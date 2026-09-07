# Copyright (C) 2026-present Naver Corporation. All rights reserved.

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_max

from blast3r.utils.geometry import xy_grid


def l2(x):
    return F.normalize(x, dim=-1)

def unpatchify(tokens, h, w, patch_size=16):
    B, N, d = tokens.shape
    if h*w//patch_size**2 == N:
        h //= patch_size
        w //= patch_size
    assert h*w == N
    return F.pixel_shuffle(tokens.transpose(1,2).view(B,d,h,w), patch_size)


def _apply_slice(obj, idx):
    if isinstance(obj, (list,tuple)):
        List = type(obj)
        if len(obj) and isinstance(obj[0], (np.ndarray, torch.Tensor)):
            return List([_apply_slice(a,idx) for a in obj])
        else:
            return List([a for a in obj[idx]])
    else:
        ndim = obj.ndim
        if ndim<2:
            assert obj.numel()==1
            return obj
        v = obj.transpose(0,1)[idx]
        return v if v.ndim < ndim else v.transpose(0,1)


class AsListOfDicts:
    def __init__(self, views):
        while isinstance(views, AsListOfDicts):
            views = views.views
        self.views = views
        assert 'img' in views
        assert isinstance(views['img'], (np.ndarray, torch.Tensor))
        self._cache = {}

    def __len__(self):
        return len(self.views['img'])

    @property
    def shape(self):
        return self.views['img'].shape

    @property
    def B(self):
        B, N, ntokens, THREE, ph, pw = self.shape
        assert THREE == 3
        return B

    @property
    def N(self):
        B, N, ntokens, THREE, ph, pw = self.shape
        assert THREE == 3
        return N

    @property
    def ntokens(self):
        B, N, ntokens, THREE, ph, pw = self.shape
        assert THREE == 3
        return ntokens

    @property
    def device(self):
        return self.views.device

    def keys(self):
        return self.views.keys()

    def to_list(self):
        return [{k:_apply_slice(v, idx) for k,v in self.views.items()} for idx in range(self.views['img'].shape[1])]

    def apply_slice(self, idx):
        res = {k:_apply_slice(v, idx) for k,v in self.views.items()}
        return res

    def cache(self, key, val=None):
        if val is not None:
            self._cache[key] = val
        else:
            return self._cache[key]

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.views[idx]
        elif isinstance(idx, (int, slice)):
            return AsListOfDicts(self.apply_slice(idx))
        else:
            raise KeyError(b'bad slice {idx=} for AsListOfDict()')


class Fraction:
    """A loss as a fraction (sum_loss / nb), so reduction can be applied at the end."""
    def __init__(self, *loss_and_nb):
        if len(loss_and_nb) == 0:
            self.sum_loss = self.nb = 0
        elif len(loss_and_nb) == 1:
            (self.sum_loss, self.nb), = loss_and_nb
        else:
            self.sum_loss, self.nb = loss_and_nb
        assert self.nb > 0 or (self.sum_loss == 0 or not self.sum_loss.isfinite())

    def __repr__(self):
        return f"Fraction({self.sum_loss:g} / {self.nb:g} = {self.sum_loss/max(1e-16,self.nb):g})"

    def __bool__(self):
        return bool(self.nb > 0)

    def reduce(self):
        return self.sum_loss / self.nb if self else 0.0

    def __rmul__(self, factor):
        assert isinstance(factor, (float,int, torch.Tensor))
        return Fraction(factor * self.sum_loss, factor * self.nb)

    def __add__(self, other):
        assert isinstance(other, Fraction)
        return Fraction(self.sum_loss + other.sum_loss, self.nb + other.nb)

null = Fraction()


def to_flow(pred, H, W, WIDTH, zero_nans=True, as_np_array=True):
    N, hw, ps, ps2 = pred.shape
    assert ps==ps2 and hw*ps*ps2 == H*W, (ps,ps2,hw,H,W,N, pred.shape)
    flow = pred.reshape(N,H//ps,W//ps,ps,ps).permute(0,1,3,2,4).reshape(N, H, W)
    if WIDTH is not None:
        flow = flow.float()
        flow[flow<0] = float('nan')
        flow = flow.floor() # make it 'integer'
        flow = torch.stack((flow % WIDTH, flow//WIDTH), dim=-1)

        flow = flow - xy_grid(W, H, device=flow.device).view(1, H, W, 2)
    if zero_nans:
        flow = flow.nan_to_num(nan=0)
    if as_np_array:
        flow = flow.cpu().numpy()
    return flow

def aggregate_sparse_flow(idxs_0, idxs_1, matching, S_0, S_1, WIDTH, patch_left_top, patch_size=16):
    """Reduce a sparse block matrix along its second axis.

    The matrix has shape S = (S_0, S_1, P**2), with blocks `matching` at rows
    `idxs_0` and columns `idxs_1`. Returns S.max(1) and S.argmax(1), both of
    shape (S_0, P**2).

    if flow_mode == 'flow_to_tgt':
        idxs_0 = b_idxs
        S_0 = B*(N-1)*ntokens # all db patches
        idxs_1 = q_idxs % ntokens
        S_1 = ntokens         # query patches
        patch_left_top = P * views['token_pos'][:,-1,...].flip(-1).reshape(-1,2)[q_idxs]

    if flow_mode == 'flow_from_src':
        idxs_0 = (b, img, q_idx) with b=q_idxs//ntokens, img=b_idxs//((N-1)*ntokens) % B, and  q_idx=q_idxs % ntokens
        S_0 = B*(N-1)*ntokens # all query*db_img patches
        idxs_1 = b_idxs % ntokens
        S_1 = ntokens         # db patches
        patch_left_top = P * views['token_pos'][:,:-1,...].flip(-1).reshape(-1,2)[b_idxs]

    Args:
        b_idxs: (n,) db patch indices.
        q_idxs: (n,) query patch indices.
        matching: (n, patch_size**2, patch_size**2 + 1) score of [pixel_0, pixel_1];
            the last bin is the null match.

    Returns:
        best_score: (S_0, P**2) best match for each db pixel.
        arg_best: (S_0, P**2) query pixel index of the best match.
    """
    P = patch_size

    # first step: select the best target pixel for each source pixel (argmax of the discrete posterior distribution)
    if isinstance(matching, torch.Tensor):
        assert matching.shape[-2:] == (P**2, P**2 + 1)
        assert len(idxs_0) == len(idxs_1) == len(matching) == len(patch_left_top)
        local_best, local_argbest = matching.max(dim=-1) # (n, ps**2)
    else:
        assert len(idxs_0) == len(idxs_1) == len(patch_left_top)
        local_best, local_argbest = matching
        assert local_best.shape == local_argbest.shape == (len(idxs_0), P**2)

    # For each pixel (b_idx, pij), we find the best_pair_idx that maximizes local_best[best_pair_idx, pij]
    # such that b_idxs[best_pair_idx] == b_idxs.
    db_patch_idx = idxs_0[:,None].expand(-1, P**2)
    best_score, arg_best = scatter_max(local_best, db_patch_idx, dim=0, dim_size=S_0) # (B*N*h*w, ps**2)

    # convert patch_idx to actual query pixels
    bx, by = patch_left_top.T # in target image
    ux, uy = (local_argbest % P), (local_argbest // P) # within-block coordinate
    local_argbest_pixel = (by[:,None] + uy) * WIDTH + (bx[:,None] + ux) # query pixel index
    # handle null matches
    local_argbest_pixel[ local_argbest == P**2 ] = -1 # bad bin = null match

    # convert arg_best to query pixels
    local_argbest_pixel = torch.cat((local_argbest_pixel, -local_argbest.new_ones((1,P**2))), dim=0)
    arg_best_pixel = local_argbest_pixel.gather(0, arg_best)

    return best_score, arg_best_pixel
