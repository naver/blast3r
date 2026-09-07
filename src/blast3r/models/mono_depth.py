# Copyright (C) 2026-present Naver Corporation. All rights reserved.

import torch

from .layers import Network, ViT, LinearPatchifier, MlpHead
from .tools import AsListOfDicts

__all__ = ['MonoDepth']

def postprocess_mode_raylogz(pred):
    z = torch.exp(pred[...,2:3])
    return torch.cat( (pred[...,:2]*z, z, pred[...,3:]), dim=-1)

class MonoDepth (Network):
    """A network that predicts monocular depthmaps."""
    def __init__(self, patch_size=16, n_z_comps=5, dim=384, head_size=64, Patchifier=LinearPatchifier, posaug_factor=None, Head=MlpHead, depth=4, predmode='raylogz', init_values=None):
        super().__init__()
        assert n_z_comps > 0, "Need at least one depth component!"
        self.dim = dim
        self.patch_size = patch_size
        self.n_z_comps = n_z_comps
        # encoder
        odim = (1 + 2 + n_z_comps) * patch_size**2 # conf + (x + y + nz)
        self.encoder = ViT(patch_size, dim=dim, output_dim=odim, Patchifier=Patchifier, posaug_factor=posaug_factor, depth=depth, null_tokens=4, Head=Head, head_size=head_size, init_values=init_values)
        self.n_cls = len(self.encoder.null_tokens)

        # initialize weights
        self.apply(self._init_weights)

        assert predmode == 'raylogz', f'{predmode=} is not supported by the released models'
        self.postprocess = postprocess_mode_raylogz

    def forward(self, views, step=None, criterion=None):
        """Predict a monocular depthmap for each view.

        `criterion` is accepted here because DistributedDataParallel forbids using
        model parameters inside the loss; passing it in makes the forward compute
        the loss itself.
        """
        assert step is not None
        views = AsListOfDicts(views)
        imgs = views['img']
        token_pos = views['token_pos']
        true_shape = views['true_shape']

        B, N, ntokens, THREE, ph, pw = imgs.shape
        assert THREE == 3
        pts3d_conf = self.encoder(imgs.flatten(0,1), token_pos.flatten(0,1), true_shape=true_shape.flatten(0,1))

        # back to image resolution
        pts3d_conf = pts3d_conf[:, self.n_cls:, :] # remove null tokens
        pts3d_conf = pts3d_conf.view(B,N,ntokens, 3+self.n_z_comps, self.patch_size, self.patch_size)
        pts3d_conf = pts3d_conf.permute(0,1,2, 4, 5, 3)

        # prepare results
        # _raw is before postprocessing
        results = {}
        query_pts3d_raw = pts3d_conf[...,:-1]
        query_pts3d = self.postprocess(pts3d_conf[...,:-1])
        query_pts3d_conf = pts3d_conf[...,-1]
        for n in range(N):
            output = dict(
                patch_size = self.patch_size,
                query_pts3d_raw = query_pts3d_raw[:,n,...],
                query_pts3d = query_pts3d[:,n,...],
                query_pts3d_conf = query_pts3d_conf[:,n,...],
                coarse_matcher = None,
                dense_matcher = None)
            results[n] = output

        if criterion is not None:
            loss, loss_details = criterion(results, views)
            return loss, loss_details
        else:
            return results
