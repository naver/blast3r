# Copyright (C) 2026-present Naver Corporation. All rights reserved.

from dataclasses import dataclass
import types
import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast
import torch_scatter
from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster

from blast3r.utils.device import get_device_str, todevice, to_numpy, isfinite, zeros_like, ones_like, concat, contiguous, cumsum, clone
from blast3r.utils.geometry import xy_grid, bilinear_sampling
from blast3r.image import detokenize
from blast3r.retrieval.processor import IncrementalRetriever
from blast3r.models import CausalTrack3R


@dataclass
class ObservedTracks:
    n_imgs: int            # number of views
    n_tracks: int          # number of tracks
    nppi: np.ndarray       # (n_img,) number of points per img
    pids: np.ndarray       # (n_kpts,) track id for each 2d keypoint in [0, n_tracks-1]
    pix2d: np.ndarray      # (n_kpts, 2) 2d observations of tracks
    pix2d_std: np.ndarray  # (n_kpts, 2) 2d observations noise (std)
    pix2d_dep: np.ndarray  # (n_kpts, n_zcfs) multi-depth at this observation
    depth_mode: str        # depth mode in {'log_multidepth', 'multidepth'}
    # zcoefs: np.ndarray     # (n_kpts, n_zcfs) initial z_coefs
    # pts3d: np.ndarray      # (n_tracks, 3) 3d tracks
    img_slices: list = None  # automatically filled from nppi

    @property
    def device(self):
        devices = {get_device_str(self.nppi), get_device_str(self.pids), get_device_str(self.pix2d),
                   get_device_str(self.pix2d_std), get_device_str(self.pix2d_dep)}
        if len(devices) == 1:
            return next(iter(devices))
        return None

    @property
    def n_kpts(self):
        return len(self.pids)

    @property
    def img_id(self):
        res = zeros_like(self.pids)
        res[cumsum(self.nppi, -1)[:-1]] = 1 # delimiters
        return cumsum(res, -1)

    def __post_init__(self):
        assert self.n_imgs == len(self.nppi)
        assert len(self.pids) == len(self.pix2d) == len(self.pix2d_std) == len(self.pix2d_dep)
        assert self.n_kpts == int(self.nppi.sum())
        if self.img_slices is None:
            cum_nkpts = cumsum(self.nppi, dim=-1)
            self.img_slices = [slice(0, int(cum_nkpts[0]))] + [slice(int(i),int(j)) for i,j in zip(cum_nkpts[:-1], cum_nkpts[1:])]

    def to(self, device):
        if self.device == device: return self
        return ObservedTracks(**todevice(vars(self), device))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # return a view of self only for a selected frame
            idx = slice(idx, idx+1)
            sl = self.img_slices[idx]
            res = ObservedTracks(1, self.n_tracks, self.nppi[idx], self.pids[sl], self.pix2d[sl], self.pix2d_std[sl], self.pix2d_dep[sl], self.depth_mode)

        elif isinstance(idx, torch.Tensor):
            def cat_sel(arr):
                return torch.cat([arr[self.img_slices[i]] for i in idx])
            res = ObservedTracks(len(idx), self.n_tracks, self.nppi[idx], cat_sel(self.pids), cat_sel(self.pix2d), cat_sel(self.pix2d_std), cat_sel(self.pix2d_dep), self.depth_mode)

        return res

    def __ror__(self, left):
        assert left is None
        return self

    def __or__(self, trks):
        assert self.n_imgs+1 == trks.n_imgs
        assert self.depth_mode == trks.depth_mode
        if self.n_tracks == 0: return trks
        nppi = clone(trks.nppi)
        nppi[:-1] += self.nppi

        pids, pix2d, pix2d_std, pix2d_dep = [], [], [], []
        for sl1, sl2 in zip(self.img_slices+[slice(0,0)], trks.img_slices):
            pids.append(self.pids[sl1])
            pids.append(trks.pids[sl2] + self.n_tracks)
            pix2d.append(self.pix2d[sl1])
            pix2d.append(trks.pix2d[sl2])
            pix2d_std.append(self.pix2d_std[sl1])
            pix2d_std.append(trks.pix2d_std[sl2])
            pix2d_dep.append(self.pix2d_dep[sl1])
            pix2d_dep.append(trks.pix2d_dep[sl2])

        assert self.pix2d_dep.shape[1] == trks.pix2d_dep.shape[1]
        res = ObservedTracks(trks.n_imgs, self.n_tracks + trks.n_tracks, nppi, concat(pids),
                             concat(pix2d), concat(pix2d_std), concat(pix2d_dep), self.depth_mode)
        return res

    def to_dense(self, n_imgs=None):
        if n_imgs is None:
            n_imgs = self.n_imgs
        else:
            assert n_imgs >= self.n_imgs

        res = torch.full((n_imgs, self.n_tracks, 2), float('nan'), device=self.pix2d.device)
        for i, sli in enumerate(self.img_slices):
            track_nums = self.pids[sli]
            res[i, track_nums] = self.pix2d[sli]
        return res.transpose(0,1)

    def remove(self, bad_track_idxs):
        if len(bad_track_idxs) == 0:
            return self, zeros_like(self.pids, shape=(0,))

        ok = ~torch.isin(self.pids, bad_track_idxs)
        nppi = torch.tensor([ok[sl].sum() for sl in self.img_slices], dtype=torch.int32, device=self.nppi.device)
        # remap track indices
        old_to_keep, pids = torch.unique(self.pids[ok], return_inverse=True)

        res = ObservedTracks(self.n_imgs, len(old_to_keep), nppi, pids.int(),
                             self.pix2d[ok], self.pix2d_std[ok], self.pix2d_dep[ok], self.depth_mode)
        return res, old_to_keep


def rm_duplicated_tracks(cur_tracks, new, dist_thr=5):
    """Remove redundant tracks.

    A track is redundant if it is old and a subset of a new track, meaning it
    appears in the same views and at a close position.

    Args:
        cur_tracks: ObservedTracks.
        new: (n_tracks, n_imgs, 2) tensor.
    """
    assert isinstance(cur_tracks, ObservedTracks)
    assert isinstance(new, torch.Tensor)
    old = cur_tracks.to_dense(n_imgs=new.shape[1])

    sels = []
    CHUNK = 1024
    for c in range(0, len(old), CHUNK):
        # compare all old tracks versus all new tracks
        dist = (old[c:c+CHUNK,None] - new[None, :]).norm(dim=-1) # (n_old_tracks, n_new_tracks, n_imgs)

        # old: undefined = nans
        # new: undefined = infs
        # we want to allow (old=nan, new=x) but not (old=x, new=inf), so we set nans to 0
        dist.nan_to_num_() # keep infs
        sel = (dist < dist_thr).all(dim=2) # set inclusion: old <= new
        sel = sel.any(dim=1) # an old track has at least one matching new track
        sel = sel.nonzero().ravel()
        sels.append(c + sel)

    assert len(sels), 'no image pairs selected'
    sel = torch.cat(sels)
    print(f'>> removing {len(sel)} duplicated tracks')
    return cur_tracks.remove(sel)


def rm_redundant_tracks(tracks, max_H, max_W, num_keep, subsample=None, bin_size=16):
    """Score how important each track is, then remove the less important ones.

    score = min(number of close tracks, over every image where it appears)
    """
    assert isinstance(tracks, ObservedTracks)
    if tracks.n_tracks <= num_keep:
        return tracks.remove(())

    w = 1 + (max_W-1) // bin_size # number of bins on the x axis
    h = 1 + (max_H-1) // bin_size # number of bins on the x axis

    # bin tracks into (n_imgs, bin_size, bin_size) histogram
    px, py = tracks.pix2d.int().T # (n_kpts, 2)
    per_img_bins = (py // bin_size) * w + (px // bin_size)
    bins = tracks.img_id * h*w + per_img_bins # (n_kpts, )

    # histogram on the bins
    _, inv, bin_count = torch.unique(bins, return_inverse=True, return_counts=True)
    # inv: n_kpts --> n_bins
    # counts: (n_bins,) with n_bins = n_imgs*bin_size*bin_size

    # track score = minimum of a bin --> torch_scatter
    track_scores, _ = torch_scatter.scatter_min(bin_count[inv], tracks.pids.long(), dim_size=tracks.n_tracks)
    sel = track_scores.argsort()[num_keep:] # removing tracks with the most

    # we need to make sure that we are not removing tracks with score == 1
    valid = (track_scores[sel] > 1)

    if subsample:
        # add them little by little
        n_valid = valid.sum()
        if n_valid == 0: return tracks.remove(())
        sel = sel[torch.multinomial(valid.float(), min(n_valid, subsample), replacement=False)]
    else:
        sel = sel[valid]

    print(f'>> removing {len(sel)} redundant tracks')
    return tracks.remove(sel)


def build_coreset(sim_matrix, coreset_method, coreset_size):
    """Build a coreset of frame indices from a similarity matrix (either FPS or hierarchical clustering)."""
    n = sim_matrix.shape[0]
    if coreset_size >= n:
        return torch.arange(n, device=sim_matrix.device)

    # symmetrize sim matrix (from lower diagonal)
    sim_matrix = sim_matrix.tril(-1) + sim_matrix.tril(-1).T + torch.diag(sim_matrix.diag())

    if coreset_method == 'hclust':
        sim_np = sim_matrix.cpu().numpy()
        dist_matrix = 1 - sim_np
        triu_indices = np.triu_indices_from(dist_matrix, k=1)
        condensed = dist_matrix[triu_indices]
        Z = scipy_linkage(condensed, method='ward')
        clusters = fcluster(Z, t=coreset_size, criterion='maxclust')

        coreset_list = []
        for c in range(1, coreset_size + 1):
            members = np.where(clusters == c)[0]
            if len(members) == 0:
                continue
            centroid = dist_matrix[members].mean(axis=0)
            closest = members[np.argmin([np.linalg.norm(dist_matrix[m] - centroid) for m in members])]
            coreset_list.append(int(closest))

        if len(coreset_list) == 0:
            return torch.tensor([], device=sim_matrix.device, dtype=torch.int64)
        return torch.tensor(coreset_list, device=sim_matrix.device).sort().values

    elif coreset_method == 'fps':
        dist = 1 - sim_matrix
        selected = [int(torch.randint(0, dist.shape[0], (1,)).item())]
        for i in range(1, coreset_size):
            d = dist[selected].min(dim=0).values
            bst = int(d.argmax().item())
            selected.append(bst)
        return torch.tensor(selected, device=sim_matrix.device).sort().values

    else:
        raise ValueError(f'Unknown coreset method: {coreset_method}')


def retrieve_imgs(sim_matrix, query, retrieval_mode, num_retrieved_images, see_future=False):
    last = None if see_future else query
    if last == 0 or retrieval_mode == 'none':
        return torch.arange(last, device=sim_matrix.device) # all images up to query (excluded)

    elif retrieval_mode == 'last':
        # just use the last frames
        first = max(0, last - num_retrieved_images)
        return torch.arange(first, last, device=sim_matrix.device)

    elif retrieval_mode == 'topsim':
        # just returns the top-k
        neg_sim = -sim_matrix[query, :last]
        ranks = neg_sim.argsort()[:num_retrieved_images]

    elif retrieval_mode.startswith('div'):
        # greedy procedure to identify the most similar AND diverse images

        # important to set diagonal to 1 and normalize in [0,1]
        sim_matrix = sim_matrix - sim_matrix.min() + 1e-6 # subtract min value
        sim_matrix = (sim_matrix / sim_matrix.diag()[:, None]).clip(min=0, max=1)

        sim_to_query = sim_matrix[query, :last].clip(min=1e-6) # cannnot be 0

        ranks = [int(sim_to_query.argmax())]
        for i in range(1, min(last,num_retrieved_images)):

            sim_to_retrieved = sim_matrix[ranks, :last]
            if retrieval_mode.endswith('max'):
                max_sim_to_retrieved = sim_to_retrieved.max(0).values
            elif retrieval_mode.endswith('avg'):
                max_sim_to_retrieved = sim_to_retrieved.mean(0)
            elif retrieval_mode.endswith('rmse'):
                max_sim_to_retrieved = sim_to_retrieved.square().mean(0).sqrt() # kinda like a soft-max
            else:
                raise NameError(f'wrong {retrieval_mode=}')

            diversity = (1 - max_sim_to_retrieved).clip(min=1e-6) # cannot be 0
            score = sim_to_query * diversity
            best = int(score.argmax())
            ranks.append(best)

            # disable re-selecting it again
            sim_to_query[best] = 0

        ranks = torch.tensor(ranks).to(sim_matrix.device)

    elif retrieval_mode.startswith('coreset'):
        parts = retrieval_mode.split('_')
        if len(parts) != 3:
            raise ValueError(f'bad {retrieval_mode=}, expected coreset_<method>_<size>')
        _, coreset_method, coreset_size_s = parts
        coreset_size = int(coreset_size_s)

        sim_matrix_c = sim_matrix
        if not see_future:
            sim_matrix_c = sim_matrix[:last, :last] # only consider past frames for coreset construction

        coreset = build_coreset(sim_matrix_c, coreset_method, coreset_size)

        selected = coreset

        if last is not None:
            available = torch.arange(last, device=sim_matrix.device)
        else:
            available = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)

        # fill with most similar frames not already selected
        need = max(0, num_retrieved_images - int(selected.numel()))
        if need > 0:
            candidates = available[~torch.isin(available, coreset)]
            if candidates.numel() > 0:
                sim_to_query = sim_matrix[query]
                sim_other = sim_to_query[candidates]
                k = min(need, int(candidates.numel()))
                topk = sim_other.argsort(descending=True)[:k]
                chosen = candidates[topk]
                selected = torch.cat((selected, chosen)) if selected.numel() else chosen

        if selected.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=sim_matrix.device)
        return selected[:num_retrieved_images].sort().values

    elif retrieval_mode == 'wgtrng':
        sim_to_query = sim_matrix[query, :last].clip(min=0)

        proba = sim_to_query / sim_to_query.sum()
        ranks = torch.multinomial(proba, num_samples=min(last,num_retrieved_images), replacement=False)

    elif retrieval_mode.startswith('keyframe'):
        first_mode, second_mode = retrieval_mode.split('-')
        keyframe_step = int(first_mode[len('keyframe'):])

        # basically retrieves again with a subset of frames
        sel = torch.zeros(len(sim_matrix), dtype=bool, device=sim_matrix.device)
        sel[::keyframe_step] = True
        sel[query] = True

        # most of retrieved images will be keyframes
        keyframes_ranks = retrieve_imgs(sim_matrix[sel][:,sel], sel.cumsum(-1)[query]-1, second_mode, 3*num_retrieved_images//4, see_future=see_future)
        keyframes_ranks = sel.nonzero().ravel()[keyframes_ranks]

        # the rest will be nearest neighbors
        sel = ~sel
        sel[query] = True
        other_ranks = retrieve_imgs(sim_matrix[sel][:,sel], sel.cumsum(-1)[query]-1, second_mode, num_retrieved_images-len(keyframes_ranks), see_future=see_future)
        other_ranks = sel.nonzero().ravel()[other_ranks]

        ranks = torch.cat((keyframes_ranks, other_ranks))
        assert len(torch.unique(ranks)) == len(ranks)

    else:
        raise ValueError(f'bad {retrieval_mode=}')

    # sort in temporal order
    return ranks.sort().values


class BaseOnlineMatcher:
    def __init__(self, device):
        self.device = torch.device(device)
        self.reverse = None

    def reset(self):
        self.n_imgs_so_far = 0 # reset when switching to backward flow

    def iter_views(self, views, reverse=False):
        assert isinstance(views, dict)
        if self.reverse != reverse:
            self.reverse = reverse
            self.reset()

        if views['img'].ndim == 3: # single image
            order = (slice(None),)
        else: # image list
            n_imgs = len(views['img'])
            order = range(n_imgs-1,-1,-1) if reverse else range(n_imgs)

        for i in order:
            yield self.batch_view(views, i)

    def batch_view(self, views, i=slice(None)):
        # selecting one image, (ntokens, 3, 16, 16)
        view = {key:views[key][i] for key in 'img token_pos true_shape'.split()}
        # expanding to (B=1, N=1, ntokens, 3, 16, 16)
        batched_view = {key:todevice(tensor, self.device).unsqueeze(0).unsqueeze(0) for key, tensor in view.items()}
        assert batched_view['img'].ndim == 6, f"expected a 6-dim batched img, got {batched_view['img'].ndim}"
        return batched_view


class OnlineMatcher (BaseOnlineMatcher):
    def __init__(self, model, n_total_imgs, n_tokens, patch_size, device, pin_memory=False):
        """`pin_memory` keeps the per-frame encoder tokens and decoder memory in
        pinned host RAM rather than on the GPU, and copies only the retrieved
        frames over at each step. Same computation; it trades ~4MB of GPU per
        frame (at size 512) for a ~170MB host-to-device copy per frame."""
        super().__init__(device)
        self.model = model
        self.hw = n_tokens
        self.P = patch_size
        self.n_total_imgs = n_total_imgs
        self.n_imgs_so_far = 0
        self.optim_memory = False
        self.pin_memory = pin_memory
        self._staging = {}  # pinned buffers the retrieved frames are gathered into

    def init(self):
        if isinstance(self.model, types.LambdaType):
            self.model = self.model() # dynamic load
        matcher, retriever = self.model
        self.matcher = matcher.to(self.device)
        assert isinstance(self.matcher, CausalTrack3R)
        assert self.matcher.patch_size == self.P

        assert retriever is not None, 'online inference needs a retrieval model: the checkpoint must ship retrieval weights, and asmk must be installed'
        self.retriever = IncrementalRetriever(retriever.model, retriever.asmk)

        self.n_tokens = self.hw + self.matcher.n_cls

        self.input_shape = torch.empty((self.n_total_imgs, 2), dtype=torch.int32, device=self.device)
        self.ji_grid = torch.empty((self.n_total_imgs, self.hw, 2), dtype=torch.int32, device=self.device)
        self.Pos = torch.empty((self.n_total_imgs, self.n_tokens, 2), device=self.device)
        self.sim_matrix = torch.eye(self.n_total_imgs, device=self.device)

        # Every previous image stays available to retrieval, so its encoder
        # tokens and decoder memory are kept for the whole run: on the GPU, or in
        # pinned host memory that the retrieved frames are copied out of.
        if self.pin_memory:
            store = dict(device='cpu', pin_memory=self.device.type == 'cuda')
        else:
            store = dict(device=self.device)
        self.enc_tokens = torch.empty((self.n_total_imgs, self.n_tokens, self.matcher.enc_dim), **store)
        self.memory = torch.empty((self.n_total_imgs, self.n_tokens, self.matcher.dec_dim), **store)

    def _gather(self, store, idxs, name):
        """`store[idxs]` on the compute device. A host store is copied frame by frame
        straight into a device buffer: from pinned memory these are asynchronous
        DMAs, ~10ms for 41 frames, where gathering on the host first costs 250ms.
        The buffer is reused across calls, which is safe because every use of it
        is on the same stream."""
        if store.device.type != 'cpu':
            return store[idxs]
        n = len(idxs)
        buffer = self._staging.get(name)
        if buffer is None or len(buffer) < n:
            buffer = torch.empty((max(n, 1),) + store.shape[1:], dtype=store.dtype, device=self.device)
            self._staging[name] = buffer
        for slot, frame in enumerate(idxs.tolist()):
            buffer[slot].copy_(store[frame], non_blocking=True)
        return buffer[:n]

    @property
    def flow_mode(self):
        return self.matcher.flow_mode

    def __call__(self, batched_view, retrieval_mode, num_retrieved_images):
        n = self.n_imgs_so_far
        self.n_imgs_so_far += 1
        if n == 0: self.init()

        if batched_view['img'].ndim != 6: # not batched yet
            batched_view = self.batch_view(batched_view)

        H, W = map(int, batched_view['true_shape'].ravel())
        self.ji_grid[n] = batched_view['token_pos'].squeeze()

        # encode all views
        with autocast(device_type=self.device.type, dtype=self.matcher.force_dtype, enabled=(self.matcher.force_dtype != torch.float32)):
            enc_n, self.Pos[n], self.input_shape[n], kpt_conf = (x.view(-1, x.shape[-1]) for x in self.matcher.encode(batched_view)[:4])
            self.enc_tokens[n] = enc_n

            # compute overall similarity with previous images
            self.sim_matrix[n,:n] = self.retriever(enc_n, add_to_db=True)

            # select past frames
            db_nums = retrieve_imgs(self.sim_matrix, n, retrieval_mode, num_retrieved_images)
            print(f'>> Retrieved images for query={n}: {db_nums.tolist()}')

            # decoder loop with selected frames
            query = enc_n[None]
            pos = self.Pos[None,n]
            mem = self._gather(self.memory, db_nums, 'memory')[None]
            mem_pos = self.Pos[None,db_nums]
            input_shapes = None # useless?
            preds = self.matcher._decoder(query, pos, input_shapes, mem.flatten(1,2), mem_pos.flatten(1,2))

            # add new tokens to the memory
            out_tokens = preds['query_decoder_tokens']
            out_tokens = torch.cat((out_tokens[-3], out_tokens[-2], out_tokens[-1]), dim=-1)
            new_mem = self.matcher.decoder_to_memory(self.matcher.norm(out_tokens))
            self.memory[n] = new_mem.squeeze(0)

            coarse_matcher = preds['coarse_matcher']
            dense_matcher = preds['dense_matcher']

            kpt_conf = dense_matcher._remove_cls(kpt_conf[None], self.hw, last=self.P**2)
            kpt_conf = detokenize(kpt_conf.view(self.hw, self.P, self.P), (H,W), chan_dim=None).float()
            kpt_conf.exp_()
            if self.n_imgs_so_far == 1:
                return n, db_nums, kpt_conf, ([], [])

            coarse_tracks = coarse_matcher(preds)
            preds['tokens_pos'] = torch.cat((mem_pos, pos.unsqueeze(1)), dim=1)
            db_nums_and_query = torch.cat((db_nums, db_nums.new([n])))
            preds['encoder_tokens'] = self._gather(self.enc_tokens, db_nums_and_query, 'enc_tokens')[None]
            preds['decoder_tokens'] = torch.cat((mem, new_mem.unsqueeze(1)), dim=1)
            matching_data = dense_matcher(preds, coarse_tracks)

            ji_grid = self.ji_grid[None, db_nums_and_query]
            pred_flow, pred_score = dense_matcher.optical_flow(matching_data, (1, db_nums_and_query.shape[0], self.hw), ji_grid, W)

        pred_flow  = detokenize(pred_flow[0], (H,W)).float()
        pred_score = detokenize(pred_score[0], (H,W), chan_dim=None).float()
        pred_score = pred_score.exp_() # these are logits

        if self.optim_memory:
            torch.cuda.empty_cache()

        valid_flow = (pred_flow[...,1] >= 0)
        return n, db_nums, kpt_conf, (pred_flow, pred_score * valid_flow)



class StoredFlows:
    """Store absolute flows and scores in a sparse layout, using as little memory as possible.

    Internally, abs_flow[src][tgt] = (flow, score), with
    flow = (H, W, 2) stored as int16/16 and score = (H, W) stored as uint8/255.
    """
    def __init__(self, n_imgs, device, imshapes=None):
        self.n_imgs = n_imgs
        self.imshapes = ([None] * n_imgs) if imshapes is None else list((h,w) for h,w in imshapes)
        self.device = device
        self.forget()

    def id_flow(self, H, W):
        id_flow = self.compact_flow(xy_grid(W, H, device=self.device) + 0.5)
        id_score = torch.full((H,W), 255, dtype=torch.uint8, device=self.device)
        return id_flow, id_score

    def forget(self):
        # abs_flow[src_view, tgt_view, px, py] --> (px', py') in tgt_view
        self.abs_flow = [[None for _ in range(self.n_imgs)] for _ in range(self.n_imgs)]

        # identities on the diagonal
        for i in range(self.n_imgs):
            if self.imshapes[i] is None: continue
            self.abs_flow[i][i] = self.id_flow(*self.imshapes[i])

    def add_one_frame(self, H, W):
        self.imshapes.append((H,W))
        # add last row
        self.abs_flow.append([None] * self.n_imgs)
        # add last column
        for row in self.abs_flow:
            row.append(None)
        # set diagonal term
        self.abs_flow[self.n_imgs][self.n_imgs] = self.id_flow(H, W)
        self.n_imgs += 1
        assert len(self.imshapes) == len(self.abs_flow) == self.n_imgs

    def compact_flow(self, flow):
        assert isinstance(flow, torch.Tensor) and flow.shape[:-1] in self.imshapes, \
            f'flow shape {tuple(flow.shape)} does not match any image shape'
        flow = (16 * flow).round().to(torch.int16)
        return flow.to(self.device)

    def uncompress_flow(self, flow):
        return flow.float() / 16

    def compact_score(self, score):
        assert isinstance(score, torch.Tensor) and score.shape in self.imshapes, \
            f'score shape {tuple(score.shape)} does not match any image shape'
        score = (255 * score).round().clip(min=0, max=255).to(torch.uint8)
        return score.to(self.device)

    def uncompress_score(self, score):
        return score.float() / 255

    def iter_idxs(self, idxs):
        if isinstance(idxs, int):
            yield from [idxs]
        elif isinstance(idxs, slice):
            yield from range(idxs.start or 0, self.n_imgs if idxs.stop is None else idxs.stop, idxs.step or 1)
        elif isinstance(idxs, (torch.Tensor, np.ndarray)) and idxs.ndim == 1:
            yield from idxs
        else:
            raise TypeError(f'bad input {idxs=}')

    def __setitem__(self, src_tgt_idxs, flows_and_scores):
        src_idxs, tgt_idxs = src_tgt_idxs
        flows, scores = map(iter, flows_and_scores)

        for src_idx in self.iter_idxs(src_idxs):
            for tgt_idx in self.iter_idxs(tgt_idxs):
                # compress to reduce memory footprint
                flow = next(flows)
                score = next(scores)
                assert self.imshapes[src_idx] == score.shape == flow.shape[:-1]
                self.abs_flow[src_idx][tgt_idx] = (self.compact_flow(flow), self.compact_score(score))

    def __getitem__(self, src_tgt_idxs):
        src_idxs, tgt_idxs = src_tgt_idxs
        res = []
        for src_idx in self.iter_idxs(src_idxs):
            for tgt_idx in self.iter_idxs(tgt_idxs):
                flow_score = self.abs_flow[src_idx][tgt_idx]
                if flow_score is None:
                    flow = torch.full(self.imshapes[src_idx], float('nan'), device=self.device)
                else:
                    flow, score = flow_score
                    flow = self.uncompress_flow(flow)
                    flow[score==0] = float('nan')
                res.append(flow)
        return res[0] if len(res) == 1 else torch.stack(res)

    def get_valid_flows(self, src_idxs, tgt_idxs):
        res = []
        for src_idx in self.iter_idxs(src_idxs):
            for tgt_idx in self.iter_idxs(tgt_idxs):
                flow_score = self.abs_flow[src_idx][tgt_idx]
                if flow_score is None:
                    res.append(torch.zeros(self.imshapes[src_idx], dtype=bool, device=self.device))
                else:
                    res.append(flow_score[1] > 0)
        return res # flows can have different shapes

    def posterior_proba(self, src_frame):
        accu = None
        for tgt, flow_and_score in enumerate(self.abs_flow[src_frame]):
            if flow_and_score is None: continue
            flow, score = flow_and_score
            score = self.uncompress_score(score)
            if accu is None:
                accu = score
            else:
                accu += score
        assert accu is not None, f'no stored flow for {src_frame=}'
        return accu

    def get_local_scaling(self, abs_flow, kpts):
        H, W = abs_flow.shape[:2]
        X = abs_flow[..., 0]
        Y = abs_flow[..., 1]

        ky, kx = kpts
        ky_0 = (ky - 1).clip(min=0)
        ky_1 = (ky + 1).clip(max=H-1)

        kx_0 = (kx - 1).clip(min=0)
        kx_1 = (kx + 1).clip(max=W-1)

        # gradients: compute (d/dy, d/dx)
        dX_dy = (X[ky_1,kx] - X[ky_0,kx]) / (ky_1 - ky_0)
        dX_dx = (X[ky,kx_1] - X[ky,kx_0]) / (kx_1 - kx_0)
        dY_dy = (Y[ky_1,kx] - Y[ky_0,kx]) / (ky_1 - ky_0)
        dY_dx = (Y[ky,kx_1] - Y[ky,kx_0]) / (kx_1 - kx_0)

        # Jacobian entries at each pixel:
        # J = [[dX_dx, dX_dy],
        #      [dY_dx, dY_dy]]

        # area magnification (signed):
        detJ = dX_dx * dY_dy - dX_dy * dY_dx    # shape (H,W)

        # “Zoom factor” as isotropic linear scale (nonnegative):
        zoom_linear = torch.sqrt(detJ.clip(min=0))  # shape (H,W)
        return zoom_linear.nan_to_num_()

    def get_tracks(self, src_frame, kpts, min_len=2):
        # conceptually, returns abs_flow[src_frame, :, kpts]
        assert len(kpts) == 2, 'kpts should be (ky, kx)'
        n_kpts = len(kpts[0])
        xy = torch.full((n_kpts, self.n_imgs, 2), float('inf'), device=self.device)
        sc = torch.zeros((n_kpts, self.n_imgs, ), device=self.device)

        for tgt, flow_and_score in enumerate(self.abs_flow[src_frame]):
            if flow_and_score is None: continue
            flow, score = flow_and_score
            flow = self.uncompress_flow(flow)

            # disable invalid keypoints
            is_valid = (score[kpts] > 0)

            xy[:, tgt] = torch.where(is_valid[:,None], flow[kpts], float('inf'))
            sc[:, tgt] = is_valid * self.get_local_scaling(flow, kpts)

        # only keep long tracks
        good = ((sc > 0).sum(1) >= min_len) # is defined in all frames
        return xy[good], sc[good]


@torch.inference_mode()
def get_absolute_flow(online_matcher, views, retrieval_mode, num_retrieved_images, optim_memory=False):
    online_matcher.optim_memory = optim_memory

    n_imgs = len(views['img'])
    flow_device = "cpu" if optim_memory else online_matcher.device
    abs_flow = StoredFlows(n_imgs, flow_device, imshapes=views['true_shape'].tolist())

    # forward flow
    kpt_confs = []
    for batched_view in online_matcher.iter_views(views):
        tgt, db_nums, kpt_conf, preds = online_matcher(batched_view, retrieval_mode, num_retrieved_images)
        if db_nums is None: continue

        kpt_confs.append( kpt_conf )
        if online_matcher.flow_mode == 'flow_from_src':
            abs_flow[tgt, db_nums] = preds
        elif online_matcher.flow_mode == 'flow_to_tgt':
            abs_flow[db_nums, tgt] = preds

    print("Starting Backward flow")

    # backward flow
    for batched_view in online_matcher.iter_views(views, reverse=True):
        tgt, db_nums, _, preds = online_matcher(batched_view, retrieval_mode, num_retrieved_images)
        if db_nums is None: continue

        tgt = n_imgs-1 - tgt # undo reverse
        rev_db_nums = slice(n_imgs-1, n_imgs-1-db_nums.stop, -1) if isinstance(db_nums, slice) else (n_imgs-1 - db_nums)
        if online_matcher.flow_mode == 'flow_from_src':
            abs_flow[tgt, rev_db_nums] = preds
        elif online_matcher.flow_mode == 'flow_to_tgt':
            abs_flow[rev_db_nums, tgt] = preds

    return abs_flow


def find_keypoints(views, abs_flow, spacing=16, start_frame=0):
    # keypoints = points that are the most repeatable
    keypoints = []
    for frame in range(start_frame, abs_flow.n_imgs):
        # check how many matches
        scoremap = abs_flow.posterior_proba(frame) # a source keypoint is defined all target frames
        kpt_pos, kpt_score = local_maxima(scoremap)
        kx, ky = nms_local_maxima(abs_flow.imshapes[frame], kpt_pos, kpt_score, spacing).T

        keypoints.append((ky, kx))

    return todevice(keypoints, abs_flow.device)


def compute_tracks(model, views, kpt_spacing=16, remove_far_kpts=True, optim_memory=False, force_dtype=torch.float32, **retrieval_kw):
    # inference with the model
    abs_flow = get_absolute_flow(model, views, optim_memory=optim_memory, **retrieval_kw)
    keypoints = find_keypoints(views, abs_flow, kpt_spacing)

    # select tracks
    tracks = [] # (n_tracks, n_views, 2)
    scales = [] # (n_tracks, n_views)
    for frame, kpts in enumerate(keypoints):
        # create tracks from the keypoints
        xy, scale = abs_flow.get_tracks(frame, kpts, min_len=2)

        if remove_far_kpts:
            # keep only tracks that are zoomed in the most
            is_kpt_zoomed = (scale.argmax(1) == frame)
            xy = xy[is_kpt_zoomed]
            scale = scale[is_kpt_zoomed]

        tracks.append(xy)
        scales.append(scale)

    tracks = torch.cat(tracks)
    scales = torch.cat(scales)

    return tracks


def local_maxima(scoremap, threshold=0):
    assert scoremap.ndim == 2
    maxpooled = F.max_pool2d(scoremap[None,None], kernel_size=3, stride=1, padding=1).squeeze()

    maxpooled -= (maxpooled <= threshold).float() # make sure that we dont select zeros

    yx = (scoremap == maxpooled).nonzero()
    return yx.flip(-1), scoremap[tuple(yx.T)]


def nms_local_maxima(shape,
                     local_maxima: np.ndarray,
                     scores: np.ndarray,
                     k: int,
                     max_points = None):
    """Non-maximum suppression over a 2D score map.

    Peaks are kept greedily in descending score order, each suppressing a k×k
    neighborhood.

    Args:
        shape: (H, W)
        local_maxima: list of [(x,y,score)]
        k: window size; after selecting a peak, its k×k neighborhood is suppressed.
        max_points: optional cap on number of returned peaks.
        threshold: optional minimum score; values below are ignored.

    Returns:
        coords: (N, 2) int array of [y, x] peak locations (N can be 0).
    """
    local_maxima = to_numpy(local_maxima)
    scores = to_numpy(scores)
    assert local_maxima.ndim == 2 and local_maxima.shape[1] == 2, "scores must be (n, 3)"
    r = k // 2

    # process pixels in descending score order
    order = np.argsort(scores, axis=None)[::-1]  # flat indices

    H, W = shape
    suppressed = np.zeros(shape, dtype=bool)
    keep_idx = []

    for idx in order:
        x, y = local_maxima[idx]
        if suppressed[y, x]:
            continue

        # keep this peak
        keep_idx.append(idx)

        # stop early if we reached our limit
        if max_points is not None and len(keep_idx) >= max_points:
            break

        # suppress its k×k neighborhood (Chebyshev radius r)
        y0 = max(0, y - r); y1 = min(H, y + r + 1)
        x0 = max(0, x - r); x1 = min(W, x + r + 1)
        suppressed[y0:y1, x0:x1] = True

    return local_maxima[keep_idx]


def gather_data_from_dense_tracks(views, tracks, tracks_std=None):
    """Gather the bundle-adjustment inputs from dense tracks.

    Args:
        tracks: (n_tracks, n_views, 2) observed tracks; NaN if invalid.
    """
    n_tracks, n_views, TWO = tracks.shape
    valids = isfinite(tracks).all(-1)
    assert TWO == 2

    # remove bad tracks
    assert (valids.sum(1) >= 2).all()

    # transpose to --> (n_views, n_tracks)
    n_tracks = len(tracks)
    tracks = tracks.swapaxes(0,1)
    valids = valids.T

    nppi = valids.sum(1)
    pids = np.broadcast_to(np.arange(n_tracks, dtype=np.int32), valids.shape)[to_numpy(valids)]
    pix2d = tracks[valids]
    pix2d_std = ones_like(pix2d) if tracks_std is None else None

    depth_mode ,= [k for k in views[0] if k in ('log_multidepth','linear_multidepth')]
    pix2d_dep = []
    for i, view in enumerate(views):
        multidepth = view[depth_mode]
        assert multidepth.ndim == 3, f'not a multi-depth: got {multidepth.ndim} dims'
        kpts_depth = bilinear_sampling(multidepth, tracks[i][valids[i]])
        pix2d_dep.append(contiguous(kpts_depth))

    pix2d_dep = concat(pix2d_dep)
    return ObservedTracks(n_views, n_tracks, nppi, pids, pix2d, pix2d_std, pix2d_dep, depth_mode)
