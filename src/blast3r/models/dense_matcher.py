# Copyright (C) 2026-present Naver Corporation. All rights reserved.

import torch
import torch.nn as nn

from .base_matcher import Matcher
from .layers import ViT, Mlp
from .tools import aggregate_sparse_flow, l2

__all__ = ['DenseDotViT2Mlp_Kpt']

class DenseMatcher (Matcher):
    """Pixel matcher for a small set of patch pairs. To be overloaded."""
    def init(self, enc_dim, dec_dim, n_cls, patch_size):
        self.n_cls = n_cls
        self.patch_size = patch_size
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        return self

    def forward(self, preds, coarse_tracks, views=None):
        # threshold and remove CLS tokens
        coarse_tracks, coarse_thr = coarse_tracks
        if coarse_tracks.shape[1] <= 1: return None

        # select a set of tracks
        ntokens = preds['hw']
        coarse_tracks = self._remove_cls(coarse_tracks, ntokens)

        b_idxs, q_idxs, hw = self._sel_patch_pairs(coarse_tracks - coarse_thr)
        res = self.forward_ex(preds, b_idxs, q_idxs, hw) # returns `matching_data`
        (b_idxs, q_idxs, matching) = res
        assert b_idxs.ndim == 1
        assert b_idxs.shape == q_idxs.shape
        if isinstance(matching, torch.Tensor):
            assert matching.shape == b_idxs.shape + (self.patch_size**2, self.patch_size**2+1)
        else:
            assert len(matching) == 2 and matching[0].shape == matching[1].shape == b_idxs.shape + (self.patch_size**2,)
        return res

    def _sel_patch_pairs(self, coarse_tracks):
        raise NotImplementedError()

    def forward_ex(self, preds, b_idxs, q_idxs, hw):
        raise NotImplementedError()

    def optical_flow(self, matching_data, batch_shape, token_pos, WIDTH, ret_xy=True):
        b_idxs, q_idxs, matching = matching_data
        B, N, ntokens = batch_shape

        # aggregate sparse scores with scatter_max
        if self.flow_mode == 'flow_from_src':
            b_img = b_idxs // ntokens # should be in [0, B*(N-1)[
            q_pid = q_idxs % ntokens  # query patch index
            q_img_idxs = b_img * ntokens + q_pid # optical flow starts from (b, img, q_pid)
            pxy = self.patch_size * token_pos[:, :N-1].flip(-1).reshape(-1,2)[b_idxs]
            pred_score, pred_pixel = aggregate_sparse_flow(q_img_idxs, b_idxs % ntokens, matching, B*(N-1)*ntokens, ntokens, WIDTH, pxy, patch_size=self.patch_size)
            # returns (B, b_img, q_pid, P, P)

        elif self.flow_mode == 'flow_to_tgt':
            pxy = self.patch_size * token_pos[:, N-1].flip(-1).reshape(-1,2)[q_idxs] # top-left patches in target (query) image
            pred_score, pred_pixel = aggregate_sparse_flow(b_idxs, q_idxs % ntokens, matching, B*(N-1)*ntokens, ntokens, WIDTH, pxy, patch_size=self.patch_size)
            # returns (B, b_img, b_pid, P, P)

        pred_score = pred_score.view(B, N-1, ntokens, self.patch_size, self.patch_size)
        pred_pixel = pred_pixel.view(B, N-1, ntokens, self.patch_size, self.patch_size)

        if ret_xy:
            pred_pixel = torch.stack((pred_pixel % WIDTH, pred_pixel // WIDTH), dim=-1)
            return pred_pixel + 0.5, pred_score
        else:
            return pred_score, pred_pixel # (B, N-1, h*w, ps, ps)

class DefaultPairs:
    """Naive pair selection: every nonzero coarse track, capped at max_pairs."""
    def _unbatchified_pair_indices(self, coarse_tracks):
        # these are the pairs that we need to focus on
        B, Nhw, hw = coarse_tracks.shape # NO CLS token here
        batch, b_idxs, q_idxs = coarse_tracks.nonzero(as_tuple=True)
        # un-batchify
        b_idxs += batch * Nhw
        q_idxs += batch *  hw
        return b_idxs, q_idxs, hw

    def _sel_patch_pairs(self, coarse_tracks_scores):
        # threshold to keep only positive tracks
        coarse_tracks = (coarse_tracks_scores > 0)

        # select all pairs
        b_idxs, q_idxs, hw = self._unbatchified_pair_indices(coarse_tracks)

        if len(b_idxs) > self.MAX_PAIRS:
            print(f'Warning: selecting only {self.MAX_PAIRS} out of {len(b_idxs)} patch pairs!')
            assert coarse_tracks_scores.shape[0] == 1, 'must be batch_size == 1'
            scores = coarse_tracks_scores.view(-1, hw)[b_idxs, q_idxs % hw]
            sel = scores.argsort(descending=True)[:self.MAX_PAIRS]

            b_idxs = b_idxs[sel]
            q_idxs = q_idxs[sel]

        return b_idxs, q_idxs, hw


class Dot (DenseMatcher):
    """One descriptor per pixel, compared by cosine similarity."""
    MAX_PAIRS = 16384 # raised to 2**20 at inference, see inference/loading.py

    def __init__(self, pix_desc_dim=32, layer_idx=-1, **kw):
        super().__init__()
        self.pix_desc_dim = pix_desc_dim
        self.layer_idx = layer_idx
        self.score_thr = nn.Parameter(torch.zeros((1,1), dtype=torch.float32))

    def init(self, flow_mode, enc_dim, dec_dim, n_cls, patch_size):
        self.flow_mode = flow_mode
        self.setup_pixel_desc(enc_dim, dec_dim, patch_size)
        return super().init(enc_dim, dec_dim, n_cls, patch_size)

    def setup_pixel_desc(self, enc_dim, dec_dim, patch_size):
        self.pixel_desc = Mlp(dec_dim, 4*dec_dim, self.pix_desc_dim * patch_size**2)

    def _gather_descs(self, tokens, idxs, hw):
        # re-integrate CLS token in the index
        img_idx = idxs // hw
        idxs = img_idx * (hw + self.n_cls) + (idxs % hw) + self.n_cls

        # select only relevant tokens
        tokens = tokens.flatten(0,1)
        subset = tokens.gather(0, idxs.unsqueeze(-1).expand(idxs.shape[:1]+tokens.shape[1:]))
        return subset

    def _pixel_descs(self, tokens, idxs, hw):
        subset = self._gather_descs(tokens, idxs, hw)
        descs = self.pixel_desc(subset)
        descs = descs.view(-1, self.patch_size**2, self.pix_desc_dim)
        return l2(descs)

    def _dot_append_null_to_matching(self, db_desc, qy_desc, full=False):
        score_thr = self.score_thr
        if full:
            matching = db_desc @ qy_desc.transpose(-1,-2)

            # matching is (pair_idx, qy_pixel, db_pixel)
            if self.flow_mode == 'flow_from_src':
                matching = matching.transpose(-1,-2) # now --> (pair_idx, qy_pixel, db_pixel)

            matching_with_null = torch.cat((matching, score_thr.expand(len(matching),self.patch_size**2,1)), dim=-1)
        else:
            # optimizing memory for eval
            CHUNK = 1024

            matching_with_null = []
            for idx in range(0, max(1,len(db_desc)), CHUNK): # max(1,) just in case there are zeros pairs
                m_w_null = self._dot_append_null_to_matching(db_desc[idx:idx+CHUNK], qy_desc[idx:idx+CHUNK], full=True)
                bests = m_w_null.max(dim=-1)
                matching_with_null.append(bests)
            matching_with_null = tuple(map(torch.cat, zip(*matching_with_null)))

        return matching_with_null

    def forward_ex(self, preds, b_idxs, q_idxs, hw):
        # gather all token descriptors
        db_desc = self._pixel_descs(preds['mem_tokens'], b_idxs, hw) # (B, ps**2, d)
        qy_desc = self._pixel_descs(preds['query_decoder_tokens'][self.layer_idx], q_idxs, hw) # (B, ps**2, d)
        matching_with_null = self._dot_append_null_to_matching(db_desc, qy_desc)

        return b_idxs, q_idxs, matching_with_null # (n, ps**2, ps**2+1)

class DotMlp (Dot):
    """An MLP over concatenated pair descriptors, propagating information locally."""
    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.pair_norm = nn.LayerNorm(2*self.dec_dim)
        self.pair_mlp = Mlp(2*self.dec_dim, 8*self.dec_dim, 2*self.dec_dim)
        return self

    def _pixel_descs(self, descs):
        descs = self.pixel_desc(descs)
        descs = descs.view(-1, self.patch_size**2, self.pix_desc_dim)
        return l2(descs)

    def forward_ex(self, preds, b_idxs, q_idxs, hw):
        # gather all token descriptors
        db_desc = self._gather_descs(preds['mem_tokens'], b_idxs, hw) # (B, ps**2, d)
        qy_desc = self._gather_descs(preds['query_decoder_tokens'][self.layer_idx], q_idxs, hw) # (B, ps**2, d)

        # make them interact
        db_qy_desc = torch.cat((db_desc, qy_desc), dim=-1)
        db_qy_desc = db_qy_desc + self.pair_mlp(self.pair_norm(db_qy_desc))

        # extract pixel descs
        db_desc, qy_desc = map(self._pixel_descs, db_qy_desc.tensor_split([self.dec_dim], dim=-1))

        matching_with_null = self._dot_append_null_to_matching(db_desc, qy_desc)

        return b_idxs, q_idxs, matching_with_null # (n, ps**2, ps**2+1)



class DotViT2Mlp (DotMlp):
    """A ViT over merged encoder/decoder descriptors, on top of DotMlp."""
    def __init__(self, *args, vit_depth=4, last_mlp=True, split_qkv=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_qkv = split_qkv
        self.last_mlp = last_mlp
        self.vit_depth = vit_depth

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.merge_enc_dec = nn.Linear(self.enc_dim + self.dec_dim, self.dec_dim)
        self.vit = ViT(dim=self.dec_dim, depth=self.vit_depth, with_CA=True, output_dim=self.dec_dim, split_qkv=self.split_qkv)

        if self.last_mlp:
            self.pair_norm = nn.LayerNorm(2*self.dec_dim)
            self.pair_mlp = Mlp(2*self.dec_dim, 8*self.dec_dim, 2*self.dec_dim)
        else:
            raise NotImplementedError('last_mlp=False is not supported by the released models')

        return self

    def forward_ex(self, preds, b_idxs, q_idxs, hw):
        # encoder tokens
        enc_tokens = preds['encoder_tokens'] # (B, N, hw, enc_dim)
        dec_tokens = preds['decoder_tokens'] # (B, N, hw, dec_dim)
        pos        = preds['tokens_pos']     # (B, N, hw, 2)
        B, N, clshw, _ = enc_tokens.shape
        assert clshw == hw+self.n_cls # enc and dec tokens contain CLS
        assert dec_tokens.shape[:3] == enc_tokens.shape[:3]
        assert pos.shape[:3] == enc_tokens.shape[:3]

        # merge both encoder and decoder tokens
        tokens = torch.cat((enc_tokens, dec_tokens), dim=-1).flatten(0,1)
        tokens = self.merge_enc_dec(tokens)
        tokens = tokens.reshape([B, N, clshw, -1])

        # split query/db
        query_tokens = tokens[:, -1]
        qpos = pos[:,-1].contiguous()

        db_tokens = tokens[:, :-1].flatten(1,2)
        db_pos = pos[:,:-1].flatten(1,2).contiguous()

        # build SA/CA masks
        sa_attn_mask = torch.zeros((B*hw), dtype=float, device=tokens.device)
        sa_attn_mask[q_idxs] = 1
        sa_attn_mask = sa_attn_mask.view([B,hw])
        sa_attn_mask = (sa_attn_mask[...,None]@sa_attn_mask[:,None]).view(B, hw, hw).bool()

        ca_attn_mask = torch.zeros((B*(N-1)*hw, hw), dtype=bool, device=tokens.device)
        ca_attn_mask[b_idxs, q_idxs%hw] = True
        ca_attn_mask = ca_attn_mask.view(B, (N-1), hw, hw)

        # add CLS tokens as ones to masks
        sa_attn_mask = torch.cat([sa_attn_mask.new_ones(B,      self.n_cls, sa_attn_mask.shape[-1]), sa_attn_mask], dim=-2)
        sa_attn_mask = torch.cat([sa_attn_mask.new_ones(B,      sa_attn_mask.shape[-2], self.n_cls), sa_attn_mask], dim=-1)
        ca_attn_mask = torch.cat([ca_attn_mask.new_ones(B, N-1, self.n_cls, ca_attn_mask.shape[-1]), ca_attn_mask], dim=-2)
        ca_attn_mask = torch.cat([ca_attn_mask.new_ones(B, N-1, ca_attn_mask.shape[-2], self.n_cls), ca_attn_mask], dim=-1)
        ca_attn_mask = ca_attn_mask.view(B, (N-1)*clshw, clshw).transpose(-2,-1).contiguous() # (B, clshw, (N-1)*clshw)


        output_query_tokens = self.vit(query_tokens, qpos, y=db_tokens, ypos=db_pos, sa_attn_mask=sa_attn_mask, ca_attn_mask=ca_attn_mask)

        # gather all token descriptors
        db_desc = self._gather_descs(db_tokens,           b_idxs, hw) # (B, ps**2, d)
        qy_desc = self._gather_descs(output_query_tokens, q_idxs, hw) # (B, ps**2, d)

        if self.last_mlp: # make them interact
            db_qy_desc = torch.cat((db_desc, qy_desc), dim=-1)
            db_qy_desc = db_qy_desc + self.pair_mlp(self.pair_norm(db_qy_desc))
            # split into query + db
            db_desc, qy_desc = db_qy_desc.tensor_split([self.dec_dim], dim=-1)

        db_desc, qy_desc = map(self._pixel_descs, (db_desc, qy_desc))

        # pixel-wise dot matching
        matching_with_null = self._dot_append_null_to_matching(db_desc, qy_desc)

        return b_idxs, q_idxs, matching_with_null # (n, ps**2, ps**2+1)

class DenseDotViT2Mlp_Kpt (DefaultPairs, DotViT2Mlp): pass
