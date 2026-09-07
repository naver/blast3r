# Copyright (C) 2026-present Naver Corporation. All rights reserved.

import torch
import torch.nn as nn

from .layers import Network, ViT, LinearPatchifier, MlpHead, Mlp
from .coarse_matcher import CoarseMatcher
from .dense_matcher import DenseMatcher
from .tools import AsListOfDicts

__all__ = 'CausalTrack3R CausalTrack3R_WithRayMaps'.split()


class Backbone (Network):
    """Lets a network be composed as `Track3R(...) * CoarseMatcher(...) * DenseMatcher(...)`."""
    def __init__(self):
        super().__init__()
        self.coarse_matcher = None
        self.dense_matcher = None

    def __mul__(self, net):
        if isinstance(net, CoarseMatcher):
            assert self.coarse_matcher is None
            self.coarse_matcher = net.init(self.flow_mode, self.dec_dim, self.n_cls)

        elif isinstance(net, DenseMatcher):
            assert self.dense_matcher is None
            self.dense_matcher = net.init(self.flow_mode, self.enc_dim, self.dec_dim, self.n_cls, self.patch_size)

        else:
            raise TypeError(f'net={type(net)} has a bad type')
        return self


class Track3R (Backbone):
    def __init__(self, patch_size=16, dim=(384,256), Patchifier=LinearPatchifier, posaug_factor=None, Head=MlpHead,
                 enc_depth=4, dec_depth=4, CA_rope=False, pts_mode='enc+dec', raymaps_mode='enc+dec', annot_proba_mode='enc+dec',
                 n_z_comps=5, pts3d_conf=True, manage_foreground=False, split_qkv=False, freeze=None, flow_mode='flow_to_tgt',
                 enc_num_heads=None, dec_num_heads=None):
        super().__init__()
        self.enc_dim, self.dec_dim = (dim,dim) if isinstance(dim,int) else dim # (encoder_dim, decoder_dim)
        self.patch_size = patch_size
        self.n_z_comps = n_z_comps
        self.pts3d_conf = pts3d_conf
        self.split_qkv = split_qkv
        self.manage_foreground = manage_foreground
        self.flow_mode = flow_mode

        # encoder
        self.encoder = ViT(patch_size, dim=self.enc_dim, Patchifier=Patchifier, posaug_factor=posaug_factor, depth=enc_depth,
                           null_tokens=4, manage_foreground=self.manage_foreground, split_qkv=self.split_qkv)
        assert enc_num_heads is None or self.encoder.num_heads == enc_num_heads, 'we should set head_size=enc_dim//enc_num_heads'
        self.n_cls = len(self.encoder.null_tokens)

        # keypoint confidence (encoder only)
        self.kpt_conf = Head(self.enc_dim, patch_size**2)

        # decoder
        self.query_to_decoder = nn.Linear(self.enc_dim, self.dec_dim)
        self.null_token_dec = nn.Parameter(torch.randn(self.dec_dim)/10)
        self.register_buffer('null_token_pos', torch.zeros(2, dtype=torch.int64), persistent=False)
        self.decoder = ViT(None, dim=self.dec_dim, depth=dec_depth, with_CA=True, with_CA_rope=CA_rope, split_qkv=self.split_qkv)
        assert dec_num_heads is None or self.decoder.num_heads == dec_num_heads, 'we should set head_size=dec_dim//dec_num_heads'

        self.pts_mode = pts_mode
        self._set_pts3d_head(Head)
        self.raymaps_mode = raymaps_mode
        self._set_raymaps_head(Head)

        self.annot_proba_mode = annot_proba_mode
        self._set_annot_proba_head(Head)

        assert freeze in [None,'encoder']
        self.freeze = freeze
        if freeze=='encoder':
            for p in self.encoder.parameters(): p.requires_grad = False

        # initialize weights
        self.apply(self._init_weights)

    def train(self, mode=True):
        super().train(mode=mode)
        if self.freeze=='encoder':
            self.encoder.eval()
            if hasattr(self.encoder, 'pos_augmentor'):
                self.encoder.pos_augmentor.train(mode=mode)
        return self

    def load_state_dict(self, *args, **kwargs):
        if 'dense_matcher.score_thr' in args[0].keys():
            args[0]['dense_matcher.score_thr'] = args[0]['dense_matcher.score_thr'].view([1,1])
        return super().load_state_dict(*args, **kwargs)

    def _set_raymaps_head(self, Head):
        pass # do nothing here, used for RayMaps class

    def _set_annot_proba_head(self, Head):
        # pts head
        inp_dim = {'enc':self.enc_dim, 'dec':self.dec_dim, 'enc+dec':self.enc_dim+self.dec_dim}[self.annot_proba_mode]
        pts_dim = self.patch_size**2 # single probabilty for each pixel
        self.head_annot_proba = Head(inp_dim, pts_dim)

    def _set_pts3d_head(self, Head):
        # pts head
        inp_dim = {'enc':self.enc_dim, 'dec':self.dec_dim, 'enc+dec':self.enc_dim+self.dec_dim}[self.pts_mode]
        pts_dim = (2 + self.n_z_comps + self.pts3d_conf) * self.patch_size**2 # desc + pts3d + conf
        self.head_pts3d = Head(inp_dim, pts_dim)

    def _head_pts3d(self, x, query_shape):
        pts3d_conf = self.head_pts3d(x)[:,self.n_cls:]
        B,ntokens,C = pts3d_conf.size()
        pts3d_conf = pts3d_conf.view(B,ntokens,self.n_z_comps+2+self.pts3d_conf,self.patch_size, self.patch_size)
        return pts3d_conf

    def _head_annot_proba(self, x, query_shape):
        annot_proba = self.head_annot_proba(x)[:,self.n_cls:]
        B,ntokens,C = annot_proba.size()
        annot_proba = annot_proba.view(B,ntokens,1,self.patch_size, self.patch_size)
        return annot_proba

    def dispatch_inputs_to_head(self, func, mode, query_tokens, y, query_shape):
        if mode == 'dec':
            out = func(y, query_shape)
        elif mode == 'enc':
            out = func(query_tokens, query_shape)
        elif mode == 'enc+dec':
            out = func(torch.cat((query_tokens, y), dim=-1), query_shape)
        return out

    def encode(self, views):
        imgs = views['img']
        B, N, ntokens, THREE, ph, pw = imgs.shape
        assert THREE == 3
        token_pos = views['token_pos']
        true_shape = views['true_shape']

        if not self.manage_foreground:
            if 'has_foreground_mask' in views:
                assert not torch.any(views['has_foreground_mask']), "You need to call with 'manage_foreground=True' with this data"
            foreground_mask = (None, None)
        else:
            foreground_mask = (views['has_foreground_mask'].flatten(0,1), views['foreground_mask'].flatten(0,1))

        with torch.set_grad_enabled(self.freeze is None or self.freeze!='encoder'):
            (tokens, pos, input_shape), patchified_imgcls = self.encoder(imgs.flatten(0,1), token_pos.flatten(0,1), true_shape=true_shape.flatten(0,1), return_patchified=True, foreground_mask=foreground_mask)
        n_tokens = ntokens + self.n_cls
        tokens = tokens.view(B, N*n_tokens, self.enc_dim)
        pos = pos.view(B, N*n_tokens, 2)

        # kpt confidence is computed based on query encoder tokens
        kpt_conf = self.kpt_conf(tokens)

        return tokens, pos, input_shape, kpt_conf, patchified_imgcls

    def _decoder(self, query_tokens, query_pos, query_shape, memory, memory_pos):
        # if there's no memory, make something up!
        if memory.shape[-2] == 0:
            memory = self.null_token_dec.view(1,1,-1).expand(*memory.shape[:-2], 1, -1)
            memory_pos = self.null_token_pos.view(1,1,2).expand(*memory.shape[:-2], 1, 2)

        # forward pass in decoder
        query_dec_input = self.query_to_decoder(query_tokens)
        all_layers_tokens, y = self.decoder(query_dec_input, query_pos, y=memory, ypos=memory_pos, ret_x='all')

        # pts3d prediction head
        pts3d_conf = self.dispatch_inputs_to_head(self._head_pts3d, self.pts_mode, query_tokens, y, query_shape)

        # annotation probability prediction head
        annot_proba = self.dispatch_inputs_to_head(self._head_annot_proba, self.pts_mode, query_tokens, y, query_shape)

        # split pts3d and multidepth if needed
        multidepth = None
        if isinstance(pts3d_conf, dict):
            multidepth = pts3d_conf['multidepth']
            pts3d_conf = pts3d_conf['pts3d_conf']

        output = dict(
            hw = query_tokens.shape[1] - self.n_cls,
            input_shape = query_shape,
            patch_size = self.patch_size,
            mem_tokens = memory,
            query_decoder_tokens = all_layers_tokens,
            query_pts3d = pts3d_conf[:,:,:,:-1 if self.pts3d_conf else None],
            query_pts3d_conf = pts3d_conf[:,:,:,-1] if self.pts3d_conf else torch.ones_like(pts3d_conf[...,0]),
            multidepth = multidepth,
            annot_proba = annot_proba,
            coarse_matcher = self.coarse_matcher,
            dense_matcher = self.dense_matcher,
            mode='pointmaps',
        )
        return output

    def forward(self, views, step=None, criterion=None):
        """Encode the views and run the matcher heads.

        `criterion` is accepted here because DistributedDataParallel forbids using
        model parameters inside the loss; passing it in makes the forward compute
        the loss itself.
        """
        assert step is not None
        views = AsListOfDicts(views)

        # encode views
        Tokens, Pos, input_shape, kpt_conf, patchified_imgcls = self.encode(views)

        # decode by comparing with previous views
        results = self.decode_loop(views, Tokens, Pos, input_shape, self.n_cls, kpt_conf=kpt_conf, patchified_imgcls=patchified_imgcls)

        if criterion is not None:
            loss, loss_details = criterion(results, views)
            return loss, loss_details
        else:
            return results


class CausalTrack3R (Track3R):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = nn.LayerNorm(3*self.dec_dim)
        self.decoder_to_memory = Mlp(3*self.dec_dim, 4*self.dec_dim, self.dec_dim)

    def get_n_mem_views_list(self, n, rng): # should be sorted!!!
        return list(range(n))

    def decode_loop(self, views, Tokens, Pos, input_shape, n_cls, **encoder_outputs):
        hw = input_shape[0,0]*input_shape[0,1]
        assert torch.all(input_shape[:,0]*input_shape[:,1]==hw)
        n_tokens = n_cls + hw
        assert Tokens.shape[-2] % n_tokens == 0

        memory = torch.empty((views.B,0,self.dec_dim), device=self.device)
        memory_pos = torch.empty((views.B,0,2), dtype=torch.int64, device=self.device)

        patchified_imgcls = encoder_outputs.pop('patchified_imgcls')
        results = {}
        for view_idx in range(views.N):
            n = view_idx * n_tokens
            query = Tokens[:, n:n+n_tokens]
            pos = Pos[:, n:n + n_tokens].contiguous()

            # compare query with decoder
            results[view_idx] = res = self._decoder(query, pos, input_shape, memory, memory_pos)
            results[view_idx].update(encoder_outputs)

            seltokens = lambda x, dim:x[:,:n+n_tokens].reshape([views.B, -1, n_tokens, dim]) # (B, N*hw, D) -> (B, N, hw, D)
            results[view_idx]['encoder_tokens'] = seltokens(Tokens, self.enc_dim) # all encoded images so far, used for some dense matchers
            results[view_idx]['tokens_pos'] = seltokens(Pos, 2)
            results[view_idx]['patchified_imgcls'] = seltokens(patchified_imgcls, self.enc_dim)

            # add them to the memory
            # one entry per decoder block plus the input, so dec_depth >= 2 here;
            # decoder_to_memory takes the last three layers concatenated
            tokens2 = res['query_decoder_tokens']
            tokens2 = torch.cat((tokens2[-3], tokens2[-2], tokens2[-1]), dim=-1)
            tokens2 = self.decoder_to_memory(self.norm(tokens2))

            memory = torch.cat((memory, tokens2), dim=1)
            memory_pos = torch.cat((memory_pos, pos), dim=1)

            results[view_idx]['decoder_tokens'] = seltokens(memory, self.dec_dim) # all decoded tokens so far, used for some dense matchers

        return results


class RayMaps_Track3R(Track3R):
    """Track3R with the pts3d head split into [raymaps+conf] and [nzcomps].

    Multi-depth and raymap predictions are made separately.
    """
    def _set_pts3d_head(self, Head):
        if self.n_z_comps == 0:
            self.head_pts3d = None # no multidepth
        else:
            # pts head
            inp_dim = {'enc':self.enc_dim, 'dec':self.dec_dim, 'enc+dec':self.enc_dim+self.dec_dim}[self.pts_mode]
            pts_dim = self.n_z_comps * self.patch_size**2 # pts3d head is now only predicting zcomps, legacy naming
            self.head_pts3d = Head(inp_dim, pts_dim)

    def _set_raymaps_head(self, Head):
        # pts head
        inp_dim = {'enc':self.enc_dim, 'dec':self.dec_dim, 'enc+dec':self.enc_dim+self.dec_dim}[self.raymaps_mode]
        raymap_dim = (2 + self.pts3d_conf) * self.patch_size**2 # raymaps are 2D + confidence
        self.head_raymaps = Head(inp_dim, raymap_dim)

    def _head_raymaps(self, x, query_shape):
        raymaps = self.head_raymaps(x)[:,self.n_cls:]
        B,ntokens,C = raymaps.size()
        raymaps = raymaps.view(B,ntokens,2+self.pts3d_conf,self.patch_size, self.patch_size)
        return raymaps

    def _head_mdepths(self, x, query_shape):
        if self.head_pts3d is None:
            mdepths=None
        else:
            mdepths = self.head_pts3d(x)[:,self.n_cls:]
            B,ntokens,C = mdepths.size()
            mdepths = mdepths.view(B,ntokens,self.n_z_comps,self.patch_size, self.patch_size)
        return mdepths

    def _decoder(self, query_tokens, query_pos, query_shape, memory, memory_pos):
        # if there's no memory, make something up!
        if memory.shape[-2] == 0:
            memory = self.null_token_dec.view(1,1,-1).expand(*memory.shape[:-2], 1, -1)
            memory_pos = self.null_token_pos.view(1,1,2).expand(*memory.shape[:-2], 1, 2)

        # forward pass in decoder
        query_dec_input = self.query_to_decoder(query_tokens)
        all_layers_tokens, y = self.decoder(query_dec_input, query_pos, y=memory, ypos=memory_pos, ret_x='all')

        # multidepth prediction head
        mdepths = self.dispatch_inputs_to_head(self._head_mdepths, self.pts_mode, query_tokens, y, query_shape)

        # raymaps prediction head
        raymaps_conf = self.dispatch_inputs_to_head(self._head_raymaps, self.raymaps_mode, query_tokens, y, query_shape)

        output = dict(
            hw = query_tokens.shape[1] - self.n_cls,
            input_shape = query_shape,
            patch_size = self.patch_size,
            mem_tokens = memory,
            query_decoder_tokens = all_layers_tokens,
            query_pts3d = raymaps_conf[...,:-1 if self.pts3d_conf else None],
            query_pts3d_conf = raymaps_conf[...,-1] if self.pts3d_conf else torch.ones_like(raymaps_conf[...,0]),
            multidepth = mdepths,
            coarse_matcher = self.coarse_matcher,
            dense_matcher = self.dense_matcher,
            mode='raymaps',

        )
        return output

class CausalTrack3R_WithRayMaps(RayMaps_Track3R, CausalTrack3R): pass # override Track3R methods and attributes with RayMaps_Track3R ones

