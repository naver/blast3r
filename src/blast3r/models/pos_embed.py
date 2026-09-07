# Copyright (C) 2026-present Naver Corporation. All rights reserved.
#
# Adapted for BLASt3R from CroCo (https://github.com/naver/croco),
# models/pos_embed.py, which in turn derives from MAE
# (https://github.com/facebookresearch/mae), licensed under CC BY-NC 4.0.

import numpy as np
import torch
from blast3r.extensions.curope import RoPE2d_func


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
# The sincos position embeddings below are from MAE (https://github.com/facebookresearch/mae) by Meta Platforms, Inc.
# Original code licensed under CC BY-NC 4.0.
def get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0):
    """2D sine-cosine position embedding for a square token grid.

    Args:
        grid_size: (grid height, grid width).

    Returns:
        pos_embed: (grid_size*grid_size, embed_dim), or
            (n_cls_token+grid_size*grid_size, embed_dim) with cls tokens.
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if n_cls_token>0:
        pos_embed = np.concatenate([np.zeros([n_cls_token, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """1D sine-cosine position embedding for a list of positions.

    Args:
        embed_dim: output dimension for each position.
        pos: (M,) positions to encode.

    Returns:
        out: (M, D).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def interpolate_pos_embed(model, checkpoint_model):
    keys = sum([[f'{encdec}_pos_embed',f'_{encdec}_cosine_embed'] for encdec in ['enc']+(['dec'] if hasattr(model,'dec_blocks') else [])],[])
    img_size = (model.img_size,model.img_size) if isinstance(model.img_size,int) else model.img_size
    for k in keys:
        if k in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model[k]
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_extra_tokens = model.n_cls_token
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = (img_size[0]//model.patch_size,img_size[1]//model.patch_size)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size[0] or orig_size != new_size[1]:
                print("Position interpolate %s from %dx%d to %dx%d" % (k, orig_size, orig_size, new_size[0], new_size[1]))
                extra_tokens = pos_embed_checkpoint[:num_extra_tokens,:]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[num_extra_tokens:,:]
                pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size[0], new_size[1]), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
                checkpoint_model[k] = new_pos_embed.squeeze(0)

# per-block embeddings (e.g. rotary)

import torch
import torch.nn as nn


class BlockPosEmbedding (nn.Module):
    """Base class for positional embeddings.

    Supports both standard and rotary embeddings, in 1d and 2d.
    """
    def __init__(self, dim, num_cls):
        super().__init__()
        dim_div = 2 # 2D data
        self.dim = dim // dim_div
        self.num_cls = num_cls
        assert self.dim * dim_div == dim

    def __repr__(self):
        return type(self).__name__ + f"(dim=2*{self.dim})"

    def apply_after_qkv(self, tokens, pos):
        return self._apply_posenc(getattr(self,'_apply_after_qkv',None), tokens, pos)

    def _apply_posenc(self, func, tokens, pos):
        if func is not None:
            assert tokens.ndim == 4 # (B, Head, Seq, Dim)
            assert pos.ndim == 3 and pos.shape[-1] == 2 # (Batch, Seq, 2)
            y, x = tokens[:,:,self.num_cls:].chunk(2, dim=-1)
            y = func(y, pos[:,self.num_cls:,0])
            x = func(x, pos[:,self.num_cls:,1])
            tokens = torch.cat((y, x), dim=-1)
            if self.num_cls > 0:
                tokens = torch.cat((tokens[:,:,self.num_cls:], tokens), dim=2)
        return tokens, pos



class cuRoPE (BlockPosEmbedding):
    def __init__(self, _dim, num_heads, num_cls, freq=None, F0=1, max_seq_len=None):
        super().__init__(_dim//num_heads, num_cls)
        self.base = freq or self.FREQ
        self.F0 = F0

    def apply_after_qkv(self, tokens, pos):
        if self.num_cls > 0:
            tokens = tokens.clone()
        RoPE2d_func.apply( tokens.transpose(1,2), pos, self.base, self.F0 )
        return tokens, pos


class cuRoPE100 (cuRoPE):
    FREQ  = 100
