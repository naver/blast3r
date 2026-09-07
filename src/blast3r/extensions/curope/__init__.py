# Copyright (C) 2026-present Naver Corporation. All rights reserved.
#
# Adapted for BLASt3R from CroCo (https://github.com/naver/croco),
# models/curope/curope2d.py.

import torch

try:
    from . import cuda_rope as _kernels
except ImportError:
    _kernels = None
    import sys
    print('Warning: the cuRoPE CUDA extension is not compiled; reinstall with '
          '`pip install --no-build-isolation -e .` and a working CUDA_HOME', file=sys.stderr)


class RoPE2d_func (torch.autograd.Function):
    """Rotary 2D position embedding, computed in place."""
    @staticmethod
    def forward(ctx, tokens, positions, base, F0=1):
        if not _kernels:
            raise RuntimeError('the cuRoPE CUDA extension is not compiled; reinstall with '
                               '`pip install --no-build-isolation -e .` and a working CUDA_HOME')
        ctx.save_for_backward(positions)
        ctx.saved_base = base
        ctx.saved_F0 = F0
        _kernels.rope_2d( tokens, positions, base, F0 )
        ctx.mark_dirty(tokens)
        return tokens

    @staticmethod
    def backward(ctx, grad_res):
        positions, base, F0 = ctx.saved_tensors[0], ctx.saved_base, ctx.saved_F0
        _kernels.rope_2d( grad_res, positions, base, -F0 )
        ctx.mark_dirty(grad_res)
        return grad_res, None, None, None


class RoPE3d_func (torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, positions, base, F0=1):
        ctx.save_for_backward(positions)
        ctx.saved_base = base
        ctx.saved_F0 = F0
        _kernels.rope_3d( tokens, positions, base, F0 )
        ctx.mark_dirty(tokens)
        return tokens

    @staticmethod
    def backward(ctx, grad_res):
        positions, base, F0 = ctx.saved_tensors[0], ctx.saved_base, ctx.saved_F0
        _kernels.rope_3d( grad_res, positions, base, -F0 )
        ctx.mark_dirty(grad_res)
        return grad_res, None, None, None
