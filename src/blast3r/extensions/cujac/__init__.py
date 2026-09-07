# Copyright (C) 2026-present Naver Corporation. All rights reserved.

import torch

# dedicated CUDA kernels to compute residuals and Jacobians

try:
    from . import ba_with_rigs_cuda_kernels
except ImportError:
    def NoCUDA(*args, **kwargs):
        raise RuntimeError('The bundle-adjustment CUDA extension is not compiled; '
                           'reinstall with `pip install --no-build-isolation -e .` and a working CUDA_HOME')
    class DummyModule:
        def __getattr__(self, n):
            return NoCUDA
    ba_with_rigs_cuda_kernels = DummyModule()

compute_residuals_with_rigs = ba_with_rigs_cuda_kernels.residuals_with_rigs
compute_jacobians_with_rigs = ba_with_rigs_cuda_kernels.jacobians_with_rigs
chol3x3_solve = ba_with_rigs_cuda_kernels.chol3x3_solve
blockcoo_matvec = ba_with_rigs_cuda_kernels.blockcoo_matvec
fused_einsum_bki_bkj = ba_with_rigs_cuda_kernels.fused_einsum_bki_bkj


# Lie-algebra update functions for SE(3)

def skew_symmetric(w, device):
    """Convert a 3D vector into a skew-symmetric matrix.

    Args:
        w: (B, 3) tensor.
    """
    wx, wy, wz = w.T
    res = w.new_zeros((len(w), 3, 3), device=device)
    res[:, 2, 1] = wx
    res[:, 1, 2] =-wx
    res[:, 2, 0] =-wy
    res[:, 0, 2] = wy
    res[:, 0, 1] =-wz
    res[:, 1, 0] = wz
    return res

def hat_se3(xi):
    """Convert a (B, 6) se(3) vector into a (B, 4, 4) pose matrix.

    Args:
        xi: (B, 6) tensor of [w_x, w_y, w_z, v_x, v_y, v_z].
    """
    w = xi[:,:3]  # rotation part
    v = xi[:,3:]  # translation part
    hat = xi.new_zeros((len(xi), 4, 4))
    hat[:, :3, :3] = skew_symmetric(w, device=xi.device)
    hat[:, :3, 3] = v
    return hat

def exp_se3(xi):
    """Matrix exponential of an se(3) element, returning an element of SE(3)."""
    return torch.matrix_exp(hat_se3(xi))
