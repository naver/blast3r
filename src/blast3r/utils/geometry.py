# Copyright (C) 2026-present Naver Corporation. All rights reserved.
#
# Adapted for BLASt3R from DUSt3R (https://github.com/naver/dust3r),
# dust3r/utils/geometry.py.

import numpy as np
import torch
import torch.nn.functional as F
import functools
import warnings

OPENGL = np.float32([[1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 0, 1]])

def clone(arr):
    if isinstance(arr,np.ndarray):
        return arr.copy()
    else:
        return arr.clone()

def cache_and_copy(func):
    # create an internal cached version of the function
    cached_func = functools.lru_cache(maxsize=None)(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        arr = cached_func(*args, **kwargs)
        # returns a deep copy to avoid mutation
        return clone(arr)

    return wrapper


@cache_and_copy
def xy_grid( W, H, device='numpy', offset=None, unsqueeze=None, cat_dim=-1, homogeneous=False, dtype=None ):
    """Output a (H, W, 2) array of pixel coordinates.

    output[j,i,0] = i + offset[0] and output[j,i,1] = j + offset[1].
    """
    if device == 'numpy':
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    elif device:
        # torch
        arange = lambda *a,**kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a, **kw: torch.ones(*a, device=device, **kw)
    else:
        raise ValueError(f'bad {device=}')

    if offset is None:
        offset = (0,0)
    elif isinstance(offset, (int,float)):
        offset = (offset,offset)
    tw, th = [arange(o,o+s, dtype=dtype) for s,o in zip((W,H), offset)]
    grid = list(meshgrid(tw, th, indexing='xy'))
    if homogeneous:
        grid.append( ones((H,W), dtype=dtype) )
    if unsqueeze is not None:
        grid = [g.unsqueeze(unsqueeze) for g in grid]
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def normalize( x, axis=-1 ):
    assert isinstance(x, np.ndarray), 'todo'
    return x / np.linalg.norm(x, axis=axis, keepdims=True)


def bmv( Trf, pts, ncol=None, norm=False, zclip=None):
    """Batched matrix-vector multiplication.

    Args:
        H: 3x3 or 4x4 projection matrix, typically a homography or SE(3) [R|T].
        p: numpy/torch/tuple of coordinates, of shape (..., 2) or (..., 3).
        ncol: number of columns of the result, 2 or 3.
        norm: if non-zero, the result is projected onto the z=norm plane.

    Returns:
        An array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
        Trf.ndim == 3 and pts.ndim == 4):
            d = pts.shape[3]
            if Trf.shape[-1] == d:
                pts = pts @ Trf[:,None,:,:].transpose(-1,-2)
            elif Trf.shape[-1] == d+1:
                pts = (pts @ Trf[:,None,:d,:d].transpose(-1,-2)) + Trf[:,None,None,:d,d]
            else:
                raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'

            if pts.ndim > Trf.ndim:
                # Trf == (*B, d, d) & pts == (*B, H, W, d) -> (*B, H*W, d)
                pts = pts.reshape(*Trf.shape[:n], -1, pts.shape[-1])
            elif pts.ndim == n+1:
                # Trf == (*B, d, d) & pts == (*B, d) -> (*B, 1, d)
                pts = pts[..., None, :]

        if pts.shape[-1]+1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1,-2) # transpose Trf
            pts = pts @ Trf[...,:-1,:] + Trf[...,-1:,:]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1,-2) # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= n+1: pts = pts.swapaxes(-1,-2)

    if norm:
        depth = pts[...,-1:]
        if zclip: depth = depth.clip(min=zclip)
        pts = pts / depth # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1: pts = pts * norm

    res = pts[...,:ncol].reshape(*output_reshape, ncol)
    return res


def depthmap_to_pts3d(depth, intrinsics, cam2world=None, stride=None):
    if stride is not None:
        sub = slice(stride//2, None, stride)
        depth = depth[..., sub, sub]
        intrinsics = clone(intrinsics)
        intrinsics[..., :2, 2] -= sub.start
        intrinsics[..., :2, :] /= stride

    H, W = depth.shape[-2:]
    device = 'numpy' if isinstance(depth, np.ndarray) else depth.device
    pixel_grid = xy_grid(W, H, offset=0.5, device=device, dtype=depth.dtype)
    pts3d = bmv(inv(intrinsics), pixel_grid, ncol=3)
    pts3d *= depth[...,None]
    if cam2world is not None:
        pts3d = bmv(cam2world, pts3d)

    return pts3d if stride is None else (pts3d, sub)


def sparse_depth_to_pts3d(pixels, depths, K, cam2world=None):
    n = len(pixels)
    assert pixels.shape == (n,2)
    assert depths.shape == (n,1)
    pc = depths * bmv(inv(K), pixels, ncol=3)

    if cam2world is not None:
        pc = bmv(cam2world, pc) # to world coords
    return pc


def bilinear_sampling(depth, pix2d, mode='bilinear', padding_mode='border'):
    to_numpy = False

    to_torch = lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x

    if isinstance(depth, np.ndarray) or isinstance(pix2d, np.ndarray):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors")
            depth = to_torch(depth)
            pix2d = to_torch(pix2d)
        to_numpy = True

    nb = depth.ndim - 3
    batch_dims = depth.shape[:nb]
    B = int(np.prod(batch_dims))
    pix_shape = pix2d.shape[nb:-1]
    assert depth.shape[:nb] == pix2d.shape[:nb], 'depth and pix2d must have the same batch dimension'
    depth = depth.reshape(B, *depth.shape[nb:]) # add a fake batch dim if necessary
    pix2d = pix2d.reshape(B, 1, np.prod(pix_shape), 2) # make sure it is 4-dim
    B, imH, imW, C = depth.shape

    # [0,W]x[0,H] now maps to [-1,1]x[-1,1]
    # in other words, the center of a pixel (i,j) is at [i+0.5,j+0.5]
    grid = pix2d * pix2d.new([2/imW,2/imH])  - 1
    res = F.grid_sample(depth.permute(0,3,1,2),
                        grid.to(depth.device),
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=False) # align with corners of the pixels
    res = res.permute(0,2,3,1).view(*batch_dims,*pix_shape,depth.shape[-1])
    if to_numpy: res = res.numpy()
    return res


def inv( mat ):
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')


def pixel_unshuffle(arr, ps): # (B, C, h*ps, w*ps) -> (B, C*ps*ps, h, w)
    if isinstance(arr, np.ndarray):
        B = arr.shape[:-3]
        C, H, W = arr.shape[-3:]
        assert H % ps == W % ps == 0
        arr = arr.reshape(*B, C, H//ps, ps, W//ps, ps)
        arr = arr.transpose(*range(len(B)+1), -3, -1, -4, -2)
        return arr.reshape(*B, C*ps*ps, H//ps, W//ps)
    elif isinstance(arr, torch.Tensor):
        return F.pixel_unshuffle(arr, ps)
    else:
        raise TypeError(f'bad {type(arr)=}')
