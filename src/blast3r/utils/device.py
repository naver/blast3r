# Copyright (C) 2026-present Naver Corporation. All rights reserved.
#
# Adapted for BLASt3R from DUSt3R (https://github.com/naver/dust3r),
# dust3r/utils/device.py.

import numpy as np
import torch


def get_device(arr):
    if arr is None:
        return None
    elif isinstance(arr, np.ndarray):
        # WARNING: arr.device='cpu' in new numpy version!
        return 'numpy'
    elif isinstance(arr, torch.Tensor):
        return arr.device
    else:
        raise ValueError(f'{arr=} is not a numpy array nor a torch tensor!')


def get_device_str(arr):
    device = get_device(arr)
    if device is None:
        return 'none'
    if isinstance(device, str):
        return device
    s = device.type
    if s == 'cuda' and device.index:
        return f'cuda:{device.index}'
    return s


def todevice(batch, device, callback=None, non_blocking=False, contiguous=False):
    """Transfer some variables to another device (GPU, CPU:torch or CPU:numpy).

    Args:
        batch: list, tuple or dict of tensors, or anything else.
        device: pytorch device, or 'numpy'.
        callback: function called on every sub-element.
    """
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k:todevice(v, device) for k,v in batch.items()}

    if isinstance(batch, (tuple,list)):
        return type(batch)(todevice(x, device, contiguous=contiguous) for x in batch)

    x = batch
    if device == 'numpy':
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            if any(s<0 for s in x.strides):
                x = x.copy() # torch does not allow negative strides
            x = torch.from_numpy(x)
        if torch.is_tensor(x) and device != 'torch':
            x = x.to(device, non_blocking=non_blocking)
        if isinstance(x, torch.Tensor) and contiguous: x=x.contiguous()
    return x

to_device = todevice # alias

def to_numpy( x ): return todevice(x, 'numpy')
def to_cuda( x ): return todevice(x, 'cuda')
def to_torch( x ): return todevice(x, 'torch')


def collate_with_cat( whatever, lists=False ):
    if isinstance(whatever, dict):
        return {k:collate_with_cat(vals, lists=lists) for k,vals in whatever.items()}

    elif isinstance(whatever, (tuple, list)):
        if len(whatever) == 0:
            return whatever
        elem = whatever[0]
        T = type(whatever)

        if elem is None:
            return None
        if isinstance(elem, (bool, float, int, str)):
            return whatever
        if isinstance(elem, tuple):
            return T(collate_with_cat(x, lists=lists) for x in zip(*whatever))
        if isinstance(elem, dict):
            return {k:collate_with_cat([e[k] for e in whatever], lists=lists) for k in elem}

        if isinstance(elem, torch.Tensor):
            return listify(whatever) if lists else torch.cat(whatever)
        if isinstance(elem, np.ndarray):
            return listify(whatever) if lists else torch.cat([torch.from_numpy(x) for x in whatever])

        # otherwise, we just chain lists
        return sum(whatever, T())


def listify( elems ):
    return [x for e in elems for x in e]


# joint numpy & torch functions

def clone(arr):
    if isinstance(arr, np.ndarray):
        return arr.copy()
    elif isinstance(arr, torch.Tensor):
        return arr.clone()
    else:
        raise ValueError(f'{arr=} is not a numpy array nor a torch tensor!')


def to_type(typename):
    np_type = getattr(np,typename)
    torch_type = getattr(torch,typename)

    def to_type_func(arr):
        if isinstance(arr, np.ndarray):
            return arr.astype(np_type)
        elif isinstance(arr, torch.Tensor):
            return arr.to(torch_type)
        else:
            raise TypeError(f'bad {type(arr)=}')

    def is_type_func(arr):
        if isinstance(arr, np.ndarray):
            return arr.dtype == np_type
        elif isinstance(arr, torch.Tensor):
            return arr.dtype == torch_type
        else:
            raise TypeError(f'bad {type(arr)=}')

    return to_type_func, is_type_func

int32, is_int32 = to_type('int32')
int64, is_int64 = to_type('int64')
float32, is_float32 = to_type('float32')
float64, is_float64 = to_type('float64')

def expand(arr, *unsqueeze__new_shape):
    # called as expand(arr, 0, 1, (4, 5,6)) --> unsqueeze dims 0 and 1 then expand to (4,5,6)
    unsqueeze = unsqueeze__new_shape[:-1]
    new_shape = unsqueeze__new_shape[-1]

    if isinstance(arr, np.ndarray):
        for dim in unsqueeze:
            arr = np.expand_dims(arr,dim)
        assert arr.ndim == len(new_shape)
        return np.broadcast_to(arr, new_shape)
    elif isinstance(arr, torch.Tensor):
        for dim in unsqueeze:
            arr = arr.unsqueeze(dim)
        assert arr.ndim == len(new_shape)
        return arr.expand(new_shape)
    else:
        raise TypeError(f'bad {type(arr)=}')

def permute(arr, axes):
    if isinstance(arr, np.ndarray):
        return arr.transpose(*axes)
    elif isinstance(arr, torch.Tensor):
        return arr.permute(*axes)
    else:
        raise TypeError(f'bad {type(arr)=}')

def def_cat_stack(np_func, torch_func):
    def func(arr_list, dim=0):
        assert isinstance(arr_list, (list, tuple)) and arr_list
        arr = arr_list[0]
        if isinstance(arr, np.ndarray):
            return np_func(arr_list, axis=dim)
        elif isinstance(arr, torch.Tensor):
            return torch_func(arr_list, dim=dim)
        else:
            return arr_list
    return func

stack = def_cat_stack(np.stack, torch.stack)
concat = def_cat_stack(np.concatenate, torch.cat)

def def_array_like(np_func, torch_func):
    # this is how it's called
    def array_like(arr, shape=None, dtype=None):
        if isinstance(arr, np.ndarray):
            return np_func(shape or arr.shape, dtype=dtype or arr.dtype)
        elif isinstance(arr, torch.Tensor):
            return torch_func(shape or arr.shape, dtype=dtype or arr.dtype, device=arr.device)
        else:
            raise ValueError(f'{arr=} is not a numpy array nor a torch tensor!')
    return array_like

zeros_like = def_array_like(np.zeros, torch.zeros)
ones_like = def_array_like(np.ones, torch.ones)

def unique(arr, **kw):
    if isinstance(arr, np.ndarray):
        return np.unique(arr, **kw)
    elif isinstance(arr, torch.Tensor):
        return torch.unique(arr, **kw)
    else:
        raise TypeError(f'bad {type(arr)=}')

def unbind(arr, dim):
    if isinstance(arr, np.ndarray):
        return tuple(np.moveaxis(arr, dim, 0))
    elif isinstance(arr, torch.Tensor):
        return torch.unbind(arr, dim=dim)
    else:
        raise TypeError(f'bad {type(arr)=}')

def moveaxis(arr, src_dim, tgt_dim):
    if isinstance(arr, np.ndarray):
        return np.moveaxis(arr, src_dim, tgt_dim)
    elif isinstance(arr, torch.Tensor):
        return torch.moveaxis(arr, src_dim, tgt_dim)
    else:
        raise TypeError(f'bad {type(arr)=}')

def norm(arr, dim):
    if isinstance(arr, np.ndarray):
        return np.linalg.norm(arr, axis=dim)
    elif isinstance(arr, torch.Tensor):
        return torch.linalg.norm(arr, dim=dim)
    else:
        raise TypeError(f'bad {type(arr)=}')

def contiguous(arr):
    if isinstance(arr, np.ndarray):
        return np.ascontiguousarray(arr)
    elif isinstance(arr, torch.Tensor):
        return arr.contiguous()
    else:
        raise TypeError(f'bad {type(arr)=}')

def def_plain_func(np_func, torch_func):
    # this is how it's called
    def boolean_test(arr):
        if isinstance(arr, np.ndarray):
            return np_func(arr)
        elif isinstance(arr, torch.Tensor):
            return torch_func(arr)
        else:
            raise ValueError(f'{arr=} is not a numpy array nor a torch tensor!')
    return boolean_test

isfinite = def_plain_func(np.isfinite, torch.isfinite)
isnan = def_plain_func(np.isnan, torch.isnan)
exp = def_plain_func(np.exp, torch.exp)
log = def_plain_func(np.log, torch.log)

def def_func_with_axis_or_dim(np_func, torch_func):
    # this is how it's called
    def func(arr, dim=None):
        if isinstance(arr, np.ndarray):
            return np_func(arr, axis=None)
        elif isinstance(arr, torch.Tensor):
            return torch_func(arr, dim=dim)
        else:
            raise ValueError(f'{arr=} is not a numpy array nor a torch tensor!')
    return func

cumsum = def_func_with_axis_or_dim(np.cumsum, torch.cumsum)
