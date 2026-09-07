# Copyright (C) 2026-present Naver Corporation. All rights reserved.

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as tvf
from PIL import Image, ImageOps
from tqdm import tqdm

from blast3r.utils.device import to_numpy, moveaxis
from blast3r.utils.geometry import xy_grid
from blast3r.utils.parallel import parallel_threads

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')

ImgNorm = tvf.Compose([tvf.ToImage(), tvf.ToDtype(torch.float32, scale=True),
                       tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def detokenize(arr, shape, chan_dim=-1):
    # arr: (*, hw, P, P, ?)
    if chan_dim is None:
        arr = arr[..., None]
    elif chan_dim != -1:
        assert chan_dim % arr.ndim == arr.ndim-3
        arr = moveaxis(arr, chan_dim, -1)
    Bs = arr.shape[:-4]
    hw, P, P_, d = arr.shape[-4:]
    assert P == P_

    H, W = shape
    h, w = H//P, W//P
    assert hw == h*w

    arr = arr.reshape(-1, h, w, P, P, d)
    arr = arr.swapaxes(2,3)
    arr = arr.reshape(*Bs, H, W, d)

    if chan_dim is None:
        arr = arr[...,0]
    elif chan_dim != -1:
        arr = moveaxis(arr, -1, chan_dim)
    return arr


def tokenize(arr, P, chan_dim=-1):
    H, W = arr.shape[1:] if chan_dim==-3 else arr.shape[:2]
    h = H//P
    w = W//P
    n = h*w
    if chan_dim is not None:
        d = arr.shape[chan_dim]
    assert h*w*P*P == H*W, f'image size ({H},{W}) is not multiple of patch_size = {P}'

    if chan_dim is None: # depth, mask, ...
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 2
        return arr.reshape(h,P,w,P).transpose(0,2,1,3).reshape(n,P,P)

    if chan_dim == -3: # an image
        assert isinstance(arr, torch.Tensor)
        assert arr.ndim == 3 and d == 3
        return arr.reshape(d,h,P,w,P).permute(1,3,0,2,4).reshape(h*w,d,P,P)

    if chan_dim == -1: # pointmap or ij_grid
        assert arr.ndim == 3
        if isinstance(arr, np.ndarray):
            return arr.reshape(h,P,w,P,d).transpose(0,2,1,3,4).reshape(n,P,P,d)
        elif isinstance(arr, torch.Tensor):
            return arr.reshape(h,P,w,P,d).permute(0,2,1,3,4).reshape(n,P,P,d)

    raise ValueError(f'bad {chan_dim=}, should be -1, -3 or None')


def RGB( x ):
    is_not_rgb = (x.min() < 0) or (x.max() <= 2)
    if is_not_rgb:
        if x.ndim == 4 and x.shape[1] == 3:
            return to_numpy(x/2+0.5).transpose(0,2,3,1)
        elif x.ndim == 3 and x.shape[0] == 3:
            return to_numpy(x/2+0.5).transpose(1,2,0)
    return to_numpy(x)


def RGB_dtk(view, batch_idx=None):
    img = view['img']
    if batch_idx is not None:
        img = img[batch_idx]

    if img.shape[-1] == img.shape[-2]:
        shape = to_numpy(view['true_shape'])
        if batch_idx is not None:
            shape = shape[batch_idx]
        if np.asarray(shape).ndim == 2:
            assert (shape[0] == shape).all(), 'cannot detokenize with multiple shapes'
            shape = shape[0]
        img = detokenize(img, shape, chan_dim=-3)
    img = RGB(img)
    return img


def _resize_pil_image(img, long_edge_size, ret_scaling=False):
    scaling = 1
    if long_edge_size is not None:
        S = max(img.size)
        if img.mode.startswith(('I','F')): # depth image
            interp = Image.NEAREST
        elif S > long_edge_size:
            interp = Image.LANCZOS
        elif S <= long_edge_size:
            interp = Image.BICUBIC

        scaling = long_edge_size / S
        new_size = tuple(int(round(x*scaling)) for x in img.size)
        img = img.resize(new_size, interp)

    return (img, scaling) if ret_scaling else img


def resize_and_crop(img, size, force_ar=None, patch_size=16, return_transform=False):
    W1, H1 = img.size
    halfw, halfh = cx, cy = W1//2, H1//2

    C = np.eye(3)
    if force_ar:
        # crop to a certain aspect ratio
        if H1 / force_ar <= W1: # crop the height
            halfw = int(cy / force_ar + 0.5)
            halfh = cy
        elif W1 / force_ar <= H1: # crop the width
            halfw = cx
            halfh = int(cx / force_ar + 0.5)
        elif W1 * force_ar <= H1: # crop height
            halfw = cx
            halfh = int(cx * force_ar + 0.5)
        elif H1 * force_ar <= W1:  # crop width
            halfw = int(cy * force_ar + 0.5)
            halfh = cy
        else:
            raise RuntimeError()
        img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        # record crop transform
        c1 = np.eye(3)
        c1[0,2] = -(cx - halfw)
        c1[1,2] = -(cy - halfh)
        C = c1 @ C

    if size is not None:
        # resize long side to given size
        img, scaling = _resize_pil_image(img, size, ret_scaling=True)
        # update size after resize
        W, H = img.size
        cx, cy = W//2, H//2

        c2 = np.eye(3)
        c2[0,0] = scaling
        c2[1,1] = scaling
        C = c2 @ C

        # make sure we have multiple of patch_size
        half_ps = patch_size // 2
        halfw, halfh = ((2*cx)//patch_size)*half_ps, ((2*cy)//patch_size)*half_ps
        img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        c3 = np.eye(3)
        c3[0,2] = -(cx-halfw)
        c3[1,2] = -(cy-halfh)
        C = c3 @ C

    if return_transform:
        return img, C

    return img


def load_image(impath, size=224, patch_size=16, force_ar=None, with_ij_grid=False, return_transform=False):
    impath = Path(impath)
    img = Image.open(impath)
    img = ImageOps.exif_transpose(img)   # applies Orientation if present
    img = img.convert("RGB")

    img = resize_and_crop(img, size=size, patch_size=patch_size, force_ar=force_ar, return_transform=return_transform)
    if return_transform:
        img, C = img
        transform = C

    img = ImgNorm(img) # to float32 in [-1,1]
    H, W = img.shape[1:]
    tokenized_img = tokenize(img, patch_size, chan_dim=-3)
    view = dict(
        instance = impath,
        img = tokenized_img,
        true_shape = np.int32((H,W)),
        patch_size = np.int32(patch_size),
        token_pos = xy_grid(W//patch_size, H//patch_size).reshape(-1,2)[:,[1,0]].astype(np.float32),
        has_foreground_mask = np.array(False),
    )
    if with_ij_grid:
        view['ij_grid'] = tokenize(xy_grid(W, H), patch_size)

    if return_transform:
        return view, transform

    return view


def load_images(image_list, **kw):
    views = parallel_threads(lambda img: load_image(img, **kw), image_list, desc='loading imgs')

    if kw.get('return_transform', False):
         views, transforms = zip(*views)
         return list(views), list(transforms)

    return views


def is_video_file(path):
    # OpenCV opens a still image as a one-frame video, so rule those out first
    if Path(path).suffix.lower() in IMAGE_SUFFIXES:
        return False

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Different backends behave differently; this is a pragmatic check.
    if frame_count > 0 and fps > 0:
        return True

    # fallback: try to read a frame
    cap = cv2.VideoCapture(str(path))
    ok, _ = cap.read()
    cap.release()
    return ok


def load_video(video_file, size=224, patch_size=16, force_ar=None):
    views = []

    cap = cv2.VideoCapture(str(video_file))
    print("Pre-loading frames...")
    for fid, (ok, frame) in enumerate(tqdm(iter(cap.read, (False, None)))):
        if ok:
            img = resize_and_crop(Image.fromarray(frame), size=size, patch_size=patch_size, force_ar=force_ar)
            img = ImgNorm(img) # to float32 in [-1,1]
            H, W = img.shape[1:]
            tokenized_img = tokenize(img, patch_size, chan_dim=-3)
            views.append(dict(
                instance = Path(video_file) / str(fid),
                img = tokenized_img,
                true_shape = np.int32((H,W)),
                patch_size = np.int32(patch_size),
                token_pos = xy_grid(W//patch_size, H//patch_size).reshape(-1,2)[:,[1,0]].astype(np.float32),
                has_foreground_mask = np.array(False),
            ))
    print(f"Found {len(views)} frames for video file {video_file}")
    return views
