# Copyright (C) 2026-present Naver Corporation. All rights reserved.
#
# Adapted for BLASt3R from DUSt3R (https://github.com/naver/dust3r),
# dust3r/utils/image.py.

import os
import numpy as np
import PIL.Image
import torch
from blast3r.image import ImgNorm, tokenize
from blast3r.utils.geometry import xy_grid


def _resize_pil_image(img, long_edge_size, ret_scaling=False):
    scaling = 1
    if long_edge_size is not None:
        S = max(img.size)
        if S > long_edge_size:
            interp = PIL.Image.LANCZOS
        elif S <= long_edge_size:
            interp = PIL.Image.BICUBIC

        scaling = long_edge_size / S
        new_size = tuple(int(round(x*scaling)) for x in img.size)
        img = img.resize(new_size, interp)

    return (img, scaling) if ret_scaling else img


def load_images(folder_or_list, size, patch_size=16, force_ar=None, verbose=True, transform=None):
    """Convert all images in a folder to the model input format."""
    if transform is None:
        def transform(img): return ImgNorm(img)[None]

    if size == 224:
        force_ar = 1
    if force_ar:
        assert 0 < force_ar <= 1

    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    imgs = []
    for path in folder_content:
        if isinstance(path, np.ndarray):
            assert path.ndim == 3 and path.shape[2] == 3
            if np.issubdtype(path.dtype, np.floating):
                assert 0 <= path.min() and path.max() <= 1
                path = np.uint8(255 * path)
            img = PIL.Image.fromarray(path)
            path = '(numpy array)'
        else:
            if not "SfM-120k" in path and not path.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                continue
            img = PIL.Image.open(os.path.join(root, path)).convert('RGB')
            if size is not None:
                img = img.convert('RGB')

        W1, H1 = img.size
        halfw, halfh = cx, cy = W1//2, H1//2
        if force_ar:
            # crop to a certain aspect ratio
            if H1 / force_ar <= W1:  # crop the height
                halfw = int(cy / force_ar + 0.5)
                halfh = cy
            elif W1 / force_ar <= H1:  # crop the width
                halfw = cx
                halfh = int(cx / force_ar + 0.5)
            elif W1 * force_ar <= H1:  # crop height
                halfw = cx
                halfh = int(cx * force_ar + 0.5)
            elif H1 * force_ar <= W1:  # crop width
                halfw = int(cy * force_ar + 0.5)
                halfh = cy
            else:
                raise RuntimeError()
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        if size is not None:
            # resize long side to given size
            img = _resize_pil_image(img, size)
            # update size after resize
            W, H = img.size
            cx, cy = W//2, H//2

            # make sure we have multiple of 16
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {img.size[0]}x{img.size[1]}')


        width, height = img.size
        true_shape = np.int32((height, width))
        h = height//patch_size
        w = width//patch_size

        img = ImgNorm(img)
        img = tokenize(img, patch_size, chan_dim=-3)
        token_pos = xy_grid(w, h).reshape(-1,2)[:,[1,0]].astype(np.float32)

        img_dict = {
            "img": img,
            "true_shape": true_shape,
            "token_pos": token_pos,
            "idx": len(imgs),
            "instance": str(len(imgs)),
            "offset": np.int32([cx-halfw, cy-halfh])
        }
        imgs.append(img_dict)

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs


class DusterInputFromImageList(torch.utils.data.Dataset):

    def __init__(self, image_list, imsize=512, transform=None, patch_size=16):
        super().__init__()
        self.image_list = image_list
        self.imsize = imsize
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        return load_images([self.image_list[index]], transform=self.transform, size=self.imsize, verbose=False,
                           patch_size=self.patch_size)[0]
