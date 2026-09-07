# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Turning bundle-adjustment output into the saved scene dictionary."""
import numpy as np
import torch

from blast3r.image import RGB_dtk


def get_at(views, idx):
    return {key: val[idx] for key, val in views.items()}


def init_view(view, conf, pts3d_in_cam, multidepth, depth_mode, optim_K, optim_P):
    new_view = {
        # displayed and saved as uint8 only, and kept on the host for every frame
        'img': compress('img', RGB_dtk(view)),
        'pts3d': pts3d_in_cam.float(),
        'conf': conf.float(),
        depth_mode: multidepth.float(),
        'not_too_far': 1,
    }

    if not any(optim_K):  # never optimized, so use the ground truth if we have it
        try:
            new_view['K'] = view['K'].float()
        except KeyError:
            print('Warning: ground-truth K is unknown, using initial estimate')

    if not any(optim_P):
        new_view['cam2w'] = view['cam2w']

    return new_view


EXPORT_KEYS = {
    'pose': ('K', 'cam2w'),
    'full': ('img', 'K', 'cam2w', 'depth', 'conf'),
}


def compress(key, val):
    """Store the display fields at the smallest dtype that loses nothing visible."""
    if key == 'img' and val.dtype != np.uint8:
        return (255 * val.clip(0, 1)).round().astype(np.uint8)
    if key == 'conf':
        return val.astype(np.float16)
    return val


def load_results(path):
    """Read a saved scene back as the (image names, views) the pipeline works with.

    Images come back as uint8 whatever the file holds, since older outputs stored
    them as floats, and confidence as float32, which is what denoising expects.
    """
    scene = torch.load(path, weights_only=False)
    views = scene['views'] if 'views' in scene else scene

    out = []
    for view in views.values():
        view = dict(view)
        if 'img' in view:
            view['img'] = compress('img', view['img'])
        if 'conf' in view:
            view['conf'] = view['conf'].astype(np.float32)
        out.append(view)

    return list(views), out


def prepare_results(result_list, image_names, extra_data=None, export_mode='full'):
    keys_to_save = EXPORT_KEYS[export_mode]

    out = {}
    for view, img_name in zip(result_list, image_names):
        view_out = {}
        for key, val in view.items():
            if key not in keys_to_save:
                continue
            val = val.cpu().numpy() if isinstance(val, torch.Tensor) else val
            view_out[key] = compress(key, val)
        out[img_name] = view_out

    out_all = {'views': out}
    if extra_data is not None:
        out_all.update(extra_data)
    return out_all
