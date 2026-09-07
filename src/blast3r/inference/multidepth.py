# Copyright (C) 2026-present Naver Corporation. All rights reserved.

import types
import torch
from torch import autocast

from blast3r.utils.device import todevice
from blast3r.image import detokenize




@torch.inference_mode()
def compute_pts3d_multidepths(model, views):
    assert not isinstance(model, types.LambdaType), 'the model must be loaded before reaching here'

    imgs_shape = views['img'].shape
    assert imgs_shape[-3:] == (3, 16, 16), 'wrong format: images are not tokenized'
    if len(imgs_shape) == 4:
        n_imgs = 1
        batching = None
    elif len(imgs_shape)  == 5:
        n_imgs = imgs_shape[0]
        batching = ()
    else:
        raise TypeError(f'imgs have the wrong shape={imgs_shape}, should be a 4- or 5-tuple')

    print(f'>> Extracting depthmaps for {n_imgs} images ...')
    batched_views = {key:views[key][batching] for key in 'img token_pos true_shape'.split()}
    if 'depth' in views: batched_views['depth'] = views['depth'][batching]
    if 'K' in views: batched_views['K'] = views['K'][batching]
    batched_views = todevice(batched_views, model.device)

    preds = []
    with autocast(device_type=model.device.type, dtype=model.force_dtype, enabled=(model.force_dtype != torch.float32)):
        for i in range(n_imgs):
            pred = model({k: v[i:i+1].unsqueeze(1) for k, v in batched_views.items()}, step=0)
            preds.append(pred[0]) # MonoDepth returns one dict per view

    depth_mode = 'log_multidepth'
    pts3d = [pred['query_pts3d'][..., :3].squeeze(0) for pred in preds]
    conf = [pred['query_pts3d_conf'].squeeze(0) for pred in preds]
    multidepth = [pred['query_pts3d_raw'][..., 2:].squeeze(0) for pred in preds]

    # detokenize (per-img because shape can vary)
    H_Ws = views['true_shape'][batching].tolist()
    assert len(H_Ws) == len(preds) and len(H_Ws[0]) == 2, \
        f'expected {len(preds)} (H, W) pairs, got {H_Ws}'
    pts3d = [detokenize(x.float(), H_W) for x, H_W in zip(pts3d, H_Ws)]
    conf = [detokenize(x.float().exp_(), H_W, chan_dim=None) for x, H_W in zip(conf, H_Ws)] # logits -> linear space
    multidepth = [detokenize(x.float(), H_W) for x, H_W in zip(multidepth, H_Ws)]

    if depth_mode == 'log_multidepth':
        multidepth = [torch.cat((m[...,:1], torch.ones_like(m[...,:1]), m[...,1:]), -1) for m in multidepth]
    print('Found', multidepth[0].shape[-1], 'Z components')

    return conf, pts3d, multidepth, depth_mode
