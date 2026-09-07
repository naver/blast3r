# Copyright (C) 2026-present Naver Corporation. All rights reserved.

from tqdm import tqdm
import numpy as np
from scipy import sparse
import torch

from blast3r.utils.device import to_numpy, to_cuda, clone
from blast3r.utils.parallel import parallel_threads
from blast3r.utils.geometry import depthmap_to_pts3d, bmv, inv, bilinear_sampling


def denoise_views(views,
                  rm_noise=0.005,
                  carve_depth=0.9,
                  conf_thr=None,
                  block_size=32,
                  use_conf_for_carving=True,
                  cc_min_size=0.01,
                 ):
    if conf_thr:
        views = denoise_depth_confidence(views, conf_thr)

    if rm_noise:    # removing obvious noise
        views = denoise_floating_outliers(views, ball_radius=rm_noise, cc_min_size=cc_min_size)

    if carve_depth:
        views = denoise_depth_carving(views, carve_coef=carve_depth, use_conf=use_conf_for_carving, block_size=block_size)

    return views


def denoise_depth_confidence(views, conf_thr=2.):
    views = copy_views(views)
    for i, view in enumerate(tqdm(views)):
        invalid = view['conf'] < conf_thr
        view['depth'][invalid] = float('nan')
        view['conf'][invalid] = 0

    return views


def denoise_floating_outliers(views, ball_radius=0.005, cc_min_size=0.01):
    views = to_numpy(copy_views(views))

    def denoise_one_image(i):
        view = views[i]
        invalid = ~filter_depth_noise(view['depth'], view['K'],
                                      ball_radius=ball_radius,
                                      cc_min_size=cc_min_size,
                                      pixel_rad=2)
        view['depth'][invalid] = float('nan')
        view['conf'][invalid] = 0

    parallel_threads(denoise_one_image, range(len(views)), desc='denoising')
    return views


def filter_depth_noise(depth, K, ball_radius=0.005, pixel_rad=2, cc_min_size=0.01):
    """Remove depth noise: isolated 3d points with no neighbor within a radius `r`.

    The radius is r = epsilon * z, with epsilon typically 0.01.

    Assuming 2 rays with depth z1 and z2 separated by angle alpha,
    then the distance between the two 3D points is:
        squared_dist = z1**2 + z2**2 - 2 z1 z2 cos(alpha)

    To make the distance threshold symmetric, we use r = epsilon * sqrt(z1 * z2)
    then
        squared_dist < r**2
    <=> r**2 - squared_dist > 0
    <=> z1 z2 epsilon**2 - (z1**2 + z2**2 - 2 z1 z2 cos(alpha)) > 0
    <=> z1 z2 (epsilon**2 - 2 cos(alpha)) - (z1**2 + z2**2) > 0
    """
    # angle between 2 neighboring rays
    H, W = depth.shape
    fov_w, fov_h = 2 * np.arctan( (W, H) / (2*np.diagonal(K[:2])) )
    alpha = min(fov_w / W, fov_h / H) # approx angle between 2 ngh rays

    def dis_test(z1, z2, cos_alpha):
        z1z2 = z1*z2
        sqr_dis = np.square(z1) + np.square(z2) - 2*z1z2*cos_alpha
        sqr_thr = ball_radius**2 * z1z2
        return sqr_dis < sqr_thr

    idx = np.arange(H*W).reshape(H, W)

    sim = []
    row_col = []
    def slices(x):
        res = (slice(None, -abs(x) or None), slice(abs(x), None))
        return res if x >= 0 else res[::-1]

    for x in range(-pixel_rad, pixel_rad+1):
        sx1, sx2 = slices(x)
        for y in range(pixel_rad+1):
            if x == y == 0: continue # useless
            if y == 0 and x < 0: continue # already done in symmetric
            sy1, sy2 = slices(y)
            ngh = dis_test(depth[sy1,sx1], depth[sy2,sx2], np.cos(alpha)).ravel()
            r, c = (idx[sy1,sx1].ravel(), idx[sy2,sx2].ravel())
            sim.append(ngh[ngh])
            row_col.append((r[ngh], c[ngh]))

    # make a sparse matrix and look for connected components
    sim = np.concatenate(sim)
    row, col = np.concatenate(row_col, -1)
    graph = sparse.coo_array((sim, (row, col)), shape=(H*W, H*W))
    nc, labels = sparse.csgraph.connected_components(graph, directed=False)
    labels = labels.reshape(H,W)

    hist = np.bincount(labels.ravel())

    # scale cc_min_size with image size
    if isinstance(cc_min_size, float):
        assert 0 < cc_min_size < 0.1
        cc_min_size = int((cc_min_size * max(H,W))**2)
    assert cc_min_size > 1
    valid_mask = (hist[labels] >= cc_min_size)

    return valid_mask


@torch.inference_mode()
def denoise_depth_carving(views, carve_coef=0.9, block_size=64, use_conf=True):
    if len(views) <= 1:
        return views
    views = to_cuda(views)
    views = copy_views(views) # copy everything before modifying it

    # allocate memory
    n_pixels = views[0]['depth'].numel() # assuming all views have the same number of pixels
    block_pts3d = torch.empty((min(len(views)-1, block_size), n_pixels, 3), device='cuda')

    for i, view in enumerate(tqdm(views, desc='depth carving')):
        cur_w2cam = inv(view['cam2w'])
        cur_conf = view['conf']
        empty_space = view['depth'] * carve_coef
        empty_space[cur_conf <= 0] = 0

        for block_start in range(0, len(views)-1, block_size):
            block_end = min(block_start + block_size, len(views)-1)

            # take the points from all other frames and project them in this view
            n = 0
            for v_idx in range(block_start, block_end):
                if v_idx == i: continue
                v = views[v_idx]
                block_pts3d[n] = depthmap_to_pts3d(v['depth'], v['K'], v['cam2w']).reshape(-1,3)
                n += 1
            pts3d = block_pts3d[:n]
            w2cam = cur_w2cam[None].expand(n, 4, 4)
            pts3d[:] = bmv(w2cam, pts3d) # in-place writing
            pts3d_in_cam = pts3d

            # now, find the depth in the current view corresponding to where they land
            K = view['K'][None].expand(n, 3, 3)
            pixels = bmv(K, pts3d_in_cam, norm=1, zclip=1e-16, ncol=2)
            empty_space = empty_space[None].expand(n, *empty_space.shape)
            empty_spaces = bilinear_sampling(empty_space.unsqueeze(-1), pixels, padding_mode='zeros').squeeze(-1)

            # also find the confidence in the current view
            if use_conf:
                conf = cur_conf[None].expand(n, *cur_conf.shape)
                my_confs = bilinear_sampling(conf.unsqueeze(-1), pixels, padding_mode='zeros').squeeze(-1)
                my_confs = iter(my_confs)

            my_depths = iter(pts3d_in_cam[...,2])
            empty_spaces = iter(empty_spaces)

            for v_idx in range(block_start, block_end):
                if v_idx == i: continue
                view2 = views[v_idx]
                conf2 = view2['conf'].ravel()

                d = next(my_depths)
                empty_space = next(empty_spaces)

                # if the point from the other view is inbetween my camera and my depth,
                # this point is wrong, if I'm more confident that it is.
                invalid = (0 < d) & (d < empty_space)

                if use_conf:
                    c = next(my_confs)
                    im_more_confident = (c >= conf2)
                    invalid &= im_more_confident

                conf2[invalid] = 0

    return views


def copy_views(views):
    # just duplicate conf and depth, these are the ones that will get modified
    views = [dict(view, depth=clone(view['depth']), conf=clone(view['conf'])) for view in views]
    return views
