# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Reconstruction of a single scene, offline or online.

The drivers in `scripts/` wrap these with argument parsing, batching and saving.
"""
import torch
from tqdm import tqdm

from blast3r.models.blocks import toggle_memory_efficient_attention
from blast3r.utils.device import to_cuda, to_numpy, todevice
from blast3r.utils.geometry import depthmap_to_pts3d
from .denoise import denoise_views
from .engine import bundle_adjustment, used_depth_channels
from .loading import get_ntokens
from .multidepth import compute_pts3d_multidepths
from .results import get_at, init_view
from .rigs import gather_pts3d, init_last_pose, init_poses_minspan_tree, update_views
from .timing import timer
from .tracks import OnlineMatcher, StoredFlows, compute_tracks, find_keypoints, gather_data_from_dense_tracks, rm_duplicated_tracks, rm_redundant_tracks


@torch.inference_mode()
def run_inference(views, depther, matcher,
                  mono_cam=False,
                  kpt_spacing=8,
                  remove_far_kpts=True,
                  retrieval_mode='coreset_fps_30',
                  num_retrieved_images=40,
                  device='cuda',
                  optim_memory=False,
                  denoise_depth=False,
                  on_init=None,
                  **BA_args):
    assert isinstance(views, dict)
    assert 'img' in views and 'true_shape' in views
    n_imgs = len(views['img'])
    n_tokens, patch_size = get_ntokens(views)
    views = todevice(views, device)

    matcher = OnlineMatcher(matcher, n_imgs, n_tokens, patch_size, device)

    if optim_memory:
        toggle_memory_efficient_attention(True)

    with timer.time('depth_pred'):
        conf, ptmap, multidepth, depth_mode = compute_pts3d_multidepths(depther, views)

    with timer.time('flow_pred'):
        tracks = compute_tracks(matcher, views,
                                kpt_spacing=kpt_spacing,
                                remove_far_kpts=remove_far_kpts,
                                num_retrieved_images=num_retrieved_images,
                                retrieval_mode=retrieval_mode,
                                optim_memory=optim_memory)
        print(f'>> Gathered {len(tracks)} tracks from {n_imgs} images')

    views = [init_view(get_at(views, i), conf[i], ptmap[i], multidepth[i], depth_mode,
                       BA_args['optim_K'], BA_args['optim_P'])
             for i in range(n_imgs)]

    with timer.time('ba_init'):
        print('>> Gathering tracks data ...')
        track_data = gather_data_from_dense_tracks(views, tracks)

        print('>> Initializing poses ...')
        rigs = init_poses_minspan_tree(views, track_data, depth_mode, mono_cam=mono_cam)
        pts3d = gather_pts3d(track_data, rigs)

        if on_init is not None:
            on_init(views, rigs)

    with timer.time('ba'):
        ptmap = []
        for i, view in enumerate(views):
            if 'pts3d_from_depth&K' in view:
                ptmap.append(view['pts3d_from_depth&K'])
            else:
                ptmap.append(depthmap_to_pts3d(view['pts3d'][:, :, 2], rigs.get_K(i)))

        new_rigs, new_pts3d = bundle_adjustment(rigs, track_data, pts3d,
                                                dense_depth=multidepth,
                                                dense_conf=conf,
                                                dense_pts3d_ref=ptmap,
                                                device='cuda',
                                                **BA_args)

    views = update_views(to_numpy(views), new_rigs.to('numpy'))
    if denoise_depth:
        views = denoise_views(views)
    return views


@torch.inference_mode()
def run_online_inference(scene, sequence, depther, matcher,
                         mono_cam=True,
                         kpt_spacing=8,
                         remove_far_kpts=True,
                         reduce_tracks=False,
                         max_num_tracks=100_000,
                         kpt_conf_thr=None,
                         retrieval_mode='coreset_fps_30',
                         num_retrieved_images=40,
                         max_iters_local=10,
                         max_iters=100,
                         gba_step=8,
                         device='cuda',
                         optim_memory=False,
                         pin_memory=False,
                         denoise_depth=False,
                         on_frame=None,
                         **BA_args):
    """`pin_memory` keeps the matcher's per-frame tokens in pinned host RAM, see
    `OnlineMatcher`; the GPU footprint then no longer grows with the sequence length."""
    assert not optim_memory, ('--optim_memory applies to offline reconstruction; online mode '
                              'already processes one frame at a time')
    print(f'>> Starting online inference for {scene}: {mono_cam=} {BA_args}')

    n_total_imgs = len(sequence)
    local_BA_args = dict(max_iters=max_iters_local, max_n_phases=1, **BA_args)

    res_views = []
    track_data = None
    rigs = None
    max_H = max_W = 0
    pts3d = torch.zeros((0, 3), device='cuda')  # sparse 3d tracks
    n_depth_channels = None  # how many multi-depth channels bundle adjustment reads

    for i, view in enumerate(tqdm(sequence)):
        with timer.time('frame'):
            print(f'>> Processing frame {i}/{n_total_imgs}')
            assert view['img'].ndim == 4, 'img.shape must be (hw, 3, 16, 16)'
            H, W = view['true_shape']
            max_H = max(max_H, H)
            max_W = max(max_W, W)

            if i == 0:
                n_tokens, patch_size = get_ntokens(view)
                abs_flow = StoredFlows(0, device)
                online_matcher = OnlineMatcher(matcher, n_total_imgs, n_tokens, patch_size, device,
                                               pin_memory=pin_memory)

            with timer.time('depth_pred'):
                conf, ptmap, multidepth, depth_mode = to_cuda(
                    compute_pts3d_multidepths(depther, view))
                # The dense maps of every frame stay on the GPU for the whole run,
                # so keep only the channels bundle adjustment will read.
                if n_depth_channels is None:
                    n_depth_channels = used_depth_channels(
                        BA_args.get('n_zcfs'), multidepth[0].shape[-1], depth_mode)
                multidepth = [m[..., :n_depth_channels].contiguous() for m in multidepth]

            with timer.time('flow_pred'):
                qidx, db_nums, kpt_conf, preds = online_matcher(view, retrieval_mode, num_retrieved_images)

                if kpt_conf_thr and kpt_conf is not None:
                    preds = (preds[0], kpt_conf * preds[1] * (preds[1] > kpt_conf_thr))

                abs_flow.add_one_frame(H, W)
                if online_matcher.flow_mode == 'flow_from_src':
                    abs_flow.forget()  # flows in previous frames are no longer needed
                    abs_flow[qidx, db_nums] = preds
                elif online_matcher.flow_mode == 'flow_to_tgt':
                    abs_flow[db_nums, qidx] = preds
                else:
                    raise NameError(f'bad {online_matcher.flow_mode=}')

                kpts, = find_keypoints([view], abs_flow, kpt_spacing, start_frame=qidx)
                tracks, scale = to_cuda(abs_flow.get_tracks(qidx, kpts, min_len=2))

            with timer.time('ba_init'):
                # the loader's frame is not kept past here: `res_view` holds what the run needs
                res_view = init_view(view, conf[0], ptmap[0], multidepth[0], depth_mode,
                                    BA_args['optim_K'], BA_args['optim_P'])
                res_views.append(res_view)

                new_track_data = gather_data_from_dense_tracks(res_views, tracks)
                rigs = init_last_pose(res_views, new_track_data, depth_mode, rigs=rigs,
                                      mono_cam=mono_cam, device=device)
                # Attaching a later frame recomputes the pointmap it compares against
                # from the depth, so the dense pointmaps are dead from here on.
                for res_view in res_views:
                    res_view.pop('pts3d', None)
                    res_view.pop('pts3d_from_depth&K', None)

                # BA is only implemented in CUDA
                new_track_data = new_track_data.to('cuda')
                rigs = rigs.to('cuda')

                new_pts3d = gather_pts3d(new_track_data, rigs, end_frame=qidx)

                if reduce_tracks and track_data and track_data.n_tracks:
                    track_data, old_to_keep = rm_duplicated_tracks(track_data, tracks)
                    pts3d = pts3d[old_to_keep]

                track_data = track_data | new_track_data
                pts3d = torch.cat((pts3d, new_pts3d))

                while max_num_tracks and track_data and track_data.n_tracks > max_num_tracks:
                    track_data, old_to_keep = rm_redundant_tracks(track_data, max_H, max_W,
                                                                 num_keep=max_num_tracks, subsample=128)
                    if not len(old_to_keep):
                        break  # cannot remove tracks anymore
                    pts3d = pts3d[old_to_keep]

                assert track_data.n_tracks == len(pts3d)
                print(f'>> Gathered {tracks.shape[0]} more tracks from {tracks.shape[1]} images '
                      f'({track_data.n_tracks} tracks in total)')

            if len(tracks) == 0:
                continue

            with timer.time('ba_step'):
                global_ba = qidx < 8 or (gba_step and qidx % gba_step == 0)
                if global_ba:
                    with timer.time('ba_step/global'):
                        rigs[:], pts3d[:] = bundle_adjustment(rigs, track_data, pts3d, **local_BA_args)
                else:
                    # optimize only the last camera, with a frozen graph
                    with timer.time('ba_step/frozen'):
                        query_rigs = rigs[qidx]
                        query_rigs[:], _ = bundle_adjustment(
                            query_rigs, new_track_data[qidx], new_pts3d, optim_X=False,
                            **dict(local_BA_args, optim_K=not mono_cam))

            if on_frame is not None:
                # a global BA moves every camera; a frozen one only moves this frame
                on_frame(i, res_views, rigs, global_ba)

    with timer.time('ba_final'):
        rigs, pts3d = bundle_adjustment(rigs, track_data, pts3d, max_iters=max_iters, **BA_args)

    views = update_views(res_views, rigs)
    if denoise_depth:
        views = denoise_views(views)
    return views
