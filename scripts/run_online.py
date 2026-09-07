#!/usr/bin/env python3
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Online (incremental) reconstruction of an image sequence.

    python scripts/run_online.py --images DIR --output DIR --name SCENE \
        --mono_cam --frame_step 4

Defaults are the values used for the published results.
"""
from pathlib import Path

import torch

from blast3r.inference.args import base_parser
from blast3r.inference.loading import load_dataset, load_depth_model, load_match_model
from blast3r.inference.pipeline import run_online_inference
from blast3r.inference.results import prepare_results
from blast3r.inference.timing import GPUMonitor, timer


def arg_parser():
    parser = base_parser('BLASt3R online (V-SLAM) reconstruction')
    parser.add_argument('--frame_step', type=int, default=1, help='only process frames[::frame_step]')
    parser.add_argument('-rdc', '--reduce_tracks', action='store_true', help='remove redundant tracks')
    parser.add_argument('-lmt', '--max_num_tracks', type=int, default=100_000,
                        help='drop tracks above this limit (0 disables)')
    parser.add_argument('--kpt_conf_thr', type=float, default=None,
                        help='remove keypoints below this confidence')
    parser.add_argument('--gba_step', type=int, default=8, help='run a global BA every `step` frames')
    parser.add_argument('--max_iters_local', type=int, default=10, help='BA iterations between frames')
    return parser


def run(images, ckpt_depth, ckpt_match, size, force_ar, output, export_mode, name=None,
        skip_existing=False, skip_failures=False, frame_step=1, seed=0,
        device='cuda', dtype=torch.float32, **options):
    dataset = load_dataset(images, size=size, force_ar=force_ar, online=True, name=name)
    depther = load_depth_model(ckpt_depth, device=device, dtype=dtype)
    matcher = load_match_model(ckpt_match, device=device, dtype=dtype)

    for i, (ds_tag, scene, sequence) in enumerate(dataset):
        print(f'Processing scene #{i+1}: {scene}')

        filename = f'{scene}.pth' if scene is not None else f'result_{i:03d}.pth'
        out_file = output if output.endswith('.pth') else str(Path(output) / filename)
        if Path(out_file).exists() and skip_existing:
            continue
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)

        sequence = list(sequence)[::frame_step]
        # reset per sequence so the coreset draw does not depend on the position in the batch
        torch.manual_seed(seed)
        try:
            torch.cuda.empty_cache()
            timer.reset()
            gpu_monitor = GPUMonitor(device=device, interval=10.0)
            with timer.time('total'), gpu_monitor.monitor():
                out = run_online_inference(f'{ds_tag}/{scene}', sequence, depther, matcher,
                                           **options, device=device)
            timer_out = timer.summary()
            gpu_monitor_out = gpu_monitor.summary()
        except Exception as e:
            if not skip_failures:
                raise
            print(f'Error processing scene #{i+1}: {e}')
            continue

        img_names = [Path(s['instance']).name for s in sequence]
        extra = {'timing': timer_out, 'gpu_usage': gpu_monitor_out}
        out = prepare_results(out, img_names, extra_data=extra, export_mode=export_mode)
        torch.save(out, out_file)


if __name__ == '__main__':
    run(**vars(arg_parser().parse_args()))
