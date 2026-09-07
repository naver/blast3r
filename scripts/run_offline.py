#!/usr/bin/env python3
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Offline reconstruction of an image collection.

    python scripts/run_offline.py --images DIR --output DIR --mono_cam

Defaults are the values used for the published results.
"""
from pathlib import Path

import torch

from blast3r.inference.args import base_parser
from blast3r.inference.loading import load_dataset, load_depth_model, load_match_model
from blast3r.inference.pipeline import run_inference
from blast3r.inference.results import prepare_results
from blast3r.inference.timing import timer


def arg_parser():
    return base_parser('BLASt3R offline reconstruction')


def run(images, ckpt_depth, ckpt_match, size, force_ar, output, export_mode, name=None,
        skip_existing=False, skip_failures=False, seed=0, device='cuda', dtype=torch.float32,
        **options):
    dataset = load_dataset(images, size=size, force_ar=force_ar, name=name)
    depther = load_depth_model(ckpt_depth, device=device, dtype=dtype)
    matcher = load_match_model(ckpt_match, device=device, dtype=dtype)

    for i, views in enumerate(dataset):
        scene = views['label'][0] if 'label' in views else None

        filename = f'{scene}.pth' if scene is not None else f'result_{i:03d}.pth'
        out_file = output if output.endswith('.pth') else str(Path(output) / filename)
        if Path(out_file).exists() and skip_existing:
            continue
        # fail before reconstructing rather than after
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)

        print(f'Processing scene #{i+1}')
        # reset per scene so the coreset draw does not depend on the position in the batch
        torch.manual_seed(seed)
        try:
            # release the previous scene's buffers before sizing this one
            torch.cuda.empty_cache()
            timer.reset()
            with timer.time('total'):
                out = run_inference(views, depther, matcher, **options, device=device)
            timer_out = timer.summary()
        except Exception as e:
            if not skip_failures:
                raise
            print(f'Error processing scene #{i+1}: {e}')
            continue

        img_names = [Path(inst).name for inst in views['instance']]
        out = prepare_results(out, img_names, extra_data={'timing': timer_out}, export_mode=export_mode)
        torch.save(out, out_file)


if __name__ == '__main__':
    run(**vars(arg_parser().parse_args()))
