#!/usr/bin/env python3
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Prepare the CO3D v2 relative-pose benchmark for offline evaluation.

    python scripts/preprocessing/prepare_co3d.py --co3d_root DIR --out DIR

`--co3d_root` is the official CO3D v2 download, one directory per category, each
holding `frame_annotations.jgz`, `sequence_annotations.jgz` and `set_lists/`.
Writes one directory per sequence under `<out>/<category>/`, with the evaluated
frames symlinked and the ground-truth poses in `gt.pkl`.

Evaluated are the `fewview_dev` test sequences of the 41 seen categories, keeping
those whose `viewpoint_quality_score` exceeds `--min_quality`. Ten frames are
drawn per sequence, by an RNG seeded from the sequence name, so the selection is
reproducible without needing to be distributed.

The conversion from CO3D's PyTorch3D cameras is adapted for BLASt3R from DUSt3R
(https://github.com/naver/dust3r), datasets_preprocess/preprocess_co3d.py,
licensed under CC BY-NC-SA 4.0.
"""
import argparse
import gzip
import json
import os.path as osp
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from blast3r.eval.prepare import sample_frames, write_scene

# The 41 seen categories. The 10 categories held out by the relative-pose
# protocol -- ball, book, couch, frisbee, hotdog, kite, remote, sandwich,
# skateboard, suitcase -- are not evaluated.
CO3D_CATEGORIES = [
    'apple', 'backpack', 'banana', 'baseballbat', 'baseballglove', 'bench', 'bicycle',
    'bottle', 'bowl', 'broccoli', 'cake', 'car', 'carrot', 'cellphone', 'chair', 'cup',
    'donut', 'hairdryer', 'handbag', 'hydrant', 'keyboard', 'laptop', 'microwave',
    'motorcycle', 'mouse', 'orange', 'parkingmeter', 'pizza', 'plant', 'stopsign',
    'teddybear', 'toaster', 'toilet', 'toybus', 'toyplane', 'toytrain', 'toytruck',
    'tv', 'umbrella', 'vase', 'wineglass',
]

# CO3D stores cameras in the PyTorch3D convention; the evaluation expects OpenCV.
PYTORCH3D_TO_OPENCV = np.diag([-1.0, -1.0, 1.0])


def camera_to_world(viewpoint):
    R = np.array(viewpoint['R'])
    T = np.array(viewpoint['T'])

    w2c = np.eye(4)
    w2c[:3, :3] = PYTORCH3D_TO_OPENCV @ R.T
    w2c[:3, 3] = PYTORCH3D_TO_OPENCV @ T
    return np.linalg.inv(w2c)


def read_category(co3d_root, category):
    """Frame paths per sequence, the camera of every frame, and per-sequence quality."""
    category_dir = Path(co3d_root) / category

    set_lists = category_dir / 'set_lists' / 'set_lists_fewview_dev.json'
    assert set_lists.is_file(), f'no such file: {set_lists}'
    with open(set_lists) as f:
        frames = defaultdict(list)
        for sequence, _, relative_path in json.load(f)['test']:
            frames[sequence].append(str(Path(co3d_root) / relative_path))

    with gzip.open(category_dir / 'frame_annotations.jgz', 'rt') as f:
        viewpoints = {(a['sequence_name'], osp.basename(a['image']['path'])): a['viewpoint']
                      for a in json.load(f)}

    with gzip.open(category_dir / 'sequence_annotations.jgz', 'rt') as f:
        quality = {a['sequence_name']: a.get('viewpoint_quality_score') for a in json.load(f)}

    return frames, viewpoints, quality


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--co3d_root', required=True, help='the official CO3D v2 download')
    parser.add_argument('--out', required=True, help='where to write the prepared scenes')
    parser.add_argument('--categories', nargs='+', default=CO3D_CATEGORIES,
                        help='categories to prepare (default: the 41 seen categories)')
    parser.add_argument('--min_quality', type=float, default=0.5,
                        help='keep sequences whose viewpoint_quality_score exceeds this')
    parser.add_argument('--n_frames', type=int, default=10,
                        help='frames to draw per sequence (default: the published 10)')
    args = parser.parse_args()

    n_scenes, n_dropped = 0, 0
    for category in tqdm(sorted(args.categories), desc='Preparing categories'):
        frames, viewpoints, quality = read_category(args.co3d_root, category)

        for sequence in sorted(frames):
            score = quality.get(sequence)
            if score is None or score <= args.min_quality:
                n_dropped += 1
                continue

            images = sample_frames(frames[sequence], sequence, n_frames=args.n_frames)
            poses = [camera_to_world(viewpoints[sequence, osp.basename(p)]) for p in images]

            write_scene(Path(args.out) / category / sequence, images, poses)
            n_scenes += 1

    print(f'\nPrepared {n_scenes} sequences over {len(args.categories)} categories in {args.out}')
    print(f'{n_dropped} sequences were below the quality threshold and were skipped')


if __name__ == '__main__':
    main()
