#!/usr/bin/env python3
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Prepare the ETH3D multi-view benchmark for offline evaluation.

    python scripts/preprocessing/prepare_eth3d.py --eth3d_root DIR --out DIR

`--eth3d_root` is the official ETH3D high-resolution multi-view download, one
directory per scene, each holding `dslr_calibration_undistorted/` and
`images/dslr_images_undistorted/`. Writes one directory per scene, its images
symlinked and its ground-truth poses in `gt.pkl`.
"""
import argparse
from pathlib import Path

from blast3r.eval.colmap import read_images as read_colmap_images
from blast3r.eval.datasets import colmap_pose
from blast3r.eval.prepare import write_scene

# The 13 scenes with public ground truth, as used for the published results.
ETH3D_SCENES = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'meadow',
                'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains']

CALIBRATION = 'dslr_calibration_undistorted'
IMAGES = 'images'


def prepare_scene(scene_dir, out_dir):
    images = read_colmap_images(Path(scene_dir) / CALIBRATION)

    names = sorted(images)
    paths = [Path(scene_dir) / IMAGES / name for name in names]
    missing = [p for p in paths if not p.is_file()]
    assert not missing, f'{len(missing)} images are missing, e.g. {missing[0]}'

    write_scene(out_dir, paths, [colmap_pose(*images[name]) for name in names])
    return len(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eth3d_root', required=True, help='the official ETH3D download')
    parser.add_argument('--out', required=True, help='where to write the prepared scenes')
    parser.add_argument('--scenes', nargs='+', default=ETH3D_SCENES,
                        help='scenes to prepare (default: the 13 published scenes)')
    args = parser.parse_args()

    for scene in args.scenes:
        scene_dir = Path(args.eth3d_root) / scene
        assert scene_dir.is_dir(), f'no such scene directory: {scene_dir}'
        n_images = prepare_scene(scene_dir, Path(args.out) / scene)
        print(f'{scene}: {n_images} images')

    print(f'\nPrepared {len(args.scenes)} scenes in {args.out}')


if __name__ == '__main__':
    main()
