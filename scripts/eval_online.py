#!/usr/bin/env python3
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Trajectory evaluation of online (SLAM) reconstructions.

Covers TUM-RGBD and ETH3D-SLAM, which share a file format. Predictions are one
`<sequence>.pth` per sequence, as written by `run_online.py`. Ground truth is
read from each sequence's `groundtruth.txt`.
Trajectories are Sim(3)-aligned before measuring absolute trajectory error,
since monocular reconstruction determines scale only up to a constant.
"""
import argparse
from pathlib import Path

from tqdm import tqdm

from blast3r.eval.datasets import ETH3D_SLAM_SEQUENCES, TUM_RGBD_SEQUENCES, read_tum_rgbd_poses
from blast3r.eval.io import read_scene_predictions
from blast3r.eval.metrics.slam import compute_ate, interpolate_trajectory
from blast3r.eval.tables import build_table, show, to_latex, write_csv

REPORT = ['ate_rmse', 'ate_mean', 'ate_median', 'coverage', 'n_frames']

BENCHMARKS = {'tum_rgbd': TUM_RGBD_SEQUENCES, 'eth3d_slam': ETH3D_SLAM_SEQUENCES}


def evaluate(data_root, preds_dir, sequences, allow_subset=False, interpolate=False):
    rows, missing = [], []
    for seq in tqdm(sequences, desc='Evaluating sequences'):
        pred_path = Path(preds_dir) / f'{seq}.pth'
        if not pred_path.exists():
            missing.append(seq)
            continue

        gt_poses = read_tum_rgbd_poses(Path(data_root) / seq)
        predictions = read_scene_predictions(pred_path)
        pred_poses = {name: view['cam2w'] for name, view in predictions.items()}
        if interpolate:
            pred_poses = interpolate_trajectory(pred_poses, gt_poses)

        stats = compute_ate(gt_poses, pred_poses, allow_subset=allow_subset)
        rows.append({'sequence': seq, **stats})

    return rows, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_root', required=True, help='dataset root, one directory per sequence')
    parser.add_argument('--preds_dir', required=True, help='directory of <sequence>.pth predictions')
    parser.add_argument('--benchmark', choices=sorted(BENCHMARKS), default='tum_rgbd',
                        help='which published sequence list to evaluate (default: tum_rgbd)')
    parser.add_argument('--sequences', nargs='+', default=None,
                        help='sequences to evaluate (default: the whole --benchmark list)')
    parser.add_argument('--allow_subset', action='store_true',
                        help='evaluate on the frames that were reconstructed, instead of '
                             'requiring a pose for every ground-truth frame. Reported as coverage.')
    parser.add_argument('--interpolate', action='store_true',
                        help='interpolate the reconstructed trajectory onto every ground-truth '
                             'frame before measuring the error, as the published results do. '
                             'Needed to compare against them when running with --frame_step.')
    parser.add_argument('--csv', default=None, help='where to write the per-sequence CSV '
                                                    '(default: <preds_dir>/slam.csv)')
    parser.add_argument('--latex', action='store_true', help='also print a LaTeX table')
    args = parser.parse_args()

    sequences = args.sequences or BENCHMARKS[args.benchmark]
    rows, missing = evaluate(args.data_root, args.preds_dir, sequences,
                             args.allow_subset, args.interpolate)
    assert rows, 'no predictions found to evaluate'
    df = build_table(rows, index='sequence')

    if missing:
        print(f'\nWarning: {len(missing)} sequences had no prediction and are excluded '
              f'from the average: {", ".join(missing)}')

    show(df, 'Absolute trajectory error after Sim(3) alignment (m)', columns=REPORT, float_format='%.4f')
    write_csv(df, args.csv or str(Path(args.preds_dir) / 'slam.csv'))
    if args.latex:
        print('\n' + to_latex(df, columns=['ate_rmse'], float_format='%.3f'))


if __name__ == '__main__':
    main()
