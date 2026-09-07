#!/usr/bin/env python3
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Relative-pose evaluation of offline reconstructions.

Ground truth is one directory per scene, holding a `gt.pkl` with `poses`
(Nx4x4 camera-to-world) and `filenames`, as written by the preprocessing
scripts. Predictions are one `<scene>.pth` per scene, as written by
`run_offline.py`.
"""
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from blast3r.eval.datasets import has_scene_gt, read_scene_gt_poses
from blast3r.eval.io import read_scene_predictions
from blast3r.eval.metrics.relpose import compute_pose_errors
from blast3r.eval.tables import build_table, show, to_latex, write_csv

REPORT = ['Auc_30', 'Racc_5', 'Tacc_5', 'n_pairs']


def eval_scene(gt_poses, predictions):
    missing = [name for name in gt_poses if name not in predictions]
    assert not missing, f'{len(missing)} ground-truth images have no prediction, e.g. {missing[:3]}'

    gt = torch.stack([torch.as_tensor(gt_poses[n]).float() for n in gt_poses])
    pred = torch.stack([torch.as_tensor(predictions[n]['cam2w']).float() for n in gt_poses])
    return compute_pose_errors(gt, pred)


def evaluate(gt_dir, preds_dir):
    scenes = sorted(p for p in Path(gt_dir).iterdir() if has_scene_gt(p))
    assert scenes, f'no scenes with ground truth found under {gt_dir}'

    rows, missing = [], []
    for scene in tqdm(scenes, desc='Evaluating scenes'):
        pred_path = Path(preds_dir) / f'{scene.name}.pth'
        if not pred_path.exists():
            missing.append(scene.name)
            # a scene with no reconstruction scores zero, as in the paper
            rows.append({'scene': scene.name, 'status': 'missing',
                         **{k: 0.0 for k in REPORT if k != 'n_pairs'}})
            continue

        errs = eval_scene(read_scene_gt_poses(scene), read_scene_predictions(pred_path))
        rows.append({'scene': scene.name, 'status': 'ok', **errs})

    return rows, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gt_dir', required=True, help='ground-truth root, one directory per scene')
    parser.add_argument('--preds_dir', required=True, help='directory of <scene>.pth predictions')
    parser.add_argument('--csv', default=None, help='where to write the per-scene CSV '
                                                    '(default: <preds_dir>/relpose.csv)')
    parser.add_argument('--latex', action='store_true', help='also print a LaTeX table')
    args = parser.parse_args()

    rows, missing = evaluate(args.gt_dir, args.preds_dir)
    df = build_table(rows)

    if missing:
        print(f'\nWarning: {len(missing)} of {len(rows)} scenes had no prediction and '
              f'were scored as zero: {", ".join(missing)}')

    show(df, 'Relative pose accuracy (%)', columns=REPORT)
    write_csv(df, args.csv or str(Path(args.preds_dir) / 'relpose.csv'))
    if args.latex:
        print('\n' + to_latex(df, columns=['Auc_30']))


if __name__ == '__main__':
    main()
