# Copyright (C) 2026-present Naver Corporation. All rights reserved.
#
# Adapted for BLASt3R from GlORIE-SLAM (https://github.com/zhangganlin/GlORIE-SLAM).
# Original code licensed under the Apache License 2.0.
import os.path as osp

import numpy as np
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from scipy.interpolate import interp1d


def _timestamp(img_name):
    return float(osp.splitext(img_name)[0])


def interpolate_trajectory(pred_poses, gt_poses):
    """Densify a keyframe trajectory onto every ground-truth frame.

    Reconstructing every `--frame_step`-th frame leaves the trajectory sampled
    more coarsely than the ground truth. Positions are interpolated linearly
    over the frame timestamps; rotations are left at identity, as the absolute
    trajectory error only uses the translation.
    """
    if len(pred_poses) == len(gt_poses):
        return pred_poses

    pred_names = sorted(pred_poses)
    gt_names = sorted(gt_poses)

    positions = np.array([pred_poses[name][:3, 3] for name in pred_names])
    interpolate = interp1d(np.array([_timestamp(n) for n in pred_names]), positions.T,
                           kind='linear', fill_value='extrapolate')
    positions = interpolate(np.array([_timestamp(n) for n in gt_names])).T

    interpolated = {}
    for name, position in zip(gt_names, positions):
        pose = np.eye(4)
        pose[:3, 3] = position
        interpolated[name] = pose

    return interpolated


def align_trajectories(gt_poses, pred_poses, allow_subset=False):
    """Sim(3)-align the predicted trajectory to the ground truth.

    Frame names are expected to be timestamps, as in the TUM-RGBD format.
    """
    if not allow_subset:
        missing = [k for k in gt_poses if k not in pred_poses]
        assert not missing, f'{len(missing)} ground-truth frames have no prediction'

    traj_est, traj_ref, timestamps = [], [], []
    for img_name, gt_pose in gt_poses.items():
        if not np.isfinite(gt_pose.sum()):
            print(f'Skipping {img_name}: ground-truth pose is not finite')
            continue
        if img_name not in pred_poses:
            continue

        traj_est.append(pred_poses[img_name])
        traj_ref.append(gt_pose)
        timestamps.append(_timestamp(img_name))

    assert timestamps, 'no frames in common between ground truth and predictions'
    coverage = len(timestamps) / len(gt_poses)

    traj_est = PoseTrajectory3D(poses_se3=traj_est, timestamps=timestamps)
    traj_ref = PoseTrajectory3D(poses_se3=traj_ref, timestamps=timestamps)
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    traj_est.align(traj_ref, correct_scale=True)

    return traj_est, traj_ref, coverage


def compute_ate(gt_poses, pred_poses, allow_subset=False):
    """Absolute trajectory error after Sim(3) alignment, in metres."""
    traj_est, traj_ref, coverage = align_trajectories(gt_poses, pred_poses, allow_subset)

    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((traj_ref, traj_est))
    stats = ape.get_all_statistics()

    return {
        'ate_rmse': stats['rmse'],
        'ate_mean': stats['mean'],
        'ate_median': stats['median'],
        'ate_std': stats['std'],
        'ate_max': stats['max'],
        'coverage': coverage * 100,
        'n_frames': traj_est.num_poses,
    }
