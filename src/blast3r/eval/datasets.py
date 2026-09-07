# Copyright (C) 2026-present Naver Corporation. All rights reserved.
#
# The TUM-RGBD reader is adapted for BLASt3R from GlORIE-SLAM
# (https://github.com/zhangganlin/GlORIE-SLAM), licensed under Apache 2.0.
import os.path as osp
import pickle
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# The 9 fr1 sequences used for the online experiments.
TUM_RGBD_SEQUENCES = [
    'rgbd_dataset_freiburg1_360',
    'rgbd_dataset_freiburg1_desk',
    'rgbd_dataset_freiburg1_desk2',
    'rgbd_dataset_freiburg1_floor',
    'rgbd_dataset_freiburg1_plant',
    'rgbd_dataset_freiburg1_room',
    'rgbd_dataset_freiburg1_rpy',
    'rgbd_dataset_freiburg1_teddy',
    'rgbd_dataset_freiburg1_xyz',
]

# The 8 ETH3D-SLAM training sequences used for the online experiments. ETH3D-SLAM
# uses the TUM-RGBD file format, so the same reader applies.
ETH3D_SLAM_SEQUENCES = [
    'cables_1',
    'camera_shake_1',
    'einstein_1',
    'plant_1',
    'plant_2',
    'sofa_1',
    'table_3',
    'table_7',
]


def _parse_list(filepath, skiprows=0):
    with warnings.catch_warnings():
        # the TUM list files have comment and blank lines that loadtxt warns about
        warnings.simplefilter('ignore', UserWarning)
        return np.loadtxt(filepath, delimiter=' ', dtype=np.str_, skiprows=skiprows)


def _pose_from_quaternion(pvec):
    """TUM ground truth is `timestamp tx ty tz qx qy qz qw`."""
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
    pose[:3, 3] = pvec[:3]
    return pose


def _associate(tstamp_image, tstamp_pose, max_dt=0.08):
    """Pair each image with the nearest-in-time ground-truth pose."""
    associations = []
    for i, t in enumerate(tstamp_image):
        k = np.argmin(np.abs(tstamp_pose - t))
        if np.abs(tstamp_pose[k] - t) < max_dt:
            associations.append((i, k))
    return associations


def read_tum_rgbd_poses(seq_dir):
    """Ground-truth camera-to-world poses for one TUM-RGBD sequence.

    Returns {image_name: 4x4 pose}, with the first associated frame at the
    origin. Images with no pose within `max_dt` are omitted.
    """
    seq_dir = Path(seq_dir)
    pose_file = seq_dir / 'groundtruth.txt'
    if not pose_file.is_file():
        pose_file = seq_dir / 'pose.txt'
    assert pose_file.is_file(), f'no groundtruth.txt or pose.txt in {seq_dir}'

    image_data = _parse_list(seq_dir / 'rgb.txt')
    pose_data = _parse_list(pose_file, skiprows=1)
    pose_vecs = pose_data[:, 1:].astype(np.float64)

    tstamp_image = image_data[:, 0].astype(np.float64)
    tstamp_pose = pose_data[:, 0].astype(np.float64)

    poses = {}
    inv_pose = None
    for i, k in _associate(tstamp_image, tstamp_pose):
        c2w = _pose_from_quaternion(pose_vecs[k])
        if inv_pose is None:
            inv_pose = np.linalg.inv(c2w)
            c2w = np.eye(4)
        else:
            c2w = inv_pose @ c2w
        poses[osp.basename(image_data[i, 1])] = c2w

    return poses


def colmap_pose(qvec, tvec):
    """Camera-to-world pose from COLMAP's world-to-camera quaternion and translation."""
    qw, qx, qy, qz = qvec
    w2c = np.eye(4)
    w2c[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    w2c[:3, 3] = tvec
    return np.linalg.inv(w2c)


def read_scene_gt_poses(scene_dir):
    """Ground-truth poses for one offline scene.

    A scene directory holds its images plus a `gt.pkl`, as written by the
    preprocessing scripts.
    """
    gt_pkl = Path(scene_dir) / 'gt.pkl'
    if not gt_pkl.is_file():
        raise FileNotFoundError(f'{scene_dir} has no gt.pkl')

    with open(gt_pkl, 'rb') as f:
        data = pickle.load(f)
    return dict(zip(data['filenames'], data['poses']))


def has_scene_gt(scene_dir):
    return (Path(scene_dir) / 'gt.pkl').is_file()
