# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Drawing reconstructed views in a viser scene.

Shared by the file viewer and the demo, which differ only in where the views come
from: one loads a finished scene, the other appends frames as the online pipeline
produces them and moves the earlier ones every time bundle adjustment revises the
poses.
"""
import numpy as np
import viser.transforms as tf

from blast3r.utils.device import to_numpy
from blast3r.utils.geometry import depthmap_to_pts3d

FRUSTUM_WIDTH = 256
CONF_INIT = 10.0  # confidence threshold the sliders start at
CONF_MAX = 50.0   # and the least they reach up to


def build_frame(view):
    """Everything the scene needs from one view, in a form the filters can slice."""
    depth = to_numpy(view['depth']).astype(np.float32)
    cam2w = to_numpy(view['cam2w'])
    K = to_numpy(view['K'])

    pts = depthmap_to_pts3d(depth, K, cam2w).reshape(-1, 3)
    img = to_numpy(view['img']) if 'img' in view else None
    col = img.reshape(-1, 3) if img is not None else np.full(pts.shape, 128, dtype=np.uint8)
    conf = to_numpy(view['conf']).reshape(-1).astype(np.float32) if 'conf' in view else None

    keep = np.isfinite(pts).all(-1)
    if conf is not None:
        keep &= conf > 0  # carving zeroes the confidence but leaves the depth in place

    return dict(pts=pts, col=col, conf=conf, img=img, K=K, cam2w=cam2w, shape=depth.shape,
                # int32 indices: one frame's draw order is kept for the whole run
                order=np.random.permutation(np.flatnonzero(keep)).astype(np.int32))


def conf_upper_bound(views):
    """A slider maximum that fits the scene: confidence is exp(logit), so its
    scale depends on the model and the range spans two orders of magnitude."""
    conf = np.concatenate([np.asarray(view['conf']).ravel() for view in views])
    return max(round(float(np.percentile(conf[::max(1, conf.size // 1_000_000)], 99))), 1)


def select_points(frames, indices, conf_thr, budget):
    """Indices to draw per frame, sharing `budget` points in proportion to what survives."""
    kept = []
    for i in indices:
        frame = frames[i]
        order = frame['order']
        if frame['conf'] is None:
            kept.append(order)
        else:
            kept.append(order[frame['conf'][order] >= conf_thr])

    total = sum(len(k) for k in kept)
    if total > budget:
        kept = [k[:int(len(k) * budget / total)] for k in kept]
    return kept


class SceneRenderer:
    """Owns the scene handles: a point cloud and a camera frustum per frame, plus
    the polyline through the camera centers.

    Sizes are in scene units; a caller showing them in other units converts.
    """

    def __init__(self, server):
        self.server = server
        self.frames = []
        self.pc_handles = []
        self.cam_handles = []
        self.trajectory = None

    def __len__(self):
        return len(self.frames)

    def clear(self):
        for handle in self.pc_handles + self.cam_handles:
            handle.remove()
        self.pc_handles, self.cam_handles = [], []
        if self.trajectory is not None:
            self.trajectory.remove()
            self.trajectory = None
        self.frames = []

    def set_frames(self, frames, point_size, cam_scale):
        self.clear()
        for i, frame in enumerate(frames):
            self.set_frame(i, frame, point_size, cam_scale)

    def set_frame(self, i, frame, point_size, cam_scale):
        """Draw frame `i`, appending it or replacing what is already drawn there."""
        assert i <= len(self.frames), f'frame {i} would leave a gap at {len(self.frames)}'
        if i == len(self.frames):
            self.frames.append(frame)
            self._add_handles(i, frame, point_size, cam_scale)
        else:
            self.frames[i] = frame
            self._move_camera(i, frame)

    def _add_handles(self, i, frame, point_size, cam_scale):
        H, W = frame['shape']
        img = frame['img']
        if img is not None:
            img = img[::max(1, W // FRUSTUM_WIDTH), ::max(1, W // FRUSTUM_WIDTH)]
        cam2w = frame['cam2w']

        self.pc_handles.append(self.server.scene.add_point_cloud(
            f'frames/{i}/points', np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8),
            point_size=point_size, point_shape='circle'))
        self.cam_handles.append(self.server.scene.add_camera_frustum(
            f'frames/{i}/camera',
            fov=2 * np.arctan2(H / 2, frame['K'][1, 1]),
            aspect=W / H,
            image=img,
            scale=cam_scale,
            thickness=2.0,
            thickness_units='screen',
            wxyz=tf.SO3.from_matrix(cam2w[:3, :3]).wxyz,
            position=cam2w[:3, 3]))

    def _move_camera(self, i, frame):
        H, W = frame['shape']
        cam2w = frame['cam2w']
        handle = self.cam_handles[i]
        handle.fov = 2 * np.arctan2(H / 2, frame['K'][1, 1])  # the focal moves while BA optimizes it
        handle.wxyz = tf.SO3.from_matrix(cam2w[:3, :3]).wxyz
        handle.position = cam2w[:3, 3]

    def refresh_points(self, first, last, conf_thr, budget):
        if not self.frames:
            return
        visible = range(first, last + 1)
        selected = select_points(self.frames, visible, conf_thr, budget)

        with self.server.atomic():
            for i, handle in enumerate(self.pc_handles):
                handle.visible = first <= i <= last
            for i, indices in zip(visible, selected):
                frame = self.frames[i]
                self.pc_handles[i].points = frame['pts'][indices]
                self.pc_handles[i].colors = frame['col'][indices]

    def refresh_cameras(self, first, last, show, scale):
        with self.server.atomic():
            for i, handle in enumerate(self.cam_handles):
                handle.visible = show and first <= i <= last
                handle.scale = scale

    def refresh_trajectory(self, first, last, show):
        if not show or last - first < 1:
            if self.trajectory is not None:
                self.trajectory.visible = False
            return

        centers = np.stack([frame['cam2w'][:3, 3] for frame in self.frames[first:last + 1]])
        segments = np.stack([centers[:-1], centers[1:]], axis=1)
        if self.trajectory is None:
            self.trajectory = self.server.scene.add_line_segments(
                'trajectory', segments, colors=(255, 170, 0), thickness=2.0, thickness_units='screen')
        else:
            self.trajectory.points = segments
        self.trajectory.visible = True

    def set_point_size(self, point_size):
        with self.server.atomic():
            for handle in self.pc_handles:
                handle.point_size = point_size
