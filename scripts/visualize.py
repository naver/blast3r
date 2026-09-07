#!/usr/bin/env python3
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Interactive viewer for saved BLASt3R reconstructions.

    python scripts/visualize.py --scenes /path/to/output
    python scripts/visualize.py --scene /path/to/output/scene.pth

Reads what `--export_mode full` writes. A pose-only scene holds no depthmaps,
so there is nothing to draw.
"""
import argparse
import threading
from pathlib import Path

import torch
import viser

from blast3r.inference.denoise import denoise_views
from blast3r.inference.results import load_results
from blast3r.utils.device import to_numpy
from blast3r.viz import CONF_INIT, SceneRenderer, build_frame, conf_upper_bound


class Viewer:
    def __init__(self, server, scene=None, scenes=None):
        self.server = server
        self.server.scene.set_up_direction('-y')
        self.renderer = SceneRenderer(server)

        self.raw = self.cleaned = None
        self.cleaned_thr = None
        self.carve_note = None
        self.range_slider = None
        self.conf_thr = None
        self.loading = threading.RLock()

        self.scene_paths = {p.name: p for p in scenes} if scenes else None
        self._build_gui()

        if scene is not None:
            self.load(scene)
        elif self.scene_paths:
            self.load(next(iter(self.scene_paths.values())))

    def _build_gui(self):
        gui = self.server.gui

        if self.scene_paths is not None:
            picker = gui.add_dropdown('Scene', list(self.scene_paths), order=0)
            picker.on_update(lambda _: self.load(self.scene_paths[picker.value]))
        else:
            self.path_box = gui.add_text('Scene file', initial_value='', order=0)
            gui.add_button('Load', order=1).on_click(
                lambda _: self.load(Path(self.path_box.value)))

        self.points_folder = gui.add_folder('Points', order=2)
        with self.points_folder:
            self.point_size = gui.add_slider('Point size (cm)', 0.01, 2, 0.01, 0.25, order=0)
            self.budget = gui.add_slider('Point budget (M)', 0.1, 20, 0.1, 2.0, order=2)
        self._rebuild_conf_slider(CONF_INIT, disabled=True)

        self.frames_folder = gui.add_folder('Frames', order=3)
        with self.frames_folder:
            self.show_cams = gui.add_checkbox('Show cameras', True, order=1)
            self.cam_scale = gui.add_slider('Camera scale', 0.005, 0.1, 0.005, 0.05, order=2)
            self.show_traj = gui.add_checkbox('Show trajectory', False, order=3)

        with gui.add_folder('Cleaning', order=4):
            self.status = gui.add_markdown('No scene loaded.', order=0)
            self.clean_button = gui.add_button('Clean point cloud', disabled=True, order=1)
            self.source = gui.add_dropdown('Source', ['Raw', 'Cleaned'], disabled=True, order=2)

        self.point_size.on_update(self._on_point_size)
        self.budget.on_update(lambda _: self.refresh_points())
        self.show_cams.on_update(lambda _: self.refresh_cameras())
        self.cam_scale.on_update(lambda _: self.refresh_cameras())
        self.show_traj.on_update(lambda _: self.refresh_trajectory())
        self.clean_button.on_click(self._on_clean)
        self.source.on_update(lambda _: self.set_source(self.source.value))

    def load(self, path):
        path = Path(path)
        with self.loading:
            self.clear()
            if not path.is_file():
                self.status.content = f'`{path}` is not a file.'
                return

            names, views = load_results(path)
            drawable = [(name, view) for name, view in zip(names, views) if 'depth' in view]
            if len(drawable) < len(views):
                print(f'{path.name}: skipping {len(views) - len(drawable)} view(s) with no depthmap')
            if not drawable:
                self.status.content = (f'**{path.name}** holds no depthmaps. '
                                       'Re-run the reconstruction with `--export_mode full`.')
                return

            self.source.value = 'Raw'

            self.raw = [view for _, view in drawable]
            self.carve_note = self._carving_obstacle()

            has_conf = 'conf' in self.raw[0]
            self._rebuild_conf_slider(conf_upper_bound(self.raw) if has_conf else CONF_INIT,
                                      disabled=not has_conf)
            self.clean_button.disabled = not has_conf

            self.build_scene()

    def _carving_obstacle(self):
        if not torch.cuda.is_available():
            return 'No CUDA device: depth carving disabled, confidence and outlier filtering only.'
        if len({view['depth'].shape for view in self.raw}) > 1:
            return 'Views have different resolutions: depth carving disabled.'
        return None

    def clear(self):
        self.renderer.clear()
        if self.range_slider is not None:
            self.range_slider.remove()
            self.range_slider = None
        self.raw = self.cleaned = None
        self.cleaned_thr = None
        self._rebuild_conf_slider(CONF_INIT, disabled=True)
        self.clean_button.disabled = True
        self.source.disabled = True

    def build_scene(self):
        views = self.cleaned if self.source.value == 'Cleaned' else self.raw
        frames = [build_frame(view) for view in views]
        self.renderer.set_frames(frames, self.point_size.value * 0.01, self.cam_scale.value)

        if self.range_slider is None:
            last = len(frames) - 1
            with self.frames_folder:
                self.range_slider = self.server.gui.add_multi_slider(
                    'Frame range', 0, max(last, 1), 1, (0, last), order=0)
            self.range_slider.on_update(lambda _: self.refresh_all())

        self.refresh_all()

    def visible_range(self):
        if self.range_slider is None:
            return 0, -1
        return self.range_slider.value

    def refresh_all(self):
        self.refresh_points()
        self.refresh_cameras()
        self.refresh_trajectory()
        self._update_status()

    def refresh_points(self):
        first, last = self.visible_range()
        self.renderer.refresh_points(first, last, self.conf_thr.value,
                                     int(self.budget.value * 1e6))

    def refresh_cameras(self):
        first, last = self.visible_range()
        self.renderer.refresh_cameras(first, last, self.show_cams.value, self.cam_scale.value)

    def refresh_trajectory(self):
        first, last = self.visible_range()
        self.renderer.refresh_trajectory(first, last, self.show_traj.value)

    def set_source(self, source):
        if self.raw is None or (source == 'Cleaned' and self.cleaned is None):
            return
        with self.loading:
            self.build_scene()

    def _rebuild_conf_slider(self, upper, disabled):
        value = self.conf_thr.value if self.conf_thr is not None else CONF_INIT
        if self.conf_thr is not None:
            self.conf_thr.remove()
        with self.points_folder:
            self.conf_thr = self.server.gui.add_slider(
                'Confidence threshold', 0, upper, 0.1, min(value, upper),
                disabled=disabled, order=1)
        self.conf_thr.on_update(self._on_conf_thr)

    def _on_conf_thr(self, _event):
        self.refresh_points()
        self._update_status()

    def _on_point_size(self, _event):
        self.renderer.set_point_size(self.point_size.value * 0.01)

    def _on_clean(self, _event):
        self.clean_button.disabled = True
        threading.Thread(target=self._clean_worker, args=(self.conf_thr.value,), daemon=True).start()

    def _clean_worker(self, conf_thr):
        self.status.content = 'Cleaning, this takes a while...'
        try:
            cleaned = denoise_views(self.raw, conf_thr=conf_thr,
                                    carve_depth=0 if self.carve_note else 0.9)
        except Exception as error:
            print(f'Cleaning failed: {error}')
            self.status.content = f'Cleaning failed: `{error}`'
            self.clean_button.disabled = False
            return

        self.cleaned = to_numpy(cleaned)
        self.cleaned_thr = conf_thr
        self.clean_button.disabled = False
        self.source.disabled = False

        already_shown = self.source.value == 'Cleaned'
        self.source.value = 'Cleaned'
        if already_shown:
            self.set_source('Cleaned')  # an unchanged assignment fires no callback

    def _update_status(self):
        self.status.content = '  \n'.join(self._status_lines())

    def _status_lines(self):
        frames = self.renderer.frames
        if not frames:
            return ['No scene loaded.']
        if frames[0]['conf'] is None:
            return ['No confidence map in this file, so filtering and cleaning are unavailable. '
                    'Re-run the reconstruction with `--export_mode full`.']

        lines = [self.carve_note] if self.carve_note else []
        if self.cleaned_thr is None:
            lines.append('Not cleaned.')
        elif abs(self.cleaned_thr - self.conf_thr.value) > 1e-6:
            lines.append(f'Cleaned at conf ≥ {self.cleaned_thr:.1f} — stale, re-clean to apply '
                         f'conf ≥ {self.conf_thr.value:.1f}.')
        else:
            lines.append(f'Cleaned at conf ≥ {self.cleaned_thr:.1f}.')
        return lines


def arg_parser():
    parser = argparse.ArgumentParser(description='BLASt3R reconstruction viewer')
    scene = parser.add_mutually_exclusive_group()
    scene.add_argument('--scene', help='a single reconstruction to view')
    scene.add_argument('--scenes', help='a directory of reconstructions to pick from')
    parser.add_argument('--host', default='127.0.0.1',
                        help='interface to serve on; 0.0.0.0 exposes the viewer, and the files '
                             'it can read, to everyone who can reach this machine')
    parser.add_argument('--port', type=int, default=8080)
    return parser


def main(scene, scenes, host, port):
    paths = None
    if scenes is not None:
        paths = sorted(Path(scenes).glob('*.pth'))
        if not paths:
            raise FileNotFoundError(f'no .pth reconstruction found in {scenes}')

    server = viser.ViserServer(host=host, port=port, label='BLASt3R')
    Viewer(server, scene=scene, scenes=paths)
    server.sleep_forever()


if __name__ == '__main__':
    main(**vars(arg_parser().parse_args()))
