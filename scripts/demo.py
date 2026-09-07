#!/usr/bin/env python3
# Copyright (C) 2026-present Naver Corporation. All rights reserved.
#
# Adapted for BLASt3R from MUSt3R (https://github.com/naver/must3r),
# must3r/demo/viser.py.

"""End-to-end demo: pick images, reconstruct, watch the result in the browser.

    python scripts/demo.py

A gradio control panel with the viser scene embedded in it. Online mode draws
the reconstruction while it runs, moving the earlier cameras whenever a global
bundle adjustment revises them; offline mode draws the spanning-tree
initialization first, then the adjusted result.

Both servers run in this process, so the browser needs two reachable ports. The
tunnel command is printed at startup.
"""
import argparse
import os
import shutil
import socket
import tempfile
import threading
import traceback
from pathlib import Path

# Every online frame allocates and frees attention buffers of a few GB, and the
# default allocator fragments on that: it ended up reserving 10GB for 4GB in use.
# Has to be set before torch initializes CUDA.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import gradio
import numpy as np
import torch
import viser

from blast3r.image import IMAGE_SUFFIXES, is_video_file
from blast3r.inference.denoise import denoise_views
from blast3r.inference.loading import (CKPT_DEPTH, CKPT_MATCH, LazyFrames, load_depth_model,
                                       load_match_model, load_scene)
from blast3r.inference.pipeline import run_inference, run_online_inference
from blast3r.inference.results import compress, prepare_results
from blast3r.inference.rigs import update_views
from blast3r.utils.device import to_numpy
from blast3r.viz import CONF_INIT, CONF_MAX, SceneRenderer, build_frame, conf_upper_bound

# The published settings. The UI exposes only the ones that change what you see.
BA_DEFAULTS = dict(kpt_spacing=8, remove_far_kpts=True, retrieval_mode='coreset_fps_30',
                   max_iters=100, pnorm=1, weight_z=[1e-4], n_zcfs=[1],
                   optim_K=[True], optim_P=[True], huber_delta=0.5, min_loss_delta=1e-4)
ONLINE_DEFAULTS = dict(max_iters_local=10, reduce_tracks=False, max_num_tracks=100_000,
                       kpt_conf_thr=None)

LIVE_BUDGET = 0.5  # millions of points; the live view pays this on every update
SPINNER = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'  # one frame per status poll, so a long stage still looks alive


class Stopped(Exception):
    """Raised on the inference thread when the panel asks the run to stop."""




def cpu_view(view):
    """The parts of a view that bundle adjustment never revises, on the host.

    The image is only ever displayed, so it is kept as uint8 the way the saved
    scene stores it: a quarter of the float32 the pipeline works with.
    """
    view = to_numpy({key: val for key, val in view.items() if key != 'pts3d'})
    if 'img' in view:
        view['img'] = compress('img', view['img'])
    return view


class Mailbox:
    """A one-slot handoff. The producer overwrites whatever the consumer has not
    picked up, so a slow renderer can never hold back inference."""

    def __init__(self):
        self._slot = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    def put(self, item):
        with self._lock:
            self._slot = item
        self._ready.set()

    def get(self, timeout):
        if not self._ready.wait(timeout):
            return None
        with self._lock:
            item, self._slot = self._slot, None
            self._ready.clear()
        return item

    def clear(self):
        with self._lock:
            self._slot = None
            self._ready.clear()


class LiveScene:
    """The viser half of the demo: the scene, every display control, and the
    render thread that keeps drawing while inference runs."""

    def __init__(self, server):
        self.server = server
        self.server.scene.set_up_direction('-y')
        self.renderer = SceneRenderer(server)

        self.views = []      # host copies, appended by the inference thread
        self.raw = None      # the finished reconstruction, once a run completes
        self.cleaned = None
        self.name = None
        self.names = None
        self.total = 0
        self.scale = 1.0
        self.conf_thr = None
        self.mailbox = Mailbox()
        self.lock = threading.RLock()
        self._written_cam = {}

        self._build_gui()
        self.server.on_client_connect(self._on_client_connect)
        threading.Thread(target=self._render_loop, daemon=True).start()

    # ------------------------------------------------------------------ GUI

    def _build_gui(self):
        gui = self.server.gui
        self.status = gui.add_markdown('No reconstruction yet.', order=0)
        self.progress = gui.add_progress_bar(0, order=1)

        self.points_folder = gui.add_folder('Points', order=2)
        with self.points_folder:
            self.point_size = gui.add_slider('Point size (cm)', 0.01, 2, 0.01, 0.25, order=0)
            self.budget = gui.add_slider('Point budget (M)', 0.1, 20, 0.1, LIVE_BUDGET, order=2)
        self._rebuild_conf_slider(CONF_MAX)

        with gui.add_folder('Frames', order=3):
            self.show_cams = gui.add_checkbox('Show cameras', True, order=0)
            self.cam_scale = gui.add_slider('Camera scale', 0.005, 0.1, 0.005, 0.05, order=1)
            self.show_traj = gui.add_checkbox('Show trajectory', True, order=2)
            self.follow = gui.add_checkbox('Follow camera', True, order=3)
            self.reset_view = gui.add_button('Reset view', order=4)

        with gui.add_folder('Cleaning', order=4):
            self.clean_button = gui.add_button('Clean point cloud', disabled=True, order=0)
            self.source = gui.add_dropdown('Source', ['Raw', 'Cleaned'], disabled=True, order=1)

        self.point_size.on_update(
            lambda _: self.renderer.set_point_size(self.point_size.value * 0.01))
        self.budget.on_update(lambda _: self.refresh_all())
        self.show_cams.on_update(lambda _: self.refresh_all())
        self.cam_scale.on_update(lambda _: self.refresh_all())
        self.show_traj.on_update(lambda _: self.refresh_all())
        self.reset_view.on_click(lambda _: self.fit_view())
        self.clean_button.on_click(self._on_clean)
        self.source.on_update(lambda _: self._on_source())

    def _rebuild_conf_slider(self, upper):
        value = self.conf_thr.value if self.conf_thr is not None else CONF_INIT
        if self.conf_thr is not None:
            self.conf_thr.remove()
        with self.points_folder:
            self.conf_thr = self.server.gui.add_slider(
                'Confidence threshold', 0, upper, 0.1, min(value, upper), order=1)
        self.conf_thr.on_update(lambda _: self.refresh_all())

    def refresh_all(self):
        count = len(self.renderer)
        if not count:
            return
        self.renderer.refresh_points(0, count - 1, self.conf_thr.value,
                                     int(self.budget.value * 1e6))
        self.renderer.refresh_cameras(0, count - 1, self.show_cams.value, self.cam_scale.value)
        self.renderer.refresh_trajectory(0, count - 1, self.show_traj.value)

    # ------------------------------------------------- inference-thread side

    def reset(self, total, message):
        """`total` of 0 means the stage count is unknown, so the bar runs full and
        animated rather than pretending to measure something."""
        with self.lock:
            self.renderer.clear()
            self.mailbox.clear()
            self.views = []
            self.raw = self.cleaned = None
            self.name = self.names = None
            self.total = total
            self.scale = 1.0
            self._written_cam.clear()
            self.follow.value = True
            self.clean_button.disabled = True
            self.source.disabled = True
            self.source.value = 'Raw'
            self.progress.value = 0 if total else 100
            self.progress.animated = True
            self.status.content = message

    def stage(self, message):
        """Report a step that runs before there is a scene to reset. Nothing counts
        its progress either, so the bar runs full and animated."""
        self.progress.value = 100
        self.progress.animated = True
        self.status.content = message

    def on_frame(self, i, res_views, rigs, global_ba):
        """Called by the online pipeline once per frame, on the inference thread.

        `rigs` is mutated in place by the next bundle adjustment, so it has to be
        copied here rather than handed over by reference.
        """
        # a frame that yielded no track never reaches here, so catch up rather
        # than assuming this is the first frame we have not seen
        while len(self.views) <= i:
            self.views.append(cpu_view(res_views[len(self.views)]))
        self.status.content = (
            'All frames tracked. Running final bundle adjustment…'
            if i + 1 >= self.total else f'Tracked {i + 1} of {self.total} frames…')
        self.mailbox.put((i, rigs.to('numpy'), global_ba))

    def on_init(self, views, rigs):
        """Called by the offline pipeline once the poses are initialized."""
        self.views = [cpu_view(view) for view in views]
        self.status.content = (f'Initialized {len(views)} poses. '
                               'Running bundle adjustment…')
        self.mailbox.put((len(views) - 1, rigs.to('numpy'), True))

    def finish(self, views, name, names):
        with self.lock:
            self.mailbox.clear()
            self.raw = to_numpy(views)
            self.views = []  # the live copies are superseded by the result
            self.cleaned = None
            self.name, self.names = name, names
            # widen to the scene's confidence range when it goes past the live cap
            self._rebuild_conf_slider(max(conf_upper_bound(self.raw), CONF_MAX))
            self._show(self.raw)
            self.clean_button.disabled = False
            self.progress.value = 100
            self.progress.animated = False
            if self.follow.value:  # the viewer never took over, so frame the result
                self.follow.value = False
                self.fit_view()
            self.status.content = f'{len(self.raw)} frames processed.'

    def halt(self):
        """A run the panel stopped: keep whatever is drawn, stop the animation."""
        self.progress.animated = False
        self.status.content = 'Stopped.'

    def fail(self, message):
        self.progress.animated = False
        self.progress.value = 0
        self.status.content = f'Failed: `{message}`'

    # ------------------------------------------------------ rendering thread

    def _render_loop(self):
        while True:
            item = self.mailbox.get(timeout=0.25)
            if item is None:
                continue
            try:
                self._draw(*item)
            except Exception:
                traceback.print_exc()

    def _draw(self, last, rigs, redraw_all):
        with self.lock:
            # the inference thread may have appended past the snapshot's frame
            views = update_views(self.views[:last + 1], rigs)
            if not views:
                return
            # The mailbox keeps only the newest update, and a frame without tracks
            # never posts one, so frames before `last` may not be drawn yet: draw
            # everything from the first missing frame, not just `last`.
            first = 0 if redraw_all else min(last, len(self.renderer.frames))
            for i in range(first, last + 1):
                self.renderer.set_frame(i, self._frame(i, views[i]),
                                        self.point_size.value * 0.01, self.cam_scale.value)
            self.refresh_all()
            self._follow(views[last])
            if self.total:
                self.progress.value = min(100.0, 100.0 * (last + 1) / self.total)

    def _frame(self, i, view):
        """Build frame `i`, keeping the draw order of a frame already on screen so
        the sampled points do not jump from one update to the next."""
        frame = build_frame(view)
        if i < len(self.renderer.frames):
            frame['order'] = self.renderer.frames[i]['order']
        return frame

    def _show(self, views):
        frames = [build_frame(view) for view in views]
        self.renderer.set_frames(frames, self.point_size.value * 0.01, self.cam_scale.value)
        self.refresh_all()

    # --------------------------------------------------------- follow camera

    def _follow(self, view):
        depth = np.asarray(view['depth'])
        finite = depth[np.isfinite(depth) & (depth > 0)]
        self.scale = float(np.median(finite)) if finite.size else 1.0
        if not self.follow.value:
            return

        cam2w = np.asarray(view['cam2w'])
        forward = cam2w[:3, 2]
        position = cam2w[:3, 3] - forward * self.scale
        look_at = cam2w[:3, 3] + forward * self.scale
        for client in self.server.get_clients().values():
            with client.atomic():
                client.camera.position = position
                client.camera.look_at = look_at
            self._written_cam[client.client_id] = position

    def view_target(self):
        """Where to look from to see the whole reconstruction, as
        (center, unit direction to look along, radius), or None if nothing is drawn.

        Percentiles rather than a bounding box: a handful of far outliers would
        otherwise push the camera back until the scene is a speck.
        """
        frames = self.renderer.frames
        sample = [frame['pts'][frame['order'][::max(1, len(frame['order']) // 2000)]]
                  for frame in frames if len(frame['order'])]
        if not sample:
            return None

        points = np.concatenate(sample)
        low, high = np.percentile(points, [2, 98], axis=0)
        center = (low + high) / 2
        radius = max(float(np.linalg.norm(high - low)) / 2, 1e-6)

        # look from where the cameras were, so the scene is seen the way it was shot
        eye = np.mean([frame['cam2w'][:3, 3] for frame in frames], axis=0)
        direction = center - eye
        distance = float(np.linalg.norm(direction))
        if distance < 1e-9:
            direction = np.array([0.0, 0.0, 1.0])
        else:
            direction = direction / distance
        return center, direction, radius

    def fit_view(self):
        """Frame the whole reconstruction for every connected viewer."""
        target = self.view_target()
        if target is None:
            return
        center, direction, radius = target

        for client in self.server.get_clients().values():
            back = radius / np.tan(client.camera.fov / 2) * 1.3
            position = center - direction * back
            with client.atomic():
                client.camera.position = position
                client.camera.look_at = center
            self._written_cam[client.client_id] = position

    def _on_client_connect(self, client):
        client.camera.on_update(self._on_camera_moved)

    def _on_camera_moved(self, camera):
        """A camera message we did not send means the viewer took over."""
        written = self._written_cam.get(camera.client.client_id)
        if written is None:
            return
        if np.linalg.norm(np.asarray(camera.position) - written) > 0.05 * self.scale:
            self.follow.value = False

    # --------------------------------------------------------------- cleaning

    def _on_clean(self, _event):
        self.clean_button.disabled = True
        threading.Thread(target=self._clean_worker, args=(self.conf_thr.value,),
                         daemon=True).start()

    def _clean_worker(self, conf_thr):
        self.status.content = 'Cleaning point cloud…'
        uniform = len({tuple(view['depth'].shape) for view in self.raw}) == 1
        try:
            cleaned = denoise_views(self.raw, conf_thr=conf_thr,
                                    carve_depth=0.9 if uniform else 0)
        except Exception as error:
            traceback.print_exc()
            self.status.content = f'Cleaning failed: `{error}`'
            self.clean_button.disabled = False
            return

        with self.lock:
            self.cleaned = to_numpy(cleaned)
            self.clean_button.disabled = False
            self.source.disabled = False
            self.status.content = f'Point cloud cleaned at confidence ≥ {conf_thr:.1f}.'
            self.source.value = 'Cleaned'
            self._on_source()

    def _on_source(self):
        views = self.cleaned if self.source.value == 'Cleaned' else self.raw
        if views is None:
            return
        with self.lock:
            self._show(views)


class Runner:
    """The gradio half: resolves the input, runs one reconstruction at a time,
    and hands frames to the scene as they are produced."""

    # the online reconstruction to run; a subclass can substitute another with the same interface
    online_pipeline = staticmethod(run_online_inference)

    def __init__(self, scene, ckpt_match, ckpt_depth, size, device, pin_memory=False):
        self.scene = scene
        self.size = size
        self.pin_memory = pin_memory
        self.downloads = Path(tempfile.mkdtemp(prefix='blast3r_scenes_'))
        self.scene_file = None
        self.stop = threading.Event()
        self.worker = None
        self.depther = load_depth_model(ckpt_depth, device=device)
        self.matcher = load_match_model(ckpt_match, device=device)

        print('>> Loading the models')
        self.matcher()
        self.depther.eval()

    def _resolve_input(self, uploads, folder):
        if folder:
            path = Path(folder).expanduser()
            if path.is_dir():
                return path, path.name
            if is_video_file(path):
                return path, path.stem
            raise ValueError(f'`{path}` is neither an image directory nor a video.')

        if not uploads:
            raise ValueError('Add images or a video first.')

        paths = [Path(upload) for upload in uploads]
        if len(paths) == 1 and is_video_file(paths[0]):
            return paths[0], paths[0].stem

        staged = Path(tempfile.mkdtemp(prefix='blast3r_demo_'))
        for path in paths:
            shutil.copy(path, staged / path.name)
        images = [path for path in staged.iterdir()
                  if path.suffix.lower() in IMAGE_SUFFIXES]
        if len(images) < 2:
            raise ValueError('Need at least 2 images, or a video.')
        return staged, 'upload'

    def run(self, *args):
        """A generator, so the panel keeps reporting the stage the scene is in;
        the spinner turns even while a stage holds the same text for minutes.

        Yields the button too, which reads Stop for as long as the run lasts.
        """
        self.stop.clear()
        self.scene_file = None
        outcome = []
        self.worker = threading.Thread(target=lambda: outcome.append(self._reconstruct(*args)),
                                       daemon=True)
        self.worker.start()
        yield self.scene.status.content, gradio.Button('Stop', variant='stop'), gradio.skip()
        tick = 0
        while self.worker.is_alive():
            self.worker.join(0.3)
            yield (f'{SPINNER[tick % len(SPINNER)]} {self.scene.status.content}',
                   gradio.skip(), gradio.skip())
            tick += 1
        yield (outcome[0] if outcome else '**The reconstruction thread died.**',
               gradio.Button('Run', variant='primary'),
               gradio.DownloadButton('Download scene', value=self.scene_file,
                                     interactive=self.scene_file is not None))

    def request_stop(self):
        """The same button, clicked while a run is live. The run notices at its next
        checkpoint: a frame boundary online, the end of pose initialization offline."""
        if self.worker is not None and self.worker.is_alive():
            self.stop.set()
            self.scene.status.content = 'Stopping…'  # the bar keeps whatever it reached

    def _checkpoint(self, callback):
        """Wrap a pipeline callback so every call is also a chance to stop."""
        def wrapped(*args):
            callback(*args)
            if self.stop.is_set():
                raise Stopped
        return wrapped

    def _until_stopped(self, items):
        for item in items:
            if self.stop.is_set():
                raise Stopped
            yield item

    def _reconstruct(self, uploads, folder, mode, mono_cam, frame_step, gba_step, num_retrieved):
        torch.cuda.empty_cache()
        torch.manual_seed(0)  # the retrieval coreset is drawn at random
        try:
            self.scene.stage('Reading input…')
            path, name = self._resolve_input(uploads, folder)
            if mode == 'online':
                views, names = self._online(path, name, mono_cam, int(frame_step),
                                            int(gba_step), int(num_retrieved))
            else:
                views, names = self._offline(path, mono_cam, int(num_retrieved))
        except Stopped:
            self.scene.halt()
            return 'Stopped. Frames reconstructed so far remain in the viewer.'
        except ValueError as error:
            return f'**{error}**'  # the input is wrong, not the reconstruction
        except Exception as error:
            traceback.print_exc()
            self.scene.fail(error)
            return f'**Reconstruction failed:** `{error}`'

        self.scene.finish(views, name, names)
        self.scene_file = self._write_scene()
        return f'{len(names)} frames processed. The scene is ready to download.'

    def _online(self, path, name, mono_cam, frame_step, gba_step, num_retrieved):
        self.scene.stage('Loading frames…')
        _tag, label, sequence = load_scene(path, online=True, name=name, size=self.size)
        if isinstance(sequence, LazyFrames):
            # an image directory: frames load as the pipeline reaches them
            sequence = sequence[::frame_step]
            names = [Path(instance).name for instance in sequence.instances]
        else:
            # a video is decoded in full up front
            sequence = list(self._until_stopped(sequence))[::frame_step]
            names = [Path(view['instance']).name for view in sequence]
        if len(sequence) < 2:
            raise ValueError(f'{len(sequence)} frame(s) after --frame_step; need at least 2.')

        self.scene.reset(len(sequence), f'Reconstructing {len(sequence)} frames…')
        views = self.online_pipeline(label, sequence, self.depther, self.matcher,
                                     mono_cam=mono_cam, gba_step=gba_step,
                                     num_retrieved_images=num_retrieved, pin_memory=self.pin_memory,
                                     on_frame=self._checkpoint(self.scene.on_frame),
                                     **ONLINE_DEFAULTS, **BA_DEFAULTS)
        return views, names

    def _offline(self, path, mono_cam, num_retrieved):
        self.scene.stage('Loading images…')
        loaded = load_scene(path, size=self.size)
        count = len(loaded['img'])
        if count < 2:
            raise ValueError(f'{count} image(s) found; need at least 2.')

        self.scene.reset(0, f'{count} images. Running depth prediction and matching…')
        views = run_inference(loaded, self.depther, self.matcher, mono_cam=mono_cam,
                              num_retrieved_images=num_retrieved,
                              on_init=self._checkpoint(self.scene.on_init), **BA_DEFAULTS)
        return views, [Path(instance).name for instance in loaded['instance']]

    def _write_scene(self):
        """The download button only serves a file it already holds, so the scene is
        written when the run ends rather than when the button is clicked."""
        out_file = self.downloads / f'{self.scene.name}.pth'
        torch.save(prepare_results(self.scene.raw, self.scene.names, export_mode='full'), out_file)
        return str(out_file)


def stage_uploads_privately():
    """gradio stages uploads in one machine-wide `<tmpdir>/gradio`. On a shared
    host the first user to upload owns it and everyone else gets EACCES, so give
    this process its own; TMPDIR still decides where that lands."""
    os.environ.setdefault('GRADIO_TEMP_DIR', tempfile.mkdtemp(prefix='blast3r_gradio_'))


def browser_host(host):
    """0.0.0.0 means every interface, which is not an address a browser resolves."""
    return socket.gethostname() if host in ('0.0.0.0', '::') else host


def print_tunnel(host, port, viser_port):
    if browser_host(host) != host:
        return  # already serving on a routable address
    print(f'\n>> Both ports have to reach your browser. From your machine:\n'
          f'   ssh -N -L {port}:localhost:{port} -L {viser_port}:localhost:{viser_port} '
          f'{socket.gethostname()}\n'
          f'   then open http://localhost:{port}, once the models have loaded\n')


def build_ui(runner, viser_port, allow_local_files):
    with gradio.Blocks(title='BLASt3R Demo') as demo:
        gradio.Markdown('## BLASt3R — reconstruct a scene end to end')
        with gradio.Row():
            with gradio.Column():
                uploads = gradio.File(label='Images, or one video', file_count='multiple',
                                      file_types=['image', 'video'], height=200)
                folder = gradio.Textbox(label='...or an image directory on the server',
                                        visible=allow_local_files)
                mode = gradio.Radio([('Offline — unordered collection', 'offline'),
                                     ('Online — sequence, drawn as it runs', 'online')],
                                    value='offline', label='Mode')
            with gradio.Column():
                mono_cam = gradio.Checkbox(True, label='One shared camera (--mono_cam)')
                num_retrieved = gradio.Slider(1, 100, value=40, step=1,
                                              label='Retrieved images per query')
                frame_step = gradio.Slider(1, 20, value=1, step=1, visible=False,
                                           label='Frame step')
                gba_step = gradio.Slider(1, 32, value=8, step=1, visible=False,
                                         label='Global bundle adjustment every N frames')
                run_btn = gradio.Button('Run', variant='primary')
                save_btn = gradio.DownloadButton('Download scene', interactive=False)
                status = gradio.Markdown('Add images and press **Run**.')

        # resize:vertical needs a non-visible overflow, and the bottom strip keeps
        # the drag handle clear of the iframe, which would swallow the pointer
        gradio.HTML(
            '<div style="width:100%; height:640px; min-height:200px; resize:vertical; '
            'overflow:auto; padding-bottom:10px; border:1px solid #e4e4e7; '
            'border-radius:4px">'
            '<div style="padding:6px 12px"><span style="color:#71717a">Reconstruction'
            '</span><span style="float:right">'
            '<a id="viser-link" target="_blank">Full screen</a></span></div>'
            '<iframe id="viser-frame" style="width:100%; height:calc(100% - 44px); '
            'border:none"></iframe></div>')

        # the viser URL has to be whatever address this browser reached the panel
        # on, which only the browser knows: a tunnel, an IP or a fully qualified
        # name all serve the same process.
        demo.load(None, js=f'''() => {{
            const url = `${{location.protocol}}//${{location.hostname}}:{viser_port}`;
            document.getElementById('viser-frame').src = url;
            document.getElementById('viser-link').href = url;
        }}''')

        mode.change(lambda chosen: [gradio.update(visible=chosen == 'online')] * 2,
                    inputs=[mode], outputs=[frame_step, gba_step])
        # the run listener ignores clicks while it is pending (trigger_mode='once'),
        # so a second click reaches the stop listener alone
        run_btn.click(runner.run,
                      inputs=[uploads, folder, mode, mono_cam, frame_step, gba_step,
                              num_retrieved],
                      outputs=[status, run_btn, save_btn], concurrency_limit=1,
                      show_progress_on=status)
        run_btn.click(runner.request_stop, queue=False)

    return demo


def arg_parser():
    parser = argparse.ArgumentParser(description='BLASt3R end-to-end demo')
    parser.add_argument('--ckpt_match', default=CKPT_MATCH,
                        help='multi-view matching model: a local directory, or a Hugging Face '
                             f'repo id to download (default: {CKPT_MATCH})')
    parser.add_argument('--ckpt_depth', default=CKPT_DEPTH,
                        help='multi-channel depth model: a local directory, or a Hugging Face '
                             f'repo id to download (default: {CKPT_DEPTH})')
    parser.add_argument('--size', type=int, default=512, help='long-edge image size for processing')
    parser.add_argument('--host', default='127.0.0.1',
                        help='interface to serve on; 0.0.0.0 exposes the demo to everyone who '
                             'can reach this machine')
    parser.add_argument('--port', type=int, default=8080, help='port for the main gradio server')
    parser.add_argument('--viser_port', type=int, default=8081,
                        help='port for the viser (visualization) server, which the browser also has to reach')
    parser.add_argument('--allow_local_files', action='store_true',
                        help='also accept an image directory on the server, which lets anyone '
                             'who can reach the demo read it')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--pin_memory', action='store_true',
                        help='online mode: keep the per-frame matcher tokens in pinned host RAM '
                             'instead of the GPU, so GPU memory stops growing with the sequence '
                             '(about 4MB per frame at size 512, for a ~170MB copy per frame)')
    return parser


def main(ckpt_match, ckpt_depth, size, host, port, viser_port, allow_local_files, device,
         pin_memory, runner_class=None):
    stage_uploads_privately()
    server = viser.ViserServer(host=host, port=viser_port, label='BLASt3R')
    scene = LiveScene(server)

    print_tunnel(host, port, viser_port)
    runner = (runner_class or Runner)(scene, ckpt_match, ckpt_depth, size, device,
                                      pin_memory=pin_memory)

    ui = build_ui(runner, viser_port, allow_local_files)
    ui.queue().launch(server_name=host, server_port=port, share=False, max_file_size='2gb',
                      show_error=True)


if __name__ == '__main__':
    main(**vars(arg_parser().parse_args()))
