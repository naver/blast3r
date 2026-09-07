# Copyright (C) 2026-present Naver Corporation. All rights reserved.

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from blast3r.image import IMAGE_SUFFIXES, is_video_file, load_image, load_images, load_video
from blast3r.models.coarse_matcher import CoarseDot2
from blast3r.models.dense_matcher import DenseDotViT2Mlp_Kpt
from blast3r.models.layers import LinearPatchifier, MlpHead, PositionAugmentor
from blast3r.models.model import CausalTrack3R_WithRayMaps
from blast3r.models.mono_depth import MonoDepth
from blast3r.utils.device import stack

# Released checkpoints, downloaded on first use unless a local directory is given.
CKPT_MATCH = 'naver/blast3r-matcher'
CKPT_DEPTH = 'naver/blast3r-depth'

# Classes a checkpoint's config.json may name. Nothing else is in scope.
MODEL_CLASSES = {
    c.__name__: c for c in (
        CausalTrack3R_WithRayMaps, CoarseDot2, DenseDotViT2Mlp_Kpt, MonoDepth,
        LinearPatchifier, MlpHead,
    )
}

# Keyword arguments whose value is itself a class name.
CLASS_KWARGS = ('Patchifier', 'Head')

# `Backbone.__mul__` only ever fills these two slots.
MATCHER_SLOTS = ('coarse_matcher', 'dense_matcher')

try:
    from asmk.asmk_method import ASMKMethod
    from asmk.codebook import Codebook
    from asmk.index import initialize_index

    from blast3r.retrieval.model import RetrievalModel
    from blast3r.retrieval.processor import Retriever, default_asmk_params, get_gpu_index
except ImportError:
    RetrievalModel = None
    print('Warning: asmk or faiss is not installed, image retrieval is disabled')


def load_dataset(images, online=False, name=None, **load_kw):
    """Yield one scene per input, each a directory of images or a video file.

    Scenes are loaded lazily so a many-scene benchmark run does not hold every
    scene's images in memory at once.
    """
    paths = [images] if isinstance(images, (str, Path)) else list(images)
    assert not (name and len(paths) > 1), '--name applies to a single input only'

    for path in paths:
        yield load_scene(Path(path), online=online, name=name, **load_kw)


class LazyFrames:
    """An image sequence loaded one frame at a time, as the online pipeline
    consumes it, so a long sequence does not sit in host memory in full.

    Behaves like the list it replaces: `len`, iteration, indexing and slicing
    (`frames[::step]` selects paths, loading nothing). `instances` gives the
    paths without loading either.
    """

    def __init__(self, paths, **load_kw):
        self.instances = list(paths)
        self.load_kw = load_kw

    def __len__(self):
        return len(self.instances)

    def __iter__(self):
        # decode the next frame while the pipeline works on the current one
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = [pool.submit(load_image, path, **self.load_kw) for path in self.instances[:1]]
            for path in self.instances[1:]:
                pending.append(pool.submit(load_image, path, **self.load_kw))
                yield pending.pop(0).result()
            if pending:
                yield pending.pop(0).result()

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return LazyFrames(self.instances[idx], **self.load_kw)
        return load_image(self.instances[idx], **self.load_kw)


def load_scene(path, online=False, name=None, **load_kw):
    if path.is_dir():
        images = sorted(p for p in path.iterdir()
                        if p.name.lower().endswith(IMAGE_SUFFIXES))
        assert images, f'no images found in directory {path}'
        label = name or path.name

        if online:
            return (path.parent.name, label, LazyFrames(images, **load_kw))

        views = load_images(images, **load_kw)
        views = {k: stack([view[k] for view in views]) for k in views[0]}
        views['label'] = [label] * len(images)
        return views

    if is_video_file(path):
        views = load_video(path, **load_kw)
        label = name or path.stem
        if online:
            return (path.parent.name, label, views)

        views = {k: stack([view[k] for view in views]) for k in views[0]}
        views['label'] = [label] * len(views['img'])
        return views

    raise FileNotFoundError(f'{path} is neither an image directory nor a readable video file')


def build_model(config):
    """Instantiate the model described by a checkpoint's `config.json`."""
    def resolve(name):
        if name not in MODEL_CLASSES:
            raise ValueError(f'config names {name!r}, which this release does not provide')
        return MODEL_CLASSES[name]

    def instantiate(spec):
        kwargs = dict(spec.get('kwargs', {}))
        for key in CLASS_KWARGS:
            if key in kwargs:
                kwargs[key] = resolve(kwargs[key])
        return resolve(spec['class'])(**kwargs)

    model = instantiate(config)
    for slot in MATCHER_SLOTS:
        if config.get(slot):
            model = model * instantiate(config[slot])
    return model


def resolve_checkpoint(path):
    """Take a local checkpoint directory or a Hugging Face repo id, return a directory."""
    if Path(path).is_dir():
        return Path(path)
    return Path(snapshot_download(str(path)))


def load_model(path, device, dtype=None):
    path = resolve_checkpoint(path)
    print('>> Loading model at', path)
    config = json.loads((path / 'config.json').read_text())

    print('>> Creating', config['class'])
    # Build and load straight on the target device: going through the host
    # costs a few GB of RAM for the two copies (init and checkpoint) of a
    # ~1GB model, which glibc does not always hand back afterwards.
    with torch.device(device):
        model = build_model(config)
    model.load_state_dict(load_file(path / 'model.safetensors', device=str(device)))
    model = model.to(device, dtype=dtype)
    model.force_dtype = dtype  # read back by the inference engine
    model.eval()

    # RoPE frequencies were trained at 512/16 tokens; rescale for other resolutions.
    for name, module in model.named_modules():
        if isinstance(module, PositionAugmentor):
            module.auto_resize = 512 // 16
            print(f'setting auto RoPE resize = {module.auto_resize} in', name)

    return model, load_retriever(path, model, device)


def load_retriever(path, model, device):
    config_file = path / 'retrieval.json'
    if not config_file.is_file():
        return None
    if RetrievalModel is None:
        print('Warning: checkpoint provides retrieval weights but asmk or faiss is missing')
        return None

    retrieval = json.loads(config_file.read_text())
    tensors = load_file(path / 'retrieval.safetensors')

    retrieval_model = RetrievalModel(model, whiten=retrieval['whiten'], nfeat=retrieval['nfeat'])
    weights = {k.removeprefix('model.'): v for k, v in tensors.items() if k.startswith('model.')}
    msg = retrieval_model.load_state_dict(weights, strict=False)
    assert all(k.startswith('backbone') for k in msg.missing_keys), msg.missing_keys
    assert not msg.unexpected_keys, msg.unexpected_keys
    retrieval_model = retrieval_model.to(device)
    retrieval_model.eval()

    asmk_params = copy.deepcopy(default_asmk_params)
    asmk_params['train_codebook']['codebook']['size'] = retrieval['nclusters']
    asmk_params['index']['gpu_id'] = get_gpu_index(device=device)

    index_factory = initialize_index(asmk_params['index']['gpu_id'])
    codebook_state = dict(retrieval['codebook'], state={
        k.removeprefix('codebook.'): v.numpy()
        for k, v in tensors.items() if k.startswith('codebook.')})
    codebook = Codebook.initialize_from_state(codebook_state, index_factory=index_factory)
    codebook.index()

    return Retriever(retrieval_model, ASMKMethod(asmk_params, {}, codebook=codebook))


class DelayedLoadingModel:
    """Defer the (slow) checkpoint load until the model is first used."""
    def __init__(self, load_func, force_dtype=None):
        self._model = None
        self._load_func = load_func
        self.force_dtype = force_dtype

    @property
    def __model(self):
        if self._model is None:
            self._model = self._load_func()
        return self._model

    def __call__(self, *args, **kwargs):
        return self.__model(*args, **kwargs)

    def __getattr__(self, key):
        return getattr(self.__model, key)


def load_depth_model(path, device='cuda', dtype=None):
    return DelayedLoadingModel(lambda: load_model(path, device=device, dtype=dtype)[0], force_dtype=dtype)


def load_match_model(path, device='cuda', dtype=None):
    # The matcher is loaded on first use and then reused: it is requested once
    # per scene, and rebuilding the ASMK index each time would exhaust the GPU.
    loaded = []

    def load_func():
        if not loaded:
            model, retriever = load_model(path, device=device, dtype=dtype)
            # the foreground-mask head is trained but unused at inference
            model.manage_foreground = False
            model.encoder.patchify.manage_foreground = False
            model.dense_matcher.MAX_PAIRS = 2**20
            loaded.append((model, retriever))
        return loaded[0]
    return load_func


def get_ntokens(views):
    H = views['true_shape'][..., 0]
    W = views['true_shape'][..., 1]
    P = views['patch_size']
    try:
        P = int(P)
    except TypeError:
        P = int(P[0])
    ntokens = (H * W // P**2)
    return int(ntokens.max()), P
