# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Turn a benchmark download into the layout the evaluation reads.

Every scene becomes one directory holding its images and a `gt.pkl` with
`filenames` and `poses` (Nx4x4 camera-to-world). The images are symlinked, so
preparing a benchmark costs no disk space beyond the ground truth.
"""
import hashlib
import pickle
from pathlib import Path

import numpy as np

# CO3D is evaluated on a fixed number of frames per scene.
N_SAMPLED_FRAMES = 10


def seed_from_string(name, base_seed=777):
    """Reproducible seed from a scene name, so a scene always draws the same frames."""
    digest = hashlib.sha256(name.encode('utf-8')).hexdigest()
    return (base_seed + int(digest, 16)) % (2**32)  # fits into np.uint32


def sample_frames(paths, scene, n_frames=N_SAMPLED_FRAMES):
    """Draw `n_frames` of a scene, as the published CO3D runs did.

    The candidate paths are sorted before shuffling: the draw depends on their
    order, so it must not depend on the order the filesystem listed them in.
    """
    paths = sorted(str(p) for p in paths)
    rng = np.random.default_rng(seed=seed_from_string(scene))
    rng.shuffle(paths)
    return paths[:n_frames]


def write_scene(out_dir, images, poses):
    """Symlink `images` into `out_dir` and write their poses as `gt.pkl`.

    Images are stored in name order: the evaluation forms its image pairs in the
    order the ground truth lists them, and the translation-angle error of a pair
    depends on which of the two images it is measured from.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images, poses = zip(*sorted(zip(images, poses), key=lambda pair: Path(pair[0]).name))

    filenames = []
    for image in images:
        src = Path(image).resolve()
        dst = out_dir / src.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
        filenames.append(src.name)

    gt = {'filenames': filenames, 'poses': [np.asarray(p, dtype=np.float32) for p in poses]}
    with open(out_dir / 'gt.pkl', 'wb') as f:
        pickle.dump(gt, f)
