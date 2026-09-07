# Copyright (C) 2026-present Naver Corporation. All rights reserved.

import torch


def read_scene_predictions(path):
    """Read one scene's BLASt3R output, returning {image_name: {K, cam2w, ...}}."""
    scene = torch.load(str(path), weights_only=False)
    return scene['views'] if 'views' in scene else scene
