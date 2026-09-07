# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Command-line arguments shared by the offline and online drivers."""
import argparse

import torch

from blast3r.inference.loading import CKPT_DEPTH, CKPT_MATCH


def torch_dtype(arg):
    return getattr(torch, arg)


def my_bool(arg):
    """Accept 'True', 'False', '0' or '1'."""
    if isinstance(arg, bool):
        return arg
    try:
        return {'True': True, 'False': False, '1': True, '0': False}[str(arg)]
    except KeyError:
        raise argparse.ArgumentTypeError(
            f"Wrong boolean parameter {arg}, expected 'True', 'False', '0' or '1'")


def base_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--images', required=True, nargs='+',
                        help='one or more image directories or video files')
    parser.add_argument('--output', required=True, help='output directory, or a .pth file')
    parser.add_argument('--name', default=None,
                        help='scene name for the output file (default: the image directory name)')
    parser.add_argument('--ckpt_match', default=CKPT_MATCH,
                        help='multi-view matching model: a local directory, or a Hugging Face '
                             f'repo id to download (default: {CKPT_MATCH})')
    parser.add_argument('--ckpt_depth', default=CKPT_DEPTH,
                        help='multi-channel depth model: a local directory, or a Hugging Face '
                             f'repo id to download (default: {CKPT_DEPTH})')

    parser.add_argument('--size', type=int, default=512, help='long-edge image size for processing')
    parser.add_argument('--force_ar', type=float, default=None, help='force aspect ratio for image cropping')
    parser.add_argument('--skip_existing', action='store_true', help='skip existing output files')
    parser.add_argument('--skip_failures', action='store_true', help='skip scenes that error out')
    parser.add_argument('--export_mode', choices=['pose', 'full'], default='full',
                        help="what to store in the output: 'pose' keeps the intrinsics and poses "
                             "only, 'full' adds the image, depthmap and confidence needed to "
                             'visualize the reconstruction')

    # matching
    parser.add_argument('--kpt_spacing', type=int, default=8, help='minimum space between keypoints')
    parser.add_argument('--keep_far_kpts', action='store_false', dest='remove_far_kpts',
                        help='keep imprecise keypoints')
    parser.add_argument('-ret', '--retrieval_mode', default='coreset_fps_30',
                        help="image retrieval strategy, e.g. 'coreset_fps_30', 'topsim', 'none'")
    parser.add_argument('-nr', '--num_retrieved_images', type=int, default=40,
                        help='number of images retrieved per query')

    # bundle adjustment
    parser.add_argument('--mono_cam', action='store_true', help='all images share one camera (single intrinsics)')
    parser.add_argument('--max_iters', type=int, default=100, nargs='+', help='max number of BA iterations')
    parser.add_argument('--pnorm', type=float, default=1, nargs='+', help='Huber norm')
    parser.add_argument('-wz', '--weight_z', type=float, default=[1e-4], nargs='+', help='weight_z in BA')
    parser.add_argument('-nz', '--n_zcfs', type=int, default=[1], nargs='+', help='number of depth components')
    parser.add_argument('-K', '--optim_K', type=my_bool, default=[True], nargs='+', help='optimize the focals')
    parser.add_argument('-P', '--optim_P', type=my_bool, default=[True], nargs='+', help='optimize the poses')
    parser.add_argument('--huber_delta', type=float, default=0.5, help='Huber loss half-width')
    parser.add_argument('--min_loss_delta', type=float, default=1e-4, help='min decrease of the loss')

    parser.add_argument('--optim_memory', action='store_true',
                        help='trade speed for GPU memory: memory-efficient attention, and keep '
                             'the optical flow on the host. Needed for large scenes on one GPU.')
    parser.add_argument('--denoise_depth', action='store_true', help='denoise the output depthmaps')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for the retrieval coreset, re-applied before each '
                             'scene so the draw does not depend on the position in a batch. '
                             'Does not make a run bit-reproducible: the CUDA bundle-adjustment '
                             'kernels accumulate with atomics, which reorders between runs.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--dtype', type=torch_dtype, default=torch.float32,
                        choices=[torch.float16, torch.bfloat16, torch.float32, torch.float64])
    return parser
