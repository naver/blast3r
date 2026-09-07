# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Builds the two CUDA extensions as part of a normal `pip install`.

Without TORCH_CUDA_ARCH_LIST, torch builds for the GPUs visible at build time
alone, which is both the fastest build and the right one for this machine. Set
it to target other GPUs, e.g. "7.0;7.5;8.0;8.6;9.0" for V100 through H100; it is
required when no GPU is visible while building. CUDA_HOME must point at a CUDA
toolkit.
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

EXT = 'src/blast3r/extensions'

setup(
    ext_modules=[
        CUDAExtension(
            name='blast3r.extensions.curope.cuda_rope',
            sources=[f'{EXT}/curope/rope.cpp', f'{EXT}/curope/kernels.cu'],
            extra_compile_args=dict(nvcc=['-O3', '--use_fast_math'], cxx=['-O3']),
        ),
        CUDAExtension(
            name='blast3r.extensions.cujac.ba_with_rigs_cuda_kernels',
            sources=[f'{EXT}/cujac/sfm_jacobian_rigs.cu'],
            extra_compile_args=dict(nvcc=['-O3'], cxx=['-O3']),
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
