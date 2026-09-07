# Copyright (C) 2026-present Naver Corporation. All rights reserved.
#
# Adapted for BLASt3R from MASt3R (https://github.com/naver/mast3r),
# mast3r/retrieval/model.py. `pcawhitenlearn_shrinkage` derives from
# how (https://github.com/gtolias/how), licensed under the MIT License.

import numpy as np
import torch
import torch.nn as nn


def pcawhitenlearn_shrinkage(X, s=1.0):
    """Learn PCA whitening with shrinkage from the given descriptors."""
    # from https://github.com/gtolias/how/blob/4d73c88e0ffb55506e2ce6249e2a015ef6ccf79f/how/utils/whitening.py#L20
    # MIT License
    N = X.shape[0]

    # learning PCA w/o annotations
    m = X.mean(axis=0, keepdims=True)
    Xc = X - m
    Xcov = np.dot(Xc.T, Xc)
    Xcov = (Xcov + Xcov.T) / (2*N)
    eigval, eigvec = np.linalg.eig(Xcov)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    eigval = np.clip(eigval, a_min=1e-14, a_max=None)
    P = np.dot(np.linalg.inv(np.diag(np.power(eigval, 0.5*s))), eigvec.T)

    return m, P.T


class Whitener(nn.Module):

    def __init__(self, dim, l2norm=None):
        super().__init__()
        self.m = torch.nn.Parameter(torch.zeros((1, dim)).double())
        self.p = torch.nn.Parameter(torch.eye(dim, dim).double())
        self.l2norm = l2norm  # if not None, apply l2 norm along a given dimension

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):
            shape = x.size()
            input_type = x.dtype
            x_reshaped = x.view(-1, shape[-1]).to(dtype=self.m.dtype)
            # center the input data
            x_centered = x_reshaped - self.m
            # apply PCA transformation
            pca_output = torch.matmul(x_centered, self.p)
            # reshape back
            pca_output_shape = shape  # list(shape[:-1]) + [shape[-1]]
            pca_output = pca_output.view(pca_output_shape)
            if self.l2norm is not None:
                return torch.nn.functional.normalize(pca_output, dim=self.l2norm).to(dtype=input_type)
            return pca_output.to(dtype=input_type)
