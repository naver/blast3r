# Copyright (C) 2026-present Naver Corporation. All rights reserved.

from .layers import Network


class Matcher (Network):
    """Base class for coarse and dense matches, providing convenience functions."""

    def init(self, flow_mode, dec_dim, n_cls, patch_size):
        raise NotImplementedError()
        return self

    def _remove_cls(self, tensor, hw, last='hw'):
        B, N_n, m = tensor.shape
        n = m if last == 'hw' else (self.n_cls + hw)

        tensor = tensor.view(B, N_n//n, n, m)
        if last != 'hw':
            tensor = tensor[:, :, self.n_cls:, :]
        elif n == self.n_cls + hw:
            tensor = tensor[:, :, self.n_cls:, self.n_cls:]
        elif n == hw:
            pass # nothing to do
        else:
            raise ValueError(f'bad {tensor.shape=} for {hw=}')
        return tensor.flatten(1,2)

    def _add_cls(self, coarse_scores):
        B,Nn,n = coarse_scores.shape
        coarse_scores = coarse_scores.view(B,Nn//n,n,n)
        res = coarse_scores.new_zeros((B, Nn//n, n+self.n_cls, n+self.n_cls))
        res[:, :, self.n_cls:, self.n_cls:] = coarse_scores
        return res.flatten(1,2)

