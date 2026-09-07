# Copyright (C) 2026-present Naver Corporation. All rights reserved.

import torch
import torch.nn as nn

from .base_matcher import Matcher
from .tools import l2

__all__ = 'CoarseDot CoarseDot2'.split()


class CoarseMatcher (Matcher):
    """Coarse patch tracker. To be overloaded."""
    def init(self, dec_dim, n_cls):
        raise NotImplementedError()

    def forward(self, preds):
        raise NotImplementedError()

class CoarseDot (CoarseMatcher):
    """One descriptor per token, compared by dot-product then thresholded.

    The threshold predicts {matching, not_matching}. Loss = BCE.
    """
    def __init__(self, *, desc_dim=512, layer=-1, Proj=nn.Linear):
        super().__init__()
        self.desc_dim = desc_dim
        self.layer = layer
        self.Proj = Proj

    def init(self, flow_mode, dec_dim, n_cls):
        self.flow_mode = flow_mode
        self.proj_query = self.Proj(dec_dim, self.desc_dim)
        self.proj_db = self.Proj(dec_dim, self.desc_dim)
        self.n_cls = n_cls
        return self

    def dot_prod(self, preds):
        # basically, we compute the dot-product
        query_feat = self.proj_query(preds['query_decoder_tokens'][self.layer]) # (B, Nq, d)
        datab_feat = self.proj_db(preds['mem_tokens']) # (B, Nb, d)
        scores = l2(datab_feat) @ l2(query_feat).transpose(-1,-2)
        return scores # (B, Nb, Nq)

    def forward(self, preds):
        score_thr = 3 / self.desc_dim ** 0.5
        return self.dot_prod(preds).abs().clip(max=1), score_thr

class CoarseDot2 (CoarseDot):
    """One descriptor per token, scored against a constant trashbin.

    InfoNCE loss:
    - if db has a positive query patch: standard infoNCE
    - if not, must be more than a constant trashbin
    """
    def __init__(self, *, temperature=0.07, **kwargs):
        super().__init__(**kwargs)
        if temperature is None:
            self._temp = nn.Parameter(-torch.ones(1))
        else:
            self.register_buffer('_temp', torch.tensor(temperature).float(), persistent=False)
            self._temp.log_()
        self.score_thr = nn.Parameter(torch.zeros(1))

    def forward(self, preds):
        scores = self.dot_prod(preds)
        res = (scores / self.temperature, self.score_thr / self.temperature)
        return res

    @property
    def temperature(self):
        return self._temp.exp()
