# Copyright (C) 2026-present Naver Corporation. All rights reserved.
#
# Adapted for BLASt3R from MASt3R (https://github.com/naver/mast3r),
# mast3r/retrieval/processor.py.

import numpy as np
import torch


class TorchFlatL2Index:
    """Exact nearest-centroid search in torch, standing in for faiss.GpuIndexFlatL2.

    Computes the same float32 distances as faiss' flat index, |x|² + |c|² - 2 x·c,
    so assignments only differ from faiss on ties at float precision. Quantizing
    the 300 descriptors of one image takes ~10ms here against ~600ms for
    faiss-cpu against a 64k codebook, and the online pipeline quantizes twice
    per frame.
    """
    QUERY_CHUNK = 1024  # rows of the distance matrix at once: 1024 x 64k floats is 256MB

    def __init__(self, points, device):
        self.points = torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32)).to(device)
        self.sq_norms = self.points.square().sum(dim=1)

    @torch.no_grad()
    def search(self, queries, k):
        """faiss' signature: (n, d) queries in, (n, k) distances and (n, k) ids out."""
        queries = torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float32))
        queries = queries.to(self.points.device)
        dists, ids = [], []
        for chunk in queries.split(self.QUERY_CHUNK):
            dist = chunk.square().sum(dim=1, keepdim=True) + self.sq_norms - 2 * chunk @ self.points.T
            best = dist.topk(k, dim=1, largest=False)
            dists.append(best.values)
            ids.append(best.indices)
        return torch.cat(dists).cpu().numpy(), torch.cat(ids).cpu().numpy()


try:
    try:
        import faiss
        faiss.StandardGpuResources()
    except AttributeError as e:
        import asmk.index

        class TorchGpuL2Index(asmk.index.FaissL2Index):
            """Codebook search on the GPU through torch. Clustering, which only
            training needs, keeps the inherited faiss CPU index."""
            def __init__(self, gpu_id):
                super().__init__()
                self.gpu_id = gpu_id

            def create_index(self, points, **index_kwargs):
                return TorchFlatL2Index(points, torch.device('cuda', self.gpu_id))

        asmk.index.FaissGpuL2Index = TorchGpuL2Index
        print('faiss has no GPU support: the retrieval codebook is searched with torch on the GPU')
except ImportError:
    print('Warning: faiss is not installed, image retrieval is disabled')

default_asmk_params = {'index': {'gpu_id': 0}, 'train_codebook': {'codebook': {'size': '64k'}},
                       'build_ivf': {'kernel': {'binary': True}, 'ivf': {'use_idf': False},
                                     'quantize': {'multiple_assignment': 1}, 'aggregate': {}},
                       'query_ivf': {'quantize': {'multiple_assignment': 5}, 'aggregate': {},
                                     'search': {'topk': None},
                                     'similarity': {'similarity_threshold': 0.0, 'alpha': 3.0}}}


def get_gpu_index(device):
    if not isinstance(device, torch.device):
        device = torch.device(device=device)

    if device.type == "cuda":
        if device.index is None:
            return torch.cuda.current_device()
        else:
            return device.index
    return None  # cpu


class Retriever(object):
    def __init__(self, retrieval_model, asmk):
        self.model = retrieval_model
        self.asmk = asmk

    def _preproc(self, feats, device):
        imids = []
        features = []
        with torch.no_grad():
            for i, feat in enumerate(feats):
                feat_ret, _, _ = self.model.forward_local(feat.unsqueeze(0).to(device))
                feat_ret = feat_ret.flatten(0, 1)  # .cpu()
                imids.append(i*torch.ones_like(feat_ret[:, 0], device=device).to(dtype=torch.int64))
                features.append(feat_ret)
        features = torch.cat(features, dim=0).cpu().numpy()
        imids = torch.cat(imids, dim=0).cpu().numpy()
        return features, imids

    def _sim_retrieval(self, asmk_dataset, qfeat, qimids):
        metadata, query_ids, ranks, ranked_scores = asmk_dataset.query_ivf(qfeat, qimids)
        # well ... scores are actually reordered according to ranks ...
        # so we redo it the other way around...
        scores = np.empty_like(ranked_scores)
        scores[np.arange(ranked_scores.shape[0])[:, None], ranks] = ranked_scores
        return torch.from_numpy(scores).to(device=qfeat.device)

    def __call__(self, query_feats, db_feats=None):
        qfeat, qimids = self._preproc(query_feats, query_feats.device)
        if db_feats is None:
            dbfeats, dbimids = qfeat, qimids
        else:
            dbfeats, dbimids = self._preproc(db_feats, db_feats.device)
        asmk_dataset = self.asmk.build_ivf(dbfeats, dbimids)
        scores = self._sim_retrieval(asmk_dataset, qfeat, qimids)
        del asmk_dataset
        return scores


class IncrementalRetriever(Retriever):
    def __init__(self, retrieval_model, asmk):
        super(IncrementalRetriever, self).__init__(retrieval_model, asmk)
        self.builder = self.asmk.create_ivf_builder()
        self.builder_count = 0

    def add(self, db_feats):
        num_new_dbs = len(db_feats)
        dbfeats, dbimids = self._preproc(db_feats, db_feats.device)
        self.builder.add(dbfeats, dbimids + self.builder_count)
        self.builder_count += num_new_dbs

    def __call__(self, query_feats, add_to_db=False):
        query_feats = query_feats.unsqueeze(0)
        qfeat, qimids = self._preproc(query_feats, query_feats.device)
        asmk_dataset = self.asmk.add_ivf_builder(self.builder)
        scores = self._sim_retrieval(asmk_dataset, qfeat, qimids)
        if add_to_db:
            self.add(query_feats)
        return scores
