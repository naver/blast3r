# Copyright (C) 2026-present Naver Corporation. All rights reserved.
#
# Adapted for BLASt3R from MASt3R (https://github.com/naver/mast3r),
# mast3r/retrieval/model.py.

import time
from tqdm import tqdm
import torch
import torch.nn as nn

from .whitener import pcawhitenlearn_shrinkage, Whitener
from blast3r.retrieval.image import DusterInputFromImageList
from blast3r.models.tools import AsListOfDicts

def weighted_spoc(feat, attn):
    """L2-normalized weighted sum-pooling of features.

    Args:
        feat: (B, N, C) features.
        attn: (B, N) weights.

    Returns:
        (B, C) pooled features.
    """
    return torch.nn.functional.normalize((feat*attn[:, :, None]).sum(dim=1), dim=1)


def how_select_local(feat, attn, nfeat):
    """Keep the `nfeat` highest-attention local features.

    Args:
        feat: (B, N, C) features.
        attn: (B, N) weights.
        nfeat: number of features to keep.
    """
    # get nfeat
    if nfeat < 0:
        assert nfeat >= -1.0
        nfeat = int(-nfeat * feat.size(1))
    else:
        nfeat = int(nfeat)
    # asort
    topk_attn, topk_indices = torch.topk(attn, min(nfeat, attn.size(1)), dim=1)
    topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, feat.size(2))
    topk_features = torch.gather(feat, 1, topk_indices_expanded)
    return topk_features, topk_attn, topk_indices



class RetrievalModel(nn.Module):

    def __init__(self, backbone, whiten=False, nfeat=300):
        super().__init__()
        self.backbone = backbone
        self.backbone_dim = backbone.enc_dim

        self.whiten = nn.Identity() if not whiten else Whitener(self.backbone_dim)
        self.do_whiten = whiten
        self.dim = self.backbone_dim
        self.attention = lambda x: x.norm(dim=-1)
        self.nfeat = nfeat

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        ss = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        ss = {k: v for k, v in ss.items() if not k.startswith('backbone')}
        return ss

    def forward_track_encoder(self, views):
        views = AsListOfDicts(views)
        imgs = views['img']
        token_pos = views['token_pos']
        true_shape = views['true_shape']

        B, ntokens, THREE, ph, pw = imgs.shape
        assert THREE == 3
        self.backbone.encoder.headless = True
        enc_feats, _, _ = self.backbone.encoder(imgs, token_pos, true_shape=true_shape)
        enc_feats = enc_feats[:, self.backbone.n_cls:, :] # remove null tokens
        return enc_feats

    def reinitialize_whitening(self, train_dataset, log_writer=None, max_nfeat_per_image=None,
                               seed=0, device='cuda', num_workers=8):
        if self.do_whiten:
            self.eval()
            sampler = train_dataset.make_sampler(batch_size=1, shuffle=True, world_size=1, rank=0, drop_last=True)
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=sampler, num_workers=num_workers,
                                                 pin_memory=True)
            loader.dataset.set_epoch(seed)
            loader.sampler.set_epoch(seed)

            print('Re-initialization of whitening')
            t = time.time()
            with torch.no_grad():
                features = []
                for d in tqdm(loader):
                    view = {k: v.to(device, non_blocking=True) for k, v in d.items()}
                    feat = self.forward_track_encoder(view)
                    feat = feat.flatten(0, 1)
                    if max_nfeat_per_image is not None and max_nfeat_per_image < feat.size(0):
                        l2norms = torch.linalg.vector_norm(feat, dim=1)
                        feat = feat[torch.argsort(-l2norms)[:max_nfeat_per_image], :]
                    features.append(feat.cpu())
            features = torch.cat(features, dim=0)
            features = features.numpy()
            m, P = pcawhitenlearn_shrinkage(features)
            self.whiten.load_state_dict({'m': torch.from_numpy(m), 'p': torch.from_numpy(P)})
            whiten_time = time.time()-t
            print(f'Done in {whiten_time:.1f} seconds')
            if log_writer is not None:
                log_writer.add_scalar('time/whiten', whiten_time, 0)

    def extract_features_and_attention(self, x):
        if isinstance(x, dict):
            backbone_feat = self.forward_track_encoder(x)
        else:
            backbone_feat = x
        backbone_feat_whitened = self.whiten(backbone_feat)
        attention = self.attention(backbone_feat_whitened)
        return backbone_feat_whitened, attention

    def forward_local(self, x):
        feat, attn = self.extract_features_and_attention(x)
        return how_select_local(feat, attn, self.nfeat)

    def forward_global(self, x):
        feat, attn = self.extract_features_and_attention(x)
        return weighted_spoc(feat, attn)

    def forward(self, x):
        return self.forward_global(x)


def identity(x):  # to avoid Can't pickle local object 'extract_local_features.<locals>
    return x


@torch.no_grad()
def extract_local_features(model, dataset, imsize, seed=0,
                           tocpu=False, max_nfeat_per_image=None, max_nfeat_per_image2=None,
                           device='cuda', num_workers=8):
    model.eval()
    if isinstance(dataset, list):
        imdataset = DusterInputFromImageList(dataset, imsize=imsize)
        loader = torch.utils.data.DataLoader(imdataset, batch_size=1, shuffle=False,
                                             num_workers=num_workers, pin_memory=True, collate_fn=identity)
    else:
        sampler = dataset.make_sampler(batch_size=1, shuffle=True, world_size=1, rank=0, drop_last=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler,
                                             num_workers=num_workers, pin_memory=True, collate_fn=identity)
        loader.sampler.set_epoch(seed)
        loader.sampler.set_epoch(seed)

    with torch.no_grad():
        features = []
        imids = []
        for i, d in enumerate(tqdm(loader)):
            dd = d[0]
            view = {
                'img': dd['img'].to(device, non_blocking=True).unsqueeze(0),
                'true_shape': torch.from_numpy(dd['true_shape']) .to(device, non_blocking=True).unsqueeze(0),
                'token_pos': torch.from_numpy(dd['token_pos']) .to(device, non_blocking=True).unsqueeze(0),
            }
            feat, _, _ = model.forward_local(view)
            feat = feat.flatten(0, 1)
            if max_nfeat_per_image is not None and feat.size(0) > max_nfeat_per_image:
                feat = feat[torch.randperm(feat.size(0))[:max_nfeat_per_image], :]
            if max_nfeat_per_image2 is not None and feat.size(0) > max_nfeat_per_image2:
                feat = feat[:max_nfeat_per_image2, :]
            features.append(feat)
            if tocpu:
                features[-1] = features[-1].cpu()
            imids.append(i*torch.ones_like(features[-1][:, 0]).to(dtype=torch.int64))
    features = torch.cat(features, dim=0)
    imids = torch.cat(imids, dim=0)
    return features, imids
