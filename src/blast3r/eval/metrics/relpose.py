# Copyright (C) 2026-present Naver Corporation. All rights reserved.
#
# Adapted for BLASt3R from PoseDiffusion (https://github.com/facebookresearch/PoseDiffusion)
# by Meta Platforms, Inc. Original code licensed under CC BY-NC 4.0.
import numpy as np
import torch

THRESHOLDS = (5, 15, 30)


def rotation_angle(rot_gt, rot_pred):
    B = rot_gt.shape[0]
    errs = []
    for i in range(B):
        cos = (torch.trace(torch.einsum('...ij, ...jk -> ...ik', rot_gt[i].T, rot_pred[i])) - 1) / 2
        cos = torch.clip(cos, -1.0, 1.0)  # numerical errors can push it out of bounds
        errs.append(torch.rad2deg(torch.abs(torch.arccos(cos))))
    return torch.tensor(errs)


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t = t / (torch.norm(t, dim=1, keepdim=True) + eps)
    t_gt = t_gt / (torch.norm(t_gt, dim=1, keepdim=True) + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def translation_angle(tvec_gt, tvec_pred):
    return compare_translation_by_angle(tvec_gt, tvec_pred) * 180.0 / np.pi


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """Area under the cumulative curve of max(rotation, translation) error."""
    max_errors = np.max(np.stack((r_error, t_error), axis=1), axis=1)
    if len(max_errors) == 0:
        return None

    histogram, _ = np.histogram(max_errors, bins=np.arange(max_threshold + 1))
    normalized = histogram.astype(float) / float(len(max_errors))
    return np.mean(np.cumsum(normalized))


def compute_pose_errors(gt_c2ws, pred_c2ws):
    """Relative-pose accuracy over all image pairs of one scene.

    Errors are computed on relative poses, so the two trajectories need not
    share a coordinate frame or scale.
    """
    gt_w2cs = torch.inverse(gt_c2ws)
    pred_w2cs = torch.inverse(pred_c2ws)

    N = len(pred_c2ws)
    i1s, i2s = torch.triu_indices(N, N, offset=1)

    gt_rel = gt_w2cs[i1s].bmm(gt_c2ws[i2s])
    pred_rel = pred_w2cs[i1s].bmm(pred_c2ws[i2s])

    rError = rotation_angle(gt_rel[:, :3, :3], pred_rel[:, :3, :3]).numpy()
    tError = translation_angle(gt_rel[:, :3, 3], pred_rel[:, :3, 3]).numpy()

    out = {'Auc_30': calculate_auc_np(rError, tError, max_threshold=30) * 100}
    for thr in THRESHOLDS:
        out[f'Racc_{thr}'] = np.mean(rError < thr) * 100
        out[f'Tacc_{thr}'] = np.mean(tError < thr) * 100
    out['Rmean'] = np.mean(rError)
    out['Tmean'] = np.mean(tError)
    out['n_pairs'] = len(rError)
    return out
