# Copyright (C) 2026-present Naver Corporation. All rights reserved.

import torch
import torch.nn.functional as F
from collections import namedtuple

from blast3r.utils.device import todevice
from blast3r.utils.geometry import depthmap_to_pts3d
from blast3r.extensions.cujac import exp_se3, compute_residuals_with_rigs as residual_func
from .ba_solver import GaussNewtonSolver


# tangent regularization on dense (log-)depth maps

def _dense_tangent_prior_terms(
    rigs: torch.Tensor,
    K: torch.Tensor,
    z_coefs: torch.Tensor,
    log_depth_maps: torch.Tensor,
    conf: torch.Tensor,
    pts3d_ref: torch.Tensor, # reference pointmap in camera coords
    *,
    weight_tan: float = 0.0,
    tan_stride: int = 4,
    optim_Z0: bool = True,
    dtype: torch.dtype = torch.float32,
    compute_gn: bool = True,
    normalize_tangents: bool = True,
    eps = 1e-4,
):
    """Compute a dense tangent prior.

    For each image, we compare the 3D tangents from the *current* combined depth
    against tangents from the *main* depth channel (channel 0). The combined depth
    is assumed to be in **log-depth** form:

        L = sum_m z_coefs[m] * L_m
        D = exp(L)

    We compute forward differences on a downsampled grid (stride = tan_stride).

    Returns:
        tan_loss: scalar tensor.
        g_z: (n_imgs, C) J^T r contribution, or None if compute_gn=False.
        H_z: (n_imgs, C, C) J^T J contribution, or None if compute_gn=False.

    Note:
        Only derivatives w.r.t. z_coefs are returned, with no cross terms w.r.t.
        K, to keep integration into the existing BA solver simple.
    """
    if weight_tan <= 0:
        n_imgs, C = z_coefs.shape
        return (
            z_coefs.new_zeros((), dtype=dtype),
            (None if not compute_gn else z_coefs.new_zeros((n_imgs, C), dtype=dtype)),
            (None if not compute_gn else z_coefs.new_zeros((n_imgs, C, C), dtype=dtype)),
        )

    assert tan_stride >= 1
    assert None not in (log_depth_maps, conf, pts3d_ref)
    n_imgs, C = z_coefs.shape
    assert len(log_depth_maps) == len(conf) == len(pts3d_ref) == n_imgs

    device = z_coefs.device
    rigs = rigs.to(device=device)
    cam_ids = rigs[2].long()  # (n_imgs,)

    tan_loss = z_coefs.new_zeros((), dtype=dtype)
    g_z = None if not compute_gn else z_coefs.new_zeros((n_imgs, C), dtype=dtype)
    H_z = None if not compute_gn else z_coefs.new_zeros((n_imgs, C, C), dtype=dtype)

    for img_idx in range(n_imgs):
        cam_idx = int(cam_ids[img_idx].item())
        sub = slice(tan_stride//2, None, tan_stride)
        K_img = K[cam_idx].to(dtype).clone()
        K_img[:2, 2] -= sub.start
        K_img[:2] /= tan_stride

        # downsampled log-depth channels and confidence
        L = log_depth_maps[img_idx][sub, sub].to(dtype)
        assert L.ndim == 3 and L.shape[2] == C, f"expected log_depth_maps n_imgs x (H,W,C), got {L.shape}"
        if min(L.shape[1:]) < 2:
            continue

        a = z_coefs[img_idx].to(dtype)  # (C,)
        L_comb = torch.einsum('c,hwc->hw', a, L)  # (H',W')
        D = torch.exp(L_comb.clamp(max=80)) # prevent float32 overflow

        # pixel coordinates of the downsample grid in original pixel units
        X = depthmap_to_pts3d(D, K_img)
        X0 = pts3d_ref[img_idx][sub, sub]
        assert X.shape == X0.shape # (H, W, 3)
        if not normalize_tangents:
            # scale the reference pointmap to similar depth than X
            z0 = X0[..., 2]
            mask = (z0 > 1e-6)
            scale = torch.median(D[mask] / z0[mask]).detach() if mask.any() else D.new_tensor(1.0)
            X0 = scale * X0

        tx = X[1:, :, :] - X[:-1, :, :]
        ty = X[:, 1:, :] - X[:, :-1, :]
        tx0 = X0[1:, :, :] - X0[:-1, :, :]
        ty0 = X0[:, 1:, :] - X0[:, :-1, :]

        # normalization coeffs for the reference map
        inv_sx0 = 1.0 / tx0.norm(dim=-1, keepdim=True).clamp(min=eps)            # (H'-1,W',1)
        inv_sy0 = 1.0 / ty0.norm(dim=-1, keepdim=True).clamp(min=eps)            # (H',W'-1,1)
        tx0_hat = tx0 * inv_sx0
        ty0_hat = ty0 * inv_sy0

        if normalize_tangents:
            inv_sx = 1.0 / tx.norm(dim=-1, keepdim=True).clamp(min=eps)            # (H'-1,W',1)
            inv_sy = 1.0 / ty.norm(dim=-1, keepdim=True).clamp(min=eps)            # (H',W'-1,1)
        else:
            inv_sx = inv_sx0 # pretend X0 is close to X and use the same normalization
            inv_sy = inv_sy0 # pretend X0 is close to X and use the same normalization
        tx_hat = tx * inv_sx
        ty_hat = ty * inv_sy

        # residuals
        rx = tx_hat - tx0_hat
        ry = ty_hat - ty0_hat

        # confidence
        Cw = confident_in_planar_regions(X0[:,:,2])
        assert Cw.ndim == 2 and Cw.shape == D.shape, f"expected conf n_imgs x (H,W), got n_imgs x {Cw.shape}"
        wx = (Cw[1:, :] * Cw[:-1, :]).clamp(min=0).sqrt()
        wy = (Cw[:, 1:] * Cw[:, :-1]).clamp(min=0).sqrt()

        if rx.numel() > 0:
            tan_loss = tan_loss + weight_tan * (wx.unsqueeze(-1) * rx.square()).mean()
        if ry.numel() > 0:
            tan_loss = tan_loss + weight_tan * (wy.unsqueeze(-1) * ry.square()).mean()

        if not compute_gn:
            continue

        # Gauss-Newton contributions w.r.t. z_coefs only
        # dD/da_m = D * L_m
        S = D.unsqueeze(0) * L.permute(2,0,1)                               # (C,H',W')
        B = depthmap_to_pts3d(torch.ones_like(D), K_img)                    # (H',W',3) rays
        dX = S.unsqueeze(-1) * B.unsqueeze(0)                               # (C,H',W',3)
        dtx = dX[:, 1:, :, :] - dX[:, :-1, :, :]
        dty = dX[:, :, 1:, :] - dX[:, :, :-1, :]

        if normalize_tangents:
            # Project out the component along t_hat, then scale by 1/||t||.
            dot_x = (dtx * tx_hat.unsqueeze(0)).sum(dim=-1, keepdim=True)         # (C,H'-1,W',1)
            dtx = (dtx - tx_hat.unsqueeze(0) * dot_x) * inv_sx.unsqueeze(0)       # (C,H'-1,W',3)
            dot_y = (dty * ty_hat.unsqueeze(0)).sum(dim=-1, keepdim=True)         # (C,H',W'-1,1)
            dty = (dty - ty_hat.unsqueeze(0) * dot_y) * inv_sy.unsqueeze(0)       # (C,H',W'-1,3)
        else:
            # just multiplication by a constant
            dtx = dtx * inv_sx
            dty = dty * inv_sy

        # Flatten and accumulate using weighted inner products.
        if rx.numel() > 0:
            wx_f = wx.reshape(-1)
            rx_f = rx.reshape(-1, 3)
            dtx_f = dtx.reshape(C, -1, 3)
            scale_x = weight_tan / rx_f.numel()  # == weight_tan / (nx*3)
            g_z[img_idx] += scale_x * torch.einsum('n,cnk,nk->c', wx_f, dtx_f, rx_f)
            H_z[img_idx] += scale_x * torch.einsum('n,cnk,dnk->cd', wx_f, dtx_f, dtx_f)

        if ry.numel() > 0:
            wy_f = wy.reshape(-1)
            ry_f = ry.reshape(-1, 3)
            dty_f = dty.reshape(C, -1, 3)
            scale_y = weight_tan / ry_f.numel()  # == weight_tan / (ny*3)
            g_z[img_idx] += scale_y * torch.einsum('n,cnk,nk->c', wy_f, dty_f, ry_f)
            H_z[img_idx] += scale_y * torch.einsum('n,cnk,dnk->cd', wy_f, dty_f, dty_f)

    if compute_gn and (not optim_Z0) and C > 0:
        g_z[:, 0] = 0
        H_z[:, 0, :] = 0
        H_z[:, :, 0] = 0

    return tan_loss, g_z, H_z


def confident_in_planar_regions(D0, alpha=1e4):
    assert D0.ndim == 2

    laplX = 2*D0
    laplX[:,  1:] -= D0[:, :-1]
    laplX[:, :-1] -= D0[:, 1: ]

    laplY = 2*D0
    laplY[1: , :] -= D0[:-1, :]
    laplY[:-1, :] -= D0[1: , :]

    # normalize so that it is invariant to depth
    norm = -F.max_pool2d(-D0[None,None], 3, stride=1, padding=1) # min-pool 2d
    norm = norm.squeeze().clip(min=1e-12)
    laplX /= norm
    laplY /= norm

    lapl2 = laplX*laplX + laplY*laplY
    conf = torch.exp(-alpha * lapl2)
    return conf


# Gauss-Newton optimization

def lm_step( rigs, pids, pix2d, pix2d_std, pix2d_dep,
             K, P_rig2cam, P_w2rig, z_coefs, pts3d, # these variables are optimized
             depth_mode = 'log',
             pnorm=1,
             clip_pix_err=100,
             huber_delta=0.5,
             min_reproj=0.01,
             dampen=1e-12,
             min_loss_delta=1e-4,
             weight_z=1,
             # tangent prior (dense) on the combined depth vs main depth (channel 0)
             log_depth_maps=None, conf=None, pts3d_ref=None, weight_tan=0.0, tan_stride=4,
             normalize_tangents=True,
             optim_K=True, optim_Z=True, optim_P=True, optim_X=True, optim_Z0=True, **ba_options):
    """Levenberg-Marquardt step.

    (a) compute the residual vector r and its Jacobian J with respect to all parameters;
    (b) solve for  J @ delta = -r
                => delta = - pseudo_inv(J) @ r; and then
                => delta = - (JᵀJ)⁻¹ Jᵀ r; and then
    (c) update the parameters.

    We are minimizing Sum || rp_error(y, K, P, X) ||
        where rp_error(y, K, P, X) = { (y_u - (K.P.X)_u ) / y_std_u
                                     { (y_v - (K.P.X)_v ) / y_std_v
                                     { f.log( (y_z @ z_img) / (K.P.X)_z )
    """
    assert depth_mode in ('lin', 'log')
    is_depth_log = depth_mode=='log'
    if weight_tan > 0:
        # Implemented for log-depth combination only.
        assert is_depth_log, 'tangent prior currently only implemented for depth_mode="log"'
    n_imgs = rigs.shape[1]
    assert shape_of(rigs) == (3, n_imgs)
    n_cams = len(K)
    assert shape_of(K) == (n_cams, 3, 3)
    assert shape_of(P_rig2cam, is_contiguous=False) == (n_cams, 4, 4)
    n_nodes = len(P_w2rig)
    assert shape_of(P_w2rig) == (n_nodes, 4, 4)
    n_zcf = z_coefs.shape[1]
    assert shape_of(z_coefs) == (n_imgs, n_zcf)
    n_tracks = len(pts3d)
    assert shape_of(pts3d) == (n_tracks, 3)
    n_obs = len(pids)
    assert shape_of(pids) == (n_obs,)
    assert shape_of(pix2d) == (n_obs, 2)
    assert shape_of(pix2d_std) == (n_obs, 2)
    assert shape_of(pix2d_dep) == (n_obs, n_zcf), \
        f'expected pix2d_dep of shape {(n_obs, n_zcf)}, got {shape_of(pix2d_dep)}'
    assert 0 <= weight_z <= 10
    assert n_obs >= n_tracks, 'there must be more 2d kpts than tracks'
    assert huber_delta >= 0.1

    # For the tangent prior, we intentionally treat intrinsics as constant *within* an LM step.
    # This avoids needing cross-derivatives w.r.t. K (we only add GN terms for z_coefs).
    K_tan = K.detach()

    def objective(K, P_w2rig, pts3d, z_coefs):
        residuals_, err_pts_, w_pts_ = residual_func(
            rigs, K, P_rig2cam, P_w2rig, z_coefs, pts3d, pids, pix2d, pix2d_std, pix2d_dep, is_depth_log,
            weight_z, pnorm, clip_pix_err, huber_delta).view(-1,3,3).unbind(dim=1)

        loss = err_pts_.mean()
        if weight_tan > 0:
            tan_loss, _, _ = _dense_tangent_prior_terms(
                rigs, K_tan, z_coefs, log_depth_maps, conf, pts3d_ref,
                weight_tan=float(weight_tan), tan_stride=int(tan_stride), optim_Z0=optim_Z0,
                dtype=loss.dtype, compute_gn=False,
                normalize_tangents=bool(normalize_tangents)
            )
            loss = loss + tan_loss

        return loss, residuals_, w_pts_

    # get the current residual vector:
    init_loss, init_residuals, w_pts = objective(K, P_w2rig, pts3d, z_coefs)

    # sparse solver for Gauss-Newton step
    jac = GaussNewtonSolver(n_cams, n_nodes, n_imgs, n_tracks, pts3d.device, dampen=dampen, **ba_options)

    # Dense tangent prior (adds only to the z_coefs normal equations).
    z_prior = None
    if weight_tan > 0:
        # compute_gn=True returns per-image (H, g) for z_coefs.
        _, g_z, H_z = _dense_tangent_prior_terms(
            rigs, K_tan, z_coefs, log_depth_maps, conf, pts3d_ref,
            weight_tan=float(weight_tan), tan_stride=int(tan_stride), optim_Z0=optim_Z0,
            dtype=jac.dtype, compute_gn=True,
            normalize_tangents=bool(normalize_tangents),
        )
        z_prior = (H_z, g_z)
    J = jac.compute_jacobians(rigs, K, P_rig2cam, P_w2rig, z_coefs,
                              pts3d, pids, pix2d_std, pix2d_dep,
                              optim_K = optim_K, optim_Z = optim_Z, optim_Z0=optim_Z0,
                              optim_P = optim_P, optim_X = optim_X,
                              is_depth_log=is_depth_log)
    delta = jac.solve_delta(J, init_residuals, w_pts, z_prior=z_prior)
    delta_K, delta_P, delta_zcf, delta_pts = jac.unpack_delta_inplace(delta)

    # Update the parameters.
    update = dict()

    last_a = 1
    for a in [0.5, 0.25, 0.125, 1/64, 1/512, 1/4096, 0]:
        update['K'] = update_K(K, delta_K)
        update['P_w2rig'] = exp_se3(-delta_P) @ P_w2rig
        update['z_coefs'] = z_coefs - delta_zcf
        update['pts3d'] = pts3d - delta_pts

        new_loss, new_residuals, _ = objective(**update)
        if new_loss.isnan(): # this should not happen
            print('Warning: loss is NaN')
            new_loss = init_loss
        new_loss += min_loss_delta
        # Real observations cannot be explained perfectly. A residual this small
        # means the geometry went degenerate -- coincident camera centres, or a
        # baseline blown up against the depths -- which fits any depth and has a
        # lower loss than the correct solution, so the loss test cannot see it.
        new_reproj = float(reproj_score(new_residuals))
        degenerate = new_reproj < min_reproj
        if degenerate:
            print(f'Rejecting step: reprojection residual collapsed to {new_reproj:.2e}')
        if new_loss < init_loss and not degenerate:
            break
        delta *= a/last_a # let's try a smaller update. This updates all (delta_K, ..., delta_pts) because there're slices
        last_a = a
    else:
        print('Warning: loss is not decreasing')
    return update, BA_Status(init_loss, init_residuals, new_loss, new_residuals)


BA_Status = namedtuple('BA_Status', 'init_loss, init_residuals, new_loss, new_residuals')


def shape_of(tensor, is_contiguous=True):
    assert tensor.dtype in (torch.float32, torch.int32)
    if is_contiguous: assert tensor.is_contiguous()
    assert tensor.isfinite().all()
    return tensor.shape


def update_K(K0, delta_focal, min_f=100):
    dfx, dfy = delta_focal.T
    K = K0.clone()
    K[:,0,0] -= dfx
    K[:,1,1] -= dfy
    K[:,0,0].clip_(min=min_f)
    K[:,1,1].clip_(min=min_f)
    return K


def reproj_score(res):
    reproj_err = res[:,:2].norm(dim=-1)
    reproj_err = reproj_err.sort().values
    n80 = len(reproj_err) * 80 // 100
    return reproj_err[:n80].mean()


def iter_on( *iterators ):
    iterable = lambda it: isinstance(it, (tuple, list))

    # find max length
    max_len = 1
    for it in iterators:
        if iterable(it):
            max_len = max(max_len, len(it))

    new_iterators = []
    for it in iterators:
        if iterable(it):
            assert len(it) in (1, max_len), f'missing {max_len-len(it)} values for parameter={it}'
            new_iterators.append(it[0] if len(it)==1 else it)
        else:
            new_iterators.append(it)
    iterators = new_iterators

    for step in range(max_len):
        yield tuple(it[step] if iterable(it) else it for it in iterators)


def used_depth_channels(n_zcfs, n_channels, depth_mode):
    """How many leading multi-depth channels `bundle_adjustment` reads for these `n_zcfs`.

    Mirrors the per-phase rule in `bundle_adjustment`: a phase with `n` coefficients
    reads `n` channels, plus the frozen main channel in log mode unless `n` is 2 or
    every channel; a phase with `n_zcfs=None` reads them all. The channels past this
    count are never read, so a caller can drop them before storing the maps.
    """
    phases = n_zcfs if isinstance(n_zcfs, (list, tuple)) else [n_zcfs]
    used = 0
    for n in phases:
        if n is None:
            return n_channels
        if depth_mode[:3] == 'log' and not (n == 2 or n == n_channels):
            n += 1  # the first channel stays in, unoptimized
        used = max(used, n)
    return min(used, n_channels)


def bundle_adjustment(rigs_data, track_data, pts3d=None, z_coefs=None,
                      max_iters = 1000,
                      pnorm = 0.5,
                      optim_P = True,
                      optim_Z = True,
                      optim_Z0 = True,
                      optim_K = True,
                      optim_X = True,
                      weight_z = 0.1,
                      # dense tangent prior inputs (optional)
                      dense_depth = None,
                      dense_conf = None,
                      dense_pts3d_ref = None,
                      tangent_reg = 0.0,
                      tan_stride = 4,
                      normalize_tangents = True,
                      n_zcfs = None,
                      no_Jfz = False,
                      max_n_phases = -1,
                      device='cuda',
                      verbose = True,
                      **other_lm_kwargs):
    """Main bundle-adjustment loop."""
    trk = track_data.to(device)
    rig = rigs_data.to(device)

    # Optional dense inputs for the tangent prior.
    dense_depth = todevice(dense_depth, device)
    dense_conf = todevice(dense_conf, device)
    dense_pts3d_ref = todevice(dense_pts3d_ref, device)

    if z_coefs is None:
        z_coefs = rig.z_coefs_from_scale(trk)
    original_pix2d_dep = trk.pix2d_dep

    pts3d = todevice(pts3d, device)
    assert pts3d is not None, 'the 3D points must be initialized before bundle adjustment'

    # variables to optimize
    variables = dict(K=rig.K, P_rig2cam=rig.P_rig2cam, P_w2rig=rig.P_w2rig, z_coefs=z_coefs, pts3d=pts3d)

    # opimize several pnorms in turns, e.g. 1 -> 0.5
    phase_num = 0
    for         pnorm, max_iters, optim_K, optim_P, optim_Z, optim_X, weight_z, n_zcfs, tangent_reg in \
        iter_on(pnorm, max_iters, optim_K, optim_P, optim_Z, optim_X, weight_z, n_zcfs, tangent_reg):

        phase_num += 1
        if phase_num > (max_n_phases % 1000): break

        # remove/add depth components
        dense_depth_use = dense_depth
        if n_zcfs is not None:
            z_coefs = variables['z_coefs']

            optim_Z0 = True
            if rig.depth_mode[:3] == 'log':
                optim_Z0 = (n_zcfs == 2) or (n_zcfs == original_pix2d_dep.shape[1]) # don't touch the first coef if less than 17
                n_zcfs += not(optim_Z0) # include the 1st channel, which is not optimized anyway

            # this is what BA will use
            trk.pix2d_dep = original_pix2d_dep[:, :n_zcfs].contiguous()

            # Keep dense log-depth maps in sync with the active number of channels.
            if dense_depth is not None:
                assert len(dense_depth) == rig.n_imgs, "log_depth_maps must be n_imgs x (H, W, C)"
                assert all(d.shape[-1] >= n_zcfs for d in dense_depth), "not enough dense depth channels for the requested n_zcfs"
                dense_depth_use = [d[..., :n_zcfs] for d in dense_depth]

            if n_zcfs < z_coefs.shape[1]: # removing depthmaps
                assert n_zcfs >= 1
                variables['z_coefs'] = z_coefs[:, :n_zcfs].contiguous()

            elif n_zcfs > z_coefs.shape[1]: # adding depthmaps
                assert trk.pix2d_dep.shape[1] == n_zcfs, 'not enough multi-depthmaps in the original data'
                variables['z_coefs'] = torch.cat((z_coefs, z_coefs.new_zeros((len(z_coefs),n_zcfs-z_coefs.shape[1]))), dim=1).contiguous()

        if verbose:
            print(f"Starting phase {phase_num}, with {variables['z_coefs'].shape[1]} z-coefs.")
        iter = 0
        while iter < max_iters:
            update, status = lm_step(rig.infos(trk), trk.pids, trk.pix2d, trk.pix2d_std, trk.pix2d_dep, **variables,
                                     log_depth_maps=dense_depth_use, conf=dense_conf, pts3d_ref=dense_pts3d_ref,
                                     weight_tan=tangent_reg, tan_stride=tan_stride, normalize_tangents=normalize_tangents,
                                     optim_K=optim_K, optim_Z=optim_Z, optim_P=optim_P, optim_X=optim_X, optim_Z0=optim_Z0,
                                     pnorm=pnorm, weight_z=weight_z, no_grad_f_from_rz=no_Jfz, depth_mode=rig.depth_mode,
                                     dampen=1e-12, adaptive_dampen=True, dtype=torch.float64, **other_lm_kwargs)
            if verbose:
                reproj_before = reproj_score(status.init_residuals)
                reproj_after = reproj_score(status.new_residuals)
                avg_focal = float((update['K'][:,0,0].mean() + update['K'][:,1,1].mean())/2)
                print(f"Loss at {iter=}: {float(status.init_loss):.3f} --> {float(status.new_loss):.3f}, reproj_score: {reproj_before:.3f} --> {reproj_after=:.3f}, {avg_focal=:.2f}")
            if status.new_loss >= status.init_loss:
                break
            variables.update(update)
            iter += 1

    trk.pix2d_dep = original_pix2d_dep # restore
    pts3d = variables.pop('pts3d')
    return rigs_data.update(**variables), pts3d
