# Copyright (C) 2026-present Naver Corporation. All rights reserved.

from dataclasses import dataclass
import numpy as np
import torch
import scipy.sparse.csgraph as csg
import roma

from blast3r.utils.device import get_device_str, todevice, zeros_like, ones_like, stack, concat, int32, cumsum, exp
from blast3r.utils.geometry import inv, bilinear_sampling, sparse_depth_to_pts3d, depthmap_to_pts3d
from blast3r.utils.post_process import estimate_focal_knowing_depth
from .tracks import ObservedTracks


@dataclass
class RigsData:
    # camera-level infos
    n_cams : int            # number of cameras (= number of rigs x number of cam/rig)
    cam_ids : np.ndarray    # img_id -> cam_id
    K : np.ndarray          # intrinsics, (n_cams, 3, 3)
    P_rig2cam : np.ndarray  # rig-to-cam poses, (n_cams, 4, 4)

    # img-level
    scale: np.ndarray       # linear scaling of the multi-depthmaps, (n_imgs,)
    z_coefs: np.ndarray     # multi-depth combination (lin or log), (n_imgs, n_zcf)
    depth_mode: str         # either 'lin' or 'log'

    # node-level infos
    n_nodes : int           # number of nodes (= number of instanciated rigs)
    nod_ids : np.ndarray    # img_id -> node_id
    P_w2rig : np.ndarray    # world-to-rig poses, (n_nodes, 4, 4)

    @property
    def n_imgs(self):
        assert len(self.cam_ids) == len(self.nod_ids)
        return len(self.cam_ids)

    @property
    def device(self):
        devices = {get_device_str(self.cam_ids), get_device_str(self.K), get_device_str(self.P_rig2cam),
                   get_device_str(self.z_coefs), get_device_str(self.nod_ids), get_device_str(self.P_w2rig)}
        if len(devices) == 1:
            return next(iter(devices))
        return None

    def to(self, device):
        if self.device == device: return self
        return RigsData(**todevice(vars(self), device))

    def infos(self, tracks):
        return int32(stack((cumsum(tracks.nppi, dim=-1), self.nod_ids, self.cam_ids)))

    def update(self, **updates):
        return RigsData(**dict(vars(self), **updates))

    def get_K(self, img_idx):
        c = self.cam_ids[img_idx] # cam id
        return self.K[c]

    def get_w2cam(self, img_idx):
        c = self.cam_ids[img_idx] # cam id
        n = self.nod_ids[img_idx] # node id
        w2cam = self.P_rig2cam[c] @ self.P_w2rig[n]
        return w2cam

    def get_cam2w(self, img_idx):
        return inv(self.get_w2cam(img_idx))

    def z_coefs_from_scale(self, tracks):
        if self.z_coefs is None:
            # create from scratch
            shape = (tracks.n_imgs, tracks.pix2d_dep.shape[-1])
            self.z_coefs = torch.zeros(shape, device=self.scale.device)
            start = 0

        elif len(self.z_coefs) < len(self.scale):
            # add missing coefs
            start = len(self.z_coefs)
            self.z_coefs = torch.cat((self.z_coefs, self.z_coefs.new_zeros((len(self.scale)-len(self.z_coefs), self.z_coefs.shape[1]))))
        else:
            # will do nothing if start == end
            start = len(self.z_coefs)

        # init values
        for i in range(start, len(self.z_coefs)):
            if self.depth_mode == 'lin':
                self.z_coefs[i, 0] = self.scale[i]
            elif self.depth_mode == 'log':
                self.z_coefs[i, 0] = 1 # main depth channel
                self.z_coefs[i, 1] = self.scale[i].log() # multiplication in logspace == addition of constant log
            else:
                raise ValueError(f'bad {self.depth_mode=}')

        return self.z_coefs

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # return a view of this RigsData() only for a selected frame
            sl1 = lambda idx: slice(int(idx), int(idx)+1)
            cam_id = sl1(self.cam_ids[idx])
            nod_id = sl1(self.nod_ids[idx])
            sidx = sl1(idx)
            zero = zeros_like(self.cam_ids[:1])

            res = RigsData( 1, zero, self.K[cam_id], self.P_rig2cam[cam_id],
                            self.scale[sidx], self.z_coefs[sidx] if idx < len(self.z_coefs) else None,
                            self.depth_mode, 1, zero, self.P_w2rig[nod_id])
        elif isinstance(idx, torch.Tensor):
            old_cam_ids = self.cam_ids[idx]
            old_nod_ids = self.nod_ids[idx]
            uniq_cam_ids, new_cam_ids = torch.unique(old_cam_ids, return_inverse=True)
            uniq_nod_ids, new_nod_ids = torch.unique(old_nod_ids, return_inverse=True)
            n_cams = len(uniq_cam_ids)
            n_nodes = len(uniq_nod_ids)

            res = RigsData( n_cams, new_cam_ids.int(), self.K[old_cam_ids], self.P_rig2cam[old_cam_ids],
                            self.scale[idx], self.z_coefs[idx],
                            self.depth_mode, n_nodes, new_nod_ids.int(), self.P_w2rig[old_nod_ids])
        return res

    def __setitem__(self, idxs, rigs):
        if idxs == slice(None):
            assert isinstance(rigs, RigsData)
            for key, new in vars(rigs).items():
                if isinstance(new, (np.ndarray, torch.Tensor)):
                    if key == 'z_coefs':
                        getattr(self, key)[:, :new.shape[1]] = new
                    else:
                        getattr(self, key)[:] = new

        elif isinstance(idxs, torch.Tensor):
            old_cam_ids = self.cam_ids[idxs]
            old_nod_ids = self.nod_ids[idxs]

            for key, new in vars(rigs).items():
                if isinstance(new, (np.ndarray, torch.Tensor)):
                    if key in 'cam_ids K P_rig2camidx':
                        idx = old_cam_ids
                    elif key in 'scale z_coefs':
                        idx = idxs
                    elif key in 'nod_ids P_w2rig':
                        idx = old_nod_ids
                    else:
                        raise NameError(f'bad {key=}')

                    mine = getattr(self, key)
                    if key == 'z_coefs':
                        mine[idx, :new.shape[1]] = new
                    else:
                        mine[idx] = new
        else:
            raise ValueError(f'bad {key=}')


def get_minspan_tree(tracks, start_frame=0, ret_affmat=False):
    assert isinstance(tracks, ObservedTracks)

    # get max spanning tree
    affinity_matrix = np.zeros((tracks.n_imgs, tracks.n_imgs))
    for i in range(start_frame, tracks.n_imgs):
        for j in range(i):
            n_matches = np.intersect1d(tracks.pids[tracks.img_slices[i]], tracks.pids[tracks.img_slices[j]]).size
            if n_matches < 3: continue # not enough to infer a transform
            affinity_matrix[i,j] = n_matches
    affinity_matrix += affinity_matrix.T

    if ret_affmat:
        return affinity_matrix

    graph = affinity_matrix > 0
    n_components, labels = csg.connected_components(graph, directed=False, return_labels=True)
    if n_components > 1:
        components = [np.where(labels == c)[0].tolist() for c in range(n_components)]
        raise ValueError(f'Disconnected track graph ({n_components} components): {components}')

    mst = csg.minimum_spanning_tree(-affinity_matrix)
    edges = [(int(i),int(j)) for i,j in zip(*mst.nonzero())]

    return edges


def init_poses_minspan_tree(views, tracks, depth_mode, mono_cam=False):
    assert isinstance(views, list) and all(isinstance(view, dict) for view in views)
    assert isinstance(tracks, ObservedTracks) and len(views) == tracks.n_imgs
    assert isinstance(tracks.pids, np.ndarray)
    device = views[0]['pts3d'].device

    # initialization
    K = estimate_all_K(views, mono_cam=mono_cam)
    P_rw  = torch.stack([torch.eye(4, device=device) for view in views])
    scale = torch.full((tracks.n_imgs,), np.nan, device=device)

    try: # copy pose if availble
        cam2w = views[0]['cam2w']
        P_rw[0] = inv(cam2w)
    except KeyError:
        pass
    scale[0] = 1
    done = lambda i: (scale[i] == scale[i])

    list_tree_edges = get_minspan_tree(tracks)

    max_stalled = 2 * len(list_tree_edges)
    stalled = 0

    while list_tree_edges:
        i,j = list_tree_edges.pop()

        if done(i) == done(j) == False:
            list_tree_edges.insert(0, (i,j)) # disconnected from the known part, so we queue it back
            stalled += 1
            if stalled > max_stalled:
                raise ValueError("Infinite loop detected. Graph may be disconnected.")
            continue
        stalled = 0
        assert done(i) != done(j)

        if done(j): # put the done one in first position
            i, j = j, i

        set_pose_scale(views, tracks, i,j, scale, P_rw)

    n_cams = n_nodes = tracks.n_imgs
    cam_ids = np.arange(tracks.n_imgs, dtype=np.int32)
    nod_ids = np.arange(tracks.n_imgs, dtype=np.int32)
    if mono_cam:
        n_cams = 1
        K = K[0:1]
        cam_ids[:] = 0

    P_cr = torch.eye(4, device=device)[None].expand(n_cams, 4, 4) # no rigs ==> identity
    return RigsData(n_cams, cam_ids, K, P_cr, scale, None, depth_mode[:3], n_nodes, nod_ids, P_rw.contiguous())


def init_last_pose(views, tracks, depth_mode, rigs=None, mono_cam=True, device='cpu'):
    assert isinstance(views, list) and all(isinstance(view, dict) for view in views)

    if len(views) == 1: # just create an empty rig
        assert rigs is None
        rigs = init_poses_minspan_tree(views, tracks, depth_mode, mono_cam=mono_cam)
        rigs.z_coefs_from_scale(tracks) # init z_coefs
        return rigs

    # augment an existing rig
    assert rigs.depth_mode == depth_mode[:3]
    assert rigs.n_nodes + 1 == len(views), f'expected {len(views)-1} nodes, got {rigs.n_nodes}'

    j = tracks.n_imgs-1
    affinity_matrix = get_minspan_tree(tracks, start_frame=j, ret_affmat=True)
    i = int(affinity_matrix[j].argmax())
    print('attaching query frame to closest past frame =', i)

    rigs.n_cams += 1
    rigs.cam_ids = concat((rigs.cam_ids, (0 if mono_cam else j)*ones_like(rigs.cam_ids, 1)))
    if not mono_cam:
        assert rigs.n_cams == len(views)
        rigs.K = concat((rigs.K, estimate_K(views[-1])[None]))
        rigs.P_rig2cam = rigs.P_rig2cam[:1].expand(rigs.n_cams, 4, 4)

    rigs.nod_ids = concat((rigs.nod_ids, j * ones_like(rigs.nod_ids, 1)))
    rigs.n_nodes += 1

    rigs.P_w2rig = concat((rigs.P_w2rig, torch.eye(4, device=rigs.P_w2rig.device)[None]))
    rigs.scale = concat((rigs.scale, torch.ones(1, device=rigs.scale.device)))

    # update pts3d in view i (for procrustes)
    new_view_i = update_views([views[i]], rigs[i])[0]
    views[i]['pts3d'] = depthmap_to_pts3d(new_view_i['depth'], new_view_i['K'])
    if mono_cam:
        views[j]['pts3d'] = depthmap_to_pts3d(views[j]['pts3d'][...,2], rigs.get_K(j))
    rigs.scale[i] = 1 # reset scale now that we have reset pts3d

    set_pose_scale(views, tracks, i, j, rigs.scale, rigs.P_w2rig)
    rigs.z_coefs_from_scale(tracks) # add new z_coefs

    return rigs


def estimate_all_K(views, mono_cam):
    K = torch.zeros((len(views),3,3), device=views[0]['pts3d'].device)

    for i, view in enumerate(views):
        if 'K' in view:
            K[i] = view['K']
            view['pts3d_from_depth&K'] = depthmap_to_pts3d(view['pts3d'][:,:,2], view['K'])
        else:
            K[i] = estimate_K(view)

    if mono_cam: # aggregate all focals, but leave the focal center alone
        median_f = K[:,0,0].median()
        K[:,0,0] = median_f
        K[:,1,1] = median_f

    return K


def estimate_K(view):
    pts3d = view['pts3d']
    H, W, THREE = pts3d.shape
    assert THREE == 3

    pp = torch.tensor((W/2, H/2), device=pts3d.device)
    f = estimate_focal_knowing_depth(pts3d.unsqueeze(0), pp, focal_mode='median')

    K = torch.eye(3, device=pts3d.device)
    K[0,0] = K[1,1] = f
    K[:2,2] = pp

    view['K_from_pts3d'] = K
    view['pts3d_from_depth&K'] = depthmap_to_pts3d(pts3d[:,:,2], K)
    return K


def set_pose_scale(views, tracks, i, j, scale, P_cw):
    # <i> is done, but not <j>
    sc_j, cam_j2i = estimate_relpose(views, tracks, i, j)

    # P_i == cam_j2i @ (sc_j * P_j)
    # P_j == cam_k2j @ (sc_k * P_k)
    # so P_i == cam_j2i @ (sc_j * cam_k2j @ (sc_k * P_k))
    scale[j] = sc_j * scale[i] # propagate scale
    try:
        # copy ground-truth if available
        cam2w = views[j]['cam2w']
        P_cw[j] = inv(cam2w)
    except KeyError:
        cam_j2i[:3,3] *= scale[i] # also propagate scale to translation
        P_cw[j] = inv(cam_j2i) @ P_cw[i] # cam_j <- cam_i <- world


def estimate_relpose(views, tracks, i, j):
    # first, get 2d matches
    device = views[i]['pts3d'].device
    sl_i, sl_j = tracks.img_slices[i], tracks.img_slices[j]
    _, idxs_i, idxs_j = np.intersect1d(tracks.pids[sl_i], tracks.pids[sl_j], return_indices=True)

    # then their corresponding 3d matches

    pix2d = todevice(tracks.pix2d, device)
    pix_i = pix2d[sl_i][idxs_i]
    pix_j = pix2d[sl_j][idxs_j]
    pts3d_i = bilinear_sampling(views[i]['pts3d'], pix_i)
    pts3d_j = bilinear_sampling(views[j]['pts3d'], pix_j)

    R, t, sj = roma.rigid_points_registration(pts3d_j, pts3d_i, compute_scaling=True)
    t = t.nan_to_num()
    sj = sj.nan_to_num(1) # security when there is no valid transformation
    # so we have s*R @ Pj + t == Pi
    #        <=> R @ (sj*Pj) + t == Pi
    #        <=> [R,t] @ (sj*Pj) == Pi
    cam_j2i = torch.eye(4, device=device)
    cam_j2i[:3, :3] = R
    cam_j2i[:3,  3] = t

    assert sj.isfinite() and cam_j2i.isfinite().all()
    return sj, cam_j2i


def combine_depth(multi_depth, Z, mode='log_multidepth'):
    if mode == 'linear_multidepth': # linear
        if Z.ndim == 0: # just a global scaling, so and addition for the last map==1 only
            depth = multi_depth[...,0] * Z
        else: # linear combination
            depth = multi_depth[...,:len(Z)] @ Z

    elif mode == 'log_multidepth':
        if Z.ndim == 0: # just a global scaling
            depth = Z * exp(multi_depth[..., 0].clip(max=80))
        else: # linear combination
            depth = exp((multi_depth[...,:len(Z)] @ Z).clip(max=80))

    else:
        raise ValueError(f'bad {mode=}')
    return depth


def update_views(views, rigs):
    assert isinstance(rigs, RigsData)
    mdkey ,= [k for k in views[0] if k in ('log_multidepth','linear_multidepth')]

    new_views = []
    for i, view in enumerate(views):
        try:
            # if they exist, we'll use them
            Z = rigs.z_coefs[i]
        except (TypeError, IndexError):
            # otherwise, just use the linear scale
            Z = rigs.scale[i]

        new_view = dict(view,
            K = rigs.get_K(i),
            cam2w = rigs.get_cam2w(i),
            depth = combine_depth(view[mdkey], Z, mode=mdkey),
        )
        new_view.pop(mdkey) # no need to keep it
        new_view.pop('pts3d', None) # remove it if it's there
        new_views.append(new_view)

    return new_views


def gather_pts3d(tracks, rigs, end_frame=None):
    assert isinstance(tracks, ObservedTracks)
    assert isinstance(rigs, RigsData)

    pts3d = zeros_like(tracks.pix2d_dep, (tracks.n_tracks,3))
    pts3d_accu = zeros_like(pts3d, (tracks.n_tracks,))

    for i in range(end_frame or tracks.n_imgs):
        try:
            # if they exist, we'll use them
            Z = rigs.z_coefs[i]
        except (TypeError, IndexError):
            # otherwise, just use the linear scale
            Z = rigs.scale[i]
            assert Z.ndim == 0

        # depth of each keypoint == 1st channel
        sel = tracks.img_slices[i]
        pixels = tracks.pix2d[sel].to(rigs.get_K(i).device)
        depths = combine_depth(tracks.pix2d_dep[sel,:], Z, mode=tracks.depth_mode)[..., None]
        xyz = sparse_depth_to_pts3d(pixels, depths, rigs.get_K(i), cam2world=rigs.get_cam2w(i))

        sel = tracks.img_slices[i]
        pts3d[tracks.pids[sel]] += xyz
        pts3d_accu[tracks.pids[sel]] += 1

    assert (pts3d_accu > 0).all()
    return pts3d / pts3d_accu[:,None]
