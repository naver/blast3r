# Copyright (C) 2026-present Naver Corporation. All rights reserved.

from abc import ABC, abstractmethod
from copy import copy as _copy
import torch

from blast3r.extensions.cujac import compute_jacobians_with_rigs, chol3x3_solve, blockcoo_matvec, fused_einsum_bki_bkj

# Column layout of the camera parameter block, mirroring sfm_jacobian_rigs.cu:
# the focal pair precedes tau in every Jacobian row, so neither can change here alone.
N_FOC = 2 # number of variables for the focal (fx, fy)
N_TAU = 6 # number of variables for the pose (3 for rot, 3 for transl)


class Param:
    def __init__(self, n):
        self.n = self.true_n = n
        self.true_idxs = None

class GaussNewtonSolver:
    """Bundle-adjustment solver for a problem with existing depth estimates."""
    def __init__(self, n_cams, n_nodes, n_imgs, n_tracks, device, no_grad_f_from_rz=False, dampen=1e-12, adaptive_dampen=True, dtype=torch.float32):
        self.cams = Param(n_cams)
        self.nodes = Param(n_nodes)
        self.imgs = Param(n_imgs)
        self.tracks = Param(n_tracks)
        self.device = device
        self.dtype = dtype
        self.dampen = dampen
        self.adaptive_dampen = adaptive_dampen
        self.no_grad_f_from_rz = no_grad_f_from_rz
        self._solver = self._solve_CG

    def compute_jacobians(self, rigs, K, P_rig2cam, P_w2rig, z_coefs, pts3d, pids, pix2d_std, pix2d_dep, is_depth_log,
                          optim_Z_1st_cam=True, optim_Z0=True, **kw):
        J_cam, J_zcf, J_pts = compute_jacobians_with_rigs(rigs, K, P_rig2cam, P_w2rig, z_coefs, pts3d, pids, pix2d_std, pix2d_dep, is_depth_log,
                                                          float64=(self.dtype == torch.float64))
        if not(optim_Z_1st_cam): J_zcf[0,0,0] = 0
        if not(optim_Z0):J_zcf[:,0,0:1] = 0
        return self.init_from_packed(rigs, pids, J_cam, J_zcf, J_pts, **kw)

    def init_from_packed(self, rigs, pids, J_cam, J_zcf, J_pts, optim_K=True, optim_Z=True, optim_P=True, optim_X=True):
        assert rigs.shape == (3, self.imgs.n)
        cum_kpts, node_ids, cam_ids = rigs.long()
        self.n_kpts, ONE, self.n_zcf = J_zcf.shape
        assert cum_kpts[-1].item() == self.n_kpts
        assert J_cam.shape == (self.n_kpts, 3, N_FOC+N_TAU)
        assert J_zcf.shape == (self.n_kpts, 1, self.n_zcf)
        assert J_pts.shape == (self.n_kpts, 3, 3)

        self.kpt2img = torch.bincount(cum_kpts)[:-1].cumsum(dim=0) # img_id <- kpt_id

        J_foc = OneBlockPerRow(J_cam[:,:,:2], cam_ids[self.kpt2img], shape=(self.n_kpts, self.cams.n))
        J_tau = OneBlockPerRow(J_cam[:,:,2:], node_ids[self.kpt2img], shape=(self.n_kpts, self.nodes.n))
        J_zcf = OneBlockPerRow(J_zcf, self.kpt2img, shape=(self.n_kpts, self.imgs.n), stride=3, offset=2)
        J_pts = OneBlockPerRow(J_pts, pids.long(), shape=(self.n_kpts, self.tracks.n))

        # fiz J_foc: makes it independant of focal on residual_z
        if self.no_grad_f_from_rz:
            J_foc._blocks[:,2].zero_()

        # enable or disable variables
        J_foc = self.enable(optim_K, J_foc, self.cams, cam_ids)
        J_tau = self.enable(optim_P, J_tau, self.nodes, node_ids)
        J_zcf = self.enable(optim_Z, J_zcf, self.imgs)
        J_pts = self.enable(optim_X, J_pts, self.tracks)

        J_cam = HStackMat(J_foc, J_tau)
        J = HStackMat(HStackMat(J_cam, J_zcf), J_pts)
        return J

    def enable(self, status, jac, count, mapping=None):
        if status is True:
            return jac
        elif status is False:
            return ZerosMat(jac._blocks, jac.block_shape, jac.shape_blocks)
        elif isinstance(status, torch.Tensor):
            assert isinstance(jac, OneBlockPerRow)
            assert status.dtype == torch.bool
            assert status.shape == (count.n,)
            assert len(jac._blocks) == len(self.kpt2img)
            if mapping is None:
                mapping = torch.arange(self.imgs.n, device=self.device)
            assert len(mapping) == self.imgs.n

            # first, remove some columns
            new_mapping = torch.zeros_like(status, dtype=jac._col_indices.dtype)
            idxs = status.nonzero().squeeze(-1)
            new_mapping[idxs] = torch.arange(len(idxs), dtype=new_mapping.dtype, device=new_mapping.device)

            if len(idxs) == 0: # ALL columns have been removed
                return ZerosMat(jac._blocks, jac.block_shape, jac.shape_blocks)

            # then, erase jacobians for disabled images
            # status: bool <- node_id
            # mapping: node_id <- img_id
            # kpt2img: img_id <- kpt_id
            jac._blocks *= status[mapping[self.kpt2img]].view(-1,1,1)
            # mat._col_indices: old_node_idx <- kpt_idx
            # new_mapping: new_node_idx <- old_node_idx
            jac._col_indices = new_mapping[jac._col_indices]
            jac._shape_blocks = (jac._shape_blocks[0], len(idxs))
            count.n = len(idxs)
            count.true_idxs = idxs

            return jac
        else:
            raise ValueError(f'bad {status=}')

    def solve_delta(self, J, residuals, w_pts, clamp_w=(1e-6, 1e6), z_prior=None):
        """Solve for delta = - (Jᵀ W J)⁻¹ Jᵀ W r, i.e. (Jᵀ W J) delta = - Jᵀ W r.

        Using the Schur complement:
        - knowing that (Jᵀ W J) = [A B; C D], with B == C.T
        - the Schur complement is S = A - B @ D_inv @ B.T
        - so the small system is S @ delta_c = -g_c + B @ D_inv @ g_p,
          with [g_c g_p] = J.T W r
        """
        assert J.shape == (3*self.n_kpts, self.cams.n * N_FOC + self.nodes.n * N_TAU + self.imgs.n * self.n_zcf + self.tracks.n * 3)
        assert residuals.shape == (self.n_kpts, 3)
        assert w_pts.shape in ((self.n_kpts, 1), (self.n_kpts, 3))

        W = w_pts.expand(-1,3)
        W = W.clamp(min=clamp_w[0], max=clamp_w[1])
        W = W.ravel()

        # compute the jacobian x weighted residuals
        JTr = (W * residuals.ravel()).to(self.dtype) @ J
        JTr = JTr.to_dense().ravel() # back to dense

        # compute Jᵀ W J
        J *= torch.sqrt(W).unsqueeze(-1)
        JTJ = J.gram_matrix()
        del J # now useless

        # optional dense tangent prior contributions, on z_coefs only.
        # z_prior is a tuple (H_z, g_z):
        #   H_z: (n_imgs, n_zcf, n_zcf) block-diagonal GN Hessian addition
        #   g_z: (n_imgs, n_zcf)        J^T r addition
        # both are added to the normal equations before damping.
        if z_prior is not None:
            H_z, g_z = z_prior
            if (H_z is not None) and (g_z is not None):
                H_z = H_z.to(self.dtype)
                g_z = g_z.to(self.dtype)


                # Add to gradient (J^T W r) in the z block.
                start_z = self.cams.n * N_FOC + self.nodes.n * N_TAU
                end_z = start_z + self.imgs.n * self.n_zcf
                assert g_z.shape == (self.imgs.n, self.n_zcf), f"bad g_z shape {g_z.shape}, expected {(self.imgs.n, self.n_zcf)}"
                assert H_z.shape == (self.imgs.n, self.n_zcf, self.n_zcf), f"bad H_z shape {H_z.shape}, expected {(self.imgs.n, self.n_zcf, self.n_zcf)}"

                JTr[start_z:end_z] += g_z.reshape(-1)

                # Add to Hessian (J^T W J) in the z-z block.
                # The non-point block is JTJ.A. When z is enabled, JTJ.A is expected
                # to be a SchurMatrix whose D block corresponds to z-z.
                if isinstance(JTJ.A, SchurMatrix):
                    assert not JTJ.transposed and not JTJ._A.transposed
                    Z_block = JTJ._A._D
                    if isinstance(Z_block, BlockDiagMat):
                        Z_block._blocks += H_z
                    elif isinstance(Z_block, ZerosMat):
                        JTJ._A._D = BlockDiagMat(H_z)
                    else:
                        # If this triggers, we need to extend the adder for this matrix type.
                        raise TypeError(f"Unsupported z-z block type for z_prior: {type(Z_block)}")
                else:
                    # If optim_Z was enabled, JTJ.A should generally be a SchurMatrix.
                    # If it isn't (edge-case), we skip the Hessian injection to avoid
                    # silently corrupting shapes.
                    raise TypeError(f"Unsupported JTJ.A type for z_prior: {type(JTJ.A)}")


        # dampen matrices to stabilize solutions (Levenberg-Marquardt)
        JTJ.dampen_(self.dampen, times_max_diag=self.adaptive_dampen)

        # solve for the delta with Schur complement
        delta = self._solver(JTJ, JTr)

        return delta.to(torch.float32)

    def _solve_CG(self, JTJ, JTr):
        g_c, g_p = JTr.tensor_split((JTJ.A.shape[0],))

        # compute the reduced right-hand side = g_c - B.T @ D.I @ g_p
        rhs = g_c - (JTJ.B @ JTJ.D.solve(g_p)).to_dense().ravel() # this is fast

        # solve the reduced system for Δc (CG with operator)
        diagA = JTJ.A.diag()
        diagA = diagA.clamp(min=1e-12)
        def M_prec(v): # this is fast
            return v / diagA

        # matvec for S = A - B D^{-1} C  (no explicit S)
        def S_matvec(v):
            Cv = JTJ.C @ v   # -> point space (size 3*P)
            DinvCv = JTJ.D.solve(Cv.ravel())  # solve with D, now fast
            return (JTJ.A @ v).ravel() - (JTJ.B @ DinvCv).ravel()

        delta_c, niter, terminated = _pcg(S_matvec, rhs, M_mv=M_prec, tol=1e-6, maxiter=300)
        delta_p = JTJ.D.solve(g_p - (JTJ.C @ delta_c).ravel()) # this is fast

        # combine into the full update vector
        return torch.cat([delta_c, delta_p])

    def _solve_dense_S(self, JTJ, JTr):
        g_c, g_p = JTr.tensor_split((JTJ.A.shape[0],))

        # form the Schur complement
        D_inv = JTJ.D.I
        B_D_inv = JTJ.B @ D_inv
        S = JTJ.A - B_D_inv @ JTJ.C   # S contains all parameters except the 3d tracks.

        # compute the reduced right-hand side
        rhs = g_c - (B_D_inv @ g_p).ravel()

        # solve for the camera updates
        delta_c = torch.linalg.solve(S.to_dense(), rhs)

        # recover the 3D point updates
        delta_p = D_inv @ (g_p.view(-1,1) - JTJ.C @ delta_c)

        # combine into the full update vector
        return delta_c, delta_p.ravel()

    def unpack_delta_inplace(self, delta):
        sizes = [self.cams.n * N_FOC, self.nodes.n * N_TAU, self.imgs.n * self.n_zcf, self.tracks.n * 3]
        cum_sizes = torch.tensor(sizes, dtype=torch.int).cumsum(dim=0)
        assert delta.shape == (cum_sizes[-1],)

        dK, dP, dZ, dX = delta.tensor_split(cum_sizes[:-1])

        def expand(deriv, param, np):
            deriv = deriv.view(param.n, np)
            if param.true_idxs is not None:
                deriv2 = deriv.new_zeros((param.true_n, np))
                deriv2[param.true_idxs] = deriv
                return deriv2
            else:
                return deriv

        dK = expand(dK, self.cams, N_FOC)
        dP = expand(dP, self.nodes, N_TAU)
        dZ = expand(dZ, self.imgs, self.n_zcf)
        dX = expand(dX, self.tracks, 3)
        return dK, dP, dZ, dX


class SparseMat (ABC):
    """Sparse block matrix stored in canonical (non-transposed) orientation.

    - transposing is just a view (a flag flip); storage is not reordered
    - _T_NAMES are the two class names to use whether transposed or not
    """
    def __init__(self, blocks, transposed: bool = False):
        assert isinstance(blocks, (torch.Tensor, SparseMat))
        self._blocks = blocks # just here for the device and dtype
        self._transposed = bool(transposed)

    # canonical helpers (never reflect .transposed)
    @property
    @abstractmethod
    def _dense_shape_base(self):
        ...

    # subclasses implement this in *canonical* orientation
    @abstractmethod
    def _to_dense_base(self) -> torch.Tensor:
        ...

    def to_dense(self) -> torch.Tensor:
        # unified densify: build in base, transpose if view says so
        base = self._to_dense_base()
        return base.T if self.transposed else base

    def ravel(self):
        res = self.to_dense()
        assert 1 in res.shape, 'ravel() is only possible with vectors'
        return res.ravel()

    # public, transposed-aware views
    @property
    def shape(self):
        H, W = self._dense_shape_base
        return (W, H) if self.transposed else (H, W)

    @property
    def dtype(self):
        return self._blocks.dtype

    @property
    def device(self):
        return self._blocks.device

    @property
    @abstractmethod
    def nnz(self):
        ...

    @property
    def infos(self):
        return ""

    def __repr__(self):
        h, w = self.shape
        nnz = self.nnz
        ratio = nnz / (h * w) if h * w else 0.0
        return f"{self._T_NAMES[self.transposed]}(shape=({h},{w}) with nnz={nnz} ({100*ratio:g}%){self.infos})"

    @property
    def transposed(self) -> bool:
        return self._transposed

    @property
    def T(self):
        # lightweight view: shallow copy that flips the flag
        res = _copy(self)
        res._transposed = not self._transposed
        return res

    def transpose(self, yes=True):
        return self.T if yes else self

    @property
    def blocks(self):
        return self._blocks.transpose(-1,-2) if self._transposed else self._blocks

    def __matmul__(self, other):
        raise NotImplementedError()

    def __rmatmul__(self, other):
        raise NotImplementedError()

    def gram_matrix(self):
        raise NotImplementedError()

    def hstack(self, other, **kw):
        return HStackMat(self, other, **kw)


class DenseMat (SparseMat):
    """A matrix that is more efficiently stored as a single dense tensor."""
    _T_NAMES = ['DenseMat', 'TransposedDenseMat']

    def __init__(self, mat: torch.Tensor, transposed: bool = False):
        assert isinstance(mat, torch.Tensor)
        super().__init__(mat, transposed=transposed)

    @property
    def _dense_shape_base(self):
        return self._blocks.shape

    def _to_dense_base(self) -> torch.Tensor:
        return self._blocks

    @property
    def nnz(self):
        return int((self._blocks != 0).sum())

    def __add__(self, other):
        if isinstance(other, DenseMat):
            A, B = self.blocks, other.blocks
            assert A.shape == B.shape
            return DenseMat(A + B)
        else:
            return other.__radd__(self)

    def __rsub__(self, left):
        if isinstance(left, torch.Tensor):
            left = DenseMat(left.view(-1,left.shape[-1]))
        assert left.shape == self.shape
        return DenseMat(left.to_dense() - self.blocks)

    def __matmul__(self, right):
        if isinstance(right, torch.Tensor):
            right = DenseMat(right.view(right.shape[0], -1))
        if isinstance(right, DenseMat):
            return self.__matmul__dense(right)
        return right.__rmatmul__(self)

    def __matmul__dense(self,right):
        return DenseMat(self.blocks @ right.blocks)

    def __rmatmul__(self, left):
        if isinstance(left, torch.Tensor):
            return DenseMat(left.view(-1,left.shape[-1])) @ self
        raise NotImplementedError(f'{left} @ {self} is not yet implemented')

    def hstack(self, other, **kw):
        if isinstance(other, DenseMat):
            return DenseMat(torch.cat((self.blocks, other.blocks), dim=-1), **kw)
        else:
            return super().hstack(other, **kw)


class SparseBlockMat (SparseMat):
    """Block-sparse matrix (abstract class).

    - `blocks`: (N, r, c), one 2D block per *block row index*, or structure-dependent
    - `shape`: (HB, WB), the matrix size in blocks, not in scalar entries
    """
    def __init__(self, blocks: torch.Tensor, shape, transposed: bool = False, stride=1, offset=0):
        super().__init__(blocks, transposed=transposed)
        assert isinstance(blocks, torch.Tensor) and blocks.ndim == 3, "`blocks` must be (N, r, c)"
        assert isinstance(shape, (tuple, list)) and len(shape) == 2
        assert all(isinstance(s, int) and s >= 0 for s in shape), "`shape` must be non-negative ints"
        self._shape_blocks = tuple(shape)  # (HB, WB) in blocks

        assert isinstance(stride, int) and stride >= 1
        assert isinstance(offset, int)
        assert 0 <= offset < stride
        self._stride = stride
        self._offset = offset

    # canonical helpers (never reflect .transposed)
    @property
    def _block_shape_base(self):
        r, c = self._blocks.shape[1:]
        return r * self._stride, c

    @property
    def _dense_shape_base(self):
        HB, WB = self._shape_blocks
        r, c = self._block_shape_base
        return HB * r, WB * c

    @property
    def shape_blocks(self):
        return self._shape_blocks[::-1 if self.transposed else 1]

    @property
    def block_shape(self):
        return self._block_shape_base[::-1 if self.transposed else 1]

    @property
    def nnz(self):
        return int(self._blocks.numel())


class ZerosMat (SparseBlockMat):
    """Fully sparse (empty) matrix.

    A block size is still needed, in case the matrix is dampened and converted
    into a BlockDiagSparse().
    """
    _T_NAMES = ['ZerosMat', 'ZerosMat']

    def __init__(self, mat, block_shape, shape, **kw):
        SparseMat.__init__(self, mat, **kw)
        assert len(block_shape) == len(shape) == 2
        self._block_shape_base_ = tuple(block_shape)
        self._shape_blocks = tuple(shape)
        self._stride = 1
        self._offset = 0

    @property
    def _block_shape_base(self):
        return self._block_shape_base_

    def _to_dense_base(self) -> torch.Tensor:
        return torch.zeros(self._dense_shape_base, dtype=self.dtype, device=self.device)

    @property
    def nnz(self):
        return 0

    def __add__(self, right):
        assert self.shape == right.shape
        return right # nothing to do

    def __radd__(self, left):
        assert self.shape == left.shape
        return left # nothing to do

    def __sub__(self, right):
        if isinstance(right, ZerosMat):
            return self
        raise NotImplementedError()

    def __rsub__(self, left):
        assert self.shape == left.shape
        return left # nothing to do

    def __imul__(self, other):
        return self # nothing to do

    def __matmul__(self, right):
        assert self.dtype == right.dtype
        assert self.device == right.device
        assert self.shape[1] == right.shape[0]
        if isinstance(right, torch.Tensor):
            right = right.view(right.shape[0], -1)
        return ZerosMat(self._blocks, (1,1), (self.shape[0], right.shape[1]))

    def __rmatmul__(self, left):
        assert left.dtype == self.dtype
        assert left.device == self.device
        if isinstance(left, torch.Tensor):
            left = left.view(-1, left.shape[-1])
        assert left.shape[-1] == self.shape[0]
        return ZerosMat(self._blocks, (1,1), (left.shape[0], self.shape[1]))

    def gram_matrix(self):
        HB, WB = self.shape_blocks
        r, c = self.block_shape
        return ZerosMat(self._blocks, (c,c), (WB, WB))

    def dampen_(self, val, times_max_diag=False):
        assert val > 0
        HB, WB = self.shape_blocks
        r, c = self.block_shape
        assert r == c and HB == WB
        eye = val * torch.eye(r, dtype=self.dtype, device=self.device)
        eye = eye.view(1,r,r).expand(HB,r,r).contiguous()
        return BlockDiagMat(eye)


class BlockDiagMat(SparseBlockMat):
    """Block-diagonal sparse matrix.

    - one block per block row, placed on the diagonal
    - `shape` is the (HB, WB) block grid; block-diagonal requires
      HB == WB == len(blocks)
    """
    _T_NAMES = ['BlockDiagMat', 'BlockDiagMat']

    def __init__(self, blocks: torch.Tensor, **kw):
        N = len(blocks)
        super().__init__(blocks, shape=(N, N), **kw)

    def _to_dense_base(self) -> torch.Tensor:
        HB, WB = self._shape_blocks
        r, c = self._block_shape_base
        out = torch.zeros((HB, self._stride, r//self._stride, WB, c), dtype=self.dtype, device=self.device)
        for i in range(len(self._blocks)):
            out[i, self._offset, :, i, :] = self._blocks[i]
        return out.view(HB*r, WB*c)

    @property
    def I(self): # inverse
        bh, bw = self._block_shape_base
        assert bh == bw and self._stride == 1
        res = _copy(self)
        res._blocks = torch.linalg.inv(res._blocks) # batched inv
        return res

    def __matmul__(self, right):
        if isinstance(right, torch.Tensor):
            right = DenseMat(right.view(right.shape[0], -1))

        if isinstance(right, HStackMat):
            return right.__rmatmul__(self)

        elif isinstance(right, DenseMat):
            bh1, bw1 = self.block_shape
            H2, W2 = right.shape
            assert H2 == len(self._blocks) * bh1 and self._stride == 1
            # (N, bh1, bw1) @ (N, bw1, W2) --> (N, bh1, W2) == (N*bh1, W2)
            rows = torch.einsum('bij,bjk->bik', self.blocks, right.blocks.view(-1, bw1, W2))
            return DenseMat(rows.view(-1, W2)) # re-assemble

        elif isinstance(right, BlockCOOMat):
            return right.__rmatmul__(self)

        elif isinstance(right, ZerosMat):
            return right.__rmatmul__(self)

        raise NotImplementedError()

    def max_diag(self):
        return self._blocks.diagonal(dim1=-1, dim2=-2).abs().max()

    def dampen_(self, val, times_max_diag=False):
        assert val > 0
        bh, bw = self._block_shape_base
        assert bh == bw and self._stride == 1
        if times_max_diag:
            val *= self.max_diag()
        self._blocks += val * torch.eye(bh, dtype=self.dtype, device=self.device).unsqueeze(0)
        return self

    def diag(self):
        bh, bw = self._block_shape_base
        assert bh == bw, 'ony works with square blocks'
        return self._blocks.diagonal(dim1=-1, dim2=-2).ravel()

    def solve(self, rhs_vec, jitter = 1e-12):
        """Solve D y = rhs_vec by batched Cholesky, assuming blocks of (n_points, 3, 3).

        Args:
            rhs_vec: (n_points*3,) dense.

        Returns:
            y of the same shape, such that D y = rhs_vec.
        """
        rhs = rhs_vec.view(len(self.blocks), 3)
        return chol3x3_solve(self.blocks, rhs, jitter).ravel()


class BlockCOOMat(SparseBlockMat):
    """Block COO matrix: a series of blocks at given coordinates."""
    _T_NAMES = ['BlockCOOMat', 'BlockCOOMat']

    def __init__(self, blocks: torch.Tensor, rows, cols, shape, **kw):
        super().__init__(blocks, shape=shape, **kw)
        assert isinstance(rows, torch.Tensor) and rows.ndim == 1
        assert isinstance(cols, torch.Tensor) and cols.ndim == 1
        assert rows.device == cols.device == blocks.device
        assert rows.dtype == torch.long, f'rows must be int64, got {rows.dtype}'
        assert cols.dtype == torch.long, f'cols must be int64, got {cols.dtype}'
        self._row_indices = rows
        self._col_indices = cols

    def _to_dense_base(self) -> torch.Tensor:
        HB, WB = self._shape_blocks
        r, c = self._block_shape_base
        out = torch.zeros((HB, self._stride, r//self._stride, WB, c), dtype=self.dtype, device=self.device)
        for n in range(len(self._blocks)):
            i = self._row_indices[n]
            j = self._col_indices[n]
            out[i, self._offset, :, j, :] = self._blocks[n]
        return out.view(HB*r, WB*c)

    @property
    def row_indices(self):
        return self._col_indices if self.transposed else self._row_indices

    @property
    def col_indices(self):
        return self._row_indices if self.transposed else self._col_indices

    def __rsub__(self, left):
        if isinstance(left, BlockDiagMat):
            HB1, WB1 = left.shape_blocks
            bh1, bw1 = left.block_shape
            HB2, WB2 = self.shape_blocks
            bh2, bw2 = self.block_shape
            assert (bh1, bw1) == (bh2, bw2), 'incompatible block sizes'
            assert (HB1, WB1) == (HB2, WB2), 'incompatible sizes'

            # make sure all the diagonal blocks are already included
            flat_idx = self.row_indices * HB1 + self.col_indices
            diag_idx = torch.arange(0, HB1*WB1, WB1+1, device=flat_idx.device)
            diag_pos = torch.searchsorted(flat_idx, diag_idx)
            assert (flat_idx[diag_pos] == diag_idx).all(), 'COO matrix is missing some diagonal blocks'

            out2 = -self.blocks.clone()
            out2[diag_pos] += left.blocks
            return BlockCOOMat(out2, self.row_indices, self.col_indices, shape=(HB1, WB1))

        raise NotImplementedError()

    def gram_matrix(self, other):
        raise NotImplementedError()

    def __matmul__(self, right):
        assert self._stride == 1
        if isinstance(right, torch.Tensor):
            right = DenseMat(right.view(right.shape[0], -1))
        if isinstance(right, DenseMat):
            return self.__matmul__dense(right)

        if isinstance(right, BlockCOOMat):
            return self.__matmul__bcoo(right)

        return right.__rmatmul__(self)

    def __matmul__dense(self, right):
        H2, W2 = right.shape
        if W2 == 1:
            out2 = blockcoo_matvec(self._blocks, self._row_indices, self._col_indices, right._blocks, *self._shape_blocks, self._transposed)
        else:
            HB1, WB1 = self.shape_blocks
            bh1, bw1 = self.block_shape
            # break it down into a lot of small blockwise matmul
            assert WB1 * bw1 == H2, 'incompatible sizes'
            # select corresponding blocks in dense matrix
            right_blocks = right.blocks.view(WB1, bw1, W2)
            right_blocks = right_blocks.index_select(0, self.col_indices)
            # (N, bh1, bw1) @ (N, bw1, W2) --> (N, bh1, W2)
            res_blocks = torch.einsum('bij,bjk->bik', self.blocks, right_blocks)
            # aggregation
            out2 = torch.zeros((HB1, bh1, W2), dtype=self.dtype, device=self.device)
            out2.index_add_(0, self.row_indices, res_blocks)      # sum blocks hitting same place
        return DenseMat(out2.view(-1, W2))

    def __matmul__bcoo(self, right):
        HB1, WB1 = self.shape_blocks
        bh1, bw1 = self.block_shape
        HB2, WB2 = right.shape_blocks
        bh2, bw2 = right.block_shape
        assert bw1 == bh2, 'incompatible block sizes'
        assert WB1 * bw1 == HB2 * bh2, 'incompatible sizes'

        # find matches
        def sort_indices(row_indices, n_rows):
            vals, idxs = row_indices.sort()
            # find delimitations
            boundaries = torch.searchsorted(vals, torch.arange(n_rows+1, device=vals.device))
            return boundaries, idxs

        indptr1, idxs1 = sort_indices(self.col_indices, WB1)
        indptr2, idxs2 = sort_indices(right.row_indices, HB2)
        i1, i2 = torch.cat([torch.cartesian_prod(idxs1[indptr1[i]:indptr1[i+1]], idxs2[indptr2[i]:indptr2[i+1]]) for i in range(WB1)]).T

        # bmm between matching blocks
        A = self.blocks .index_select(0, i1)   # (M, r, k)
        B = right.blocks.index_select(0, i2)   # (M, k, c)
        C = torch.einsum('bij,bjk->bik', A, B)

        # now combine duplicates (same row, col)
        out_rows = self.row_indices .index_select(0, i1)
        out_cols = right.col_indices.index_select(0, i2)
        flat_idx = out_rows * WB2 + out_cols

        # unique and inverse index mapping
        uniq, inv = torch.unique(flat_idx, return_inverse=True)
        Nc = uniq.numel()

        if Nc >= (HB1 * WB2) // 2: # should we output a dense tensor?
            out2 = torch.zeros((HB1 * WB2, bh1, bw2), device=C.device, dtype=C.dtype)
            out2.index_add_(0, flat_idx, C)
            out2 = out2.view(HB1, WB2, bh1, bw2).transpose(1,2).reshape(HB1*bh1, WB2*bw2)
            return DenseMat(out2)
        else:
            out2 = torch.zeros((Nc, bh1, bw2), device=C.device, dtype=C.dtype)
            out2.index_add_(0, inv, C)  # accumulate duplicates
            return BlockCOOMat(out2, uniq // WB2, uniq % WB2, (HB1, WB2))

    def __rmatmul__(self, left):
        assert self._stride == 1
        if isinstance(left, BlockDiagMat):
            return self.__rmatmul__bdiag(left)

        if isinstance(left, DenseMat):
            return self.__rmatmul__dense(left)

        if isinstance(left, torch.Tensor):
            return self.__rmatmul__(DenseMat(left.view(1,-1) if left.ndim == 1 else left))

        raise NotImplementedError(f'{left} @ {self} is not yet implemented')

    def __rmatmul__bdiag(self, left):
        # break it down into a lot of small blockwise matmul
        HB1, WB1 = left.shape_blocks
        bh1, bw1 = left.block_shape
        HB2, WB2 = self.shape_blocks
        bh2, bw2 = self.block_shape
        assert bw1 == bh2, 'incompatible block sizes'
        assert WB1 * bw1 == HB2 * bh2, 'incompatible sizes'
        # select corresponding blocks in block-diag matrix
        left_blocks = left._blocks.index_select(0, self.row_indices)
        # (N, bh1, bw1) @ (N, bh2, bw2) --> (N, bh1, bw2)
        res_blocks = torch.einsum('bij,bjk->bik', left_blocks, self.blocks)
        # no aggregation because it's a block-diag matrix
        return BlockCOOMat(res_blocks, self.row_indices, self.col_indices, (HB1, WB2))

    def __rmatmul__dense(self, left):
        H1, W1 = left.shape
        if H1 == 1:
            out2 = blockcoo_matvec(self._blocks, self._row_indices, self._col_indices, left._blocks, *self._shape_blocks, not self._transposed)
        else:
            HB2, WB2 = self.shape_blocks
            bh2, bw2 = self.block_shape
            assert W1 == HB2 * bh2, 'incompatible sizes'
            # break it down into a lot of small blockwise matmul
            # select corresponding blocks in block-diag matrix
            left_blocks = left.blocks.view(H1, HB2, bh2).transpose(1, 0)
            left_blocks = left_blocks.index_select(0, self.row_indices)
            # (N, H1, bh2) @ (N, bh2, bw2) --> (N, H1, bw2)
            res_blocks = torch.einsum('bij,bjk->bik', left_blocks, self.blocks)
            # aggregation
            out2 = torch.zeros((WB2, H1, bw2), dtype=self.dtype, device=self.device)
            out2.index_add_(0, self.col_indices, res_blocks)      # sum blocks hitting same place
            out2 = out2.transpose(0,1)
        return DenseMat(out2.reshape(H1, -1))


class OneBlockPerRow(SparseBlockMat):
    """Block-sparse matrix with exactly one block per row.

    - exactly one nonzero block per *block-row* i, at block-column j = col_indices[i]
    - the transpose view naturally behaves like "one block per column"
    """
    _T_NAMES = ['OneBlockPerRow', 'OneBlockPerCol']

    def __init__(self, blocks: torch.Tensor, col_indices: torch.Tensor, shape, **kw):
        assert isinstance(blocks, torch.Tensor) and blocks.ndim == 3
        super().__init__(blocks, shape, **kw)
        assert isinstance(col_indices, torch.Tensor) and col_indices.ndim == 1
        assert len(blocks) == len(col_indices), 'there should be as many col_indices as blocks'
        HB, WB = self._shape_blocks
        assert len(self._blocks) == HB, "len(blocks) must equal number of block-rows (HB)"
        assert col_indices.device == blocks.device and col_indices.dtype == torch.long, \
            f'col_indices must be int64 on {blocks.device}, got {col_indices.dtype} on {col_indices.device}'
        self._col_indices = col_indices

    def _to_dense_base(self) -> torch.Tensor:
        # Always fill as if not transposed; the base class will transpose at the end if needed.
        HB, WB = self._shape_blocks
        r, c = self._block_shape_base
        out = torch.zeros((HB, self._stride, r//self._stride, WB, c), dtype=self.dtype, device=self.device)
        for i in range(len(self._blocks)):
            j = self._col_indices[i]
            out[i, self._offset, :, j, :] = self._blocks[i]
        return out.view(HB*r, WB*c)

    def __imul__(self, other):
        assert isinstance(other, torch.Tensor) and other.ndim <= 2
        if other.ndim == 2 and other.shape[1] == 1:
            # multiply each row
            if self.transposed:
                raise NotImplementedError()
            else:
                HB, WB = self._shape_blocks
                bh, bw = self._block_shape_base
                col = self._blocks.view(HB * bh // self._stride, bw)
                col *= other[self._offset::self._stride]
                return self
        else:
            raise NotImplementedError()

    def __matmul__(self, right):
        if isinstance(right, torch.Tensor):
            right = DenseMat(right.view(right.shape[0], -1))

        if self.transposed:
            if isinstance(right, OneBlockPerRow) and not right.transposed:
                return self.__matmul__obpc_obpr(right)
            elif isinstance(right, DenseMat):
                return (right.T @ self.T).T
            else:
                return right.__rmatmul__(self)
        else:
            if isinstance(right, DenseMat):
                return self.__matmul__obpr_dense(right)

        raise NotImplementedError(f'{self} @ {right} is not yet implemented')

    def __matmul__obpr_dense(self, right):
        # my grid and block sizes
        HB1, WB1 = self._shape_blocks
        bh1, bw1 = self._block_shape_base
        H2, W2 = right.shape
        assert WB1*bw1 == H2
        # we slice the dense (H2, W2) --> (H2//c1, c1, W2)
        right_blocks = right.blocks.view(WB1, bw1, W2)
        # then we have blockwise multiplication: (H1,r2) x (r2,c2) -> (H1,c2)
        per_row = torch.einsum('bij,bjk->bik', self._blocks, right_blocks.index_select(0, self._col_indices))
        return DenseMat(per_row.reshape(HB1 * bh1, W2))

    def __matmul__obpc_obpr(self, right):
        # my grid and block sizes
        HB1, WB1 = self._shape_blocks
        bh1, bw1 = self._block_shape_base
        # his grid and block sizes
        HB2, WB2 = right._shape_blocks
        bh2, bw2 = right._block_shape_base
        assert HB1 == HB2 and bh1 == bh2, 'block sizes are incompatible'

        self_blocks = self._blocks
        right_blocks = right._blocks
        if self._stride == right._stride == 1:
            pass # nothing to do
        elif self._stride == 1 and right._stride > 1:
            self_blocks = self._blocks.view(HB1, right._stride, bh1//right._stride, bw1)[:, right._offset]
        elif self._stride > 1 and right._stride == 1:
            right_blocks = right._blocks.view(HB2, self._stride, bh2//self._stride, bw2)[:, self._offset]
        else:
            raise NotImplementedError()

        # (HB, bw1, bh) @ (HB, bh, bw2) -> (HB, bw1, bw2)
        blocks = fused_einsum_bki_bkj(self_blocks, right_blocks)

        # let's find out which blocks are occupied: output matrix has size (WB1, WB2)
        flat_idx = self._col_indices * WB2 + right._col_indices
        uniq, inv = torch.unique(flat_idx, return_inverse=True)
        Nc = uniq.numel()

        if one_block_per_col(Nc, right._col_indices, WB2): # one block per column!
            out2 = torch.zeros((WB2, bw1, bw2), dtype=self.dtype, device=self.device)
            out2.index_add_(0, right._col_indices, blocks)      # sum blocks hitting same place
            row_idxs = torch.empty((WB2,), dtype=torch.long, device=out2.device)
            row_idxs.index_put_((right._col_indices,), self._col_indices)
            return OneBlockPerRow(out2.transpose(-1,-2), row_idxs, (WB2, WB1), transposed=True)

        if one_block_per_col(Nc, self._col_indices, WB1): # one block per row!
            out2 = torch.zeros((WB1, bw1, bw2), dtype=self.dtype, device=self.device)
            out2.index_add_(0, self._col_indices, blocks)      # sum blocks hitting same place
            col_idxs = torch.empty((WB1,), dtype=torch.long, device=out2.device)
            col_idxs.index_put_((self._col_indices,), right._col_indices)
            return OneBlockPerRow(out2, col_idxs, (WB1, WB2))

        if Nc > 0.5*WB1*WB2: # more than half of blocks are occupied ==> dense
            out2 = torch.zeros((WB1*WB2, bw1, bw2), dtype=self.dtype, device=self.device)
            out2.index_add_(0, flat_idx, blocks)      # sum blocks hitting same place
            dense_out2 = out2.view(WB1, WB2, bw1, bw2).permute(0,2,1,3).reshape(WB1*bw1, WB2*bw2)
            return DenseMat(dense_out2)

        # default choice = BlockCOO
        if Nc == flat_idx.numel():
            # no need to do any aggregation
            return BlockCOOMat(blocks, self._col_indices, right._col_indices, (WB1, WB2))
        else:
            out2 = torch.zeros((Nc, bw1, bw2), device=self.device, dtype=self.dtype)
            out2.index_add_(0, inv, blocks)  # accumulate duplicates
            return BlockCOOMat(out2, uniq // WB2, uniq % WB2, (WB1, WB2))

    def gram_matrix(self):
        HB, WB = self._shape_blocks                   # grid size in blocks
        bh, bw = self._block_shape_base                 # block size (rows, cols)

        if self.transposed:
            raise NotImplementedError()
        else:
            # (HB, bw, bh) @ (HB, bh, bw) --> (HB, bw, bw)
            blocks = fused_einsum_bki_bkj(self._blocks, self._blocks)

            # accumulator into block-diagonal matrix
            out2 = torch.zeros((WB, bw*bw), dtype=blocks.dtype, device=blocks.device)
            out2.index_add_(0, self._col_indices, blocks.view(HB, bw*bw))      # sum rows hitting same column
            return BlockDiagMat(out2.view(WB, bw, bw))

    def __rmatmul__(self, left):
        if isinstance(left, torch.Tensor):
            left = DenseMat(left.view(-1, left.shape[-1]))
        if isinstance(left, DenseMat):
            if self.transposed:
                return self.__rmatmul__dense_obpc(left)
            else:
                return self.__rmatmul__dense_obpr(left)
        else:
            raise NotImplementedError()

    def __rmatmul__dense_obpc(self, left):
        H1, W1 = left.shape
        HB2, WB2 = self.shape_blocks
        r2, c2 = self.block_shape
        assert W1 == HB2*r2, 'incompatible sizes'

        left_blocks = left.blocks.view(H1, W1//r2, r2)
        # then we have blockwise multiplication: (WB2,H1,r2) x (WB2,r2,c2) -> (WB2,H1,c2)
        per_row = torch.einsum('bij,bjk->bik', left_blocks.index_select(1, self._col_indices).transpose(0,1), self.blocks)
        if self._stride > 1:
            raise NotImplementedError()
        return DenseMat(per_row.reshape(H1, WB2 * c2))

    def __rmatmul__dense_obpr(self, left):
        H1, W1 = left.shape
        HB2, WB2 = self.shape_blocks
        r2, c2 = self.block_shape
        # we want to multiply (H1, W1) x (HB2*r2, WB2*c2) --> (H1, WB2*c2)
        assert W1 == HB2*r2, 'incompatible sizes'

        # we have blockwise multiplication: (H1,r2) x (r2,c2) -> (H1,c2)
        left_blocks = left.blocks.view(H1, W1//r2, self._stride, r2//self._stride)[:, :, self._offset]
        # einsum should be faster for small-size blocks (single fused kernel)
        per_row = torch.einsum('bij,bjk->bik', left_blocks.transpose(0,1), self.blocks)

        # accumulate into block-columns (WB, c)
        out2 = torch.zeros((WB2, H1, c2), dtype=self.dtype, device=self.device)
        out2.index_add_(0, self._col_indices, per_row)        # sum rows hitting same column
        return DenseMat(out2.view(H1, WB2 * c2))


def one_block_per_col(n_unique_cells, col_idxs, n_columns):
    # first, make sure that there's as many unique cells as number of columns
    if n_unique_cells == n_columns:
        # then, make sure that all columns are occupied
        uniq = torch.unique(col_idxs)
        return uniq.numel() == n_columns
    return False


class HStackMat (SparseMat):
    """Horizontally-adjacent stack of two matrices [A B].

    A and B must have the same number of rows.
    """
    _T_NAMES = ['HStackMat', 'VStackMat']

    def __init__(self, mat1, mat2, **kw):
        super().__init__(mat1, **kw)
        assert isinstance(mat1, SparseMat), f'mat1 must be a SparseMat, got {type(mat1).__name__}'
        assert isinstance(mat2, SparseMat), f'mat2 must be a SparseMat, got {type(mat2).__name__}'
        assert mat1.dtype == mat2.dtype
        assert mat1.device == mat2.device
        assert mat1.shape[0] == mat2.shape[0], f'incompatible sizes: {mat1.shape} vs {mat2.shape}'
        self._mat1 = mat1 # those are never tranposed
        self._mat2 = mat2

    @property
    def A(self):
        return self._mat1.transpose(self.transposed)

    @property
    def B(self):
        return self._mat2.transpose(self.transposed)

    @property
    def infos(self):
        h1,w1 = self.A.shape
        h2,w2 = self.B.shape
        return f" with blocks A={h1}x{w1} and B={h2}x{w2}"

    @property
    def _dense_shape_base(self):
        h1, w1 = self._mat1.shape
        h2, w2 = self._mat2.shape
        assert h1 == h2
        return (h1, w1+w2)

    @property
    def nnz(self):
        return self._mat1.nnz + self._mat2.nnz

    def _to_dense_base(self):
        h1, w1 = self._mat1.shape
        res = torch.zeros(self._dense_shape_base, dtype=self.dtype, device=self.device)
        if not isinstance(self._mat1, ZerosMat): res[:, :w1] = self._mat1.to_dense()
        if not isinstance(self._mat2, ZerosMat): res[:, w1:] = self._mat2.to_dense()
        return res

    def __sub__(self, right):
        assert self.shape == right.shape
        if isinstance(right, HStackMat) and self.transposed == right.transposed:
            return HStackMat(self._mat1 - right._mat1, self._mat2 - right._mat2, transposed=self.transposed)
        else:
            return right.__rsub__(self)

    def __imul__(self, other):
        assert isinstance(other, torch.Tensor) and other.ndim <= 2
        if other.ndim == 2 and other.shape[1] == 1:
            # multiply each row
            if self.transposed:
                raise NotImplementedError()
            else:
                self._mat1 *= other
                self._mat2 *= other
                return self
        else:
            raise NotImplementedError()

    def __matmul__(self, right):
        if isinstance(right, torch.Tensor):
            right = DenseMat(right.view(right.shape[0], -1))
        if self.transposed: # we are a vertical stack of two matrices
            right_T = right.T
            A_T = right_T @ self._mat1
            B_T = right_T @ self._mat2
            return A_T.hstack(B_T, transposed=True)

        elif isinstance(right, DenseMat):
            n = self._mat1.shape[1]
            A = self._mat1 @ DenseMat(right.blocks[:n])
            B = self._mat2 @ DenseMat(right.blocks[n:])
            return A + B
        else:
            raise NotImplementedError()

    def gram_matrix(self):
        """Compute self.T @ self."""
        if self.transposed:
            raise NotImplementedError()
        else:
            A = self._mat1.gram_matrix()
            B = self._mat1.T @ self._mat2
            D = self._mat2.gram_matrix()
            return SchurMatrix(A, B, B.T, D)

    def __rmatmul__(self, left):
        if self.transposed:
            raise NotImplementedError()
        else:
            A = left @ self._mat1
            B = left @ self._mat2
            return A.hstack(B)


class SchurMatrix (SparseMat):
    """Square Schur matrix of four blocks, [A B] over [C D]."""
    _T_NAMES = ['SchurMatrix', 'SchurMatrix']

    def __init__(self, A, B, C, D):
        super().__init__(A)
        assert isinstance(A, SparseMat)
        assert isinstance(B, SparseMat)
        assert isinstance(C, SparseMat)
        assert isinstance(D, SparseMat)
        HA, WA = A.shape
        HB, WB = B.shape
        HC, WC = C.shape
        HD, WD = D.shape
        assert HA == WA and HD == WD, 'matrices A and D should be square'
        assert HA == HB and WB == WD, 'incompatible matrices shapes'
        assert WA == WC and HC == HD, 'incompatible matrices shapes'
        self._A = A
        self._B = B
        self._C = C
        self._D = D

    @property
    def infos(self):
        n,n = self._A.shape
        m,m = self._D.shape
        return f" with blocks A={n}x{n} and D={m}x{m}"

    @property
    def _dense_shape_base(self):
        h1, w1 = self._A.shape
        h2, w2 = self._D.shape
        return (h1+h2, w1+w2)

    def _to_dense_base(self):
        h1, w1 = self._A.shape
        res = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        res[:h1, :w1] = self._A.to_dense()
        res[:h1, w1:] = self._B.to_dense()
        res[h1:, :w1] = self._C.to_dense()
        res[h1:, w1:] = self._D.to_dense()
        return res

    @property
    def A(self):
        return self._A.transpose(self.transposed)
    @property
    def B(self):
        return self._B.transpose(self.transposed)
    @property
    def C(self):
        return self._C.transpose(self.transposed)
    @property
    def D(self):
        return self._D.transpose(self.transposed)

    @property
    def nnz(self):
        return self._A.nnz + self._B.nnz + self._C.nnz + self._D.nnz

    def __sub__(self, right):
        if isinstance(right, DenseMat):
            return DenseMat(self.to_dense() - right.blocks)

        right = self.as_schur_matrix(right)
        return SchurMatrix( self._A - right._A,
                            self._B - right._B,
                            self._C - right._C,
                            self._D - right._D)

    def __matmul__(self, right):
        assert isinstance(right, torch.Tensor) and right.ndim in (1,2)
        n = self.A.shape[1]
        r1, r2 = right[:n], right[n:]

        A_r1 = (self.A @ r1).to_dense()
        C_r1 = (self.C @ r1).to_dense()
        B_r2 = (self.B @ r2).to_dense()
        D_r2 = (self.D @ r2).to_dense()
        return torch.cat((A_r1 + B_r2, C_r1 + D_r2))

    def as_schur_matrix(self, mat):
        if isinstance(mat, SchurMatrix):
            return mat
        if (isinstance(mat, HStackMat) and
            isinstance(mat._mat1, HStackMat) and
            isinstance(mat._mat2, HStackMat) and
            mat.transposed != mat.A.transposed and
            mat.transposed != mat.B.transposed):

            A, B = mat.A.A, mat.B.A
            C, D = mat.A.B, mat.B.B
            if mat.transposed:
                B, C = C, B
            return SchurMatrix(A, B, C, D)

        elif isinstance(mat, ZerosMat):
            A = ZerosMat(self._A, (1,1), self.A.shape)
            B = ZerosMat(self._B, (1,1), self.B.shape)
            C = ZerosMat(self._C, (1,1), self.C.shape)
            D = ZerosMat(self._D, (1,1), self.D.shape)
            return SchurMatrix(A, B, C, D)

        else:
            raise NotImplementedError()

    def dampen_(self, val, **kw):
        self._A = self._A.dampen_(val, **kw)
        self._D = self._D.dampen_(val, **kw)
        return self

    def diag(self):
        return torch.cat((self.A.diag(), self.D.diag()))


def _pcg(A_mv, b, M_mv=None, x0=None, tol=1e-6, maxiter=200):
    """Minimal preconditioned conjugate gradient.

    Args:
        A_mv: callable v -> A v, an SPD operator.
        b: right-hand side, a 1D tensor.
        M_mv: callable v -> M^{-1} v, a preconditioner, or None.
        x0: initial guess.

    Returns:
        (x, iters, converged).
    """
    dtype, device = b.dtype, b.device
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - A_mv(x)
    z = M_mv(r) if M_mv is not None else r
    p = z.clone()
    rz_old = torch.dot(r, z)

    best_so_far = float('inf'), x, 0

    bnorm = torch.norm(b).clamp(min=1e-12)
    for k in range(1, maxiter + 1):
        Ap = A_mv(p)
        denom = torch.dot(p, Ap).clamp(min=1e-32)
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        r_norm = r.norm()
        if r_norm < best_so_far[0]:
            best_so_far = r_norm, x, k
        if torch.norm(r) <= tol * bnorm:
            return x, k, True
        z = M_mv(r) if M_mv is not None else r
        rz_new = torch.dot(r, z)
        if not rz_new.isfinite():
            break
        beta = rz_new / rz_old.clamp(min=1e-32)
        p = z + beta * p
        rz_old = rz_new

    return best_so_far[1:] + (False,)
