// Copyright (C) 2026-present Naver Corporation. All rights reserved.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK_CUDA(var)         TORCH_CHECK(var.is_cuda(), #var " is not on CUDA device");
#define CHECK_CONT(var)         TORCH_CHECK(var.is_contiguous(), #var " is not memory-contiguous");
#define CHECK_NDIM(var,n)       TORCH_CHECK(var.dim() == n, #var ".ndim should be == ", n, ", not ", var.dim()); 
#define CHECK_LASTDIM(var,n)    TORCH_CHECK(var.size(var.dim()-1) == n, #var ".shape[-1] should be == ", n, ", not ", var.size(var.dim()-1)); 
#define CHECK_CUDA_AND_CONTIGUOUS(var, nd) {CHECK_CUDA(var) CHECK_CONT(var) CHECK_NDIM(var,nd)}
#define CHECK_CUDA_AND_CONTIGUOUS_SHAPE(var, nd, ld) {CHECK_CUDA_AND_CONTIGUOUS(var, nd) CHECK_LASTDIM(var, ld)}

// #define PRINT0(var)         if (idx == 0) _print(#var,&var,1,1);
// #define PRINT1(var,n)       if (idx == 0) _print(#var,var,1,n);
// #define PRINT2(var,n,m)     if (idx == 0) _print(#var,var,n,m);
#define PRINT0(var)         ; // do nothing
#define PRINT1(var,n)       ; // do nothing
#define PRINT2(var,n,m)     ; // do nothing

const int threads_per_block = 128;


template <typename T>
__device__ __forceinline__ T my_sqrt(T x) { return sqrt(x); }

template <>
__device__ __forceinline__ float my_sqrt<float>(float x) { return sqrtf(x); }

template <typename T>
__device__ __forceinline__ 
void atomicAddT(T* addr, T val) { atomicAdd(addr, val); }

template <>
__device__ __forceinline__ void atomicAddT<double>(double* addr, double val) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(addr, val);
#else
    // Fallback for very old archs
    unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
}

template <typename T>
__device__ 
void _print(const char* name, const T* var, const int n, const int m)
{
    printf("%s = [", name);
    for(int i = 0; i < n; i++) {
        int j;
        for(j = 0; j < m-1; j++) 
        {
            double v = var[i*m+j];
            printf("%g, ", v);
        }
        double v = var[i*m+j];
        printf("%g]", v);
        if (i < n-1) printf(",[");
    }
    printf("\n");
}

__device__ __forceinline__
int searchsorted(const int* __restrict__ infos, const int n_img, const int kpt_idx)
{
    int a = 0;
    int b = n_img;
    while (b - a > 1) 
    {
        int m = (a + b) / 2;
        int sum_nkpts = infos[m-1];

        if (kpt_idx < sum_nkpts)
            b = m;
        else
            a = m;
    }
    return a;
}

template <int M, int K, int N, typename Type>
__device__ __forceinline__
void matmul(const Type* __restrict__ A, 
            const Type* __restrict__ B, 
            Type* __restrict__ C,
            const Type* __restrict__ bias = NULL)
{
    #pragma unroll
    for (int row = 0; row < M; row++) {
        #pragma unroll
        for (int col = 0; col < N; col++) {
            Type sum = 0;

            #pragma unroll
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            if (bias)
                sum += bias[row];
            C[row * N + col] = sum;
        }
    }
}

// CUDA kernel to compute the residuals for bundle adjustment
// Inputs:
//   infos     : [3, n_imgs] each column = (node_id, cam_id, cum_n_kpts)
//   K         : [n_cams, 3, 3] camera intrinsics (row-major)
//   P_rig2c   : [n_cams, 4, 4] rig-to-cam poses (row-major)
//   P_w2rig   : [n_node, 4, 4] world-to-rig poses (row-major)
//   z_coefs   : [n_imgs, C] global image shape parameters (C comes from shape)
//   pts3d     : [n_pts, 3] 3D points
//   pids      : [n_obs,] for each observed keypoint this is the pts3d index
//   pix2d     : [n_obs, 2] per–observation keypoint pixels (x,y)
//   pix2d_std : [n_obs, 2] per–observation standard deviations
//   pix2d_dep : [n_obs, C] local point shape parameters
// Outputs:
//   residuals : [n_obs, 3] residuals on (x,y,z)
//
template <bool is_depth_log>
__global__
void compute_residuals_rigs_cuda_kernel(const int* __restrict__ infos,            // (n_imgs x 3)
                                        const float* __restrict__ K,             // (n_cams x 9)
                                        const float* __restrict__ P_rig2c,       // (n_cams x 16)
                                        const int64_t P_rig2c_stride,                // P_rig2c stride 
                                        const float* __restrict__ P_w2rig,       // (n_nodes x 16)
                                        const float* __restrict__ z_coefs,       // (n_imgs x C)
                                        const float* __restrict__ pts3d,         // (n_pts x 3)
                                        const int* __restrict__ pids,            // (n_obs x 1)
                                        const float* __restrict__ pix2d,         // (n_obs x 2)
                                        const float* __restrict__ pix2d_std,     // (n_obs x 2)
                                        const float* __restrict__ pix2d_dep,     // (n_obs x C)
                                        float* __restrict__ residuals,           // (n_obs x 3)
                                        int n_imgs, int n_obs, int C,
                                        float weight_z, 
                                        float pnorm,
                                        float max_pix_err, 
                                        float huber_delta,
                                        float w_pts_null
                                        )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    PRINT0(idx)
    if (idx >= n_obs)
        return;

    // ---------- Load observation indices ----------
    // PRINT2(&infos,3,n_imgs)
    int img_idx = searchsorted(infos, n_imgs, idx);
    int node_idx = infos[1*n_imgs + img_idx];
    int cam_idx  = infos[2*n_imgs + img_idx];
    int pt_idx   = pids[idx];
    PRINT0(img_idx)
    PRINT0(pt_idx)

    // ---------- Load 3D point (pts3d[j]) ----------
    float X_world[3];
    X_world[0] = pts3d[pt_idx * 3 + 0];
    X_world[1] = pts3d[pt_idx * 3 + 1];
    X_world[2] = pts3d[pt_idx * 3 + 2];
    PRINT1(X_world,3)

    // ---------- Load camera intrinsics K_i (3x3) ----------
    // Extract fx, cx, fy, cy (we assume K has the usual form)
    float fx = K[cam_idx * 9 + 0];
    float cx = K[cam_idx * 9 + 2];
    float fy = K[cam_idx * 9 + 4];
    float cy = K[cam_idx * 9 + 5];

    // ---------- Load camera pose P_w2rig (4x4) and extract R (3x3) and t (3-vector) ----------
    const float* __restrict__ P_rw = P_w2rig + node_idx * 16;
    // R is the top-left 3x3 of P_i (row-major)
    // t is the translation (first 3 entries of the fourth column)
    float R_rw[9], t_rw[3];
    R_rw[0] = P_rw[0];  R_rw[1] = P_rw[1];  R_rw[2] = P_rw[2];  t_rw[0] = P_rw[3];
    R_rw[3] = P_rw[4];  R_rw[4] = P_rw[5];  R_rw[5] = P_rw[6];  t_rw[1] = P_rw[7];
    R_rw[6] = P_rw[8];  R_rw[7] = P_rw[9];  R_rw[8] = P_rw[10]; t_rw[2] = P_rw[11];
    PRINT2(R_rw,3,3)
    PRINT1(t_rw,3)

    // ---------- Load camera pose P_rig2c (4x4) and extract R (3x3) and t (3-vector) ----------
    const float* __restrict__ P_cr = P_rig2c + cam_idx * P_rig2c_stride;
    // R is the top-left 3x3 of P_i (row-major)
    // t is the translation (first 3 entries of the fourth column)
    float R_cr[9], t_cr[3];
    R_cr[0] = P_cr[0];  R_cr[1] = P_cr[1];  R_cr[2] = P_cr[2];  t_cr[0] = P_cr[3];
    R_cr[3] = P_cr[4];  R_cr[4] = P_cr[5];  R_cr[5] = P_cr[6];  t_cr[1] = P_cr[7];
    R_cr[6] = P_cr[8];  R_cr[7] = P_cr[9];  R_cr[8] = P_cr[10]; t_cr[2] = P_cr[11];
    PRINT2(R_cr,3,3)
    PRINT1(t_cr,3)

    // ---------- Compute X_rig = R_rw * X_world + t_rw ----------
    float X_rig[3];
    matmul<3,3,1>(R_rw, X_world, X_rig, t_rw);
    PRINT1(X_rig,3)

    // ---------- Compute X_cam = R_cr * X_rig + t_cr ----------
    float X_cam[3];
    matmul<3,3,1>(R_cr, X_rig, X_cam, t_cr);
    PRINT1(X_cam,3)

    float z = X_cam[2];  // depth
    if (z <= 1e-3) return; // invalid point, no residual

    // ---------- Project into pixel coordinates: p = K_i * X_cam ----------
    // Here we assume that K has the standard form so that p[2] is X_cam[2].
    float p[2];
    p[0] = (fx * X_cam[0] + cx * X_cam[2]) / z;
    p[1] = (fy * X_cam[1] + cy * X_cam[2]) / z;
    PRINT1(p,2)

    // ---------- residual over x and y -------------
    int pix2d_offset = idx * 2;
    float kpt[2] = {pix2d[pix2d_offset + 0], pix2d[pix2d_offset + 1]};
    float std[2] = {pix2d_std[pix2d_offset + 0], pix2d_std[pix2d_offset + 1]};
    float res[3];
    #pragma unroll
    for (int i=0; i<2; i++)
        residuals[i*n_obs + idx] = res[i] = (p[i] - kpt[i]) / std[i];

    // ---------- residual over z -------------
    float guidance_z = 0.0f;
    for (int i = 0; i < C; i++)
        guidance_z += z_coefs[img_idx*C + i] * pix2d_dep[idx*C + i];
    float avg_focal = 0.5 * (fx + fy);
    if (is_depth_log)
        residuals[2*n_obs + idx] = res[2] = avg_focal * (guidance_z - log(z));
    else
        residuals[2*n_obs + idx] = res[2] = avg_focal * log(guidance_z / z);

    if (pnorm > 0) {
        // huber loss parameters
        float a = a = 0.5f * pnorm * powf(huber_delta, pnorm-2);
        float b = powf(huber_delta, pnorm) - a * huber_delta*huber_delta;
        float wz[3] = {1, 1, weight_z};

        #pragma unroll
        for (int i=0; i<3; i++) {
            float r = fabs(res[i]);

            // write err_pts = HuberLoss(min(res, max_pix_err) ** pnorm)
            r = (r > max_pix_err) ? max_pix_err : r; // limit impact of outliers
            residuals[(3+i)*n_obs + idx] = wz[i] * ((r <= huber_delta) ? a*r*r : powf(r, pnorm) - b);

            // write w_pts = max(res_norm, huber_delta) ** (pnorm-2)
            r = (r < huber_delta) ? huber_delta : r; // now weight for the Huber loss
            residuals[(6+i)*n_obs + idx] = (r == max_pix_err) ? w_pts_null : wz[i] * powf(r, pnorm-2);
        }
    }
}

// ---------------------------------------------------------------------
// Host function to launch the CUDA kernel.
// This function assumes that device memory is already allocated and
// pointers (d_K, d_P, etc.) refer to device buffers.
// ---------------------------------------------------------------------
void compute_residuals_rigs_cuda(const int* infos,             // device pointer: (n_imgs, 3)
                                 const float* d_K,             // device pointer: (n_cams, 9)
                                 const float* d_P_rc,          // device pointer: (n_cams, 16)
                                 const int64_t s_P_rc,         // P_rig2c stride 
                                 const float* d_P_wr,          // device pointer: (n_nodes, 16)
                                 const float* d_z_coefs,       // device pointer: (n_imgs, C)
                                 const float* d_pts3d,         // device pointer: (n_pts, 3)
                                 const int* d_pids,            // device pointer: (n_obs, )
                                 const float* d_pix2d,         // device pointer: (n_obs, 2)
                                 const float* d_pix2d_std,     // device pointer: (n_obs, 2)
                                 const float* d_pix2d_dep,     // device pointer: (n_obs, C)
                                 bool is_depth_log,
                                 float* d_res,                 // device pointer: (n_obs, 3, 7)
                                 int n_imgs, 
                                 int n_obs,
                                 int C,
                                 float weight_z, 
                                 float pnorm,
                                 float max_pix_err, 
                                 float huber_delta,
                                 float w_pts_null)
{
    // Choose thread block size and grid dimensions.
    int num_blocks = (n_obs + threads_per_block - 1) / threads_per_block;

    // Launch the kernel.
    if (is_depth_log) 
        compute_residuals_rigs_cuda_kernel<true><<<num_blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
            infos, d_K, d_P_rc, s_P_rc, d_P_wr, d_z_coefs, d_pts3d, d_pids, d_pix2d, d_pix2d_std, d_pix2d_dep, d_res, n_imgs, n_obs, C, weight_z, pnorm, max_pix_err, huber_delta, w_pts_null);
    else
        compute_residuals_rigs_cuda_kernel<false><<<num_blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
            infos, d_K, d_P_rc, s_P_rc, d_P_wr, d_z_coefs, d_pts3d, d_pids, d_pix2d, d_pix2d_std, d_pix2d_dep, d_res, n_imgs, n_obs, C, weight_z, pnorm, max_pix_err, huber_delta, w_pts_null);

    // It is recommended to check for launch errors and synchronize.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    // cudaDeviceSynchronize();
}


// CUDA kernel to compute the Jacobians for bundle adjustment
// Inputs:
//   infos     : [3, n_imgs] each column = (node_id, cam_id, cum_n_kpts)
//   K         : [n_cams, 3, 3] camera intrinsics (row-major)
//   P_rig2c   : [n_cams, 4, 4] rig-to-cam poses (row-major)
//   P_w2rig   : [n_node, 4, 4] world-to-rig poses (row-major)
//   z_coefs   : [n_imgs, C] global image shape parameters (C comes from shape)
//   pts3d     : [n_pts, 3] 3D points
//   pids      : [n_obs,] for each observed keypoint this is the pts3d index
//   pix2d_std : [n_obs, 2] per–observation standard deviations
//   pix2d_dep : [n_obs, C] local point shape parameters
// Outputs:
//   J_tau_zcf : [n_obs, 3, 7+C] Jacobian wrt camera update parameters and shape parameters
//   J_pts     : [n_obs, 3, 3] Jacobian wrt 3D point coordinates
//
template <typename Tfloat, bool is_depth_log>
__global__
void compute_jacobians_rigs_cuda_kernel(const int* __restrict__ infos,            // (n_imgs x 3)
                                       const float* __restrict__ K,             // (n_cams x 9)
                                       const float* __restrict__ P_rig2c,       // (n_cams x 16)
                                       const int64_t P_rig2c_stride,
                                       const float* __restrict__ P_w2rig,       // (n_nodes x 16)
                                       const float* __restrict__ z_coefs,       // (n_imgs x C)
                                       const float* __restrict__ pts3d,         // (n_pts x 3)
                                       const int* __restrict__ pids,            // (n_obs x 1)
                                       const float* __restrict__ pix2d_std,     // (n_obs x 2)
                                       const float* __restrict__ pix2d_dep,     // (n_obs x C)
                                       Tfloat* __restrict__ J_cam,               // (n_obs x 3 x 7)
                                       Tfloat* __restrict__ J_zcf,               // (n_obs x 1 x C)
                                       Tfloat* __restrict__ J_pts,               // (n_obs x 3 x 3)
                                       int n_imgs, int n_obs, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    PRINT0(idx)
    if (idx >= n_obs)
        return;

    // ---------- Load observation indices ----------
    // PRINT2(&infos,3,n_imgs)
    int img_idx = searchsorted(infos, n_imgs, idx);
    int node_idx = infos[1*n_imgs + img_idx];
    int cam_idx  = infos[2*n_imgs + img_idx];
    int pt_idx   = pids[idx];
    PRINT0(img_idx)
    PRINT0(pt_idx)

    // ---------- Load 3D point (pts3d[j]) ----------
    Tfloat X_world[3];
    X_world[0] = pts3d[pt_idx * 3 + 0];
    X_world[1] = pts3d[pt_idx * 3 + 1];
    X_world[2] = pts3d[pt_idx * 3 + 2];
    PRINT1(X_world,3)

    // ---------- Load camera intrinsics K_i (3x3) ----------
    // Extract fx, cx, fy, cy (we assume K has the usual form)
    Tfloat fx = K[cam_idx * 9 + 0];
    Tfloat cx = K[cam_idx * 9 + 2];
    Tfloat fy = K[cam_idx * 9 + 4];
    Tfloat cy = K[cam_idx * 9 + 5];

    // ---------- Load camera pose P_w2rig (4x4) and extract R (3x3) and t (3-vector) ----------
    const float* __restrict__ P_rw = P_w2rig + node_idx * 16;
    // R is the top-left 3x3 of P_i (row-major)
    // t is the translation (first 3 entries of the fourth column)
    Tfloat R_rw[9], t_rw[3];
    R_rw[0] = P_rw[0];  R_rw[1] = P_rw[1];  R_rw[2] = P_rw[2];  t_rw[0] = P_rw[3];
    R_rw[3] = P_rw[4];  R_rw[4] = P_rw[5];  R_rw[5] = P_rw[6];  t_rw[1] = P_rw[7];
    R_rw[6] = P_rw[8];  R_rw[7] = P_rw[9];  R_rw[8] = P_rw[10]; t_rw[2] = P_rw[11];
    PRINT2(R_rw,3,3)
    PRINT1(t_rw,3)

    // ---------- Load camera pose P_rig2c (4x4) and extract R (3x3) and t (3-vector) ----------
    const float* __restrict__ P_cr = P_rig2c + cam_idx * P_rig2c_stride;
    // R is the top-left 3x3 of P_i (row-major)
    // t is the translation (first 3 entries of the fourth column)
    Tfloat R_cr[9], t_cr[3];
    R_cr[0] = P_cr[0];  R_cr[1] = P_cr[1];  R_cr[2] = P_cr[2];  t_cr[0] = P_cr[3];
    R_cr[3] = P_cr[4];  R_cr[4] = P_cr[5];  R_cr[5] = P_cr[6];  t_cr[1] = P_cr[7];
    R_cr[6] = P_cr[8];  R_cr[7] = P_cr[9];  R_cr[8] = P_cr[10]; t_cr[2] = P_cr[11];
    PRINT2(R_cr,3,3)
    PRINT1(t_cr,3)

    // ---------- Compute X_rig = R_rw * X_world + t_rw ----------
    Tfloat X_rig[3];
    matmul<3,3,1>(R_rw, X_world, X_rig, t_rw);
    PRINT1(X_rig,3)

    // ---------- Compute X_cam = R_cr * X_rig + t_cr ----------
    Tfloat X_cam[3];
    matmul<3,3,1>(R_cr, X_rig, X_cam, t_cr);
    PRINT1(X_cam,3)

    // ---------- Project into pixel coordinates: p = K_i * X_cam ----------
    // Here we assume that K has the standard form so that p[2] is X_cam[2].
    Tfloat p[2];
    p[0] = fx * X_cam[0] + cx * X_cam[2];
    p[1] = fy * X_cam[1] + cy * X_cam[2];
    PRINT1(p,2)

    Tfloat z = X_cam[2];  // depth
    if (z <= 1e-3) return; // invalid point, no gradients

    // ---------- Compute the ∂(i,j,k)/∂X_cam, J_proj (3x3) ----------
    Tfloat inv_z = 1.0f / z;
    Tfloat inv_z2 = inv_z * inv_z;
    Tfloat avg_focal = 0.5 * (fx + fy);
    // Allocate as a 2x3 array in registers:
    Tfloat J_proj_cam[9];
    // Row 0:
    J_proj_cam[0] = fx * inv_z;
    J_proj_cam[1] = 0.0f;
    J_proj_cam[2] = cx * inv_z - p[0] * inv_z2;
    // Row 1:
    J_proj_cam[3] = 0.0f;
    J_proj_cam[4] = fy * inv_z;
    J_proj_cam[5] = cy * inv_z - p[1] * inv_z2;
    // Row 2: (derivative wrt
    J_proj_cam[6] = 0.0f;
    J_proj_cam[7] = 0.0f;
    J_proj_cam[8] = -avg_focal * inv_z;

    // Divide each row by the corresponding standard-deviation.
    int pix2d_offset = idx * 2;
    Tfloat std[2] = {pix2d_std[pix2d_offset + 0], pix2d_std[pix2d_offset + 1]};
    J_proj_cam[0] /= std[0];
    J_proj_cam[2] /= std[0];
    J_proj_cam[4] /= std[1];
    J_proj_cam[5] /= std[1];
    PRINT2(J_proj_cam,3,3)

    // ---------- Compute the ∂(i,j,k) / ∂X_rig (3x3) ----------
    Tfloat J_proj_rig[9];
    #define J_cam_rig R_cr
    matmul<3,3,3>(J_proj_cam, J_cam_rig, J_proj_rig);

    // ---------- Compute ∂X_rig/∂tau = [ -skew(X_rig) , I_3 ]   (3x6) ----------
    Tfloat J_rig_tau[18];
    // Row 0
    J_rig_tau[0] = 0.0f;        J_rig_tau[1] = X_rig[2];    J_rig_tau[2] = -X_rig[1];
    J_rig_tau[3] = 1.0f;        J_rig_tau[4] = 0.0f;        J_rig_tau[5] = 0.0f;
    // Row 1
    J_rig_tau[6] = -X_rig[2];   J_rig_tau[7] = 0.0f;        J_rig_tau[8] = X_rig[0];
    J_rig_tau[9] = 0.0f;        J_rig_tau[10] = 1.0f;       J_rig_tau[11] = 0.0f;
    // Row 2
    J_rig_tau[12] = X_rig[1];   J_rig_tau[13] = -X_rig[0];  J_rig_tau[14] = 0.0f;
    J_rig_tau[15] = 0.0f;       J_rig_tau[16] = 0.0f;       J_rig_tau[17] = 1.0f;
    PRINT2(J_rig_tau,3,6)

    // ---------- Jacobian w.r.t. camera update parameters, J_proj_tau = J_proj_rig (3x3) @ J_rig_tau (3x6)  ----------
    Tfloat J_proj_tau[18];
    matmul<3,3,6>(J_proj_rig, J_rig_tau, J_proj_tau);

    // ---------- Jacobian w.r.t. 3D point coordinates, J_pts_obs = J_proj (2x3) * R (3x3)  ----------
    Tfloat J_pts_obs[9];
    #define J_rig_world R_rw
    matmul<3,3,3>(J_proj_rig, J_rig_world, J_pts_obs);

    // Write J_cam: shape (n_obs, 3, 2+6)
    Tfloat guidance_z = 0.0f;
    for (int i = 0; i < C; i++)
        guidance_z += z_coefs[img_idx*C + i] * pix2d_dep[idx*C + i];
    PRINT0(guidance_z)

    int row_size = 2 + 6;
    int cam_offset = idx * 3 * row_size;
    J_cam[cam_offset + 0*row_size + 0] = X_cam[0] * inv_z / std[0];          // Write ∂(i) / ∂(f_x)
    #pragma unroll
    for (int i = 0; i < 6; i++)
        J_cam[cam_offset + 0*row_size + 2 + i] = J_proj_tau[0*6 + i];    // Write ∂(i) / ∂(tau)
    J_cam[cam_offset + 1*row_size + 1] = X_cam[1] * inv_z / std[1];          // Write ∂(j) / ∂(f_y)
    #pragma unroll
    for (int i = 0; i < 6; i++)
        J_cam[cam_offset + 1*row_size + 2 + i] = J_proj_tau[1*6 + i];    // Write ∂(j) / ∂(tau)
    float J_k_f = 0.0f;
    if (guidance_z > 0) {
        if (is_depth_log)
            J_k_f = 0.5f * (guidance_z + log(inv_z));
        else
            J_k_f = 0.5f * log(guidance_z * inv_z);
    }
    J_cam[cam_offset + 2*row_size + 0] = J_k_f;                          // Write ∂(k) / ∂(f_x)
    J_cam[cam_offset + 2*row_size + 1] = J_k_f;                          // Write ∂(k) / ∂(f_y)
    #pragma unroll
    for (int i = 0; i < 6; i++)
        J_cam[cam_offset + 2*row_size + 2 + i] = J_proj_tau[2*6 + i];    // Write ∂(k) / ∂(tau)

    // Write J_zcf: shape (n_obs, 1, C)                                  // Write ∂(k) / ∂(zcf)
    int zcf_offset = idx * C;
    for (int i = 0; i < C; i++) {
        if (is_depth_log)
            J_zcf[zcf_offset + i] = avg_focal * pix2d_dep[zcf_offset + i];
        else
            J_zcf[zcf_offset + i] = (guidance_z != 0) ? avg_focal / guidance_z * pix2d_dep[zcf_offset + i] : 0.0f;
    }

    // Write J_pts: shape (n_obs, 3, 3)
    int pts_offset = idx * 9;
    #pragma unroll
    for (int c = 0; c < 3*3; c++) {
        J_pts[pts_offset + c] = J_pts_obs[c];                            // Write ∂(i,j,k) / ∂(X_world)
    }
}

// ---------------------------------------------------------------------
// Host function to launch the CUDA kernel.
// This function assumes that device memory is already allocated and
// pointers (d_K, d_P, etc.) refer to device buffers.
// ---------------------------------------------------------------------
template <typename Tfloat>
void compute_jacobians_rigs_cuda(const int* infos,              // device pointer: (n_imgs, 3)
                                 const float* d_K,             // device pointer: (n_cams, 9)
                                 const float* d_P_rc,          // device pointer: (n_cams, 16)
                                 const int64_t s_P_rc,         // P_rig2c stride 
                                 const float* d_P_wr,          // device pointer: (n_nodes, 16)
                                 const float* d_z_coefs,       // device pointer: (n_imgs, C)
                                 const float* d_pts3d,         // device pointer: (n_pts, 3)
                                 const int* d_pids,            // device pointer: (n_obs, )
                                 const float* d_pix2d_std,     // device pointer: (n_obs, 2)
                                 const float* d_pix2d_dep,     // device pointer: (n_obs, C)
                                 bool is_depth_log,
                                 Tfloat* d_J_cam,               // device pointer: (n_obs, 3, 7)
                                 Tfloat* d_J_zcf,               // device pointer: (n_obs, 1, C)
                                 Tfloat* d_J_pts,               // device pointer: (n_obs, 3, 3)
                                 int n_imgs, 
                                 int n_obs,
                                 int C)
{
    // Choose thread block size and grid dimensions.
    int num_blocks = (n_obs + threads_per_block - 1) / threads_per_block;

    // Launch the kernel.
    if (is_depth_log)
        compute_jacobians_rigs_cuda_kernel<Tfloat, true><<<num_blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
            infos, d_K, d_P_rc, s_P_rc, d_P_wr, d_z_coefs, d_pts3d, d_pids, d_pix2d_std, d_pix2d_dep, d_J_cam, d_J_zcf, d_J_pts, n_imgs, n_obs, C);
    else
        compute_jacobians_rigs_cuda_kernel<Tfloat, false><<<num_blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
            infos, d_K, d_P_rc, s_P_rc, d_P_wr, d_z_coefs, d_pts3d, d_pids, d_pix2d_std, d_pix2d_dep, d_J_cam, d_J_zcf, d_J_pts, n_imgs, n_obs, C);

    // It is recommended to check for launch errors and synchronize.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    // cudaDeviceSynchronize();
}


// ----------------------------------------------------------------------------
// ------------------- Cholesky linear solve with 3x3 matrices ----------------
// ----------------------------------------------------------------------------

// Each thread processes exactly one 3x3 SPD matrix and one 3x1 rhs.
template <typename T>
__global__ void chol3x3_solve_kernel_cuda(
    const T* __restrict__ blocks,   // [N,3,3] row-major contiguous
    const T* __restrict__ rhs,      // [N,3]
    T* __restrict__ out,            // [N,3]
    int64_t N,
    T jitter)
{
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Base pointers
    const T* A = blocks + i * 9; // 3x3
    const T* b = rhs    + i * 3;
    T* y = out          + i * 3;

    // Load symmetric A (row-major). We assume SPD but add jitter to diagonal.
    // A = [a00 a01 a02; a10 a11 a12; a20 a21 a22] with a10=a01, a20=a02, a21=a12
    T a00 = A[0] + jitter;
    T a01 = A[1];
    T a02 = A[2];
    T a10 = A[3]; // should equal a01
    T a11 = A[4] + jitter;
    T a12 = A[5];
    T a20 = A[6];
    T a21 = A[7];
    T a22 = A[8] + jitter;

    // Enforce symmetry just in case (optional; costs a few cycles)
    // average the off-diagonals
    T a01s = (a01 + a10) * T(0.5);
    T a02s = (a02 + a20) * T(0.5);
    T a12s = (a12 + a21) * T(0.5);
    a01 = a10 = a01s;
    a02 = a20 = a02s;
    a12 = a21 = a12s;

    // Unrolled Cholesky: A = L L^T
    // l00
    T l00_sq = a00;
    l00_sq = l00_sq > T(0) ? l00_sq : (l00_sq + jitter); // guard
    T l00 = my_sqrt(l00_sq);

    // l10, l20
    T l10 = a01 / l00;
    T l20 = a02 / l00;

    // l11
    T tmp11 = a11 - l10*l10;
    tmp11 = tmp11 > T(0) ? tmp11 : (tmp11 + jitter);
    T l11 = my_sqrt(tmp11);

    // l21
    T l21 = (a12 - l10*l20) / l11;

    // l22
    T tmp22 = a22 - l20*l20 - l21*l21;
    tmp22 = tmp22 > T(0) ? tmp22 : (tmp22 + jitter);
    T l22 = my_sqrt(tmp22);

    // Forward solve: L z = b
    T z0 = b[0] / l00;
    T z1 = (b[1] - l10*z0) / l11;
    T z2 = (b[2] - l20*z0 - l21*z1) / l22;

    // Backward solve: L^T x = z
    T x2 = z2 / l22;
    T x1 = (z1 - l21*x2) / l11;
    T x0 = (z0 - l10*x1 - l20*x2) / l00;

    // Write result
    y[0] = x0;
    y[1] = x1;
    y[2] = x2;
}

template <typename T>
void chol3x3_solve_kernel(const torch::Tensor& blocks, const torch::Tensor& rhs, torch::Tensor& out, double jitter) {
    const int64_t N = blocks.size(0);
    const int blocksGrid = (int)((N + threads_per_block - 1) / threads_per_block);

    chol3x3_solve_kernel_cuda<T><<<blocksGrid, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        blocks.data_ptr<T>(),
        rhs.data_ptr<T>(),
        out.data_ptr<T>(),
        N,
        (T)jitter
    );
}


// ------------------------ Fused kernel for matrix-vector 

/*
// ----- Core kernels: one thread = one sparse block -----
template <int bh, int bw, bool TRANSPOSE, typename scalar_t, typename index_t>
__global__ void bcoo_matvec_generic(
    const scalar_t* __restrict__ blocks,   // [N, bh, bw]
    const index_t*  __restrict__ row_idx,  // [N]
    const index_t*  __restrict__ col_idx,  // [N]
    const scalar_t* __restrict__ x,        // [in_len]
    scalar_t*       __restrict__ y,        // [out_len]
    int64_t N)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int64_t block_offset = i * (int64_t)bh * (int64_t)bw;

    if (!TRANSPOSE) {
        // y[rb*bh + r] += sum_c A[r,c] * x[cb*bw + c]
        const int rb = (int)row_idx[i];
        const int cb = (int)col_idx[i];
        const int y_base = rb * bh;
        const int x_base = cb * bw;

        #pragma unroll
        for (int r = 0; r < bh; ++r) {
            const int row_offset = r * bw;

            scalar_t acc = scalar_t(0);
            #pragma unroll
            for (int c = 0; c < bw; ++c) {
                acc += blocks[block_offset + row_offset + c] * x[x_base + c];
            }
            atomicAddT(y + y_base + r, acc);
        }
    } else {
        // y[cb*bw + c] += sum_r A[r,c] * x[rb*bh + r]   (A^T * x)
        const int rb = (int)row_idx[i];
        const int cb = (int)col_idx[i];
        const int x_base = rb * bh;
        const int y_base = cb * bw;

        #pragma unroll
        for (int c = 0; c < bw; ++c) {
            scalar_t acc = scalar_t(0);

            // column c: walk rows
            #pragma unroll
            for (int r = 0; r < bh; ++r) {
                acc += blocks[block_offset + r * bw + c] * x[x_base + r];
            }
            atomicAddT(y + y_base + c, acc);
        }
    }
}

template <typename scalar_t, typename index_t>
void blockcoo_matvec_kernel(const torch::Tensor& blocks,
                            const torch::Tensor& row_idx,
                            const torch::Tensor& col_idx,
                            const torch::Tensor& x,
                            torch::Tensor& y,
                            bool transpose)
{
    const int64_t  N  = blocks.size(0);
    const int      bh = (int)blocks.size(1);
    const int      bw = (int)blocks.size(2);

    const int threads = 256;
    const int grid = (int)((N + threads - 1) / threads);

    const scalar_t* bptr = blocks.data_ptr<scalar_t>();
    const index_t*  rptr = row_idx.data_ptr<index_t>();
    const index_t*  cptr = col_idx.data_ptr<index_t>();
    const scalar_t* xptr = x.data_ptr<scalar_t>();
    scalar_t*       yptr = y.data_ptr<scalar_t>();

    if (bh == 2 && bw == 2) {
        if (!transpose)
            bcoo_matvec_generic<2,2,false><<<grid,threads, 0, at::cuda::getCurrentCUDAStream()>>>(bptr,rptr,cptr,xptr,yptr,N);
        else
            bcoo_matvec_generic<2,2,true><<<grid,threads, 0, at::cuda::getCurrentCUDAStream()>>>(bptr,rptr,cptr,xptr,yptr,N);
    }
    else if (bh == 3 && bw == 2) {
        if (!transpose)
            bcoo_matvec_generic<3,2,false><<<grid,threads, 0, at::cuda::getCurrentCUDAStream()>>>(bptr,rptr,cptr,xptr,yptr,N);
        else
            bcoo_matvec_generic<3,2,true><<<grid,threads, 0, at::cuda::getCurrentCUDAStream()>>>(bptr,rptr,cptr,xptr,yptr,N);
    }
    else if (bh == 2 && bw == 6) {
        if (!transpose)
            bcoo_matvec_generic<2,6,false><<<grid,threads, 0, at::cuda::getCurrentCUDAStream()>>>(bptr,rptr,cptr,xptr,yptr,N);
        else
            bcoo_matvec_generic<2,6,true><<<grid,threads, 0, at::cuda::getCurrentCUDAStream()>>>(bptr,rptr,cptr,xptr,yptr,N);
    }
    else if (bh == 3 && bw == 6) {
        if (!transpose) 
            bcoo_matvec_generic<3,6,false><<<grid,threads, 0, at::cuda::getCurrentCUDAStream()>>>(bptr,rptr,cptr,xptr,yptr,N);
        else 
            bcoo_matvec_generic<3,6,true><<<grid,threads, 0, at::cuda::getCurrentCUDAStream()>>>(bptr,rptr,cptr,xptr,yptr,N);
    }
    else {
        TORCH_CHECK(false, "cannot handle COO blocks of shape=(", bh, ", ", bw, ")");
    }
}*/

// TRANSPOSE=false: per-block threads = BH; dot length = BW; y idx uses row_idx
// TRANSPOSE=true : per-block threads = BW; dot length = BH; y idx uses col_idx
template <int BH, int BW, bool TRANSPOSE, typename scalar_t, typename index_t>
__global__ void bcoo_matvec_oneval_kernel(  const scalar_t* __restrict__ blocks,   // [N,BH,BW], contiguous
                                            const index_t*  __restrict__ row_idx,  // [N]
                                            const index_t*  __restrict__ col_idx,  // [N]
                                            const scalar_t* __restrict__ x,        // [in_len]
                                            scalar_t*       __restrict__ y,        // [out_len]
                                            int64_t N)
{
    constexpr int PER_BLOCK = TRANSPOSE ? BW : BH;
    const int64_t total = N * (int64_t)PER_BLOCK;

    // 1D grid-stride over all (block, local_out) pairs
    for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += (int64_t)blockDim.x * gridDim.x)
    {
        const int64_t bi = tid / PER_BLOCK;          // which sparse block
        const int      lo = (int)(tid - bi * PER_BLOCK); // which output in that block
        const int rb = (int)row_idx[bi];
        const int cb = (int)col_idx[bi];

        const scalar_t* A = blocks + bi * (BH * BW);
        scalar_t acc = scalar_t(0);

        if (!TRANSPOSE) {
            // compute dot of row lo with x segment for column block
            const scalar_t* Arow = A + lo * BW;
            const scalar_t* xseg = x + cb * BW;
            #pragma unroll
            for (int c = 0; c < BW; ++c) {
                acc += Arow[c] * xseg[c];
            }
            atomicAddT(y + rb * BH + lo, acc);
        } else {
            // compute dot of column lo with x segment for row block
            const scalar_t* xseg = x + rb * BH;
            #pragma unroll
            for (int r = 0; r < BH; ++r) {
                acc += A[r * BW + lo] * xseg[r];
            }
            atomicAddT(y + cb * BW + lo, acc);
        }
    }
}

// ----------------------- Launch / dispatch -----------------------
template <typename scalar_t, typename index_t>
void blockcoo_matvec_kernel(const torch::Tensor& blocks,
                            const torch::Tensor& row_idx,
                            const torch::Tensor& col_idx,
                            const torch::Tensor& x,
                            torch::Tensor& y,
                            bool transpose)
{
    const int64_t  N  = blocks.size(0);
    const int      bh = (int)blocks.size(1);
    const int      bw = (int)blocks.size(2);

    const scalar_t* bptr = blocks.data_ptr<scalar_t>();
    const index_t*  rptr = row_idx.data_ptr<index_t>();
    const index_t*  cptr = col_idx.data_ptr<index_t>();
    const scalar_t* xptr = x.data_ptr<scalar_t>();
    scalar_t*       yptr = y.data_ptr<scalar_t>();

    const int64_t per_block = transpose ? bw : bh;
    const int64_t total = N * per_block;

    // Heuristic: enough blocks to cover large outputs
    const int grid = (int)((total + threads_per_block - 1) / threads_per_block);

    #define LAUNCH_KERNEL(BH, BW) \
        if (bh == BH && bw == BW) { \
            if (!transpose) \
                bcoo_matvec_oneval_kernel<BH,BW,false><<<grid, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(bptr,rptr,cptr,xptr,yptr,N); \
            else \
                bcoo_matvec_oneval_kernel<BH,BW,true><<<grid, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(bptr,rptr,cptr,xptr,yptr,N); \
            return; \
        }

        LAUNCH_KERNEL(1, 1)
        LAUNCH_KERNEL(1, 2)
        LAUNCH_KERNEL(1, 3)
        LAUNCH_KERNEL(1, 6)
        LAUNCH_KERNEL(2, 2)
        LAUNCH_KERNEL(2, 3)
        LAUNCH_KERNEL(2, 6)
        LAUNCH_KERNEL(2, 9)
        LAUNCH_KERNEL(3, 1)
        LAUNCH_KERNEL(3, 2)
        LAUNCH_KERNEL(3, 3)
        LAUNCH_KERNEL(3, 5)
        LAUNCH_KERNEL(3, 6)
        LAUNCH_KERNEL(3, 9)
        LAUNCH_KERNEL(3, 16)
        LAUNCH_KERNEL(3, 17)
        LAUNCH_KERNEL(5, 2)
        LAUNCH_KERNEL(5, 5)
        LAUNCH_KERNEL(6, 1)
        LAUNCH_KERNEL(6, 6)
        LAUNCH_KERNEL(9, 9)
        LAUNCH_KERNEL(9, 2)
        LAUNCH_KERNEL(9, 3)
        LAUNCH_KERNEL(9, 6)
        LAUNCH_KERNEL(16, 2)
        LAUNCH_KERNEL(16, 3)
        LAUNCH_KERNEL(16, 6)
        LAUNCH_KERNEL(16, 16)
        LAUNCH_KERNEL(17, 2)
        LAUNCH_KERNEL(17, 6)
        LAUNCH_KERNEL(17, 17)

        TORCH_CHECK(false, "cannot handle COO blocks of shape=(", bh, ", ", bw, ")");

    #undef LAUNCH_KERNEL
}


// ----------------- einsum bki x bkj --> bij ----------------------------

// Overloadless version (fix the pointer arithmetic above):
template <typename T, int I, int J, int K>
__global__ void einsum_bki_bkj_kernel_cuda(
    const T* __restrict__ left,   // [B, K, I]
    int64_t l_bs, int64_t l_ks,
    const T* __restrict__ right,  // [B, K, J]
    int64_t r_bs, int64_t r_ks , 
    T* __restrict__ out,          // [B, I, J]
    int64_t B)
{
    // int64_t l_ks = I;
    int64_t l_is = 1;
    // int64_t r_ks = J;
    int64_t r_js = 1;
    int64_t o_bs = I*J;
    int64_t o_is = J;
    int64_t o_js = 1;

    const int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    T acc[I * J];
    #pragma unroll
    for (int t = 0; t < I * J; ++t) acc[t] = T(0);

    const T* l_b = left  + b * l_bs;
    const T* r_b = right + b * r_bs;

    #pragma unroll
    for (int k = 0; k < K; ++k) {
        const T* l_row = l_b + k * l_ks;
        const T* r_row = r_b + k * r_ks;

        T L[I], R[J];

        #pragma unroll
        for (int ii = 0; ii < I; ++ii) L[ii] = *(l_row + ii * l_is);

        #pragma unroll
        for (int jj = 0; jj < J; ++jj) R[jj] = *(r_row + jj * r_js);

        // Outer product accumulate
        #pragma unroll
        for (int ii = 0; ii < I; ++ii) {
            const T li = L[ii];
            #pragma unroll
            for (int jj = 0; jj < J; ++jj) {
                acc[ii * J + jj] += li * R[jj];
            }
        }
    }

    // Store
    T* o_b = out + b * o_bs;
    #pragma unroll
    for (int ii = 0; ii < I; ++ii) {
        T* o_row = o_b + ii * o_is;
        #pragma unroll
        for (int jj = 0; jj < J; ++jj) {
            o_row[jj * o_js] = acc[ii * J + jj];
        }
    }
}

template <typename T, int I, int J, int K>
void fused_einsum_bki_bkj_kernel2( const torch::Tensor& left, 
                                   const torch::Tensor& right, 
                                   torch::Tensor& out,
                                   int64_t B)
{
    const int grid = (int)((B + threads_per_block - 1) / threads_per_block);
    TORCH_CHECK(out.is_contiguous(), "output should be contiguous");

    einsum_bki_bkj_kernel_cuda<T, I, J, K><<<grid, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        left.data_ptr<T>(),
        left.stride(0), left.stride(1), //left.stride(2),
        right.data_ptr<T>(),
        right.stride(0), right.stride(1), //right.stride(2),
        out.data_ptr<T>(),
        //out.stride(0), out.stride(1), out.stride(2),
        B
    );
}

template <typename T>
void fused_einsum_bki_bkj_kernel( const torch::Tensor& left, 
                                  const torch::Tensor& right, 
                                  torch::Tensor& out,
                                  int I, int J, int64_t B, int64_t K)
{
    // Supported (I,J): (3,2) (3,3) (3,6) (1,3) (1,2) (1,6)
    if (K == 1) {
        if (I == 1 && J == 1) return fused_einsum_bki_bkj_kernel2<T,1,1,1>(left,right,out,B);
        if (I == 1 && J == 2) return fused_einsum_bki_bkj_kernel2<T,1,2,1>(left,right,out,B);
        if (I == 1 && J == 6) return fused_einsum_bki_bkj_kernel2<T,1,6,1>(left,right,out,B);
        if (I == 2 && J == 2) return fused_einsum_bki_bkj_kernel2<T,2,2,1>(left,right,out,B);
        if (I == 2 && J == 3) return fused_einsum_bki_bkj_kernel2<T,2,3,1>(left,right,out,B);
        if (I == 2 && J == 6) return fused_einsum_bki_bkj_kernel2<T,2,6,1>(left,right,out,B);
        if (I == 3 && J == 1) return fused_einsum_bki_bkj_kernel2<T,3,1,1>(left,right,out,B);
        if (I == 3 && J == 2) return fused_einsum_bki_bkj_kernel2<T,3,2,1>(left,right,out,B);
        if (I == 3 && J == 3) return fused_einsum_bki_bkj_kernel2<T,3,3,1>(left,right,out,B);
        if (I == 3 && J == 5) return fused_einsum_bki_bkj_kernel2<T,3,5,1>(left,right,out,B);
        if (I == 3 && J == 6) return fused_einsum_bki_bkj_kernel2<T,3,6,1>(left,right,out,B);
        if (I == 3 && J == 9) return fused_einsum_bki_bkj_kernel2<T,3,9,1>(left,right,out,B);
        if (I == 3 && J == 16) return fused_einsum_bki_bkj_kernel2<T,3,16,1>(left,right,out,B);
        if (I == 3 && J == 17) return fused_einsum_bki_bkj_kernel2<T,3,17,1>(left,right,out,B);
        if (I == 5 && J == 2) return fused_einsum_bki_bkj_kernel2<T,5,2,1>(left,right,out,B);
        if (I == 5 && J == 5) return fused_einsum_bki_bkj_kernel2<T,5,5,1>(left,right,out,B);
        if (I == 6 && J == 6) return fused_einsum_bki_bkj_kernel2<T,6,6,1>(left,right,out,B);
        if (I == 9 && J == 2) return fused_einsum_bki_bkj_kernel2<T,9,2,1>(left,right,out,B);
        if (I == 9 && J == 9) return fused_einsum_bki_bkj_kernel2<T,9,9,1>(left,right,out,B);
        if (I == 16 && J == 2) return fused_einsum_bki_bkj_kernel2<T,16,2,1>(left,right,out,B);
        if (I == 16 && J == 6) return fused_einsum_bki_bkj_kernel2<T,16,6,1>(left,right,out,B);
        if (I == 16 && J == 16) return fused_einsum_bki_bkj_kernel2<T,16,16,1>(left,right,out,B);
        if (I == 17 && J == 2) return fused_einsum_bki_bkj_kernel2<T,17,2,1>(left,right,out,B);
        if (I == 17 && J == 6) return fused_einsum_bki_bkj_kernel2<T,17,6,1>(left,right,out,B);
        if (I == 17 && J == 17) return fused_einsum_bki_bkj_kernel2<T,17,17,1>(left,right,out,B);

    } else if (K == 3) {
        if (I == 2 && J == 2) return fused_einsum_bki_bkj_kernel2<T,2,2,3>(left,right,out,B);
        if (I == 2 && J == 3) return fused_einsum_bki_bkj_kernel2<T,2,3,3>(left,right,out,B);
        if (I == 2 && J == 6) return fused_einsum_bki_bkj_kernel2<T,2,6,3>(left,right,out,B);
        if (I == 3 && J == 2) return fused_einsum_bki_bkj_kernel2<T,3,2,3>(left,right,out,B);
        if (I == 3 && J == 3) return fused_einsum_bki_bkj_kernel2<T,3,3,3>(left,right,out,B);
        if (I == 3 && J == 6) return fused_einsum_bki_bkj_kernel2<T,3,6,3>(left,right,out,B);
        if (I == 6 && J == 6) return fused_einsum_bki_bkj_kernel2<T,6,6,3>(left,right,out,B);
    }

    TORCH_CHECK(false, "unsupported (I,J,K)=(", I, ",", J, ",", K, "). Please add more cases in fused_einsum_bki_bkj_kernel()");
}

// ----------------------------------------------
// ------------ entry call functions ------------

std::tuple<int, int, int> check_inputs(torch::Tensor infos,
                                       torch::Tensor K,
                                       torch::Tensor P_rig2c,
                                       torch::Tensor P_w2rig, 
                                       torch::Tensor z_coefs,
                                       torch::Tensor pts3d,
                                       torch::Tensor pids,
                                       torch::Tensor pix2d,
                                       torch::Tensor pix2d_std,
                                       torch::Tensor pix2d_dep
                                      ) {
    CHECK_CUDA_AND_CONTIGUOUS(infos, 2)
    CHECK_CUDA_AND_CONTIGUOUS_SHAPE(K, 3, 3)
    CHECK_CUDA(P_rig2c)
    TORCH_CHECK(P_rig2c.dim() == 3 && P_rig2c.stride(1) == 4 && P_rig2c.stride(2) == 1, "P_rig2c should be (N,4,4) and contiguous inside 4x4 blocks");
    CHECK_CUDA_AND_CONTIGUOUS_SHAPE(P_w2rig, 3, 4)
    CHECK_CUDA_AND_CONTIGUOUS(z_coefs, 2)
    CHECK_CUDA_AND_CONTIGUOUS_SHAPE(pts3d, 2, 3)
    CHECK_CUDA_AND_CONTIGUOUS(pids, 1)
    CHECK_CUDA_AND_CONTIGUOUS_SHAPE(pix2d, 2, 2)
    CHECK_CUDA_AND_CONTIGUOUS_SHAPE(pix2d_std, 2, 2)
    CHECK_CUDA_AND_CONTIGUOUS(pix2d_dep, 2)

    // Get dimensions.
    TORCH_CHECK(infos.size(0) == 3, "infos should have 3 rows = [nkpt_cum, node_ids, cam_ids]");
    int n_imgs = infos.size(1);
    int n_obs = pids.size(0);
    int C = pix2d_dep.size(1);  // assuming z_hyps shape is [n_obs, C]
    TORCH_CHECK(C == z_coefs.size(1), "Number of Z coefs is not consistent");

    return {n_imgs, n_obs, C};
}

// ------------------------------------------------------------------
// PyTorch wrapper (exposed to Python)
torch::Tensor residuals_with_rigs( torch::Tensor infos,
                                   torch::Tensor K,
                                   torch::Tensor P_rig2c,
                                   torch::Tensor P_w2rig, 
                                   torch::Tensor z_coefs,
                                   torch::Tensor pts3d,
                                   torch::Tensor pids,
                                   torch::Tensor pix2d,
                                   torch::Tensor pix2d_std,
                                   torch::Tensor pix2d_dep,
                                   bool is_depth_log,
                                   float weight_z, 
                                   float pnorm,
                                   float max_pix_err, 
                                   float huber_delta,
                                   float w_pts_null
                                 ) {
    auto [n_imgs, n_obs, C] = check_inputs(infos, K, P_rig2c, P_w2rig, z_coefs, pts3d, pids, pix2d, pix2d_std, pix2d_dep);
    TORCH_CHECK( (pnorm == -1) || (pnorm > 0 && pnorm <= 2), "pnorm must be in {-1} U ]0,2]");

    // other outputs if requested
    int n_output = 3;
    if (pnorm > 0) 
        n_output += 6; // we also output err_pts and w_pts

    // Allocate output tensors.
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor output = torch::zeros({n_output, n_obs}, options);

    // Launch CUDA kernel wrapper.
    compute_residuals_rigs_cuda( infos.data_ptr<int>(),
                                 K.data_ptr<float>(),
                                 P_rig2c.data_ptr<float>(),
                                 P_rig2c.stride(0),
                                 P_w2rig.data_ptr<float>(),
                                 z_coefs.data_ptr<float>(),
                                 pts3d.data_ptr<float>(),
                                 pids.data_ptr<int>(),
                                 pix2d.data_ptr<float>(),
                                 pix2d_std.data_ptr<float>(),
                                 pix2d_dep.data_ptr<float>(),
                                 is_depth_log,
                                 output.data_ptr<float>(),
                                 n_imgs, 
                                 n_obs,
                                 C,
                                 weight_z,
                                 pnorm,
                                 max_pix_err,
                                 huber_delta,
                                 w_pts_null
                                 );
    return output.transpose(0,1);
}

// ------------------------------------------------------------------
// PyTorch wrapper (exposed to Python)
std::vector<torch::Tensor> jacobians_with_rigs( torch::Tensor infos,
                                                torch::Tensor K,
                                                torch::Tensor P_rig2c,
                                                torch::Tensor P_w2rig, 
                                                torch::Tensor z_coefs,
                                                torch::Tensor pts3d,
                                                torch::Tensor pids,
                                                torch::Tensor pix2d_std,
                                                torch::Tensor pix2d_dep,
                                                bool is_depth_log,
                                                bool output_float64
                                              ) {

    int n_imgs, n_obs, C;
    std::tie(n_imgs, n_obs, C) = check_inputs(infos, K, P_rig2c, P_w2rig, z_coefs, pts3d, pids, pix2d_std, pix2d_std, pix2d_dep);

    // Allocate output tensors.
    auto options = torch::TensorOptions().dtype(output_float64 ? torch::kFloat64 : torch::kFloat32).device(torch::kCUDA);
    torch::Tensor J_pts = torch::zeros({n_obs, 3, 3}, options);
    torch::Tensor J_cam = torch::zeros({n_obs, 3, 8}, options);
    torch::Tensor J_zcf = torch::zeros({n_obs, 1, C}, options);

    AT_DISPATCH_FLOATING_TYPES(
        J_pts.type().scalarType(), "compute_jacobians_rigs_cuda", ([&] {
        compute_jacobians_rigs_cuda( infos.data_ptr<int>(),
                                     K.data_ptr<float>(),
                                     P_rig2c.data_ptr<float>(),
                                     P_rig2c.stride(0),
                                     P_w2rig.data_ptr<float>(),
                                     z_coefs.data_ptr<float>(),
                                     pts3d.data_ptr<float>(),
                                     pids.data_ptr<int>(),
                                     pix2d_std.data_ptr<float>(),
                                     pix2d_dep.data_ptr<float>(),
                                     is_depth_log,
                                     J_cam.data_ptr<scalar_t>(),
                                     J_zcf.data_ptr<scalar_t>(),
                                     J_pts.data_ptr<scalar_t>(),
                                     n_imgs, 
                                     n_obs,
                                     C);
    }));
    return {J_cam, J_zcf, J_pts};
}

// C++/ATen entry
torch::Tensor chol3x3_solve(torch::Tensor blocks, torch::Tensor rhs, double jitter) {
    TORCH_CHECK(blocks.is_cuda(), "blocks must be CUDA tensor");
    TORCH_CHECK(rhs.is_cuda(), "rhs must be CUDA tensor");
    TORCH_CHECK(blocks.is_contiguous(), "blocks must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(blocks.dim() == 3 && blocks.size(1) == 3 && blocks.size(2) == 3,
                "blocks must have shape [N,3,3]");
    TORCH_CHECK(rhs.dim() == 2 && rhs.size(1) == 3 && rhs.size(0) == blocks.size(0),
                "rhs must have shape [N,3]");
    TORCH_CHECK(blocks.scalar_type() == rhs.scalar_type(), "dtype mismatch between blocks and rhs");
    auto out = torch::empty_like(rhs);

    AT_DISPATCH_FLOATING_TYPES(
        blocks.type().scalarType(),  "chol3x3_solve_kernel", ([&] {
        chol3x3_solve_kernel<scalar_t>(blocks, rhs, out, jitter);
    }));

    return out;
}

torch::Tensor blockcoo_matvec( torch::Tensor blocks,      // [N,bh,bw]
                               torch::Tensor row_indices, // [N] int32/int64
                               torch::Tensor col_indices, // [N] int32/int64
                               torch::Tensor x,           // [in_len]
                               int64_t HB1,               // number of block rows
                               int64_t WB1,               // number of block cols
                               bool transpose_blocks)     // false: y=A x; true: y=A^T x
{
    TORCH_CHECK(blocks.is_cuda() && x.is_cuda(), "blocks and x must be CUDA tensors");
    TORCH_CHECK(row_indices.is_cuda() && col_indices.is_cuda(), "indices must be CUDA tensors");
    TORCH_CHECK(blocks.is_contiguous(), "blocks must be contiguous [N,bh,bw]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(blocks.dim()==3, "blocks must be [N,bh,bw]");
    TORCH_CHECK(row_indices.dim()==1 && col_indices.dim()==1 && row_indices.size(0)==blocks.size(0) && col_indices.size(0)==blocks.size(0),
                "indices must be [N] and match blocks.size(0)");
    TORCH_CHECK(blocks.scalar_type()==x.scalar_type(), "dtype mismatch");
    TORCH_CHECK(col_indices.scalar_type()==row_indices.scalar_type(), "index dtypes must match");

    const int64_t bh = blocks.size(1);
    const int64_t bw = blocks.size(2);

    int64_t in_len  = transpose_blocks ? (HB1 * bh) : (WB1 * bw);
    int64_t out_len = transpose_blocks ? (WB1 * bw) : (HB1 * bh);

    TORCH_CHECK(x.numel()==in_len, "x has wrong length for the provided shapes");
    auto y = torch::zeros({out_len}, blocks.options());

    // dtype/index dispatch
    if (row_indices.scalar_type()==torch::kInt) {
        if (blocks.scalar_type()==torch::kFloat) {
            blockcoo_matvec_kernel<float, int>(blocks,row_indices,col_indices,x,y,transpose_blocks);
        } else if (blocks.scalar_type()==torch::kDouble) {
            blockcoo_matvec_kernel<double,int>(blocks,row_indices,col_indices,x,y,transpose_blocks);
        } else {
            TORCH_CHECK(false, "Only float32/float64 supported");
        }
    } else if (row_indices.scalar_type()==torch::kLong) { // int64 indices
        if (blocks.scalar_type()==torch::kFloat) {
            blockcoo_matvec_kernel<float, int64_t>(blocks,row_indices,col_indices,x,y,transpose_blocks);
        } else if (blocks.scalar_type()==torch::kDouble) {
            blockcoo_matvec_kernel<double,int64_t>(blocks,row_indices,col_indices,x,y,transpose_blocks);
        } else {
            TORCH_CHECK(false, "Only float32/float64 supported");
        }
    } else {
        TORCH_CHECK(false, "Only int32/int64 indicessupported");
    }
    return y;
}


torch::Tensor fused_einsum_bki_bkj(torch::Tensor left, torch::Tensor right)
{
    TORCH_CHECK(left.is_cuda() && right.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(left.dim() == 3 && right.dim() == 3, "expected [B,K,I] and [B,K,J]");
    TORCH_CHECK(left.size(0) == right.size(0), "B mismatch");
    TORCH_CHECK(left.size(1) == right.size(1), "K mismatch");
    TORCH_CHECK(left.scalar_type() == right.scalar_type(), "dtype mismatch");

    // Fast-path assumptions: per-row contiguous inside each block
    TORCH_CHECK(left.stride(2) == 1, "left must have stride(2)==1 (got ", left.stride(2), ")");
    TORCH_CHECK(right.stride(2) == 1, "right must have stride(2)==1 (got ", right.stride(2), ")");

    const int64_t B = left.size(0);
    const int64_t K = left.size(1);
    const int I = (int)left.size(2);
    const int J = (int)right.size(2);

    auto out = torch::empty({B, I, J}, left.options());

    if (left.scalar_type() == torch::kFloat) {
        fused_einsum_bki_bkj_kernel<float>(left, right, out, I, J, B, K);
    } else if (left.scalar_type() == torch::kDouble) {
        fused_einsum_bki_bkj_kernel<double>(left, right, out, I, J, B, K);
    } else {
        TORCH_CHECK(false, "only float32/float64 supported");
    }

    return out;
}


// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("residuals_with_rigs", &residuals_with_rigs,
          "Compute residuals for bundle adjustment (CUDA)",
          py::arg("infos"), py::arg("K"), py::arg("P_rig2c"), py::arg("P_w2rig"), py::arg("z_coefs"), py::arg("pts3d"), py::arg("pids"), py::arg("pix2d"), py::arg("pix2d_std"), py::arg("pix2d_dep"), py::arg("is_depth_log"),
          py::arg("weight_z") = 1.f, py::arg("pnorm") = -1.f, py::arg("max_pix_err") = 100.f, py::arg("huber_delta") = 0.5f, py::arg("w_pts_nul") = 1e-16f);

    m.def("jacobians_with_rigs", &jacobians_with_rigs,
          "Compute explicit Jacobians for bundle adjustment (CUDA)",
          py::arg("infos"), py::arg("K"), py::arg("P_rig2c"), py::arg("P_w2rig"), py::arg("z_coefs"), py::arg("pts3d"), py::arg("pids"), py::arg("pix2d_std"), py::arg("pix2d_dep"), py::arg("is_depth_log"), py::arg("float64") = false);

    m.def("chol3x3_solve", &chol3x3_solve,
          "Fused batched 3x3 SPD solve via Cholesky (CUDA)",
          py::arg("blocks"), py::arg("rhs"), py::arg("jitter") = 1e-12);

    m.def("blockcoo_matvec", &blockcoo_matvec,
          "Sparse block-COO matvec / vecmat (CUDA, fused)",
          py::arg("blocks"), py::arg("row_indices"), py::arg("col_indices"),
          py::arg("x"), py::arg("HB1"), py::arg("WB1"), py::arg("transpose_blocks") = false);

    m.def("fused_einsum_bki_bkj", &fused_einsum_bki_bkj,
          "Fused einsum('bki,bkj->bij') with strided batches (CUDA)");
}
