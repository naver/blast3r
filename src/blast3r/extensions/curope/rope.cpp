// Copyright (C) 2026-present Naver Corporation. All rights reserved.
//
// Adapted for BLASt3R from CroCo (https://github.com/naver/croco),
// models/curope/curope.cpp.

#include <torch/extension.h>

#define MIN(a,b) (((a) < (b)) ? (a) : (b))

// forward declaration
void rope_2d_cuda( torch::Tensor tokens, const torch::Tensor pos, const float base, const float fwd );
void rope_3d_cuda( torch::Tensor tokens, const torch::Tensor pos, const float base, const float fwd );

void rope_2d_cpu( torch::Tensor tokens, const torch::Tensor positions, const float base, const float fwd )
{
    const int B = tokens.size(0);
    const int N = tokens.size(1);
    const int H = tokens.size(2);
    const int D = tokens.size(3) / 4;

    auto tok = tokens.accessor<float, 4>();
    auto pos = positions.accessor<float, 3>();

    for (int b = 0; b < B; b++) {
      for (int x = 0; x < 2; x++) { // y and then x (2d)
        for (int n = 0; n < N; n++) {
        
            // grab the token position
            const float p = pos[b][n][x];

            for (int h = 0; h < H; h++) {
                for (int d = 0; d < D; d++) {
                    // grab the two values
                    float u = tok[b][n][h][d+0+x*2*D];
                    float v = tok[b][n][h][d+D+x*2*D];

                    // grab the cos,sin
                    const float inv_freq = fwd * p / powf(base, d/float(D));
                    float c = cosf(inv_freq);
                    float s = sinf(inv_freq);

                    // write the result
                    tok[b][n][h][d+0+x*2*D] = u*c - v*s;
                    tok[b][n][h][d+D+x*2*D] = v*c + u*s;
                }
            }
        }
      }
    }
}


void rope_3d_cpu( torch::Tensor tokens, const torch::Tensor positions, const float base, const float fwd )
{
    const int B = tokens.size(0);
    const int N = tokens.size(1);
    const int H = tokens.size(2);
    const int D = tokens.size(3);
    const int F0 = (D + 5) / 6;  

    auto tok = tokens.accessor<float, 4>();
    auto pos = positions.accessor<float, 3>();

    for (int b = 0; b < B; b++) {
      for (int x = 0; x < 3; x++) { // x, y, then z (3d)

        // number of frequency per position channel
        const int F_offset = 2*x*F0;
        const int F = MIN(F0, (D - F_offset)/2); // last channel is maybe smaller

        for (int n = 0; n < N; n++) {

            // grab the token position
            const float p = pos[b][n][x];

            for (int h = 0; h < H; h++) {
                for (int d = 0; d < F; d++) {
                    // grab the two values
                    float u = tok[b][n][h][F_offset+0+d];
                    float v = tok[b][n][h][F_offset+F+d];

                    // grab the cos,sin
                    const float theta = fwd * p / powf(base, d/float(F));
                    // if (b == 0 && h == 0 && n == 1 + 64 + 3072)
                        // printf("x=%d p=%f d=%d u=%f v=%f => theta = %f\n", x, p, d, u, v, theta);
                    float c = cosf(theta);
                    float s = sinf(theta);

                    // write the result
                    tok[b][n][h][F_offset+0+d] = u*c - v*s;
                    tok[b][n][h][F_offset+F+d] = v*c + u*s;
                }
            }
        }
      }
    }
}

void rope_2d( torch::Tensor tokens,     // B,N,H,D
        const torch::Tensor positions,  // B,N,2
        const float base, 
        const float fwd )
{
    TORCH_CHECK(tokens.dim() == 4, "tokens must have 4 dimensions");
    TORCH_CHECK(positions.dim() == 3, "positions must have 3 dimensions");
    TORCH_CHECK(tokens.size(0) == positions.size(0), "batch size differs between tokens & positions");
    TORCH_CHECK(tokens.size(1) == positions.size(1), "seq_length differs between tokens & positions");
    TORCH_CHECK(tokens.size(3) % 2 == 0, "positions.shape[3] must be even");
    TORCH_CHECK(positions.size(2) == 2, "positions.shape[2] must be equal to 2");
    TORCH_CHECK(tokens.is_cuda() == positions.is_cuda(), "tokens and positions are not on the same device" );

    if (tokens.is_cuda())
        rope_2d_cuda( tokens, positions, base, fwd );
    else
        rope_2d_cpu( tokens, positions, base, fwd );
}

void rope_3d( torch::Tensor tokens,     // B,N,H,D
        const torch::Tensor positions,  // B,N,3
        const float base, 
        const float fwd )
{
    TORCH_CHECK(tokens.dim() == 4, "tokens must have 4 dimensions");
    TORCH_CHECK(positions.dim() == 3, "positions must have 3 dimensions");
    TORCH_CHECK(tokens.size(0) == positions.size(0), "batch size differs between tokens & positions");
    TORCH_CHECK(tokens.size(1) == positions.size(1), "seq_length differs between tokens & positions");
    TORCH_CHECK(tokens.size(3) % 2 == 0, "positions.shape[3] must be even");
    TORCH_CHECK(positions.size(2) == 3, "positions.shape[2] must be equal to 3");
    TORCH_CHECK(tokens.is_cuda() == positions.is_cuda(), "tokens and positions are not on the same device" );

    if (tokens.is_cuda())
        rope_3d_cuda( tokens, positions, base, fwd );
    else
        rope_3d_cpu( tokens, positions, base, fwd );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_2d", &rope_2d, "RoPE 2d forward/backward without cos/sin");
  m.def("rope_3d", &rope_3d, "RoPE 3d forward/backward without cos/sin");
}
