#!/usr/bin/env bash
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

# Reproduce the online SLAM results on the 8 ETH3D-SLAM sequences.
#
#   scripts/xps/eth3d_slam_online.sh <eth3d_slam_dir> <output_dir> [matcher_ckpt] [depth_ckpt]
#
# <eth3d_slam_dir> holds one directory per sequence, each with rgb/, rgb.txt and
# groundtruth.txt as distributed by ETH3D. Only the training sequences carry
# ground truth, so only those can be evaluated.
set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 4 ]; then
    sed -n '4,10p' "$0" | sed 's/^# \?//'
    exit 1
fi
ETH3D_SLAM=$1; OUTPUT=$2
CKPT_MATCH=${3:-naver/blast3r-matcher}
CKPT_DEPTH=${4:-naver/blast3r-depth}

SEQUENCES=(
    cables_1
    camera_shake_1
    einstein_1
    plant_1
    plant_2
    sofa_1
    table_3
    table_7
)

# One sequence per invocation, because each needs its own scene name.
for seq in "${SEQUENCES[@]}"; do
    python "$(dirname "$0")/../run_online.py" \
        --images "$ETH3D_SLAM/$seq/rgb" \
        --output "$OUTPUT" \
        --name "$seq" \
        --ckpt_match "$CKPT_MATCH" \
        --ckpt_depth "$CKPT_DEPTH" \
        --size 512 \
        --mono_cam \
        --frame_step 4 \
        --export_mode pose \
        --gba_step 8 \
        --max_iters_local 10 \
        --max_num_tracks 100000 \
        -wz 1e-4 1 -nz 1 17 -K True False -P True False \
        --skip_existing \
        --skip_failures
done

python "$(dirname "$0")/../eval_online.py" \
    --data_root "$ETH3D_SLAM" \
    --preds_dir "$OUTPUT" \
    --benchmark eth3d_slam \
    --interpolate
