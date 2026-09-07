#!/usr/bin/env bash
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

# Reproduce the online SLAM results on the TUM-RGBD fr1 sequences.
#
#   scripts/xps/tum_rgbd_online.sh <tum_dir> <output_dir> [matcher_ckpt] [depth_ckpt]
#
# <tum_dir> holds one directory per sequence, each with rgb/, rgb.txt and
# groundtruth.txt as distributed by TUM.
set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 4 ]; then
    sed -n '4,9p' "$0" | sed 's/^# \?//'
    exit 1
fi
TUM=$1; OUTPUT=$2
CKPT_MATCH=${3:-naver/blast3r-matcher}
CKPT_DEPTH=${4:-naver/blast3r-depth}

SEQUENCES=(
    rgbd_dataset_freiburg1_360
    rgbd_dataset_freiburg1_desk
    rgbd_dataset_freiburg1_desk2
    rgbd_dataset_freiburg1_floor
    rgbd_dataset_freiburg1_plant
    rgbd_dataset_freiburg1_room
    rgbd_dataset_freiburg1_rpy
    rgbd_dataset_freiburg1_teddy
    rgbd_dataset_freiburg1_xyz
)

# One sequence per invocation, because each needs its own scene name.
for seq in "${SEQUENCES[@]}"; do
    python "$(dirname "$0")/../run_online.py" \
        --images "$TUM/$seq/rgb" \
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
    --data_root "$TUM" \
    --preds_dir "$OUTPUT" \
    --benchmark tum_rgbd \
    --interpolate
