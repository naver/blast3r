#!/usr/bin/env bash
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

# Reproduce the offline relative-pose results on ETH3D.
#
#   scripts/xps/eth3d_offline.sh <eth3d_dir> <output_dir> [matcher_ckpt] [depth_ckpt]
#
# <eth3d_dir> holds one directory per scene, each with the scene's images and a
# gt.pkl. All other settings are the CLI defaults, which are the published ones.
set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 4 ]; then
    sed -n '4,9p' "$0" | sed 's/^# \?//'
    exit 1
fi
ETH3D=$1; OUTPUT=$2
CKPT_MATCH=${3:-naver/blast3r-matcher}
CKPT_DEPTH=${4:-naver/blast3r-depth}

python "$(dirname "$0")/../run_offline.py" \
    --images "$ETH3D"/*/ \
    --output "$OUTPUT" \
    --ckpt_match "$CKPT_MATCH" \
    --ckpt_depth "$CKPT_DEPTH" \
    --size 512 \
    --mono_cam \
    --export_mode pose \
    --skip_existing

python "$(dirname "$0")/../eval_offline.py" --gt_dir "$ETH3D" --preds_dir "$OUTPUT"
