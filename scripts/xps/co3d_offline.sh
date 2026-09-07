#!/usr/bin/env bash
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

# Reproduce the offline relative-pose results on CO3D v2.
#
#   scripts/xps/co3d_offline.sh <prepared_dir> <output_dir> [matcher_ckpt] [depth_ckpt]
#
# <prepared_dir> is the output of scripts/preprocessing/prepare_co3d.py: one
# directory per sequence under <category>/, holding the ten evaluated frames and
# a gt.pkl. Ten images per scene, so retrieval is off and the two-stage bundle
# adjustment the published CO3D runs used is requested explicitly.
#
# Metrics are reported per category. The published table averages over scenes
# within a category, then over the 41 categories.
set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 4 ]; then
    sed -n '4,14p' "$0" | sed 's/^# \?//'
    exit 1
fi
CO3D=$1; OUTPUT=$2
CKPT_MATCH=${3:-naver/blast3r-matcher}
CKPT_DEPTH=${4:-naver/blast3r-depth}

for category_dir in "$CO3D"/*/; do
    category=$(basename "$category_dir")

    python "$(dirname "$0")/../run_offline.py" \
        --images "$category_dir"*/ \
        --output "$OUTPUT/$category" \
        --ckpt_match "$CKPT_MATCH" \
        --ckpt_depth "$CKPT_DEPTH" \
        --size 512 \
        --mono_cam \
        --export_mode pose \
        --kpt_spacing 8 \
        -ret none \
        -wz 1e-4 1 -nz 1 17 -K True False -P True False \
        --skip_existing \
        --skip_failures

    python "$(dirname "$0")/../eval_offline.py" \
        --gt_dir "$category_dir" \
        --preds_dir "$OUTPUT/$category"
done
