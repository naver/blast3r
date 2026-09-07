# Benchmarks

Reproducing the published results. None of this is needed to reconstruct your
own scenes; see the [README](../README.md) for that.

BLASt3R is evaluated on ETH3D and CO3D v2 offline, and on TUM-RGBD and
ETH3D-SLAM online. Each is rebuildable from its official download.

## Preparing the data

The two offline benchmarks need their download converted into the layout the
evaluation reads: one directory per scene, holding the evaluated images and a
`gt.pkl`. The images are symlinked, so this costs no disk space beyond the
ground truth.

```bash
# ETH3D: the official high-resolution multi-view download
python scripts/preprocessing/prepare_eth3d.py \
  --eth3d_root /path/to/ETH3D --out data/eth3d

# CO3D v2: the official download, 41 seen categories
python scripts/preprocessing/prepare_co3d.py \
  --co3d_root /path/to/co3d --out data/co3d
```

CO3D is evaluated on ten frames per scene, drawn by an RNG seeded from the scene
name, and the prepared directories record which frames those are. The scene set
is the `fewview_dev` test sequences of the 41 seen categories whose
`viewpoint_quality_score` exceeds `--min_quality` (0.5), 2080 sequences in all.

The two online benchmarks are used as distributed, with no preparation step.

## Running a benchmark

One script per benchmark, each running inference over every scene and then
evaluating. They take the data directory and an output directory:

```bash
scripts/xps/eth3d_offline.sh     data/eth3d          out/eth3d
scripts/xps/co3d_offline.sh      data/co3d           out/co3d
scripts/xps/tum_rgbd_online.sh   /path/to/TUM_RGBD   out/tum
scripts/xps/eth3d_slam_online.sh /path/to/ETH3D_slam out/eth3d_slam
```

The released checkpoints download from the Hugging Face Hub on first use. To
evaluate your own instead, pass them as two further arguments:
`<script> <data_dir> <output_dir> <matcher_ckpt> <depth_ckpt>`.

All four are safe to re-run: finished scenes are skipped. Each states its own
settings in a header comment, including the bundle-adjustment schedule, which
differs between benchmarks.

On ETH3D, averaged over the 13 scenes:

| Metric | Value |
|---|---|
| AUC@30 | 98.5 |
| RRA@5 | 100.0 |
| RTA@5 | 97.9 |

Expect small per-scene variation between machines. The bundle adjustment stops
when the loss stops decreasing, and that test can fall either side of the
threshold depending on the GPU and the CUDA build, moving individual scenes by a
few tenths of a point.

## Evaluating on your own

The benchmark scripts call the two evaluation scripts, which also run
standalone. Both print a table and write a CSV.

**Relative pose**, for offline reconstructions. Ground truth is one directory
per scene holding a `gt.pkl` with `poses` (Nx4x4 camera-to-world) and
`filenames`, as the preprocessing scripts write:

```bash
python scripts/eval_offline.py --gt_dir /path/to/gt --preds_dir /path/to/output
```

It reports AUC@30, RRA@5 and RTA@5 per scene plus an average. A scene with no
prediction is scored as zero and reported as such, as in the paper.

**Trajectory error**, for online reconstructions on TUM-RGBD or ETH3D-SLAM,
which share a file format:

```bash
python scripts/eval_online.py --data_root /path/to/TUM_RGBD --preds_dir /path/to/output \
  --benchmark tum_rgbd --interpolate
```

It reports absolute trajectory error after Sim(3) alignment, since monocular
reconstruction fixes scale only up to a constant. `--benchmark` selects the
published sequence list (`tum_rgbd` or `eth3d_slam`); `--sequences` overrides it.

`--frame_step` reconstructs a subset of the frames, so the trajectory is sampled
more coarsely than the ground truth. `--interpolate`, which the published
numbers use, interpolates it onto every ground-truth frame before measuring the
error. The alternative, `--allow_subset`, measures only the reconstructed frames
and reports the fraction covered.
