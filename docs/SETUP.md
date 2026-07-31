# GuideNav: full workflow

The [README](../README.md) covers installation and a self-contained offline run. This document
covers the rest: recording a demonstration, building a topomap from it, and deploying to a
robot.

## 1. Demonstration Collection

Traverse the route once, recording RGB. Depth and odometry are optional — they are only used by
the odometry-based topomap builder below.

```bash
python sensor/extract_data_two.py --output-dir ./data/teaching_run
```

Writes a run directory named after the start time, containing `d435_color/`, `depth/`, and
`odom.csv`. Edit the topic names in `sensor/extract_data_two.py` to match your camera driver.

## 2. Build Topomap

The topomap is a folder of keyframes named `0.png`, `1.png`, … (`.jpg` also works). The
numbering is the route order and is parsed as an integer, so non-numeric names will fail.

```bash
# DINOv3 adaptive keyframe selection (used in the paper)
python topogen/gen_dinov3.py \
    --input ./data/teaching_run/<timestamp>/d435_color \
    --output ./data/topomap_raw \
    --dinov3-repo /path/to/dinov3 --weights /path/to/dinov3_vitl16.pth

# gen_dinov3.py writes keyframe_000000.jpg; rename to the numeric scheme
mkdir -p ./data/topomap
i=0; for f in $(ls -v ./data/topomap_raw/keyframe_*.jpg); do \
    cp "$f" "./data/topomap/$i.jpg"; i=$((i+1)); done

# pre-compute place recognition descriptors (optional; done automatically on first run)
python -m guidenav.place_recognition.extract_database --topomap-dir ./data/topomap
```

DINOv3 is not on torch.hub, so `--dinov3-repo` / `--weights` must point at a local
[DINOv3](https://github.com/facebookresearch/dinov3) clone and checkpoint. Omitting them falls
back to DINOv2, which works but selects different keyframes than the paper.

Alternatively `python sensor/build_topomap.py <run_dir> <out_dir> --distance 1.0` selects
keyframes by odometry spacing; it needs depth and `odom.csv`, expects RGB in a folder named
`color/`, and writes an already-numbered `topo/` subfolder.

`--img-size` must match between `extract_database` and navigation (default `85 64` in both), or
the descriptors will not be comparable.

## 3. Robot Deployment

The offline command without `--offline-images`, so frames come from the live camera. Real
velocity commands are published — make sure the robot is clear to move.

```bash
source /opt/ros/humble/setup.bash

python guidenav/navigate.py \
    --robot mc \
    --robot-config-path ./config/robots.yaml \
    --topomap-base-dir ./data -d topomap \
    --model-weight-dir model_weights \
    --model-config-path config/models.yaml
```

Use `--robot go2` for a Unitree Go2; `--robot` selects only the velocity limits.

> **Topics are hardcoded, not read from `robots.yaml`.** `navigate.py` subscribes to
> `/d435i/color/image_raw` and publishes to `/cmd_vel` regardless of `--robot`. For different
> topic names, either edit them in `guidenav/navigate.py` or remap at launch:
> `--ros-args -r /cmd_vel:=/mobile_base/commands/velocity`

The deployment camera should match the one used for the demonstration; place recognition is
sensitive to changes in field of view or mounting height.

## Useful flags

`guidenav/parser.py` has the full list; `navigate.sh` has wrapped invocations.

| flag | default | notes |
|---|---|---|
| `--filter-mode` | `bayesian` | or `sliding_window` |
| `--lookahead` | `1` | nodes ahead of the match to aim for |
| `--img-size` | `85 64` | must match `extract_database` |
| `--device` | `cuda` | or `cpu` |
| `--offline-fps` | `30` | use `3` when saving debug frames |
| `--recompute-place-recognition-db` | off | rebuild descriptors |

## Troubleshooting

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'models.blocks'` | reloc3r setup step 2 was skipped |
| `Warning, cannot find cuda-compiled version of RoPE2D` | Expected — reloc3r's optional CUDA kernel is not built. Navigation works, slightly slower. |
| `UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy` | Expected — the venv uses `--system-site-packages` for `rclpy`, so ROS's scipy sits alongside the newer numpy pinned by `requirements.txt`. Harmless. |
| First run seems to hang with no output | reloc3r is fetching ~1.4 GB of weights from Hugging Face; there is no progress bar when stdout is redirected. |
| Debug video is missing frames | Lower `--offline-fps` (see the README quickstart). |
| Offline run never exits | Expected — the replay loops. Ctrl+C after `Goal reached! Stopping navigation.` |

## Project Structure

```
GuideNav/
├── guidenav/               # Core navigation system
│   ├── navigate.py         # Main navigation node
│   ├── parser.py           # Argument parser
│   ├── match_to_control/   # reloc3r pose estimation + control
│   │   └── methods/        # reloc3r repo (cloned by you)
│   ├── models/pr_models/   # Place recognition models
│   └── place_recognition/  # VPR filtering (Bayesian / sliding window)
├── sensor/                 # Data collection
├── topogen/                # Topomap generation
├── debug/                  # Debug frame saver and video rendering
├── config/                 # robots.yaml, models.yaml
└── model_weights/          # Checkpoints (not included)
```
