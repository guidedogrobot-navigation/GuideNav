# GuideNav: User-Informed Development of a Vision-Only Robotic Navigation Assistant For Blind Travelers


  <div align="center">                                   
    <video src="https://github.com/user-attachments/asset
  s/057ec63f-41ca-4d35-89ea-f578838eb2ae" width="800"    
  controls autoplay muted loop>                          
    </video>                                             
  </div> 
  
<!-- To embed video: drag-drop teaser.mp4 into GitHub issue/PR, then paste the generated link here -->

<p align="center">
  <a href="https://arxiv.org/abs/2512.06147"><img src="https://img.shields.io/badge/arXiv-2512.06147-b31b1b.svg" alt="arXiv"></a>
  <a href="https://guidedogrobot-navigation.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

This repository contains the official implementation of **GuideNav**, a vision-only teach-and-repeat navigation system that enables kilometer-scale route following in sidewalk environments.

GuideNav is **RGB-only** and **untethered**: it learns a route from a single demonstration and repeats it using visual place recognition (CosPlace) for localization and reloc3r for relative pose estimation. No depth or LiDAR is used at navigation time.

## Setup

Requires Ubuntu 22.04, Python 3.10, CUDA, and ROS2 Humble. ROS2 is needed for
all stages, including offline evaluation.

```bash
git clone https://github.com/guidedogrobot-navigation/GuideNav.git
cd GuideNav

# venv must see the ROS2 python packages (rclpy, cv_bridge)
source /opt/ros/humble/setup.bash
python3.10 -m venv --system-site-packages ~/guidenav-venv
source ~/guidenav-venv/bin/activate

# install torch for your CUDA version first, e.g. CUDA 12.8
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### reloc3r

Relative pose estimation uses [reloc3r](https://github.com/ffrivera0/reloc3r),
which is a separate repository. Clone it into
`guidenav/match_to_control/methods/reloc3r` and apply the three steps below —
all are required, and a plain `git clone` alone will not run.

```bash
# 1. clone WITH the croco submodule (reloc3r's ViT blocks live there)
git clone --recurse-submodules https://github.com/ffrivera0/reloc3r \
    guidenav/match_to_control/methods/reloc3r

# 2. make croco/models importable as a package
touch guidenav/match_to_control/methods/reloc3r/croco/models/__init__.py

# 3. add load_images_reloc3r(), which accepts in-memory frames
#    instead of file paths so ROS images can be fed in directly
git -C guidenav/match_to_control/methods/reloc3r apply \
    ../../../../third_party/reloc3r_load_images_in_memory.patch
```

Notes:
- Step 2 is needed because `croco/models/` ships without an `__init__.py`;
  without it you get `ModuleNotFoundError: No module named 'models.blocks'`.
- Step 3 appends one function to `reloc3r/utils/image.py`. The patch is against
  upstream commit `761fac6`; if reloc3r has moved on and the patch no longer
  applies, copy the function out of the patch file by hand.
- reloc3r's CUDA RoPE kernel is optional. If it is not compiled you will see
  `Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch
  version instead` — navigation still works.
- reloc3r downloads its own weights (`siyan824/reloc3r-512`) from Hugging Face
  on first use.

Download the CosPlace place recognition weights from
[PlaceNav](https://github.com/lasuomela/placenav) and place them at
`model_weights/efficientnet_85x85.pth` (the filename is set by
`checkpoint_path` in `config/models.yaml`). reloc3r fetches its own weights from
Hugging Face on first use.

## Usage

### 1. Demonstration Collection

Traverse the route once, recording RGB. Depth and odometry are optional — they
are only used by the odometry-based topomap builder below.

```bash
python sensor/extract_data_two.py --output-dir ./data/teaching_run
```

This writes a run directory named after the start time, containing
`d435_color/`, `depth/`, and `odom.csv`. Edit the topic names in
`sensor/extract_data_two.py` to match your camera driver.

### 2. Build Topomap

The topomap is a folder of keyframes named `0.jpg`, `1.jpg`, … The numbering is
the route order and is parsed as an integer, so non-numeric names will fail.

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

DINOv3 is not on torch.hub, so `--dinov3-repo` / `--weights` must point at a
local [DINOv3](https://github.com/facebookresearch/dinov3) clone and checkpoint.
Omitting them falls back to DINOv2, which works but selects different keyframes
than the paper.

Alternatively `python sensor/build_topomap.py <run_dir> <out_dir> --distance 1.0`
selects keyframes by odometry spacing; it needs depth and `odom.csv`, expects
RGB in a folder named `color/`, and writes an already-numbered `topo/` subfolder.

`--img-size` must match between `extract_database` and navigation (default
`85 64` in both), or the descriptors will not be comparable.

### 3. Evaluate Offline

Replay recorded frames through the full stack — place recognition, subgoal
selection, pose estimation, control — with no robot. Add `--offline-images` for
offline; omitting it is live mode (there is no `--online` flag).

```bash
source /opt/ros/humble/setup.bash

python guidenav/navigate.py \
    --robot mc \
    --robot-config-path ./config/robots.yaml \
    --topomap-base-dir ./data -d topomap \
    --model-weight-dir model_weights \
    --model-config-path config/models.yaml \
    --offline-images --img-dir ./data/test_run/color
```

`--img-dir` accepts either a directory containing a `color/` subfolder or the
frame folder itself. Frames are sorted numerically.

**Saving the run.** `navigate.py` does not write images; with `--enable-debug`
it publishes `/debug/image/compressed` and `/debug/nav_data`. Run the saver as a
second process, started first so it does not miss frames:

```bash
# terminal 1
python debug/save_data.py --topo_dir ./data/topomap --output_dir ./debug_results

# terminal 2
python guidenav/navigate.py ... --offline-images --img-dir ./data/test_run/color --enable-debug
```

Each frame shows the observation beside the matched subgoal, plus the estimated
relative pose and commanded `v`/`ω`. Render them to video with
`python debug/img2vid.py --input ./debug_results` (requires ffmpeg).

### 4. Robot Deployment

Same command without `--offline-images`, so frames come from the live camera.
Real velocity commands are published — make sure the robot is clear to move.

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

> **Topics are hardcoded, not read from `robots.yaml`.** `navigate.py`
> subscribes to `/d435i/color/image_raw` and publishes to `/cmd_vel` regardless
> of `--robot`. For different topic names, either edit them in
> `guidenav/navigate.py` or remap at launch:
> `--ros-args -r /cmd_vel:=/mobile_base/commands/velocity`

The deployment camera should match the one used for the demonstration; place
recognition is sensitive to changes in field of view or mounting height.

Other useful flags: `--start-node-idx` / `--goal-node-idx` to run a route
segment, `--filter-mode` (`bayesian` or `sliding_window`), `--lookahead`,
`--device`. See `guidenav/parser.py` for the full list, and `navigate.sh` for
wrapped invocations.

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

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{hwang2026guidenav,
  title={Guidenav: User-informed development of a vision-only robotic navigation assistant for blind travelers},
  author={Hwang, Hochul and Yang, Soowan and Monon, Jahir Sadik and Giudice, Nicholas A and Lee, Sunghoon Ivan and Biswas, Joydeep and Kim, Donghyun},
  booktitle={Proceedings of the 21st ACM/IEEE International Conference on Human-Robot Interaction},
  pages={1129--1139},
  year={2026}
}
```

## Acknowledgment
We would like to express our gratitude to the authors and contributors of the following repositories:

- [PlaceNav](https://github.com/lasuomela/placenav)
- [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer)
- [CosPlace](https://github.com/gmberton/CosPlace)
- [reloc3r](https://github.com/ffrivera0/reloc3r)


## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
