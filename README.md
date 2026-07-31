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

Official implementation of **GuideNav**, a vision-only teach-and-repeat navigation system for
kilometer-scale route following in sidewalk environments.

GuideNav is **RGB-only** and **untethered**: it learns a route from a single demonstration and
repeats it using visual place recognition (CosPlace) for localization and reloc3r for relative
pose estimation. No depth or LiDAR is used at navigation time.

## Setup

Requires Ubuntu 22.04, Python 3.10, CUDA, and ROS2 Humble. ROS2 is needed for all stages,
including offline evaluation.

```bash
sudo apt install python3.10-venv ffmpeg   # venv: Ubuntu ships python3.10 without ensurepip
                                          # ffmpeg: only needed for debug/img2vid.py

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

Relative pose estimation uses [reloc3r](https://github.com/ffrivera0/reloc3r), a separate
repository. All three steps are required — a plain `git clone` will not run.

```bash
# 1. clone WITH the croco submodule (reloc3r's ViT blocks live there)
git clone --recurse-submodules https://github.com/ffrivera0/reloc3r \
    guidenav/match_to_control/methods/reloc3r

# 2. make croco/models importable (it ships without __init__.py, otherwise
#    ModuleNotFoundError: No module named 'models.blocks')
touch guidenav/match_to_control/methods/reloc3r/croco/models/__init__.py

# 3. add load_images_reloc3r(), which accepts in-memory frames instead of file
#    paths so ROS images can be fed in directly
git -C guidenav/match_to_control/methods/reloc3r apply \
    ../../../../third_party/reloc3r_load_images_in_memory.patch
```

The patch is against upstream commit `761fac6`; if it no longer applies, copy the function out
of the patch file by hand. reloc3r's CUDA RoPE kernel is optional — without it you will see
`Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead`,
and navigation still works.

### Model weights

```bash
# CosPlace place recognition, from PlaceNav (18,952,687 bytes; filename must not change --
# it is set by checkpoint_path in config/models.yaml)
mkdir -p model_weights
curl -L -o model_weights/efficientnet_85x85.pth \
  "https://drive.usercontent.google.com/download?id=1M1rvlRYiV9F0VHKggAOyzun4PwGFTlZ1&export=download"
```

reloc3r downloads its own weights (`siyan824/reloc3r-512`, ~1.4 GB) from Hugging Face on first
use. This can take a long time and shows no progress bar when stdout is redirected — it is not
hung.

## Quickstart: offline evaluation

Replay recorded frames through the full stack — place recognition, subgoal selection, pose
estimation, control — with no robot. Expected layout:

```
data/
├── topomap/          # route keyframes: 0.png, 1.png, ... (numbered in route order)
└── test_run/color/   # observation frames from a repeat traversal   <- --img-dir
```

`--img-dir` accepts either a directory containing a `color/` subfolder or the frame folder
itself.

Topomap keyframes **must** be named `<int>.png` — the index is parsed with
`int(name.split(".")[0])` and anything else raises. Observation frames are sorted by the digits
in their filename, so raw recorder output like `1753734170.224759102.png` usually works, but
only while every name has the same number of digits. Renumbering observations to `0.png`,
`1.png`, … is the safe option:

```bash
# subsample a raw run (every 10th frame) into the numeric scheme
python3 - <<'PY'
import os, shutil
src, dst = "/path/to/raw_run/color", "./data/test_run/color"
os.makedirs(dst, exist_ok=True)
for i, f in enumerate(sorted(os.listdir(src), key=lambda f: float(os.path.splitext(f)[0]))[::10]):
    shutil.copy2(os.path.join(src, f), os.path.join(dst, f"{i}.png"))
PY
```

`navigate.py` does not write images; with `--enable-debug` it publishes
`/debug/image/compressed` and `/debug/nav_data`. Run the saver as a second process, started
first so it does not miss frames:

```bash
source /opt/ros/humble/setup.bash

# terminal 1
python debug/save_data.py --topo_dir ./data/topomap --output_dir ./debug_results

# terminal 2
python guidenav/navigate.py \
    --robot mc \
    --robot-config-path ./config/robots.yaml \
    --topomap-base-dir ./data -d topomap \
    --model-weight-dir model_weights \
    --model-config-path config/models.yaml \
    --offline-images --img-dir ./data/test_run/color \
    --offline-fps 3 \
    --enable-debug
```

Each frame shows the observation beside the matched subgoal, plus the estimated relative pose
and commanded `v`/`ω`. Render to video with
`python debug/img2vid.py --input ./debug_results` (requires ffmpeg).

Three things to expect:
- **The replay loops.** Offline streaming restarts at frame 0 after the last frame. Stop with
  Ctrl+C once `Goal reached! Stopping navigation.` appears.
- **Keep `--offline-fps` low.** The debug topics are best-effort with queue depth 1 and the
  saver renders a plot per frame, so a fast replay drops frames from the video. On one 154-frame
  run, `--offline-fps 3` saved 148 frames where the default `30` saved only 80.
- **The first 4 observations produce no output** — the model needs 5 frames of context — so
  debug frame `N` corresponds to input frame `N+4`.

## Next steps

See [docs/SETUP.md](docs/SETUP.md) for recording a demonstration, building a topomap from it,
deploying to a robot, and the flag reference.

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
