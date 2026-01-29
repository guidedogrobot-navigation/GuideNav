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

## Overview

GuideNav is an **RGB-only**, **untethered** visual navigation system designed for autonomous robot deployment. The system learns navigation routes from demonstration and can reliably repeat them using visual place recognition and relative pose estimation.

### Key Features

- **RGB-only Navigation**: No depth sensors or LiDAR required during deployment
- **Topological Mapping**: Efficient sparse map representation using keyframes
- **Place Recognition**: Robust localization using visual place recognition models
- **Real-time Performance**: Optimized for edge deployment on NVIDIA Jetson platforms
- **Multiple Feature Matching Methods**: Support for LoFTR, RoMa, MAST3R, LiftFeat

## Installation

### Prerequisites

- Ubuntu 22.04
- Python 3.8+
- CUDA 11.x or 12.x
- ROS2 Humble (for robot deployment)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/guidedogrobot-navigation/GuideNav.git
cd GuideNav
```

2. Create a conda environment:
```bash
conda create -n guidenav python=3.10
conda activate guidenav
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download model weights:
```bash
# Place recognition models
mkdir -p model_weights
# Download CosPlace weights from: https://github.com/gmberton/CosPlace
# Download feature matching weights as needed
```

## Usage

### 1. Build a Topological Map (Teaching Phase)

First, collect images along the desired route:

```bash
# Record RGB-D data with odometry
python sensor/extract_data_two.py --output-dir ./data/teaching_run

# Build topological map from recorded data (using naive approach)
python sensor/build_topomap.py ./data/teaching_run ./data/topomap --distance 1.0
```

For adaptive keyframe selection using visual features:
```bash
python topogen/gen_dinov3.py --input ./data/raw_images --output ./data/topomap
```

### 2. Extract Place Recognition Features

```bash
# Features are automatically extracted after the demonstration
# Or pre-compute them:
python -m guidenav.place_recognition.extract_database --topomap-dir ./data/topomap
```

### 3. Navigation (Repeat Phase)

For real-time navigation with ROS2:
```bash
python guidenav/navigate.py \
    --robot mc \
    --robot-config-path ./config/robots.yaml \
    --topomap-base-dir ./data \
    -d topomap \
    --model-weight-dir ./model_weights \
    --model-config-path ./config/models.yaml \
    --feature-matching reloc3r
```

## Configuration

### Robot Configuration (`config/robots.yaml`)

Configure robot-specific parameters including:
- Maximum linear/angular velocities
- Camera intrinsics
- Control parameters


## Project Structure

```
GuideNav/
├── guidenav/               # Core navigation system
│   ├── navigate.py         # Main navigation node
│   ├── parser.py           # Argument parser
│   ├── match_to_control/   # Feature matching and control
│   │   ├── feature_match.py
│   │   ├── control.py
│   │   └── se2_estimate.py
│   ├── models/             # Neural network models
│   │   └── pr_models/      # Place recognition models
│   └── place_recognition/  # VPR filtering modules
├── sensor/                 # Data collection tools
├── topogen/               # Topological map generation
├── config/                # Configuration files
└── model_weights/         # Model checkpoints (not included)
```


## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{hwang2025guidenav,
  title={GuideNav: User-Informed Development of a Vision-Only Robotic Navigation Assistant For Blind Travelers},
  author={Hwang, Hochul and Yang, Soowan and Monon, Jahir Sadik and Giudice, Nicholas A and Lee, Sunghoon Ivan and Biswas, Joydeep and Kim, Donghyun},
  journal={arXiv preprint arXiv:2512.06147},
  year={2025}
}
```

## Acknowledgment
We would like to express our gratitude to the authors and contributors of the following repositories:

- [PlaceNav](https://github.com/lasuomela/PlaceNav)
- [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer)
- [CosPlace](https://github.com/gmberton/CosPlace)


## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
