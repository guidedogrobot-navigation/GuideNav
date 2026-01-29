#!/bin/bash
# GuideNav navigation script
# Usage: ./navigate.sh

# === Online Navigation (with real camera) ===
python guidenav/navigate.py \
    --robot mc \
    --robot-config-path ./config/robots.yaml \
    -d topomap \
    --topomap-base-dir ./data \
    --model-weight-dir model_weights \
    --model-config-path config/models.yaml \
    --vis-dir ./output_visualization \
    --feature-matching reloc3r

# === Offline Navigation (with recorded images) ===
# python guidenav/navigate.py \
#     --robot mc \
#     --robot-config-path ./config/robots.yaml \
#     -d topomap \
#     --topomap-base-dir ./data \
#     --model-weight-dir model_weights \
#     --model-config-path config/models.yaml \
#     --vis-dir ./output_visualization \
#     --feature-matching reloc3r \
#     --offline-images \
#     --img-dir /path/to/test/images/color
