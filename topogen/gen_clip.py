import os
import cv2
import torch
import clip
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import pdb

# --- Configuration ---
IMAGE_DIR = "/media/hochul/2TB/IJRR_Topomaps/topo_gail2library_1m"  # Folder containing 0.png, 1.png, ...
OUTPUT_DIR = "/media/hochul/2TB/IJRR_Topomaps/topo_gail2library_1m_topogen_08"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SIM_THRESHOLD = 0.95
MAX_FRAMES = 100

# --- Load CLIP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_clip_feature(img_bgr):
    image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(image)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().squeeze()

# --- Load and Sort Images ---
image_paths = sorted(
    [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(".png")],
    key=lambda x: int(os.path.basename(x).split(".")[0])
)

# --- Keyframe Selection ---
last_feat = None
key_idx = 0

for i, img_path in enumerate(image_paths):
    img = cv2.imread(img_path)
    feat = extract_clip_feature(img)


    if last_feat is None or cosine_similarity([feat], [last_feat])[0][0] < SIM_THRESHOLD:
        save_path = os.path.join(OUTPUT_DIR, f"keyframe_{i:06d}.jpg")
        cv2.imwrite(save_path, img)
        print(f"Saved keyframe: {save_path}")
        last_feat = feat
        key_idx += 1

        if key_idx >= MAX_FRAMES:
            break
