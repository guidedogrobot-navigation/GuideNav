"""
DINOv3-based adaptive keyframe selection for topological map generation.

Usage:
    python gen_dinov3.py --input /path/to/images --output /path/to/output
"""
import os
import cv2
import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
from collections import deque
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DINOv3 keyframe selection")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input image directory")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output directory for keyframes")
    parser.add_argument("--method", "-m", type=str, default="multi_criteria",
                       choices=["adaptive_threshold", "diversity_buffer",
                               "temporal_spacing", "multi_criteria"],
                       help="Keyframe selection method")
    parser.add_argument("--dinov3-repo", type=str, default=None,
                       help="Path to DINOv3 repository (if using local)")
    parser.add_argument("--weights", type=str, default=None,
                       help="Path to DINOv3 weights file")
    return parser.parse_args()

# === ADAPTIVE KEYFRAME SELECTION STRATEGIES ===

class AdaptiveKeyframeSelector:
    def __init__(self, method='adaptive_threshold'):
        """
        method options:
        - 'adaptive_threshold': Dynamic similarity threshold
        - 'diversity_buffer': Buffer-based diversity selection  
        - 'temporal_spacing': Minimum temporal distance
        - 'multi_criteria': Combined approach (recommended)
        """
        self.method = method
        self.reset()
    
    def reset(self):
        """Reset selector state"""
        self.feature_history = deque(maxlen=20)  # Keep recent features for diversity
        self.last_keyframe_idx = -1
        self.similarity_history = deque(maxlen=50)  # Track similarity trends
        self.keyframe_count = 0
        
    def should_select_keyframe(self, current_feat, frame_idx):
        """Determine if current frame should be a keyframe"""
        
        if self.method == 'adaptive_threshold':
            return self._adaptive_threshold_selection(current_feat, frame_idx)
        elif self.method == 'diversity_buffer':
            return self._diversity_buffer_selection(current_feat, frame_idx)
        elif self.method == 'temporal_spacing':
            return self._temporal_spacing_selection(current_feat, frame_idx)
        elif self.method == 'multi_criteria':
            return self._multi_criteria_selection(current_feat, frame_idx)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _adaptive_threshold_selection(self, current_feat, frame_idx):
        """Dynamic similarity threshold based on recent similarity trends"""
        if len(self.feature_history) == 0:
            self.feature_history.append(current_feat)
            return True
        
        # Compute similarity to last keyframe
        last_keyframe_feat = self.feature_history[-1]
        similarity = cosine_similarity([current_feat], [last_keyframe_feat])[0][0]
        self.similarity_history.append(similarity)
        
        # Adaptive threshold based on recent similarity variance
        if len(self.similarity_history) > 10:
            recent_sims = list(self.similarity_history)[-10:]
            sim_std = np.std(recent_sims)
            sim_mean = np.mean(recent_sims)
            
            # Dynamic threshold: lower when similarities are stable, higher when variable
            base_threshold = 0.93
            adaptive_threshold = base_threshold - (sim_std * 2)  # Adjust multiplier as needed
            adaptive_threshold = max(0.85, min(0.98, adaptive_threshold))  # Clamp
        else:
            adaptive_threshold = 0.93
        
        should_select = similarity < adaptive_threshold
        if should_select:
            self.feature_history.append(current_feat)
            self.last_keyframe_idx = frame_idx
            
        return should_select
    
    def _diversity_buffer_selection(self, current_feat, frame_idx):
        """Select based on diversity to recent keyframes buffer"""
        if len(self.feature_history) == 0:
            self.feature_history.append(current_feat)
            return True
        
        # Compute minimum similarity to all recent keyframes
        similarities = []
        for hist_feat in self.feature_history:
            sim = cosine_similarity([current_feat], [hist_feat])[0][0]
            similarities.append(sim)
        
        min_similarity = min(similarities)
        
        # Select if sufficiently different from all recent keyframes
        diversity_threshold = 0.90
        should_select = min_similarity < diversity_threshold
        
        if should_select:
            self.feature_history.append(current_feat)
            self.last_keyframe_idx = frame_idx
            
        return should_select
    
    def _temporal_spacing_selection(self, current_feat, frame_idx):
        """Ensure minimum temporal distance between keyframes"""
        min_frame_distance = 10  # Minimum frames between keyframes
        
        if self.last_keyframe_idx == -1:
            self.feature_history.append(current_feat)
            self.last_keyframe_idx = frame_idx
            return True
        
        # Check temporal distance
        if frame_idx - self.last_keyframe_idx < min_frame_distance:
            return False
        
        # Check similarity
        last_feat = self.feature_history[-1]
        similarity = cosine_similarity([current_feat], [last_feat])[0][0]
        
        similarity_threshold = 0.92
        should_select = similarity < similarity_threshold
        
        if should_select:
            self.feature_history.append(current_feat)
            self.last_keyframe_idx = frame_idx
            
        return should_select
    
    def _multi_criteria_selection(self, current_feat, frame_idx):
        """Combined approach using multiple criteria"""
        if len(self.feature_history) == 0:
            self.feature_history.append(current_feat)
            self.last_keyframe_idx = frame_idx
            return True
        
        # Criteria 1: Minimum temporal distance
        min_distance = 5
        temporal_ok = (frame_idx - self.last_keyframe_idx) >= min_distance
        
        # Criteria 2: Similarity to last keyframe
        last_feat = self.feature_history[-1]
        last_similarity = cosine_similarity([current_feat], [last_feat])[0][0]
        
        # Criteria 3: Diversity to recent keyframes (if we have enough)
        if len(self.feature_history) > 1:
            recent_similarities = []
            for hist_feat in list(self.feature_history)[-5:]:  # Last 5 keyframes
                sim = cosine_similarity([current_feat], [hist_feat])[0][0]
                recent_similarities.append(sim)
            min_recent_sim = min(recent_similarities)
        else:
            min_recent_sim = last_similarity
        
        # Adaptive thresholds based on keyframe density
        if self.keyframe_count < 20:
            # Early phase: be more selective
            sim_threshold = 0.95
            diversity_threshold = 0.92
        elif self.keyframe_count < 100:
            # Middle phase: balanced selection
            sim_threshold = 0.93
            diversity_threshold = 0.90
        else:
            # Later phase: ensure coverage
            sim_threshold = 0.91
            diversity_threshold = 0.88
        
        # Decision logic
        should_select = (
            temporal_ok and 
            (last_similarity < sim_threshold or min_recent_sim < diversity_threshold)
        )
        
        # Force selection if too much time has passed
        if (frame_idx - self.last_keyframe_idx) > 50:  # Force every 50 frames max
            should_select = True
        
        if should_select:
            self.feature_history.append(current_feat)
            self.last_keyframe_idx = frame_idx
            self.keyframe_count += 1
            
        return should_select

# === MAIN PROCESSING ===

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load DINOv3 model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load model
    if args.dinov3_repo and args.weights:
        model = torch.hub.load(args.dinov3_repo, "dinov3_vitl16", source='local',
                              weights=args.weights)
    else:
        # Use pretrained from torch hub
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.to(device)
    model.eval()

    def extract_dinov3_feature(img_bgr):
        """Extract DINOv3 features from BGR image"""
        image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(image_tensor)

        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().squeeze()

    # Load images
    extensions = ('.png', '.jpg', '.jpeg')
    image_paths = sorted(
        [os.path.join(args.input, f) for f in os.listdir(args.input)
         if f.lower().endswith(extensions)],
        key=lambda x: int(os.path.basename(x).split(".")[0])
    )

    print(f"Found {len(image_paths)} images")

    # Choose selection method
    selector = AdaptiveKeyframeSelector(method=args.method)

    # Process all frames
    selected_keyframes = []
    start_time = time.time()

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            continue

        feat = extract_dinov3_feature(img)

        if selector.should_select_keyframe(feat, i):
            save_path = os.path.join(args.output, f"keyframe_{len(selected_keyframes):06d}.jpg")
            cv2.imwrite(save_path, img)
            selected_keyframes.append((i, img_path, save_path))

            print(f"Selected keyframe {len(selected_keyframes):3d}: frame {i:4d} -> {save_path}")

        # Progress update
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            fps = (i + 1) / elapsed
            print(f"Processed {i+1}/{len(image_paths)} frames ({fps:.1f} fps), "
                  f"selected {len(selected_keyframes)} keyframes")

    # Final statistics
    total_time = time.time() - start_time
    reduction_ratio = len(selected_keyframes) / len(image_paths) if image_paths else 0

    print(f"\n=== KEYFRAME SELECTION COMPLETE ===")
    print(f"Total frames processed: {len(image_paths)}")
    print(f"Keyframes selected: {len(selected_keyframes)}")
    print(f"Reduction ratio: {reduction_ratio:.1%} (kept {reduction_ratio:.1%} of frames)")
    if total_time > 0:
        print(f"Processing time: {total_time:.1f} seconds ({len(image_paths)/total_time:.1f} fps)")
    print(f"Output directory: {args.output}")

    # Save selection log
    with open(os.path.join(args.output, "selection_log.txt"), 'w') as f:
        f.write("Keyframe Selection Log\n")
        f.write("=====================\n")
        f.write(f"Method: {selector.method}\n")
        f.write(f"Total frames: {len(image_paths)}\n")
        f.write(f"Selected keyframes: {len(selected_keyframes)}\n")
        f.write(f"Reduction ratio: {reduction_ratio:.1%}\n")
        f.write(f"\nSelected frames:\n")
        for i, (frame_idx, orig_path, save_path) in enumerate(selected_keyframes):
            f.write(f"{i:3d}: frame {frame_idx:4d} ({os.path.basename(orig_path)})\n")


if __name__ == '__main__':
    main()
