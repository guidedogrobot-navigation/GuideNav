"""
Relative pose estimation with reloc3r.

reloc3r lives in its own upstream repository, which must be cloned into
``methods/reloc3r`` next to this file (see the Setup section of the README).
Imports are performed lazily so that importing this module does not require
reloc3r to be present.
"""

import os
import sys

import cv2
import numpy as np
import torch

# Directory holding the reloc3r repo, resolved relative to this file so the
# code is portable across machines.
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_METHODS_DIR = os.path.join(_CURRENT_DIR, 'methods')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_depth_image(path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # Preserve depth precision
    if depth is None:
        raise ValueError(f"Cannot read depth image {path}.")
    return depth


def _import_reloc3r():
    """Import reloc3r, raising a setup-oriented error if it is not cloned."""
    path = os.path.join(_METHODS_DIR, 'reloc3r')
    if path not in sys.path:
        sys.path.append(path)
    try:
        from reloc3r.utils.image import load_images_reloc3r, check_images_shape_format
        from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model, inference_relpose
        from reloc3r.utils.device import to_numpy
    except ImportError as e:
        raise ImportError(
            f"reloc3r must be cloned into {path} and patched with "
            f"load_images_reloc3r.\n"
            f"See the README (Setup) for the exact steps.\n"
            f"Original error: {e}"
        ) from e
    return (load_images_reloc3r, check_images_shape_format,
            setup_reloc3r_relpose_model, inference_relpose, to_numpy)


def init_reloc3r():
    *_, setup_reloc3r_relpose_model, _, _ = _import_reloc3r()
    img_reso = '512' # or 224
    reloc3r_relpose = setup_reloc3r_relpose_model(model_args=img_reso, device=device)
    return reloc3r_relpose, img_reso


# direct inference of relative position
def matching_features_reloc3r_inv(img1, img2, model, img_reso):
    (load_images_reloc3r, check_images_shape_format,
     _, inference_relpose, to_numpy) = _import_reloc3r()

    images = load_images_reloc3r([img1, img2], size=int(img_reso))
    images = check_images_shape_format(images, device)

    # Relative pose estimation
    batch = [images[0], images[1]]
    pose2to1 = to_numpy(inference_relpose(batch, model, device, use_amp=True)[0])

    # Normalize translation to unit scale
    pose2to1[0:3, 3] = pose2to1[0:3, 3] / np.linalg.norm(pose2to1[0:3, 3])

    # Extract relative position (camera frame to robot frame)
    x_rel = pose2to1[2, 3]   # Z translation (forward/backward)
    y_rel = -pose2to1[0, 3]  # X translation (right/left)

    # Extract yaw rotation
    yaw = np.arctan2(-pose2to1[0, 2], pose2to1[2, 2])

    return x_rel, y_rel, np.degrees(yaw)
