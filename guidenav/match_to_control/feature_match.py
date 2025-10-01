import numpy as np
import random
import cv2
from math import atan2
from PIL import Image
from romatch import roma_outdoor
from romatch import tiny_roma_v1_outdoor
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys
# sys.path.append('/home/orin2/Repository/GuideNav/src/guidenav/match_to_control/methods') 
sys.path.append('/home/orin2/Repository/GuideNav/src/guidenav/match_to_control/methods') 
import torch
from LoFTR.src.loftr import LoFTR, default_cfg
from LoFTR.src.config.default import get_cfg_defaults
import pdb
import time
# LiftFeat
# sys.path.append('/home/orin2/Repository/GuideNav/src/guidenav/match_to_control/methods/LiftFeat') 
sys.path.append('/home/orin2/Repository/GuideNav/src/guidenav/match_to_control/methods/LiftFeat') 
from LiftFeat.models.liftfeat_wrapper import LiftFeat,MODEL_PATH

sys.path.append('/home/orin2/Repository/GuideNav/src/guidenav/match_to_control/methods/mast3r') 
sys.path.append('/home/orin2/Repository/GuideNav/src/guidenav/match_to_control/methods/mast3r/dust3r')  # Add this line
# sys.path.append('/home/orin2/Repository/GuideNav/src/guidenav/match_to_control/methods/mast3r/dust3r/dust3r')  # Add this line

import dust3r.utils.path_to_croco  # noqa: F401
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

# import sys
# sys.path.append('/home/orin2/Repository/GuideNav/src/guidenav/match_to_control/methods/mast3r')
# sys.path.append('/home/orin2/Repository/GuideNav/src/guidenav/match_to_control/methods/mast3r/dust3r')
# sys.path.append('/home/orin2/Repository/GuideNav/src/guidenav/match_to_control/methods/mast3r/dust3r/dust3r')
# sys.path.append('/home/orin2/Repository/GuideNav/src/guidenav/match_to_control/methods/mast3r/dust3r/croco')

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images, load_images_online

# Reloc3r (END2END)
sys.path.append('/home/orin2/Repository/GuideNav/src/guidenav/match_to_control/methods/reloc3r') 
from reloc3r.utils.image import parse_video, load_images_reloc3r, check_images_shape_format
from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model, inference_relpose
from reloc3r.utils.device import to_numpy



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_depth_image(path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # Preserve depth precision
    if depth is None:
        raise ValueError(f"Cannot read depth image {path}.")
    return depth

def init_loftr():
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load("/home/orin2/Repository/GuideNav/src/guidenav/match_to_control/methods/LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
    matcher = matcher.eval().to(device=device)
    return matcher
    
def matching_features_loftr(img1, img2, matcher):
    
    # Preprocess images
    # img1_tensor = loftr_utils_preprocess(img1, resize=(640, 480), device=device)
    # img2_tensor = loftr_utils_preprocess(img2, resize=(640, 480), device=device)
    img1_tensor = loftr_utils_preprocess(img1, resize=(640, 360), device=device)
    img2_tensor = loftr_utils_preprocess(img2, resize=(640, 360), device=device)
    
    batch = {'image0': img1_tensor, 'image1': img2_tensor}
    with torch.no_grad():
        matcher(batch)
        
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    
    kp1 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in mkpts0]
    kp2 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in mkpts1]
    
    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(mkpts0))]

    return kp1, kp2, matches

def process_resize(w, h, resize):
    if len(resize) == 2:
        return resize
    elif len(resize) == 1 and resize[0] > 0:
        max_dim = resize[0]
        scale = max_dim / max(w, h)
        return int(w * scale), int(h * scale)
    elif len(resize) == 1 and resize[0] == -1:
        return w, h  
    else:
        raise ValueError("Invalid resize config")

def loftr_utils_preprocess(img: np.ndarray, resize=(640, 480), device='cuda'):
    w, h = img.shape[1], img.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    img_resized = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(gray / 255.).float()[None, None].to(device)

def init_roma():
    roma_model = roma_outdoor(device=device)
    return roma_model

def matching_features_roma(img1_path, img2_path, roma_model):
    W_1, H_1 = Image.open(img1_path).size
    W_2, H_2 = Image.open(img2_path).size

    # start_1_time_roma = time.time()
    # roma_model = roma_outdoor(device=device)
    # end_1_time_roma = time.time()
    # print(f"[INFO] ROMA model loading time: {end_1_time_roma - start_1_time_roma:.4f} seconds")

    # Match
    start_time_inner_roma = time.time()
    warp, certainty = roma_model.match(img1_path, img2_path)
    end_time_inner_roma = time.time()
    print(f"[INFO] ROMA inner matching time: {end_time_inner_roma - start_time_inner_roma:.4f} seconds")
    matches_raw, certainty = roma_model.sample(warp, certainty)
    top_k = int(0.2 * matches_raw.shape[0])  # top 20%, 50, 80
    indices = torch.topk(certainty, top_k).indices
    matches_raw = matches_raw[indices]

    # convert to pixel coord (roma generates in [-1, 1] x [-1, 1])
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches_raw, H_1, W_1, H_2, W_2)

    kpts1 = kpts1.cpu().numpy()
    kpts2 = kpts2.cpu().numpy()
    
    kp1 = [cv2.KeyPoint(float(x), float(y), size=1) for x, y in kpts1]
    kp2 = [cv2.KeyPoint(float(x), float(y), size=1) for x, y in kpts2]

    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(kp1))]
    print(f"ROMA matches: {len(matches)}")

    return kp1, kp2, matches

def init_mast3r():
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    matcher = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    return matcher
    
def matching_features_mast3r(img1, img2, model):
    
    # Preprocess images
    # img1_tensor = loftr_utils_preprocess(img1, resize=(640, 480), device=device)
    # img2_tensor = loftr_utils_preprocess(img2, resize=(640, 480), device=device)
    images = [img1, img2]
    # pdb.set_trace()
    # processed_images = load_images([str(p) for p in images], size=512)
    processed_images = load_images_online(images, size=512)

    # pdb.set_trace()
    # start_mastr_inf_time = time.time()
    output = inference([tuple(processed_images)], model, device, batch_size=1, verbose=False)
    # end_mastr_inf_time = time.time()
    # print(f"[INFO] MAST3R INF time: {end_mastr_inf_time - start_mastr_inf_time}")
    
    # pdb.set_trace()
    # Extract predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    
    # Extract descriptors
    desc1 = pred1['desc'].squeeze(0).detach()
    desc2 = pred2['desc'].squeeze(0).detach()

    # Find 2D-2D matches between the two images
    # These coordinates are already in the original image coordinate system
    # start_mastr_mat_time = time.time()
    matches_im0, matches_im1 = fast_reciprocal_NNs(
        desc1, desc2, 
        subsample_or_initxy1=8,
        device=device, 
        dist='dot', 
        block_size=2**13
    )
    # end_mastr_mat_time = time.time()
    # print(f"[INFO] MAST3R MATCHING time: {end_mastr_mat_time - start_mastr_mat_time}")
    
    # Get original image dimensions (true_shape contains original dimensions)
    H0, W0 = view1['true_shape'][0]
    H1, W1 = view2['true_shape'][0]
    
    # Filter out matches near borders
    border = 3
    valid_matches_im0 = (
        (matches_im0[:, 0] >= border) & (matches_im0[:, 0] < int(W0) - border) & 
        (matches_im0[:, 1] >= border) & (matches_im0[:, 1] < int(H0) - border)
    )
    
    valid_matches_im1 = (
        (matches_im1[:, 0] >= border) & (matches_im1[:, 0] < int(W1) - border) & 
        (matches_im1[:, 1] >= border) & (matches_im1[:, 1] < int(H1) - border)
    )
    
    valid_matches = valid_matches_im0 & valid_matches_im1
    final_matches_im0 = matches_im0[valid_matches]
    final_matches_im1 = matches_im1[valid_matches]
    
    # Convert to cv2.KeyPoint format
    kp1 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in final_matches_im0]
    kp2 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in final_matches_im1]
    
    # Create cv2.DMatch objects
    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(final_matches_im0))]
    
    return kp1, kp2, matches

def init_liftFeat():
    matcher=LiftFeat(weight=MODEL_PATH,detect_threshold=0.35)
    return matcher

def matching_features_liftFeat(img1, img2, model):
    mkpts0,mkpts1=model.match_liftfeat(img1,img2)
    
    # Apply RANSAC filtering
    if len(mkpts0) > 4:
        H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
        # H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 5.0, maxIters=2_000, confidence=0.95)
        if mask is not None:
            mask = mask.flatten()
            mkpts0 = mkpts0[mask.astype(bool)]
            mkpts1 = mkpts1[mask.astype(bool)]

    kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in mkpts0]
    kp2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in mkpts1]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mkpts0))]

    return kp1, kp2, matches

def init_reloc3r():
    img_reso = '512' # or 224
    reloc3r_relpose = setup_reloc3r_relpose_model(model_args=img_reso, device=device)
    return reloc3r_relpose, img_reso

# direct inference of relative position
def matching_features_reloc3r_inv(img1, img2, model, img_reso):
    images = load_images_reloc3r([img1, img2], size=int(img_reso))
    images = check_images_shape_format(images, device)

    # relpose
    batch = [images[0], images[1]]
    start_time_inner_reloc3r = time.time()
    pose2to1 = to_numpy(inference_relpose(batch, model, device, use_amp=True)[0])
    # pose2to1 = to_numpy(inference_relpose(batch, model, device, use_amp=False)[0])
    end_time_inner_reloc3r = time.time()
    # pose2to1[0:3,3] = pose2to1[0:3,3] / np.linalg.norm(pose2to1[0:3,3])  # normalize the scale to 1 meter
    print(f"[INFO] Reloc3 inner matching time: {end_time_inner_reloc3r - start_time_inner_reloc3r:.4f} seconds")

    # T is obs to target: how obs cam looks from target
    # pose1to2 = np.linalg.inv(pose2to1) # target relative to obs
    

    # Convert to SE(2)
    # x_rel = pose2to1[2, 3]   # forward (Z in cam)
    # y_rel = -pose2to1[0, 3]  # left (X in cam = -Y in robot)
    # yaw = np.arctan2(pose2to1[0, 2], pose2to1[2, 2])  # camera yaw

    # print(f"[Reloc3r] Pose est: {pose2to1}")

    
    pose2to1[0:3,3] = pose2to1[0:3,3] / np.linalg.norm(pose2to1[0:3,3])

    # Extract relative position of image 2 from image 1
    # pose2to1 gives us how camera 2 is positioned relative to camera 1
    # y_rel = pose2to1[1, 3]  # Y translation (up/down) 
    x_rel = pose2to1[2, 3]  # Z translation (forward/backward)
    y_rel = -pose2to1[0, 3]  # X translation (right/left)

    # Extract yaw rotation (rotation around Y-axis, assuming camera looks along Z)
    yaw = np.arctan2(-pose2to1[0, 2], pose2to1[2, 2])

    return x_rel, y_rel, np.degrees(yaw)
def main():
    # obs_img_pth = "/home/hochul/Downloads/data_go2/preprocessed/color/1480.png" # 1450
    # obs_dep_pth = "/home/hochul/Downloads/data_go2/preprocessed/depth/1480.png"
    # top_img_pth = "/home/hochul/Downloads/data_go2/preprocessed/color/1500.png"
    # top_dep_pth = "/home/hochul/Downloads/data_go2/preprocessed/depth/1500.png"
    
    # obs_img_pth = "/home/hochul/Downloads/data_go2/preprocessed/color/0.png"
    # obs_dep_pth = "/home/hochul/Downloads/data_go2/preprocessed/depth/0.png"
    # top_img_pth = "/home/hochul/Downloads/data_go2/preprocessed/color/50.png"
    # top_dep_pth = "/home/hochul/Downloads/data_go2/preprocessed/depth/50.png"

    # Here Syn - works
    # top_img_pth = "/media/2t/relpose/syn/000000_lcam_front.png"
    # top_dep_pth = "/media/2t/relpose/syn/000000_lcam_front_depth.png"
    # obs_img_pth = "/media/2t/relpose/syn/000002_lcam_front.png"
    # obs_dep_pth = "/media/2t/relpose/syn/000002_lcam_front_depth.png"
    # top_img_pth = "/home/hochul/Repository/Tools/sample_data/tartanair/ArchVizTinyHouseDay/Data_easy/P000/image_lcam_front/000000_lcam_front.png"
    # top_dep_pth = "/home/hochul/Repository/Tools/sample_data/tartanair/ArchVizTinyHouseDay/Data_easy/P000/depth_lcam_front/000000_lcam_front_depth.png"
    # obs_img_pth = "/home/hochul/Repository/Tools/sample_data/tartanair/ArchVizTinyHouseDay/Data_easy/P000/image_lcam_front/000002_lcam_front.png"
    # obs_dep_pth = "/home/hochul/Repository/Tools/sample_data/tartanair/ArchVizTinyHouseDay/Data_easy/P000/depth_lcam_front/000002_lcam_front_depth.png"
    # obs_img_pth = "/home/hochul/Repository/Tools/sample_data/tartanair/ArchVizTinyHouseDay/Data_easy/P000/image_lcam_front/000004_lcam_front.png"
    # obs_dep_pth = "/home/hochul/Repository/Tools/sample_data/tartanair/ArchVizTinyHouseDay/Data_easy/P000/depth_lcam_front/000004_lcam_front_depth.png"
    # obs_img = cv2.imread(obs_img_pth, cv2.IMREAD_COLOR)
    # top_img = cv2.imread(top_img_pth, cv2.IMREAD_COLOR)
    # obs_dep = cv2.imread(obs_dep_pth, cv2.IMREAD_UNCHANGED).astype(np.float32)  # 16-bit PNG
    # top_dep = cv2.imread(top_dep_pth, cv2.IMREAD_UNCHANGED).astype(np.float32)  # not used

    # Real
    top_img_pth = "/media/2t/ijrr/final/02_L2CSFAIL/topomap/topomap/1.jpg"
    obs_img_pth = "/media/2t/ijrr/final/02_L2CSFAIL/observation/obs/0_4.jpg"
    obs_dep_pth = "/media/2t/ijrr/final/02_L2CSFAIL/observation/obs_depth/0_4.png"
    
    obs_img = cv2.imread(obs_img_pth, cv2.IMREAD_COLOR)
    top_img = cv2.imread(top_img_pth, cv2.IMREAD_COLOR)
    obs_dep = read_depth_image(obs_dep_pth)
    if obs_dep.max() > 1000:  # likely in mm
        obs_dep = obs_dep / 1000.0

    fx = 454.102 # 320
    fy = 453.927 # 320
    cx = 325.456 # 320
    cy = 181.126 # 240
    
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)

    kp1, kp2, matches = matching_features_roma(obs_img_pth, top_img_pth)
    # x, y, yaw_deg = estimate_relative_pose_from_essential(kp1, kp2, matches, K)
    x, y, yaw_deg = estimate_relative_pose_from_pnp(kp1, kp2, matches, obs_dep, K)


    # print(f"Estimated relative pose:\n    forward = {x:.3f} m\n    lateral = {y:.3f} m\n    yaw = {yaw_deg:.2f} deg")

    # 1450
    # gt_pose0 = {'x': 17.432792664, 'y': 33.322921753, 'z': -0.100337669,
    #             'qx': 0.205125467, 'qy': 0.008371016, 'qz': -0.537061955, 'qw': 0.818179618}
    # 1480
    # gt_pose0 = {'x': 17.639974594, 'y': 31.918531418, 'z': -0.555517733,
    #             'qx': 0.162791434, 'qy': 0.041651172, 'qz': -0.704905957, 'qw': 0.689109590}
    # 0
    # gt_pose0 = {'x': 17.639974594, 'y': 31.918531418, 'z': -0.555517733,
    #             'qx': 0.162791434, 'qy': 0.041651172, 'qz': -0.704905957, 'qw': 0.689109590}

    # 1500
    # gt_pose1 = {'x': 17.859683990, 'y': 30.391685486, 'z': -1.443254709,
    #             'qx': 0.141535088, 'qy': 0.049727997, 'qz': -0.693597860, 'qw': 0.704568648}
    # 50
    # gt_pose1 = {'x': 17.859683990, 'y': 30.391685486, 'z': -1.443254709,
    #             'qx': 0.141535088, 'qy': 0.049727997, 'qz': -0.693597860, 'qw': 0.704568648}

    # TARTAN AIR
    # 2
    gt_pose0 = {'x': 7.488675713539123535e-01, 'y': 5.103784799575805664e-01, 'z': -8.929691910743713379e-01,
                'qx': 1.192193478345870972e-02, 'qy': 1.985714212059974670e-02,
                'qz': -9.898579120635986328e-01, 'qw': 1.401601582765579224e-01}

    # 5
    # gt_pose0 = {'x': 1.185005784034729004e+00, 'y': 5.535679459571838379e-01, 'z': -9.330671429634094238e-01,
    #             'qx': 9.964801371097564697e-03, 'qy': 3.555731475353240967e-02,
    #             'qz': -9.904983043670654297e-01, 'qw': 1.324747502803802490e-01}
    # 0
    gt_pose1 = {'x': 1.561090797185897827e-01, 'y': 4.587774574756622314e-01, 'z': -8.409901261329650879e-01, 
                'qx': -0.000000000000000000e+00, 'qy': -0.000000000000000000e+00, 
                'qz': -9.915279746055603027e-01, 'qw': 1.298935413360595703e-01}
    # gt_pose0 = {'x': 17.639974594, 'y': 31.918531418, 'z': -0.555517733,
    #             'qx': 0.162791434, 'qy': 0.041651172, 'qz': -0.704905957, 'qw': 0.689109590}

    draw_feature_matches(obs_img, top_img, kp1, kp2, matches)
    plot_relative_pose(gt_pose0, gt_pose1, (x, y, yaw_deg))


def draw_feature_matches(img1, img2, kp1, kp2, matches, max_matches=1000):
    matched_vis = cv2.drawMatches(
        img1, kp1, img2, kp2,
        matches[:max_matches], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    # save
    cv2.imwrite("feature_matches.png", matched_vis)
    # cv2.imshow("Feature Matches", matched_vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


        