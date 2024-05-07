import os
from PIL import Image
import numpy as np
import open3d as o3d
import cv2
from scipy.linalg import svd
import kornia
import torch
from unidepth.utils.visualization import save_file_ply

from os import listdir
from os.path import join

def apply_intrinsics(points, intrinsics):
    my_intrinsics = {'fx': intrinsics[0][0],
                    'fy': intrinsics[1][1],
                    'cx': intrinsics[0][2],
                    'cy': intrinsics[1][2], }
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    z = z  
    x = (x - my_intrinsics['cx']) * z / my_intrinsics['fx']
    y = (y - my_intrinsics['cy']) * z / my_intrinsics['fy']
    final = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))
    return final

def estimate_initial_transform(source_points, target_points):
    Rt, scale = cv2.estimateAffine3D(source_points, target_points, force_rotation=True)
    print(scale)
    return Rt, scale

def compose_transform_matrix(Rt):
    row_to_add = np.array([0, 0, 0, 1])
    transform_matrix = np.vstack([Rt, row_to_add])
    return transform_matrix

def pairwise_pose(img1_path, img2_path, model):
    img1 = kornia.io.load_image(img1_path, kornia.io.ImageLoadType.RGB32)[None, ...]
    img2 = kornia.io.load_image(img2_path, kornia.io.ImageLoadType.RGB32)[None, ...]
    matcher = kornia.feature.LoFTR(pretrained="indoor")
    input_dict = {
        "image0": kornia.color.rgb_to_grayscale(img1),  # LofTR works on grayscale images only
        "image1": kornia.color.rgb_to_grayscale(img2),
    }
    
    with torch.inference_mode():
        correspondences = matcher(input_dict)

    mkpts1 = correspondences["keypoints0"].cpu().numpy()
    mkpts2 = correspondences["keypoints1"].cpu().numpy()

    rgb = np.array(Image.open(img1_path).convert('RGB'))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    predictions = model.infer(rgb_torch)
    depth1 = predictions["depth"].squeeze().cpu().numpy()
    
    H, W = rgb.shape[:2]
    points = predictions["points"].squeeze().cpu().permute(1,2,0).numpy().reshape(H*W, 3)
    new_rgb = rgb.reshape(H*W, 3)
    file = "ply2/im1.ply"
    save_file_ply(points, new_rgb, file)
    intrinsics1 = predictions["intrinsics"].squeeze().cpu().numpy()
    torch_intrinsic1 = predictions["intrinsics"]
    torch_pts1 = predictions["points"]

    rgb = np.array(Image.open(img2_path).convert('RGB'))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    predictions = model.infer(rgb_torch)
    depth2 = predictions["depth"].squeeze().cpu().numpy()

    H, W = rgb.shape[:2]
    points = predictions["points"].squeeze().cpu().permute(1,2,0).numpy().reshape(H*W, 3)
    new_rgb = rgb.reshape(H*W, 3)
    file = "ply2/im2.ply"
    save_file_ply(points, new_rgb, file)
    intrinsics2 = predictions["intrinsics"].squeeze().cpu().numpy()

    corresponding_points_3d_img1 = np.zeros((len(mkpts1), 3))
    for i, (x, y) in enumerate(mkpts1):
        depth = depth1[int(y), int(x)]
        corresponding_points_3d_img1[i] = [x, y, depth]
    
    corresponding_points_3d_img2 = np.zeros((len(mkpts2), 3))
    for i, (x, y) in enumerate(mkpts2):
        depth = depth2[int(y), int(x)]
        corresponding_points_3d_img2[i] = [x, y, depth]
    
    cp1 = apply_intrinsics(corresponding_points_3d_img1, intrinsics1)
    cp2 = apply_intrinsics(corresponding_points_3d_img2, intrinsics2)

    Rt, scale = estimate_initial_transform(cp2, cp1)

    initial_transform_matrix = compose_transform_matrix(Rt)

    return initial_transform_matrix, scale, torch_intrinsic1, torch_pts1

def create_depth_map(pointcloud, intrinsics_matrix, image_shape):
    # Project 3D points onto image plane
    projected_points, _ = cv2.projectPoints(pointcloud, np.eye(3), np.zeros(3), intrinsics_matrix, None)
    H, W = image_shape
    # Initialize depth map and count map
    depth_map = [[[] for _ in range(W)] for _ in range(H)]


    # Aggregate depth values that project to the same pixel
    for i, point in enumerate(projected_points):
        x, y = point.ravel()
        z = pointcloud[i, 2]

        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            depth_map[int(y)][int(x)].append(z)
    
    final_depth = np.zeros(image_shape)

    for i in range(H):
        for j in range(W):
            data = np.array(depth_map[i][j])
            if len(data) == 0: 
                final_depth[i][j] = 0
                continue
            median = np.median(data)
            final_depth[i][j] = median

    return final_depth
