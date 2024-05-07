import os
from PIL import Image
import numpy as np
import open3d as o3d
import cv2
from scipy.linalg import svd
import kornia
import torch
from unidepth.utils import colorize, image_grid
from unidepth.models import UniDepthV1
from unidepth.utils.visualization import save_file_ply

from os import listdir
from os.path import join
from unidepth.utils.geometric import project_points
from poselib import estimate_relative_pose


# These depend on Unity camera intrinsics but I think they are important

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
    print(x.shape)
    final = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))
    print(final.shape)
    return final


def estimate_initial_transform(source_points, target_points):
    Rt, _ = cv2.estimateAffine3D(source_points, target_points, force_rotation=True)
    return Rt

def compose_transform_matrix(Rt):
    row_to_add = np.array([0, 0, 0, 1])
    transform_matrix = np.vstack([Rt, row_to_add])
    return transform_matrix

def pairwise_pose(img1, img2):
    matcher = kornia.feature.LoFTR(pretrained="indoor")

    input_dict = {
        "image0": kornia.color.rgb_to_grayscale(img1),  # LofTR works on grayscale images only
        "image1": kornia.color.rgb_to_grayscale(img2),
    }
    
    with torch.inference_mode():
        correspondences = matcher(input_dict)

    mkpts1 = correspondences["keypoints0"].cpu().numpy()
    mkpts2 = correspondences["keypoints1"].cpu().numpy()

    model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #depth1 = np.load('im1depth.npy')
    #depth2 = np.load('im2depth.npy')

    rgb = np.array(Image.open('pair2/im0.png'))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    predictions = model.infer(rgb_torch)
    depth1 = predictions["depth"].squeeze().cpu().numpy()
    
    np.save('im1depth', depth1)
    H, W = rgb.shape[:2]
    points = predictions["points"].squeeze().cpu().permute(1,2,0).numpy().reshape(H*W, 3)
    new_rgb = rgb.reshape(H*W, 3)
    file = "ply2/im1.ply"
    save_file_ply(points, new_rgb, file)
    intrinsics1 = predictions["intrinsics"].squeeze().cpu().numpy()
    torch_intrinsic1 = predictions["intrinsics"]
    torch_pts1 = predictions["points"]

    depth_pred_col = colorize(depth1, vmin=0.01, vmax=10.0, cmap="magma_r")
    Image.fromarray(depth_pred_col).save("original.png")


    rgb = np.array(Image.open('pair2/im1.png'))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    predictions = model.infer(rgb_torch)
    depth2 = predictions["depth"].squeeze().cpu().numpy()
    np.save('im2depth', depth2)
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

    print('Before intrinsics:', corresponding_points_3d_img1)
    print('After intrinsics:', apply_intrinsics(corresponding_points_3d_img1, intrinsics1))
    
    cp1 = apply_intrinsics(corresponding_points_3d_img1, intrinsics1)
    cp2 = apply_intrinsics(corresponding_points_3d_img2, intrinsics2)

    Rt = estimate_initial_transform(cp2, cp1)
   
  #  camera1 = {'model': 'SIMPLE_PINHOLE', 'width': 736, 'height': 468, 'params': [intrinsics1[0][0], intrinsics1[1][1],  intrinsics1[0][2],  intrinsics1[1][2]]}
  #  camera2 = {'model': 'SIMPLE_PINHOLE', 'width': 736, 'height': 468, 'params': [intrinsics2[0][0], intrinsics2[1][1],  intrinsics2[0][2],  intrinsics2[1][2]]}

  #  Rt2 = estimate_relative_pose(mkpts2, mkpts1, camera2, camera1)

    initial_transform_matrix = compose_transform_matrix(Rt)
    print(initial_transform_matrix)
    return initial_transform_matrix, torch_intrinsic1, torch_pts1


img1 = kornia.io.load_image('pair2/im0.png', kornia.io.ImageLoadType.RGB32)[None, ...]
img2 = kornia.io.load_image('pair2/im1.png', kornia.io.ImageLoadType.RGB32)[None, ...]
    
_ , _ , H,W = img1.size()

T, intrinsics1, points1 = pairwise_pose(img1,img2)
pcd1 = o3d.io.read_point_cloud("ply2/im1.ply")
pcd2 = o3d.io.read_point_cloud("ply2/im2.ply")
pcd2 = pcd2.transform(T)


#pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

#reg_p2p = o3d.pipelines.registration.registration_icp(
#    pcd2, pcd1, 5, np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]),
#    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))

#pcd2 = pcd2.transform(reg_p2p.transformation)

o3d.io.write_point_cloud("ply2/im2new.ply", pcd2)

pcd_comb = pcd1 

points = np.asarray(pcd_comb.points)
points_tensor = torch.tensor(points, dtype=torch.float)
points_tensor = points_tensor.view(-1, 3).unsqueeze(0).cuda()

#print(points_tensor)
depth = project_points(points_tensor, intrinsics1, (H,W))
print(depth.shape)
depth_pred_col = colorize(depth.squeeze().cpu().numpy(), vmin=0.01, vmax=10.0, cmap="magma_r")
Image.fromarray(depth_pred_col).save("averaged.png")
