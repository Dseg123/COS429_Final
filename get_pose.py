import os
from PIL import Image
import numpy as np
import open3d as o3d
import cv2
from scipy.linalg import svd


# These depend on Unity camera intrinsics but I think they are important
intrinsic = o3d.camera.PinholeCameraIntrinsic(720, 480, 1500, 1000, 360, 240)
my_intrinsics = {'depth_scale': 1000,
              'width': 720,
              'height': 480,
              'fx': 1500,
              'fy': 1000,
              'cx': 360,
              'cy': 240}

def apply_intrinsics(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    z = z  # / my_intrinsics['depth_scale']?
    x = (x - my_intrinsics['cx']) * z / my_intrinsics['fx']
    y = (y - my_intrinsics['cy']) * z / my_intrinsics['fy']
    print(x.shape)
    final = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))
    print(final.shape)
    return final


def estimate_initial_transform(source_points, target_points):
    # Center the points
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)
    centered_source = source_points - centroid_source
    centered_target = target_points - centroid_target

    # Singular Value Decomposition
    H = centered_source.T @ centered_target
    U, _, Vt = svd(H)

    # Rotation matrix
    R = Vt.T @ U.T

    # Translation vector
    t = centroid_target - R @ centroid_source

    return R, t

def compose_transform_matrix(rotation_matrix, translation_vector):
    # Create a 4x4 identity matrix
    transform_matrix = np.eye(4)

    # Assign the rotation matrix to the top-left 3x3 block
    transform_matrix[:3, :3] = rotation_matrix

    # Assign the translation vector to the rightmost column
    transform_matrix[:3, 3] = translation_vector

    return transform_matrix


def pairwise_pose(ind_1, ind_2):
    # def get_init_pos2(col1, depth1, col2, depth2):
    img_paths = os.listdir('data/images')
    depth_paths = os.listdir('data/mono_depths')
    img_path_1 = 'data/images/' + img_paths[ind_1]
    img_path_2 = 'data/images/' + img_paths[ind_2]
    depth_path_1 = 'data/mono_depths/' + depth_paths[ind_1]
    depth_path_2 = 'data/mono_depths/' + depth_paths[ind_2]



    # Assuming you have two grayscale images: img1 and img2
    img1 = np.asarray(Image.open(img_path_1))
    img2 = np.asarray(Image.open(img_path_2))
    depth1 = np.load(depth_path_1)
    depth2 = np.load(depth_path_2)


    # Step 1: Detect ORB keypoints and descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Step 2: Match descriptors using a matcher (e.g., BFMatcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    print("Matches:", matches)

    # Step 3: Filter matches based on a distance threshold
    distance_threshold = 50  # Adjust as needed
    good_matches = [match for match in matches if match.distance < distance_threshold]

    # Step 4: Extract corresponding 3D points from the matched keypoints
    corresponding_points_img1 = np.array([keypoints1[match.queryIdx].pt for match in good_matches])
    corresponding_points_img2 = np.array([keypoints2[match.trainIdx].pt for match in good_matches])

    # print(corresponding_points_img1)
    # print(corresponding_points_img2)

    # Convert 2D points to homogeneous 3D points by adding depth information (from depth images)
    # Use camera calibration parameters if available
    depth_img1 = depth1  # Depth image corresponding to img1
    depth_img2 = depth2  # Depth image corresponding to img2

    # # Assuming (u, v) are the pixel coordinates of the keypoints
    corresponding_points_3d_img1 = np.hstack((corresponding_points_img1, depth_img1[corresponding_points_img1[:, 1].astype(int), corresponding_points_img1[:, 0].astype(int)].reshape(-1, 1)))
    corresponding_points_3d_img2 = np.hstack((corresponding_points_img2, depth_img2[corresponding_points_img2[:, 1].astype(int), corresponding_points_img2[:, 0].astype(int)].reshape(-1, 1)))
    print('Before intrinsics:', corresponding_points_3d_img1)
    print('After intrinsics:', apply_intrinsics(corresponding_points_3d_img1, my_intrinsics))
    
    # might need to multiply by a coefficient here... intrinsics are weird
    cp1 = apply_intrinsics(corresponding_points_3d_img1, my_intrinsics)
    cp2 = apply_intrinsics(corresponding_points_3d_img2, my_intrinsics)


    print(cp1)
    print(cp2)


    # print(corresponding_points_3d_img1)
    # print(corresponding_points_3d_img2)
    # # Optionally, you might want to further filter or refine the correspondences

    # # Now, use the corresponding 3D points for initial transformation estimation
    initial_rotation, initial_translation = estimate_initial_transform(cp1, cp2)

    # Continue with ICP or other alignment methods
    # print(initial_rotation)
    # print(initial_translation)

    initial_transform_matrix = compose_transform_matrix(initial_rotation, initial_translation)
    return initial_transform_matrix