import get_pose
import os
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors


def convert_to_pcd(depth):
    base_pixels = np.array((depth.shape[0] * depth.shape[1], 3))
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            base_pixels[i * depth.shape[1] + j, 0] = i
            base_pixels[i * depth.shape[1] + j, 1] = j
            base_pixels[i * depth.shape[1] + j, 2] = depth[i, j]
    
    return get_pose.apply_intrinsics(base_pixels)

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def get_depth_at_point(u, v, ind_base, ind_support, match_thresh):
    T = get_pose.pairwise_pose(ind_base, ind_support)
    
    depth_paths = os.listdir('data/mono_depths')
    depth_path_1 = 'data/mono_depths/' + depth_paths[ind_base]
    depth_path_2 = 'data/mono_depths/' + depth_paths[ind_support]

    # Assuming you have two grayscale images: img1 and img2
    depth1 = np.load(depth_path_1)
    depth2 = np.load(depth_path_2)


    support_pcd = convert_to_pcd(depth2)
    support_pcd_homo = np.hstack((support_pcd, np.ones(support_pcd.shape[0])))
    support_pcd_transform = support_pcd_homo @ T
    for i in range(support_pcd_transform.shape[0]):
        support_pcd_transform[i, :] /= support_pcd_transform[i, 3]
    support_pcd_transform = support_pcd_transform[:, :3]

    base_depth = depth1[u, v]
    base_point = get_pose.apply_intrinsics([u, v, base_depth])
    best_ind = -1
    best_dist = 10000000
    for i in range(support_pcd_transform.shape[0]):
        point = support_pcd_transform[i, :]
        dist = np.sqrt(np.sum(np.square(point - base_point)))
        if dist < match_thresh and dist < best_dist:
            best_ind = i
            best_dist = dist
    
    depths_arr = [base_depth]
    if best_ind != -1:
        depths_arr.append(support_pcd_transform[best_ind, 2])
    
    depths_arr = np.array(depths_arr)
    mean_depth = np.mean(depths_arr)
    if len(depths_arr) > 0:
        std_depth = np.std(depths_arr)
    else:
        std_depth = np.nan
    
    return mean_depth, std_depth

def get_depth_at_points(ind_base, ind_support, match_thresh):
    T = get_pose.pairwise_pose(ind_base, ind_support)
    
    depth_paths = os.listdir('data/mono_depths')
    depth_path_1 = 'data/mono_depths/' + depth_paths[ind_base]
    depth_path_2 = 'data/mono_depths/' + depth_paths[ind_support]

    # Assuming you have two grayscale images: img1 and img2
    depth1 = np.load(depth_path_1)
    depth2 = np.load(depth_path_2)


    support_pcd = convert_to_pcd(depth2)
    support_pcd_homo = np.hstack((support_pcd, np.ones(support_pcd.shape[0])))
    support_pcd_transform = support_pcd_homo @ T
    for i in range(support_pcd_transform.shape[0]):
        support_pcd_transform[i, :] /= support_pcd_transform[i, 3]
    support_pcd_transform = support_pcd_transform[:, :3]

    base_pcd = convert_to_pcd(depth1)
    dists, inds = nearest_neighbor(base_pcd, support_pcd_transform)

    depth_map = np.zeros_like(depth1)
    std_map = np.zeros_like(depth1)
    for u in range(depth_map.shape[0]):
        for v in range(depth_map.shape[1]):
            base_point = base_pcd[u*depth_map.shape[1] + v, :]
            depths_arr = [base_point[2]]

            dist = dists[u*depth_map.shape[1] + v]
            ind = inds[u*depth_map.shape[1] + v]
            support_point = support_pcd_transform[ind, :]

            if dist < match_thresh:
                depths_arr.append(support_point[2])
            
            depths_arr = np.array(depths_arr)
            depth_map[u, v] = np.mean(depths_arr)
            if len(depths_arr) > 1:
                std_map[u, v] = np.std(depths_arr)
            else:
                std_map[u, v] = np.nan
            
    
    return depth_map, std_map


