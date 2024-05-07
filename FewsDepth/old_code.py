
def pairwise_pose2():
    img1 = np.array(Image.open('pair2/im0.png'))
    img2 = np.array(Image.open('pair2/im1.png'))
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

    # Convert 2D points to homogeneous 3D points by adding depth information (from depth images)
    # Use camera calibration parameters if available

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

    depth_img1 = depth1  # Depth image corresponding to img1
    depth_img2 = depth2  # Depth image corresponding to img2

    # # Assuming (u, v) are the pixel coordinates of the keypoints

    corresponding_points_3d_img1 = np.hstack((corresponding_points_img1, depth_img1[corresponding_points_img1[:, 1].astype(int), corresponding_points_img1[:, 0].astype(int)].reshape(-1, 1)))
    corresponding_points_3d_img2 = np.hstack((corresponding_points_img2, depth_img2[corresponding_points_img2[:, 1].astype(int), corresponding_points_img2[:, 0].astype(int)].reshape(-1, 1)))
    print('Before intrinsics:', corresponding_points_3d_img1)
    print('After intrinsics:', apply_intrinsics(corresponding_points_3d_img1, intrinsics1))
    
    cp1 = apply_intrinsics(corresponding_points_3d_img1, intrinsics1)
    cp2 = apply_intrinsics(corresponding_points_3d_img2, intrinsics2)

    Rt = estimate_initial_transform(cp1, cp2)
    initial_transform_matrix = compose_transform_matrix(Rt)
    print(initial_transform_matrix)
    return initial_transform_matrix

def estimate_initial_transform2(source_points, target_points):
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