import numpy as np
import cv2
import random
from utility import rodriguez_to_mat, compose_extrinsics, find_camera_location

def solve_pnp_and_locations(common_matches, cloud0, inliers0, kp_left1, K, tvec0):
    idx_map = {m.queryIdx: i for i, m in enumerate(inliers0)}
    success = False
    attempts = 0
    while not success and attempts < 100:
        # Randomly sample 4 matches from common_matches
        sample = random.sample(common_matches, 4)
        pts3D = np.array([cloud0[idx_map[m.queryIdx]] for m in sample], dtype=np.float32)
        pts2D = np.array([kp_left1[m.trainIdx].pt for m in sample], dtype=np.float32)

        # Solve PnP to get the rotation and translation vectors
        success, rvec, tvec = cv2.solvePnP(pts3D, pts2D, K, None, flags=cv2.SOLVEPNP_P3P)
        attempts += 1
    
    if not success:
        raise RuntimeError("Failed to solve PnP after 100 attempts")
    extrinsic_matrix_l1 =  rodriguez_to_mat(rvec, tvec)
    location_l1 = find_camera_location(extrinsic_matrix_l1)
    relative_extrinsic_matrix_r1 = np.hstack((np.eye(3), tvec0.reshape(3, 1)))
    extrinsic_matrix_r1 = compose_extrinsics(extrinsic_matrix_l1, relative_extrinsic_matrix_r1)
    location_r1 = find_camera_location(extrinsic_matrix_r1)
    return extrinsic_matrix_l1, extrinsic_matrix_r1, location_l1, location_r1
