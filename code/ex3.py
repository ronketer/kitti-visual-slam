from utility import read_cameras, read_images, find_camera_location, find_stereo_temporal_matches, coordinate_transform
import cv2
import numpy as np
import matplotlib.pyplot as plt
from alg import *
from detector_config import detector
from ex3code.q1 import q1
from ex3code.q2 import q2
from ex3code.q3 import q3
from ex3code.q4 import q4
from ex3code.q5 import q5   
from ex3code.q6 import q6
from consts import OUTPUT_RELATIVE_PATH

ORANGE = (0, 165, 255)
CYAN = (255, 255, 0)

def create_visualization_with_matches(img, keypoints, supporters, idx_pos):
    """Create an image with keypoints colored based on supporter status"""
    supporter_indices = [support[idx_pos] for support in supporters]
    supporters_kp = [keypoints[idx] for idx in supporter_indices]
    non_supporter_indices = [i for i in range(len(keypoints)) if i not in supporter_indices]
    non_supporters_kp = [keypoints[idx] for idx in non_supporter_indices]


    img_with_matches = cv2.drawKeypoints(img, non_supporters_kp, None, color=CYAN, flags=0)
    img_with_matches = cv2.drawKeypoints(img_with_matches, supporters_kp, None, color=ORANGE, flags=0)

    
    # Add legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_with_matches, f'{len(supporters)} Supporters', (10, 30), font, 1, ORANGE, 2)
    cv2.putText(img_with_matches, f'{len(non_supporters_kp)} Non-Supporters', (10, 60), font, 1, CYAN, 2)

    return img_with_matches

def plot_q3(location_l0, location_r0, location_l1, location_r1):
    plt.figure()
    plt.scatter(location_l0[0], location_l0[2], color='r', label='Left Camera 0')
    plt.scatter(location_r0[0], location_r0[2], color='b', label='Right Camera 0')
    plt.scatter(location_l1[0], location_l1[2], color='g', label='Left Camera 1')
    plt.scatter(location_r1[0], location_r1[2], color='y', label='Right Camera 1')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.legend()
    plt.savefig(OUTPUT_RELATIVE_PATH + 'camera_locations.png')

def plot_q4(img_left0, img_left1, kp_left0, kp_left1, supporters):   
    # Create visualizations with matches - use index 0 for left0 and index 2 for left1
    left0_vis = create_visualization_with_matches(img_left0, kp_left0, supporters, 0)
    left1_vis = create_visualization_with_matches(img_left1, kp_left1, supporters, 2)
    
    # Show matches visualization
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(left0_vis, cv2.COLOR_BGR2RGB))
    plt.title('Left Image 0')
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(left1_vis, cv2.COLOR_BGR2RGB))
    plt.title('Left Image 1')
    plt.savefig(OUTPUT_RELATIVE_PATH + 'matches_visualization.png')

def plot_q5(cloud0, cloud1, best_extrinsic_l1, img_left0, img_left1, kp_left0, kp_left1, best_inliers, common_matches_indices):
    """Plot the transformed point clouds and inliers/outliers visualization for q5."""
    # Part 1: Plot transformed point clouds
    transformed_cloud0 = coordinate_transform(cloud0, best_extrinsic_l1)
    transformed_cloud0 = transformed_cloud0.T  # Convert to same format as ex2.py (3xN)
    cloud1 = cloud1.T  # Convert to same format as ex2.py (3xN)
    
    # Create 3D visualization of point clouds
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    
    # Plot both clouds with different colors
    ax.scatter(transformed_cloud0[0], transformed_cloud0[2], transformed_cloud0[1], 
              c='red', marker='o', label='Transformed Cloud 0')
    ax.scatter(cloud1[0], cloud1[2], cloud1[1], 
              c='blue', marker='o', label='Cloud 1')
    
    ax.set_xlabel('X-axis (Meters)')
    ax.set_ylabel('Z-axis (Meters)')
    ax.set_zlabel('Y-axis (Meters)')
    
    plt.gca().invert_zaxis()
    ax.set_xlim((-50, 50))
    ax.set_ylim((-10, 100))
    ax.set_zlim((-10, 20))
    ax.view_init(elev=20, azim=-70)
    ax.set_aspect('equal')
    ax.set_title('Transformed Point Clouds Comparison')
    ax.legend()
    
    plt.savefig(OUTPUT_RELATIVE_PATH + 'transformed_clouds.png')
    plt.close(fig)

    # Part 2: Plot inliers and outliers
    left0_vis = create_visualization_with_matches(img_left0, kp_left0, best_inliers, 0)
    left1_vis = create_visualization_with_matches(img_left1, kp_left1, best_inliers, 2)
    
    # Show matches visualization
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(left0_vis, cv2.COLOR_BGR2RGB))
    plt.title('Left Image 0')
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(left1_vis, cv2.COLOR_BGR2RGB))
    plt.title('Left Image 1')
    plt.savefig(OUTPUT_RELATIVE_PATH + 'inliers_outliers_q5.png')
    plt.close()
    
if __name__ == "__main__":
    K, M1, M2 = read_cameras()
    camera_left, camera_right = K @ M1, K @ M2

    img_left0, img_right0 = read_images(0)
    kp_left0, des_left0 = detector.detectAndCompute(img_left0, None)
    kp_right0, des_right0 = detector.detectAndCompute(img_right0, None)
    inliers_matches0, cloud0 = q1(camera_left, camera_right, kp_left0, des_left0, kp_right0, des_right0)

    img_left1, img_right1 = read_images(1)
    kp_left1, des_left1 = detector.detectAndCompute(img_left1, None)
    kp_right1, des_right1 = detector.detectAndCompute(img_right1, None)
    inliers_matches1, cloud1 = q1(camera_left, camera_right, kp_left1, des_left1, kp_right1, des_right1)

    matches_between_img0_img1 = q2(des_left0, des_left1)

    common_matches, common_matches_indices = find_stereo_temporal_matches(
        inliers_matches0, inliers_matches1, matches_between_img0_img1
    )

    tvec0 = M2[:,3]
    extrinsic_matrix_l1, extrinsic_matrix_r1, location_l1, location_r1 = q3(common_matches, cloud0, inliers_matches0, kp_left1, K, tvec0)
    location_l0 = find_camera_location(M1)
    location_r0 = find_camera_location(M2)
    plot_q3(location_l0, location_r0, location_l1, location_r1)

    # TODO: maybe use draw matches
    supporters = q4(common_matches_indices, cloud0, kp_left0,kp_right0,kp_left1,kp_right1, K, extrinsic_matrix_l1,extrinsic_matrix_r1,M1,M2, inliers_matches0)
    plot_q4(img_left0, img_left1, kp_left0, kp_left1, supporters)

    best_extrinsic_l1, best_extrinsic_r1, best_inliers =  q5(common_matches_indices, common_matches, cloud0, cloud1, inliers_matches0, kp_left0, kp_right0, kp_left1, kp_right1, K, M1, M2, tvec0)
    plot_q5(cloud0, cloud1, best_extrinsic_l1, img_left0, img_left1, kp_left0, kp_left1, best_inliers, common_matches_indices)

    q6()
