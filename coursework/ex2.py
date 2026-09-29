from kitti_slam.dataset import read_images
from reports.plots import plot_and_save
from kitti_slam.dataset import read_cameras
from reports.paths import COURSEWORK_OUTPUT_DIR
from kitti_slam.stereo import classify_matches_by_deviation, solveLLST
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


ORANGE = (0, 165, 255)
CYAN = (255, 255, 0)
np.random.seed(2)

#region helper functions
def createImgWithMatches(img, inliers, outliers):
    img_with_inliers = cv2.drawKeypoints(
        img,
        [kp0[match.queryIdx] for match in inliers],
        None,  # Specify None for the outImage argument
        color=ORANGE,
        flags=0,
    )
    img_with_inliers_and_outliers = cv2.drawKeypoints(
        img_with_inliers,
        [kp0[match.queryIdx] for match in outliers],
        None,  # Specify None for the outImage argument
        color=CYAN,
        flags=0,
    )
    return img_with_inliers_and_outliers

def plot_and_save_3d(points3D, title, output_path, optimize_visualization=True):
    fig = plt.figure(figsize=(10, 10))  # Create figure explicitly
    ax = fig.add_subplot(projection='3d')  # Add axes to this figure
    ax.scatter(points3D[0], points3D[2], points3D[1], c='blue', marker='o')
    ax.set_xlabel('X-axis (Meters)')
    ax.set_ylabel('Z-axis (Meters)')
    ax.set_zlabel('Y-axis (Meters)')
    if optimize_visualization:
        plt.gca().invert_zaxis()
        ax.set_xlim((-50,50))
        ax.set_ylim((-10,100))
        ax.set_zlim((-10,20))
        ax.view_init(elev=20, azim=-70)  # Set elevation and azimuth for the best view
        ax.set_aspect('equal')
    ax.set_title(title)
    plt.savefig(output_path)
    plt.close(fig)

def triangulateWithCV(camera1, camera2, kp0, kp1, inliers):
    # Extract the 2D points from the inliers
    train_keypoints = np.array([kp1[inliers[idx].trainIdx].pt for idx in range(len(inliers))])
    query_keypoints = np.array([kp0[inliers[idx].queryIdx].pt for idx in range(len(inliers))])
    # Triangulate the points
    points4D = cv2.triangulatePoints(camera1, camera2, query_keypoints.T, train_keypoints.T)
    # Convert to 3D points
    points3D = points4D[:3] / points4D[3]
    plot_and_save_3d(points3D, "3D Points from CV", os.path.join(COURSEWORK_OUTPUT_DIR, 'ex2_3D_points_CV.png'))

def linearLeastSquaresTriangulation(camera1, camera2, kp0, kp1, inliers):
    # Extract the 2D points from the inliers
    train_keypoints = np.array([kp1[inliers[idx].trainIdx].pt for idx in range(len(inliers))])
    query_keypoints = np.array([kp0[inliers[idx].queryIdx].pt for idx in range(len(inliers))])

    points3D = np.zeros((3, len(inliers)))
    for i in range(len(inliers)):
        point3D = solveLLST(query_keypoints[i], train_keypoints[i], camera1, camera2)
        points3D[:, i] = point3D
    plot_and_save_3d(points3D, "3D Points from Linear Least Squares", os.path.join(COURSEWORK_OUTPUT_DIR, 'ex2_3D_points_LLS.png'))
#endregion

def q1(matches):
    deviations = []
    for match in matches:
        train_keypoint = kp1[match.trainIdx]
        query_keypoint = kp0[match.queryIdx]
        cur_deviation = abs(query_keypoint.pt[1] - train_keypoint.pt[1])
        deviations.append(cur_deviation)

    plt.hist(deviations, bins=30)
    plt.title("Histogram of Deviations")
    plt.xlabel("Deviation in Y-axis")
    plt.ylabel("Number of Matches")
    plt.savefig(os.path.join(COURSEWORK_OUTPUT_DIR, "ex2_deviation_histogram.png"))
    plt.close()

    percentage_deviations = (sum(deviation > 2 for deviation in deviations) / len(deviations)) * 100
    print(f"Percentage of matches deviating by more than 2 pixels: {percentage_deviations:.2f}%")

def q2(matches, kp0, kp1):
    inliers, outliers = classify_matches_by_deviation(matches, kp0, kp1)
    print(f"Proportion of inliers: {len(inliers) / len(matches) * 100:.2f}%")
    print(f"Number of inliers: {len(inliers)}")


    img0_with_inliers_and_outliers = createImgWithMatches(img0, inliers, outliers)
    img1_with_inliers_and_outliers = createImgWithMatches(img1, inliers, outliers)
    combined_img = cv2.hconcat([img0_with_inliers_and_outliers, img1_with_inliers_and_outliers])
    plot_and_save(
        combined_img,
        "Inliers (orange colored dots) and Outliers (cyan colored dots)",
        os.path.join(COURSEWORK_OUTPUT_DIR, "ex2_inliers_outliers.png"),
        (20, 10),
        100,
    )

    return inliers, outliers

def q3(inliers, k, m1, m2):
    linearLeastSquaresTriangulation(k @ m1, k @ m2, kp0, kp1, inliers)
    triangulateWithCV(k @ m1, k @ m2, kp0, kp1, inliers)

def q4(k, m1, m2):
    camera1 = k @ m1
    camera2 = k @ m2
    three_imgs_idx = np.random.randint(0, 2599, 3)
    for idx in three_imgs_idx:
        img0, img1 = read_images(idx)
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        kp0, des0 = orb.detectAndCompute(img0, None)
        kp1, des1 = orb.detectAndCompute(img1, None)
        matches = bf.match(des0, des1)
        inliers, _ = classify_matches_by_deviation(matches, kp0, kp1)
        # Extract the 2D points from the inliers
        train_keypoints = np.array([kp1[inliers[idx].trainIdx].pt for idx in range(len(inliers))])
        query_keypoints = np.array([kp0[inliers[idx].queryIdx].pt for idx in range(len(inliers))])
        # Triangulate the points
        points4D = cv2.triangulatePoints(camera1, camera2, query_keypoints.T, train_keypoints.T)
        # Convert to 3D points
        points3D = points4D[:3] / points4D[3]
        plot_and_save_3d(points3D, f"3D Points from {idx}", os.path.join(COURSEWORK_OUTPUT_DIR, f'ex2_3D_points_{idx}.png'))

if __name__  == "__main__":
    from reports.paths import ensure_output_directories
    ensure_output_directories()
    img0, img1 = read_images(0)
    orb = cv2.ORB_create()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    kp0, des0 = orb.detectAndCompute(img0, None)
    kp1, des1 = orb.detectAndCompute(img1, None)
    matches = bf.match(des0, des1)
    q1(matches)
    inliers, _ = q2(matches, kp0, kp1)
    k, m1, m2 = read_cameras()
    print("Camera Matrix K:")
    print(k)
    print("Projection Matrix M1:")
    print(m1)
    print("Projection Matrix M2:")
    print(m2)
    q3(inliers, k, m1, m2)
    q4(k, m1, m2)
