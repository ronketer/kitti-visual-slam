from stereo import classify_matches_by_deviation
import numpy as np
import cv2
# from alg import classify_matches_by_deviation
from .. import detector_config

def q1(camera1, camera2, kp_left, des_left, kp_right, des_right):
    matches = detector_config.matcher.match(des_left, des_right)
    inliers_matches, _ = classify_matches_by_deviation(matches, kp_left, kp_right)
    pts_left = np.float32([kp_left[m.queryIdx].pt for m in inliers_matches]).T
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in inliers_matches]).T
    cloud4D = cv2.triangulatePoints(camera1, camera2, pts_left, pts_right)
    return inliers_matches, (cloud4D[:3] / cloud4D[3]).T
