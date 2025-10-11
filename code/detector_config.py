import cv2

DETECTOR = "AKAZE"

detector_config = {
    "AKAZE": {
        "detector": cv2.AKAZE_create(),
        "matcher": cv2.BFMatcher(cv2.NORM_HAMMING),
    },
    "ORB": {
        "detector": cv2.ORB_create(),
        "matcher": cv2.BFMatcher(cv2.NORM_HAMMING),
    },
    "SIFT": {
        "detector": cv2.SIFT_create(),
        "matcher": cv2.BFMatcher(cv2.NORM_L2),
    },
}

detector = detector_config[DETECTOR]["detector"]

matcher = detector_config[DETECTOR]["matcher"]
