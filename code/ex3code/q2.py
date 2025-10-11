import cv2
from .. import detector_config

LOWE_RATIO = 0.7
STRICT_RATIO = 0.8
def q2(des0, des1):
    matches = detector_config.matcher.match(des0, des1)
    return matches
