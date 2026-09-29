"""Construct OpenCV components explicitly, with lazy legacy exports."""
import cv2
from .config import DEFAULT_DETECTOR

DETECTOR = DEFAULT_DETECTOR


def create_detector_and_matcher(name=DEFAULT_DETECTOR):
    constructors = {
        "AKAZE": (cv2.AKAZE_create, cv2.NORM_HAMMING),
        "ORB": (cv2.ORB_create, cv2.NORM_HAMMING),
        "SIFT": (cv2.SIFT_create, cv2.NORM_L2),
    }
    constructor, norm = constructors[name]
    return constructor(), cv2.BFMatcher(norm)


__all__ = ["DETECTOR", "create_detector_and_matcher", "detector_config", "detector", "matcher"]


def __getattr__(name):
    if name not in {"detector_config", "detector", "matcher"}:
        raise AttributeError(name)
    if "detector_config" not in globals():
        components = {}
        for detector_name in ("AKAZE", "ORB", "SIFT"):
            detector_obj, matcher_obj = create_detector_and_matcher(detector_name)
            components[detector_name] = {"detector": detector_obj, "matcher": matcher_obj}
        globals()["detector_config"] = components
        globals()["detector"] = components[DETECTOR]["detector"]
        globals()["matcher"] = components[DETECTOR]["matcher"]
    return globals()[name]
