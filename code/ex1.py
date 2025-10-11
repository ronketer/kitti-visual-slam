import cv2
import matplotlib.pyplot as plt
import random
import os
from utility import plot_and_save
from alg import *

random.seed(0)
cwd = os.getcwd()

# path related constants
DATASET_RELATIVE_PATH = r"./dataset/sequences/05/"
OUTPUT_RELATIVE_PATH = r"./code/output/"

DATA_PATH = os.path.join(cwd, DATASET_RELATIVE_PATH)
OUTPUT_PATH = os.path.join(cwd, OUTPUT_RELATIVE_PATH)


LEFT_IMG_DIR = os.path.join(DATA_PATH, "image_0")
RIGHT_IMG_DIR = os.path.join(DATA_PATH, "image_1")

# output files names
OUTPUT_FILES = {
    "keypoints": "ex1_keypoints.png",
    "random_matches": "ex1_random_matches.png",
    "good_matches": "ex1_random_good_matches_ratio_{}.png",
    "failed_matches": "ex1_potential_false_negative_ratio_{}.png",
}

# plotting visual configuration
VISUAL_CONFIG = {
    "figsize": (20, 10),
    "dpi": 300,
    "color_kp": (0, 255, 0),
    "titles": {
        "keypoints": "Keypoints in Left and Right Images",
        "random_matches": "Random 20 Matches Between Images",
        "significant_matches": "Matches Passed Ratio Test = {:.2f}",
        "failed_matches": "Potential Valid Matches Rejected by Ratio {:.2f}",
    },
}

# algorithms parameters
STRICT_RATIO = 0.75
LOWE_RATIO = 0.8
KNN_MATCHES = 2


def read_images(idx):
    """Read left and right images from the dataset."""
    img_name = "{:06d}.png".format(idx)
    img0 = cv2.imread(os.path.join(LEFT_IMG_DIR, img_name), 0)
    img1 = cv2.imread(os.path.join(RIGHT_IMG_DIR, img_name), 0)
    return img0, img1


def q1(img0, kp0, img1, kp1):
    """Question 1.1 - Display keypoints on both stereo images."""
    img0_with_kp = cv2.drawKeypoints(
        img0, kp0, img0, color=VISUAL_CONFIG["color_kp"], flags=0
    )
    img1_with_kp = cv2.drawKeypoints(
        img1, kp1, img1, color=VISUAL_CONFIG["color_kp"], flags=0
    )
    combined_img = cv2.hconcat([img0_with_kp, img1_with_kp])
    plot_and_save(
        combined_img,
        VISUAL_CONFIG["titles"]["keypoints"],
        os.path.join(OUTPUT_PATH, OUTPUT_FILES["keypoints"]),
        VISUAL_CONFIG["figsize"],
        VISUAL_CONFIG["dpi"],
    )


def q2(des0, des1):
    """Question 1.2 - Print feature descriptors."""
    print("The descriptors of the two first features in image 0 are:")
    print(des0[0:2])
    print("The descriptors of the first two features in image 1 are:")
    print(des1[0:2])


def q3(bf: cv2.BFMatcher, img0, kp0, des0, img1, kp1, des1):
    """Question 1.3 - Match features between stereo pairs."""
    matches = bf.match(des0, des1)
    random_matches = random.sample(matches, 20)
    matched_img = cv2.drawMatches(
        img0,
        kp0,
        img1,
        kp1,
        random_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    plot_and_save(
        matched_img,
        VISUAL_CONFIG["titles"]["random_matches"],
        os.path.join(OUTPUT_PATH, OUTPUT_FILES["random_matches"]),
        VISUAL_CONFIG["figsize"],
        VISUAL_CONFIG["dpi"],
    )


def q4(bf: cv2.BFMatcher, img0, kp0, des0, img1, kp1, des1):
    """Question 1.4 - Match features using k-nearest neighbors."""
    matches = bf.knnMatch(des0, des1, k=KNN_MATCHES)
    results = filter_by_ratio_test(matches, [STRICT_RATIO, LOWE_RATIO])
    random_good_matches_idx = random.sample(results["good"][STRICT_RATIO], 20)
    random_good_matches = [[matches[i][0]] for i in random_good_matches_idx]

    print(f"number of matches before significance test: {len(matches)}")
    print(
        f"number of matches discarded: {len(matches) - len(results['good'][STRICT_RATIO])}"
    )

    good_matches_img = cv2.drawMatchesKnn(
        img0,
        kp0,
        img1,
        kp1,
        random_good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    good_output = OUTPUT_FILES["good_matches"].format(STRICT_RATIO)
    plot_and_save(
        good_matches_img,
        VISUAL_CONFIG["titles"]["significant_matches"].format(STRICT_RATIO),
        os.path.join(OUTPUT_PATH, good_output),
        VISUAL_CONFIG["figsize"],
        VISUAL_CONFIG["dpi"],
    )

    strict_bad = set(results["bad"][STRICT_RATIO])
    lowe_good = set(results["good"][LOWE_RATIO])
    potential_false_negatives = list(strict_bad & lowe_good)
    false_negatives = sorted(
        [(matches[i][0], i) for i in potential_false_negatives],
        key=lambda x: x[0].distance,
    )
    best_false_negatives = [[m[0]] for m in false_negatives[:5]]
    failed_matches_img = cv2.drawMatchesKnn(
        img0,
        kp0,
        img1,
        kp1,
        best_false_negatives,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    failed_output = OUTPUT_FILES["failed_matches"].format(STRICT_RATIO)
    plot_and_save(
        failed_matches_img,
        VISUAL_CONFIG["titles"]["failed_matches"].format(STRICT_RATIO),
        os.path.join(OUTPUT_PATH, failed_output),
        VISUAL_CONFIG["figsize"],
        VISUAL_CONFIG["dpi"],
    )


if __name__ == "__main__":
    img0, img1 = read_images(0)
    orb = cv2.ORB_create()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    kp0, des0 = orb.detectAndCompute(img0, None)
    kp1, des1 = orb.detectAndCompute(img1, None)

    q1(img0, kp0, img1, kp1)
    q2(des0, des1)
    q3(bf, img0, kp0, des0, img1, kp1, des1)
    q4(bf, img0, kp0, des0, img1, kp1, des1)
