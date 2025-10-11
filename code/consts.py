import os

DATASET_RELATIVE_PATH = r"./dataset/sequences/05/"
OUTPUT_RELATIVE_PATH = r"./code/output/"
FINAL_PLOTS_RELATIVE_PATH = r"./code/final/plots/"
FIRSTFRAME = 0
LASTFRAME = 2599

# Additional constants needed by utility.py
CALIB_FILE = DATASET_RELATIVE_PATH + "calib.txt"
GT_POSES_FILE = "./dataset/poses/05.txt"
cwd = os.getcwd()
DATASET_RELATIVE_PATH = r"./dataset/sequences/05/"
OUTPUT_RELATIVE_PATH = r"./code/output/"

DATA_PATH = os.path.join(cwd, DATASET_RELATIVE_PATH)
OUTPUT_PATH = os.path.join(cwd, OUTPUT_RELATIVE_PATH)

LEFT_IMG_DIR = os.path.join(DATA_PATH, "image_0")
RIGHT_IMG_DIR = os.path.join(DATA_PATH, "image_1")