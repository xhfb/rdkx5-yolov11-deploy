import os
import cv2
import numpy as np
import sys
from glob import glob

if len(sys.argv) < 3:
    print("Usage: python prepare_data.py <input_dir> <output_dir>")
    sys.exit(1)

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
TARGET_SIZE = 640

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

images = glob(os.path.join(INPUT_DIR, '*'))#[:100]
print(f"Processing {len(images)} images from {INPUT_DIR}...")

for idx, path in enumerate(images):
    img = cv2.imread(path)
    if img is None: continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = letterbox(img)
    img = img.transpose(2, 0, 1).astype(np.float32)
    img.tofile(os.path.join(OUTPUT_DIR, f'calib_{idx}.rgb'))