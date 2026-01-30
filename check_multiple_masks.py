import cv2
import os
import numpy as np

MASK_DIR = "data/raw/masks"

non_empty = 0
checked = 0

for name in os.listdir(MASK_DIR):
    mask = cv2.imread(os.path.join(MASK_DIR, name), 0)
    if mask is None:
        continue

    checked += 1
    if mask.sum() > 0:
        non_empty += 1
        print("Non-empty mask found:", name)
        break

print("Checked:", checked)
print("Found non-empty:", non_empty)