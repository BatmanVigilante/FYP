import cv2
import random
import os

IMG_DIR = "data/patches/images"
MASK_DIR = "data/patches/masks"

name = random.choice(os.listdir(IMG_DIR))

img = cv2.imread(os.path.join(IMG_DIR, name))
mask = cv2.imread(os.path.join(MASK_DIR, name), 0)

overlay = img.copy()
overlay[mask > 0] = [0, 255, 0]

cv2.imwrite("results/visualizations/patch_overlay.png", overlay)

print("Saved overlay for:", name)