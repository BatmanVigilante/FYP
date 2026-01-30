import cv2
import os

IMG_DIR = "data/raw/images"
MASK_DIR = "data/raw/masks"

name = os.listdir(IMG_DIR)[0]

img = cv2.imread(os.path.join(IMG_DIR, name))
mask = cv2.imread(os.path.join(MASK_DIR, name), 0)

print("Image shape:", img.shape)
print("Mask shape:", mask.shape)
print("Mask unique values:", set(mask.flatten()))