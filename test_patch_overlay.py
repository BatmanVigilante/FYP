import cv2
import os

IMG_DIR = "data/patches/images"
MASK_DIR = "data/patches/masks"

os.makedirs("results/visualizations", exist_ok=True)

files = os.listdir(IMG_DIR)

for i in range(5):
    name = files[i]

    img = cv2.imread(os.path.join(IMG_DIR, name))
    mask = cv2.imread(os.path.join(MASK_DIR, name), 0)

    overlay = img.copy()
    overlay[mask > 0] = [0, 255, 0]   # create overlay HERE

    cv2.imwrite(
        f"results/visualizations/patch_overlay_{i}.png",
        overlay
    )

    print("Saved:", f"patch_overlay_{i}.png")