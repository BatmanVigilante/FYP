import torch
import cv2
import os
import numpy as np

from model.adc_res_transxnet import ADCResTransXNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "results/adc_res_transxnet.pth"
IMG_DIR = "data/patches/images"
MASK_DIR = "data/patches/masks"
OUT_DIR = "results/visualizations"

os.makedirs(OUT_DIR, exist_ok=True)

model = ADCResTransXNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

files = os.listdir(IMG_DIR)[:5]

for i, name in enumerate(files):
    img_path = os.path.join(IMG_DIR, name)
    mask_path = os.path.join(MASK_DIR, name)

    img = cv2.imread(img_path)
    gt_mask = cv2.imread(mask_path, 0)

    orig = img.copy()

    img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = img_input / 255.0
    img_input = torch.tensor(img_input, dtype=torch.float32).permute(2, 0, 1)
    img_input = img_input.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img_input)[0][0].cpu().numpy()

    pred_mask = (pred > 0.5).astype(np.uint8)

    # Prediction overlay
    pred_overlay = orig.copy()
    pred_overlay[pred_mask == 1] = [0, 255, 0]

    # Ground truth overlay
    gt_overlay = orig.copy()
    gt_overlay[gt_mask > 0] = [255, 0, 0]

    combined = np.hstack([orig, gt_overlay, pred_overlay])

    cv2.imwrite(f"{OUT_DIR}/comparison_{i}.png", combined)

print("Comparison images saved.")