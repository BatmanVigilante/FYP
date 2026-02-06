import torch
import cv2
import os
import numpy as np

from model.adc_res_transxnet import ADCResTransXNet

def predict_image(image_path, model):
    img = cv2.imread(image_path)
    orig = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    img = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img)[0][0].cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8)
    overlay = orig.copy()
    overlay[mask == 1] = [0, 255, 0]

    return overlay

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "results/adc_res_transxnet.pth"
IMG_DIR = "data/patches/images"
OUT_DIR = "results/visualizations"

os.makedirs(OUT_DIR , exist_ok=True)

# Load model
model = ADCResTransXNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Pick one image
img_name = os.listdir(IMG_DIR)[0]
img = cv2.imread(os.path.join(IMG_DIR, img_name))
orig = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255.0
img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
img = img.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = model(img)[0][0].cpu().numpy()

# Threshold
mask = (pred > 0.5).astype(np.uint8)

# Overlay
overlay = orig.copy()
overlay[mask == 1] = [0, 255, 0]

cv2.imwrite(f"{OUT_DIR}/prediction.png", overlay)

print("Saved prediction to results/visualizations/prediction.png")