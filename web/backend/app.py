import os
import torch
import cv2
import numpy as np
from flask import Flask, request, send_file
from model.adc_res_transxnet import ADCResTransXNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "results/adc_res_transxnet.pth"
UPLOAD_FOLDER = "web/backend/uploads"
OUTPUT_FOLDER = "web/backend/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)

# Load model once
model = ADCResTransXNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


def predict_image(image_path):
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


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result = predict_image(filepath)

    output_path = os.path.join(OUTPUT_FOLDER, file.filename)
    cv2.imwrite(output_path, result)

    return send_file(output_path, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)