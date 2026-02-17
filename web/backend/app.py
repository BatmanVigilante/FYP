import os
import torch
import cv2
import numpy as np
from flask import Flask, request, send_file
from flask_cors import CORS
from model.adc_res_transxnet import ADCResTransXNet

# Base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths (absolute)
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../results/adc_res_transxnet.pth"))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)
CORS(app)

# Load model once
model = ADCResTransXNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


def predict_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("ERROR: Image not loaded")
        return None

    print("Loaded image shape:", img.shape)

    orig = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    img = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img)[0][0].cpu().numpy()

    print("Prediction shape:", pred.shape)

    mask = (pred > 0.5).astype(np.uint8)

    overlay = orig.copy().astype("uint8")
    overlay[mask == 1] = [0, 255, 0]

    return overlay


@app.route("/")
def home():
    return "Flask server running. Use /predict endpoint."


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]

    if file.filename == "":
        return "Empty filename", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    print("Saved upload to:", filepath)

    result = predict_image(filepath)

    if result is None:
        return "Prediction failed", 500

    output_path = os.path.join(OUTPUT_FOLDER, file.filename)

    cv2.imwrite(output_path, result)
    print("Saved output to:", output_path)

    return send_file(output_path, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)