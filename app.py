import requests
import os
import numpy as np
import cv2
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# Google Drive file ID extracted from the link
file_id = "1zRS2cP562T28-QjxbO5uZR-BFXV3R7YH"  # Extracted from the shared link
model_path = "model.pkl"

# Function to download model from Google Drive
def download_model(file_id, model_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url, stream=True)
    
    if "Content-Disposition" not in response.headers:
        print("Google Drive security check detected. Manual download may be needed.")
        return False
    
    total_size = int(response.headers.get("content-length", 0))
    downloaded_size = 0

    with open(model_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)
                percent_complete = (downloaded_size / total_size) * 100
                print(f"\rDownloading Model: {percent_complete:.2f}% complete", end="")

    print("\nModel downloaded successfully!")
    return True

# Check if model exists, else download it
if not os.path.exists(model_path):
    print("Model file not found. Downloading...")
    success = download_model(file_id, model_path)
    if not success:
        raise FileNotFoundError("Model download failed. Please check the link or download manually.")

# Load the model
model = joblib.load(model_path)

# Preprocessing function
def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image.")
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    file_path = "temp.jpg"
    file.save(file_path)  # Save temporarily

    try:
        image = preprocess_image(file_path)
        prediction = model.predict(image)[0][0]  # Predict
        result = "Infection" if prediction > 0.5 else "No Infection"
        return jsonify({"prediction": result, "confidence": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
