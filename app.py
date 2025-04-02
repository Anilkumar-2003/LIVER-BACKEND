import os
import numpy as np
import cv2
import requests
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings

app = Flask(__name__)
CORS(app)

# Model URL and path
MODEL_URL = "https://huggingface.co/Vagicharla/liver_disease_h5/resolve/main/hepatitis_detection_model.h5"
MODEL_PATH = "model.h5"

# Function to download model
def download_model(url, path):
    if not os.path.exists(path):
        print("Downloading model...")
        response = requests.get(url, stream=True, timeout=60)
        if response.status_code == 200:
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded successfully.")
        else:
            print(f"Model download failed: {response.status_code}")
            raise Exception("Failed to download model.")

# Download model if not exists
download_model(MODEL_URL, MODEL_PATH)

# ✅ FIX: Load TensorFlow Model Instead of joblib
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Preprocess image
def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image.")
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    file_path = "https://i.postimg.cc/nrpZNMLy/temp.png"
    file.save(file_path)

    try:
        image = preprocess_image(file_path)
        prediction = model.predict(image)[0][0]
        result = "Infection" if prediction > 0.5 else "No Infection"
        return jsonify({"prediction": result, "confidence": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
