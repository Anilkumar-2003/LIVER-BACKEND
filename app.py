import os
import numpy as np
import cv2
import joblib
import requests
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS) for frontend requests

# Corrected Hugging Face model URL
MODEL_URL = "https://huggingface.co/Vagicharla/liver_disease_h5/resolve/main/hepatitis_detection_model.h5"
MODEL_PATH = "model.h5"

# Function to download model with retries
def download_model(url, path, retries=5, delay=5):
    for i in range(retries):
        try:
            print(f"Attempt {i+1}: Downloading model...")
            response = requests.get(url, stream=True, timeout=60)
            if response.status_code == 200:
                with open(path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Model downloaded successfully.")
                return
            else:
                print(f"Failed to download model. Status code: {response.status_code}")
        except Exception as e:
            print(f"Download failed: {e}")
        time.sleep(delay)  # Wait before retrying
    raise FileNotFoundError("Failed to download model after multiple attempts.")

# Download model if not present
if not os.path.exists(MODEL_PATH):
    download_model(MODEL_URL, MODEL_PATH)

# Load the trained machine learning model
model = joblib.load(MODEL_PATH)

# Function to preprocess the input image
def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)  # Read the image from file
    if image is None:
        raise ValueError("Could not read the image.")
    image = cv2.resize(image, target_size)  # Resize image to match model input size
    image = image.astype("float32") / 255.0  # Normalize pixel values to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400  # Return error if no file is provided

    file = request.files["file"]  # Get the uploaded file
    file_path = "temp.jpg"
    file.save(file_path)  # Save file temporarily

    try:
        image = preprocess_image(file_path)  # Preprocess the image
        prediction = model.predict(image)[0][0]  # Make a prediction
        result = "Infection" if prediction > 0.5 else "No Infection"  # Interpret the prediction
        return jsonify({"prediction": result, "confidence": float(prediction)})  # Return the result
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Handle any errors during processing

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use the port assigned by Render
    app.run(host="0.0.0.0", port=port, debug=True)  # Run the Flask app
