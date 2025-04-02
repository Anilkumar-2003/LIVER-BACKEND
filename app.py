import os
import numpy as np
import cv2
import joblib
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS) for frontend requests

# Define Hugging Face model URL
MODEL_URL = "https://huggingface.co/spaces/Vagicharla/LIVER_DISEASE/resolve/main/model.pkl"
MODEL_PATH = "model.pkl"

# Download model if it does not exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Model downloaded successfully.")
    else:
        raise FileNotFoundError("Failed to download model from Hugging Face.")

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
    app.run(host="0.0.0.0", port=5000, debug=True)  # Run the Flask app on port 5000

