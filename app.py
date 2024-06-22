import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to the directory containing symbol templates
template_dir = os.path.join(os.path.dirname(__file__), "Emoji-Directory")

# Load template images from the directory
template_images = []
for file in os.listdir(template_dir):
    if file.endswith(".png") or file.endswith(".jpg"):
        template_path = os.path.join(template_dir, file)
        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        template_images.append(template_img)

# Function to resize and preprocess the query image
def preprocess_image(img):
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if image has more than 1 channel

    # Check if image depth is CV_8U or CV_32F, convert if necessary
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    resized_img = cv2.resize(img, (50, 50))  # Resize to a fixed size
    _, thresh_img = cv2.threshold(resized_img, 127, 255, cv2.THRESH_BINARY)  # Apply thresholding
    return thresh_img

# Function to recognize symbol using template matching
def recognize_symbol(query_image):
    query_image = preprocess_image(query_image)
    best_match = None
    best_match_score = 0

    for i, template_img in enumerate(template_images):
        resized_template_img = cv2.resize(template_img, (50, 50))
        result = cv2.matchTemplate(query_image, resized_template_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val > best_match_score:
            best_match_score = max_val
            best_match = i

    if best_match is not None and best_match_score >= 0.70:
        confidence_percent = "{:.2f}".format(best_match_score * 100)  # Convert confidence to percentage with 2 decimal places
        filename = os.listdir(template_dir)[best_match]
        result_text = f"Recognized symbol: '{os.path.splitext(filename)[0]}' with confidence: {confidence_percent}%"
        return result_text
    else:
        return "Symbol Not Recognized"

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        result = recognize_symbol(img)
        return jsonify({'result': result})
    else:
        return jsonify({'result': 'No file uploaded'})

if __name__ == '__main__':
    app.run(debug=True)
