from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import os
import cv2
import asyncio
from utils import load_image, preprocess_image, extract_colors, overlay_logo, create_banner, generate_image_from_text

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return "API is running"

@app.route('/extract_colors', methods=['POST'])
def extract_colors_endpoint():
    file = request.files['logo']
    logo_path = os.path.join('static', 'logo.png')
    file.save(logo_path)
    
    logo_image = load_image(logo_path)
    processed_logo = preprocess_image(logo_image)
    branding_colors = extract_colors(processed_logo)
    
    return jsonify({"branding_colors": branding_colors.tolist()})

@app.route('/generate_image', methods=['POST'])
def generate_image_endpoint():
    data = request.json
    prompt = data['prompt']
    branding_colors = np.array(data['branding_colors'])
    
    # Generate image based on the prompt
    generated_image_path = os.path.join('static', 'generated_image.png')
    
    # Use asyncio.run() to run the asynchronous function synchronously
    asyncio.run(generate_image_from_text(prompt, generated_image_path))

    logo_path = os.path.join('static', 'logo.png')
    
    modified_image = overlay_logo(generated_image_path, logo_path)
    modified_image_path = os.path.join('static', 'modified_image.jpg')
    cv2.imwrite(modified_image_path, modified_image)
    
    return send_file(modified_image_path, mimetype='image/jpeg')

@app.route('/generate_banner', methods=['POST'])
def generate_banner_endpoint():
    # Get the uploaded logo
    if 'logo' not in request.files:
        return jsonify({"error": "No logo file part"}), 400
    file = request.files['logo']
    if file.filename == '':
        return jsonify({"error": "No selected logo file"}), 400
    logo_path = os.path.join('static', 'logo.png')
    file.save(logo_path)

    # Get other form data
    branding_colors = request.form.get('branding_colors')
    occasion = request.form.get('occasion')
    additional_text = request.form.get('text')

    if not branding_colors or not occasion or not additional_text:
        return jsonify({"error": "Missing form data"}), 400

    # Convert branding colors from string to numpy array
    branding_colors = np.array([list(map(float, color.split(','))) for color in branding_colors.split(';')])

    # Generate the banner
    banner_path = 'output/banner.png'
    create_banner(branding_colors, occasion, additional_text, banner_path)

    return jsonify({"message": "Banner created successfully", "banner_path": banner_path})


if __name__ == '__main__':
    app.run(debug=True)
