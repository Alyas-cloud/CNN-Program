from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import google.generativeai as genai

# Initialize Flask app
app = Flask(__name__)

# Configure Google Generative AI
genai.configure(api_key="AIzaSyAGWeWOxFevp8qEStBxJO4AoEJY3W17FgE")

# Load the trained TensorFlow model
model = tf.keras.models.load_model(r"C:\Users\Rapid Computer's\Desktop\cnn fruits\fruit_classifier.h5")
print("Model loaded successfully.")

# Define class labels (ensure they match your training data order)
class_labels = ['apple', 'grape', 'mango', 'orange', 'watermelon']  # Adjust as needed

# Path to save uploaded images temporarily
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(img_path):
    """
    Predict the class of a fruit image.
    Args:
        img_path (str): Path to the image file.
    Returns:
        tuple: Predicted class label and confidence score.
    """
    img = image.load_img(img_path, target_size=(128, 128))  # Match input size of the model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index]
    return class_labels[predicted_class_index], float(confidence)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def upload_and_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Predict the class
    predicted_class, confidence = predict_image(file_path)
    os.remove(file_path)  # Remove the uploaded file after prediction

    # Generate content using Google Generative AI
    try:
        gen_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gen_model.generate_content(f"Describe the benefits of {predicted_class}")
        generated_text = response.text
    except Exception as e:
        generated_text = f"Could not generate content: {str(e)}"

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2),
        "generated_text": generated_text
    })

if __name__ == "__main__":
    app.run(debug=True)
