from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model.h5')

# Define image preprocessing function
def preprocess_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Define prediction function
def predict_breed(image_path):
    image_array = preprocess_image(image_path)
    predictions = model.predict(image_array)
    class_index = np.argmax(predictions, axis=1)[0]
    return class_index

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image_path = "temp.jpg"
    file.save(image_path)
    class_index = predict_breed(image_path)
    # Assuming you have a mapping of class indices to breed names
    class_labels = ["breed1", "breed2", "breed3", "breed120"]  # Example labels
    breed_name = class_labels[class_index]
    return jsonify({"breed": breed_name})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
