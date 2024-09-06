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
    class_labels = [
    'affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
    'american_staffordshire_terrier', 'appenzeller', 'australian_terrier',
    'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog',
    'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick',
    'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres',
    'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
    'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber',
    'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole',
    'dingo', 'doberman', 'english_foxhound', 'english_setter', 'english_springer',
    'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog', 
    'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever',
    'gordon_setter', 'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog',
    'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel',
    'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie',
    'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
    'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog', 'mexican_hairless',
    'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'newfoundland',
    'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
    'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone',
    'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke',
    'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog', 
    'shih-tzu', 'siberian_husky', 'silky_terrier', 'soft-coated_wheaten_terrier', 
    'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
    'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla',
    'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier',
    'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier'
]  # Example labels
    breed_name = class_labels[class_index]
    return jsonify({"breed": breed_name})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
