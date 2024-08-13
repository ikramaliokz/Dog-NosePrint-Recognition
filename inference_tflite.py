import os
import heapq
from itertools import combinations
from collections import defaultdict
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

# Function to preprocess the image
def preprocess_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.BILINEAR)  # Resize to the target size
    img_array = np.array(img)
    
    # Convert image array to float and normalize to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Function to run inference on a single image
def run_inference(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

image_path = 'image.jpg'
model_path = 'keras_model.tflite'

# Load the TFLite model
interpreter = load_tflite_model(model_path)

# Preprocess the image
image = preprocess_image(image_path)

print("Image Shape:", image.shape)

# Run inference on the preprocessed image
output = run_inference(interpreter, image)

print(output)