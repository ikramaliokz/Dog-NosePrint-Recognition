import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/home/athena/Documents/GitHub/Dog-NosePrint-Recognition/keras_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess the image
def preprocess_image(image_path):
    # Open the image file
    img = Image.open(image_path)
    
    # Resize the image to 256x256
    img = img.resize((256, 256))
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array / 255.0 - mean) / std
    
    # Add batch dimension and ensure the shape matches the model's input shape
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    return img_array

# Path to the image you want to use for inference
image_path = "/home/athena/Documents/GitHub/Dog-NosePrint-Recognition/dataset2/classes_with_6imgs/5672/O1CN01fjcDg01yHK0TN1bE3_!!0-mtopupload.jpg"

# Preprocess the image
input_data = preprocess_image(image_path)

# Set the tensor to point to the input data to be inferred
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run the inference
interpreter.invoke()

# Get the output data
output_data = interpreter.get_tensor(output_details[0]['index'])

# Print the output
print("Inference Output:", output_data)

pt_output = np.load("output.npy")

# Compare the output with the PyTorch output
print("Output Matches:", np.allclose(output_data, pt_output, atol=1e-1))

def cosine_similarity(arr1, arr2):
    dot_product = np.dot(arr1, arr2.T)
    norm_a = np.linalg.norm(arr1)
    norm_b = np.linalg.norm(arr2)
    similarity = dot_product / (norm_a * norm_b)
    return similarity[0][0]

# Calculate the cosine similarity between the outputs
similarity = cosine_similarity(output_data, pt_output)

print("Cosine Similarity:", similarity)