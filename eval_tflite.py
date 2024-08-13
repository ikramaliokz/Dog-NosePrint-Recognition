import os
import heapq
from itertools import combinations
from collections import defaultdict
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

# Function to preprocess the image
# def preprocess_image(image_path, target_size=(256, 256)):
#     img = Image.open(image_path).convert('RGB')
#     img = img.resize(target_size, Image.BILINEAR)  # Resize to the target size
#     img_array = np.array(img)
    
#     # Convert image array to float and normalize to [0, 1]
#     img_array = img_array.astype(np.float32) / 255.0
    
#     # Add batch dimension
#     img_array = np.expand_dims(img_array, axis=0)
    
#     return img_array

def preprocess_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path)  # Load image
    img = img.resize(target_size, Image.BILINEAR)
    img_array = np.array(img)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


class DogNoseDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.label_img_pairs = []

        # Walk through the directory to list all images
        for class_id, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_filename in os.listdir(class_dir):
                    self.label_img_pairs.append((os.path.join(class_dir, img_filename), class_name))     

    def __getitem__(self, index):
        img_path, label = self.label_img_pairs[index]
        img = preprocess_image(img_path)

        return img, label

    def __len__(self):
        return len(self.label_img_pairs)

class QueryDataset:
    def __init__(self, root_dir):
        """
        root_dir: Directory with all the subdirectories of classes.
        """
        self.root_dir = root_dir
        self.image_paths = []  # Store image paths
        self.labels = []       # Store class labels for each image

        # Walk through the directory to list all images
        for class_id, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_filename in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, img_filename))
                    self.labels.append(class_name)

        print(f"Found {len(self.image_paths)} images in {len(set(self.labels))} classes.")

    def __getitem__(self, index):
        img_path, label = self.image_paths[index], self.labels[index]
        img = preprocess_image(img_path)

        return img, label

    def __len__(self):
        return len(self.image_paths)


# Function to load the TFLite model
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

# Function to generate embeddings for the dataset
def generate_embeddings(interpreter, dataset):
    embeddings = defaultdict(list)
    for img, class_Id in tqdm(dataset):
        emb = run_inference(interpreter, img)[0]
        embeddings[class_Id].append(emb)
    return embeddings

# Function to calculate accuracy
def calculate_accuracy(interpreter, query_dataset, db_embeddings):
    total = 0
    correct = 0
    correct_distances = []
    incorrect_distances = []

    for img1, label in tqdm(query_dataset):
        query_emb = run_inference(interpreter, img1)[0]
        
        # List to store distances and corresponding class ids
        distances = []
        
        for db_class_id, embeddings in db_embeddings.items():
            for emb in embeddings:
                distance = np.linalg.norm(query_emb - emb)
                distances.append((distance, db_class_id))
        
        # Get the closest class
        closest_class, closest_class_id = min(distances, key=lambda x: x[0])
        
        if label == closest_class_id:
            correct += 1
            correct_distances.append(closest_class)
        else:
            incorrect_distances.append(closest_class)

        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct_distances, incorrect_distances


# Main script
if __name__ == '__main__':
    train_dir = '400_classes_dataset/train'
    db_dir = '101_classes_dataset_for_testing/db_images'
    q_dir = '101_classes_dataset_for_testing/query_images'
    
    # Load datasets
    db_dataset = DogNoseDataset(root_dir=db_dir)
    query_dataset = QueryDataset(root_dir=q_dir)
    
    # Load the TFLite model
    model_path = '/home/athena/Documents/GitHub/Dog-NosePrint-Recognition/dog_nose_quant_f16.tflite'
    interpreter = load_tflite_model(model_path)
    
    # check if the embeddings are already generated, then load them

    if os.path.exists('db_embeddings_tflite.npy'):
        db_embeddings = np.load('db_embeddings_tflite.npy', allow_pickle=True).item()
    else:

        # Generate embeddings for the database images
        print("Generating embeddings for the database images...")
        db_embeddings = generate_embeddings(interpreter, db_dataset)

        # Save the database embeddings
        try:
            np.save('db_embeddings_tflite.npy', db_embeddings)
        except Exception as e:
            print("Error saving database embeddings:", e)
    
    # print(db_embeddings)

    # Calculate accuracy on the query dataset
    print("Calculating accuracy on the query dataset...")
    accuracies, correct_distances_list,incorrect_distances_list = calculate_accuracy(interpreter, query_dataset, db_embeddings)
    correct_distances_array = np.array(correct_distances_list)
    np.save('correct_distances.npy', correct_distances_array)
    
    correct_min_distance = min(correct_distances_list)
    correct_max_distance = max(correct_distances_list)
    correct_avg_distance = sum(correct_distances_list) / len(correct_distances_list)
    print("Correct distances")
    print(f"Min: {correct_min_distance}, Max: {correct_max_distance}, Avg: {correct_avg_distance}")
    incorrect_distances_array = np.array(incorrect_distances_list)
    np.save('incorrect_distances.npy', incorrect_distances_array)
    incorrect_min_distance = min(incorrect_distances_list)
    incorrect_max_distance = max(incorrect_distances_list)
    print("Incorrect distances")
    incorrect_avg_distance = sum(incorrect_distances_list) / len(incorrect_distances_list)
    print(f"Min: {incorrect_min_distance}, Max: {incorrect_max_distance}, Avg: {incorrect_avg_distance}")
    print(f"Top-1 Accuracy: {accuracies * 100:.4f}%")
