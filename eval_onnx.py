import os
import heapq
from collections import defaultdict
import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm

def preprocess_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.BILINEAR)  # Resize to the target size
    img_array = np.array(img)
    
    # Convert image array to float, rearrange dimensions, and normalize to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # Reorder dimensions to CxHxW
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
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
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []

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

# Function to load the ONNX model
def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)
    return session

# Function to run inference on a single image
def run_inference(session, image):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: image})
    return output

# Function to generate embeddings for the dataset
def generate_embeddings(session, dataset):
    embeddings = defaultdict(list)
    for img, class_id in tqdm(dataset):
        emb = run_inference(session, img)[0]
        embeddings[class_id].append(emb)
    return embeddings

# Function to calculate accuracy
def calculate_accuracy(session, query_dataset, db_embeddings, top_k=(1, 3, 5)):
    top_k_correct = {k: 0 for k in top_k}
    total = 0

    for img1, label in tqdm(query_dataset):
        query_emb = run_inference(session, img1)[0]
        
        # List to store distances and corresponding class ids
        distances = []
        
        for db_class_id, embeddings in db_embeddings.items():
            for emb in embeddings:
                distance = np.linalg.norm(query_emb - emb)
                distances.append((distance, db_class_id))
        
        # Get the closest classes
        closest_classes = [class_id for _, class_id in heapq.nsmallest(max(top_k), distances)]
        
        for k in top_k:
            if label in closest_classes[:k]:
                top_k_correct[k] += 1

        total += 1
    
    top_k_accuracies = {k: correct / total for k, correct in top_k_correct.items()}
    return top_k_accuracies

# Main script
if __name__ == '__main__':
    train_dir = '400_classes_dataset/train'
    db_dir = '101_classes_dataset_for_testing/db_images'
    q_dir = '101_classes_dataset_for_testing/query_images'
    
    # Load datasets
    db_dataset = DogNoseDataset(root_dir=db_dir)
    query_dataset = QueryDataset(root_dir=q_dir)
    
    # Load the ONNX model
    model_path = 'model-pet.onnx'
    session = load_onnx_model(model_path)
    
    # Check if the embeddings are already generated, then load them
    if os.path.exists('db_embeddings_onnx.npy'):
        db_embeddings = np.load('db_embeddings_onnx.npy', allow_pickle=True).item()
    else:
        # Generate embeddings for the database images
        print("Generating embeddings for the database images...")
        db_embeddings = generate_embeddings(session, db_dataset)

        # Save the database embeddings
        try:
            np.save('db_embeddings_onnx.npy', db_embeddings)
        except Exception as e:
            print("Error saving database embeddings:", e)
    
    # Calculate accuracy on the query dataset
    print("Calculating accuracy on the query dataset...")
    accuracies = calculate_accuracy(session, query_dataset, db_embeddings, top_k=(1, 3, 5))
    
    print(f"Top-1 Accuracy: {accuracies[1] * 100:.4f}%")
    print(f"Top-3 Accuracy: {accuracies[3] * 100:.4f}%")
    print(f"Top-5 Accuracy: {accuracies[5] * 100:.4f}%")
