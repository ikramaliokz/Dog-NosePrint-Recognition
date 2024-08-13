from collections import defaultdict
import numpy as np
import torch
import os
from tqdm import tqdm
from ultralytics import YOLO
import heapq


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
        return img_path, label

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

        return img_path, label

    def __len__(self):
        return len(self.image_paths)


# Function to run inference on a single image
def run_inference(model, image_path):
    return model.predict(image_path, embed=[74], verbose=False)

# Function to generate embeddings for the dataset
def generate_embeddings(model, dataset):
    embeddings = defaultdict(list)
    for img, class_Id in tqdm(dataset):
        emb = run_inference(model, img)
        embeddings[class_Id].append(emb)
    return embeddings

def cosine_similarity(embedding1, embedding2):
    return torch.nn.CosineSimilarity(dim=0)(embedding1[0][0], embedding2[0][0]).item()

def calculate_accuracy(model, query_dataset, db_embeddings, top_k=(1, 3, 5)):
    top_k_correct = {k: 0 for k in top_k}
    total = 0

    for img, label in tqdm(query_dataset):

        query_emb = run_inference(model, img)
        
        similarities = []
        for db_class_id, embeddings in db_embeddings.items():
            for emb in embeddings:
                similarity = cosine_similarity(query_emb, emb)
                similarities.append((similarity, db_class_id))
        
        closest_classes = [class_id for _, class_id in heapq.nlargest(max(top_k), similarities)]
        
        for k in top_k:
            if label in closest_classes[:k]:
                top_k_correct[k] += 1
        total += 1
    
    top_k_accuracies = {k: correct / total for k, correct in top_k_correct.items()}
    return top_k_accuracies


# Main script
if __name__ == '__main__':
    db_dir = '101_classes_dataset_for_testing/db_images'
    q_dir = '101_classes_dataset_for_testing/query_images'
    
    # Load datasets
    db_dataset = DogNoseDataset(root_dir=db_dir)
    query_dataset = QueryDataset(root_dir=q_dir)
    
    # Load the TFLite model
    model_path = 'runs/classify/train5/weights/best.pt'
    model = YOLO(model_path)
    
    # check if the embeddings are already generated, then load them


    # Generate embeddings for the database images
    print("Generating embeddings for the database images...")
    db_embeddings = generate_embeddings(model, db_dataset)
    
    # print(db_embeddings)

    # Calculate accuracy on the query dataset
    accuracies = calculate_accuracy(model, query_dataset, db_embeddings, top_k=(1, 3, 5))
    print(f"Top-1 Accuracy: {accuracies[1] * 100:.4f}%")
    print(f"Top-3 Accuracy: {accuracies[3] * 100:.4f}%")
    print(f"Top-5 Accuracy: {accuracies[5] * 100:.4f}%")