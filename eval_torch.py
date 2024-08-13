import os
import heapq
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        base_model = resnet152(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.reduce_channels = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.reduce_channels(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels

    def forward(self, x):
        # x: input feature map with shape (batch_size, C, H, W)
        batch_size, C, H, W = x.size()
        
        # Reshape x to (batch_size, C, H*W)
        x_reshaped = x.view(batch_size, C, -1)
        
        # Compute the channel attention map
        channel_attention_map = torch.bmm(x_reshaped, x_reshaped.transpose(1, 2))
        channel_attention_map = F.softmax(channel_attention_map, dim=1)
        
        # Multiply the attention map by the input feature map
        x_weighted = torch.bmm(channel_attention_map, x_reshaped)
        
        # Reshape back to (batch_size, C, H, W)
        x_weighted = x_weighted.view(batch_size, C, H, W)
        
        # Apply scale parameter and element-wise summation
        beta = torch.nn.Parameter(torch.zeros(1)).to(device)
        out = beta * x_weighted + x
        
        return out

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # x: input feature map with shape (batch_size, C, H, W)
        batch_size, Ch, H, W = x.size()
        
        # Obtain new feature maps B and C
        B = self.conv1(x)
        C = self.conv2(x)
        
        # Reshape B and C to (batch_size, C, H*W)
        B_reshaped = B.view(batch_size, Ch, -1)
        C_reshaped = C.view(batch_size, Ch, -1)
        
        # Compute the spatial attention map
        spatial_attention_map = torch.bmm(B_reshaped.transpose(1, 2), C_reshaped)
        spatial_attention_map = F.softmax(spatial_attention_map, dim=1)
        
        # Multiply the attention map by the input feature map
        D = self.conv3(x)
        D_reshaped = D.view(batch_size, Ch, -1)
        x_weighted = torch.bmm(spatial_attention_map, D_reshaped.transpose(1, 2))
        
        # Reshape back to (batch_size, C, H, W)
        x_weighted = x_weighted.view(batch_size, Ch, H, W)
        
        # Apply scale parameter and element-wise summation
        alpha = torch.nn.Parameter(torch.zeros(1)).to(device)
        out = alpha * x_weighted + x
        
        return out

class DualAttentionNetwork(nn.Module):
    def __init__(self, in_channels):
        super(DualAttentionNetwork, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention(in_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels * 3, 1024)

    def forward(self, x):
        # Apply channel attention
        channel_attention_map = self.channel_attention(x)
        
        # Apply spatial attention
        spatial_attention_map = self.spatial_attention(x)
        
        # Concatenate the input feature map with channel and spatial attention maps
        concatenated = torch.cat([x, channel_attention_map, spatial_attention_map], dim=1)
        
        # Global average pooling
        gap = self.global_avg_pool(concatenated)
        gap = gap.view(gap.size(0), -1)
        
        # Fully connected layer
        out = self.fc(gap)
        
        return out

class SiameseNetworkX(nn.Module):
    def __init__(self):
        super(SiameseNetworkX, self).__init__()
        self.backbone = ResNetBackbone()
        self.attention = DualAttentionNetwork(512)
        # self.pooling = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512, 1024)  # Output 1024-dimensional embedding

    def forward(self, x1):
        out1 = self.backbone(x1)
        out1 = self.attention(out1)
        return out1


class DogNoseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir

        self.label_img_pairs = []
        self.transform = transform

        # Walk through the directory to list all images
        for class_id, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_filename in os.listdir(class_dir):
                    self.label_img_pairs.append((os.path.join(class_dir, img_filename), class_name))     

    def __getitem__(self, index):
        img_path, label = self.label_img_pairs[index]
        img = Image.open(img_path)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.label_img_pairs)

class QueryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Directory with all the subdirectories of classes.
        """
        self.root_dir = root_dir
        self.image_paths = []  # Store image paths
        self.labels = []       # Store class labels for each image
        self.transform = transform

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
        img = Image.open(img_path)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image_paths)


def load_model(model_path):
    model = SiameseNetworkX().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def generate_embeddings(model, dataset):
    embeddings = defaultdict(list)
    for img, class_Id in tqdm(dataset):
        img = img.to(device)
        with torch.no_grad():
            emb = model(img).detach().cpu().numpy()  # Convert to NumPy array after moving to CPU
        embeddings[class_Id].append(emb)
    return embeddings

def calculate_accuracy(model, query_dataset, db_embeddings, top_k=(1, 3, 5)):
    top_k_correct = {k: 0 for k in top_k}
    total = 0

    for img, label in tqdm(query_dataset):
        img = img.to(device)
        with torch.no_grad():
            query_emb = model(img).detach().cpu().numpy()  # Convert to NumPy array after moving to CPU
        
        distances = []
        for db_class_id, embeddings in db_embeddings.items():
            for emb in embeddings:
                distance = np.linalg.norm(query_emb - emb)
                distances.append((distance, db_class_id))
        
        closest_classes = [class_id for _, class_id in heapq.nsmallest(max(top_k), distances)]
        
        for k in top_k:
            if label in closest_classes[:k]:
                top_k_correct[k] += 1
        total += 1
    
    top_k_accuracies = {k: correct / total for k, correct in top_k_correct.items()}
    return top_k_accuracies

if __name__ == '__main__':
    train_dir = '400_classes_dataset/train'
    db_dir = '101_classes_dataset_for_testing/db_images'
    q_dir = '101_classes_dataset_for_testing/query_images'
    
    db_dataset = DogNoseDataset(root_dir=db_dir, transform=transform)
    query_dataset = QueryDataset(root_dir=q_dir, transform=transform)

    db_dataset_loader = DataLoader(db_dataset, batch_size=1, shuffle=False)
    query_dataset_loader = DataLoader(query_dataset, batch_size=1, shuffle=False)
    
    model_path = 'best_acc_400.pth'
    model = load_model(model_path)

    db_embeddings = generate_embeddings(model, db_dataset_loader)
    
    accuracies = calculate_accuracy(model, query_dataset_loader, db_embeddings, top_k=(1, 3, 5))
    print(f"Top-1 Accuracy: {accuracies[1] * 100:.4f}%")
    print(f"Top-3 Accuracy: {accuracies[3] * 100:.4f}%")
    print(f"Top-5 Accuracy: {accuracies[5] * 100:.4f}%")