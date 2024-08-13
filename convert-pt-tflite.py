import torch
import torch.nn as nn
from torchvision.models import resnet152
import torch.nn.functional as F
import ai_edge_torch
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F



device = torch.device("cpu")
print(device)

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        # Load a pre-trained ResNet-152 and remove the last GAP and FC
        base_model = resnet152(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-2])

        # Additional blocks to reduce channel dimensions
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

model = SiameseNetworkX()
model.load_state_dict(torch.load('/home/athena/Documents/GitHub/Dog-NosePrint-Recognition/best_acc_400.pth', map_location=torch.device('cpu')))
model.eval()
transform_val = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to 256x256
    # transforms.ToTensor(),          # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])
dummy_input = (transform_val(torch.randn( 3, 256, 256)).unsqueeze(0).to(device),)
edge_model = ai_edge_torch.convert(model.eval(), dummy_input)
