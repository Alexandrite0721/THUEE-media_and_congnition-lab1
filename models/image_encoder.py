# import torch.nn as nn
# import torchvision.models as models

# class ImageEncoder(nn.Module):
#     def __init__(self, embed_dim=256):
#         super().__init__()
#         self.resnet = models.resnet18(pretrained=True)
#         self.resnet.fc = nn.Identity()  # 移除最后的分类层
#         self.fc = nn.Linear(512, embed_dim)  # 映射到共享空间

#     def forward(self, images):
#         features = self.resnet(images)  # [batch, 512]
#         embeddings = self.fc(features)  # [batch, embed_dim]
#         return embeddings
# image_encoder.py
import torch.nn as nn
from .resnet_custom import ResNet18  # 替换 torch 的 resnet18

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.resnet = ResNet18()  # 自己写的网络
        self.fc = nn.Linear(256, embed_dim)  # 将 ResNet 输出映射到共享空间

    def forward(self, images):
        features = self.resnet(images)  # [batch, 512]
        embeddings = self.fc(features)  # [batch, embed_dim]
        return embeddings
