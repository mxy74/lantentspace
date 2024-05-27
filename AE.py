import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import os
from PIL import Image

# 原始图像做输入，3072
# class AutoencoderWithAttributeLoss(nn.Module):
#     def __init__(self, input_dim, encoding_dim):
#         super(AutoencoderWithAttributeLoss, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, encoding_dim),
#             nn.ReLU()
#         )
#         self.attribute_layer = nn.Linear(encoding_dim, 10)
#         self.decoder = nn.Sequential(
#             nn.Linear(encoding_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, input_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         attribute_output = self.attribute_layer(encoded)
#         decoded = self.decoder(encoded)
#         return attribute_output, decoded

# resnet_20feature做输入，64
class AutoencoderWithAttributeLoss(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoencoderWithAttributeLoss, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.ReLU()
        )
        self.attribute_layer = nn.Linear(encoding_dim, 10)
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        attribute_output = self.attribute_layer(encoded)
        decoded = self.decoder(encoded)
        return attribute_output, decoded
# 自定义损失函数
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, reconstructed, original, attribute_output, confidence):
        # 计算重构损失
        recon_loss = nn.MSELoss()(reconstructed, original)

        # 计算属性层和置信度的损失
        attribute_loss = nn.CrossEntropyLoss()(attribute_output, confidence)  # 使用最高置信度作为属性值
        # 将重构损失和属性层损失结合成最终的损失
        loss = recon_loss + attribute_loss

        return loss, recon_loss, attribute_loss

        # return attribute_loss


class Mydata_sets(Dataset):

    def __init__(self, path, transform=None):
        super(Mydata_sets, self).__init__()
        self.root_dir = path
        self.img_names = os.listdir(self.root_dir)
        self.transform = transform

    def __getitem__(self, index): # 返回transform后的图像和文件的id序号
        img_name = self.img_names[index]
        img = Image.open(os.path.join(self.root_dir, img_name))
        id_name = torch.tensor(int(img_name[4:-4])) #pic_xx.jpg
        if self.transform is not None:
            img = self.transform(img)
        return img, id_name

    def __len__(self):
        return len(self.img_names)

class Mydata_sets1(Dataset):

    def __init__(self, path,label, transform=None):
        super(Mydata_sets1, self).__init__()
        self.root_dir = path
        self.img_names = os.listdir(self.root_dir)
        self.transform = transform
        self.label = label

    def __getitem__(self, index): # 返回transform后的图像和文件的id序号
        img_name = self.img_names[index]
        img = Image.open(os.path.join(self.root_dir, img_name))
        id_name = torch.tensor(int(img_name[4:-4])) #pic_xx.jpg
        if self.transform is not None:
            img = self.transform(img)
        img = img.reshape(-1)
        return img, id_name, self.label[id_name]

    def __len__(self):
        return len(self.img_names)


class Mydata_sets2(Dataset):

    def __init__(self, zs,labels):
        super(Mydata_sets2, self).__init__()
        self.zs = zs
        self.labels = labels

    def __getitem__(self, index):
        z = self.zs[index]
        labels = self.labels[index]
        return z, labels

    def __len__(self):
        return len(self.zs)