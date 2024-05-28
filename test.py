import pickle
import torch
import torch.nn as nn
import torchvision
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
import csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import model_files as model_all #模型的连接在__init__.py中体现

import argparse

'''
arg主要是模型选择参数
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="CIFAR10", help="数据集类别: CIFAR10或者MNIST")
parser.add_argument('--model_name', type=str, default="ResNet20", help="模型名字，例如ResNet20")
parser.add_argument('--weight_dir', type=str, default="./model_files/CIFAR10/checkpoints/classify_model/", help="模型参数的位置")
parser.add_argument('--epoch', type=str, default='199', help="决定用那个epoch的模型参数")
parser.add_argument('--gpu_id', type=str, default='0')
args = parser.parse_args()

# 设备检测
device = torch.device("cuda:" + args.gpu_id  if torch.cuda.is_available() else "cpu")

model = model_all.get_DNN_model(args.dataset,args.model_name)
model.load_state_dict(torch.load(args.weight_dir + args.model_name + '.pt'))
model.eval()
model = model.to(device)


#读取数据
pic_path = "./static/data/"+ args.dataset +"/pic/random_50k"
filenames = os.listdir(pic_path)
filenames.sort(key=lambda x: int(x[4:-4])) # pic_0.jpg,根据编号进行排序


if args.dataset == "MNIST":
    transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([32,32])
            ]
        )
    preprocessing = dict(mean=[0.5], std=[0.5], axis=-3)
elif args.dataset == "CIFAR10":
    transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4914,0.4822,0.4465], [0.2023,0.1994,0.2010])
            ]
        )
    preprocessing = dict(mean=[0.4914,0.4822,0.4465], std=[0.2023,0.1994,0.2010], axis=-3)


pic_path = "./static/data/CIFAR10/pic/random_50k"
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914,0.4822,0.4465], [0.2023,0.1994,0.2010])])


class Mydata_sets(Dataset):
    
    def __init__(self, pic_path, train, transform):
        super(Mydata_sets, self).__init__()
        self.root_dir = pic_path
        self.transform = transform
        filenames = os.listdir(pic_path)
        filenames.sort(key=lambda x: int(x[4:-4])) # pic_0.jpg,根据编号进行排序
        if(train == True):
            self.pic_filenames = filenames[:40000]
        else:
            self.pic_filenames = filenames[40000:]

    def __getitem__(self, index):
        img_name = self.pic_filenames[index]
        img = Image.open(os.path.join(self.root_dir, img_name))
        img = self.transform(img)
        return img, img_name

    def __len__(self):
        return len(self.pic_filenames)


#读取模型数据
model2 = model_all.get_DNN_model("CIFAR10","ResNet20")
model2.load_state_dict(torch.load("./model_files/CIFAR10/checkpoints/classify_model/ResNet20.pt"))
model2.eval()
model2 = model2.to(device)

train_dataloader = DataLoader(Mydata_sets(pic_path, True, transform = transform), batch_size=1, shuffle=False)


for i, (img, img_name) in enumerate(train_dataloader):
    img = img.to(device)
    print("img_name: ",img_name)
    position = pic_path + "/" + img_name[0]
    image = Image.open(position)
    print("position: ",position)
    input_image = transform(image)
    input_image = input_image.unsqueeze(0).to(device)  # 增加一个维度维batch维度
    print("input_image.shape:",input_image.shape)
    print("img.shape:",img.shape)
    output1 = model(input_image)
    output2 = model2(img)
    print("output1: ",output1)
    print("output2: ",output2)
    a = input("press 0 to break: ")
    if a == "0":
        break




#数据
# robustness = []
# for i, file in enumerate(filenames):
#     position = pic_path + "/" + file
#     if args.dataset == "MNIST":
#         image = Image.open(position).convert('L')
#     elif args.dataset == "CIFAR10":
#         image = Image.open(position)
#     print("position: ",position)
#     input_image = transform(image)
#     input_image = input_image.unsqueeze(0).to(device)  # 增加一个维度维batch维度
#     print("image.shape:",input_image.shape)
#     output = model(input_image)
#     print(output)
#     a = input("press 0 to break: ")
#     if a == "0":
#         break