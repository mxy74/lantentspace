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

from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import model_files as model_all #模型的连接在__init__.py中体现

import argparse

'''
arg主要是模型选择参数
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help="数据集类别: CIFAR10或者MNIST")
parser.add_argument('--model_name', type=str, required=True, help="模型名字，例如ResNet20")
parser.add_argument('--weight_dir', type=str, default="./model_files/CIFAR10/checkpoints/classify_model/", help="模型参数的位置")
parser.add_argument('--epoch', type=str, default='199', help="决定用那个epoch的模型参数")
parser.add_argument('--gpu_id', type=str, default='0')
args = parser.parse_args()

# 设备检测
device = torch.device("cuda:" + args.gpu_id)

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
                transforms.Resize([32,32]),
                transforms.Normalize([0.5], [0.5])
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


fmodel = PyTorchModel(model, bounds=(0, 1), device = device, preprocessing=preprocessing)
attack = LinfPGD()
arr=[]

#数据
robustness = []
for i, file in enumerate(filenames):
    position = pic_path + "/" + file
    if args.dataset == "MNIST":
        image = Image.open(position).convert('L')
    elif args.dataset == "CIFAR10":
        image = Image.open(position)
    input_image = transform(image)
    input_image = input_image.unsqueeze(0).to(device)  # 增加一个维度维batch维度
    print("image.shape:",input_image.shape)
    output = model(input_image)
    _, label = torch.max(output.data, 1)
    label = label.to(device)
#    print(file)
#    print(label)
#    a = input("press 0 to break")
#    if a != "0":
#        continue
    flag = False
    epsilon = 0.001
    print("进入计算")
    transform2 = transforms.Compose([transforms.ToTensor()])
    image = transform2(image).unsqueeze(0).to(device)
    while(flag == False):
        if epsilon >= 0.033:
            break
        raw_advs, clipped_advs, success = attack(fmodel, image, label, epsilons=epsilon)
        flag=success
        epsilon = epsilon + 0.001

    print("type(epsilon)",type(epsilon))
    print("epsilon: ", epsilon)
    print("*"*100)

    robustness.append(epsilon-0.001)

    print("len(robutsness): ", len(robustness))

torch.save(robustness, "./系统以外的资源/对抗鲁棒性PDG/" + args.dataset + "/" + args.model_name + "/PDG_robustness_" + args.model_name + ".pt") 
    