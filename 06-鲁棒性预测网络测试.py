import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os 
import time
from tqdm import tqdm
import codecs
import model_files as model_all #模型的连接在__init__.py中体现
from PIL import Image 
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import scipy.stats as stats
from sklearn.metrics import mean_absolute_error

import argparse
import copy

'''
arg主要是模型选择参数
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--gpu_id', type=str, default="0")
args = parser.parse_args()

device = torch.device("cuda:" + args.gpu_id)

if args.dataset == "SteeringAngle":
    #读取模型数据
    model = model_all.get_DNN_model(args.dataset,args.model_name)
    model.load_state_dict(torch.load("./model_files/" + args.dataset + "/checkpoints/regre_model/" + args.model_name + '.pt')["net_state_dict"])
    model.eval()
    model = model.to(device)
    sub_string = "_wrongAngel=10.0epsilon=0.001epsilon_step=0.001"
    
else:
    #读取模型数据
    model = model_all.get_DNN_model(args.dataset,args.model_name)
    model.load_state_dict(torch.load("./model_files/" + args.dataset + "/checkpoints/classify_model/" + args.model_name + '.pt'))
    model.eval()
    model = model.to(device)
    sub_string = ""
    

#读取数据
pic_path = "./static/data/"+ args.dataset +"/pic/random_50k_png"
robustness = torch.load("./系统以外的资源/对抗鲁棒性PDG/" + args.dataset + "/" + args.model_name + "/PDG_robustness_" + args.model_name + sub_string + ".pt")


class Mydata_sets(Dataset):
    
    def __init__(self, pic_path, pic_robustness, train, transform):
        super(Mydata_sets, self).__init__()
        self.root_dir = pic_path
        self.transform = transform
        filenames = os.listdir(pic_path)
        filenames.sort(key=lambda x: int(x[4:-4])) # pic_0.jpg,根据编号进行排序
        if(train == True):
            self.pic_filenames = filenames[:40000]
            self.pic_robustness = pic_robustness[:40000]
        else:
            self.pic_filenames = filenames[40000:]
            self.pic_robustness = pic_robustness[40000:]

    def __getitem__(self, index):
        img_name = self.pic_filenames[index]
        img = Image.open(os.path.join(self.root_dir, img_name))
        img = self.transform(img)
        pic_robustness = self.pic_robustness[index]
        return img, pic_robustness

    def __len__(self):
        return len(self.pic_robustness)

if args.dataset == "SteeringAngle":
    # 定义一个字典来存储中间层的输出
    activation = {}
    # 针对回归模型，需要hook函数获取倒数第二层的输出
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    # 注册钩子函数到第二层
    model.pool1.register_forward_hook(get_activation('pool1'))

if args.dataset == "MNIST":
	transform = transforms.Compose([transforms.ToTensor(),transforms.Resize([32,32]), transforms.Normalize([0.5], [0.5])])
	train_dataloader = DataLoader(Mydata_sets(pic_path, robustness, True, transform = transform), batch_size=10, shuffle=False)
	test_dataloader = DataLoader(Mydata_sets(pic_path, robustness, False, transform = transform), batch_size=10, shuffle=False)
elif args.dataset == "CIFAR10":
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914,0.4822,0.4465], [0.2023,0.1994,0.2010])]) #CIFAR10数据集的均值和方差，多处网络验证
	train_dataloader = DataLoader(Mydata_sets(pic_path, robustness, True, transform = transform), batch_size=10, shuffle=False)
	test_dataloader = DataLoader(Mydata_sets(pic_path, robustness, False, transform = transform), batch_size=10, shuffle=False)
elif args.dataset == "SteeringAngle":
    print("数据集为SteeringAngle")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])]) #CIFAR10数据集的均值和方差，多处网络验证
    train_dataloader = DataLoader(Mydata_sets(pic_path, robustness, True, transform = transform), batch_size=10, shuffle=False)
    test_dataloader = DataLoader(Mydata_sets(pic_path, robustness, False, transform = transform), batch_size=32, shuffle=False)


rob_predictor = model_all.get_rob_predictor(args.dataset, args.model_name).to(device)
rob_predictor.load_state_dict(torch.load("./model_files/" + args.dataset + "/checkpoints/rob_predictor/kjl_rob_predictor_" + args.model_name + sub_string + ".pt"))
rob_predictor.eval()
print(rob_predictor)

#测试
flag = 0
test_dataloader = tqdm(test_dataloader)
with torch.no_grad():
    for i, (inputs, robs) in enumerate(test_dataloader):
        inputs, robs = inputs.to(device),torch.tensor(robs).to(torch.float32).to(device)
        # =========== forward ==========
        penultimate_layer = model(inputs)
        if args.dataset == "SteeringAngle": #回归模型通过hook钩子获得倒数第二层的输出
            penultimate_layer = activation["pool1"]
            penultimate_layer = penultimate_layer.view(penultimate_layer.size(0), -1)
            # print("penultimate_layer.shape: ", penultimate_layer.shape)

        outputs = rob_predictor(penultimate_layer)
        outputs = outputs.squeeze(1)
        # print("penultimate_layer: ", penultimate_layer)
        if flag == 0:
            print("outputs: ",outputs)
            print("robs: ",robs)
            a = input("press 0 to break: ")
        if a == "0":
            flag = 1

        if i == 0:
            predict_robs = outputs
            true_robs = robs
        else:
            predict_robs = torch.cat((predict_robs, outputs))
            true_robs = torch.cat((true_robs, robs))

pcc = stats.pearsonr(predict_robs.cpu(), true_robs.cpu())[0]
mae =  mean_absolute_error(predict_robs.cpu(), true_robs.cpu())
print("原始pcc: ", pcc)
print("mae: ", mae)

test_dataloader = tqdm(test_dataloader)
test_rob = copy.deepcopy(rob_predictor)
test_rob.eval()
with torch.no_grad():
    for i, (inputs, robs) in enumerate(test_dataloader):
        inputs, robs = inputs.to(device),torch.tensor(robs).to(torch.float32).to(device)
        # =========== forward ==========
        penultimate_layer = model(inputs)
        if args.dataset == "SteeringAngle": #回归模型通过hook钩子获得倒数第二层的输出
            penultimate_layer = activation["pool1"]
            penultimate_layer = penultimate_layer.view(penultimate_layer.size(0), -1)
        outputs = test_rob(penultimate_layer)
        outputs = outputs.squeeze(1) # 让输出和rob的格式一样
        if i == 0:
            predict_robs = outputs
            true_robs = robs
        else:
            predict_robs = torch.cat((predict_robs, outputs))
            true_robs = torch.cat((true_robs, robs))

print(predict_robs.shape)
print(true_robs.shape)


pcc = stats.pearsonr(predict_robs.cpu(), true_robs.cpu())[0]
mae =  mean_absolute_error(predict_robs.cpu(), true_robs.cpu())
print("训练的pcc: ", pcc)
print("mae: ", mae)