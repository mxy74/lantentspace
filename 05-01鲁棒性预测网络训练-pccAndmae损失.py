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
import copy

import scipy.stats as stats
from sklearn.metrics import mean_absolute_error

import argparse

'''
arg主要是模型选择参数
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--gpu_id', type=str, default="0")
args = parser.parse_args()

device = torch.device("cuda:" + args.gpu_id)

#读取模型数据
if args.dataset == "SteeringAngle":
    model = model_all.get_DNN_model(args.dataset,args.model_name)
    model.load_state_dict(torch.load("./model_files/" + args.dataset + "/checkpoints/regre_model/" + args.model_name + '.pt')["net_state_dict"])
    model.eval()
    model = model.to(device)
else: 
    model = model_all.get_DNN_model(args.dataset,args.model_name)
    model.load_state_dict(torch.load("./model_files/" + args.dataset + "/checkpoints/classify_model/" + args.model_name + '.pt'))
    model.eval()
    model = model.to(device)


#读取数据
pic_path = "./static/data/"+ args.dataset +"/pic/random_50k"
robustness = torch.load("./系统以外的资源/对抗鲁棒性PDG/" + args.dataset + "/" + args.model_name + "/PDG_robustness_" + args.model_name + ".pt")

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

if args.dataset == "MNIST":
	transform = transforms.Compose([transforms.ToTensor(),transforms.Resize([32,32]), transforms.Normalize([0.5], [0.5])])
	train_dataloader = DataLoader(Mydata_sets(pic_path, robustness, True, transform = transform), batch_size=128, shuffle=False)
	test_dataloader = DataLoader(Mydata_sets(pic_path, robustness, False, transform = transform), batch_size=32, shuffle=False)
elif args.dataset == "CIFAR10":
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914,0.4822,0.4465], [0.2023,0.1994,0.2010])]) #CIFAR10数据集的均值和方差，多处网络验证
	train_dataloader = DataLoader(Mydata_sets(pic_path, robustness, True, transform = transform), batch_size=128, shuffle=False)
	test_dataloader = DataLoader(Mydata_sets(pic_path, robustness, False, transform = transform), batch_size=32, shuffle=False)
elif args.dataset == "SteeringAngle":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
    train_dataloader = DataLoader(Mydata_sets(pic_path, robustness, True, transform = transform), batch_size=128, shuffle=False)
    test_dataloader = DataLoader(Mydata_sets(pic_path, robustness, False, transform = transform), batch_size=32, shuffle=False)


rob_predictor = model_all.get_rob_predictor(args.dataset, args.model_name).to(device)
rob_predictor.train()
print(rob_predictor)

# 使用皮尔僧和MAE相关系数作为模型的损失
class PearsonMAELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(PearsonMAELoss, self).__init__()
        self.alpha = alpha
        self.mae_loss = nn.L1Loss()

    def forward(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        pearson_loss = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        pearson_loss = 1 - pearson_loss  # 将相关系数转换为距离，即损失函数，取1减去相关系数
        mae_loss = self.mae_loss(x, y)
        loss = self.alpha * pearson_loss + (1 - self.alpha) * mae_loss
        return loss

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

optimizer = optim.Adam(rob_predictor.parameters(), lr=0.0002)
criterion = PearsonMAELoss()

epoches = 200
threshold = 0.01
#进行训练
for epoch in range(epoches):
    running_loss = 0.0
    total = 0
    correct = 0
    train_dataloader = tqdm(train_dataloader)
    for i, (inputs, robs) in enumerate(train_dataloader):
        inputs, robs = inputs.to(device),torch.tensor(robs).to(torch.float32).to(device)
        # =========== forward ==========
        penultimate_layer = model(inputs)
        if args.dataset == "SteeringAngle": #回归模型通过hook钩子获得倒数第二层的输出
            penultimate_layer = activation["pool1"]
            penultimate_layer = penultimate_layer.view(penultimate_layer.size(0), -1)
        outputs = rob_predictor(penultimate_layer)
        outputs = outputs.squeeze(1) # 让输出和rob的格式一样
        loss = criterion(outputs, robs)
        # =========== backward ==========
        optimizer.zero_grad() # zero the parameter gradients
        loss.backward()
        optimizer.step()
        # =========== logging ==========
        running_loss += loss.data
        # ========== show process ==========
        description = 'epoch: %d , current_loss: %.4f, running_loss: %.4f' % (epoch, loss.item(), running_loss)
        train_dataloader.set_description(description)

    
    # 使用测试集判断是否停止训练pcc比较高就停止训练
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

    pcc = stats.pearsonr(predict_robs.cpu(), true_robs.cpu())[0]
    mae =  mean_absolute_error(predict_robs.cpu(), true_robs.cpu())
    print("pcc: ", pcc)
    print("mae: ", mae)
    # 判断是否达到阈值，如果是则停止训练
    if running_loss <= threshold:
        print("达到阈值，停止训练")
        break
    #print(str(epoch) + "[**running_loss**]: ",running_loss.item())

torch.save(rob_predictor.state_dict(), "./model_files/" + args.dataset + "/checkpoints/rob_predictor/kjl_rob_predictor_" + args.model_name + ".pt")
print('Finished Training')