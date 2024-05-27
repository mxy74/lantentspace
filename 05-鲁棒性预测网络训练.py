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
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#读取模型数据
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
        return len(self.pic_filenames)

if args.dataset == "MNIST":
	transform = transforms.Compose([transforms.ToTensor(),transforms.Resize([32,32]), transforms.Normalize([0.5], [0.5])])
	train_dataloader = DataLoader(Mydata_sets(pic_path, robustness, True, transform = transform), batch_size=128, shuffle=False)
	test_dataloader = DataLoader(Mydata_sets(pic_path, robustness, False, transform = transform), batch_size=32, shuffle=False)
elif args.dataset == "CIFAR10":
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914,0.4822,0.4465], [0.2023,0.1994,0.2010])]) #CIFAR10数据集的均值和方差，多处网络验证
	train_dataloader = DataLoader(Mydata_sets(pic_path, robustness, True, transform = transform), batch_size=128, shuffle=False)
	test_dataloader = DataLoader(Mydata_sets(pic_path, robustness, False, transform = transform), batch_size=32, shuffle=False)


rob_predictor = model_all.get_rob_predictor(args.dataset, args.model_name).to(device)
rob_predictor.train()
print(rob_predictor)

optimizer = optim.Adam(rob_predictor.parameters(), lr=0.0002)
criterion = nn.MSELoss()

epoches = 200
batch_size = 256
threshold = 0.7
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
    if pcc >= threshold:
        print("达到阈值，停止训练")
        break
    #print(str(epoch) + "[**running_loss**]: ",running_loss.item())

torch.save(rob_predictor.state_dict(), "./model_files/" + args.dataset + "/checkpoints/rob_predictor/kjl_rob_predictor_" + args.model_name + ".pt")
print('Finished Training')