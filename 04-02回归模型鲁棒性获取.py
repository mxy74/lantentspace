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
# from foolbox.criteria import Misclassification
from foolbox.criteria import Criterion
from foolbox.attacks.gradient_descent_base import BaseGradientDescent
from foolbox import distances
import eagerpy as ep
from typing import TypeVar, Any, Tuple
T = TypeVar("T")
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
parser.add_argument('--dataset', type=str, required=True, help="数据集类别")
parser.add_argument('--model_name', type=str, required=True, help="模型名字，例如ResNet20")
parser.add_argument('--weight_dir', type=str, default="./model_files/SteeringAngle/checkpoints/regre_model/", help="模型参数的位置")
parser.add_argument('--angle', type=float, default=5, help="多少的误差被认为是扰动成功")
parser.add_argument('--epsilon', type=float, default=0.001, help="起始的扰动强度是多少")
parser.add_argument('--epsilon_step', type=float, default=0.001, help="每次迭代增加多少扰动强度")
parser.add_argument('--epoch', type=str, default='199', help="决定用那个epoch的模型参数")
parser.add_argument('--gpu_id', type=str, default='1')
args = parser.parse_args()

# 设备检测
device = torch.device("cuda:" + args.gpu_id)

model = model_all.get_DNN_model(args.dataset, args.model_name)
model.load_state_dict(torch.load(args.weight_dir + args.model_name + '.pt', map_location=device)['net_state_dict'])
model.eval()
model = model.to(device)


#读取数据
pic_path = "./static/data/"+ args.dataset +"/pic/random_50k"
filenames = os.listdir(pic_path)
filenames.sort(key=lambda x: int(x[4:-4])) # pic_0.jpg,根据编号进行排序

if args.dataset == "SteeringAngle":
    transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ]
        )
    preprocessing = dict(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], axis=-3)


# 自定义梯度下降对抗攻击 
# 攻击连接：https://foolbox.readthedocs.io/en/stable/modules/attacks.html#
# 参照/home/kuangjielong/.conda/envs/python3_7/lib/python3.7/site-packages/foolbox/attacks/gradient_descent_base.py
# 以及 /home/kuangjielong/.conda/envs/python3_7/lib/python3.7/site-packages/foolbox/attacks/porjected_gradient_decent.py
class RegreAttack():
    def __init__(self, rel_stepsize: float = 0.01 / 0.3, steps: int = 40, bounds = (0, 1)):
        self.rel_stepsize = rel_stepsize
        self.steps = steps
        self.bounds = bounds
        self.loss_fn = nn.MSELoss()

    def get_random_start(self, x0, epsilon: float):
        x0, restore_type = ep.astensor_(x0)
        x = x0 + ep.uniform(x0, x0.shape, -epsilon, epsilon)
        return restore_type(x)

    def normalize(self, gradients):
        return gradients.sign()

    # 将
    def project(self, x, x0, epsilon: float):
        x, x_restore_type = ep.astensor_(x)
        x0, x0_restore_type = ep.astensor_(x0)
        temp = x0 + ep.clip(x - x0, -epsilon, epsilon)
        return x_restore_type(temp)
    
    def run(self, model, inputs, epsilon: float):
        x0 = inputs.clone().detach().requires_grad_()
        _min, _max = self.bounds
        init_label = model(x0)
        # print("攻击中的init_label: ", init_label)

        # 随机添加扰动，生成新的样本
        x = self.get_random_start(x0, epsilon)
        x = ep.clip(x, _min, _max)
        x = x.clone().detach().requires_grad_()

        # 梯度下降攻击
        for _ in range(self.steps):
            model.zero_grad()  # 清空模型参数的梯度信息
            # 计算扰动后的损失
            output = model(x)
            mse_loss = self.loss_fn(init_label, output)
            mse_loss.backward(retain_graph=True)
            gradients = x.grad.data

            gradients = self.normalize(gradients)
            x = x + self.rel_stepsize * gradients * epsilon
            x = self.project(x, x0, epsilon)
            x = ep.clip(x, _min, _max)
            x.grad = None  # 清空 x 的梯度信息
            x = x.clone().detach_().requires_grad_()
        return x

attack = RegreAttack()


#数据
robustness = []
for i, file in enumerate(filenames):
    position = pic_path + "/" + file
    if args.dataset == "SteeringAngle":
        image = Image.open(position)
    input_image = transform(image)
    input_image = input_image.unsqueeze(0).to(device)  # 增加一个维度维batch维度
    print("image.shape:",input_image.shape)
    label = model(input_image)[0]
    print(file)
    # print("外边的label：", label)
    # a = input("press 0 to break")
    # if a != "0":
    #     continue
    flag = False
    epsilon = args.epsilon
    epsilon_step = args.epsilon_step
    worng_angel = args.angle / 160.0
    print("进入计算")
    while(flag == False):
        if epsilon >= 0.033:
            break
        adv = attack.run(model, input_image, epsilon=epsilon)
        label2 = model(adv)[0]
        print("误差: ", abs(label2 - label))
        if abs(label2 - label) >= worng_angel: # 0~1对应-80~80度，所以0.03125对应的是误差超过5度
            flag=True
        epsilon = epsilon + epsilon_step
    print("type(epsilon)",type(epsilon))
    print("epsilon: ", epsilon-epsilon_step)
    print("*"*100)

    robustness.append(epsilon-epsilon_step)

    print("len(robutsness): ", len(robustness))

    log_string = "_wrongAngel=" + str(args.angle) + "epsilon=" + str(args.epsilon) + "epsilon_step=" + str(args.epsilon_step)
    torch.save(robustness, "./系统以外的资源/对抗鲁棒性PDG/" + args.dataset + "/" + args.model_name + "/PDG_robustness_" + args.model_name + log_string + ".pt") 
    