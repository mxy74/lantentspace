import torch
import torch.nn as nn
import sys
import torchvision.transforms.functional as TF
from sklearn.metrics import confusion_matrix

python_files_dir = "./python_files/"  # python工具包位置
sys.path.append(python_files_dir)
import fid_score as official_fid
import os

model_files_dir = "./model_files/"  # 模型位置
sys.path.append(model_files_dir)
from model_files.CIFAR10.models import ResNet20 as ResNet
from model_files.CIFAR10.models import Rob_predictor as my_Rob_predictor

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = '{"max_split_size_mb": 1024}'
import foolbox as fb
from foolbox.attacks import LinfPGD

import pickle
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy import spatial
import time
import PIL.Image
import copy
import eagerpy as ep

import torch
import torchvision.utils as utils
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
import random

device = torch.device("cuda:0")
cpu = torch.device("cpu")
kd_tree_number = 20  # 通过多少个最近邻生成样本


# 用来处理zs的类，方便使用batchsize
class Mydata_sets(Dataset):

    def __init__(self, zs,labels):
        super(Mydata_sets, self).__init__()
        self.zs = zs
        self.labels = labels

    def __getitem__(self, index):
        z = self.zs[index]
        labels = self.labels[index]
        return z, labels

    def __len__(self):
        return len(self.zs)


# 定义中间变量字典
activation = {}


# 用来获取模型中间层输出的hook
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


# 自定义梯度下降对抗攻击
# 攻击连接：https://foolbox.readthedocs.io/en/stable/modules/attacks.html#
# 参照/home/kuangjielong/.conda/envs/python3_7/lib/python3.7/site-packages/foolbox/attacks/gradient_descent_base.py
# 以及 /home/kuangjielong/.conda/envs/python3_7/lib/python3.7/site-packages/foolbox/attacks/porjected_gradient_decent.py
class RegreAttack():
    def __init__(self, rel_stepsize: float = 0.01 / 0.3, steps: int = 40, bounds=(0, 1)):
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


### -------------------------------------------------------------------------------
### 以下不同的插值公式
### -------------------------------------------------------------------------------
# 将对应坐标的z计算出来（网上的插值公式，最开始使用的版本）
def get_zs(nearest_distance, nearest_index, dict_zs):
    '''
    nearest_distance: n*k维
    nearest_index: n*k维
    dict_zs: 键是文件的id号, 值是对应的z（后面直接改成数组了，问题不大）
    '''
    n = len(nearest_index)  # n个样本
    k = len(nearest_index[0])  # k个近邻（默认每个都是k近邻）
    # print(nearest_distance)
    for iter in range(n):
        iter_distance = nearest_distance[iter]
        iter_index = nearest_index[iter]

        sum_distanceForIter = np.sum(iter_distance)  # 这k个近邻的距离总和
        for i, index in enumerate(iter_index):
            temp_z = torch.tensor(dict_zs[index])
            temp_distance = iter_distance[i]
            w = (sum_distanceForIter - temp_distance) / ((k - 1) * sum_distanceForIter)  # 对z进行权重
            if i == 0:
                z = temp_z * w
            else:
                z += temp_z * w
        z = z.unsqueeze(0)  # shape:[1*latent_dim]
        if iter == 0:
            zs = z
        else:
            zs = torch.cat((zs, z), dim=0)
        # print(zs.shape)
    return zs


# 尝试各种插值方式（使用get_information_other调用）##########################################################################################
# 最开始的版本，优化了一下代码
def get_zs_new(coordinates, kdTree_2D, latent_z, k=10):
    '''
    coordinates: n个要插值的坐标
    kdTree_2D: 降维后的2D坐标
    latent_z: 生成模型的输入潜向量，和kdTree_2D是一一对应关系
    k: 近邻的数量
    '''
    print("最开始的版本，优化了一下代码~~~~~~~~~~~ k:", k)
    # 直接一次查询所有坐标的k个近邻
    nearest_distance, nearest_index = kdTree_2D.query(coordinates, k=k)  # 这里的k为固定值
    origin_coordinates = kdTree_2D.data  # 获取kdtree中原始的坐标
    for iter in range(len(coordinates)):
        iter_distance = nearest_distance[iter]
        iter_index = nearest_index[iter]

        sum_distanceForIter = np.sum(iter_distance)  # 这k个近邻的距离总和
        for i, index in enumerate(iter_index):
            temp_z = torch.tensor(latent_z[index])
            temp_distance = iter_distance[i]
            w = (sum_distanceForIter - temp_distance) / ((k - 1) * sum_distanceForIter)  # 对z进行权重
            if i == 0:
                z = temp_z * w
            else:
                z += temp_z * w
        z = z.unsqueeze(0)  # shape:[1*latent_dim]
        if iter == 0:
            zs = z
        else:
            zs = torch.cat((zs, z), dim=0)
        # print(zs.shape)
    return zs


# 回归模型的版本，只插值z，不动y(y为最近的那一个)
def get_zs_new_regre(coordinates, kdTree_2D, latent_z, k=2):
    '''
    coordinates: n个要插值的坐标
    kdTree_2D: 降维后的2D坐标
    latent_z: 生成模型的输入潜向量，和kdTree_2D是一一对应关系
    k: 近邻的数量，也就是lagrange插值的节点数量
    '''
    print("最开始的版本，优化了一下代码, 不动y，y只取最近的~~~~~~~~~~~ k:", k)
    # 直接一次查询所有坐标的k个近邻
    nearest_distance, nearest_index = kdTree_2D.query(coordinates, k=k)  # 这里的k为固定值
    origin_coordinates = kdTree_2D.data  # 获取kdtree中原始的坐标
    for iter in range(len(coordinates)):
        iter_distance = nearest_distance[iter]
        iter_index = nearest_index[iter]

        sum_distanceForIter = np.sum(iter_distance)  # 这k个近邻的距离总和
        # 取最近的那一个y
        label_y_embed = latent_z[iter_index[0]][256:].clone()
        for i, index in enumerate(iter_index):
            temp_z = torch.tensor(latent_z[index])
            temp_distance = iter_distance[i]
            w = (sum_distanceForIter - temp_distance) / ((k - 1) * sum_distanceForIter)  # 对z进行权重
            if i == 0:
                z = temp_z[:256] * w
            else:
                z += temp_z[:256] * w
        # print(label_y_embed)
        # print("label_y_embed.shape: ", label_y_embed.shape)
        # print("前z.shape: ", z.shape)
        z = torch.cat((z, label_y_embed))
        # print("后z.shape: ", z.shape)
        z = z.unsqueeze(0)  # shape:[1*latent_dim]
        if iter == 0:
            zs = z
        else:
            zs = torch.cat((zs, z), dim=0)
        # print(zs.shape)
    return zs


# 拉格朗日插值(最没用)
def get_zs_lagrange(coordinates, kdTree_2D, latent_z, k=3):
    '''
    coordinates: n个要插值的坐标
    kdTree_2D: 降维后的2D坐标
    latent_z: 生成模型的输入潜向量，和kdTree_2D是一一对应关系
    k: 近邻的数量，也就是lagrange插值的节点数量
    '''
    # 直接一次查询所有坐标的k个近邻
    nearest_distance, nearest_index = kdTree_2D.query(coordinates, k=k)
    origin_coordinates = kdTree_2D.data  # 获取kdtree中原始的坐标
    for i, pos in enumerate(coordinates):  # 对每一个坐标进行插值
        # pos坐标对应的近邻下标
        pos_nearst_index = nearest_index[i]  # 其中有k个index，每个index对应kdTree_2D中的一个2维坐标
        for y in range(k):  # 这一层循环时循环lagrange公式中y那一层的
            w = 1.0
            count = 0
            for x in range(k):  # 这一层循环时循环lagrange公式中x那一层的，即x的连成公式
                if y != x:
                    count += 1
                    w *= np.linalg.norm(pos - origin_coordinates[pos_nearst_index[x]]) / np.linalg.norm(
                        origin_coordinates[pos_nearst_index[y]] - origin_coordinates[pos_nearst_index[x]])
            print("w: ", w)
            print("count: ", count)
            temp_z = latent_z[pos_nearst_index[y]].clone().detach()
            if y == 0:
                z_new = w * temp_z
            else:
                z_new += w * temp_z

        z_new = z_new.unsqueeze(0)
        if i == 0:
            zs = z_new
        else:
            zs = torch.cat((zs, z_new), dim=0)

    return zs


# 均值插值
def get_zs_average(coordinates, kdTree_2D, latent_z, k=50):
    '''
    coordinates: n个要插值的坐标
    kdTree_2D: 降维后的2D坐标
    latent_z: 生成模型的输入潜向量，和kdTree_2D是一一对应关系
    k: 近邻的数量，也就是lagrange插值的节点数量
    '''
    # 直接一次查询所有坐标的k个近邻
    nearest_distance, nearest_index = kdTree_2D.query(coordinates, k=k)
    origin_coordinates = kdTree_2D.data  # 获取kdtree中原始的坐标
    for i, pos in enumerate(coordinates):  # 对每一个坐标进行插值
        # pos坐标对应的近邻下标
        pos_nearst_index = nearest_index[i]  # 其中有k个index，每个index对应kdTree_2D中的一个2维坐标
        for j in range(k):
            temp_z = latent_z[pos_nearst_index[j]].clone().detach()
            if j == 0:
                z_new = temp_z
            else:
                z_new += temp_z
        z_new = z_new / k
        z_new = z_new.unsqueeze(0)
        if i == 0:
            zs = z_new
        else:
            zs = torch.cat((zs, z_new), dim=0)
    return zs


# 取最近的点为基点进行偏移插值
def get_zs_basic_point(coordinates, kdTree_2D, latent_z, k=10):
    '''
    coordinates: n个要插值的坐标
    kdTree_2D: 降维后的2D坐标
    latent_z: 生成模型的输入潜向量，和kdTree_2D是一一对应关系
    k: 近邻的数量，也就是lagrange插值的节点数量
    '''
    print("进入get_zs_basic_point~~~~~~~~~~~~~~~~")
    # 直接一次查询所有坐标的k个近邻
    nearest_distance, nearest_index = kdTree_2D.query(coordinates, k=k)
    origin_coordinates = kdTree_2D.data  # 获取kdtree中原始的坐标
    for i, pos in enumerate(coordinates):  # 对每一个坐标进行插值
        # pos坐标对应的近邻下标
        pos_nearst_index = nearest_index[i]  # 其中有k个index，每个index对应kdTree_2D中的一个2维坐标
        for j in range(k):
            temp_z = latent_z[pos_nearst_index[j]].clone().detach()
            if j == 0:
                z_add = temp_z
            else:
                z_add += temp_z

        # 取平均的z
        average_z = z_add / k
        # print(average_z)
        # 在最近的一个点上加上偏移
        nearest_point_v = latent_z[pos_nearst_index[0]].clone().detach()
        # print(nearest_point_v)
        offset = 0.1
        # print("offset: ", offset)
        z_new = nearest_point_v + offset * average_z
        z_new = z_new.unsqueeze(0)

        if i == 0:
            zs = z_new
        else:
            zs = torch.cat((zs, z_new), dim=0)
    return zs


# 网上的插值公式改版
def get_zs_online(coordinates, kdTree_2D, latent_z):
    '''
    coordinates: n个要插值的坐标
    kdTree_2D: 降维后的2D坐标
    latent_z: 生成模型的输入潜向量，和kdTree_2D是一一对应关系
    k: 近邻的数量，也就是lagrange插值的节点数量
    '''
    print("进入了网上插值公式的改版~~~~~~~~~~~")
    # 直接一次查询所有坐标的k个近邻
    nearest_distance, nearest_index = kdTree_2D.query(coordinates, k=2)  # 这里的k为固定值
    origin_coordinates = kdTree_2D.data  # 获取kdtree中原始的坐标
    for i, pos in enumerate(coordinates):  # 对每一个坐标进行插值
        # pos坐标对应的近邻下标
        pos_nearst_index = nearest_index[i]  # 其中有k个index，每个index对应kdTree_2D中的一个2维坐标
        pos_nearst_distance = nearest_distance[i]
        # 以最近邻的两个坐标之间的距离为分母
        fix_s = np.linalg.norm(origin_coordinates[pos_nearst_index[0]] - origin_coordinates[pos_nearst_index[1]])
        # 新坐标到这两个坐标之间的距离总和
        sum_distance = np.sum(pos_nearst_distance)
        print(fix_s)

        # 改的地方就在这里，分母为最近邻两个点之间的距离，而不是新的坐标点和原始点之间的距离总和了
        temp_z_0 = latent_z[pos_nearst_index[0]].clone().detach()
        temp_z_1 = latent_z[pos_nearst_index[1]].clone().detach()
        z_new = (sum_distance - pos_nearst_distance[0]) / (fix_s + sum_distance) * temp_z_0 + (
                sum_distance - pos_nearst_distance[1]) / (fix_s + sum_distance) * temp_z_1

        z_new = z_new.unsqueeze(0)
        if i == 0:
            zs = z_new
        else:
            zs = torch.cat((zs, z_new), dim=0)

    return zs


# 使用高斯函数作为权重函数的径向基函数插值（chatGPT给的）
def get_zs_gauss_GTP(coordinates, kdTree_2D, latent_z, k=2):
    '''
    coordinates: n个要插值的坐标
    kdTree_2D: 降维后的2D坐标
    latent_z: 生成模型的输入潜向量，和kdTree_2D是一一对应关系
    k: 近邻的数量，也就是lagrange插值的节点数量
    '''
    # 直接一次查询所有坐标的k个近邻
    nearest_distance, nearest_index = kdTree_2D.query(coordinates, k=k)
    origin_coordinates = kdTree_2D.data  # 获取kdtree中原始的坐标
    for i, pos in enumerate(coordinates):  # 对每一个坐标进行插值
        # pos坐标对应的近邻下标
        pos_nearst_index = nearest_index[i]  # 其中有k个index，每个index对应kdTree_2D中的一个2维坐标
        pos_nearst_distance = nearest_distance[i]
        sum_distance = np.sum(pos_nearst_distance)
        for y in range(k):  # 这一层循环时循环lagrange公式中y那一层的
            dist = pos_nearst_distance[y]
            w = np.exp(-dist ** 2 / (2 * sum_distance ** 2))
            print("w: ", w)
            temp_z = latent_z[pos_nearst_index[y]].clone().detach()
            if y == 0:
                z_new = w * temp_z
            else:
                z_new += w * temp_z

        z_new = z_new.unsqueeze(0)
        if i == 0:
            zs = z_new
        else:
            zs = torch.cat((zs, z_new), dim=0)

    return zs


# 想办法防止近邻的点粘在一块 效果最好的！！！！！！！！
def get_zs_prevent_stick(coordinates, kdTree_2D, latent_z, k=20):
    '''
    nearest_distance: n*k维
    nearest_index: n*k维
    dict_zs: 键是文件的id号, 值是对应的z（后面直接改成数组了，问题不大）
    '''
    print("进入了防止粘在一块~~~~~~~~~~~~~~~~~~;k=", k)
    # 直接一次查询所有坐标的k个近邻
    nearest_distance, nearest_index = kdTree_2D.query(coordinates, k=k)
    origin_coordinates = kdTree_2D.data  # 获取kdtree中原始的坐标
    for i, pos in enumerate(coordinates):  # 对每一个坐标进行插值
        # pos坐标对应的近邻下标
        pos_nearst_index = nearest_index[i]  # 其中有k个index，每个index对应kdTree_2D中的一个2维坐标
        pos_nearst_distance = nearest_distance[i]

        # 最近邻的坐标点，以及最近的距离
        most_nearst_pos = origin_coordinates[pos_nearst_index[0]]
        most_nearst_dis = pos_nearst_distance[0]

        # 利用三角形,找到第二个插值基点，让两边之和越接近第三边，就越是钝角，就越合理
        s1 = most_nearst_dis
        best_index = 1  # 默认第二个最近邻最好
        min_dif = 100
        for j in range(1, k):
            cur_pos = origin_coordinates[pos_nearst_index[j]]
            s2 = pos_nearst_distance[j]
            s3 = np.linalg.norm(most_nearst_pos - cur_pos)
            if (s1 + s2) - s3 < min_dif:  # 两边之和大于等于第三边，所以不用绝对值
                min_dif = (s1 + s2) - s3
                best_index = j

        temp_z_0 = latent_z[pos_nearst_index[0]].clone().detach()
        temp_z_1 = latent_z[pos_nearst_index[best_index]].clone().detach()
        sum_distance = most_nearst_dis + pos_nearst_distance[best_index]
        z_new = (sum_distance - most_nearst_dis) / (sum_distance) * temp_z_0 + (
                sum_distance - pos_nearst_distance[best_index]) / (sum_distance) * temp_z_1

        z_new = z_new.unsqueeze(0)
        if i == 0:
            zs = z_new
        else:
            zs = torch.cat((zs, z_new), dim=0)

    return zs


# 插值函数，不对类向量进行插值，类取最近的那一个点
def get_zs_prevent_stick_not_class(coordinates, kdTree_2D, latent_z, k=200):
    '''
    nearest_distance: n*k维
    nearest_index: n*k维
    dict_zs: 键是文件的id号, 值是对应的z（后面直接改成数组了，问题不大）
    '''
    print("进入了防止粘在一块，并且不对控制类别的向量进行插值~~~~~~~~~~~~~~~~~~")
    # 直接一次查询所有坐标的k个近邻
    nearest_distance, nearest_index = kdTree_2D.query(coordinates, k=k)
    origin_coordinates = kdTree_2D.data  # 获取kdtree中原始的坐标
    for i, pos in enumerate(coordinates):  # 对每一个坐标进行插值
        # pos坐标对应的近邻下标
        pos_nearst_index = nearest_index[i]  # 其中有k个index，每个index对应kdTree_2D中的一个2维坐标
        pos_nearst_distance = nearest_distance[i]

        ##########################test
        # print("当前坐标：", pos)
        # for j in range(len(pos_nearst_index)):
        #     print("最近的坐标：", origin_coordinates[pos_nearst_index[j]])
        #     print("最近的距离：", pos_nearst_distance[j])
        #     print("对应的类向量：", latent_z[pos_nearst_index[j]][-128:])
        # print("暂时结束")
        # return
        ##########################end

        # 最近邻的坐标点，以及最近的距离
        most_nearst_pos = origin_coordinates[pos_nearst_index[0]]
        most_nearst_dis = pos_nearst_distance[0]

        # 利用三角形,找到第二个插值基点，让两边之和越接近第三边，就越是钝角，就越合理
        s1 = most_nearst_dis
        best_index = 1  # 默认第二个最近邻最好
        min_dif = 100

        fun_dic = {}
        for j in range(1, k):
            cur_pos = origin_coordinates[pos_nearst_index[j]]
            s2 = pos_nearst_distance[j]
            s3 = np.linalg.norm(most_nearst_pos - cur_pos)
            if (s1 + s2) - s3 < min_dif:  # 两边之和大于等于第三边，所以不用绝对值
                min_dif = (s1 + s2) - s3
                best_index = j

            # 统计一下最近邻居中最多的类别###################test
        #     class_v = str(latent_z[pos_nearst_index[j]][-128:].detach().numpy())
        #     # print("当前类别向量：",class_v)
        #     if class_v in fun_dic.keys():
        #         fun_dic[class_v] += 1
        #     else:
        #         fun_dic[class_v] = 1
        #     # print(latent_z[pos_nearst_index[j]][-128:])
        # print("当前数量向量数量为：", len(fun_dic.keys()))
        # print("对应的值：", list(fun_dic.values()))
        # if i > 20:
        #     return
        ################################################end

        temp_z_0 = latent_z[pos_nearst_index[0]].clone().detach()
        temp_z_1 = latent_z[pos_nearst_index[best_index]].clone().detach()
        sum_distance = most_nearst_dis + pos_nearst_distance[best_index]
        z_new = (sum_distance - most_nearst_dis) / (sum_distance) * temp_z_0 + (
                sum_distance - pos_nearst_distance[best_index]) / (sum_distance) * temp_z_1

        z_new[-128:] = temp_z_0[-128:]  # 不修改类标签

        z_new = z_new.unsqueeze(0)
        if i == 0:
            zs = z_new
        else:
            zs = torch.cat((zs, z_new), dim=0)

    return zs


# 回归模型的插值, 只变z不变y(y就取最近的)
def get_zs_prevent_stick_regre(coordinates, kdTree_2D, latent_z, k=20):
    '''
    nearest_distance: n*k维
    nearest_index: n*k维
    dict_zs: 键是文件的id号, 值是对应的z（后面直接改成数组了，问题不大）
    '''
    print("进入了防止回归模型专用粘在一块~~~~~~~~~~~~~~~~~~;k=", k)
    # 直接一次查询所有坐标的k个近邻
    nearest_distance, nearest_index = kdTree_2D.query(coordinates, k=k)
    origin_coordinates = kdTree_2D.data  # 获取kdtree中原始的坐标
    for i, pos in enumerate(coordinates):  # 对每一个坐标进行插值
        # pos坐标对应的近邻下标
        pos_nearst_index = nearest_index[i]  # 其中有k个index，每个index对应kdTree_2D中的一个2维坐标
        pos_nearst_distance = nearest_distance[i]

        # 最近邻的坐标点，以及最近的距离
        most_nearst_pos = origin_coordinates[pos_nearst_index[0]]
        most_nearst_dis = pos_nearst_distance[0]

        # 利用三角形,找到第二个插值基点，让两边之和越接近第三边，就越是钝角，就越合理
        s1 = most_nearst_dis
        best_index = 1  # 默认第二个最近邻最好
        min_dif = 100
        for j in range(1, k):
            cur_pos = origin_coordinates[pos_nearst_index[j]]
            s2 = pos_nearst_distance[j]
            s3 = np.linalg.norm(most_nearst_pos - cur_pos)
            if (s1 + s2) - s3 < min_dif:  # 两边之和大于等于第三边，所以不用绝对值
                min_dif = (s1 + s2) - s3
                best_index = j

        temp_z_0 = latent_z[pos_nearst_index[0]].clone().detach()[:256]
        label_0 = latent_z[pos_nearst_index[0]].clone().detach()[256:]
        temp_z_1 = latent_z[pos_nearst_index[best_index]].clone().detach()[:256]
        sum_distance = most_nearst_dis + pos_nearst_distance[best_index]
        z_new = (sum_distance - most_nearst_dis) / (sum_distance) * temp_z_0 + (
                sum_distance - pos_nearst_distance[best_index]) / (sum_distance) * temp_z_1
        # print("cat前，z_new.shape: ", z_new.shape)
        z_new = torch.cat((z_new, label_0))
        # print("cat后，z_new.shape: ", z_new.shape)

        z_new = z_new.unsqueeze(0)
        if i == 0:
            zs = z_new
        else:
            zs = torch.cat((zs, z_new), dim=0)

    return zs


# 插值函数，最远最近插值方法
def get_zs_farthest_and_closeet(coordinates, kdTree_2D, latent_z, k=50000):
    '''
    nearest_distance: n*k维
    nearest_index: n*k维
    dict_zs: 键是文件的id号, 值是对应的z（后面直接改成数组了，问题不大）
    '''
    print("进入了最近最远插值~~~~~~~~~~~~~~~~~~")
    # 直接一次查询所有坐标的k个近邻
    nearest_distance, nearest_index = kdTree_2D.query(coordinates, k=k)
    origin_coordinates = kdTree_2D.data  # 获取kdtree中原始的坐标
    for i, pos in enumerate(coordinates):  # 对每一个坐标进行插值
        # pos坐标对应的近邻下标
        pos_nearst_index = nearest_index[i]  # 其中有k个index，每个index对应kdTree_2D中的一个2维坐标
        pos_nearst_distance = nearest_distance[i]

        # 最近邻的距离
        most_nearst_dis = pos_nearst_distance[0]

        # 最远点的的距离
        most_farthest_dis = pos_nearst_distance[-1]

        temp_z_0 = latent_z[pos_nearst_index[0]].clone().detach()
        temp_z_1 = latent_z[pos_nearst_index[-1]].clone().detach()
        sum_distance = most_nearst_dis + most_farthest_dis
        z_new = (sum_distance - most_nearst_dis) / (sum_distance) * temp_z_0 + (sum_distance - most_farthest_dis) / (
            sum_distance) * temp_z_1

        z_new = z_new.unsqueeze(0)
        if i == 0:
            zs = z_new
        else:
            zs = torch.cat((zs, z_new), dim=0)

    return zs


# 插值函数，反距离权重插值
def get_zs_idw_not_class(coordinates, kdTree_2D, latent_z, k=20, p=50):
    '''
    nearest_distance: n*k维
    nearest_index: n*k维
    dict_zs: 键是文件的id号, 值是对应的z（后面直接改成数组了，问题不大）
    '''
    print(f"进入了反距离权重插值（Inverse Distance Weighting, IDW）, 系数k={k}，系数p={p}~~~~~~~~~~~~~~~~~~")
    # 直接一次查询所有坐标的k个近邻
    nearest_distance, nearest_index = kdTree_2D.query(coordinates, k=k)
    origin_coordinates = kdTree_2D.data  # 获取kdtree中原始的坐标
    for i, pos in enumerate(coordinates):  # 对每一个坐标进行插值
        # pos坐标对应的近邻下标
        pos_nearst_index = nearest_index[i]  # 其中有k个index，每个index对应kdTree_2D中的一个2维坐标
        pos_nearst_distance = nearest_distance[i]
        weights = torch.tensor(1.0 / (pos_nearst_distance + 1e-8))  # 避免除以零，加上一个小的常数

        # 计算加权平均
        weighted_values = latent_z[pos_nearst_index].clone().detach() * weights[:, None] ** p
        z_new = torch.sum(weighted_values, dim=0) / torch.sum(weights ** p)

        z_new[-128:] = latent_z[pos_nearst_index[0]][-128:].clone().detach()  # 类向量为最近的那一个点的向量
        z_new = z_new.unsqueeze(0)
        if i == 0:
            zs = z_new
        else:
            zs = torch.cat((zs, z_new), dim=0)
    return zs


# 尝试各种插值方式##########################################################################################################################


### -------------------------------------------------------------------------------
### 以下是针对全局事件
### -------------------------------------------------------------------------------
# 获取坐标对应的鲁棒性------------------------------------------
def get_robustness_data(coordinates, tree_2D, dict_zs, G, DNN_model, Rob_predictor):
    # 获取最近邻的坐标
    print("获取最近的坐标中.....")
    sys.stdout.flush()
    time1 = time.time()
    nearest_distance, nearest_index = tree_2D.query(coordinates, k=kd_tree_number)
    print("nearest_distance.shape: ", nearest_distance.shape)
    print("nearest_index.shape: ", nearest_index.shape)
    sys.stdout.flush()
    time2 = time.time()
    print("获取坐标消耗时间：", time2 - time1)
    sys.stdout.flush()

    # 根据k个最近邻坐标，计算出坐标对应的z
    print("取z中.....")
    sys.stdout.flush()
    time3 = time.time()
    zs = get_zs(nearest_distance, nearest_index, dict_zs)
    zs_datasets = Mydata_sets(zs)
    zs_loader = DataLoader(zs_datasets, batch_size=200, shuffle=False, num_workers=8)  # 指定读取配置信息
    time4 = time.time()
    print("取z消耗时间：", time4 - time3)
    sys.stdout.flush()

    first = 0  # 判断是否第一次进入循环
    print("生成图片中.....")
    sys.stdout.flush()
    time5 = time.time()
    with torch.no_grad():  # 取消梯度计算，加快运行速度
        for batch_z in zs_loader:
            z = torch.tensor(batch_z).to(torch.float32).to(device)  # latent code
            # label = torch.tensor( [[random.randrange(10)] for i in range(batch_z.shape[0])] ).to(device)
            imgs = G(z)  # NCHW, float32, dynamic range [-1, +1]
            layers = DNN_model(imgs)  # 分类模型分类图片
            if first == 0:
                robustness = Rob_predictor(layers)  # 鲁棒性预测网络预测图片
                first = 1
            else:
                robustness = torch.cat((robustness, Rob_predictor(layers)), dim=0)
        print("robustness.shape: ", robustness.shape)
        sys.stdout.flush()
    time6 = time.time()
    print("生成图片消耗时间：", time6 - time5)
    sys.stdout.flush()
    return robustness.detach().cpu().numpy()


# 获取坐标对应的鲁棒性、类别、以及右边概览图片的坐标------------------------------------------
def get_information(coordinates, tree_2D, dict_zs, G, DNN_model, Rob_predictor, dataset_type):
    # 获取最近邻的坐标
    print("获取最近的坐标中.....")
    sys.stdout.flush()
    time1 = time.time()
    nearest_distance, nearest_index = tree_2D.query(coordinates, k=kd_tree_number)
    print("nearest_distance.shape: ", nearest_distance.shape)
    print("nearest_index.shape: ", nearest_index.shape)
    sys.stdout.flush()
    time2 = time.time()
    print("获取坐标消耗时间：", time2 - time1)
    sys.stdout.flush()

    # 根据k个最近邻坐标，计算出坐标对应的z
    print("取z中.....")
    sys.stdout.flush()
    time3 = time.time()
    zs = get_zs(nearest_distance, nearest_index, dict_zs)
    zs_datasets = Mydata_sets(zs)
    zs_loader = DataLoader(zs_datasets, batch_size=200, shuffle=False, num_workers=1)  # 指定读取配置信息
    time4 = time.time()
    print("取z消耗时间：", time4 - time3)
    sys.stdout.flush()

    print("zs.shape:", zs.shape)

    first = 0  # 判断是否第一次进入循环
    print("生成图片中.....")
    sys.stdout.flush()
    time5 = time.time()
    with torch.no_grad():  # 取消梯度计算，加快运行速度
        for batch_z in zs_loader:
            z = torch.tensor(batch_z).to(torch.float32).to(device)  # latent code
            imgs = G(z)
            layers = DNN_model(imgs)  # 分类模型分类图片
            label = torch.argmax(layers, dim=1)
            if first == 0:
                robustness = Rob_predictor(layers)  # 鲁棒性预测网络预测图片
                all_imgs = imgs
                labels = label
                first = 1
            else:
                robustness = torch.cat((robustness, Rob_predictor(layers)), dim=0)
                all_imgs = torch.cat((all_imgs, imgs), dim=0)
                labels = torch.cat((labels, label), dim=0)
        print("robustness.shape: ", robustness.shape)
        sys.stdout.flush()
    time6 = time.time()
    print("生成图片消耗时间：", time6 - time5)
    sys.stdout.flush()

    # -------------------
    # 从总体图片中取400张图片，及其类别信息、二维坐标
    img_labels_lst_400 = []
    img_coords_lst_400 = []
    image_num_in_row = len(robustness) ** 0.5  # 每行图片为开根号的值
    all_imgs = all_imgs / 2 + 0.5  # [0,1] 归一化图片的范围到0~1区间

    for i in range(image_num_in_row):
        for j in range(20):
            # 确定400张在总样本中的位置
            index = int(image_num_in_row * (image_num_in_row / 20 * i) + 1 * (image_num_in_row / 20 * j))
            img_scaled_single = all_imgs[index]

            utils.save_image(img_scaled_single.detach().cpu(),
                             f'./static/data/' + dataset_type + f'/pic/grid_images/grid_image_{20 * i + j}.png')

            img_labels_lst_400.append((labels[index]).detach().cpu())
            img_coords_lst_400.append((coordinates[index]))

    return robustness.detach().cpu().numpy(), np.float_(img_labels_lst_400), np.float_(img_coords_lst_400)


def get_backmix(z, G, DNN_model, CAMmethod, mask_threshold, dataset_type, background_img):

    imgs = G(z)
    # DNN_model.layer3.register_forward_hook(get_activation('layer3'))

    layers = DNN_model(imgs)  # 分类模型分类图片
    # CAMlayer = activation['layer3']
    max_value, label = torch.max(layers, dim=1)

    # 每张图固定比例取前景
    # 将 CAM 张量展平为一维数组
    # flat_cams = cams.view(-1)

    # 对展平后的数组进行排序
    # sorted_cams, indices = torch.sort(flat_cams, descending=True)

    # 计算阈值索引
    # threshold_index = int(len(sorted_cams) * mask_threshold)

    # 设置阈值
    # threshold = sorted_cams[threshold_index]

    #
    # binary_cams = (cams >= threshold).to(dtype=torch.float32)

    # 在 binary_cams 上添加额外的维度以匹配 imgs 的通道数，并复制到三个通道
    # expanded_binary_cams = binary_cams.repeat(1, 3, 1, 1)
    # 创建一个全零的 32x32 的 tensor
    mask = torch.zeros(32, 32)
    mask_size = int(mask_threshold * 32)
    # 生成随机的左上角坐标
    start_row = random.randint(0, 32 - mask_size)
    start_col = random.randint(0, 32 - mask_size)

    # 将一个 8x8 的方块设置为 1
    mask[start_row:start_row + mask_size, start_col:start_col + mask_size] = 1

    # # 打印掩码
    # print(mask)
    mask_tensor = mask.unsqueeze(0).unsqueeze(0)
    expanded_mask_tensor = mask_tensor.repeat(1, 3, 1, 1)

    # 使用 torch.where() 函数将不包含 CAM 的部分设置为空白
    # 将 binary_cams 中值为 0 的位置的像素设置为 0，其余位置保持不变
    # masked_imgs = torch.where(expanded_binary_cams == 0, torch.tensor(0, device=device), imgs)
    # print(background_img.device,expanded_mask_tensor.device,imgs.device)
    expanded_mask_tensor = expanded_mask_tensor.to(device)
    masked_random_image_tensor = background_img * expanded_mask_tensor + imgs * (1 - expanded_mask_tensor)

    fore_layers = DNN_model(masked_random_image_tensor)
    fore_max_value, fore_label = torch.max(fore_layers, dim=1)

    max_pro_id = fore_layers.squeeze(0).argmax().item()
    # 解释图像并获取 CAM
    cams = CAMmethod(max_pro_id, fore_layers)
    # 展开成32*32

    cams = torch.nn.functional.interpolate(cams[0].unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False)
    # [1,1,32,32]
    return imgs, layers, max_value, label, masked_random_image_tensor, fore_layers, fore_max_value, fore_label, cams


# 返回label的插值
#


def get_zs_idw_class(coordinates, kdTree_2D, latent_z, data_z_labels, k=20, p=50):
    '''
        nearest_distance: n*k维
        nearest_index: n*k维
        dict_zs: 键是文件的id号, 值是对应的z（后面直接改成数组了，问题不大）
        '''
    print(f"进入了反距离权重插值（Inverse Distance Weighting, IDW）, 系数k={k}，系数p={p}~~~~~~~~~~~~~~~~~~")
    # print(data_z_labels.shape)
    # 直接一次查询所有坐标的k个近邻
    nearest_distance, nearest_index = kdTree_2D.query(coordinates, k=k)
    origin_coordinates = kdTree_2D.data  # 获取kdtree中原始的坐标
    for i, pos in enumerate(coordinates):  # 对每一个坐标进行插值
        # pos坐标对应的近邻下标
        pos_nearst_index = nearest_index[i]  # 其中有k个index，每个index对应kdTree_2D中的一个2维坐标
        pos_nearst_distance = nearest_distance[i]
        weights = torch.tensor(1.0 / (pos_nearst_distance + 1e-8))  # 避免除以零，加上一个小的常数

        # 计算加权平均
        weighted_values = latent_z[pos_nearst_index].clone().detach() * weights[:, None] ** p
        z_new = torch.sum(weighted_values, dim=0) / torch.sum(weights ** p)

        z_new[-128:] = latent_z[pos_nearst_index[0]][-128:].clone().detach()  # 类向量为最近的那一个点的向量
        z_new = z_new.unsqueeze(0)
        # 类标签为最近的那个点的类标签
        label_new = data_z_labels[pos_nearst_index[0]].clone().detach()
        if i == 0:
            zs = z_new
            labels = label_new.reshape(1)
        else:
            zs = torch.cat((zs, z_new), dim=0)
            # print(labels,label_new)
            labels = torch.cat((labels, label_new.reshape(1)), dim=0)
    return zs, labels


# 获取两组图片（原始图片，混合图片，背景取一块贴到前景图上）坐标对应的置信度、类别、以及右边概览图片的坐标(使用其他插值法)------------------------------------------
def get_information_backmix(coordinates, tree_2D, dict_zs, data_z_labels, G, DNN_model, CAMmethod, dataset_type, idw_p=50,
                            mask_threshold=0.25):
    # 根据k个最近邻坐标，计算出坐标对应的z
    print("取z中.....")
    time3 = time.time()

    zs, zs_labels = get_zs_idw_class(coordinates, tree_2D, dict_zs, data_z_labels, p=idw_p)

    zs_datasets = Mydata_sets(zs,zs_labels)
    # 不需要前向传播，一次可以多处理一些
    zs_loader = DataLoader(zs_datasets, batch_size=1, shuffle=False, num_workers=0)  # 指定读取配置信息
    time4 = time.time()
    print("取z消耗时间：", time4 - time3)
    sys.stdout.flush()
    print("zs.shape:", zs.shape)


    first = 0  # 判断是否第一次进入循环
    print("生成图片中.....")
    sys.stdout.flush()
    time5 = time.time()
    # print("准备CAM方法。。。")
    # # Global_CAM_method_dict[model_id] = GradCAM(Global_DNN_model_dict[model_id])
    # print("CAM准备完毕。。。")
    background_imgs = torch.load('./static/data/' + dataset_type + f'/background/background_remove_fore.pt')

    with torch.no_grad():  # 取消梯度计算，加快运行速度
        for batch_z, batch_labels in zs_loader:
            # z = torch.tensor(batch_z, dtype=torch.float32, device=device).detach()
            z = torch.tensor(batch_z).to(torch.float32).to(device).detach()  # latent code
            if dataset_type == "CIFAR10":
                # 前景图
                # imgs, layers,max_value, label, cams, binary_cam, fore_imgs, fore_layers,fore_max_value, fore_label = get_foreground_by_cam(z, G, DNN_model, CAMmethod, mask_threshold)
                # 混合图
                random_image_number = random.randint(0, 49999)  # 假设图片编号从0到49999
                background_img = background_imgs[random_image_number]
                background_img = background_img.unsqueeze(0)
                imgs, layers, max_value, label, fore_imgs, fore_layers, fore_max_value, fore_label, cams = get_backmix(z, G,
                                                                                                                 DNN_model,
                                                                                                                 CAMmethod,
                                                                                                                 mask_threshold,
                                                                                                                 dataset_type,
                                                                                                                 background_img)
                # 利用 torch.eq() 函数比较两个标签张量是否相同
                # print(labels.device, batch_labels.device)
                same_labels_mask = torch.eq(label, batch_labels.to(device))

                same_labels_mask_fore = torch.eq(fore_label, batch_labels.to(device))

                max_value = torch.where(same_labels_mask, max_value, -max_value)
                fore_max_value = torch.where(same_labels_mask_fore, fore_max_value, -fore_max_value)
                # imgs = G(z)
                # # DNN_model.layer3.register_forward_hook(get_activation('layer3'))
                # layers = DNN_model(imgs) #分类模型分类图片
                # # CAMlayer = activation['layer3']
                # label = torch.argmax(layers, dim=1)

            elif dataset_type == "SteeringAngle":
                imgs = G(z[:, :256], z[:, 256:])
                # 注册钩子函数到第二层
                DNN_model.pool1.register_forward_hook(get_activation('pool1'))
                label = DNN_model(imgs)
                layers = activation['pool1']
                # cam= layers
                layers = layers.view(layers.size(0), -1)

            if first == 0:
                # robustness = Rob_predictor(layers) #鲁棒性预测网络预测图片
                # fore_robustness = Rob_predictor(fore_layers)
                confidence_imgs = max_value
                confidence_fore_imgs = fore_max_value
                all_imgs = imgs
                all_fore_imgs = fore_imgs
                labels = label
                fore_labels = fore_label
                CAMlayers_mix = cams
                # binary_cams = binary_cam
                first = 1
            else:
                # robustness = torch.cat((robustness, Rob_predictor(layers)), dim=0)
                # fore_robustness= torch.cat((fore_robustness, Rob_predictor(fore_layers)), dim=0)
                confidence_imgs = torch.cat((confidence_imgs, max_value), dim=0)
                confidence_fore_imgs = torch.cat((confidence_fore_imgs, fore_max_value), dim=0)
                all_imgs = torch.cat((all_imgs, imgs), dim=0)
                all_fore_imgs = torch.cat((all_fore_imgs, fore_imgs), dim=0)
                labels = torch.cat((labels, label), dim=0)
                fore_labels = torch.cat((fore_labels, fore_label), dim=0)
                CAMlayers_mix = torch.cat((CAMlayers_mix, cams), dim=0)
                # binary_cams = torch.cat((binary_cams, binary_cam), dim=0)
        # print("robustness.shape: ", robustness.shape)
        # print("fore_robustness.shape: ", fore_robustness.shape)
        confidence_imgs = confidence_imgs.unsqueeze(1)
        confidence_fore_imgs = confidence_fore_imgs.unsqueeze(1)
        print("confidence_imgs.shape: ", confidence_imgs.shape)
        max_value = torch.max(confidence_imgs)  # 找到整个张量中的最大值
        min_value = torch.min(confidence_imgs)  # 找到整个张量中的最小值
        # 对展平后的张量进行排序
        sorted_values, _ = torch.sort(confidence_imgs)
        # 计算每个区间的长度
        num_intervals = 5
        interval_length = len(sorted_values) // num_intervals
        # 计算划分区间的索引值
        split_indices = [i * interval_length for i in range(1, num_intervals)]
        # 在排序后的张量中找到划分区间的值
        split_values = sorted_values[split_indices]
        print(split_values)

        print(max_value, min_value)
        print("confidence_fore_imgs.shape: ", confidence_fore_imgs.shape)
        print("all_imgs.shape: ", all_imgs.shape)
        print("all_fore_imgs.shape: ", all_fore_imgs.shape)
        print("labels.shape: ", labels.shape)
        print("fore_labels.shape: ", fore_labels.shape)
        print("zs_labels shape", zs_labels.shape)
        print("cams shape", CAMlayers_mix.shape)
        # 获取混淆矩阵
        # 将标签转换为NumPy数组
        fore_labels_np = fore_labels.cpu().numpy()
        zs_labels_np = zs_labels.cpu().numpy()

        # 计算混淆矩阵
        num_classes = 10
        conf_matrix = confusion_matrix(zs_labels_np, fore_labels_np, labels=np.arange(num_classes))

        # 将混淆矩阵转换为嵌套列表（用于JSON序列化）
        conf_matrix_list = conf_matrix.tolist()

        # print("CAMlayers.shape: ", CAMlayers.shape)
        # print("binary_cams.shape: ", binary_cams.shape)

        sys.stdout.flush()

    time6 = time.time()
    print("生成图片消耗时间：", time6 - time5)
    sys.stdout.flush()

    # -------------------
    # 从总体1600图片中取400张图片，及其类别信息、二维坐标
    img_labels_lst_400 = []
    img_coords_lst_400 = []
    foreimg_labels_lst_400 = []
    zs_lst_400 = []  # 测试用，保存生成的向量
    image_num_in_row = len(confidence_imgs) ** 0.5  # 每行图片为开根号的值40
    # # 保存生成模型的生成图片，不进行归一化
    all_generate_imgs_tensor = all_imgs
    all_generate_foreimgs_tensor = all_fore_imgs
    #  # [0,1] 归一化图片的范围到0~1区间
    all_imgs = ((all_imgs + 1) / 2).clamp(0.0, 1.0)
    all_fore_imgs = ((all_fore_imgs + 1) / 2).clamp(0.0, 1.0)

    save_generate_imgs_tensor = []
    save_generate_imgs_fore_tensor = []
    # sava_CAMLayer_tensor = []
    for i in range(20):
        for j in range(20):
            # 确定400张在总样本中的位置
            index = int(image_num_in_row * (image_num_in_row / 20 * i) + 1 * (image_num_in_row / 20 * j))

            # 原始图片/前景图片存储，用来方格展示图片，所以用归一化处理之后的
            img_single = all_imgs[index]
            utils.save_image(img_single.detach().cpu(),
                             f'./static/data/' + dataset_type + f'/pic/grid_images/grid_image_{20 * i + j}.png')

            foreimg_single = all_fore_imgs[index]
            # print("foreimg_single.shape", foreimg_single.shape)
            utils.save_image(foreimg_single.detach().cpu(),
                             f'./static/data/' + dataset_type + f'/pic/grid_fore_images/grid_fore_image_{20 * i + j}.png')
            cam_single = CAMlayers_mix[index]
            # print("cam_single.shape", cam_single.shape)
            # cam_single.shape torch.Size([1, 32, 32])
            # foreimg_single.shape torch.Size([3, 32, 32])
            # 将 tensor 转换为 numpy 数组并移除批次维度
            cam_array = cam_single.squeeze().cpu().numpy()

            # 归一化到 0 到 1 的范围内
            cam_array = (cam_array - cam_array.min()) / (cam_array.max() - cam_array.min())

            # 将归一化后的数组转换为 0 到 255 的范围
            cam_array = (cam_array * 255).astype(np.uint8)

            # 使用 Matplotlib 将灰度图转换为彩色图



            # 获取颜色映射
            cmap = plt.colormaps['jet']
            # 将灰度图应用颜色映射，并转换为 (32, 32, 4) 的 RGBA 图像
            colored_cam = cmap(cam_array)

            # 转换为 Pillow 图像
            colored_cam_img = Image.fromarray((colored_cam[:, :, :3] * 255).astype(np.uint8))

            # 添加 Alpha 通道
            alpha = 0.3  # 可以根据需要调整透明度
            alpha_channel = (colored_cam[:, :, 3] * alpha * 255).astype(np.uint8)
            colored_cam_img.putalpha(Image.fromarray(alpha_channel))

            # 保存为 PNG 文件
            output_path = f'./static/data/' + dataset_type + f'/pic/cam_image/cam_image_{20 * i + j}.png'
            colored_cam_img.save(output_path)

            # save_generate_imgs_tensor.append(all_generate_imgs_tensor[index])

            # 未归一化的进行存储，后边用来辅助单个图片的相关信息
            save_generate_imgs_tensor.append(all_generate_imgs_tensor[index])
            save_generate_imgs_fore_tensor.append(all_generate_foreimgs_tensor[index])

            # 存储坐标，原始图片及前景图片的label
            img_labels_lst_400.append((labels[index]).detach().cpu())
            img_coords_lst_400.append((coordinates[index]))
            foreimg_labels_lst_400.append((fore_labels[index]).detach().cpu())

            # 生成这些图片的z
            zs_lst_400.append(zs[index])

    # save_generate_imgs_tensor = torch.stack(save_generate_imgs_tensor, dim=0)
    # print("save_generate_imgs_tensor.shape",save_generate_imgs_tensor.shape)
    # print("save_generate_imgs_fore_tensor.shape",save_generate_imgs_fore_tensor.shape)
    torch.save(save_generate_imgs_tensor,
               './static/data/' + dataset_type + '/pic/grid_images_tensor/save_generate_imgs_tensor.pt')
    torch.save(save_generate_imgs_fore_tensor,
               './static/data/' + dataset_type + '/pic/grid_fore_images_tensor/save_generate_fore_imgs_tensor.pt')

    torch.save(zs_lst_400, "./临时垃圾-随时可删/向量保存/zs_lst_400.pt")
    return confidence_imgs.detach().cpu().numpy(), confidence_fore_imgs.detach().cpu().numpy(), img_labels_lst_400, foreimg_labels_lst_400, img_coords_lst_400,conf_matrix_list


# 分类过程中获取类别激活图，二值掩码，获得只包括前景的图片
def get_foreground_by_cam(z, G, DNN_model, CAMmethod, mask_threshold):
    imgs = G(z)
    # DNN_model.layer3.register_forward_hook(get_activation('layer3'))
    layers = DNN_model(imgs)  # 分类模型分类图片
    # CAMlayer = activation['layer3']
    max_value, label = torch.max(layers, dim=1)
    max_pro_id = layers.squeeze(0).argmax().item()
    # 解释图像并获取 CAM
    cams = CAMmethod(max_pro_id, layers)
    # 展开成32*32
    cams = torch.nn.functional.interpolate(cams[0].unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False)
    binary_cams = (cams >= mask_threshold).to(dtype=torch.float32)

    # 在 binary_cams 上添加额外的维度以匹配 imgs 的通道数，并复制到三个通道
    expanded_binary_cams = binary_cams.repeat(1, 3, 1, 1)

    # 使用 torch.where() 函数将不包含 CAM 的部分设置为空白
    # 将 binary_cams 中值为 0 的位置的像素设置为 0，其余位置保持不变
    masked_imgs = torch.where(expanded_binary_cams == 0, torch.tensor(0, device=device), imgs)

    # CAMmethod.remove_hooks()
    fore_layers = DNN_model(masked_imgs)
    fore_max_value, fore_label = torch.max(fore_layers, dim=1)
    return imgs, layers, max_value, label, cams, binary_cams, masked_imgs, fore_layers, fore_max_value, fore_label


# # 分类过程中获取类别激活图，二值掩码，获得随机混合背景的图片
def get_mix_random_by_cam(z, G, DNN_model, CAMmethod, mask_threshold, dataset_type):
    imgs = G(z)
    # DNN_model.layer3.register_forward_hook(get_activation('layer3'))
    layers = DNN_model(imgs)  # 分类模型分类图片
    # CAMlayer = activation['layer3']
    max_value, label = torch.max(layers, dim=1)
    max_pro_id = layers.squeeze(0).argmax().item()
    # 解释图像并获取 CAM
    cams = CAMmethod(max_pro_id, layers)
    # 展开成32*32

    cams = torch.nn.functional.interpolate(cams[0].unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False)

    # 每张图固定比例取前景
    # 将 CAM 张量展平为一维数组
    flat_cams = cams.view(-1)

    # 对展平后的数组进行排序
    sorted_cams, indices = torch.sort(flat_cams, descending=True)

    # 计算阈值索引
    threshold_index = int(len(sorted_cams) * mask_threshold)

    # 设置阈值
    threshold = sorted_cams[threshold_index]

    #
    binary_cams = (cams >= threshold).to(dtype=torch.float32)

    # 在 binary_cams 上添加额外的维度以匹配 imgs 的通道数，并复制到三个通道
    expanded_binary_cams = binary_cams.repeat(1, 3, 1, 1)

    # Step 1: 从文件夹中随机选择一张图片
    random_image_number = random.randint(0, 49999)  # 假设图片编号从0到49999
    random_image_filename = './static/data/' + dataset_type + f'/pic/origin_50k_png/pic_{random_image_number}.png'
    random_image = Image.open(random_image_filename)
    random_image_tensor = TF.to_tensor(random_image).unsqueeze(0).to(device)
    # 因为是从图片直接读取的，要跟image范围匹配，这样后边进行+1/2的操作不会颜色变浅
    random_image_tensor = random_image_tensor * 2 - 1

    # 使用 torch.where() 函数将不包含 CAM 的部分设置为空白
    # 将 binary_cams 中值为 0 的位置的像素设置为 0，其余位置保持不变
    # masked_imgs = torch.where(expanded_binary_cams == 0, torch.tensor(0, device=device), imgs)

    masked_random_image_tensor = random_image_tensor * (1 - expanded_binary_cams) + imgs * expanded_binary_cams

    fore_layers = DNN_model(masked_random_image_tensor)
    fore_max_value, fore_label = torch.max(fore_layers, dim=1)
    return imgs, layers, max_value, label, cams, binary_cams, masked_random_image_tensor, fore_layers, fore_max_value, fore_label


# 获取两组图片（原始图片，混合图片，前景贴到背景图上）坐标对应的置信度、类别、以及右边概览图片的坐标(使用其他插值法)------------------------------------------
def get_information_mix_conf(coordinates, tree_2D, dict_zs, G, DNN_model, CAMmethod, dataset_type, idw_p=50,
                             mask_threshold=0.3):
    # 根据k个最近邻坐标，计算出坐标对应的z
    print("取z中.....")
    time3 = time.time()

    zs = get_zs_idw_not_class(coordinates, tree_2D, dict_zs, p=idw_p)

    zs_datasets = Mydata_sets(zs)
    # 前向传播每次只能一张图
    zs_loader = DataLoader(zs_datasets, batch_size=1, shuffle=False, num_workers=0)  # 指定读取配置信息
    time4 = time.time()
    print("取z消耗时间：", time4 - time3)
    sys.stdout.flush()
    print("zs.shape:", zs.shape)

    first = 0  # 判断是否第一次进入循环
    print("生成图片中.....")
    sys.stdout.flush()
    time5 = time.time()
    # print("准备CAM方法。。。")
    # # Global_CAM_method_dict[model_id] = GradCAM(Global_DNN_model_dict[model_id])
    # print("CAM准备完毕。。。")
    with torch.no_grad():  # 取消梯度计算，加快运行速度
        for batch_z in zs_loader:
            # z = torch.tensor(batch_z, dtype=torch.float32, device=device).detach()
            z = torch.tensor(batch_z).to(torch.float32).to(device).detach()  # latent code
            if dataset_type == "CIFAR10":
                # 前景图
                # imgs, layers,max_value, label, cams, binary_cam, fore_imgs, fore_layers,fore_max_value, fore_label = get_foreground_by_cam(z, G, DNN_model, CAMmethod, mask_threshold)
                # 混合图
                imgs, layers, max_value, label, cams, binary_cam, fore_imgs, fore_layers, fore_max_value, fore_label = get_mix_random_by_cam(
                    z, G, DNN_model, CAMmethod, mask_threshold, dataset_type)

                # imgs = G(z)
                # # DNN_model.layer3.register_forward_hook(get_activation('layer3'))
                # layers = DNN_model(imgs) #分类模型分类图片
                # # CAMlayer = activation['layer3']
                # label = torch.argmax(layers, dim=1)

            elif dataset_type == "SteeringAngle":
                imgs = G(z[:, :256], z[:, 256:])
                # 注册钩子函数到第二层
                DNN_model.pool1.register_forward_hook(get_activation('pool1'))
                label = DNN_model(imgs)
                layers = activation['pool1']
                # cam= layers
                layers = layers.view(layers.size(0), -1)

            if first == 0:
                # robustness = Rob_predictor(layers) #鲁棒性预测网络预测图片
                # fore_robustness = Rob_predictor(fore_layers)
                confidence_imgs = max_value
                confidence_fore_imgs = fore_max_value
                all_imgs = imgs
                all_fore_imgs = fore_imgs
                labels = label
                fore_labels = fore_label
                CAMlayers = cams
                binary_cams = binary_cam
                first = 1
            else:
                # robustness = torch.cat((robustness, Rob_predictor(layers)), dim=0)
                # fore_robustness= torch.cat((fore_robustness, Rob_predictor(fore_layers)), dim=0)
                confidence_imgs = torch.cat((confidence_imgs, max_value), dim=0)
                confidence_fore_imgs = torch.cat((confidence_fore_imgs, fore_max_value), dim=0)
                all_imgs = torch.cat((all_imgs, imgs), dim=0)
                all_fore_imgs = torch.cat((all_fore_imgs, fore_imgs), dim=0)
                labels = torch.cat((labels, label), dim=0)
                fore_labels = torch.cat((fore_labels, fore_label), dim=0)
                CAMlayers = torch.cat((CAMlayers, cams), dim=0)
                binary_cams = torch.cat((binary_cams, binary_cam), dim=0)
        # print("robustness.shape: ", robustness.shape)
        # print("fore_robustness.shape: ", fore_robustness.shape)
        confidence_imgs = confidence_imgs.unsqueeze(1)
        confidence_fore_imgs = confidence_fore_imgs.unsqueeze(1)
        print("confidence_imgs.shape: ", confidence_imgs.shape)
        max_value = torch.max(confidence_imgs)  # 找到整个张量中的最大值
        min_value = torch.min(confidence_imgs)  # 找到整个张量中的最小值
        # 对展平后的张量进行排序
        sorted_values, _ = torch.sort(confidence_imgs)
        # 计算每个区间的长度
        num_intervals = 5
        interval_length = len(sorted_values) // num_intervals
        # 计算划分区间的索引值
        split_indices = [i * interval_length for i in range(1, num_intervals)]
        # 在排序后的张量中找到划分区间的值
        split_values = sorted_values[split_indices]
        print(split_values)

        print(max_value, min_value)
        print("confidence_fore_imgs.shape: ", confidence_fore_imgs.shape)
        print("all_imgs.shape: ", all_imgs.shape)
        print("all_fore_imgs.shape: ", all_fore_imgs.shape)
        print("labels.shape: ", labels.shape)
        print("fore_labels.shape: ", fore_labels.shape)
        print("CAMlayers.shape: ", CAMlayers.shape)
        print("binary_cams.shape: ", binary_cams.shape)

        sys.stdout.flush()

    time6 = time.time()
    print("生成图片消耗时间：", time6 - time5)
    sys.stdout.flush()

    # -------------------
    # 从总体1600图片中取400张图片，及其类别信息、二维坐标
    img_labels_lst_400 = []
    img_coords_lst_400 = []
    foreimg_labels_lst_400 = []
    zs_lst_400 = []  # 测试用，保存生成的向量
    image_num_in_row = len(confidence_imgs) ** 0.5  # 每行图片为开根号的值40
    # # 保存生成模型的生成图片，不进行归一化
    all_generate_imgs_tensor = all_imgs
    all_generate_foreimgs_tensor = all_fore_imgs
    #  # [0,1] 归一化图片的范围到0~1区间
    all_imgs = ((all_imgs + 1) / 2).clamp(0.0, 1.0)
    all_fore_imgs = ((all_fore_imgs + 1) / 2).clamp(0.0, 1.0)

    save_generate_imgs_tensor = []
    save_generate_imgs_fore_tensor = []
    # sava_CAMLayer_tensor = []
    for i in range(20):
        for j in range(20):
            # 确定400张在总样本中的位置
            index = int(image_num_in_row * (image_num_in_row / 20 * i) + 1 * (image_num_in_row / 20 * j))

            # 原始图片/前景图片存储，用来方格展示图片，所以用归一化处理之后的
            img_single = all_imgs[index]
            utils.save_image(img_single.detach().cpu(),
                             f'./static/data/' + dataset_type + f'/pic/grid_images/grid_image_{20 * i + j}.png')

            foreimg_single = all_fore_imgs[index]
            utils.save_image(foreimg_single.detach().cpu(),
                             f'./static/data/' + dataset_type + f'/pic/grid_fore_images/grid_fore_image_{20 * i + j}.png')

            save_generate_imgs_tensor.append(all_generate_imgs_tensor[index])

            # 未归一化的进行存储，后边用来辅助单个图片的相关信息
            save_generate_imgs_tensor.append(all_generate_imgs_tensor[index])
            save_generate_imgs_fore_tensor.append(all_generate_foreimgs_tensor[index])

            # 存储坐标，原始图片及前景图片的label
            img_labels_lst_400.append((labels[index]).detach().cpu())
            img_coords_lst_400.append((coordinates[index]))
            foreimg_labels_lst_400.append((fore_labels[index]).detach().cpu())
            # 生成这些图片的z
            zs_lst_400.append(zs[index])

    # save_generate_imgs_tensor = torch.stack(save_generate_imgs_tensor, dim=0)
    # print("save_generate_imgs_tensor.shape",save_generate_imgs_tensor.shape)
    # print("save_generate_imgs_fore_tensor.shape",save_generate_imgs_fore_tensor.shape)
    torch.save(save_generate_imgs_tensor,
               './static/data/' + dataset_type + '/pic/grid_images_tensor/save_generate_imgs_tensor.pt')
    torch.save(save_generate_imgs_fore_tensor,
               './static/data/' + dataset_type + '/pic/grid_fore_images_tensor/save_generate_fore_imgs_tensor.pt')

    torch.save(zs_lst_400, "./临时垃圾-随时可删/向量保存/zs_lst_400.pt")
    return confidence_imgs.detach().cpu().numpy(), confidence_fore_imgs.detach().cpu().numpy(), img_labels_lst_400, foreimg_labels_lst_400, img_coords_lst_400


# 获取坐标对应的鲁棒性、类别、以及右边概览图片的坐标(使用其他插值法)------------------------------------------
def get_information_other(coordinates, tree_2D, dict_zs, G, DNN_model, Rob_predictor, dataset_type, idw_p=50):
    # 根据k个最近邻坐标，计算出坐标对应的z
    print("取z中.....")
    time3 = time.time()
    if dataset_type == "SteeringAngle":
        zs = get_zs_new_regre(coordinates, tree_2D, dict_zs)
    else:
        zs = get_zs_idw_not_class(coordinates, tree_2D, dict_zs, p=idw_p)

    zs_datasets = Mydata_sets(zs)
    zs_loader = DataLoader(zs_datasets, batch_size=200, shuffle=False, num_workers=1)  # 指定读取配置信息
    time4 = time.time()
    print("取z消耗时间：", time4 - time3)
    sys.stdout.flush()
    print("zs.shape:", zs.shape)

    first = 0  # 判断是否第一次进入循环
    print("生成图片中.....")
    sys.stdout.flush()
    time5 = time.time()
    with torch.no_grad():  # 取消梯度计算，加快运行速度
        for batch_z in zs_loader:
            # z = torch.tensor(batch_z, dtype=torch.float32, device=device).detach()
            z = torch.tensor(batch_z).to(torch.float32).to(device).detach()  # latent code
            if dataset_type == "CIFAR10":
                imgs = G(z)
                layers = DNN_model(imgs)  # 分类模型分类图片
                label = torch.argmax(layers, dim=1)
            elif dataset_type == "SteeringAngle":
                imgs = G(z[:, :256], z[:, 256:])
                # 注册钩子函数到第二层
                DNN_model.pool1.register_forward_hook(get_activation('pool1'))
                label = DNN_model(imgs)
                layers = activation['pool1']
                layers = layers.view(layers.size(0), -1)

            if first == 0:
                robustness = Rob_predictor(layers)  # 鲁棒性预测网络预测图片
                all_imgs = imgs
                labels = label
                first = 1
            else:
                robustness = torch.cat((robustness, Rob_predictor(layers)), dim=0)
                all_imgs = torch.cat((all_imgs, imgs), dim=0)
                labels = torch.cat((labels, label), dim=0)
        print("robustness.shape: ", robustness.shape)
        sys.stdout.flush()
    time6 = time.time()
    print("生成图片消耗时间：", time6 - time5)
    sys.stdout.flush()

    # -------------------
    # 从总体图片中取400张图片，及其类别信息、二维坐标
    img_labels_lst_400 = []
    img_coords_lst_400 = []
    zs_lst_400 = []  # 测试用，保存生成的向量
    image_num_in_row = len(robustness) ** 0.5  # 每行图片为开根号的值
    all_generate_imgs_tensor = all_imgs  # 保存生成模型的生成图片，不进行归一化

    all_imgs = ((all_imgs + 1) / 2).clamp(0.0, 1.0)  # [0,1] 归一化图片的范围到0~1区间

    save_generate_imgs_tensor = []
    for i in range(20):
        for j in range(20):
            # 确定400张在总样本中的位置
            index = int(image_num_in_row * (image_num_in_row / 20 * i) + 1 * (image_num_in_row / 20 * j))
            img_scaled_single = all_imgs[index]
            save_generate_imgs_tensor.append(all_generate_imgs_tensor[index])

            utils.save_image(img_scaled_single.detach().cpu(),
                             f'./static/data/' + dataset_type + f'/pic/grid_images/grid_image_{20 * i + j}.png')

            img_labels_lst_400.append((labels[index]).detach().cpu())
            img_coords_lst_400.append((coordinates[index]))
            zs_lst_400.append(zs[index])
    # 单独图片数据有用
    torch.save(save_generate_imgs_tensor,
               './static/data/' + dataset_type + '/pic/grid_images_tensor/save_generate_imgs_tensor.pt')
    torch.save(zs_lst_400, "./临时垃圾-随时可删/向量保存/zs_lst_400.pt")
    return robustness.detach().cpu().numpy(), img_labels_lst_400, img_coords_lst_400


### -------------------------------------------------------------------------------
### 以下是针对单击事件
### -------------------------------------------------------------------------------
# 计算一个坐标点的z
def get_z(nearest_distance, nearest_index, dict_zs):
    k = len(nearest_index)  # k个近邻（默认每个都是k近邻）

    iter_distance = nearest_distance
    iter_index = nearest_index

    sum_distanceForIter = np.sum(iter_distance)  # 这k个近邻的距离总和
    for i, index in enumerate(iter_index):
        temp_z = torch.tensor(dict_zs[index])
        temp_distance = iter_distance[i]
        w = (sum_distanceForIter - temp_distance) / ((k - 1) * sum_distanceForIter)  # 对z进行权重
        if i == 0:
            z = temp_z * w
        else:
            z += temp_z * w
    z = z.unsqueeze(0)  # shape:[1*512]

    return z


# 通过对抗扰动获取图片真实的鲁棒性
def adversarial_robustness(model, image_path, label, device, preprocessing=True):
    print("获取对抗鲁棒性中.....")
    sys.stdout.flush()
    time1 = time.time()

    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # 增加一个维度维batch维度
    copy_model = copy.deepcopy(model)
    copy_model.eval()

    bounds = (0, 1)
    if preprocessing:
        preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
        fmodel = fb.PyTorchModel(copy_model, bounds=bounds, device=device, preprocessing=preprocessing)
    else:
        fmodel = fb.PyTorchModel(copy_model, bounds=bounds, device=device)
    attack = fb.attacks.LinfPGD()
    epsilon = 0.001
    # 直到出现对抗样本为止
    flag = False
    while (flag == False):
        if epsilon >= 0.033:
            break
        raw_advs, clipped_advs, success = attack(fmodel, image, label, epsilons=epsilon)
        flag = success
        epsilon = epsilon + 0.001

    sys.stdout.flush()
    time2 = time.time()
    print("获取对抗鲁棒性消耗时间：", time2 - time1)
    return epsilon - 0.001


# 通过对抗扰动获取图片真实鲁棒性（回归模型）
def adversarial_robustness_regre(model, image_path, label, device):
    print("获取对抗鲁棒性中.....")
    sys.stdout.flush()
    time1 = time.time()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # 增加一个维度维batch维度
    copy_model = copy.deepcopy(model)
    copy_model.eval()

    attack = RegreAttack()
    epsilon = 0.001
    flag = False
    while (flag == False):
        if epsilon >= 0.033:
            break
        adv = attack.run(copy_model, image, epsilon=epsilon)
        label2 = model(adv)[0]
        print("误差: ", abs(label2 - label))
        if abs(label2 - label) >= 0.05:  # 0~1对应-80~80度，所以0.03125对应的是误差超过5度，0.05对应8度, 0.0625对应10度
            flag = True
        epsilon = epsilon + 0.001

    sys.stdout.flush()
    time2 = time.time()
    print("获取对抗鲁棒性消耗时间：", time2 - time1)
    return epsilon - 0.001


# 获取点击位置图片的类别、鲁棒性、置信度
def get_image_information(points, tree_2D, dict_zs, G, DNN_model, Rob_predictor, img_name="one",
                          dataset_type="CIFAR10"):
    # 获取最近邻的坐标
    print("获取最近的坐标中.....")
    sys.stdout.flush()
    time1 = time.time()
    nearest_distance, nearest_index = tree_2D.query(points, k=kd_tree_number)
    print("nearest_index: ", nearest_index)
    print("nearest_distance.shape: ", nearest_distance.shape)
    print("nearest_index.shape: ", nearest_index.shape)
    sys.stdout.flush()
    time2 = time.time()
    print("获取坐标消耗时间：", time2 - time1)
    sys.stdout.flush()

    # 根据k个最近邻坐标，计算出坐标对应的z
    print("取z中.....")
    sys.stdout.flush()
    time3 = time.time()
    z = get_z(nearest_distance, nearest_index, dict_zs)
    print("z.shape: ", z.shape)
    time4 = time.time()
    print("取z消耗时间：", time4 - time3)
    sys.stdout.flush()

    print("生成图片中.....")
    sys.stdout.flush()
    time5 = time.time()
    # with torch.no_grad(): # 取消梯度计算，加快运行速度
    z = torch.tensor(z).to(torch.float32).to(device)  # latent code
    img = G(z)
    layer = DNN_model(img)  # 分类模型分类图片
    label = torch.argmax(layer, dim=1)

    img = img / 2 + 0.5  # [0,1] 归一化图片的范围到0~1区间
    utils.save_image(img.detach().cpu(), f'./static/example/pic/' + img_name + '.png')

    robustness = Rob_predictor(layer)  # 鲁棒性预测网络预测图片
    robustness = robustness.detach().cpu()
    # print("robustness.shape: ", robustness.shape)
    image_path = './static/example/pic/' + img_name + '.png'
    # robustness = adversarial_robustness(DNN_model, image_path, label, device) # 对抗性攻击算法获得鲁棒性
    print("robustness: ", robustness)
    sys.stdout.flush()

    time6 = time.time()
    print("生成图片消耗时间：", time6 - time5)
    sys.stdout.flush()
    return label.detach().cpu(), robustness, layer.detach().cpu()


# 使用其他插值方式获取坐标对应的图片
def get_image_information_other(points, img_type, tree_2D, dict_zs, G, DNN_model, Rob_predictor, img_name="one",
                                dataset_type="CIFAR10", idw_p=50):
    # 根据k个最近邻坐标，计算出坐标对应的z
    print("取z中.....")
    sys.stdout.flush()
    time3 = time.time()
    if dataset_type == "SteeringAngle":
        z = get_zs_new_regre([points], tree_2D, dict_zs)
    else:
        z = get_zs_idw_not_class([points], tree_2D, dict_zs, p=idw_p)
    print("z.shape: ", z.shape)
    time4 = time.time()
    print("取z消耗时间：", time4 - time3)
    sys.stdout.flush()

    print("生成图片中.....")
    sys.stdout.flush()
    time5 = time.time()
    # with torch.no_grad(): # 取消梯度计算，加快运行速度
    z = torch.tensor(z).to(torch.float32).to(device)  # latent code

    if dataset_type == "CIFAR10":
        img = G(z)
        layer = DNN_model(img)  # 分类模型分类图片
        # label = torch.argmax(layer, dim=1)
        _, label = torch.max(layer.data, 1)
    elif dataset_type == "SteeringAngle":
        img = G(z[:, :256], z[:, 256:])
        # 注册钩子函数到第二层
        DNN_model.pool1.register_forward_hook(get_activation('pool1'))
        label = DNN_model(img)  # 分类模型分类图片
        layer = activation["pool1"]
        layer = layer.view(layer.size(0), -1)

    img = img / 2 + 0.5  # [0,1] 归一化图片的范围到0~1区间
    utils.save_image(img.detach().cpu(), f'./static/example/pic/' + img_name + '.png')

    # 鲁棒性预测获取鲁棒性的方法 
    robustness = Rob_predictor(layer)  # 鲁棒性预测网络预测图片
    robustness = robustness.detach().cpu()
    # print("robustness.shape: ", robustness.shape)

    # 对抗性攻击获取鲁棒性的方法

    # image_path = './static/example/pic/'+ img_name +'.png'
    # if dataset_type == "CIFAR10":
    #     robustness = adversarial_robustness(DNN_model, image_path, label, device, preprocessing=False) # 对抗性攻击算法获得鲁棒性
    # elif dataset_type == "SteeringAngle":
    #     robustness = adversarial_robustness_regre(DNN_model, image_path, label, device) # 对抗性攻击算法获得鲁棒性
    print("robustness: ", robustness)
    sys.stdout.flush()

    time6 = time.time()
    print("生成图片消耗时间：", time6 - time5)
    sys.stdout.flush()
    return label.detach().cpu(), robustness, layer.detach().cpu()


# 对输入的图片进行评估(通过get_information_other中保存的tensor进行复现)
def evaluate_img(img_number, img_type, img_name, G, DNN_model, Rob_predictor, dataset_type):
    # with torch.no_grad(): # 取消梯度计算，加快运行速度
    if img_type == 0:
        imgs_tensor_path = "./static/data/" + dataset_type + "/pic/grid_images_tensor/save_generate_imgs_tensor.pt"
    else:
        imgs_tensor_path = "./static/data/" + dataset_type + "/pic/grid_fore_images_tensor/save_generate_fore_imgs_tensor.pt"

    all_generate_imgs_tesnor = torch.load(imgs_tensor_path)
    img = all_generate_imgs_tesnor[int(img_number)]
    img = img.unsqueeze(0).to(device)

    if dataset_type == "CIFAR10":
        layer = DNN_model(img)  # 分类模型分类图片
        label = torch.argmax(layer, dim=1)
    elif dataset_type == "SteeringAngle":
        # 注册钩子函数到第二层
        DNN_model.pool1.register_forward_hook(get_activation('pool1'))
        label = DNN_model(img)  # 分类模型分类图片
        layer = activation["pool1"]
        layer = layer.view(layer.size(0), -1)

    # robustness = Rob_predictor(layer) #鲁棒性预测网络预测图片
    # robustness = robustness.detach().cpu()
    print("当前点击的图片id：", img_name)
    img = img / 2 + 0.5  # [0,1] 归一化图片的范围到0~1区间
    utils.save_image(img, f'./static/example/pic/' + img_name + '.png')

    img_path = f'./static/example/pic/' + img_name + '.png'
    if dataset_type == "CIFAR10":
        robustness = adversarial_robustness(DNN_model, img_path, label, device)  # 对抗性攻击算法获得鲁棒性
    elif dataset_type == "SteeringAngle":
        robustness = adversarial_robustness_regre(DNN_model, img_path, label, device)  # 对抗性攻击算法获得鲁棒性

    return label.detach().cpu(), robustness, layer.detach().cpu()
