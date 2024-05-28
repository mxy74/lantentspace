import torch
from torchvision.transforms.functional import normalize, resize, to_pil_image
# 看一下这一批数据对应哪个原始图片-----random_50k_png

#
# labels_path = "E:/project/lantentspace/static/data/CIFAR10/labels/BigGAN_random_png_208z_50000_2023-08-30_labels.pt"
#
# # data_z_path = "./static/data/CIFAR10/latent_z/BigGAN_208z_50000.pt"
# labels = torch.load(labels_path, map_location="cpu")  # 因为我之前保存数据到了GPU上，所以要回到cpu上才不会出错
#
# for i in range(10):
#     z = labels[i]
#     print(z)

# cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#
#
# class_number = cifar10_classes.index('bird')
# print(class_number)


# 构建random_discovery和random_0文件夹
# import os
# import random
# import shutil
#
# # 定义原始图片文件夹和目标文件夹路径
# original_folder = img_dir = "E:/project/lantentspace/static/data/CIFAR10/pic/random_50k_png"
# target_folder = "./SOURCE_DIR/random500_1"
#
# # 获取原始图片文件夹下文件名中的数字范围（假设文件名格式为pic_0.png, pic_1.png, ..., pic_49999.png）
# start_num = 0
# end_num = 49999
#
# # 随机选择5000个数字作为文件编号
# random_nums = random.sample(range(start_num, end_num + 1), 5000)
#
# # 创建目标文件夹（如果不存在）
# if not os.path.exists(target_folder):
#     os.makedirs(target_folder)
#
# # 复制随机选择的5000张图片到目标文件夹
# for num in random_nums:
#     file_name = "pic_{}.png".format(num)
#     file_path = os.path.join(original_folder, file_name)
#     target_path = os.path.join(target_folder, file_name)
#     shutil.copyfile(file_path, target_path)
#

# imagenet类别信息输出

# print("Successfully copied 5000 random images to the target folder.")
# import json
# import requests
#
# # 下载 ImageNet 标签文件
# LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
# response = requests.get(LABELS_URL)
# labels = json.loads(response.text)
#
# # 打印前10个类别
# for i, label in enumerate(labels[239:245]):
#     print(f"Class {i}: {label}")


from torchvision.transforms.functional import normalize, resize, to_pil_image
# 看一下这一批数据对应哪个原始图片-----random_50k_png


import torch

# 加载潜在向量和标签
data_z_path = "E:/project/lantentspace/static/data/CIFAR10/latent_z/BigGAN_random_png_208z_50000_2023-08-30.pt"
label_path = "E:/project/lantentspace/static/data/CIFAR10/labels/BigGAN_random_png_208z_50000_2023-08-30_labels.pt"

dict_zs = torch.load(data_z_path, map_location="cpu")  # 因为我之前保存数据到了GPU上，所以要回到cpu上才不会出错
data_z_labels = torch.load(label_path, map_location="cpu")

# 初始化存储后128个元素的字典
category_tail_128_elements = {}

# 找到每个类别的潜在向量
for category in range(10):  # CIFAR-10 有10个类别，编号从0到9
    category_zs = [z for z, label in zip(dict_zs, data_z_labels) if label == category]
    if category_zs:  # 确保该类别存在潜在向量
        tail_128_elements = category_zs[0][-128:]  # 获取该类别潜在向量的后128个元素
        category_tail_128_elements[category] = tail_128_elements

# 将所有类别的后128个元素存储在一个张量中
tail_128_tensor = torch.stack([category_tail_128_elements[category] for category in range(10)])

print(tail_128_tensor.shape)
# 保存到.pt文件
torch.save(tail_128_tensor, 'E:/project/lantentspace/GANLatentDiscovery-master/models/BigGAN_CIFAR10/label_z_128.pt')

print("各类别的后128个元素已保存为 category_tail_128_elements.pt")
