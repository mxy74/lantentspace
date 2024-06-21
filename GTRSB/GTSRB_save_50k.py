import csv
import itertools
import os
import pickle
import time
import random

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.utils as utils
import fid_score as official_fid
from GTRSB import CDCGAN_size32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrafficSignDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images, self.labels = self.read_traffic_signs(root_dir)
        self.transform = transform

    def read_traffic_signs(self, rootpath):
        images = []
        labels = []
        target_count = 1100
        for c in range(0, 43):
            prefix = rootpath + '/' + format(c, '05d') + '/'
            gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')
            gtReader = csv.reader(gtFile, delimiter=';')
            next(gtReader)

            class_images = []
            class_labels = []
            selected_images = []
            selected_labels = []
            for row in gtReader:
                img = plt.imread(os.path.join(prefix, row[0]))
                class_images.append(img)
                class_labels.append(int(row[7]))

                # images.append(plt.imread(prefix + row[0]))
                # labels.append(int(row[7]))
            gtFile.close()

            if len(class_images) > target_count:
                indices = random.sample(range(len(class_images)), target_count)
                selected_images = [class_images[i] for i in indices]
                selected_labels = [class_labels[i] for i in indices]
            elif len(class_images) < target_count:
                extra_images = target_count - len(class_images)
                selected_images = class_images[:]
                selected_labels = class_labels[:]

                while extra_images > 0:
                    if extra_images >= len(class_images):
                        selected_images.extend(class_images)
                        selected_labels.extend(class_labels)
                        extra_images -= len(class_images)
                    else:
                        indices = random.sample(range(len(class_images)), extra_images)
                        selected_images.extend([class_images[i] for i in indices])
                        selected_labels.extend([class_labels[i] for i in indices])
                        extra_images = 0
            print(c, 'classes:',len(selected_images), len(selected_labels))
            images.extend(selected_images)
            labels.extend(selected_labels)

        return images, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image.to(device), label


def save_orgin_pic():
    # 生成原始图片和生成图片到origin
    img_size = 32

    save_path = 'origin_size32_50k/'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        # 存储的图片会黑
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_dataset = TrafficSignDataset('GTSRB_Final_Training_Images/GTSRB/Final_Training/Images', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    index = 0
    for data in train_loader:
        img, labels = data
        utils.save_image(img, save_path + 'pic_' + str(index) + '.png', nrow=10)
        index += 1


def save_gene_pic():
    # 加载你的生成器模型
    G = CDCGAN_size32.generator(128).to(device)
    G.load_state_dict(torch.load('GTSRB_cDCGAN_results/GTSRB_cDCGAN_generator_param_size32_epoch20.pth'))
    G.eval()
    save_path = 'random_size32_50k/'

    # 每个类别的onehot
    onehot = torch.zeros(43, 43)
    onehot = onehot.scatter_(1, torch.LongTensor(list(range(43))).view(43, 1), 1).view(43,
                                                                                       43, 1, 1)
    z_100dim = []
    all_label = []
    for num in range(50000):
        z = torch.tensor(np.random.RandomState(num).randn(1, 100, 1, 1)).to(torch.float32).to(device)  # latent code
        label = torch.tensor(random.randrange(43)).unsqueeze(0).to(device)
        # 获取对应标签的one-hot编码
        label_onehot = onehot[label].to(device)
        # 存储
        if num == 0:
            z_100dim = z
            all_label = label
        else:
            z_100dim = torch.cat((z_100dim, z))
            all_label = torch.cat((all_label, label))
        # print(z.shape, label.shape)
        # 1,100,1,1
        # 1
        img = G(z, label_onehot)  # NCHW, float32, dynamic range [-1, +1]
        img = ((img + 1) / 2).clamp(0.0, 1.0)  # 变换到[0,1]范围内
        # print(z.shape) # [1, 80]
        # print(shared_label.shape) # [1, 128]
        # print(z_and_shared_label.shape) # [1, 208]
        utils.save_image(img.detach().cpu(), save_path + '/pic_' + str(num) + '.png')
    print(z_100dim.shape)
    print(all_label.shape)
    torch.save(z_100dim, 'GTSRB_cDCGAN_results/cGAN_100z_size32_50k.pt')
    torch.save(all_label, 'GTSRB_cDCGAN_results/cGAN_label_size32_50k.pt')


# fid
def fid_judge():
    # fid计算模型
    dims = 2048

    # num_avail_cpus = len(os.sched_getaffinity(0))
    num_workers = min(1, 8)
    block_idx = official_fid.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    fid_model = official_fid.InceptionV3([block_idx]).to(device)
    print('fid_model load success!')

    pic_path_fid1 = 'random_size32_50k'
    pic_path_fid2 = 'origin_size32_50k'

    batch_size = 100
    m1, s1 = official_fid.compute_statistics_of_path(pic_path_fid1, fid_model, batch_size,
                                                     dims, device, num_workers)
    m2, s2 = official_fid.compute_statistics_of_path(pic_path_fid2, fid_model, batch_size,
                                                     dims, device, num_workers)
    fid_value = official_fid.calculate_frechet_distance(m1, s1, m2, s2)
    print(fid_value)

#
# def

def main():
    save_orgin_pic()
    save_gene_pic()
    fid_judge()
#     看一下哪个类别效果差


if __name__ == '__main__':
    main()
