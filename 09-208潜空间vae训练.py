import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

from torch.nn import functional as F
from tqdm import tqdm

import model_files as model_all #模型的连接在__init__.py中体现

# 参数控制
batch_size = 128
num_of_images = 50000
learning_rate = 0.005
device = torch.device('cuda:1')
kl_alf= batch_size / num_of_images
num_epoches = 100

# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
# 加载数据
latent_z_path="./static/data/CIFAR10/latent_z/BigGAN_208z_" + str(num_of_images) + ".pt"
latent_z = torch.load(latent_z_path, map_location="cpu") #因为我之前保存数据到了GPU上，所以要回到cpu上才不会出错

dataset = MyDataset(latent_z)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

vae = model_all.get_VAE(dataset="CIFAR10", in_dim=208, latent_dim=2).to(device)
vae.train()
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

log_id = 7 # 日志id
# 参数保存到日志
with open("./model_files/CIFAR10/checkpoints/vae/log"+ str(log_id) +".txt","a+") as f:
    f.write("batch_size :{}, learning_rate: {:.4f}, kl_alf = batch_size / num_of_images: {:.4f}, num_epochs: {}, latent_z_path: {}\n {}\n " .format(batch_size, learning_rate, kl_alf, num_epoches, latent_z_path, vae))

tq_epoch = tqdm(range(num_epoches))
for epoch in tq_epoch:
    all_loss = 0
    all_reconst_loss = 0
    all_kl_loss = 0
    for inputs in dataloader:
        optimizer.zero_grad()

        inputs = inputs.to(device)
        recons, inputs, mu, log_var = vae(inputs)
        loss, reconst_loss, kl_loss= vae.loss_function(recons, inputs, mu, log_var, M_N=kl_alf)
        all_loss += loss
        all_reconst_loss += reconst_loss
        all_kl_loss += kl_loss

        loss.backward()
        optimizer.step()

    txt = f"all_loss: {all_loss:.4f}, all_reconst_loss: {all_reconst_loss:.4f}, all_kl_loss: {all_kl_loss:.4f}"

    with open("./model_files/CIFAR10/checkpoints/vae/log"+ str(log_id) +".txt","a+") as f:
        f.write("epoch :{}, Loss: {:.4f}, Reconst Loss: {:.4f}, KL Div: {:.4f}\n" .format(epoch+1, all_loss.item(), all_reconst_loss.item(), all_kl_loss.item()))
    tq_epoch.set_description(txt)
    tq_epoch.update()

torch.save(vae.state_dict(), "./model_files/CIFAR10/checkpoints/vae/vae"+ str(log_id) +".pt")
