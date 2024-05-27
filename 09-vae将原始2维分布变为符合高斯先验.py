import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size * 2) # 输出均值向量和方差向量
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid() # 输出二元数据
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1) # 将输出向量拆分为均值向量和方差向量
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std # 重参数化
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# 定义训练函数
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x in dataloader:
        x = x.to(device)
        x_recon, mu, logvar = model(x)
        recon_loss = criterion(x_recon, x) # 重构损失
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # KL散度
        loss = recon_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 定义测试函数
def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            x_recon, mu, logvar = model(x)
            recon_loss = criterion(x_recon, x)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_div
            total_loss += loss.item()
    return total_loss / len(dataloader)

# 加载数据
kdTree = torch.load("./static/data/CIFAR10/2D_kdTree/2D_kdTree_200000.pt")
data = kdTree.data
dataset = MyDataset(torch.tensor(data, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 定义超参数和设备
input_size = 2
hidden_size = 128
latent_size = 2
learning_rate = 1e-3
num_epochs = 100
device = torch.device('cuda: 1')

# 创建模型和优化器
model = VAE(input_size, hidden_size, latent_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss(reduction='sum') # 二元交叉熵损失

# 训练模型
for epoch in range(num_epochs):
    train_loss = train(model, dataloader, optimizer, criterion, device)
    test_loss = test(model, dataloader, criterion, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# 生成新数据
model.eval()
with torch.no_grad():
    z = torch.randn(100, latent_size).to(device)
    x_gen = model.decode(z).cpu().numpy()
    print(x_gen)