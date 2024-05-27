import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(208, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 208),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 208))
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# 计算VAE的损失函数
def vae_loss(x, x_recon, mu, logvar):
    weight = torch.ones_like(x)  # 创建与输入数据相同形状的权重向量，初始权重为1
    weight[:, -128:] = 1  # 将后面128维的权重设置为x
    recon_loss = nn.BCELoss(reduction='none')(x_recon, x.view(-1, 208)) * weight.view(-1, 208)  # 根据权重调整重构损失
    recon_loss = torch.sum(recon_loss) # 对损失求和
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# 自定义datasets
class Mydata_sets(Dataset):
    def __init__(self, path, device, transform=None):
        super(Mydata_sets, self).__init__()
        self.latent_z = torch.load(path, map_location=device)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = Image.open(os.path.join(self.root_dir, img_name))
        id_name = torch.tensor(int(img_name[4:-4])) #pic_xx.jpg
        if self.transform is not None:
            img = self.transform(img)
        return img, id_name

    def __len__(self):
        return len(self.img_names)

# 设置训练参数
latent_dim = 2
batch_size = 128
epochs = 10
lr = 1e-3

# 加载MNIST数据集
train_dataset = MNIST(root='./data', train=True, download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化VAE模型和优化器
vae = VAE(latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=lr)

# 训练VAE模型
vae.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        optimizer.zero_grad()

        x_recon, mu, logvar = vae(x)

        loss = vae_loss(x, x_recon, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader.dataset):.4f}")

# 使用训练好的VAE生成样本
vae.eval()
with torch.no_grad():
    z = torch.randn(16, latent_dim)
    samples = vae.decode(z)

# 在此处处理生成的样本