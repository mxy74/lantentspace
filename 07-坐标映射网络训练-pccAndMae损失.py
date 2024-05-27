import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(hidden_size, 208)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Mydata_sets(Dataset):
    def __init__(self, two_Ds, high_Ds, train=True):
        super(Mydata_sets, self).__init__()
        # 输入类型都要是tensor
        if train:
            self.two_Ds = two_Ds[:180000]
            self.high_Ds = high_Ds[:180000]
        else:
            self.two_Ds = two_Ds[180000:]
            self.high_Ds = high_Ds[180000:]

    def __getitem__(self, index):
        two_D = self.two_Ds[index]
        high_D = self.high_Ds[index]
        return two_D, high_D

    def __len__(self):
        return len(self.two_Ds)
# 设备
device = torch.device("cuda:2")

# 读取数据
two_Ds_tree = torch.load("./static/data/CIFAR10/2D_kdTree/2D_kdTree_200000.pt")
two_Ds = torch.tensor(two_Ds_tree.data).to(device).to(torch.float32)
high_Ds = torch.load("./static/data/CIFAR10/latent_z/BigGAN_208z_200000.pt", map_location=device).to(device)


# 定义训练数据集
train_datasets = Mydata_sets(two_Ds=two_Ds, high_Ds=high_Ds, train=True)  
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=128, shuffle=False)  # 指定读取配置信息
# 定义测试数据集
test_datasets = Mydata_sets(two_Ds=two_Ds, high_Ds=high_Ds, train=False)  
test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=1, shuffle=False)  # 指定读取配置信息



# 初始化模型和优化器
net = Net(hidden_size=64).to(device)
net.train()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# 定义综合损失函数
def loss_fn(y_pred, y_true, alpha=0.2):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    corr = torch.tensor(pearsonr(y_pred.flatten().detach().numpy(), y_true.flatten().detach().numpy())[0])
    mae = nn.L1Loss()(y_pred, y_true)
    loss = alpha*corr + (1-alpha)*mae
    return loss

# 训练模型
threshold = 0.0001
window_size=8
converged = False
for epoch in range(500):
    pbar = tqdm(total=len(train_dataloader), desc='Progress')
    losses = [float('inf')] * window_size # 初始化损失值列表
    for i, (x_batch, y_batch) in enumerate(train_dataloader):
        optimizer.zero_grad()
        y_pred = net(x_batch)
        loss = loss_fn(y_pred, y_batch, alpha=0.2)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        pbar.update(1)
    pbar.close()

    losses[epoch % window_size] = loss.item()  # 更新损失值列表
    avg_loss = sum(losses) / min(epoch+1, window_size)  # 计算平均损失值
    if epoch >= window_size-1 and abs(avg_loss - losses[epoch % window_size]) < threshold:
        print('Training finished: loss converged')
        torch.save(net.state_dict(), "./model_files/CIFAR10/checkpoints/mapping_network/mapping_network.pt")
        break



# 在测试集上评估模型
flag = 0
test_dataloader = tqdm(test_dataloader)
for x_batch, y_batch in test_dataloader:
    optimizer.zero_grad()
    y_pred = net(x_batch)
    if flag == 0:
        print("y_pred: ",y_pred)
        print("y_batch: ",y_batch)
        a = input("press 0 to break: ")
    if a == "0":
        flag = 1