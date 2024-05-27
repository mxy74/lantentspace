import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 208)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 定义训练数据集
x_train = torch.randn(1000, 2)
y_train = torch.randn(1000, 208)

# 定义数据加载器
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 初始化模型和优化器
net = Net(hidden_size=64)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# 训练模型
for epoch in range(10):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = net(x_batch)
        loss = nn.MSELoss()(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
# 在测试集上评估模型
x_test = torch.tensor([[1.0, 2.0]])
y_pred = net(x_test)
print(y_pred)






