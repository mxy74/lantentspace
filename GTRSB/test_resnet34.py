import csv
import warnings


import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from classify_model.resnet import ResNet18
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning)

# # 1、同时加载模型和参数
# if torch.cuda.is_available():
#       resnet_model = torch.load(args.model_path)
#       print("GPU加载模型")
# else:
#       resnet_model = torch.load(args.model_path, map_location=torch.device('cpu'))
#       print("CPU加载模型")

# 分别加载结构和参数

# resnet_model = torch.load('classify_model/resnet34_model.pth').to(device)
# resnet_model_weights = torch.load('classify_model/train_fc_model_param_22.pth')
# resnet_model.load_state_dict(resnet_model_weights)
# print("GPU加载模型")
# 将模型切换到评估模式进行预测
# resnet_model.eval()

# 加载整个 state_dict
checkpoint = torch.load('classify_model/ResNet18.pth', map_location=device)

# 提取模型参数
state_dict = checkpoint['models']
model = ResNet18().to(device)
model.load_state_dict(state_dict)
model.eval()

class TrafficSignDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images, self.labels = self.read_traffic_signs(root_dir)
        self.transform = transform

    def read_traffic_signs(self, rootpath):
        images = []
        labels = []
        prefix = rootpath + '/'
        gtFile = open(prefix + 'GT-final_test.csv')
        gtReader = csv.reader(gtFile, delimiter=';')
        next(gtReader)
        for row in gtReader:
            images.append(plt.imread(prefix + row[0]))
            labels.append(int(row[7]))
        gtFile.close()
        return images, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image.to(device), label
# def models(input):
#     """
#     用于分类的model，返回对抗样本输入后的模型预测的向量
#     :param input: 需要分类的图像
#     :return: 置信度向量 和 对应标签
#     """
#
#     # 输入图像为单张图像，增加一个batch维度
#     # input_img = transform(input).unsqueeze(0).to(device)
#
#     output = resnet_model(input)
#     output = torch.squeeze(output, dim=0)
#     output = F.softmax(output, dim=0)
#     label = output.argmax(dim=0, keepdim=True)
#
#     return output, label


# 用于测试对训练数据进行分类的效果
if __name__ == "__main__":

    # data_loader
    img_size = 32
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_dataset = TrafficSignDataset('GTSRB_Final_Test_Images/GTSRB/Final_Test/Images', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    label_list = []
    real_label_list = []
    counts = 0
    number = 0
    # 顺序读取每一行，取出其中指定的列
    for index, (img, y_) in enumerate(train_loader):

        output = model(img)
        output = output.detach()
        label = output.max(1, keepdim=True)[1]

        # 输出当前的置信度和标签，并加入label_list中
        label_list.append(label.item())

        real_label_list.append(y_)

        # 如果输出结果与标注结果一致，则+1
        if label_list[index] == real_label_list[index]:
            counts = counts + 1
        # 输出当前的准确率
        print(f"curr_pred_accuracy : {counts / (index + 1):.4f}")
        number = index

    # 计算预测准确率
    # 似乎最后的index+1也能表示测试集图像的数量
    print(f"pred_accuracy : {counts / (number + 1):.4f}")