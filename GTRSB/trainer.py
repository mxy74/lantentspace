import csv
import itertools
import os
import pickle
import time
import random

import imageio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import CDCGAN_size32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrafficSignDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images, self.labels = self.read_traffic_signs(root_dir)
        self.transform = transform

    def read_traffic_signs(self, rootpath):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = rootpath + '/' + format(c, '05d') + '/'
            gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')
            gtReader = csv.reader(gtFile, delimiter=';')
            next(gtReader)

            # for row in gtReader:
            #     images.append(plt.imread(prefix + row[0]))
            #     labels.append(int(row[7]))
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

            if len(class_images) < 1500:
                extra_images = 1500 - len(class_images)
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
            else:
                selected_images = class_images
                selected_labels = class_labels
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


num_classes = 43
# fixed noise & label
temp_z_ = torch.randn(10, 100)
fixed_z_ = temp_z_
fixed_y_ = torch.zeros(10, 1)
# Randomly select 9 numbers from 0 to 42
random_indices = torch.randint(0, 42, (9,))

for i in range(9):
    fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
    temp = torch.ones(10, 1) + random_indices[i]
    fixed_y_ = torch.cat([fixed_y_, temp], 0)

fixed_z_ = fixed_z_.view(-1, 100, 1, 1).to(device)
fixed_y_label_ = torch.zeros(100, num_classes)
fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
fixed_y_label_ = fixed_y_label_.view(-1, num_classes, 1, 1).to(device)
with torch.no_grad():
    fixed_z_, fixed_y_label_ = Variable(fixed_z_.cuda()), Variable(fixed_y_label_.cuda())


def show_result(num_epoch, G, fixed_z_, fixed_y_label_, save=False, path='result.png'):
    G.eval()
    test_images = G(fixed_z_, fixed_y_label_)
    G.train()

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(10 * 10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        # ax[i, j].imshow(test_images[k].detach().cpu().permute(1, 2, 0).numpy())
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)
    plt.close()


# def show_result(num_epoch, show=False, save=False, path='result.png'):
#     G.eval()
#     test_images = G(fixed_z_, fixed_y_label_)
#     G.train()
#
#     size_figure_grid = 10
#     fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
#     for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
#         ax[i, j].get_xaxis().set_visible(False)
#         ax[i, j].get_yaxis().set_visible(False)
#
#     for k in range(10 * 10):
#         i = k // 10
#         j = k % 10
#         ax[i, j].cla()
#         ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')
#
#     label = 'Epoch {0}'.format(num_epoch)
#     fig.text(0.5, 0.04, label, ha='center')
#     plt.savefig(path)
#
#     if show:
#         plt.show()
#     else:
#         plt.close()


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# training parameters
batch_size = 256
lr = 0.0002
train_epoch = 20

# data_loader
img_size = 32
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_dataset = TrafficSignDataset('GTSRB_Final_Training_Images/GTSRB/Final_Training/Images', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('data', train=True, download=True, transform=transform),
#     batch_size=batch_size, shuffle=True)

# network
G = CDCGAN_size32.generator(128).to(device)
D = CDCGAN_size32.discriminator(128).to(device)
# 加载已经训练好的模型参数
# model_path_G = 'GTSRB_cDCGAN_results/GTSRB_cDCGAN_generator_param_epoch20.pth'  # 替换为你保存的生成器模型参数的路径
# model_path_D = 'GTSRB_cDCGAN_results/GTSRB_cDCGAN_discriminator_param_epoch20.pth'  # 替换为你保存的判别器模型参数的路径

# 加载生成器模型参数
# checkpoint_G = torch.load(model_path_G, map_location=device)
# G.load_state_dict(checkpoint_G)  # 如果有模型状态字典的键，使用对应的键

# 加载判别器模型参数
# checkpoint_D = torch.load(model_path_D, map_location=device)
# D.load_state_dict(checkpoint_D)  # 如果有模型状态字典的键，使用对应的键

G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
# scheduler_G = lr_scheduler.StepLR(G_optimizer, step_size=4, gamma=0.1)
# scheduler_D = lr_scheduler.StepLR(D_optimizer, step_size=4, gamma=0.1)
# results save folder
root = 'GTSRB_cDCGAN_results/'
model = 'GTSRB_cDCGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
# with open('GTSRB_cDCGAN_results/GTSRB_cDCGAN_train_hist.pkl', 'rb') as f:
#     train_hist = pickle.load(f)
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# label preprocess
onehot = torch.zeros(num_classes, num_classes)
onehot = onehot.scatter_(1, torch.LongTensor(list(range(num_classes))).view(num_classes, 1), 1).view(num_classes,
                                                                                                     num_classes, 1, 1)
fill = torch.zeros([num_classes, num_classes, img_size, img_size])
for i in range(num_classes):
    fill[i, i, :, :] = 1

print('training start!')
start_time = time.time()

for epoch in range(train_epoch):
    D_losses = []
    G_losses = []

    # learning rate decay
    if (epoch + 1) == 11:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch + 1) == 16:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")
    if (epoch + 1) == 21:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")
    if (epoch + 1) == 26:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")
    epoch_start_time = time.time()
    # 真实还是虚假，用来计算二分类损失
    y_real_ = torch.ones(batch_size).to(device)
    y_fake_ = torch.zeros(batch_size).to(device)
    y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())
    for x_, y_ in train_loader:
        # print(x_.shape)

        G.zero_grad()

        # train discriminator D
        D.zero_grad()
        # 最后一批数量不够
        mini_batch = x_.size()[0]

        if mini_batch != batch_size:
            y_real_ = torch.ones(mini_batch).to(device)
            y_fake_ = torch.zeros(mini_batch).to(device)
            y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())
        y_label_ = onehot[y_].to(device)
        y_fill_ = fill[y_].to(device)
        x_, y_fill_ = Variable(x_.cuda()), Variable(y_fill_.cuda())

        # train G
        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).to(device)
        z_ = Variable(z_.cuda())
        # 假图像
        G_result = G(z_, y_label_)
        # print(G_result.shape)
        # print(y_fill_.shape)
        D_result_fake = D(G_result, y_fill_).squeeze()
        # G_train_loss = BCE_loss(D_result_fake, y_real_)
        # G_train_loss.backward()
        # G_optimizer.step()

        # train D

        D_result = D(x_, y_fill_).squeeze()
        D_real_loss = BCE_loss(D_result, y_real_)
        # 判别器判别

        D_fake_loss = BCE_loss(D_result_fake, y_fake_)
        D_fake_score = D_result.data.mean()

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        D_losses.append(D_train_loss.item())

        # train generator G
        G.zero_grad()

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).to(device)
        z_ = Variable(z_.cuda())

        G_result = G(z_, y_label_)
        D_result_fake = D(G_result, y_fill_).squeeze()

        G_train_loss = BCE_loss(D_result_fake, y_real_)

        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.item())

        # y_fill_ = fill[y_].to(device)
        # x_, y_fill_ = Variable(x_.cuda()), Variable(y_fill_.cuda())
        #
        # D_result = D(x_, y_fill_).squeeze()
        # D_real_loss = BCE_loss(D_result, y_real_)
        #
        # z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).to(device)
        # y_ = (torch.rand(mini_batch, 1) * num_classes).type(torch.LongTensor).squeeze().to(device)
        # y_label_ = onehot[y_].to(device)
        # y_fill_ = fill[y_].to(device)
        # z_, y_label_, y_fill_ = Variable(z_.cuda()), Variable(y_label_.cuda()), Variable(y_fill_.cuda())
        # # 假图像
        # G_result = G(z_, y_label_)
        # # 判别器判别
        # D_result = D(G_result, y_fill_).squeeze()
        #
        # D_fake_loss = BCE_loss(D_result, y_fake_)
        # D_fake_score = D_result.data.mean()
        #
        # D_train_loss = D_real_loss + D_fake_loss
        #
        # D_train_loss.backward()
        # D_optimizer.step()
        #
        # D_losses.append(D_train_loss.item())
        #
        # # train generator G
        # G.zero_grad()
        #
        # z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).to(device)
        # y_ = (torch.rand(mini_batch, 1) * num_classes).type(torch.LongTensor).squeeze().to(device)
        # y_label_ = onehot[y_].to(device)
        # y_fill_ = fill[y_].to(device)
        # z_, y_label_, y_fill_ = Variable(z_.cuda()), Variable(y_label_.cuda()), Variable(y_fill_.cuda())
        #
        # G_result = G(z_, y_label_)
        # D_result = D(G_result, y_fill_).squeeze()
        #
        # G_train_loss = BCE_loss(D_result, y_real_)
        #
        # G_train_loss.backward()
        # G_optimizer.step()
        #
        # G_losses.append(G_train_loss.item())

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % (
        (epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
        torch.mean(torch.FloatTensor(G_losses))))

    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '_fixed.png'
    show_result((epoch + 1), G, fixed_z_, fixed_y_label_, save=True, path=fixed_p)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    # scheduler_G.step()
    # scheduler_D.step()

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
    torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), root + model + f'generator_param_size32_epoch{train_epoch}.pth')
torch.save(D.state_dict(), root + model + f'discriminator_param_size32_epoch{train_epoch}.pth')
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

images = []
for e in range(train_epoch):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '_fixed.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)
