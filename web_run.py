import numpy as np
from flask import Flask, render_template, request, jsonify
import logging
from torchcam.methods import SmoothGradCAMpp, GradCAMpp, SSCAM, CAM, GradCAM

import os
import time

import sys

sys.path.append("./python_files")
import python_files.my_tools as my_tools

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = '{"max_split_size_mb": 1024}'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import sys
import pickle
from scipy import spatial

python_files_dir = "./python_files/"  # python工具包位置
sys.path.append(python_files_dir)

model_files_dir = "./model_files/"  # 模型位置
sys.path.append(model_files_dir)
import model_files as model_all

device_id = 0
device = torch.device("cuda:" + str(device_id))

# 分类模型字典
Global_DNN_model_dict = {}
# 模型鲁棒性预测器字典
Global_rob_predictor_dict = {}
# CAM方法字典
Global_CAM_method_dict = {}
# 当前研究的数据类型
dataset_type = ""
# 反距离系数
idw_p = 50


app = Flask(__name__, template_folder="./templates", static_folder="./static")


# 刚进入时候返回mani.html页面
@app.route('/')
def go_to_main_html():
    # print("进入开始界面")
    # 这两行取消一些输出，可以加速响应过程
    log = logging.getLogger('werkzeug')
    log.disabled = True
    # 每次刷新都置空
    # 分类模型字典
    Global_DNN_model_dict.clear()
    # 模型鲁棒性预测器字典
    Global_rob_predictor_dict.clear()
    # CAM方法字典
    Global_CAM_method_dict.clear()
    return render_template('main.html')


# 这里准备的是两个模型共有的数据：数据集的坐标和高维向量，生成模型
@app.route('/prepare_shared_data', methods=["post"])
def prepare_shared_data():
    print("开始准备公共数据....")
    data = request.get_json(silent=True)
    global dataset_type  # 数据类型为共有全局变量
    global tree_2D, dict_zs, data_z_labels  # 2D树和潜向量字典设置为全局变量

    dataset_type = data["dataset_type"]

    # CIFAR10生成模型
    if dataset_type == "CIFAR10":
        # checkpoints_path = "./model_files/CIFAR10/checkpoints/BigGAN/model=G-best-weights-step=162000.pth"
        checkpoints_path = "./model_files/CIFAR10/checkpoints/BigGAN/model=G-best-weights-step=392000.pth"  # kjl测试#############
        global G
        print("数据类型为：", dataset_type)
        G = model_all.get_generative_model("CIFAR10").to(device)
        G.load_state_dict(torch.load(checkpoints_path, map_location=device)["state_dict"])
        G.eval()

        # 提前加载预处理的数据（降维后的2维坐标和对应的高维向量）
        # tree_2D_path="./static/data/CIFAR10/2D_kdTree/2D_kdTree_200000.pt"
        # data_z_path="./static/data/CIFAR10/latent_z/BigGAN_208z_200000.pt"

        # tree_2D = torch.load(tree_2D_path)
        # dict_zs = torch.load(data_z_path, map_location="cpu") #因为我之前保存数据到了GPU上，所以要回到cpu上才不会出错

        # kjl测试superviced-cnn-ae####################################start
        # 原始tsne降维
        # tree_2D_path = "./static/data/CIFAR10/2D_kdTree/2D_kdTree_50000_png_2023-08-30.pt"
        # AE加入confidence损失
        # tree_2D_path = "./static/data/CIFAR10/2D_kdTree/2D_kdTree_50000_png_2024-3-25.pt"
        tree_2D_path = "./static/data/CIFAR10/2D_kdTree/2D_kdTree_50000_png_2024-3-27_cof.pt"
        data_z_path = "./static/data/CIFAR10/latent_z/BigGAN_random_png_208z_50000_2023-08-30.pt"
        label_path = "./static/data/CIFAR10/labels/BigGAN_random_png_208z_50000_2023-08-30_labels.pt"
        tree_2D = torch.load(tree_2D_path)
        dict_zs = torch.load(data_z_path, map_location="cpu")  # 因为我之前保存数据到了GPU上，所以要回到cpu上才不会出错
        data_z_labels = torch.load(label_path,map_location="cpu")
        ##############################################################end

        # kjl测试VAE####################################################start
        # 定义VAE模型
        # class VAE(nn.Module):
        #     def __init__(self, latent_dim):
        #         super(VAE, self).__init__()

        #         self.latent_dim = latent_dim

        #         # 编码器
        #         self.encoder = nn.Sequential(
        #             nn.Linear(208, 512),
        #             nn.ReLU(),
        #             nn.Linear(512, 256),
        #             nn.ReLU(),
        #         )
        #         self.fc_mu = nn.Linear(256, latent_dim)
        #         self.fc_logvar = nn.Linear(256, latent_dim)

        #         # 解码器
        #         self.decoder = nn.Sequential(
        #             nn.Linear(latent_dim, 256),
        #             nn.ReLU(),
        #             nn.Linear(256, 512),
        #             nn.ReLU(),
        #             nn.Linear(512, 208),
        #             nn.Sigmoid()
        #         )

        #     def encode(self, x):
        #         x = self.encoder(x)
        #         mu = self.fc_mu(x)
        #         logvar = self.fc_logvar(x)
        #         return mu, logvar

        #     def reparameterize(self, mu, logvar):
        #         std = torch.exp(0.5 * logvar)
        #         eps = torch.randn_like(std)
        #         z = mu + eps * std
        #         return z

        #     def get_2D(self, x):
        #         mu, logvar = self.encode(x.view(-1, 208))
        #         z = self.reparameterize(mu, logvar)
        #         return z

        #     def decode(self, z):
        #         x = self.decoder(z)
        #         return x

        #     def forward(self, x):
        #         mu, logvar = self.encode(x.view(-1, 208))
        #         z = self.reparameterize(mu, logvar)
        #         x_recon = self.decode(z)
        #         return x_recon, mu, logvar

        # vae_path = os.path.join("./临时垃圾-随时可删/20230814vae训练", "vae_state_dict_epoch = 100 kl = 0.01 cl_w = 3.0 loss = 0.13633339115142823 loss_fn=official 2023-08-17 20:33:18.pt")
        # # 初始化VAE模型
        # vae = VAE(2)
        # vae.load_state_dict(torch.load(vae_path, map_location=device))
        # vae = vae.to(device)
        # vae.eval()
        # # 加载数据集
        # # 自定义datasets
        # class Mydata_sets__(Dataset):
        #     def __init__(self, path, device, transform=None):
        #         super(Mydata_sets__, self).__init__()
        #         self.latent_z = torch.load(path, map_location=device)

        #     def __getitem__(self, index):
        #         z = self.latent_z[index].detach()
        #         return z

        #     def __len__(self):
        #         return len(self.latent_z)

        # datasets = Mydata_sets__('./static/data/CIFAR10/latent_z/BigGAN_random_50k_png_208z_50000.pt', device = device)
        # dataLoader = DataLoader(datasets, batch_size=32, shuffle=False)
        # with torch.no_grad():
        #     for i, data in enumerate(dataLoader):
        #         data = data.to(device)
        #         mu, logvar = vae.encode(data.view(-1, 208))
        #         zs = vae.reparameterize(mu, logvar)
        #         if i == 0:
        #             z_2ds = zs
        #         else:
        #             z_2d = zs
        #             z_2ds = torch.cat((z_2ds, z_2d))
        # print("z_2ds.shape: ", z_2ds.shape)
        # # 将张量转换为 NumPy 数组
        # zs_np = z_2ds.to(torch.device("cpu")).detach().numpy()
        # # 读取类别标签（每一个向量对应一个类别）
        # labels = torch.load("./临时垃圾-随时可删/labels.pt")
        # # 获取类别数量和颜色映射
        # color_map = plt.get_cmap("tab10")
        # cifar10_labels = [
        #     "airplane", "automobile", "bird", "cat", "deer",
        #     "dog", "frog", "horse", "ship", "truck"
        # ]

        # # 画图检查
        # plt.scatter(zs_np[:, 0], zs_np[:, 1], s=0.05, c=labels, cmap=color_map)
        # plt.xlabel('Dimension 1')
        # plt.ylabel('Dimension 2')
        # plt.colorbar(label="Class")
        # # 添加类别标签
        # for i in range(10):
        #     label = cifar10_labels[i]
        #     x = zs_np[labels == i, 0].mean()
        #     y = zs_np[labels == i, 1].mean()
        #     plt.text(x, y, label, fontsize=8, ha='center', va='center', weight='bold')
        # # 保存图像
        # plt.savefig('test_vae.png') 

        # # 建立搜索树
        # tree_2D = spatial.KDTree(data=zs_np)
        # data_z_path="./static/data/CIFAR10/latent_z/BigGAN_random_50k_png_208z_50000.pt"
        # dict_zs = torch.load(data_z_path, map_location="cpu") #因为我之前保存数据到了GPU上，所以要回到cpu上才不会出错

        ###############################################################################end


    # SteerginAngle生成模型
    elif dataset_type == "SteeringAngle":
        checkpoints_path = "./model_files/SteeringAngle/checkpoints/CcGAN/ckpt_CcGAN_niters_40000_seed_2020_hard.pth"
        G = model_all.get_generative_model("SteeringAngle")
        G = nn.DataParallel(G, device_ids=[device_id])
        G.load_state_dict(torch.load(checkpoints_path, map_location=device)["netG_state_dict"])
        G = G.to(device)
        G.eval()
        # 提前加载预处理的数据（降维后的2维坐标和对应的高维向量）
        tree_2D_path = "./static/data/SteeringAngle/2D_kdTree/2D_kdTree_200000.pt"
        data_z_path = "./static/data/SteeringAngle/latent_z/CcGAN_384z_200000.pt"
        angles_path = "./static/data/SteeringAngle/angle/angles_200000.pt"

        tree_2D = torch.load(tree_2D_path)
        dict_zs = torch.load(data_z_path, map_location="cpu")  # 因为我之前保存数据到了GPU上，所以要回到cpu上才不会出错
        global angles
        angles = torch.load(angles_path, map_location='cuda:0')

    print("公共数据准本完毕！")
    return "0"


# 这里是准备分类模型相关的数据，包括分类模型它本身和相应的预测器
@app.route('/prepare_DNN_data', methods=["post"])
def prepare_DNN_data():
    print("开始准备DNN数据....")
    data = request.get_json(silent=True)

    model_id = data["model_id"]
    model_name = data["model_name"]
    print("model_id: ", model_id)
    print("model_name: ", model_name)

    # 申明使用的是全局
    global Global_DNN_model_dict
    global Global_rob_predictor_dict
    global Global_CAM_method_dict

    # 把选择None的model_id缓存的数据删掉
    if model_name == "None":
        del Global_DNN_model_dict[model_id]
        del Global_rob_predictor_dict[model_id]
        del Global_CAM_method_dict[model_id]
        return "0"

    # 分类模型（初始模型为resNet20）
    Global_DNN_model_dict[model_id] = model_all.get_DNN_model(dataset_type, model_name)
    if dataset_type == "SteeringAngle":
        print("准备回归模型。。。")
        Global_DNN_model_dict[model_id].load_state_dict(
            torch.load("./model_files/" + dataset_type + "/checkpoints/regre_model/" + model_name + ".pt",
                       map_location=device)["net_state_dict"])
    else:
        print("准备分类模型。。。")
        Global_DNN_model_dict[model_id].load_state_dict(
            torch.load("./model_files/" + dataset_type + "/checkpoints/classify_model/" + model_name + ".pt",
                       map_location=device))
    Global_DNN_model_dict[model_id].eval()
    Global_DNN_model_dict[model_id].to(device)

    # print("准备CAM方法。。。")
    # Global_CAM_method_dict[model_id] = CAM(Global_DNN_model_dict[model_id])
    # print("CAM准备完毕。。。")

    # 预测模型
    print("DNN模型准备完毕，准备鲁棒性预测模型。。。")
    Global_rob_predictor_dict[model_id] = model_all.get_rob_predictor(dataset_type, model_name)
    if dataset_type == "SteeringAngle":
        Global_rob_predictor_dict[model_id].load_state_dict((torch.load(
            "./model_files/" + dataset_type + "/checkpoints/rob_predictor/kjl_rob_predictor_" + model_name + "_wrongAngel=8.0epsilon=0.001epsilon_step=0.001.pt",
            map_location=device)))  # 自动驾驶使用的8度鲁棒性
    else:
        Global_rob_predictor_dict[model_id].load_state_dict((torch.load(
            "./model_files/" + dataset_type + "/checkpoints/rob_predictor/kjl_rob_predictor_" + model_name + ".pt",
            map_location=device)))  # 自动驾驶使用的8度鲁棒性

    # Global_rob_predictor_dict[model_id].load_state_dict((torch.load("./model_files/"+ dataset_type + "/checkpoints/rob_predictor/kjl_rob_predictor_" + model_name + "_wrongAngel=10.0epsilon=0.001epsilon_step=0.001.pt", map_location=device)))
    # Global_rob_predictor_dict[model_id].load_state_dict((torch.load("./model_files/"+ dataset_type + "/checkpoints/rob_predictor/kjl_rob_predictor_" + model_name + ".pt", map_location=device)))
    # Global_rob_predictor_dict[model_id].load_state_dict((torch.load("./model_files/"+ dataset_type + "/checkpoints/rob_predictor/Rob_predictor_ResNet20.pt", map_location=device)))
    Global_rob_predictor_dict[model_id].eval()
    Global_rob_predictor_dict[model_id].to(device)

    print("所有模型准备完毕！")
    return "0"


# 输入坐标，获取坐标对应的鲁棒性值，以及相应的图片类别，和图片对应的坐标。
@app.route('/get_information_data', methods=["post"])
def get_information_data():
    print("进入后端处理...")
    sys.stdout.flush()
    data = request.get_json(silent=True)
    coordinates = data["coordinates"]

    print("coordinates:", coordinates[:5])
    global idw_p
    idw_p = int(data["idw_p"])  # 反距离指数p

    # 将结果储存在字典中
    # robustness_dict = {}
    confidence_dict = {}
    confidence_fore_dict = {}
    img_DNN_output_lst_400_dict = {}
    img_DNN_for_output_lst_400_dict = {}
    img_coords_lst_400_dict = {}
    print("Global_DNN_model_dict.keys: ", Global_DNN_model_dict.keys())
    # 循环遍历模型字典中的每一个
    copy_dict = Global_DNN_model_dict.copy()  # 需要复制一下，避免运行时添加字典出错
    for key in copy_dict:
        DNN_model = Global_DNN_model_dict[key]
        # Rob_predictor = Global_rob_predictor_dict[key]
        # CAMmethod = Global_CAM_method_dict[key]
        # robustness = my_tools.get_robustness_data(coordinates, data_2D, dict_zs, G=G, DNN_model=DNN_model, Rob_predictor=Rob_predictor)
        #师兄的部分
        # robustness, img_labels_lst_400, img_coords_lst_400 = my_tools.get_information_other(coordinates, tree_2D, dict_zs, G=G, DNN_model=DNN_model, Rob_predictor=Rob_predictor, dataset_type=dataset_type, idw_p=idw_p)
        # 对前景图使用CAM，然后贴到back上
        # confidence_imgs, confidence_fore_imgs, img_labels_lst_400, foreimg_labels_lst_400, img_coords_lst_400 = my_tools.get_information_mix_conf(
        #     coordinates, tree_2D, dict_zs, G=G, DNN_model=DNN_model, CAMmethod=CAMmethod, dataset_type=dataset_type,
        #     idw_p=idw_p,
        #     mask_threshold=0.5)
        # 不对前景使用cam，背景取一个方块
        confidence_imgs, confidence_fore_imgs, img_labels_lst_400, foreimg_labels_lst_400, img_coords_lst_400 = my_tools.get_information_backmix(
            coordinates, tree_2D, dict_zs, data_z_labels, G=G, DNN_model=DNN_model,  dataset_type=dataset_type,
            idw_p=idw_p,
            mask_threshold=0.25)
        # robustness_dict[key] = list(format(float(n), '.3f')  for n in robustness)
        confidence_dict[key] = list(format(float(n), '.3f') for n in confidence_imgs)
        confidence_fore_dict[key] = list(format(float(n), '.3f') for n in confidence_fore_imgs)
        img_DNN_output_lst_400_dict[key] = list(format(float(n), '.3f') for n in img_labels_lst_400)
        img_DNN_for_output_lst_400_dict[key] = list(format(float(n), '.3f') for n in foreimg_labels_lst_400)
        img_coords_lst_400_dict[key] = list(
            [format(float(n[0]), '.3f'), format(float(n[1]), '.3f')] for n in img_coords_lst_400)
        # print(f'%s的51号类别：%.2f'%(key, float(img_DNN_output_lst_400_dict[key][51])))
    return jsonify(
        {
            # "robustness_dict": robustness_dict,  # 坐标对应的鲁棒性
            "confidence_dict": confidence_dict,
            "confidence_fore_dict": confidence_fore_dict,
            "img_DNN_output_lst_400_dict": img_DNN_output_lst_400_dict,  # 400张图片对应的类别
            "img_DNN_for_output_lst_400_dict": img_DNN_for_output_lst_400_dict,
            "img_coords_lst_400_dict": img_coords_lst_400_dict  # 400张图片对应的坐标（后面用来当作图片的唯一id）
        })


# 获取坐标对应的图片信息（主要是配合点击鲁棒性地图事件：评估生成后的图片）
@app.route('/get_image_information', methods=["post"])
def get_image_information():
    print("开始获取鲁棒性地图中坐标对应的信息~")
    data = request.get_json(silent=True)
    points = data["points"]
    img_name = data["img_name"]
    img_type = data["img_type"]
    # print("points: ", points)

    # 将不同模型的结果保存到字典中
    label_dict = {}
    img_robustness_dict = {}
    layer_dict = {}
    # 循环遍历模型字典中的每一个
    copy_dict = Global_DNN_model_dict.copy()  # 需要复制一下，避免运行时添加字典出错

    for key in copy_dict:
        DNN_model = Global_DNN_model_dict[key]
        Rob_predictor = Global_rob_predictor_dict[key]
        label, img_robustness, layer = my_tools.get_image_information_other(points, img_type, tree_2D, dict_zs, G=G,
                                                                            DNN_model=DNN_model,
                                                                            Rob_predictor=Rob_predictor,
                                                                            img_name=img_name,
                                                                            dataset_type=dataset_type, idw_p=idw_p)
        soft_max = nn.Softmax(dim=1)
        soft_layer = soft_max(layer)[0]
        # print("soft_layer: ", soft_layer)

        label_dict[key] = float(label)
        img_robustness = float(img_robustness)
        img_robustness_dict[key] = max(img_robustness, 0)
        layer_dict[key] = list([format(float(n), '.5f')] for n in soft_layer)
        print("confidence:",layer)
        print("softmax_conf:", layer_dict)
    return jsonify(
        {
            "label": label_dict,
            "img_robustness": img_robustness_dict,
            "layer": layer_dict
        }
    )


# 评估图片（主要是配合点击样本图片事件：评估已经存在的图片）
@app.route('/evaluate_image', methods=["post"])
def evaluate_image():
    data = request.get_json(silent=True)
    img_number = data["img_number"]
    img_name = data["img_name"]
    img_type = data["img_type"]
    # 将不同模型的结果保存到字典中
    label_dict = {}
    img_robustness_dict = {}
    layer_dict = {}
    # 循环遍历模型字典中的每一个
    copy_dict = Global_DNN_model_dict.copy()  # 需要复制一下，避免运行时添加字典出错
    for key in copy_dict:
        DNN_model = Global_DNN_model_dict[key]
        Rob_predictor = Global_rob_predictor_dict[key]

        label, img_robustness, layer = my_tools.evaluate_img(img_number, img_type, img_name=img_name, G=G, DNN_model=DNN_model,
                                                             Rob_predictor=Rob_predictor, dataset_type=dataset_type)
        soft_max = nn.Softmax(dim=1)
        soft_layer = soft_max(layer)[0]
        # print("soft_layer: ", soft_layer)
        label_dict[key] = float(label)
        img_robustness = float(img_robustness)
        img_robustness_dict[key] = max(img_robustness, 0)
        layer_dict[key] = list([format(float(n), '.3f')] for n in soft_layer)
        # print(f'%s的类别：%f'%(key, label_dict[key]))
    return jsonify(
        {
            "label": label_dict,
            "img_robustness": img_robustness_dict,
            "layer": layer_dict  # 置信度
        }
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8825)
