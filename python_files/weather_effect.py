import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as TF

from PIL import Image, ImageDraw, ImageFilter
import random
import os

from python_files.weather_effect_example import generate_reflection_texture

device = torch.device("cuda:0")
cpu = torch.device("cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 定义天气效果函数
# 定义天气效果函数
def apply_sunny_effect(tensor):
    mean_brightness = torch.mean(tensor)
    enhancement_factor = 2.0 / (1 + torch.exp(-mean_brightness))  # 例如，使用 sigmoid 函数调整因子

    # 实现晴天效果，这里只是一个示例
    modified_tensor = tensor * max(enhancement_factor, 1.3)
    modified_tensor = modified_tensor.clamp(-1.0, 1.0)
    return modified_tensor
def apply_rainy_effect(tensor):
    # Load and resize/crop the snowy texture image to match tensor size
    # rainy_texture = Image.open('weather_effect_img/rain_texture.png')

    rainy_texture = Image.open('python_files/weather_effect_img/rain_texture.png')
    # Randomly crop a 32x32 patch from the snowy texture image
    i, j, h, w = transforms.RandomCrop.get_params(rainy_texture, output_size=(32, 32))
    rainy_texture = transforms.functional.crop(rainy_texture, i, j, h, w)
    # 0-1
    rainy_texture = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])(rainy_texture)
    mask = (rainy_texture > 0.3).float()
    mask = mask.expand(3, -1, -1).to(device)
    rainy_texture = rainy_texture.expand(3, -1, -1)  # Expand single channel to three channels

    # Normalize snowy_texture to range [-1, 1]
    rainy_texture = rainy_texture * 2 - 1  # assuming original range is [0, 1]

    # Ensure snowy_texture and tensor are both tensors
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)
    rainy_texture = rainy_texture.to(device)
    # Add snowy effect to tensor
    # tensor_with_snow = tensor + 0 * snowy_texture
    tensor_with_rain = tensor * (1 - mask) + (rainy_texture * 0.5 + tensor) * mask

    # Clamp the values to ensure they are within [-1, 1]
    tensor_with_rain = torch.clamp(tensor_with_rain, min=-1, max=1)

    return tensor_with_rain

def apply_snowy_effect(tensor):
    # Load and resize/crop the snowy texture image to match tensor size
    snowy_texture = Image.open('python_files/weather_effect_img/snow_texture.png')
    # snowy_texture = Image.open('weather_effect_img/snow_texture.png')

    # Randomly crop a 32x32 patch from the snowy texture image
    i, j, h, w = transforms.RandomCrop.get_params(snowy_texture, output_size=(32, 32))
    snowy_texture = transforms.functional.crop(snowy_texture, i, j, h, w)
    # 0-1
    snowy_texture = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])(snowy_texture)
    mask = (snowy_texture > 0.5).float()
    mask = mask.expand(3, -1, -1).to(device)
    snowy_texture = snowy_texture.expand(3, -1, -1)  # Expand single channel to three channels

    # Normalize snowy_texture to range [-1, 1]
    snowy_texture = snowy_texture * 2 - 1  # assuming original range is [0, 1]

    # Ensure snowy_texture and tensor are both tensors
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)
    snowy_texture = snowy_texture.to(device)
    # Add snowy effect to tensor
    # tensor_with_snow = tensor + 0 * snowy_texture
    tensor_with_snow = tensor * (1 - mask) + (snowy_texture + tensor) * mask

    # Clamp the values to ensure they are within [-1, 1]
    tensor_with_snow = torch.clamp(tensor_with_snow, min=-1, max=1)

    return tensor_with_snow
def apply_foggy_effect(tensor):
    # 实现雾天效果，这里只是一个示例
    # Load and resize/crop the snowy texture image to match tensor size

    fog_texture = Image.open('python_files/weather_effect_img/fog_texture.png')
    # fog_texture = Image.open('weather_effect_img/fog_texture.png')


    # Randomly crop a 32x32 patch from the snowy texture image
    i, j, h, w = transforms.RandomCrop.get_params(fog_texture, output_size=(32, 32))
    fog_texture = transforms.functional.crop(fog_texture, i, j, h, w)
    # 0-1
    fog_texture = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])(fog_texture)
    # mask = (snowy_texture > 0.5).float()
    # mask = mask.expand(3, -1, -1).to(device)
    fog_texture = fog_texture.expand(3, -1, -1)  # Expand single channel to three channels

    # Normalize snowy_texture to range [-1, 1]
    fog_texture = fog_texture * 2 - 1  # assuming original range is [0, 1]

    # Ensure snowy_texture and tensor are both tensors
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)
    fog_texture = fog_texture.to(device)
    # Add snowy effect to tensor
    # tensor_with_snow = tensor + 0 * snowy_texture
    tensor_with_fog = tensor * (1 - 0.3) + fog_texture * 0.3

    # Clamp the values to ensure they are within [-1, 1]
    tensor_with_fog = torch.clamp(tensor_with_fog, min=-1, max=1)

    return tensor_with_fog




# 定义强光反射效果函数
def apply_reflection_effect(tensor):
    """
    Apply reflection effect to an image tensor.

    Args:
    tensor (torch.Tensor): Input image tensor with values in the range [-1, 1].
    reflection_texture (torch.Tensor): Reflection texture tensor with values in the range [0, 1].
    alpha (float): Blending factor for the reflection effect.

    Returns:
    torch.Tensor: Image tensor with reflection effect applied.
    """
    # reflection_texture = Image.open('weather_effect_img/reflection_texture.png')
    reflection_texture = generate_reflection_texture(15, 10, 15)
    # reflection_texture = transforms.Compose([
    #     transforms.ToTensor()
    # ])(reflection_texture)
    reflection_texture = reflection_texture.expand(3, -1, -1)  # Expand to 3 channels
    reflection_texture = reflection_texture * 2 - 1  # Convert reflection texture to range [-1, 1]

    # Randomly determine the position to add the reflection
    # print(tensor.shape)
    _, height, width = tensor.shape
    start_x = np.random.randint(0, width - reflection_texture.shape[2])
    start_y = np.random.randint(0, height - reflection_texture.shape[1])

    # Create an empty tensor for the reflection effect
    reflection_effect = torch.zeros_like(tensor)

    # Add the reflection texture at the random position
    reflection_effect[:, start_y:start_y + reflection_texture.shape[1],
    start_x:start_x + reflection_texture.shape[2]] = reflection_texture

    mask = (reflection_effect > 0.1).float()

    # Blend the reflection effect with the original image
    tensor_with_reflection = tensor * (1 - mask) + (reflection_effect*4+tensor) * mask

    # Clamp the values to ensure they are within [-1, 1]
    tensor_with_reflection = torch.clamp(tensor_with_reflection, min=-1, max=1)

    return tensor_with_reflection


# # 加载tensor并应用效果
# file_path = '../static/data/GTSRB/pic/grid_fore_images_tensor/save_generate_fore_imgs_tensor.pt'  # 替换为你的路径
# tensor_images = torch.load(file_path)
# # tensor_images = torch.stack(tensor_images)
#
# # 确保tensor格式为[1600, 3, 32, 32]
# # assert tensor_images.shape == (1600, 3, 32, 32), "Tensor shape should be [1600, 3, 32, 32]"
# # 随机选择10个图像
# random_indices = random.sample(range(400), 10)
# selected_tensors = []
# print(random_indices)
# print(len(tensor_images))
# for i in random_indices:
#     selected_tensors.append(tensor_images[i])
#
# # selected_tensors = tensor_images[random_indices]
# # selected_tensors = selected_tensors.cpu()
#
# # 定义一个函数应用某种天气效果
# # 定义一个函数应用某种天气效果
# def apply_weather_effect(tensors, effect_fn):
#     modified_tensors = []
#     for tensor in tensors:
#         modified_tensor = effect_fn(tensor)
#         modified_tensors.append(modified_tensor)
#     return modified_tensors
#
# # 应用不同的天气效果
# effects = {
#     "Sunny": apply_sunny_effect,
#     "Rainy": apply_rainy_effect,
#     "Snowy": apply_snowy_effect,
#     "Foggy": apply_foggy_effect,
#     "Reflection": apply_reflection_effect
# }
#
# effect_names = list(effects.keys())
# num_effects = len(effects)
# fig, axes = plt.subplots(len(selected_tensors), num_effects + 1, figsize=(15, 15))
#
# ## 遍历选择的图像和效果
# for i, tensor in enumerate(selected_tensors):
#     # print(tensor.shape)
#     # 显示原始图像
#     # tensor = tensor.unsqueeze(0)  # 添加批次维度
#     # tensor = tensor.detach().cpu()
#     original_img = ((tensor + 1) / 2).clamp(0.0, 1.0)
#     axes[i, 0].imshow(transforms.ToPILImage()(original_img))
#     axes[i, 0].set_title("Original")
#     axes[i, 0].axis('off')
#
#     # 应用并显示每种效果
#     for j, (effect_name, effect_fn) in enumerate(effects.items(), start=1):
#         modified_tensor = effect_fn(tensor)
#         modified_img = ((modified_tensor + 1) / 2).clamp(0.0, 1.0)
#         axes[i, j].imshow(transforms.ToPILImage()(modified_img))
#         axes[i, j].set_title(effect_name)
#         axes[i, j].axis('off')
#
# # 调整布局
# plt.tight_layout()
# plt.show()