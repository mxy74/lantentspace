import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import utils

device = torch.device("cuda:0")
cpu = torch.device("cpu")
# Set the size of the image
width, height = 800, 800


# Function to add snowflakes of a certain size
def add_snowflakes(snow_texture, num_snowflakes, size):
    x_positions = np.random.randint(0, width, num_snowflakes)
    y_positions = np.random.randint(0, height, num_snowflakes)
    for x, y in zip(x_positions, y_positions):
        snow_texture[max(0, y - size):min(height, y + size + 1), max(0, x - size):min(width, x + size + 1)] = 255
    return snow_texture


def generate_snow_texture():
    # Create a blank image with all pixels set to black
    snow_texture = np.zeros((height, width), dtype=np.uint8)

    # Define the number of snowflakes for different sizes
    num_small_snowflakes = 10000
    num_medium_snowflakes = 500
    num_large_snowflakes = 500
    # Add small snowflakes
    snow_texture = add_snowflakes(snow_texture, num_small_snowflakes, 1)

    # Add medium snowflakes
    # snow_texture = add_snowflakes(snow_texture, num_medium_snowflakes, 2)

    # Add large snowflakes
    # add_snowflakes(num_large_snowflakes, 3)

    # Apply Gaussian Blur to the snow texture
    blurred_snow_texture = cv2.GaussianBlur(snow_texture, (3, 3), 0)
    print(blurred_snow_texture.shape)
    blurred_snow_texture = blurred_snow_texture / 255.0
    # Plot the blurred snow texture
    # plt.imshow(blurred_snow_texture, cmap='gray')
    # plt.axis('off')
    # plt.title('Blurred Snow Texture')
    # plt.show()
    tensor_snow_texture = torch.from_numpy(blurred_snow_texture)

    # Ensure correct data type (float32) and range (0-1)
    tensor_snow_texture = tensor_snow_texture.float()  # Ensure float32 data type

    # Save the tensor as an image using utils.save_image
    utils.save_image(tensor_snow_texture, 'weather_effect_img/snow_texture.png')
    # cv2.imwrite('weather_effect_img/snow_texture.png', blurred_snow_texture)


def generate_rain_texture(size, num_drops=500, drop_length=10):
    rain_texture = np.zeros((size, size), dtype=np.float32)

    for _ in range(num_drops):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        length = np.random.randint(10, drop_length)
        angle = np.random.uniform(-np.pi / 6, np.pi / 6)  # Random angle between -45 and 45 degrees
        for i in range(length):
            xi = int(x + i * np.sin(angle))
            yi = int(y + i * np.cos(angle))
            if 0 <= xi < size and 0 <= yi < size:
                rain_texture[yi, xi] = 1.0

    # # Apply motion blur
    kernel_size = 3
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[:,int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    rain_texture = cv2.filter2D(rain_texture, -1, kernel)
    print(rain_texture.shape)
    tensor_rain_texture = torch.from_numpy(rain_texture)

    # Ensure correct data type (float32) and range (0-1)
    tensor_rain_texture = tensor_rain_texture.float()  # Ensure float32 data type

    utils.save_image(tensor_rain_texture, 'weather_effect_img/rain_texture.png')

    return rain_texture


def generate_fog_texture(size, intensity=0.5, blur_size=400):
    """
    Generate a fog texture using random noise and Gaussian blur.

    Args:
    size (tuple): Size of the texture (height, width).
    intensity (float): Intensity of the fog.
    blur_size (int): Size of the Gaussian blur kernel.

    Returns:
    torch.Tensor: Fog texture tensor with values in the range [0, 1].
    """
    # Generate random noise
    fog_texture = np.random.normal(loc=0.5, scale=intensity, size=(size, size)).astype(np.float32)
    fog_texture = np.clip(fog_texture, 0, 1)  # Ensure values are within [0, 1]

    # Apply Gaussian blur to create a fog-like effect
    fog_texture = cv2.GaussianBlur(fog_texture, (blur_size, blur_size), 0)

    # Convert to tensor
    fog_texture = torch.tensor(fog_texture)
    utils.save_image(fog_texture, 'weather_effect_img/fog_texture.png')

    print(fog_texture.shape)

    return fog_texture

 # 定义强光反射效果函数
# 不规则形状
def generate_random_shape(size):

    shape = np.zeros((size, size), dtype=np.float32)
    # 多边形
    num_points = np.random.randint(5, 10)  # Random number of points
    points = np.random.randint(0, size, size=(num_points, 2))  # Random points within image size

    cv2.fillPoly(shape, [points], color=1.0)  # Fill the shape with white (1.0)

    # 随机生成圆心坐标，确保圆完全位于图像内
    # radius = np.random.randint(size // 8, size // 4)  # 随机半径
    # center = np.random.randint(radius, size - radius, size=2)  # 随机圆心坐标
    #
    # # 在图像上绘制圆形
    # cv2.circle(shape, tuple(center), radius, color=1.0, thickness=-1)  # -1 表示填充圆形

    return shape
def generate_reflection_texture(size, square_size, blur_size):
    """
    Generate a reflection texture by creating a white square and applying Gaussian blur.

    Args:
    size (tuple): Size of the texture (height, width).
    square_size (int): Size of the white square.
    blur_size (int): Size of the Gaussian blur kernel (must be a positive odd number).

    Returns:
    torch.Tensor: Reflection texture tensor with values in the range [0, 1].
    """
    reflection_texture = generate_random_shape(size)

    # Create an empty black image
    # reflection_texture = np.zeros((size, size), dtype=np.float32)

    # # Determine the center of the image
    # center_x, center_y = size // 2, size // 2
    #
    # # Create a white square in the center of the image
    # start_x, start_y = center_x - square_size // 2, center_y - square_size // 2
    # end_x, end_y = start_x + square_size, start_y + square_size
    # reflection_texture[start_y:end_y, start_x:end_x] = 1.0
    #
    # # Ensure blur_size is a positive odd number
    # if blur_size % 2 == 0:
    #     blur_size += 1

    # Apply Gaussian blur to create a circular shape with soft edges
    reflection_texture = cv2.GaussianBlur(reflection_texture, (blur_size, blur_size), 0)

    # Convert to tensor
    reflection_texture = torch.tensor(reflection_texture)
    # utils.save_image(reflection_texture, 'weather_effect_img/reflection_texture.png')


    return reflection_texture
# generate_rain_texture(800, 10000, 15)
# generate_snow_texture()
# generate_fog_texture(800, 0.5, 21)
# generate_reflection_texture(16, 10, 15)