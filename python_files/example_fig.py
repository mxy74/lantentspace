import os
from PIL import Image
import torch
import random

from matplotlib.gridspec import GridSpec
from torchvision import transforms
import matplotlib.pyplot as plt

from weather_effect import apply_rainy_effect, apply_snowy_effect, apply_foggy_effect, \
    apply_reflection_effect
device = torch.device("cuda:0")

pic_num = [2, 15, 21, 76, 74, 156, 172]

# Directory containing images
image_dir = r'E:\project\lantentspace\GTRSB\origin_size32_50k'

# Function to load images from the specified directory
def load_images(image_dir, pic_num):
    images = []
    for num in pic_num:
        image_path = os.path.join(image_dir, f'pic_{num}.png')
        image = Image.open(image_path)
        images.append(image)
    return images
# Load the images
imgs = load_images(image_dir, pic_num)



transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),  # Convert to float32 and scale to [0, 1]
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Convert images to tensors
imgs_tensor = [transform(img) for img in imgs]


# 随机加天气效果
effects = {
    # "Sunny": apply_sunny_effect,
    "Rainy": apply_rainy_effect,
    "Snowy": apply_snowy_effect,
    "Foggy": apply_foggy_effect,
    "Reflection": apply_reflection_effect
}

# 随机选择一个效果
# chosen_effect_name = random.choice(list(effects.keys()))
# chosen_effect_func = effects[chosen_effect_name]

# imgs = imgs.squeeze(0)
# masked_random_image_tensor = chosen_effect_func(imgs)

# Set up the plot
num_effects = len(effects)
num_images = len(imgs_tensor)
fig, axes = plt.subplots(2 * num_effects, num_images, figsize=(4 * num_images, 8 * num_effects))




# Process and plot each image with each weather effect
for i, (effect_name, effect_func) in enumerate(effects.items()):
    for j, img_tensor in enumerate(imgs_tensor):
        # Apply the weather effect
        img_tensor = img_tensor.to(device)
        # print(img_tensor.shape)
        processed_img_tensor = effect_func(img_tensor)
        processed_img_tensor = ((processed_img_tensor + 1) / 2).clamp(0.0, 1.0)

        # Convert back to PIL Image for visualization
        processed_img = transforms.ToPILImage()(processed_img_tensor.squeeze(0))

        # Plotting
        ax = axes[2*i, j]
        ax.imshow(processed_img)
        ax.axis('off')


    for j in range(num_images):
        ax_text = axes[2 * i + 1, j]  # Odd rows for text
        if j == num_images // 2:  # Center the text label in the middle column
            ax_text.text(0.5, 0.5, effect_name, ha='center', va='center', fontsize=40)
        ax_text.axis('off')

# Adjust layout to avoid overlapping titles
plt.tight_layout()
plt.show()