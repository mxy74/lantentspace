from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
import random

def apply_sunny_effect(image):
    # 增加亮度和对比度
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.5)  # 提高亮度
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)  # 提高对比度
    return image

def apply_rainy_effect(image):
    # 创建一个雨滴图层
    rain_layer = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(rain_layer)
    for _ in range(1000):
        x1 = random.randint(0, image.size[0])
        y1 = random.randint(0, image.size[1])
        x2 = x1 + random.randint(-3, 3)
        y2 = y1 + random.randint(10, 20)
        draw.line((x1, y1, x2, y2), fill=(255, 255, 255, 128))
    rain_layer = rain_layer.filter(ImageFilter.BLUR)
    return Image.alpha_composite(image.convert("RGBA"), rain_layer)

def apply_snowy_effect(image):
    # 创建一个雪花图层
    snow_layer = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(snow_layer)
    for _ in range(1000):
        x = random.randint(0, image.size[0])
        y = random.randint(0, image.size[1])
        draw.ellipse((x, y, x+2, y+2), fill=(255, 255, 255, 128))
    return Image.alpha_composite(image.convert("RGBA"), snow_layer)

# 加载图像
image_path = 'pic_57.png'  # 替换为你的图像路径
image = Image.open(image_path)

# 应用效果
sunny_image = apply_sunny_effect(image)
rainy_image = apply_rainy_effect(image)
snowy_image = apply_snowy_effect(image)

# 保存结果
sunny_image.save('sunny_image.jpg')
rainy_image.save('rainy_image.jpg')
snowy_image.save('snowy_image.jpg')
