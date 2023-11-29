import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

def reduce_blue_green(image_path, output_path, reduction_factor=0.95):
    img = Image.open(image_path)
    pixels = img.load()

    for i in range(img.width):
        for j in range(img.height):
            r, g, b = pixels[i, j]
            new_b = int(b * reduction_factor)
            new_g = int(g * reduction_factor)
            pixels[i, j] = (r, new_g, new_b)

    img.save(output_path)

def reduce_blacks(image_path, output_path, reduction_factor=0.8):
    img = Image.open(image_path)
    pixels = img.load()

    for i in range(img.width):
        for j in range(img.height):
            r, g, b = pixels[i, j]

            new_r = int(r + (255 - r) * (1 - reduction_factor))
            new_g = int(g + (255 - g) * (1 - reduction_factor))
            new_b = int(b + (255 - b) * (1 - reduction_factor))

            pixels[i, j] = (new_r, new_g, new_b)

    img.save(output_path)

def stretch_dynamic_range(image_path, output_path, lower_percent=5, upper_percent=100):
    img = Image.open(image_path)
    img_array = img.convert("L").getdata()
    
    lower_bound = int(np.percentile(img_array, lower_percent))
    upper_bound = int(np.percentile(img_array, upper_percent))
    
    img_stretched = Image.eval(img, lambda x: 255 * (x - lower_bound) / (upper_bound - lower_bound) if x > lower_bound else 0)
    img_stretched = img_stretched.convert("RGB")
    img_stretched.save(output_path)

def make_brighter(image_path, output_path, brightness_factor=1.3):
    img = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(img)
    img_brighter = enhancer.enhance(brightness_factor)
    img_brighter.save(output_path)

def double_image_size(image_path, output_path):
    img = Image.open(image_path)
    img_doubled = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
    img_doubled.save(output_path)

def apply_median_filter(image_path, output_path, size=3):
    img = Image.open(image_path)
    filtered_img = img.filter(ImageFilter.MedianFilter(size=size))
    filtered_img.save(output_path)

if __name__ == "__main__":
    input_image_path = "317_MLLE.jpg"
    output_directory = "output_images"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_image_path = os.path.join(output_directory, "output_image.jpg")

    reduce_blue_green(input_image_path, output_image_path)

    reduce_blacks(output_image_path, output_image_path)

    stretch_dynamic_range(output_image_path, output_image_path)

    make_brighter(output_image_path, output_image_path)

    double_image_size(output_image_path, output_image_path)

    apply_median_filter(output_image_path, output_image_path)
