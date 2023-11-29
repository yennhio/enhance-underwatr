import numpy as np
from PIL import Image

def calculate_means(image):
    r_mean, g_mean, b_mean = np.mean(image[:, :, 0]), np.mean(image[:, :, 1]), np.mean(image[:, :, 2])
    return r_mean, g_mean, b_mean

def classify_means(r_mean, g_mean, b_mean):
    if r_mean < g_mean < b_mean:
        return 0, 1, 2
    elif r_mean < b_mean < g_mean:
        return 0, 2, 1
    elif g_mean < r_mean < b_mean:
        return 1, 0, 2
    elif g_mean < b_mean < r_mean:
        return 1, 2, 0
    elif b_mean < r_mean < g_mean:
        return 2, 0, 1
    elif b_mean < g_mean < r_mean:
        return 2, 1, 0

def color_loss(large_mean, medium_mean, small_mean):
    return (large_mean - medium_mean) + (large_mean - small_mean)

def increase_dynamic_range(channel):
    min_val, max_val = np.min(channel), np.max(channel)
    dynamic_range_channel = (channel - min_val) * 255 / (max_val - min_val)
    return dynamic_range_channel

def color_compensation(channel, large_mean, mean_medium_small):
    return channel + (large_mean - mean_medium_small) * channel

def perform_color_transfer(image):
    r_mean, g_mean, b_mean = calculate_means(image)
    large_index, medium_index, small_index = classify_means(r_mean, g_mean, b_mean)

    large_channel = image[:, :, large_index]
    medium_channel = image[:, :, medium_index]
    small_channel = image[:, :, small_index]

    color_loss_threshold = 1e-2
    current_color_loss = color_loss(np.mean(large_channel), np.mean(medium_channel), np.mean(small_channel))

    while current_color_loss > color_loss_threshold:
        large_channel_dynamic = increase_dynamic_range(large_channel)
        medium_channel_compensated = color_compensation(medium_channel, np.mean(large_channel), np.mean(medium_channel))
        small_channel_compensated = color_compensation(small_channel, np.mean(large_channel), np.mean(small_channel))

        current_color_loss = color_loss(np.mean(large_channel_dynamic), np.mean(medium_channel_compensated), np.mean(small_channel_compensated))

        large_channel = large_channel_dynamic
        medium_channel = medium_channel_compensated
        small_channel = small_channel_compensated

    enhanced_image = np.stack([large_channel, medium_channel, small_channel], axis=-1)
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)

    return enhanced_image

input_image_path = "317.jpg"
input_image = np.array(Image.open(input_image_path))
enhanced_image = perform_color_transfer(input_image)
final_output_image = Image.fromarray(enhanced_image.astype('uint8'))
final_output_image.save('color_transfer_img.jpg')

Image.fromarray(input_image).show(title="Input Image")
Image.fromarray(enhanced_image).show(title="Enhanced Image")
