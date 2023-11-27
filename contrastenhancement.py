import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter
from PIL import Image

def calculate_mean_variance(local_block):
    mean_block = np.mean(local_block)
    variance_block = np.var(local_block)
    return mean_block, variance_block

def enhance_high_frequency(local_block, mean_block, alpha, beta, sigma_G, sigma_B):
    alpha = sigma_G / sigma_B
    enhancement_factor = np.minimum(alpha, beta)
    high_frequency_component = (local_block - mean_block) * enhancement_factor
    enhanced_block = mean_block + high_frequency_component
    return enhanced_block

def guided_filter(refined_block, guided_image, epsilon=1e-8):
    mean_guided = convolve(guided_image, np.ones_like(guided_image) / guided_image.size, mode='reflect')
    mean_refined = convolve(refined_block, np.ones_like(refined_block) / refined_block.size, mode='reflect')
    correlation = convolve(guided_image * refined_block, np.ones_like(refined_block) / refined_block.size, mode='reflect')
    covariance = correlation - mean_guided * mean_refined

    a = covariance / (guided_image.var() + epsilon)
    b = mean_refined - a * mean_guided

    mean_a = convolve(a, np.ones_like(a) / a.size, mode='reflect')
    mean_b = convolve(b, np.ones_like(b) / b.size, mode='reflect')

    refined_block = mean_a * guided_image + mean_b

    return refined_block

def minimize_difference(refined_block, enhanced_block):
    loss = np.sum((refined_block - enhanced_block)**2)
    return loss

def operate_on_blocks(input_image, block_size=10, alpha=1.0, beta=2.0, sigma_G=1.0):
    enhanced_image = np.zeros_like(input_image)

    for i in range(0, input_image.shape[0], block_size):
        for j in range(0, input_image.shape[1], block_size):
            local_block = input_image[i:i+block_size, j:j+block_size, 0]

            mean_block, variance_block = calculate_mean_variance(local_block)

            enhanced_block = enhance_high_frequency(local_block, mean_block, alpha, beta, sigma_G, variance_block)

            min_value = np.min(enhanced_block)
            max_value = np.max(enhanced_block)

            if max_value - min_value != 0:
                guided_image = (enhanced_block - min_value) / (max_value - min_value)
            else:
                guided_image = enhanced_block

            refined_block = guided_filter(enhanced_block, guided_image)

            enhanced_image[i:i+block_size, j:j+block_size, 0] = refined_block

    return enhanced_image

def contrast_enhancement(input_image_path, output_image_path):
    img = Image.open(input_image_path)

    M = np.asarray(img)

    block_size = 10
    alpha = 2.0
    beta = 2.0
    sigma_G = 1.0

    enhanced_image = operate_on_blocks(M, block_size, alpha, beta)

    final_output_image = Image.fromarray(enhanced_image.astype('uint8'))

    final_output_image.save(output_image_path)
    final_output_image.show()

contrast_enhancement("color_transfer_img.jpg", "contrast_enhanced_img.jpg")
