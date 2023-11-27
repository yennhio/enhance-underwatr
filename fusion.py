import numpy as np
from PIL import Image, ImageFilter

def max_attenuation_map(image, gamma=1.2):
    attenuation_map = np.maximum(1 - image[:, :, 0]**gamma, np.maximum(1 - image[:, :, 1]**gamma, 1 - image[:, :, 2]**gamma))
    return attenuation_map

def detailed_image(image, blur_radius=5):
    blurred_image = image.filter(ImageFilter.GaussianBlur(blur_radius))
    detailed_image = np.array(image) - np.array(blurred_image)
    return Image.fromarray(detailed_image.astype(np.uint8))

def fusion(input_image, max_attenuation, color_transfer_image):
    max_attenuation_expanded = np.expand_dims(max_attenuation, axis=-1)
    
    input_image_expanded = np.array(input_image) / 255.0
    
    fused_image = color_transfer_image + max_attenuation_expanded + input_image_expanded * (1 - max_attenuation_expanded)
    return Image.fromarray(np.clip(fused_image * 255, 0, 255).astype(np.uint8))

input_image_path = "317.jpg"
input_image = Image.open(input_image_path)

#Step 1: Max Attenuation Map
max_attenuation = max_attenuation_map(np.array(input_image) / 255.0)

#Step 2: Detailed Image
detailed_img = detailed_image(input_image)

#Step 3: Fusion
color_corrected_img = fusion(np.array(input_image) / 255.0, max_attenuation, np.array(detailed_img) / 255.0)

color_corrected_img.save("color_corrected_img.jpg")

input_image.show(title="Input Image")
detailed_img.show(title="Detailed Image")
Image.fromarray((max_attenuation * 255).astype(np.uint8)).show(title="Max Attenuation Map")
color_corrected_img.show(title="Color Corrected Image")
