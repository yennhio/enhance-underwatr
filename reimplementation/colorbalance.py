import numpy as np
from PIL import Image

def color_balance(a_channel, b_channel):
    Ia = np.mean(a_channel)
    Ib = np.mean(b_channel)

    if Ia > Ib:
        Ibc = Ib + (Ia - Ib) / (Ia + Ib) * Ib
        Iac = a_channel + (b_channel - a_channel) / (b_channel + a_channel) * a_channel
    else:
        Iac = a_channel + (b_channel - a_channel) / (b_channel + a_channel) * a_channel
        Ibc = Ib + (Ia - Ib) / (Ia + Ib) * Ib

    return Iac, Ibc

def apply_color_balance(image_path, output_path):
    img = Image.open(image_path)

    LAB_image = np.asarray(img.convert('LAB')).copy()

    a_channel = LAB_image[:, :, 1]
    b_channel = LAB_image[:, :, 2]

    Iac, Ibc = color_balance(a_channel, b_channel)

    LAB_image[:, :, 1] = Iac
    LAB_image[:, :, 2] = Ibc

    RGB_image = Image.fromarray(LAB_image, mode='LAB').convert('RGB')

    #Save the color balanced image as JPEG
    RGB_image.save(output_path, format='JPEG')
    RGB_image.show()

apply_color_balance("contrast_enhanced_img.jpg", "color_balanced_image.jpg")
