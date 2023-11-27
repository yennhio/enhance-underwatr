import numpy as np
from PIL import Image

# Open the image
img = Image.open("317.jpg")

# Convert the image to a NumPy array
M = np.asarray(img)

# Get the height and width
height, width, _ = M.shape

# Print the dimensions
print("Height:", height)
print("Width:", width)

def get_channel_means(red_channel, green_channel, blue_channel):
    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)

    means = [red_mean, green_mean, blue_mean]
    means.sort()
    
    if red_mean == means[0]:
        red_order = 0
        small_channel = 0
    elif red_mean == means[1]:
        red_order = 1
        medium_channel = 0
    else:
        red_order = 2
        large_channel = 0

    if green_mean == means[0]:
        green_order = 0
        small_channel = 1
    elif green_mean == means[1]:
        green_order = 1
        medium_channel = 1
    else:
        green_order = 2
        large_channel = 1

    if blue_mean == means[0]:
        blue_order = 0
        small_channel = 2
    elif blue_mean == means[1]:
        blue_order = 1
        medium_channel = 2
    else:
        blue_order = 2
        large_channel = 2

    #channels based on mean values: [[row 1: means in increasing order],
    #                                [row 2: where each channel is in the first row],
    #                                [row 3: where each channel is in second row]


    channels = np.array([[means[0], means[1], means[2]],
                [red_order, green_order, blue_order],
                [small_channel, medium_channel, large_channel]
                ])

    return channels


def get_color_loss(channels):
    return (channels[0, 2] - channels[0,1]) + (channels[0,2] - channels[0,0])

def stretch_channel(color_channel, old_img):
    return 0 + (color_channel - np.min(old_img)) * (255-0) / (np.max(old_img)-np.min(old_img))

def do_color_compensation(color_channel, large_color_channel):
    return color_channel + (np.mean(large_color_channel) - np.mean(color_channel)) * large_color_channel

def iterate_mcc():
    corrected_channels = []

    # Separate the color channels
    red_channel = M[:, :, 0]
    green_channel = M[:, :, 1]
    blue_channel = M[:, :, 2]
    color_loss = 1

    rgb_array = np.array([red_channel, green_channel, blue_channel])

    while color_loss > 0.1:
        channels = get_channel_means(rgb_array[0], rgb_array[1], rgb_array[2])

        color_loss = get_color_loss(channels)
        print(color_loss)
        
        input_img = update_input_img(rgb_array)
        color_corrected_large_channel = stretch_channel(rgb_array[int(channels[2,2])], input_img)
        print(color_corrected_large_channel)
        rgb_array[int(channels[2,2])] = color_corrected_large_channel

        if color_loss <= 0.1:
            stretched_med_channel = stretch_channel(rgb_array[int(channels[2,1])], input_img)
            rgb_array[int(channels[2,1])] = stretched_med_channel
            stretched_small_channel = stretch_channel(rgb_array[int(channels[2,0])], input_img)
            rgb_array[int(channels[2,0])] = stretched_small_channel
            print(rgb_array)
            break

        color_compensated_med_channel = do_color_compensation(rgb_array[int(channels[2,1])], rgb_array[int(channels[2,2])])
        rgb_array[int(channels[2,1])] = color_compensated_med_channel
        color_compensated_small_channel = do_color_compensation(rgb_array[int(channels[2,0])], rgb_array[int(channels[2,2])])
        rgb_array[int(channels[2,0])] = color_compensated_small_channel

        print(color_compensated_med_channel)
        print(color_compensated_small_channel)

        # channels = np.array([[means[0], means[1], means[2]],
        #         [red_order, green_order, blue_order],
        #         [small_channel, medium_channel, large_channel]
        #         ])

        rgb_array = [rgb_array[0], rgb_array[1], rgb_array[2]]

    # Convert the NumPy array to PIL Image
    image = Image.fromarray((input_img * 255).astype('uint8'))

    # Save the image to a file (change "output_image.jpg" to your desired filename)
    image.save("output_image.jpg")

    # Display the saved image
    image.show()

    return rgb_array

def update_input_img(channels):
    return np.stack((channels[0], channels[1], channels[2]), axis=-1)

def get_color_transfer_img(corrected_channels):

    color_transfer_img_array = np.stack((corrected_channels[0], corrected_channels[1], corrected_channels[2]), axis=-1)

    color_transfer_img = Image.fromarray(color_transfer_img_array.astype('uint8'))

    color_transfer_img.save("color_transfer_img.jpg")
    color_transfer_img.show()

    return color_transfer_img


    
def get_max_attenuation_map(gamma=1.2):
    attenuation_map = []

    for i in range(len(M)):
        for j in range(len(M)):
            attenuation_map[i][j] = max(1-red_channel[i][j]**gamma, 1-green_channel[i][j]**gamma, 1-blue_channel[i][j]**gamma)
        
    return attenuation_map


def gaussian_kernel(size, sigma):
    #Generate a 1D Gaussian kernel
    kernel = np.fromfunction(
        lambda x: (1/(2*np.pi*sigma**2)) * np.exp(-(x - (size-1)/2)**2 / (2*sigma**2)),
        (size,),
        dtype=np.float64
    )
    return kernel / np.sum(kernel)

def detailed_image(input_image, sigma):
    # Ensure the input image is a NumPy array
    input_array = np.array(input_image, dtype=float)

    # Create a 1D Gaussian kernel
    kernel_size = int(6 * sigma + 1)
    gaussian_kernel_1d = gaussian_kernel(kernel_size, sigma)

    # Create a 2D Gaussian kernel by taking the outer product
    gaussian_kernel_2d = np.outer(gaussian_kernel_1d, gaussian_kernel_1d)

    # Convolve the input image with the Gaussian kernel
    blurred_image = convolve(input_array, gaussian_kernel_2d, mode='constant', cval=0.0)

    # Calculate the detailed image
    detailed_image = input_array - blurred_image

    return detailed_image


corrected_channels = iterate_mcc()
get_color_transfer_img(corrected_channels)