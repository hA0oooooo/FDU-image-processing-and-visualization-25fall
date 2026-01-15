import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import os

def bilinear_interpolation_single_channel(channel_array, N):
    """
    implement bilinear interpolation on single channel
    Parameters:
        channel_array: numpy array
        N: the scale factor
    Return: 
        new_channel: scaled numpy array
    """
    height, width = channel_array.shape
    new_height = int(N * height)
    new_width = int(N * width)
    
    i = np.arange(new_height)
    j = np.arange(new_width)
    y, x = np.meshgrid(i, j, indexing='ij')
    
    x = np.clip(x / N, 0, width - 1)
    y = np.clip(y / N, 0, height - 1)
    x1, y1 = x.astype(int), y.astype(int)
    x2 = np.minimum(x1 + 1, width - 1)
    y2 = np.minimum(y1 + 1, height - 1)
    
    dx, dy = x - x1, y - y1
    new_channel = (channel_array[y1, x1] * (1-dx) * (1-dy) +
                   channel_array[y1, x2] * dx * (1-dy) +
                   channel_array[y2, x1] * (1-dx) * dy +
                   channel_array[y2, x2] * dx * dy)
    
    return new_channel.astype(np.uint8)

def bilinear_interpolation_color(image_path, N):
    """
    implement bilinear interpolation on image (every channel of the image)
    Parameters:
        image_path
        N: the scale factor
    Return: 
        scaled_array: numpy array of scaled image
    """
    img = Image.open(image_path)
    if img.mode == 'L':
        # single channel
        img_array = np.array(img)
        scaled_array = bilinear_interpolation_single_channel(img_array, N)
    else:
        # convert to RGB format
        img = img.convert('RGB')
        img_array = np.array(img)
        _, _, c = img_array.shape
        
        scaled_channels = []
        for channel in range(c):
            channel_array = img_array[:, :, channel]
            scaled_channel = bilinear_interpolation_single_channel(channel_array, N)
            scaled_channels.append(scaled_channel)
        
        # compose all channels
        scaled_array = np.stack(scaled_channels, axis=2)
    
    return scaled_array

def downsample_and_restore(image_path, N):
    """
    restore the image directly
    Parameters:
        image_path
        N: scaled facotr
    Return: 
        restored_array
    """
    img = Image.open(image_path)
    original_size = img.size  
    
    new_width = original_size[0] // N
    new_height = original_size[1] // N
    
    small_img = img.resize((new_width, new_height), Image.NEAREST)
    restored_img = small_img.resize(original_size, Image.NEAREST)

    restored_array = np.array(restored_img)
    restored_img = Image.fromarray(restored_array)

    file_name = os.path.basename(image_path)
    name, ext = os.path.splitext(file_name)
    output_path = f"simple_restore_{name}{ext}"
    restored_img.save(output_path)

    return restored_array

def downsample_and_bilinear_restore(image_path, N):
    """
    restore the image using bilinear interporlation
    Parameters:
        image_path
        N: scaled facotr
    Return: 
        restored_array
    """
    img = Image.open(image_path)
    original_size = img.size 
    
    new_width = original_size[0] // N
    new_height = original_size[1] // N

    small_img = img.resize((new_width, new_height), Image.NEAREST)
    
    file_name = os.path.basename(image_path)
    name, ext = os.path.splitext(file_name)
    temp_path = f"temp_small_{name}{ext}"
    small_img.save(temp_path)
    
    restored_array = bilinear_interpolation_color(temp_path, N)
    
    os.remove(temp_path)
    
    restored_img = Image.fromarray(restored_array)
    output_path = f"bilinear_restore_{name}{ext}"
    restored_img.save(output_path)

    return restored_array

# plot
def plot_comparison_interpolation(image_path, simple_array, bilinear_array, N):

    fig, axes = plt.subplots(1, 3, figsize=(21, 9))
    
    # original image path
    axes[0].imshow(simple_array)
    axes[0].set_title(f'downsample and restore directly with scaled factor {N}')
    axes[0].axis('off')
    
    # scaled image array
    axes[1].imshow(bilinear_array)
    axes[1].set_title(f'bilinear interpolation')
    axes[1].axis('off')
    
    # original image
    original_img = Image.open(image_path)
    axes[2].imshow(original_img)
    axes[2].set_title(f'original image')
    axes[2].axis('off')

    plt.tight_layout()
    file_name = os.path.basename(image_path)
    name, _ = os.path.splitext(file_name)
    comparison_path = f"comparison_interpolation_{N}_{name}.png"
    plt.savefig(comparison_path, bbox_inches='tight', dpi=300)
    plt.close()
    
if __name__ == "__main__":

    image_path = "2-3.png"
    N = 3

    start_time = time.time()
    simple_array = downsample_and_restore(image_path, N)
    print(f"downsample and restore directly: {time.time() - start_time:.2f}s")
    
    start_time = time.time()
    bilinear_array = downsample_and_bilinear_restore(image_path, N)
    print(f"downsample and bilinear restore: {time.time() - start_time:.2f}s")

    plot_comparison_interpolation(image_path, simple_array, bilinear_array, N)