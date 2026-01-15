import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import time
import math

### (1) smooth
def spatial_filter_smooth_one_channel(N, sigma, channel_array):
    """
    Parameter:
        N: kernal size (2N+1)x(2N+1)
        sigma: standard devia
        channel_array: single channel array
    Return:
        smooth_channel_array: smoothed channel array with N padding
    """
    height, width = channel_array.shape
    kernel_size = 2 * N + 1
    
    # construct gaussion kernal
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - N, j - N
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()  
    
    # pad with edge's pixel
    padded = np.pad(channel_array, pad_width = N, mode = 'edge')
    smooth_channel_array = np.zeros((height, width))
    for i in range(kernel_size):
        for j in range(kernel_size):
            smooth_channel_array += kernel[i, j] * padded[i: i + height, j: j + width]
    
    return smooth_channel_array.astype(channel_array.dtype)

def spartial_filter_smooth(N, sigma, image_path, save=True):
    """
    Parameter:
        N: kernel radius
        sigma: gaussian sigma
        image_path
        save
    """
    img = Image.open(image_path)
    img_array = np.array(img)
    # if gray
    start_time = time.time()
    if len(img_array.shape) == 2:
        smooth_array = spatial_filter_smooth_one_channel(N, sigma, img_array)
    # if rgb
    else:
        smooth_array = np.zeros_like(img_array)
        for channel in range(img_array.shape[2]): 
            smooth_array[:, :, channel] = spatial_filter_smooth_one_channel(
                N, sigma, img_array[:, :, channel]
            )
    print(f"spartial filter smooth of {image_path}: {time.time() - start_time:.2f}s")
    # save 
    if save:
        image_dir = "image"
        file_name = os.path.basename(image_path)
        output_name = "smooth_" + str(N) + "_" + file_name
        save_path = os.path.join(image_dir, output_name)
        
        result_img = Image.fromarray(smooth_array)
        result_img.save(save_path)
    
    return smooth_array

### (2) sharpen
def sharpen_one_channel(c, channel_array):
    """
    Parameter:
        c: for the laplacian kernal defined here, c is a positive number
    sharpen one channel using Laplacian filter
    Return:
        sharpen_channel_array: sharpened channel array after trimming
        laplacian = normalized \nable^{2} f
    """
    height, width = channel_array.shape
    laplacian_kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float64)
    
    kernel_size = 3
    pad_width = kernel_size // 2
    padded_channel = np.pad(channel_array, pad_width = pad_width, mode = 'edge')
    
    # for actual sharpen: dont normalize laplacian and trim sharpened channel array
    laplacian = np.zeros((height, width))
    for i in range(kernel_size):
        for j in range(kernel_size):
            laplacian += laplacian_kernel[i, j] * padded_channel[i: i + height, j: j + width]
    
    sharpen_channel_array = channel_array + c * laplacian
    sharpen_channel_array = np.clip(sharpen_channel_array, 0, 255)

    # for demonstrate and comparison: normalize laplacian to [0, 255] for visualization
    laplacian_min = (-laplacian).min()
    laplacian_max = (-laplacian).max()
    if laplacian_max - laplacian_min > 0:
        normalized_laplacian = (-laplacian - laplacian_min) / (laplacian_max - laplacian_min) * 255
    else:
        normalized_laplacian = np.zeros_like(-laplacian)
    
    return sharpen_channel_array, normalized_laplacian

def spatial_filter_sharpen(c, image_path, save=True):
   
    img = Image.open(image_path)
    img_array = np.array(img)
    start_time = time.time()
    # if grey
    if len(img_array.shape) == 2:
        sharpen_array, laplacian_array = sharpen_one_channel(c, img_array)
    # if RGB
    else:
        sharpen_array = np.zeros_like(img_array)
        laplacian_array = np.zeros_like(img_array)
        for channel in range(img_array.shape[2]):  
            sharpen_array[:, :, channel], laplacian_array[:, :, channel] = sharpen_one_channel(
                c, img_array[:, :, channel]
            )
    print(f"spartial filter sharpen of {image_path}: {time.time() - start_time:.2f}s")
    # save
    if save:
        image_dir = "image"
        file_name = os.path.basename(image_path)
        # save sharpened image
        sharpen_name = "sharpen_" + str(c) + "_" + file_name
        sharpen_path = os.path.join(image_dir, sharpen_name)
        sharpen_img = Image.fromarray(sharpen_array.astype(np.uint8))
        sharpen_img.save(sharpen_path)
        # save laplacian image
        laplacian_name = "laplacian_" + str(c) + "_" + file_name
        laplacian_path = os.path.join(image_dir, laplacian_name)
        if len(laplacian_array.shape) == 3 and laplacian_array.shape[2] == 4:
            laplacian_array = laplacian_array[:, :, :3]
        laplacian_img = Image.fromarray(laplacian_array)
        laplacian_img.save(laplacian_path)
    
    return sharpen_array, laplacian_array

### plot
def smooth_comparison(image_path, N, sigma, save_original=True):

    img = Image.open(image_path)
    img_array = np.array(img)
    
    # smooth
    smooth_result = spartial_filter_smooth(N, sigma, image_path, save_original)
    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im1 = axes[0].imshow(img_array, cmap='gray' if len(img_array.shape) == 2 else None)
    axes[0].set_title('original image')
    axes[0].set_xlabel('width')
    axes[0].set_ylabel('height')
    axes[0].axis('on')

    im2 = axes[1].imshow(smooth_result.astype(np.uint8), cmap='gray' if len(img_array.shape) == 2 else None)
    axes[1].set_title(f'smoothed image with {2*N+1}x{2*N+1}, σ={sigma} Gassian kernal')
    axes[1].axis('off')
    
    plt.tight_layout()
    # save 
    image_dir = "image"
    file_name = os.path.basename(image_path)
    name, _ = os.path.splitext(file_name)
    plt.savefig(os.path.join(image_dir, f'comparison_smooth_{N}_{name}.png'), bbox_inches='tight', dpi=150)
    plt.close() 


def sharpen_comparison(image_path, c, save_original):

    img = Image.open(image_path)
    img_array = np.array(img)
    
    # sharpen
    sharpen_result, laplacian_result = spatial_filter_sharpen(c, image_path, save_original)
    # plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im1 = axes[0].imshow(img_array, cmap='gray' if len(img_array.shape) == 2 else None)
    axes[0].set_title('original Image')
    axes[0].set_xlabel('width')
    axes[0].set_ylabel('height')
    axes[0].axis('on')

    if len(laplacian_result.shape) == 3 and laplacian_result.shape[2] == 4:
        laplacian_result = laplacian_result[:, :, :3]  
    im2 = axes[1].imshow(laplacian_result, cmap='gray' if len(laplacian_result.shape) == 2 else None)
    axes[1].set_title(f'laplacian (edge detection)')
    axes[1].axis('off')
    
    # sharpened image
    im3 = axes[2].imshow(sharpen_result.astype(np.uint8), cmap='gray' if len(img_array.shape) == 2 else None)
    axes[2].set_title(f'sharpened image (c={c})')
    axes[2].axis('off')
    
    plt.tight_layout()
    # save
    image_dir = "image"
    file_name = os.path.basename(image_path)
    name, _ = os.path.splitext(file_name)
    plt.savefig(os.path.join(image_dir, f'comparison_sharpen_{c}_{name}.png'), bbox_inches='tight', dpi=150)
    plt.close()  


if __name__ == "__main__":
    # smooth
    sigma = 2.5
    N = math.floor(3 * sigma)
    # sharpen
    c = 3
    # result comparison
    image_path = os.path.join("image", "3-1.png")
    smooth_comparison(image_path, N , sigma, save_original = True)
    sharpen_comparison(image_path, c, save_original = False)
    smooth_image = os.path.join("image", f"smooth_{N}_3-1.png")
    sharpen_comparison(smooth_image, c, save_original = True)