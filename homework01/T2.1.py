import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def rgb_to_grey(image_path):
    # turn rgb to grey
    img = Image.open(image_path)
    if img.mode == 'RGBA': img = img.convert('RGB')
    
    if img.mode == 'L': 
        return np.array(img)
    else:
        img_array = np.array(img)
        grey_array = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        return grey_array.astype(np.uint8)

def compute_joint_histogram(n, image_paths, bins):
    """
    Parameters:
        n: dimension
        image_paths: list of image path
        bins: total number of intensity level
    Returns:
        histogram: array for plotting histogram, not histogram itself
    """
    first_img = rgb_to_grey(image_paths[0])
    height, width = first_img.shape
    
    images = [first_img]
    for i in range(1, n):
        grey_img = rgb_to_grey(image_paths[i])
        # make sure all images having the same size
        if grey_img.shape != (height, width):
            img_pil = Image.fromarray(grey_img)
            img_pil = img_pil.resize((width, height))
            grey_img = np.array(img_pil)
        
        images.append(grey_img)
    
    # n-dimension histogram array
    hist_shape = tuple([bins] * n)
    histogram = np.zeros(hist_shape, dtype=np.int32)
    
    for i in range(height):
        for j in range(width):
            indices = tuple(images[k][i, j] * bins // 256 for k in range(n))
            histogram[indices] += 1
    
    return histogram

def plot_2d_histogram(histogram):

    plt.figure(figsize=(10, 8))
    
    log_histogram = np.log10(histogram + 1)
    plt.imshow(log_histogram, cmap='hot', interpolation='none', origin='lower', 
               extent=[0, 255, 0, 255])
    
    plt.colorbar(label='log10(Frequency + 1)')
    plt.xlabel('intensity of original image')
    plt.ylabel('intensity of transformed image')
    plt.title('2-dimensional joint histogram')
    plt.xticks([0, 100, 200])
    plt.yticks([0, 100, 200])
    
    plt.tight_layout()
    plt.savefig('2d_joint_histogram.png', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    image_paths = ["transformed_1.jpg", "1.jpg"]
    hist = compute_joint_histogram(2, image_paths, bins=256)
    plot_2d_histogram(hist)