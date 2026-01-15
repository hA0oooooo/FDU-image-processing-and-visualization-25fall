import numpy as np
from PIL import Image
import os
import time
import matplotlib.pyplot as plt

# T3(1)
def rgb_to_grey(image_path):
    
    img = Image.open(image_path)
    if img.mode == 'RGBA': img = img.convert('RGB')
    
    if img.mode == 'L': 
        return np.array(img)
    else:
        img_array = np.array(img)
        grey_array = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        return grey_array.astype(np.uint8)

def global_histogram_equalization(image_path, L=256, save_result=True):

    grey_array = rgb_to_grey(image_path)
    height, width = grey_array.shape
    
    histogram = np.zeros(L, dtype=np.int32)
    for pixel_value in grey_array.flatten():
        histogram[pixel_value] += 1
    
    # compute cdf and make the equilization
    freq = np.zeros(256, dtype=np.float64)
    freq[0] = histogram[0]
    for i in range(1, 256):
        freq[i] = freq[i-1] + histogram[i]
    
    total_pixels = height * width
    cdf_normalized = freq / total_pixels
    
    transform = np.round(cdf_normalized * (L - 1)).astype(np.uint8)
    
    equalized_image = np.zeros_like(grey_array)
    for i in range(height):
        for j in range(width):
            original_value = grey_array[i, j]
            equalized_image[i, j] = transform[original_value]
    
    # save equilizad image
    if save_result:
        equalized_img_pil = Image.fromarray(equalized_image)
        file_name = os.path.basename(image_path)
        name, ext = os.path.splitext(file_name)
        output_name = "global_equalized_" + name + ext
        equalized_img_pil.save(output_name)
    
    return equalized_image


# T3(2)
def local_histogram_equalization(k, image_path, L=256, save_result=True):
    """
    the local window uses Z-move. for each window, only equilize the center pixel
    Parameters:
        k: size of local window
        image_path
    Return:
        equilized: 2-dimension array
    """
    grey_array = rgb_to_grey(image_path)
    height, width = grey_array.shape
    equalized = np.zeros_like(grey_array)
    
    # initialize first window at (0, 0)
    half = k // 2
    top = max(0, 0 - half)
    bottom = min(height, 0 + half + 1)
    left = max(0, 0 - half)
    right = min(width, 0 + half + 1)
    
    # build initial histogram
    current_hist = np.zeros(256, dtype=np.int32)
    for i in range(top, bottom):
        for j in range(left, right):
            current_hist[grey_array[i, j]] += 1
    current_count = (bottom - top) * (right - left)
    
    # compute cdf and equalize center pixel
    center_value = grey_array[0, 0]
    cdf = np.sum(current_hist[:center_value + 1]) / current_count
    equalized[0, 0] = np.round(cdf * (L-1)).astype(np.uint8)
    
    # first row: left to right
    for col in range(1, width):
        new_left = max(0, col - half)
        new_right = min(width, col + half + 1)
        # remove left column
        if new_left > left:
            for i in range(top, bottom):
                current_hist[grey_array[i, left]] -= 1
            current_count -= (bottom - top)
        # add right column
        if new_right > right:
            for i in range(top, bottom):
                current_hist[grey_array[i, new_right - 1]] += 1
            current_count += (bottom - top)
        
        left, right = new_left, new_right
        
        # equalize center pixel
        center_value = grey_array[0, col]
        cdf = np.sum(current_hist[:center_value + 1]) / current_count
        equalized[0, col] = np.round(cdf * (L-1)).astype(np.uint8)
    
    # other rows: Z-shape
    for row in range(1, height):
        new_top = max(0, row - half)
        new_bottom = min(height, row + half + 1)
        # remove top row
        if new_top > top:
            for j in range(left, right):
                current_hist[grey_array[top, j]] -= 1
            current_count -= (right - left)
        # add bottom row
        if new_bottom > bottom:
            for j in range(left, right):
                current_hist[grey_array[bottom, j]] += 1
            current_count += (right - left)
        
        top, bottom = new_top, new_bottom
        
        # even row: left to right
        if row % 2 == 0: 
            for col in range(width):
                if col > 0:
                    new_left = max(0, col - half)
                    new_right = min(width, col + half + 1)
                    # remove left column
                    if new_left > left:
                        for i in range(top, bottom):
                            current_hist[grey_array[i, left]] -= 1
                        current_count -= (bottom - top)
                    # add right column
                    if new_right > right:
                        for i in range(top, bottom):
                            current_hist[grey_array[i, new_right - 1]] += 1
                        current_count += (bottom - top)
                    
                    left, right = new_left, new_right
                
                # equalize center pixel
                center_value = grey_array[row, col]
                cdf = np.sum(current_hist[:center_value + 1]) / current_count
                equalized[row, col] = np.round(cdf * (L-1)).astype(np.uint8)

        # odd row: right to left
        else: 
            for col in range(width - 1, -1, -1):
                if col < width - 1:
                    new_left = max(0, col - half)
                    new_right = min(width, col + half + 1)
                    # remove right column
                    if new_right < right:
                        for i in range(top, bottom):
                            current_hist[grey_array[i, right - 1]] -= 1
                        current_count -= (bottom - top)
                    # add left column
                    if new_left < left:
                        for i in range(top, bottom):
                            current_hist[grey_array[i, new_left]] += 1
                        current_count += (bottom - top)
                    
                    left, right = new_left, new_right
                
                # equalize center pixel
                center_value = grey_array[row, col]
                cdf = np.sum(current_hist[:center_value + 1]) / current_count
                equalized[row, col] = np.round(cdf * (L-1)).astype(np.uint8)
    
    # save equalized image
    if save_result:
        equalized_img_pil = Image.fromarray(equalized)
        file_name = os.path.basename(image_path)
        name, ext = os.path.splitext(file_name)
        output_name = "local_equalized_" + name + ext
        equalized_img_pil.save(output_name)
    
    return equalized

# plot
def plot_comparison(original_array, global_equalized, local_equalized, k):

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # original image
    im1 = axes[0].imshow(original_array, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('original image')
    axes[0].set_xlabel('image width')
    axes[0].set_ylabel('image height')
    axes[0].axis('on')
    plt.colorbar(im1, ax=axes[0], shrink=0.8, pad=0.05)

    # global equilization
    im2 = axes[1].imshow(global_equalized, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('global histogram equalization')
    axes[1].set_xlabel('image width')
    axes[1].set_ylabel('image height')
    axes[1].axis('on')
    plt.colorbar(im2, ax=axes[1], shrink=0.8, pad=0.05)

    # local equilization
    im3 = axes[2].imshow(local_equalized, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f'local equalization with {k}x{k} window')
    axes[2].set_xlabel('image width')
    axes[2].set_ylabel('image height')
    axes[2].axis('on')
    plt.colorbar(im3, ax=axes[2], shrink=0.8, pad=0.05)
    
    plt.tight_layout()
    plt.savefig('comparison_equilization.png', bbox_inches='tight', dpi=150)
    plt.show()


if __name__ == "__main__":

    original_path = "3.png"
    original = rgb_to_grey(original_path)
    k = 21
    
    start_time = time.time()
    global_result = global_histogram_equalization(original_path)
    print(f"global equilization: {time.time() - start_time:.2f}s")
    start_time = time.time()
    local_result = local_histogram_equalization(k, original_path)
    print(f"local equilization: {time.time() - start_time:.2f}s")

    plot_comparison(original, global_result, local_result, k)