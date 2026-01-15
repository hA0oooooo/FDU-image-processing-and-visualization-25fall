import numpy as np
from PIL import Image
from numba import njit
import os
import matplotlib.pyplot as plt
import time

@njit()
def erode(array, k):
    
    height, width = array.shape
    erodeArray = np.zeros((height, width), dtype=array.dtype)

    half = k // 2
    for i in range(height):
        for j in range(width):
            top = max(0, i - half)
            bottom = min(height, i + half + 1)
            left = max(0, j - half)
            right = min(width, j + half + 1)
            erodeArray[i, j] = np.min(array[top:bottom, left:right])

    return erodeArray

@njit()
def dilate(array, k):
    
    height, width = array.shape
    dilateArray = np.zeros((height, width), dtype=array.dtype)

    half = k // 2
    for i in range(height):
        for j in range(width):
            top = max(0, i - half)
            bottom = min(height, i + half + 1)
            left = max(0, j - half)
            right = min(width, j + half + 1)
            dilateArray[i, j] = np.max(array[top:bottom, left:right])

    return dilateArray

@njit()
def opening(array, k):

    erodeArray = erode(array, k)
    openArray = dilate(erodeArray, k)

    return openArray

@njit()
def closing(array, k):

    dilateArray = dilate(array, k)
    closeArray = erode(dilateArray, k)

    return closeArray

if __name__ == "__main__":

    image_dir = "image"
    image_path = os.path.join(image_dir, "zmic_fdu_noise.bmp")
    image = Image.open(image_path)
    array = np.array(image)

    kclose = 5
    kopen = 5

    start = time.time()
    # process
    closeArray = closing(array, kclose)
    processArray = opening(closeArray, kopen)
    print(time.time() - start)

    # load target image
    target_path = os.path.join(image_dir, "zmic_fdu.bmp")
    target_image = Image.open(target_path)
    target_array = np.array(target_image)

    # comparison: original, processed, target
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    axes[0].imshow(array, cmap='gray')
    axes[0].set_title('original')
    axes[0].axis('off')

    axes[1].imshow(processArray, cmap='gray')
    axes[1].set_title('processed')
    axes[1].axis('off')

    axes[2].imshow(target_array, cmap='gray')
    axes[2].set_title('target')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, f"comparison_6-2"), bbox_inches='tight', dpi=150)
    plt.close()