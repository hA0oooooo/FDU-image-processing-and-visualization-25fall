import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time

def rgb_to_grey(image_path):
    
    img = Image.open(image_path)
    if img.mode == 'RGBA': img = img.convert('RGB')
    
    if img.mode == 'L': 
        return np.array(img)
    else:
        img_array = np.array(img)
        grey_array = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        return grey_array.astype(np.uint8)

def save_grey_image(image_path):
  
    grey_array = rgb_to_grey(image_path)
    grey_img_pil = Image.fromarray(grey_array)
    
    file_name = os.path.basename(image_path)
    name, ext = os.path.splitext(file_name)
    output_name = "grey_" + name + ext
    grey_img_pil.save(output_name)
    
    return output_name

def local_otsu_threshold(k, image_path, L=256, save_result=True):
    """
    the local window uses Z-move to conduct locally adaptive thresholding using method of local OTSU or maximum local entropy
    Parameters:
        k: size of local window
        image_path
        method: 
            local OTSU: for every threshold k, compute sigma2 = [m_{G}P1 - m]^{2} / [P1(1-P1)]
                        where P1 and m is the cdf and the weighted average of pixel of which intensity up to k, m is the global weighted average
    Return:
        threshold: 2-dimension array
    """
    grey_array = rgb_to_grey(image_path)
    height, width = grey_array.shape
    threshold = np.zeros_like(grey_array)
    
    # initialize first window at (0, 0)
    half = k // 2
    top = max(0, 0 - half)
    bottom = min(height, 0 + half + 1)
    left = max(0, 0 - half)
    right = min(width, 0 + half + 1)
    
    # build initial histogram
    current_hist = np.zeros(L, dtype=np.int32)
    for i in range(top, bottom):
        for j in range(left, right):
            current_hist[grey_array[i, j]] += 1
    current_count = (bottom - top) * (right - left)
    # compute initial threshold for center pixel
    threshold[0, 0] = compute_otsu_threshold(current_hist, current_count, grey_array[0, 0], L)
    
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
        # compute threshold for center pixel
        threshold[0, col] = compute_otsu_threshold(current_hist, current_count, grey_array[0, col], L)
    
    # other rows: Z-shape movement
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

                # compute threshold for center pixel
                threshold[row, col] = compute_otsu_threshold(current_hist, current_count, grey_array[row, col], L)

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
                
                # compute threshold for center pixel
                threshold[row, col] = compute_otsu_threshold(current_hist, current_count, grey_array[row, col], L)
    
    # save processed image
    if save_result:
        threshold_img = Image.fromarray(threshold)
        file_name = os.path.basename(image_path)
        name, ext = os.path.splitext(file_name)
        output_name = "local_otsu_threshold_" + name + ext
        threshold_img.save(output_name)

    return threshold

# auxiliary function of local OTSU
def compute_otsu_threshold(hist, count, center_value, L):
    """
    compute otsu threshold for a given histogram with adaptive strategy
    Parameters:
        hist: histogram array
        count: total pixel count
        center_value: center pixel value for threshold
    Return:
        thresholded value (0 or L-1)
    """
    if count == 0:
        return 0
    
    # normalize histogram and compute cdf
    p = hist.astype(np.float64) / count
    P1 = np.zeros(L)
    m1 = np.zeros(L)
    P1[0] = p[0]
    m1[0] = 0
    # local mean
    for k in range(1, L):
        P1[k] = P1[k-1] + p[k]
        m1[k] = m1[k-1] + k * p[k]
    # global mean
    mG = np.sum(np.arange(L) * p)
    
    # compute between-class variance for each threshold
    max_variance = -1
    best_threshold = 0
    for k in range(0, L, 4):
        if P1[k] > 0 and P1[k] < 1:
            variance = (mG * P1[k] - m1[k])**2 / (P1[k] * (1 - P1[k]))
            if variance > max_variance:
                max_variance = variance
                best_threshold = k

    best_threshold = min(best_threshold, 0.5 * mG)
    return L-1 if center_value > best_threshold else 0


def maximum_local_entropy_threshold(k, image_path, L=256, save_result=True):
    """
    the local window uses Z-move to conduct locally adaptive thresholding using method of local OTSU or maximum local entropy
    Parameters:
        k: size of local window
        image_path
        method: 
            maximum local entropy: for every threshold k, compute entropy = -sum((p_{i}/P1) * log2(p_{i}/P1)) for i=0 to k 
                                                                          + -sum((p_{i}/P2) * log2(p_{i}/P2)) for i=k+1 to L-1
                                   where P1 + P2 = 1
                                   after simplification, we have entropy =  - 1/P1 sum(p_{i} * log2(p_{i})) for i=0 to k + log2(P1) 
                                                                            + 1/(1-P1) (H(I) + sum(p_{i} * log2(p_{i})) for i=0 to k) + log2(1-P1)
                                   where H(I) = - sum(p_{i} * log2(p_{i})) for i=0 to L-1
    Return:
        threshold: 2-dimension array
    """
    grey_array = rgb_to_grey(image_path)
    height, width = grey_array.shape
    threshold = np.zeros_like(grey_array)
    
    # initialize first window at (0, 0)
    half = k // 2
    top = max(0, 0 - half)
    bottom = min(height, 0 + half + 1)
    left = max(0, 0 - half)
    right = min(width, 0 + half + 1)
    
    # build initial histogram
    current_hist = np.zeros(L, dtype=np.int32)
    for i in range(top, bottom):
        for j in range(left, right):
            current_hist[grey_array[i, j]] += 1
    current_count = (bottom - top) * (right - left)
    
    # compute initial threshold for center pixel
    threshold[0, 0] = compute_entropy_threshold(current_hist, current_count, grey_array[0, 0], L)
    
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
        
        # Compute threshold for center pixel
        threshold[0, col] = compute_entropy_threshold(current_hist, current_count, grey_array[0, col], L)
    
    # other rows: Z-move
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
                
                # compute threshold for center pixel
                threshold[row, col] = compute_entropy_threshold(current_hist, current_count, grey_array[row, col], L)

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
                
                # compute threshold for center pixel
                threshold[row, col] = compute_entropy_threshold(current_hist, current_count, grey_array[row, col], L)
    
    # save processed image
    if save_result:
        threshold_img = Image.fromarray(threshold)
        file_name = os.path.basename(image_path)
        name, ext = os.path.splitext(file_name)
        output_name = "local_entropy_threshold_" + name + ext
        threshold_img.save(output_name)

    return threshold

# auxiliary function of maximum local entropy
def compute_entropy_threshold(hist, count, center_value, L):
    """
    compute maximum entropy threshold for a given histogram
    Parameters:
        hist: histogram array
        count: total pixel count
        center_value: center pixel value for thresholding
    Return:
        threshold value (0 or L-1)
    """
    if count == 0:
        return 0
    
    # normalize histogram and compute cdf
    p = hist.astype(np.float64) / count
    P1 = np.zeros(L)
    S = np.zeros(L)
    P1[0] = p[0]    
    S[0] = p[0] * np.log2(p[0]) if p[0] > 0 else 0
    # entropy =  - 1/P1 sum(p_{i} * log2(p_{i})) for i=0 to k + log2(P1) 
    #            + 1/(1-P1) (H(I) + sum(p_{i} * log2(p_{i})) for i=0 to k) + log2(1-P1)
    for k in range(1, L):
        P1[k] = P1[k-1] + p[k]
        S[k] = S[k-1] + p[k] * np.log2(p[k]) if p[k] > 0 else S[k-1]

    max_entropy = -1
    best_threshold = 0

    for k in range(0, L, 4):
        if P1[k] > 0 and P1[k] < 1:
            P2 = 1 - P1[k]
            # maximum weighted local entropy, more attention to low intensity pixel, the writing 
            total_entropy = 0.1 * (-1/P1[k] * S[k] + np.log2(P1[k])) + 0.9 * (1/P2 * (-S[L-1] + S[k]) + np.log2(P2))
            if total_entropy > max_entropy:
                max_entropy = total_entropy
                best_threshold = k
    
    # apply threshold to center pixel
    return L-1 if center_value > best_threshold else 0


# plot
def plot_comparison(original_array, otsu_threshold, entropy_threshold, input_image, k):

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # original image
    im1 = axes[0].imshow(original_array, cmap='grey', vmin=0, vmax=255)
    axes[0].set_title('original image')
    axes[0].set_xlabel('image width')
    axes[0].set_ylabel('image height')
    axes[0].axis('on')
    plt.colorbar(im1, ax=axes[0], shrink=0.8, pad=0.05)

    # local otsu threshold
    im2 = axes[1].imshow(otsu_threshold, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'local otsu threshold with {k}x{k} window')
    axes[1].set_xlabel('image width')
    axes[1].set_ylabel('image height')
    axes[1].axis('on')
    plt.colorbar(im2, ax=axes[1], shrink=0.8, pad=0.05)

    # maximum local entropy
    im3 = axes[2].imshow(entropy_threshold, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f'maximum local entropy with {k}x{k} window')
    axes[2].set_xlabel('image width')
    axes[2].set_ylabel('image height')
    axes[2].axis('on')
    plt.colorbar(im3, ax=axes[2], shrink=0.8, pad=0.05)
    
    plt.tight_layout()
    input_name = os.path.basename(input_image)
    name, _ = os.path.splitext(input_name)
    plt.savefig(f'comparison_threshold_{name}.png', bbox_inches='tight', dpi=150)
    plt.close()  


def main(input_image, k):

    grey_image_array = rgb_to_grey(input_image)     
    
    start_time = time.time()
    otsu_image = local_otsu_threshold(k, input_image)
    print(f"local otsu threshold of {input_image}: {time.time() - start_time:.2f}s")
    start_time = time.time()
    entropy_image = maximum_local_entropy_threshold(k, input_image)
    print(f"maximum local entropy threshold of {input_image}: {time.time() - start_time:.2f}s")
    print("\n")
    
    plot_comparison(grey_image_array, otsu_image, entropy_image, input_image, k)

if __name__ == "__main__":

    k = 21
    main("2-2-1.png", k)
    main("2-2-2.png", k)