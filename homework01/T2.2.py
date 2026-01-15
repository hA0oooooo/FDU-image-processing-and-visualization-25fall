import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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

def compute_histogram(image_path):
    # global statistics of the image
    img = Image.open(image_path)
    grey_array = rgb_to_grey(image_path)
    histogram = np.zeros(256, dtype=np.int32)
    for pixel_value in grey_array.flatten():
        histogram[pixel_value] += 1
    return histogram

def compute_local_mean(k, image_path):
    """
    the local window uses Z-move to compute the information of the next window based on the previous one
    Parameters:
        k: size of local window
        image_path
    Return:
        mean: 2-dimension array
    """
    grey_array = rgb_to_grey(image_path)
    height, width = grey_array.shape
    mean = np.zeros((height, width), dtype=np.float64)

    # initial window
    half = k // 2
    top = max(0, 0 - half)
    bottom = min(height, 0 + half + 1)
    left = max(0, 0 - half)
    right = min(width, 0 + half + 1)

    current_window = grey_array[top:bottom, left:right]
    current_sum = np.sum(current_window)
    current_count = current_window.shape[0] * current_window.shape[1]
    mean[0, 0] = current_sum / current_count

    # first row update
    for col in range(1, width):
        new_left = max(0, col - half)
        new_right = min(width, col + half + 1)
        # remove left col if exists
        if new_left > left:
            removed_col = grey_array[top: bottom, left]
            current_sum -= np.sum(removed_col)
            current_count -= (bottom - top)
        # add right col if exists
        if new_right > right:
            added_col = grey_array[top: bottom, right]
            current_sum += np.sum(added_col)
            current_count += (bottom - top)
        # renew boundary
        left, right = new_left, new_right
        mean[0, col] = current_sum / current_count

    # other row update
    for row in range(1, height):
        new_top = max(0, row - half)
        new_bottom = min(height, row + half + 1)
        # remove top row if exists
        if new_top > top:
            removed_row = grey_array[top, left:right]
            current_sum -= np.sum(removed_row)
            current_count -= (right - left)
        # add bottom row if exists
        if new_bottom > bottom:
            added_row = grey_array[bottom, left:right]
            current_sum += np.sum(added_row)
            current_count += (right - left)
        # renew boundary
        top, bottom = new_top, new_bottom

        # if even row, the window moves from the left to right
        if row % 2 == 0:  
            for col in range(1, width):
                # the same as first row update
                new_left = max(0, col - half)
                new_right = min(width, col + half + 1)
                # left, remove
                if new_left > left:
                    removed_col = grey_array[top:bottom, left]
                    current_sum -= np.sum(removed_col)
                    current_count -= (bottom - top)
                # right, add
                if new_right > right:
                    added_col = grey_array[top:bottom, new_right - 1]
                    current_sum += np.sum(added_col)
                    current_count += (bottom - top)
                
                left, right = new_left, new_right
                mean[row, col] = current_sum / current_count
        # if odd row, the window moves from the right to left
        else:  
            for col in range(width - 2, -1, -1):
                # the oppsite as first row update
                new_left = max(0, col - half)
                new_right = min(width, col + half + 1)
                # left, add
                if new_left < left:
                    added_col = grey_array[top:bottom, new_left]
                    current_sum += np.sum(added_col)
                    current_count += (bottom - top)
                # right, remove
                if new_right < right:
                    removed_col = grey_array[top:bottom, right-1]
                    current_sum -= np.sum(removed_col)
                    current_count -= (bottom - top)

                left, right = new_left, new_right
                mean[row, col] = current_sum / current_count

    return mean

def compute_local_std(k, image_path):
    # compute local standard error, but cannot use 
    grey_array = rgb_to_grey(image_path)
    height, width = grey_array.shape
    std = np.zeros((height, width), dtype=np.float64)
    
    for row in range(height):
        for col in range(width):
            half = k // 2
            top = max(0, row - half)
            bottom = min(height, row + half + 1)
            left = max(0, col - half)
            right = min(width, col + half + 1)
            
            local_window = grey_array[top:bottom, left:right]
            std[row, col] = np.std(local_window)

    return std

def compute_local_entropy(k, image_path):
    """
    the local window uses Z-move
    Return:
        entropy: 2-dimension array
    """
    grey_array = rgb_to_grey(image_path)
    height, width = grey_array.shape
    entropy = np.zeros((height, width), dtype=np.float64)

    # initial window
    half = k // 2
    top = max(0, 0 - half)
    bottom = min(height, 0 + half + 1)
    left = max(0, 0 - half)
    right = min(width, 0 + half + 1)

    current_window = grey_array[top:bottom, left:right]
    current_hist = np.zeros(256, dtype=np.int32)
    for pixel in current_window.flatten():
        current_hist[pixel] += 1
    
    current_count = current_window.shape[0] * current_window.shape[1]
    entropy[0, 0] = compute_entropy_from_hist(current_hist, current_count)

    # first row update
    for col in range(1, width):
        new_left = max(0, col - half)
        new_right = min(width, col + half + 1)
        # remove left col if exists
        if new_left > left:
            removed_col = grey_array[top: bottom, left]
            for pixel in removed_col.flatten():
                current_hist[pixel] -= 1
            current_count -= (bottom - top)
        # add right col if exists
        if new_right > right:
            added_col = grey_array[top: bottom, right]
            for pixel in added_col.flatten():
                current_hist[pixel] += 1
            current_count += (bottom - top)
        # renew boundary
        left, right = new_left, new_right
        entropy[0, col] = compute_entropy_from_hist(current_hist, current_count)

    # other row update
    for row in range(1, height):
        new_top = max(0, row - half)
        new_bottom = min(height, row + half + 1)
        # remove top row if exists
        if new_top > top:
            removed_row = grey_array[top, left:right]
            for pixel in removed_row.flatten():
                current_hist[pixel] -= 1
            current_count -= (right - left)
        # add bottom row if exists
        if new_bottom > bottom:
            added_row = grey_array[bottom, left:right]
            for pixel in added_row.flatten():
                current_hist[pixel] += 1
            current_count += (right - left)
        # renew boundary
        top, bottom = new_top, new_bottom

        # if even row, the window moves from the left to right
        if row % 2 == 0:  
            for col in range(1, width):
                # the same as first row update
                new_left = max(0, col - half)
                new_right = min(width, col + half + 1)
                
                if new_left > left:
                    removed_col = grey_array[top:bottom, left]
                    for pixel in removed_col.flatten():
                        current_hist[pixel] -= 1
                    current_count -= (bottom - top)
                
                if new_right > right:
                    added_col = grey_array[top:bottom, right]
                    for pixel in added_col.flatten():
                        current_hist[pixel] += 1
                    current_count += (bottom - top)
                
                left, right = new_left, new_right
                entropy[row, col] = compute_entropy_from_hist(current_hist, current_count)
        # if odd row, the window moves from the right to left
        else:  
            for col in range(width - 2, -1, -1):
                # the opposite as first row update
                new_left = max(0, col - half)
                new_right = min(width, col + half + 1)
                
                if new_right < right:
                    removed_col = grey_array[top:bottom, right-1]
                    for pixel in removed_col.flatten():
                        current_hist[pixel] -= 1
                    current_count -= (bottom - top)
                
                if new_left < left:
                    added_col = grey_array[top:bottom, new_left]
                    for pixel in added_col.flatten():
                        current_hist[pixel] += 1
                    current_count += (bottom - top)
                
                left, right = new_left, new_right
                entropy[row, col] = compute_entropy_from_hist(current_hist, current_count)

    return entropy

def compute_entropy_from_hist(hist, count):
    entropy = 0.0
    for freq in hist:
        if freq > 0:
            p = freq / count
            entropy -= p * np.log2(p)
    return entropy

# plot function, three statistics together
def plot_local_statistics(mean, std, entropy, k):

    fig = plt.figure(figsize=(15, 6))
    height, width = mean.shape
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    
    # local mean
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, mean, cmap='Blues', alpha=1.0)
    ax1.set_title(f'local mean with {k}x{k} window')
    ax1.set_xlabel('image width')
    ax1.set_ylabel('image height')
    ax1.set_zlabel('local mean')
    plt.colorbar(surf1, ax=ax1, shrink=0.5, pad=0.1)
    
    # local std
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, std, cmap='Reds', alpha=1.0)
    ax2.set_title(f'local standard deviation with {k}x{k} window')
    ax2.set_xlabel('image width')
    ax2.set_ylabel('image height')
    ax2.set_zlabel('local std')
    plt.colorbar(surf2, ax=ax2, shrink=0.5, pad=0.1)
    
    # local entropy
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, Y, entropy, cmap='Greens', alpha=1.0)
    ax3.set_title(f'local entropy with {k}x{k} window')
    ax3.set_xlabel('image width')
    ax3.set_ylabel('image height')
    ax3.set_zlabel('local entropy')
    plt.colorbar(surf3, ax=ax3, shrink=0.5, pad=0.1)
    
    plt.tight_layout()
    plt.savefig('local_statistics_3d.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    image_path = "grey_1.jpg"
    k = 5  
    
    # compute
    start_time = time.time()
    mean = compute_local_mean(k, image_path)
    print(f"mean: {time.time() - start_time:.2f}s")
    start_time = time.time()
    std = compute_local_std(k, image_path)
    print(f"std: {time.time() - start_time:.2f}s")
    start_time = time.time()
    entropy = compute_local_entropy(k, image_path)
    print(f"entropy: {time.time() - start_time:.2f}s")
    
    # plot
    plot_local_statistics(mean, std, entropy, k)