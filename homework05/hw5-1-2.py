import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def gaussian_noise(image_path, k, plot=True):
    """
    Parameter:
        k: noise standard deviation is (k * sigma_original)
    Return:
        noisy_image: noise-added and clipped image array
        noise_array: noise array
    """
    image = Image.open(image_path).convert('L')
    image_array = np.array(image).astype(np.float32)
    M, N = image_array.shape
    
    # construct Gaussian noise in spatial domain, mean = 0 and std = (k * sigma)
    sigma = image_array.std()
    noise_std = k * sigma
    noise = np.random.normal(loc=0.0, scale=noise_std, size=(M, N)).astype(np.float32)
    # add noise in spatial domain and clip
    noisy_image = image_array + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    
    # noise array
    noise_array = noisy_image - image_array

    # empirical v.s. theoretical
    noise_mean = float(noise_array.mean())
    noise_std = float(noise_array.std())
    print(image_path)
    print(f"practical noise mean = {noise_mean:.2f}")
    print(f"practical noise std  = {noise_std:.2f}")
    print(f"k * sigma(original image) = {k * sigma:.2f}" + "\n")

    image_dir = "image"
    file_name = os.path.basename(image_path)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # original
        axes[0].imshow(image_array, cmap='gray')
        axes[0].set_title('original image')
        axes[0].axis('off')
        # add Gaussian noise
        axes[1].imshow(noisy_image, cmap='gray')
        axes[1].set_title(f'image with Gaussian noise (k={k}, std={noise_std:.2f})')
        axes[1].axis('off')
        # noise histogram
        axes[2].hist(noise_array.ravel(), bins=256)
        axes[2].set_title('histogram of noise')
        axes[2].set_xlim(-100, 100)

        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, f"gaussian_noise_{file_name}"), bbox_inches='tight', dpi=150)
        plt.close()

    return noisy_image, noise_array

def salt_pepper_noise(image_path, a, b, pa, pb, plot=True):
    """
    Parameter:
        a: intensity value for pepper noise
        b: intensity value for salt noise
        pa: probability of pepper noise
        pb: probability of salt noise
    Return:
        noisy_image: image with salt-and-pepper noise
        noise_array: noise array
    """
    image = Image.open(image_path).convert('L')
    image_array = np.array(image).astype(np.float32)
    M, N = image_array.shape

    # mask matrix
    rand = np.random.uniform(0, 1, (M, N))
    pepper_mask = rand < pa
    salt_mask = (rand >= pa) & (rand < pa + pb)
    keep_mask = rand >= (pa + pb)

    # multiplicative mask: keep original pixel
    mul_mask = keep_mask.astype(np.float32)
    # additive mask: replace original pixel with intensity a or b
    add_mask = np.zeros((M, N), dtype=np.float32)
    add_mask[pepper_mask] = a
    add_mask[salt_mask] = b

    # apply masks
    noisy_image = image_array * mul_mask + add_mask

    # noise array
    noise_array = noisy_image - image_array

    image_dir = "image"
    file_name = os.path.basename(image_path)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # original
        axes[0].imshow(image_array, cmap='gray')
        axes[0].set_title('original image')
        axes[0].axis('off')
        # salt and pepper noise image
        axes[1].imshow(noisy_image, cmap='gray')
        axes[1].set_title(f'salt and pepper noise (a={a}, pa={pa}, b={b}, pb={pb})')
        axes[1].axis('off')
        # noise histogram
        axes[2].hist(noise_array.ravel(), bins=256)
        axes[2].set_title('histogram of noise')
        axes[2].set_xlim(-15, 270)

        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, f"salt_pepper_{file_name}"), bbox_inches='tight', dpi=150)
        plt.close()

    return noisy_image, noise_array

if __name__ == "__main__":

    image_path1 = os.path.join("image", "brain.png")
    image_path2 = os.path.join("image", "heart.png")
    # gassian noise
    k = 0.5
    gaussian_noise(image_path1, k, plot=True)
    gaussian_noise(image_path2, k, plot=True)
    # salt and pepper noise
    a, b, pa, pb = 55, 200, 0.2, 0.2
    salt_pepper_noise(image_path1, a, b, pa, pb, plot=True)
    salt_pepper_noise(image_path2, a, b, pa, pb, plot=True)