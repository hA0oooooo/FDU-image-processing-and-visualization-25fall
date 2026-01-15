import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


def white_noise(image_path, k, plot=True):
    """
    Parameter:
        k: the magnitude of white noise is (k * sigma * sqrt(MN))
    Return:
        noisy_image_clipped: noise-added and clipped image array
        noise_array: noise array
    """
    image = Image.open(image_path).convert('L')
    image_array = np.array(image).astype(np.float32)
    M, N = image_array.shape

    F = np.fft.fft2(image_array)
    # construct white noise: constant magnitude + uniform phase
    phase = 2 * np.pi * np.random.uniform(0, 1, (M, N))
    sigma = image_array.std()
    noise_magnitude = k * sigma * np.sqrt(M * N)
    noise_F = noise_magnitude * (np.cos(phase) + 1j * np.sin(phase))
    # add frequency domain
    F_noisy = F + noise_F
    noisy_image_complex = np.fft.ifft2(F_noisy)
    noisy_image = np.real(noisy_image_complex)

    # noise-added image array, clip
    noisy_image_clipped = np.clip(noisy_image, 0, 255).astype(np.uint8)
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
        # add white noise
        axes[1].imshow(noisy_image_clipped, cmap='gray')
        axes[1].set_title(f'image with white noise (k={k}, magnitude={noise_magnitude:.0f})')
        axes[1].axis('off')
        # noise histogram
        axes[2].hist(noise_array.ravel(), bins=256)
        axes[2].set_title('histogram of noise')

        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, f"white_noise_{file_name}"), bbox_inches='tight', dpi=150)
        plt.close()

    return noisy_image_clipped, noise_array


if __name__ == "__main__":

    image_path1 = os.path.join("image", "brain.png")
    image_path2 = os.path.join("image", "heart.png")
    # white noise
    k = 0.5
    white_noise(image_path1, k, plot=True)
    white_noise(image_path2, k, plot=True)