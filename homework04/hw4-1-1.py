import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def frequency_filter_smooth_one_channel(d0, channel_array):
    """
    Parameter:
        d0: cutoff frequency for Gaussian lowpass filter
        channel_array: single channel array
    Return:
        smooth_channel_array: smoothed channel array
        original_spectrum: log-magnitude of original frequency domain
        filtered_spectrum: log-magnitude of filtered frequency domain
    """
    channel = channel_array.astype(np.float32)
    M, N = channel.shape

    # Step 1: Padding
    P, Q = 2 * M, 2 * N
    fp = np.zeros((P, Q), dtype=np.float32)
    fp[0:M, 0:N] = channel

    # Step 2: Centering and DFT
    x, y = np.arange(P), np.arange(Q)
    X, Y = np.meshgrid(x, y, indexing='ij')
    center_matrix = (-1) ** (X + Y)
    fp_centered = fp * center_matrix
    F = np.fft.fft2(fp_centered)
    original_spectrum = np.log(np.abs(F) + 1)

    # Step 3: Filtering
    u, v = np.arange(P), np.arange(Q)
    U, V = np.meshgrid(u, v, indexing='ij')
    center_u, center_v = P / 2, Q / 2
    distance_square = (U - center_u) ** 2 + (V - center_v) ** 2
    H = np.exp(- distance_square / (2 * d0 ** 2))
    G = F * H
    filtered_spectrum = np.log(np.abs(G) + 1)

    # Step 4: Inverse transform and de-centering
    gp_prim = np.fft.ifft2(G)
    gp = np.real(gp_prim * center_matrix)

    # Step 5: Cropping
    g = np.clip(gp[0: M, 0: N], 0, 255)
    smooth_channel_array = g.astype(channel_array.dtype)

    return smooth_channel_array, original_spectrum, filtered_spectrum

def frequency_filter_smooth(image_path, d0, save=True, plot=True):
    """
    Parameter:
        image_path: path of image to smooth
        d0: cutoff frequency for Gaussian lowpass filter
        save: save smoothed image and spectrum comparison
        plot: plot comparison figure
    Return:
        smooth_image_array: smoothed image array
    """
    image = Image.open(image_path)
    image_array = np.array(image)

    if image_array.ndim == 2:
        smooth_image_array, original_spectrum, filtered_spectrum = frequency_filter_smooth_one_channel(d0, image_array)
    else:
        smooth_image_array = np.zeros_like(image_array)
        for channel in range(image_array.shape[2]):
            smooth_image_array[:, :, channel], _, _ = frequency_filter_smooth_one_channel(d0, image_array[:, :, channel])
            if channel == 0:
                _, original_spectrum, filtered_spectrum = frequency_filter_smooth_one_channel(d0, image_array[:, :, channel])

    image_dir = "image"
    file_name = os.path.basename(image_path)

    if save:
        output_name = f"smooth_{d0}_{file_name}"
        save_path = os.path.join(image_dir, output_name)
        output_image = Image.fromarray(smooth_image_array.astype(np.uint8))
        output_image.save(save_path)

    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))

        axes[0, 0].imshow(image_array, cmap='gray' if image_array.ndim == 2 else None)
        axes[0, 0].set_title('original image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(smooth_image_array.astype(np.uint8), cmap='gray' if image_array.ndim == 2 else None)
        axes[0, 1].set_title(f'smoothed image (d0={d0})')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(original_spectrum, cmap='gray')
        axes[1, 0].set_title('original spectrum (log magnitude)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(filtered_spectrum, cmap='gray')
        axes[1, 1].set_title('filtered spectrum (log magnitude)')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, f"comparison_smooth_{d0}_{file_name}"), bbox_inches='tight', dpi=150)
        plt.close()

    return smooth_image_array

if __name__ == "__main__":

    d0 = 100
    image_path = os.path.join("image", "4-1.png")
    frequency_filter_smooth(image_path, d0, save=True, plot=True)
