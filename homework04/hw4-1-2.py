import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def convert_to_gray(image_array):

    if len(image_array.shape) == 2:
        return image_array
    elif len(image_array.shape) == 3:
        if image_array.shape[2] == 3:
            gray_array = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
        elif image_array.shape[2] == 4:
            gray_array = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
        else:
            gray_array = image_array[:, :, 0]
        return gray_array.astype(image_array.dtype)

def frequency_filter_sharpen(image_path, k, d0, save = True, plot = True):
    r"""
    Parameter:
        k: Laplace - $g(x, y) = \mathcal{F}^{-1}\{[1 + kH_{Lap}(u, v)]F(u, v)\}$
           High Boosting - $g(x, y) = \mathcal{F}^{-1}\{[1 + kH_{HP}(u, v)]F(u, v)\}$
        d0: cutoff frequency for Gaussian lowpass filter (1 - Gaussian lowpass filter = high pass filter) 
        image_path:
    Return:
        sharpened_laplace_array: use frequency kernal $H_{Lap}(u,v) = -4\pi^{2}(u^{2}+v^{2})$         
        sharpened_highboosting_array: use frequency kernal $1 + k(1 - H_{LP}(u, v))$, $H_{LP}(u, v)$ is Gaussian kernal
    """
    image = Image.open(image_path)
    image_array = np.array(image)
    image_array = convert_to_gray(image_array)
    M, N = image_array.shape

    # Step 1: Padding
    P, Q = 2 * M, 2 * N
    fp = np.zeros((P, Q), dtype=np.float32)
    fp[0: M, 0: N] = image_array.astype(np.float32)

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
    
    # laplace kernal: normalize frequency coordinates
    U_norm = (U - center_u) / P
    V_norm = (V - center_v) / Q
    distance_square_norm = U_norm ** 2 + V_norm ** 2
    H_laplace = -4 * np.pi ** 2 * distance_square_norm
    G_laplace = F * (1 - k * H_laplace)
    laplace_spectrum = np.log(np.abs(G_laplace) + 1)

    # high boosting
    distance_square = (U - center_u) ** 2 + (V - center_v) ** 2
    H_highboosting = 1 - np.exp(- distance_square / (2 * d0 ** 2))
    G_highboosting = F * (1 + k * H_highboosting)
    highboosting_spectrum = np.log(np.abs(G_highboosting) + 1)

    # Step 4 method 1: Inverse transform and de-centering for Laplace
    original_gp_laplace = np.fft.ifft2(G_laplace)
    gp_laplace = np.real(original_gp_laplace * center_matrix)

    # Step 4 method 2: Inverse transform and de-centering for High Boosting
    original_gp_highboosting = np.fft.ifft2(G_highboosting)
    gp_highboosting = np.real(original_gp_highboosting * center_matrix)

    # Step 5: Cropping
    g_laplace = np.clip(gp_laplace[0: M, 0: N], 0, 255)
    sharpen_laplace_array = g_laplace.astype(image_array.dtype)
    
    g_highboosting = np.clip(gp_highboosting[0: M, 0: N], 0, 255)
    sharpen_highboosting_array = g_highboosting.astype(image_array.dtype)

    image_dir = "image"
    file_name = os.path.basename(image_path)

    if save:
        output_name_laplace = f"sharpen_laplace_{k}_{file_name}"
        save_path_laplace = os.path.join(image_dir, output_name_laplace)
        output_image_laplace = Image.fromarray(sharpen_laplace_array)
        output_image_laplace.save(save_path_laplace)
        
        output_name_highboosting = f"sharpen_highboosting_{d0}_{k}_{file_name}"
        save_path_highboosting = os.path.join(image_dir, output_name_highboosting)
        output_image_highboosting = Image.fromarray(sharpen_highboosting_array)
        output_image_highboosting.save(save_path_highboosting)

    if plot:

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        axes[0, 0].imshow(image_array, cmap='gray')
        axes[0, 0].set_title('original image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(sharpen_laplace_array, cmap='gray')
        axes[0, 1].set_title(f'sharpened image (laplace kernal, k={k})')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(sharpen_highboosting_array, cmap='gray')
        axes[0, 2].set_title(f'sharpened image (high boosting, d0={d0}, k={k})')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(original_spectrum, cmap='gray')
        axes[1, 0].set_title('original spectrum (log magnitude)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(laplace_spectrum, cmap='gray')
        axes[1, 1].set_title('laplace spectrum (log magnitude)')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(highboosting_spectrum, cmap='gray')
        axes[1, 2].set_title('high boosting spectrum (log magnitude)')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, f"comparison_sharpen_{k}_{file_name}"), bbox_inches='tight', dpi=150)
        plt.close()

    return sharpen_laplace_array, sharpen_highboosting_array


if __name__ == "__main__":

    k = 3
    d0 = 100
    image_path = os.path.join("image", "4-1.png")
    frequency_filter_sharpen(image_path, k, d0, save=False, plot=True)