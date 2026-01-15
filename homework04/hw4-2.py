import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def show_spectrum(image_path, plot=True):
    """
    Return:
        image_array: original image array 
        F: frequency domain 
    """
    image = Image.open(image_path)
    image_array = np.array(image)
    
    if len(image_array.shape) == 3:
        channel_array = image_array[:, :, 0] 
    else:
        channel_array = image_array
    M, N = channel_array.shape
    
    # Step 1: Padding
    P, Q = 2 * M, 2 * N
    fp = np.zeros((P, Q), dtype=np.float32)
    fp[0: M, 0: N] = channel_array.astype(np.float32)
    
    # Step 2: Centering and DFT
    x, y = np.arange(P), np.arange(Q)
    X, Y = np.meshgrid(x, y, indexing='ij')
    center_matrix = (-1) ** (X + Y)
    fp_centered = fp * center_matrix
    F = np.fft.fft2(fp_centered)
    spectrum = np.log(np.abs(F) + 1)
    
    image_dir = "image"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    file_name = os.path.basename(image_path)
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        
        axes[0].imshow(channel_array, cmap='gray')
        axes[0].set_title('original image')
        axes[0].axis('off')
        
        axes[1].imshow(spectrum, cmap='gray')
        axes[1].set_title('frequency spectrum (log magnitude)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, f"spectrum_{file_name}"), bbox_inches='tight', dpi=150)
        plt.close()
    
    return channel_array, F

def gaussian_highpass_filter_pnorm(channel_array, F, num, C0, W, p, save=True, plot=True):
    """
    Parameter:
        channel_array: original image array from show_spectrum
        F: frequency domain from show_spectrum
        num: number of filters
        C0: list of center frequency parameters for each filter
        W: list of bandwidth parameters for each filter
        p: list of p-norm parameters for distance calculation for each filter
    Return:
        filtered_array: filtered image array
    """
    M, N = channel_array.shape
    P, Q = 2 * M, 2 * N

    x, y = np.arange(P), np.arange(Q)
    X, Y = np.meshgrid(x, y, indexing='ij')
    center_matrix = (-1) ** (X + Y)
    
    original_spectrum = np.log(np.abs(F) + 1)
    
    # Step 3: Filtering with p-norm
    u, v = np.arange(P), np.arange(Q)
    U, V = np.meshgrid(u, v, indexing='ij')
    center_u, center_v = P / 2, Q / 2
    
    # Initialize combined filter H
    H_combined = np.ones((P, Q), dtype=np.float64)
    
    # Apply each filter and combine by multiplication
    for i in range(num):
        # p-norm distance: D = (|U-center_u|^p + |V-center_v|^p)^(1/p)
        U_diff = np.abs(U - center_u)
        V_diff = np.abs(V - center_v)
        D = (U_diff ** p[i] + V_diff ** p[i]) ** (1.0 / p[i])
        
        # Gaussian band-reject filter: H = 1 - exp(-[ (D^2 - C0^2) / (D*W) ]^2)
        epsilon = 1e-10
        D_safe = np.where(D < epsilon, epsilon, D)
        D_square = D ** 2
        ratio = (D_square - C0[i] ** 2) / (D_safe * W[i])
        H = 1.0 - np.exp(-(ratio ** 2))
        
        # Multiply filters together
        H_combined = H_combined * H
    
    G = F * H_combined
    filtered_spectrum = np.log(np.abs(G) + 1)
    
    # Step 4: Inverse transform and de-centering
    gp_prim = np.fft.ifft2(G)
    gp = np.real(gp_prim * center_matrix)
    
    # Step 5: Cropping
    g = np.clip(gp[0: M, 0: N], 0, 255)
    filtered_array = g.astype(channel_array.dtype)
    
    image_dir = "image"
    
    if save:
        output_name = f"filter_4-2.png"
        save_path = os.path.join(image_dir, output_name)
        output_image = Image.fromarray(filtered_array)
        output_image.save(save_path)
    
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        
        axes[0, 0].imshow(channel_array, cmap='gray')
        axes[0, 0].set_title('original image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(filtered_array, cmap='gray')
        axes[0, 1].set_title(f'filtered image')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(original_spectrum, cmap='gray')
        axes[1, 0].set_title('original spectrum (log magnitude)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(filtered_spectrum, cmap='gray')
        axes[1, 1].set_title('filtered spectrum (log magnitude)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, f"comparison_filter_4-2.png"), bbox_inches='tight', dpi=150)
        plt.close()
    
    return filtered_array


if __name__ == "__main__":

    num = 4
    C0 = [245, 400, 470, 1080]
    W = [200, 100, 100, 300]
    p = [1.0, 1.0, 1.0, 0.50]
    image_path = os.path.join("image", "4-2.png")

    channel_array, F = show_spectrum(image_path, plot=True)
    filtered_array = gaussian_highpass_filter_pnorm(channel_array, F, num, C0, W, p, save=True, plot=True)