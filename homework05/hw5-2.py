import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def spectrum(image_path, plot=True):
    """
    Return:
        image_array: original image array 
        F: 2d fourier transform of the image
    """
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
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
    spectrum = np.log(np.abs(F) + 1)
    
    image_dir = "image"
    file_name = os.path.basename(image_path)
    # spectrum
    spectrum_norm = spectrum - spectrum.min()
    spectrum_norm = spectrum_norm / spectrum_norm.max()
    spectrum_img = (spectrum_norm * 255).astype(np.uint8)
    spectrum_image = Image.fromarray(spectrum_img)
    spectrum_image.save(os.path.join(image_dir, f"original_spectrum.png"))
    # comparison
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(8, 6))
        
        axes[0].imshow(image_array, cmap='gray')
        axes[0].set_title('original image')
        axes[0].axis('off')
        
        axes[1].imshow(spectrum, cmap='gray')
        axes[1].set_title('frequency spectrum (log magnitude)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, f"comparison_spectrum"), bbox_inches='tight', dpi=150)
        plt.close()

    return image_array, F


def select_notch_points(F):
    """
    interactively select notch points in the frequency domain
    Parameters:
        F: 2d fourier transform of the images
    Returns:
        points: list of (row, col) needed to notch
    """
    spectrum = np.log(np.abs(F) + 1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(spectrum, cmap='gray')
    ax.set_title("click peaks (press SPACE to finish)")
    ax.axis('off')

    pts = []
    finished = False

    def on_click(event):
        nonlocal finished
        if event.inaxes == ax and not finished and event.button == 1:
            x, y = event.xdata, event.ydata
            pts.append((x, y))
            ax.plot(x, y, 'r+')
            fig.canvas.draw()

    def on_key(event):
        nonlocal finished
        if event.key == ' ':
            finished = True
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=True)

    indices = [(int(round(y)), int(round(x))) for x, y in pts]
    print("selected notch points (row, col) in frequency domain: ")
    for u, v in indices:
        print(f"({u}, {v})")
    return indices

def notch_filter(F, notch_points, D0=10, n=2, plot=True):
    """
    optimal notch reject filter using Butterworth filter
    Parameters:
        F: 2d fourier transform (already centered), shape (P, Q) where P=2M, Q=2N
        notch_points: list of (row, col) selected points
        D0: cutoff frequency (radius) for notch filter
        n: order of Butterworth filter
    Returns:
        F_noise: frequency domain of noise (bandpass filtered)
        F_filtered: frequency domain after notch reject filtering
    """
    P, Q = F.shape
    u, v = np.arange(P), np.arange(Q)
    U, V = np.meshgrid(u, v, indexing='ij')
    
    # notch reject filter
    H_NR = np.ones((P, Q), dtype=np.float64)
    # for each selected point, create a notch pair
    for u_k, v_k in notch_points:
        D_k = np.sqrt((U - u_k) ** 2 + (V - v_k) ** 2)
        u_sym = P - u_k
        v_sym = Q - v_k
        D_k_sym = np.sqrt((U - u_sym) ** 2 + (V - v_sym) ** 2)
        
        epsilon = 1e-10
        D_k = np.where(D_k < epsilon, epsilon, D_k)
        D_k_sym = np.where(D_k_sym < epsilon, epsilon, D_k_sym)
        
        # Butterworth notch reject filter
        H_k = 1.0 / (1.0 + (D0 / D_k) ** n)
        H_k_sym = 1.0 / (1.0 + (D0 / D_k_sym) ** n)
        H_NR = H_NR * H_k * H_k_sym
    
    # notch pass filter = 1 - notch reject filter
    H_NP = 1.0 - H_NR
    F_noise = F * H_NP     
    F_filtered = F * H_NR   
    if plot:
        # plot
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        # original F (log)
        original_spectrum = np.log(np.abs(F) + 1)
        axes[0].imshow(original_spectrum, cmap='gray')
        axes[0].set_title('original F (log)')
        axes[0].axis('off')
        # noise frequency
        noise_spectrum = np.log(np.abs(F_noise) + 1)
        axes[1].imshow(noise_spectrum, cmap='gray')
        axes[1].set_title(f'noise frequency(0={D0}, n={n})')
        axes[1].axis('off')
        # filtered frequency
        filtered_spectrum = np.log(np.abs(F_filtered) + 1)
        axes[2].imshow(filtered_spectrum, cmap='gray')
        axes[2].set_title(f'filter frequency (D0={D0}, n={n})')
        axes[2].axis('off')
        
        plt.tight_layout()
        image_dir = "image"
        plt.savefig(os.path.join(image_dir, f"notch_spectrum.png"), bbox_inches='tight', dpi=150)
        plt.close()
    
    return F_noise, F_filtered


def frequency_to_spatial(F):
    """
    convert frequency domain array to spatial domain array
    implements steps 3-5 of frequency domain transformation:
    Step 3: Inverse transform
    Step 4: De-centering
    Step 5: Cropping
    Parameters:
        F: frequency domain array, shape (P, Q) where P=2M, Q=2N
    Returns:
        image_array: spatial domain array, shape (M, N)
    """
    P, Q = F.shape
    M, N = P // 2, Q // 2
    
    # Step 3: Inverse transform
    gp_prim = np.fft.ifft2(F)
    
    # Step 4: De-centering
    x, y = np.arange(P), np.arange(Q)
    X, Y = np.meshgrid(x, y, indexing='ij')
    center_matrix = (-1) ** (X + Y)
    gp = np.real(gp_prim * center_matrix)
    
    # Step 5: Cropping
    g = np.clip(gp[0: M, 0: N], 0, 255)
    image_array = g.astype(np.uint8)
    
    return image_array


def best_notch_filter(image_array, noise_array, k, save=True):
    """
    optimal notch filter using adaptive weighting function
    Parameters:
        image_array: original noisy image array (g)
        noise_array: noise array in spatial domain (n)
        k: window size (odd number)
    Returns:
        weight_array: weighting function w(x,y)
        best_notch_array: filtered image f̂(x,y) = g(x,y) - w(x,y)n(x,y)
    """
    image_float = image_array.astype(np.float64)
    noise_float = noise_array.astype(np.float64)
    height, width = image_float.shape
    weight_array = np.zeros_like(image_array, dtype=np.float32)
    best_notch_array = np.zeros_like(image_float, dtype=np.float64)
    
    half = k // 2
    
    # w(x,y) = \frac{\mu_{g\eta|S_{xy}} - \mu_{g|S_{xy}}\mu_{\eta|S_{xy}}}{\sigma_{\eta|S_{xy}}^{2}}
    # For each pixel, compute its window and calculate statistics directly
    for row in range(height):
        for col in range(width):
            # calculate window boundaries 
            top = max(0, row - half)
            bottom = min(height, row + half + 1)
            left = max(0, col - half)
            right = min(width, col + half + 1)
            
            # extract window
            window_g = image_float[top:bottom, left:right]
            window_n = noise_float[top:bottom, left:right]
            
            # calculate statistics for this window
            current_count = window_g.size
            sum_g = np.sum(window_g)
            sum_n = np.sum(window_n)
            sum_gn = np.sum(window_g * window_n)
            sum_n_sq = np.sum(window_n ** 2)

            mu_g = sum_g / current_count
            mu_n = sum_n / current_count
            mu_gn = sum_gn / current_count
            mu_n_sq = sum_n_sq / current_count
            sigma_n_sq = mu_n_sq - mu_n ** 2
            
            # compute weight
            if sigma_n_sq > 1e-10:
                w = (mu_gn - mu_g * mu_n) / sigma_n_sq
                w = max(0.0, w)
            else:
                w = 0.0
            
            weight_array[row, col] = w
            best_notch_array[row, col] = image_float[row, col] - w * noise_float[row, col]
    
    # clip to valid range
    best_notch_array = np.clip(best_notch_array, 0, 255).astype(np.uint8)
    if save:
        output_path = os.path.join("image", "best_notch.png")
        best_notch_image = Image.fromarray(best_notch_array)
        best_notch_image.save(output_path)

    return weight_array, best_notch_array

if __name__ == "__main__":

    image_path = os.path.join("image", "5-2.png")
    image_array, F = spectrum(image_path, plot=False)

    notch_points = select_notch_points(F)
    
    # notch in frequency domain
    D0 = 10   # cutoff frequency 
    n = 6    # Butterworth order 
    F_noise, F_filter = notch_filter(F, notch_points, D0, n, plot=False)

    # transform back into spatial domain
    simple_notch_array = frequency_to_spatial(F_filter)
    noise_array = frequency_to_spatial(F_noise)

    # plot frequency domain
    fig_freq, axes_freq = plt.subplots(2, 3, figsize=(8, 8))
    axes_freq[0, 0].imshow(image_array, cmap='gray')
    axes_freq[0, 0].set_title('original image')
    axes_freq[0, 0].axis('off')
    axes_freq[0, 1].imshow(noise_array, cmap='gray')
    axes_freq[0, 1].set_title('noise')
    axes_freq[0, 1].axis('off')
    axes_freq[0, 2].imshow(simple_notch_array, cmap='gray')
    axes_freq[0, 2].set_title('simple notch filtered')
    axes_freq[0, 2].axis('off')
    axes_freq[1, 0].imshow(np.log(np.abs(F) + 1), cmap='gray')
    axes_freq[1, 0].set_title('original F (log)')
    axes_freq[1, 0].axis('off')
    axes_freq[1, 1].imshow(np.log(np.abs(F_noise) + 1), cmap='gray')
    axes_freq[1, 1].set_title('noise frequency (log)')
    axes_freq[1, 1].axis('off')
    axes_freq[1, 2].imshow(np.log(np.abs(F_filter) + 1), cmap='gray')
    axes_freq[1, 2].set_title('simple notch filtered frequency (log)')
    axes_freq[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join("image", "frequency_domain.png"), bbox_inches='tight', dpi=150)
    plt.close(fig_freq)

    # best notch filter
    k = 11
    weight_array, best_notch_array = best_notch_filter(image_array, noise_array, k)
    
    # visualize weight array
    fig_weight, axes_weight = plt.subplots(1, 2, figsize=(12, 5))
    # weight array heatmap
    im = axes_weight[0].imshow(weight_array, cmap='hot', interpolation='nearest')
    axes_weight[0].set_title(f'Weight Array (k={k})')
    axes_weight[0].axis('off')
    plt.colorbar(im, ax=axes_weight[0], label='Weight Value')
    # weight histogram
    axes_weight[1].hist(weight_array.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes_weight[1].set_xlabel('Weight Value')
    axes_weight[1].set_ylabel('Frequency')
    axes_weight[1].set_title('Weight Distribution')
    axes_weight[1].grid(True, alpha=0.3)
    # add statistics text
    stats_text = f'Min: {weight_array.min():.2f}\n'
    stats_text += f'Max: {weight_array.max():.2f}\n'
    stats_text += f'Mean: {weight_array.mean():.2f}\n'
    stats_text += f'Median: {np.median(weight_array):.2f}\n'
    stats_text += f'0 <= Weights < 1: {np.sum((weight_array >= 0) & (weight_array < 1))}\n'
    stats_text += f'1 <= Weights < 2: {np.sum((weight_array >= 1) & (weight_array < 2))}\n'
    stats_text += f'Weights >= 2: {np.sum(weight_array >= 2)}'
    axes_weight[1].text(0.98, 0.98, stats_text, transform=axes_weight[1].transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                        fontsize=9, family='monospace')
    plt.tight_layout()
    plt.savefig(os.path.join("image", "weight.png"), bbox_inches='tight', dpi=150)
    plt.close(fig_weight)
    
    # comparison
    fig_compare, axes_compare = plt.subplots(1, 3, figsize=(8, 4))
    axes_compare[0].imshow(image_array, cmap='gray')
    axes_compare[0].set_title('original image')
    axes_compare[0].axis('off')
    axes_compare[1].imshow(simple_notch_array, cmap='gray')
    axes_compare[1].set_title('simple notch result')
    axes_compare[1].axis('off')
    axes_compare[2].imshow(best_notch_array, cmap='gray')
    axes_compare[2].set_title('best notch result')
    axes_compare[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join("image", "notch_comparison.png"), bbox_inches='tight', dpi=150)
    plt.close(fig_compare)
