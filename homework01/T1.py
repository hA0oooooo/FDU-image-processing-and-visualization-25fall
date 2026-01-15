import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

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

def piecewise_linear_function(r, r1, s1, r2, s2, L=256):
    """
    Parameters:
        r: input intensity value
        r1: input of the first breakpoint
        s1: output of the first breakpoint
        r2: input of the second breakpoint
        s2: output of the second breakpoint
        L: total number of intensity levels
    Returns:
        s: output intensity value
    """
    if r < r1:
        s = (s1 / r1) * r
    elif r < r2:
        s = s1 + ((s2 - s1) / (r2 - r1)) * (r - r1)
    else:
        s = s2 + ((L - 1 - s2) / (L - 1 - r2)) * (r - r2)
    return s

def lookup_table(r1, s1, r2, s2, L=256):

    table = np.zeros(256, dtype=np.uint8)
    for r in range(L):
        s = piecewise_linear_function(r, r1, s1, r2, s2, L)
        table[r] = np.clip(s, 0, L-1)

    return table

def piecewise_linear_transformation(file_path, r1, s1, r2, s2, L=256):

    img = Image.open(file_path)
    if img.mode == 'RGBA': img = img.convert('RGB')    
    img_array = np.array(img)

    table = lookup_table(r1, s1, r2, s2, L)
    transformed_array = table[img_array]
    output_img = Image.fromarray(transformed_array)
    
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    output_path = "transformed_" + name + ext
    output_img.save(output_path)

    return output_path

def show_comparison(original_path, transformed_path, output_name):

    original = Image.open(original_path)
    transformed = Image.open(transformed_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    if original.mode == 'L':
        axes[0].imshow(original, cmap='gray')
    else:
        axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    if transformed.mode == 'L':
        axes[1].imshow(transformed, cmap='gray')
    else:
        axes[1].imshow(transformed)
    axes[1].set_title('Transformed Image')
    axes[1].axis('off')
    plt.tight_layout()
    
    comparison_filename = output_name
    plt.savefig(comparison_filename, bbox_inches='tight')
    plt.close()
    
    return None

if __name__ == "__main__":

    input_file = "1.jpg"
    # input: 3/8L, output: 1/8L
    r1, s1 = 96, 32  
    # input: 5/8L, output: 7/8L
    r2, s2 = 160, 224 
    
    grey_input = save_grey_image(input_file)
    output_file = piecewise_linear_transformation(input_file, r1, s1, r2, s2)
    grey_output = save_grey_image(output_file)
    
    show_comparison(input_file, output_file, "comparison.png")
    show_comparison(grey_input, grey_output, "comparison_grey.png")
