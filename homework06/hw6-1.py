from tkinter import N
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# get image with salt and pepper noise
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
    withnoise_array = image_array * mul_mask + add_mask
    
    # save
    image_dir = "image"
    file_name = os.path.basename(image_path)
    withnoise_array = withnoise_array.astype(np.uint8)
    withnoise_image = Image.fromarray(withnoise_array)
    save_path = os.path.join(image_dir, f"saltPepper_{file_name}")
    withnoise_image.save(save_path)

    return withnoise_array

# global Otsu threshold algorithm
def otsu_threshold(image_path, L=256, save_result=True):

    image = Image.open(image_path).convert('L')
    array = np.array(image)
    height, width = array.shape
    count = height * width

    hist = np.zeros(L, dtype = np.int32)
    for i in range(height):
        for j in range(width):
            hist[array[i, j]] += 1
    p = hist.astype(np.float64) / count
    
    P1 = np.zeros(L) 
    m1 = np.zeros(L)
    P1[0] = p[0]
    m1[0] = 0
    for k in range(1, L):
        P1[k] = P1[k-1] + p[k]
        m1[k] = m1[k-1] + k * p[k]
    mG = np.sum(np.arange(L) * p)
    
    # compute between-class variance
    max_variance = -1
    best_threshold = 0
    for k in range(L):
        if P1[k] > 0 and P1[k] < 1:
            variance = (mG * P1[k] - m1[k])**2 / (P1[k] * (1 - P1[k]))
            if variance > max_variance:
                max_variance = variance
                best_threshold = k
    
    threshold = np.zeros_like(array)
    threshold[array > best_threshold] = L - 1
    
    # save processed image
    if save_result:

        image_dir = "image"
        file_name = os.path.basename(image_path)
        threshold_img = Image.fromarray(threshold)
        save_path = os.path.join(image_dir, f"Otsu_{file_name}")
        threshold_img.save(save_path)       

    return threshold

# kmeans algorithm
def kmeans_seg(k, image_path, save_result=True):

    image = Image.open(image_path).convert('L')
    array = np.array(image)
    height, width = array.shape
    
    mean = np.random.uniform(np.min(array) + 1, np.max(array) -  1, k)
    mask = np.zeros((height, width))

    epsilon = 1
    maxIteration = 100
    iteration = 0

    while iteration < maxIteration:
        sum, count = np.zeros(k), np.zeros(k)
        for i in range(height):
            for j in range(width):
                value = array[i, j]
                distance = np.abs(mean - value)
                index = np.argmin(distance)
                mask[i, j] = index
                sum[index] += value
                count[index] += 1
        
        newMean = np.zeros(k)
        for index in range(k):
            if count[index] > 0:
                newMean[index] = sum[index] / count[index]
            else:
                newMean[index] = mean[index]
                
        if np.linalg.norm(newMean - mean) < epsilon:
            print(f"iteration: {iteration}")
            break

        mean = newMean
        iteration += 1
        if iteration == maxIteration:
            print(f"iteration: {iteration}")
    
    segarray = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            segarray[i, j] = mean[int(mask[i, j])]

    segarray = segarray.astype(np.uint8)

    # save processed image
    if save_result:

        image_dir = "image"
        file_name = os.path.basename(image_path)
        segarray_img = Image.fromarray(segarray)
        save_path = os.path.join(image_dir, f"kmeans{k}_{file_name}")
        segarray_img.save(save_path) 

    return segarray

# GMM algorithm
def gmm_seg(k, image_path, save_result=True):

    image = Image.open(image_path).convert('L')
    array = np.array(image)
    height, width = array.shape
    N = height * width
    X = array.flatten().astype(np.float64)

    # initialize
    pi = np.ones(k) / k
    mu = np.random.uniform(np.min(array) + 1, np.max(array) - 1, k)
    sigma2 = np.ones(k) * 25
    P = np.zeros((N, k))

    epsilon = 1
    maxIteration = 100
    iteration = 0

    while iteration < maxIteration:
        # E-step: update P(zi | xi, theta)
        for i in range(N):
            numerator = np.zeros(k)
            for j in range(k):
                numerator[j] = pi[j] * (1.0 / np.sqrt(2 * np.pi * sigma2[j])) * \
                              np.exp(-0.5 * (X[i] - mu[j]) ** 2 / sigma2[j])
            denominator = np.sum(numerator)
            P[i, :] = numerator / denominator

        # M-step: update theta
        newMu = np.zeros(k)
        newSigma2 = np.zeros(k)
        newPi = np.zeros(k)

        for j in range(k):
            sumProb = np.sum(P[:, j])
            newMu[j] = np.sum(P[:, j] * X) / sumProb
            newSigma2[j] = np.sum(P[:, j] * (X - newMu[j]) ** 2) / sumProb
            newSigma2[j] = max(newSigma2[j], 1e-2) 
            newPi[j] = sumProb / N

        # check if converges
        muDiff = np.linalg.norm(newMu - mu)
        sigmaDiff = np.linalg.norm(newSigma2 - sigma2)
        piDiff = np.linalg.norm(newPi - pi)
        if muDiff < epsilon and sigmaDiff < epsilon and piDiff < epsilon:
            print(f"iteration: {iteration}")
            break
        
        mu = newMu
        sigma2 = newSigma2
        pi = newPi
        iteration += 1
        if iteration == maxIteration:
            print(f"iteration: {iteration}")

    # segragate
    mask = np.argmax(P, axis = 1).reshape(height, width)
    segarray = np.zeros((height, width), dtype = np.uint8)
    for i in range(height):
        for j in range(width):
            segarray[i, j] = mu[mask[i, j]]

    # save processed image
    if save_result:

        image_dir = "image"
        file_name = os.path.basename(image_path)
        segarray_img = Image.fromarray(segarray)
        save_path = os.path.join(image_dir, f"gmm{k}_{file_name}")
        segarray_img.save(save_path) 

    return segarray


if __name__ == "__main__":

    # get image with salt and pepper noise
    image_dir = "image"
    image_path = os.path.join(image_dir, "heart.png")
    salt_pepper_noise(image_path, a = 0, b = 255, pa = 0.0005, pb = 0.0005)

    withnoise_image_path = os.path.join(image_dir, "saltPepper_heart.png")
    withnoise_image = Image.open(withnoise_image_path)
    withnoise_array = np.array(withnoise_image)

    # [HW6-1](1)
    # global OSTU threshold
    threshold = otsu_threshold(withnoise_image_path, L = 256, save_result=True)

    # kmeans and GMM segregation
    kmeans2 = kmeans_seg(2, withnoise_image_path, save_result=True)
    print("kmeans: k=2 finished")
    gmm2 = gmm_seg(2, withnoise_image_path, save_result=True)
    print("gmm: k=2 finished")

    # comparison: original, otsu, kmeans, gmm    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(withnoise_array, cmap='gray')
    axes[0, 0].set_title('original image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(threshold, cmap='gray')
    axes[0, 1].set_title('Otsu threshold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(kmeans2, cmap='gray')
    axes[1, 0].set_title('K-means segmentation')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(gmm2, cmap='gray')
    axes[1, 1].set_title('GMM segmentation')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, f"comparison_6-1-1.png"), bbox_inches='tight', dpi=150)
    plt.close()

    # [HW6-1](2)
    kmeans3 = kmeans_seg(3, withnoise_image_path, save_result=True)
    print("kmeans: k=3 finished")
    kmeans4 = kmeans_seg(4, withnoise_image_path, save_result=True)
    print("kmeans: k=4 finished")
    kmeans5 = kmeans_seg(5, withnoise_image_path, save_result=True)
    print("kmeans: k=5 finished")

    # comparison: original, kmeans with class 3, 4, 5
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(withnoise_array, cmap='gray')
    axes[0, 0].set_title('original image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(kmeans3, cmap='gray')
    axes[0, 1].set_title('Kmeans (k=3)')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(kmeans4, cmap='gray')
    axes[1, 0].set_title('Kmeans (k=4)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(kmeans5, cmap='gray')
    axes[1, 1].set_title('Kmeans (k=5)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, f"comparison_6-1-2.png"), bbox_inches='tight', dpi=150)
    plt.close()