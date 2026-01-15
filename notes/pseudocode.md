### 1. 灰度变换 (Gray Level Transformation)

**输入**: 原图像 `old_image`，变换函数 `T`  
**输出**: 新图像 `new_image`

```python
# T 是 [0, L-1] 到 [0, L-1] 的映射函数
# L 是灰度级的数量（例如 L=256）
for i = 1 to height:
    for j = 1 to width:
        new_image[i, j] = T(old_image[i, j])  
```


### 2. n维联合直方图 (nD-joint Histogram)

**输入**: n个图像 `images[0..n-1]`，每个图像的灰度级数 `L`  
**输出**: n维数组 `histogram`（初始化为全0）

```python
# 初始化n维直方图数组，大小为 L×L×...×L（n个L）
histogram = zeros(L, L, ..., L) 
for i = 1 to height:
	for j = 1 to width:
		histogram[images[0][i, j], images[1][i, j], ..., images[n-1][i, j]] += 1
```


### 3. 直方图均衡化 (Histogram Equalization)

**输入**: 原图像 `image`，灰度级数 `L`  
**输出**: 均衡化后的图像 `equalized_image`

```python
# 步骤1: 计算原始直方图
hist = [0] * L  
for i = 1 to height:
    for j = 1 to width:
        hist[image[i, j]] += 1

# 步骤2: 计算累积分布函数 (CDF)
total_pixels = height * width
cdf = [0] * L
cdf[0] = hist[0]
for k = 1 to L-1:
    cdf[k] = cdf[k-1] + hist[k]
cdf = cdf / total_pixels

# 步骤3: 使用CDF进行变换
# S = T(R) = round((L-1) × CDF(R))
for i = 1 to height:
    for j = 1 to width:
        equalized_image[i, j] = round((L - 1) * cdf[image[i, j]])  
```


### 4. 局部统计量计算 (Local Statistics)

**输入**: 原图像 `image`，窗口大小 `k`  
**输出**: 局部均值 `local_mean`，局部标准差 `local_std`，局部熵 `local_entropy`

```python
window = get_window(image[0, 0])
local_statistics[0, 0] = calculate_local_statistics(window)

for i = 1 to height:
    update_window_vertical()  # 行方向到终点后向下移动
    if i % 2 == 0:  # 偶数行：从左到右
        for j = 1 to width:
            update_window_horizontal()
            local_statistics[i, j] = calculate_local_statistics(window)
    else:  # 奇数行：从右到左
        for j = width-1 to 1:
            update_window_horizontal()
            local_statistics[i, j] = calculate_local_statistics(window)
```


### 5. 局部直方图均衡化 (Local Histogram Equalization)

**输入**: 原图像 `image`，窗口大小 `k`，灰度级数 `L`  
**输出**: 均衡化后的图像 `local_equalized_image`

```python
for i = 1 to height:
    for j = 1 to width:
        local_window = window(i, j)
        # 步骤1: 计算局部直方图
        local_hist = histogram(local_window)
        # 步骤2: 计算局部CDF
        local_cdf = [0] * L
        local_cdf[0] = local_hist[0]
        for k = 1 to L-1:
            local_cdf[k] = local_cdf[k-1] + local_hist[k]
        window_count = count(local_window)
        local_cdf = local_cdf / window_count
        # 步骤3: 使用局部CDF对中心像素进行变换
        # S = round((L-1) × CDF(R))
        center_value = image[i, j]
        local_equalized_image[i, j] = round((L-1) * local_cdf[center_value])
```


### 6. 基本全局阈值处理 (Basic Global Thresholding, BGT)

**输入**: 原图像 `image`  
**输出**: 阈值 `threshold`

```python
# 初始化两个类中心
mu1 = initial_value_1
mu2 = initial_value_2

while not converged:
    # 步骤1: 根据与类中心的距离分配类别
    for i = 1 to height:
        for j = 1 to width:
            if abs(image[i, j] - mu1) < abs(image[i, j] - mu2):
                flag[i, j] = 1
            else:
                flag[i, j] = 2
    
    # 步骤2: 更新类中心
    new_mu1 = mean(image[flag[i, j] == 1])
    new_mu2 = mean(image[flag[i, j] == 2])
    
    # 退出循环条件：类中心不再变化
    if abs(new_mu1 - mu1) < epsilon and abs(new_mu2 - mu2) < epsilon:
        converged = true
    mu1 = new_mu1
    mu2 = new_mu2

threshold = (mu1 + mu2) / 2
```


### 7. Otsu算法 (最大类间方差法)

**输入**: 直方图 `hist`
**输出**: 最佳阈值 `threshold`

```python
# 步骤1: 计算概率分布
P = [0] * L
for i = 0 to L-1:
    P[i] = hist[i] / total_pixels
# 步骤2: 计算全局均值
mu_glb = sum(i * P[i] for i = 0 to L-1)
# 步骤3: 遍历所有阈值，寻找使类间方差最大的阈值
max_variance = 0
threshold = 0
w1 = 0
mu1_sum = 0

for i = 0 to L-1:
    w1 += P[i]
    w2 = 1 - w1
    mu1_sum += P[i] * i
    mu1 = mu1_sum / w1
    mu2 = (mu_glb - w1 * mu1) / w2
    # 类间方差: σ²_bet = w1(μ1 - μ_glb)² + w2(μ2 - μ_glb)²
    variance = w1 * (mu1 - mu_glb)^2 + w2 * (mu2 - mu_glb)^2
    if variance > max_variance:
        max_variance = variance
        threshold = i
```

### 8. 熵方法阈值 (Entropy Method Threshold)

**输入**: 直方图 `hist`  
**输出**: 最佳阈值 `threshold`

```python
# 步骤1: 计算概率分布
P = [0] * L
for i = 0 to L-1:
    P[i] = hist[i] / total_pixels

max_entropy = 0
threshold = 0

for k = 0 to L-1:
    P1 = sum(P[0:k+1])
    P2 = 1 - P1
    if P1 == 0 or P2 == 0:
        continue
    
    # H1 = -sum(i=0 to k) (p_i/P1) * log2(p_i/P1)
    H1 = 0
    for i = 0 to k:
        if P[i] > 0:
            H1 -= (P[i] / P1) * log2(P[i] / P1)
    
    # H2 = -sum(i=k+1 to L-1) (p_i/P2) * log2(p_i/P2)
    H2 = 0
    for i = k+1 to L-1:
        if P[i] > 0:
            H2 -= (P[i] / P2) * log2(P[i] / P2)
    
    H = H1 + H2
    if H > max_entropy:
        max_entropy = H
        threshold = k
```


### 9. 局部自适应阈值 (Local Adaptive Threshold)

**输入**: 原图像 `image`，窗口大小 `k`，参数 `k_factor`  
**输出**: 二值化图像 `binary_image`

```python
for i = 1 to height:
    for j = 1 to width:
        local_window = window(i, j)
        # 计算局部均值和局部方差
        local_mean = mean(local_window)
        local_variance = variance(local_window)
        # 阈值 = 局部均值 + k × 局部方差
        threshold = local_mean + k_factor * local_variance
        # 判断当前像素点
        if image[i, j] >= threshold:
            binary_image[i, j] = 1
        else:
            binary_image[i, j] = 0
```


### 10. 双线性插值 (Bilinear Interpolation)

**输入**: 原图像 `image`，缩放因子 `N`  
**输出**: 放大后的图像 `scaled_image`

```python
height, width = image.shape
new_height = N * height
new_width = N * width

for i = 0 to new_height-1:
    for j = 0 to new_width-1:
        # 将目标像素位置映射回原图坐标（浮点数）
        x = j / N
        y = i / N
        # 分离整数部分和小数部分
        x1 = floor(x)
        y1 = floor(y)
        dx = x - x1  # dx ∈ [0, 1)
        dy = y - y1  # dy ∈ [0, 1)
        # 双线性插值公式：使用4个最近邻像素点
        # v(x,y) = (1-dx)(1-dy)I(x1,y1) + dx(1-dy)I(x2,y1) + (1-dx)dyI(x1,y2) + dx*dy*I(x2,y2)
        scaled_image[i, j] = image[y1, x1] * (1-dx) * (1-dy) + 
                             image[y1, x2] * dx * (1-dy) + 
                             image[y2, x1] * (1-dx) * dy + 
                             image[y2, x2] * dx * dy
```


### 11. 双三次插值 (Bicubic Interpolation, 使用三次样条)

**输入**: 原图像 `image`，目标位置坐标 `(x, y)`（浮点数）  
**输出**: 插值后的像素值 `value`

```python
# 分离整数部分和小数部分
i = floor(x)
j = floor(y)
u = x - i  # u ∈ [0, 1)
v = y - j  # v ∈ [0, 1)

# 三次样条权重函数
def cubic_spline(u, index):
    if index == -1:
        return (1-u)^3 / 6
    elif index == 0:
        return (3*u^3 - 6*u^2 + 4) / 6
    elif index == 1:
        return (-3*u^3 + 3*u^2 + 3*u + 1) / 6
    elif index == 2:
        return u^3 / 6

# 使用周围16个点（4×4网格）进行双三次插值
value = 0
for di = -1 to 2:
    for dj = -1 to 2:
        # 获取周围像素点坐标（边界处理）
        px = clip(i + di, 0, width - 1)
        py = clip(j + dj, 0, height - 1)
        # 计算权重：三次样条在x和y方向的权重乘积
        weight = cubic_spline(u, di) * cubic_spline(v, dj)
        value += image[py, px] * weight
```


### 12. 高斯平滑 (Gaussian Smoothing)

**输入**: 原图像 `image`，核半径 `N`，标准差 `sigma`  
**输出**: 平滑后的图像 `smooth_image`

```python
# 步骤1: 构建高斯核 (2N+1) × (2N+1)
kernel_size = 2 * N + 1
kernel = zeros(kernel_size, kernel_size)

for i = 0 to kernel_size-1:
    for j = 0 to kernel_size-1:
        x = i - N
        y = j - N
        kernel[i, j] = exp(-(x^2 + y^2) / (2 * sigma^2))

# 归一化核
kernel = kernel / sum(kernel)
# 步骤2: 边界填充（edge模式：复制边缘像素）
padded_image = pad(image, N, mode='edge')
# 步骤3: 卷积操作
smooth_image = zeros(height, width)
for i = 0 to kernel_size-1:
    for j = 0 to kernel_size-1:
        smooth_image += kernel[i, j] * padded_image[i:i+height, j:j+width]
```


### 13. 拉普拉斯锐化 (Laplacian Sharpening)

**输入**: 原图像 `image`，参数 `c`  
**输出**: 锐化后的图像 `sharpened_image`

```python
# 步骤1: 定义拉普拉斯核（8邻域，中心为-8）
laplacian_kernel = [[1, 1, 1],
                     [1, -8, 1],
                     [1, 1, 1]]
# 步骤2: 边界填充
pad_width = 1
padded_image = pad(image, pad_width, mode='edge')
# 步骤3: 计算拉普拉斯算子
laplacian = zeros(height, width)
for i = 0 to 2:
    for j = 0 to 2:
        laplacian += laplacian_kernel[i, j] * padded_image[i:i+height, j:j+width]
# 步骤4: 锐化公式 g = f - c * ∇²f
sharpened_image = image - c * laplacian
sharpened_image = clip(sharpened_image, 0, 255)
```


### 14. 反锐化掩膜 (Unsharp Masking)

**输入**: 原图像 `image`，核半径 `N`，标准差 `sigma`  
**输出**: 锐化后的图像 `sharpened_image`

```python
# 步骤1: 平滑原图像（使用高斯平滑）
blurred_image = gaussian_smooth(image, N, sigma)
# 步骤2: 计算掩膜（边缘信息）g_mask = f - f_blur
mask = image - blurred_image
# 步骤3: 将掩膜加回原图 g = f + g_mask
sharpened_image = image + mask
sharpened_image = clip(sharpened_image, 0, 255)
```


### 15. 高提升滤波 (High-boosting Filtering)

**输入**: 原图像 `image`，核半径 `N`，标准差 `sigma`，参数 `k`  
**输出**: 锐化后的图像 `sharpened_image`

```python
# 步骤1: 平滑原图像（使用高斯平滑）
blurred_image = gaussian_smooth(image, N, sigma)
# 步骤2: 计算掩膜（边缘信息）g_mask = f - f_blur
mask = image - blurred_image
# 步骤3: 加权增强 g = f + k * g_mask
sharpened_image = image + k * mask
sharpened_image = clip(sharpened_image, 0, 255)
```


### 16. 频域滤波通用算法 (Frequency Domain Filtering - General Algorithm)

**输入**: 原图像 `image`，滤波函数 `H`  
**输出**: 滤波后的图像 `filtered_image`

```python
M, N = image.shape
# 步骤1: 填充 (Padding) - 防止缠绕错误
P, Q = 2 * M, 2 * N
fp = zeros(P, Q)
fp[0:M, 0:N] = image

# 步骤2: 中心化 (Centering) 与 DFT
for x = 0 to P-1:
    for y = 0 to Q-1:
        fp[x, y] = fp[x, y] * (-1)^(x+y)
F = FFT2(fp)

# 步骤3: 滤波 (Filtering) - 根据卷积定理 G = F * H
G = F * H

# 步骤4: 反变换与反中心化
gp_prim = IFFT2(G)
for x = 0 to P-1:
    for y = 0 to Q-1:
        gp[x, y] = real(gp_prim[x, y]) * (-1)^(x+y)

# 步骤5: 提取 (Cropping)
filtered_image = clip(gp[0:M, 0:N], 0, 255)
```


### 17. 高斯低通滤波 (Gaussian Lowpass Filter - Frequency Domain)

**输入**: 原图像 `image`，截止频率 `d0`  
**输出**: 平滑后的图像 `smooth_image`

```python
# 使用频域滤波通用算法
M, N = image.shape
P, Q = 2 * M, 2 * N

# 构建高斯低通滤波器 H(u,v) = exp(-D²(u,v) / (2*d0²))
center_u = P / 2
center_v = Q / 2
for u = 0 to P-1:
    for v = 0 to Q-1:
        D_square = (u - center_u)^2 + (v - center_v)^2
        H[u, v] = exp(-D_square / (2 * d0^2))

# 应用频域滤波通用算法
smooth_image = frequency_filter(image, H)
```


### 18. 拉普拉斯锐化 (Laplacian Sharpening - Frequency Domain)

**输入**: 原图像 `image`，参数 `c`  
**输出**: 锐化后的图像 `sharpened_image`

```python
# 使用频域滤波通用算法
M, N = image.shape
P, Q = 2 * M, 2 * N

# 构建拉普拉斯锐化滤波器
# H_Lap(u,v) = -4π²(u²+v²)，锐化滤波器 H = 1 - c*H_Lap
center_u = P / 2
center_v = Q / 2
for u = 0 to P-1:
    for v = 0 to Q-1:
        u_norm = u - center_u
        v_norm = v - center_v
        H_Lap = -4 * π^2 * (u_norm^2 + v_norm^2)
        H[u, v] = 1 - c * H_Lap

# 应用频域滤波通用算法
sharpened_image = frequency_filter(image, H)
sharpened_image = clip(sharpened_image, 0, 255)
```


### 19. 高提升滤波 (High-boosting Filtering - Frequency Domain)

**输入**: 原图像 `image`，截止频率 `d0`，参数 `k`  
**输出**: 锐化后的图像 `sharpened_image`

```python
# 使用频域滤波通用算法
M, N = image.shape
P, Q = 2 * M, 2 * N

# 构建高斯低通滤波器 H_LP
center_u = P / 2
center_v = Q / 2
for u = 0 to P-1:
    for v = 0 to Q-1:
        D_square = (u - center_u)^2 + (v - center_v)^2
        H_LP[u, v] = exp(-D_square / (2 * d0^2))

# 构建高提升滤波器 H = 1 + k*(1 - H_LP) = 1 + k*H_HP
for u = 0 to P-1:
    for v = 0 to Q-1:
        H[u, v] = 1 + k * (1 - H_LP[u, v])

# 应用频域滤波通用算法
sharpened_image = frequency_filter(image, H)
sharpened_image = clip(sharpened_image, 0, 255)
```


### 20. 高斯陷波滤波器 (Gaussian Notch Filter)

**输入**: 原图像 `image`，陷波频率位置列表 `notch_points`（每个元素为 `(uk, vk)`），标准差 `sigma`  
**输出**: 滤波后的图像 `filtered_image`

```python
M, N = image.shape
# 步骤1: 填充 (Padding) - 防止缠绕错误
P, Q = 2 * M, 2 * N
fp = zeros(P, Q)
fp[0:M, 0:N] = image

# 步骤2: 中心化 (Centering) 与 DFT
for x = 0 to P-1:
    for y = 0 to Q-1:
        fp[x, y] = fp[x, y] * (-1)^(x+y)
F = FFT2(fp)

# 步骤3: 滤波 (Filtering) - 构建高斯陷波滤波器
# 高斯陷波滤波器 H_NP = 所有陷波点的带阻滤波器乘积
center_u = P / 2
center_v = Q / 2
H = ones(P, Q)

for each (uk, vk) in notch_points:
    # 计算陷波位置及其对称点
    notch_u = center_u + uk
    notch_v = center_v + vk
    notch_u' = center_u - uk
    notch_v' = center_v - vk
    # 构建高斯带阻滤波器 H_k = 1 - exp(-[(u-notch_u)²+(v-notch_v)²]/(2*sigma²))
    H_k = ones(P, Q)
    H_k' = ones(P, Q)
    for u = 0 to P-1:
        for v = 0 to Q-1:
            H_k[u, v] = 1 - exp(-((u - notch_u)^2 + (v - notch_v)^2) / (2 * sigma^2))
            H_k'[u, v] = 1 - exp(-((u - notch_u')^2 + (v - notch_v')^2) / (2 * sigma^2))
    # 多个陷波滤波器相乘
    H = H * H_k * H_k'

# 根据卷积定理 G = F * H
G = F * H

# 步骤4: 反变换与反中心化
gp_origin = IFFT2(G)
for x = 0 to P-1:
    for y = 0 to Q-1:
        gp[x, y] = real(gp_origin[x, y]) * (-1)^(x+y)

# 步骤5: 提取 (Cropping)
filtered_image = clip(gp[0:M, 0:N], 0, 255)
```


### 21. 最优陷波滤波 (Optimal Notch Filtering)

**输入**: 原图像 `image`（含周期性噪声），陷波滤波器 `H_NP`，局部邻域窗口大小 `window_size`  
**输出**: 去噪后的图像 `denoised_image`

```python
M, N = image.shape
# 步骤1: 通过五步法提取噪声的时空域模式
# 步骤1.1: 填充 (Padding)
P, Q = 2 * M, 2 * N
fp = zeros(P, Q)
fp[0:M, 0:N] = image

# 步骤1.2: 中心化 (Centering) 与 DFT
for x = 0 to P-1:
    for y = 0 to Q-1:
        fp[x, y] = fp[x, y] * (-1)^(x+y)
G = FFT2(fp)

# 步骤1.3: 滤波 (Filtering) - 提取干扰模式的主频率分量
N = H_NP * G

# 步骤1.4: 反变换与反中心化
noise_origin = IFFT2(N)
for x = 0 to P-1:
    for y = 0 to Q-1:
        noise_padded[x, y] = real(noise_origin[x, y]) * (-1)^(x+y)

# 步骤1.5: 提取 (Cropping) - 获得空间域噪声模式
noise = noise_padded[0: M, 0: N]

# 步骤2: 遍历原图像每个像素，计算局部统计量并得到最佳去噪图像
denoised_image = zeros(M, N)
half_window = window_size / 2

for i = 0 to M-1:
    for j = 0 to N-1:
        # 定义局部邻域区域边界
        top = max(0, i - half_window)
        bottom = min(M-1, i + half_window)
        left = max(0, j - half_window)
        right = min(N-1, j + half_window)
        local_window = image[top:bottom+1, left:right+1]
        local_noise = noise[top:bottom+1, left:right+1]
        
        # 计算局部统计量
        mean_g = mean(local_window)
        mean_noise = mean(local_noise)
        mean_g_noise = mean(local_window * local_noise)
        var_noise = variance(local_noise)
        
        # 计算加权函数 w = (mean_{g*noise} - mean_g * mean_noise) / var_noise
        if var_noise > 0:
            w = (mean_g_noise - mean_g * mean_noise) / var_noise
        else:
            w = 0
        
        # 计算最佳去噪图像 f̂ = g - w * noise
        denoised_image[i, j] = image[i, j] - w * noise[i, j]

denoised_image = clip(denoised_image, 0, 255)
```


### 22. 维纳滤波 (Wiener Filtering)

**输入**: 退化图像 `image`，退化函数 `H`，噪声功率谱 `S_n`，图像功率谱 `S_f`  
**输出**: 复原后的图像 `restored_image`

```python
M, N = image.shape
# 步骤1: 填充 (Padding) - 防止缠绕错误
P, Q = 2 * M, 2 * N
fp = zeros(P, Q)
fp[0:M, 0:N] = image

# 步骤2: 中心化 (Centering) 与 DFT
for x = 0 to P-1:
    for y = 0 to Q-1:
        fp[x, y] = fp[x, y] * (-1)^(x+y)
G = FFT2(fp)

# 步骤3: 滤波 (Filtering) - 构建并应用维纳滤波器
# 维纳滤波公式: F̂(u,v) = [1/H(u,v)] * [|H(u,v)|² / (|H(u,v)|² + S_n(u,v)/S_f(u,v))] * G(u,v)
H_w = zeros(P, Q)
for u = 0 to P-1:
    for v = 0 to Q-1:
        if H[u, v] != 0:
            H_abs_square = abs(H[u, v])^2
            K = S_n[u, v] / S_f[u, v]
            # 维纳滤波器 H_w = [1/H] * [|H|² / (|H|² + K)]
            H_w[u, v] = (1 / H[u, v]) * (H_abs_square / (H_abs_square + K))
        else:
            H_w[u, v] = 0

F_hat = H_w * G

# 步骤4: 反变换与反中心化
fp_origin = IFFT2(F_hat)
for x = 0 to P-1:
    for y = 0 to Q-1:
        fp[x, y] = real(fp_origin[x, y]) * (-1)^(x+y)

# 步骤5: 提取 (Cropping)
restored_image = clip(fp[0:M, 0:N], 0, 255)
```


### 23. K-means聚类算法 (K-means Clustering)

**输入**: 像素点集合 `pixels`（N个像素点），聚类数量 `K`  
**输出**: 像素点分类结果 `labels`，簇中心 `centers`

```python
N = len(pixels)
# 初始化：随机选择K个像素值作为初始簇中心
centers = random_select(pixels, K)

while not converged:
    # E-step（分类步骤）：根据当前簇中心更新所有像素点的分类
    labels = zeros(N)
    for i = 0 to N-1:
        min_distance = infinity
        for k = 0 to K-1:
            distance = (pixels[i] - centers[k])^2
            if distance < min_distance:
                min_distance = distance
                labels[i] = k
    
    # M-step（更新步骤）：根据当前像素点分类更新所有簇中心
    old_centers = copy(centers)
    for k = 0 to K-1:
        sum_points = 0
        count = 0
        for i = 0 to N-1:
            if labels[i] == k:
                sum_points += pixels[i]
                count += 1
        if count > 0:
            centers[k] = sum_points / count
    
    # 检查是否收敛：簇中心不再变化或达到最大迭代次数
    if max(abs(centers - old_centers)) < epsilon or iteration >= max_iter:
        converged = true
```


### 24. 高斯混合模型分割算法 (Gaussian Mixture Model Segmentation - EM Algorithm)

**输入**: 图像像素值 `pixels`（N个像素点），类别数量 `K`  
**输出**: 像素点分类结果 `labels`，模型参数 `theta = {pi, mu, sigma}`

```python
N = len(pixels)
# 初始化：设定初始参数 theta[0] = {pi_k, mu_k, sigma_k}
pi = ones(K) / K  # 混合系数，初始化为均匀分布
mu = random_select(pixels, K)  # 均值，随机选择K个像素值
sigma = ones(K) * initial_sigma  # 标准差，初始化为相同值

while not converged:
    # E-step（期望步骤）：根据当前参数计算后验概率 P(z_i=k | x_i, theta)
    # P_ik = P(z_i=k | x_i, theta) = pi_k * Phi(x_i | mu_k, sigma_k) / sum_j(pi_j * Phi(x_i | mu_j, sigma_j))
    P = zeros(N, K)
    for i = 0 to N-1:
        sum_prod = 0
        for k = 0 to K-1:
            # 高斯概率密度函数 Phi(x | mu, sigma) = exp(-(x-mu)²/(2*sigma²)) / (sqrt(2*pi)*sigma)
            Phi = exp(-(pixels[i] - mu[k])^2 / (2 * sigma[k]^2)) / (sqrt(2 * pi) * sigma[k])
            P[i, k] = pi[k] * Phi
            sum_prod += P[i, k]
        # 归一化
        for k = 0 to K-1:
            P[i, k] = P[i, k] / sum_prod
    
    # M-step（最大化步骤）：根据后验概率更新参数
    old_pi = copy(pi)
    old_mu = copy(mu)
    old_sigma = copy(sigma)
    
    for k = 0 to K-1:
        # 更新均值 mu_k = sum_i(P_ik * x_i) / sum_i(P_ik)
        sum_px = 0
        sum_p = 0
        for i = 0 to N-1:
            sum_px += P[i, k] * pixels[i]
            sum_p += P[i, k]
        mu[k] = sum_px / sum_p
        
        # 更新方差 sigma_k² = sum_i(P_ik * (x_i - mu_k)²) / sum_i(P_ik)
        sum_p_diff = 0
        for i = 0 to N-1:
            sum_p_diff += P[i, k] * (pixels[i] - mu[k])^2
        sigma[k] = sqrt(sum_p_diff / sum_p)
        
        # 更新混合系数 pi_k = sum_i(P_ik) / N
        pi[k] = sum_p / N
    
    # 检查是否收敛：参数不再变化或达到最大迭代次数
    if max(abs(pi - old_pi)) < epsilon and 
       max(abs(mu - old_mu)) < epsilon and
       max(abs(sigma - old_sigma)) < epsilon or iteration >= max_iter:
        converged = true

# 根据最终的后验概率确定每个像素的类别标签
labels = zeros(N)
for i = 0 to N-1:
    max_p = 0
    for k = 0 to K-1:
        if P[i, k] > max_p:
            max_p = P[i, k]
            labels[i] = k
```


### 25.1 腐蚀 (Erosion)

**输入**: 原图像 `image`，结构元素大小 `k`  
**输出**: 腐蚀后的图像 `eroded_image`

```python
height, width = image.shape
eroded_image = zeros(height, width)
half = k / 2

for i = 0 to height-1:
    for j = 0 to width-1:
        # 定义局部窗口边界
        top = max(0, i - half)
        bottom = min(height, i + half + 1)
        left = max(0, j - half)
        right = min(width, j + half + 1)
        # 腐蚀：取局部窗口内的最小值
        eroded_image[i, j] = min(image[top:bottom, left:right])
```


### 25.2 膨胀 (Dilation)

**输入**: 原图像 `image`，结构元素大小 `k`  
**输出**: 膨胀后的图像 `dilated_image`

```python
height, width = image.shape
dilated_image = zeros(height, width)
half = k / 2

for i = 0 to height-1:
    for j = 0 to width-1:
        # 定义局部窗口边界
        top = max(0, i - half)
        bottom = min(height, i + half + 1)
        left = max(0, j - half)
        right = min(width, j + half + 1)
        # 膨胀：取局部窗口内的最大值
        dilated_image[i, j] = max(image[top:bottom, left:right])
```


### 25.3 开运算 (Opening)

**输入**: 原图像 `image`，结构元素大小 `k`  
**输出**: 开运算后的图像 `opened_image`

```python
# 开运算 = 先腐蚀后膨胀
eroded_image = erode(image, k)
opened_image = dilate(eroded_image, k)
```


### 25.4 闭运算 (Closing)

**输入**: 原图像 `image`，结构元素大小 `k`  
**输出**: 闭运算后的图像 `closed_image`

```python
# 闭运算 = 先膨胀后腐蚀
dilated_image = dilate(image, k)
closed_image = erode(dilated_image, k)
```


### 26. 距离变换 (Distance Transform)

**输入**: 二值图像 `image`（前景为1，背景为0），距离类型 `distance_type`（'D4'或'D8'）  
**输出**: 距离变换图像 `distance_image`

```python
height, width = image.shape
distance_image = zeros(height, width)

# 初始化：边界像素为1，内部像素为无穷大 
for i = 0 to height-1:
    for j = 0 to width-1:
        if image[i, j] == 1:  # 前景像素
            distance_image[i, j] = infinity
            # 检查8邻域是否有背景
            for di = -1 to 1:
                for dj = -1 to 1:
                    if di == 0 and dj == 0:
                        continue
                    if image[i + di, j + dj] == 0:
                        distance_image[i, j] = 1
                        break
                if distance_image[i, j] == 1:
                    break

# 迭代传播：由外向内，遍历到相同像素点时取最小值
changed = true
while changed:
    changed = false
    for i = 0 to height-1:
        for j = 0 to width-1:
            if image[i, j] == 1:  # 前景像素
                # 根据距离类型选择邻域
                if distance_type == 'D4':
                    neighbors = [(-1,0), (1,0), (0,-1), (0,1)]
                else:  # D8
                    neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
                
                # 计算邻域最小值
                min_neighbor = infinity
                for (di, dj) in neighbors:
                    if distance_image[i + di, j + dj] < min_neighbor:
                        min_neighbor = distance_image[i + di, j + dj]
                
                # 更新为当前值与（最小值+1）的较小者
                new_value = min_neighbor + 1
                if new_value < distance_image[i, j]:
                    distance_image[i, j] = new_value
                    changed = true
```


### 27. 图像配准 (Image Registration)

**输入**: 参考图像 `reference_image`，浮动图像 `moving_image`，初始变换参数 `T_init`  
**输出**: 最优变换参数 `T_optimal`，配准后的浮动图像 `registered_image`

```python
# 目标：T̂ = argmax_T (S(I₁, I₂, T) + R(·))
# S: Similarity（相似度，如SSD），T: Transformation（变换，如仿射或FFD），R: Regularization（正则化）

# 初始化
T = T_init

# 迭代配准循环
while not converged:
    # 步骤1: 空间变换（反向变换） - 计算参考图像每个像素Y=(i,j)对应到浮动图像的坐标X=(x,y)
    # S(Y) = T⁻¹(Y) = X，其中S为反向映射函数
    backmap = inverse_transform(reference_image, T)
    # 步骤2: 图像建模和插值算法（双线性插值） - 在浮动图像上插值得出坐标X的灰度值f(x,y)
    warped_image = bilinear_interpolation(backmap, moving_image)
    # 步骤3: 相似性测度/代价函数 - 目标函数：灰度值SSD（Sum of Squared Differences）
    # SSD = mean((reference_image - warped_image)^2)
    similarity = compute_SSD(reference_image, warped_image)
    # 步骤4: 正则化项 - 计算变换的平滑性约束（此例中可设为0）
    regularization = compute_regularization(T)
    # 步骤5: 优化过程 - 梯度下降更新变换参数
    # 梯度计算：有限差分法 ∂f(x)/∂x ≈ (f(x+step) - f(x-step)) / (2*step)
    gradient = compute_gradient_finite_difference(reference_image, moving_image, T)
    T = T - learning_rate * gradient
    
    # 检查收敛条件
    if change_in_cost < epsilon or iteration >= max_iter:
        converged = true

T_optimal = T
registered_image = bilinear_interpolation(inverse_transform(reference_image, T_optimal), moving_image)
```


### 28. FFD自由形变 (Free-Form Deformation)

**输入**: 原图像 `image`，控制点位移 `phi[i,j]`  
**输出**: 变换后的图像 `transformed_image`

```python
# 三次B样条核函数
def cubic_spline(u, index):
    if index == -1:
        return (1 - u)^3 / 6
    elif index == 0:
        return (3*u^3 - 6*u^2 + 4) / 6
    elif index == 1:
        return (-3*u^3 + 3*u^2 + 3*u + 1) / 6
    elif index == 2:
        return u^3 / 6

height, width = image.shape
transformed_image = zeros(height, width)

for x = 0 to height-1:
    for y = 0 to width-1:
        # FFD变换：Q_local(X) = Σ Σ phi[i,j] * β_i(u) * β_j(v)
        # 计算归一化坐标和网格索引
        ix = floor(x / lx)
        iy = floor(y / ly)
        u = x / lx - ix
        v = y / ly - iy
        
        # 计算加权位移（遍历周围4×4=16个控制点）
        Q_local = [0.0, 0.0]
        for i = -1 to 2:
            for j = -1 to 2:
                phi = control_shift[ix + i, iy + j]  # 控制点位移
                weight = cubic_spline(u, i) * cubic_spline(v, j)
                # B样条权重
                Q_local += phi * weight
        
        # 反向映射坐标：原坐标 + 位移
        [new_x, new_y] = [x, y] + Q_local 
        
        # 双线性插值获取像素值
        transformed_image[x, y] = bilinear_interpolation(image, [new_x, new_y])
```


### 29. 局部仿射变换 (Local Affine Transformation)

**输入**: 图像坐标 `X`，局部区域列表 `regions = [[Ω_0, G_0], [Ω_1, G_1], ...]`（每个元素为 \[区域范围, 仿射变换矩阵]），权重指数 `e`  
**输出**: 变换后的坐标 `T(X)`

```python
# 局部仿射变换：T(X) = Σ w_i(X) * G_i(X)（所有区域变换的加权平均）
if X in Ω_i for some i:
    # 区域内点：直接应用该区域的仿射变换
    T(X) = G_i(X)
else:
    # 区域外点：所有区域变换的加权平均（Shepard方法）
    T(X) = 0
    sum_weight = 0
    for [Ω_i, G_i] in regions:
        d_i = distance(X, Ω_i)  # 点到区域的距离
        w_i = 1 / (d_i^e)  # 权重与距离的e次方成反比
        T(X) += w_i * G_i(X)
        sum_weight += w_i
    T(X) = T(X) / sum_weight  # 归一化权重
```


### 30. 前向变换 (Forward Transformation)

**输入**: 原图像 `source_image`，变换函数 `T`  
**输出**: 变换后的图像 `target_image`

```python
height, width = source_image.shape
target_image = zeros(height, width)

for x = 0 to height-1:
    for y = 0 to width-1:
        # 正向变换：获取变换后的坐标
        new_coord = T(x, y)
        new_coord = clip(new_coord, [0, 0], [height-1, width-1])
        new_x = round(new_coord[0])
        new_y = round(new_coord[1])
        
        # 直接将原像素值赋给变换后的位置
        target_image[new_x, new_y] = source_image[x, y]
        # 注意：可能存在重叠或空洞问题
```


### 31. 反向变换 (Backward Transformation)

**输入**: 原图像 `original_image`，反向映射坐标 `backmap`  
**输出**: 变换后的图像 `transformed_image`

```python
height, width = original_image.shape
transformed_image = zeros(height, width)

for x = 0 to height-1:
    for y = 0 to width-1:
        # 获取反向映射坐标（在原图像中的位置）
        orig_x = backmap[x, y, 0]
        orig_y = backmap[x, y, 1]
        
        # 分离整数部分和小数部分
        i = floor(orig_x)
        j = floor(orig_y)
        u = orig_x - i
        v = orig_y - j
        
        # 双线性插值：使用周围4个整数像素点
        i1 = i
        j1 = j
        i2 = min(i + 1, height - 1)
        j2 = min(j + 1, width - 1)
        
        value00 = original_image[i1, j1]
        value01 = original_image[i1, j2]
        value10 = original_image[i2, j1]
        value11 = original_image[i2, j2]
        
        # 双线性插值公式
        transformed_image[x, y] = (1-u)*(1-v)*value00 + 
                                   (1-u)*v*value01 + 
                                   u*(1-v)*value10 + 
                                   u*v*value11
```


### 32. 等值面渲染 (Iso-surface Rendering)

**输入**: 三维标量场 `volume`，等值阈值 `C`  
**输出**: 等值面网格 `surface_mesh`

```python
# 等值面提取：逐个处理单元（体素），比较顶点值，线性插值，连接插值点
surface_mesh = []

for each voxel in volume:
    # 步骤1: 比较体素8个顶点的值与等值C
    vertex_values = get_voxel_vertices_values(volume, voxel)
    
    # 步骤2: 沿边进行线性插值（找到等值面与边的交点）
    interpolated_points = []
    for each edge in voxel.edges:
        v1, v2 = edge.vertices
        if (vertex_values[v1] >= C) != (vertex_values[v2] >= C):
            t = (C - vertex_values[v1]) / (vertex_values[v2] - vertex_values[v1])
            point = (1 - t) * v1 + t * v2
            interpolated_points.append(point)
    
    # 步骤3: 连接插值点构建三角面片
    if len(interpolated_points) >= 3:
        triangles = connect_points(interpolated_points)
        surface_mesh.extend(triangles)
```


### 33. 光线投射法体渲染 (Ray Casting Volume Rendering)

**输入**: 光线 `R`，三维标量场 `volume`，传输函数 `transfer_function`  
**输出**: 光线颜色 `C`，累积透明度 `A`

```python
# TraceRay算法：沿光线采样并合成
C = 0
A = 0
# 计算光线进入和离开数据场的位置（对象空间）
x1 = First(R)  # 在对象空间中光线进入的位置
x2 = Last(R)   # 在对象空间中光线离开的位置
# 转换到图像空间
u1 = Image(x1)  # 转换到图像空间的位置
u2 = Image(x2)  # 转换到图像空间的位置

# 等距离采样重建
for S = u1 to u2:
    # 转换回对象空间
    Sx = Object(S)  # 转换到对象空间的位置
    # 早期射线终止：如果透明度已饱和则停止
    if A < 1:
        # 获取Sx点处的颜色C(Sx)和透明参数a(Sx)
        value = trilinear_interpolation(volume, Sx)
        C_Sx = transfer_function_color(value)
        a_Sx = transfer_function_alpha(value)
        # 光学积分：从前向后合成（Front-to-Back）
        C = C + (1 - A) * C_Sx
        A = A + (1 - A) * a_Sx
```


### 34. 投雪球法体渲染 (Splatting Volume Rendering)

**输入**: 三维标量场 `volume`，传输函数 `transfer_function`  
**输出**: 渲染图像 `image`

```python
# 投雪球法：从数据空间出发向图像平面传递数据信息，累积光亮度贡献
height, width = get_image_size()
image = zeros(height, width)
image_alpha = zeros(height, width)

# 遍历每个体素（正向扫描）
for each voxel (i, j, k) in volume:
    # 数据分类 - 获取体素的颜色和不透明度
    value = volume[i, j, k]
    C_voxel = transfer_function_color(value)
    alpha = transfer_function_alpha(value)
    
    # 投影到图像平面 - 计算体素的投影足迹（footprint）
    footprint = compute_footprint(voxel, projection_matrix)
    
    for each pixel (u, v) in footprint:
        # 计算权重（高斯核函数或重构核）
        weight = compute_weight(voxel, pixel)
        
        # 累积颜色和透明度贡献
        image[u, v] += C_voxel * alpha * weight
        image_alpha[u, v] += alpha * weight

# 归一化
for u = 0 to height-1:
    for v = 0 to width-1:
        if image_alpha[u, v] > 0:
            image[u, v] = image[u, v] / image_alpha[u, v]
```


### 35. Blinn-Phong光照模型 (Blinn-Phong Illumination Model)

**输入**: 采样点颜色 `C_TF`，法线方向 `N`，光线方向 `L`，视线方向 `V`，光照系数 `ka, kd, ks, n`  
**输出**: 光照后的颜色 `C`

```python
# Blinn-Phong光照模型：C = (ka + kd(N·L)) · C_TF + ks(N·H)^n
# 归一化向量
N = normalize(N)
L = normalize(L)
V = normalize(V)

# 计算半角向量H（V和L的平均方向）
H = normalize(V + L)
# 计算环境光项
ambient = ka * C_TF

# 计算漫反射项：N·L（法线与光线方向的点积）
N_dot_L = dot(N, L)
diffuse = kd * N_dot_L * C_TF

# 计算镜面反射项：N·H（法线与半角向量的点积）的n次方
N_dot_H = dot(N, H)
specular = ks * (N_dot_H)^n

# 最终颜色
C = ambient + diffuse + specular
```
