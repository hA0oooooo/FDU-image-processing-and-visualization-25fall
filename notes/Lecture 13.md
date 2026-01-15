### 连续与Folding

在非线性变换中，形变场的拓扑性质至关重要，Folding即折叠现象，指的是形变场在局部发生了重叠或翻转，直观表现为变换后的网格线出现相交

从数学角度看，当变换函数的雅可比行列式在某些点小于或等于零时，即 $det(J) \le 0$，就会发生Folding 1：变换在这些局部区域不再是双射，破坏了图像的拓扑结构，导致变换不可逆

针对Folding问题，存在两种处理思路：

- TPS方法的软约束改善

    TPS（薄板样条）模型通过在能量函数中引入正则化项来抑制剧烈的形变。这种方法利用弯曲能量来惩罚过大的二阶导数，从而倾向于生成更平滑的变换场，但这仅能降低Folding发生的概率

- 微分同胚的硬约束解决

    为了从根本上解决折叠问题，可以采用级联（Concatenation）的策略，其核心思想是将一个大的形变分解为一系列微小的形变叠加。在数学上，这对应于微分同胚（Diffeomorphic）变换，通过积分速度场来生成形变，这种方法能够保证变换在整个过程中始终保持连续、光滑且可逆，从而彻底避免Folding

### 局部仿射变换 Local Affine

局部仿射变换是一种介于全局线性变换和自由非线性形变之间的折中模型，旨在解决多目标不同步运动的问题

1. 场景引入与直观理解

    在医学图像中，往往存在多个具有独立运动模式的解剖结构。例如心脏图像中，左心室和右心室可能分别向不同的方向收缩或旋转
    如果使用全局仿射变换，无法同时满足两个心室的运动需求；如果使用像素级的自由形 变，计算量大且容易丢失解剖结构的刚性特征
    局部仿射的核心在于：将图像划分为若干个具有解剖意义的子区域（如心室区域），在区域内部
    保持刚性或仿射运动，而在区域之间通过平滑过渡来维持图像的连续性
    
2. 核心机制：区域控制与全局影响

    该模型将变换控制从单个的点扩展到了区域。虽然变换是基于局部区域定义的，但通过加权机制，这些局部变换会影响到整个图像空间，因此属于区域控制、全局影响的模型 

3. 数学定义与分段函数

    假设图像中存在 $n$ 个局部区域 $\Omega_i$，每个区域对应一个独立的仿射变换矩阵 $G_i$。对于图像空间中的任意一点 $X$，其变换函数 $T(X)$ 定义如下
	- 区域内点：如果点 $X$ 落在某个特定区域 $\Omega_i$ 内部，它完全遵循该区域的仿射变换 $G_i(X)$ 进行运动
	- 区域外点：如果点 $X$ 不属于任何定义的区域（即处于背景或过渡地带），其运动由所有区域变换矩阵 $G_i(X)$ 的加权平均来决定
$$T(X) = \begin{cases} G_{i}(X), & X \in \Omega_{i} \\ \sum_{j=1}^{n} w_{j}(X)G_{j}(X), & X \notin \cup \Omega_{i} \end{cases}$$

4. 权重计算与空间插值

    为了保证图像在不同区域之间过渡时不会发生撕裂，通常采用Shepard方法
$$w_{i}(X) = \frac{1/d_{i}(X)^{e}}{\sum_{k=1}^{n} 1/d_{k}(X)^{e}}$$

- $d_i(X)$ 表示像素点 $X$ 到第 $i$ 个区域 $\Omega_i$ 的距离
- $e$ 是控制衰减速率的指数，距离越近，分母越小，权重越大；距离越远，权重迅速衰减


### FFD (Free-Form Deformation)

##### FFD 直觉：点控制与局部影响

FFD 的核心直觉可以比喻为操作一张覆盖在图像上的弹性网格

1. **网格隐喻**：想象你在图像上铺了一层透明的橡胶网格（Lattice）。网格的交叉点就是控制点 (Control Points)
2. **间接形变**：你并不直接移动图像上的像素，而是用手捏住网格上的某个控制点 $P$ 并移动
3. **局部影响**：当你移动一个控制点时，只有这个点附近的网格会被拉扯变形，从而带动该区域内的图像像素移动。离这个控制点很远的网格和图像完全不受影响
4. **平滑过渡**：由于网格是弹性的（由 B-样条函数定义），控制点附近的形变是平滑过渡的，不会出现尖锐的折痕

这就是 **点控制**（通过操作离散的网格点）和 **局部影响**（一个点只改变邻域）的含义

##### FFD 核心公式与物理意义

FFD 将一个像素点 $X$ 的最终位置 $Y$ 定义为原始位置加上一个 局部位移量
$$Y = T(X) = X + Q_{local}(X)$$
其中 $Q_{local}(X)$ 是局部位移场，它是由周围控制点的位移加权求和得到的：

$$Q_{local}(X) = \sum_{k} \phi_{k} W(X, P_{k})$$

- $X$：图像中某个像素点的原始坐标
- $\phi_{k}$：第 $k$ 个控制点的 位移向量（即控制点动了多少）
- $P_{k}$：第 $k$ 个控制点的原始坐标
- $W(X, P_{k})$：权重函数

##### 核函数：三次 B-样条 (Cubic B-Spline)

为了实现平滑且局部的形变，FFD 选择 三次 B-样条 作为权重函数 $W$
$$\beta^{(3)}(a) = \begin{cases} \frac{1}{6}(4 - 6a^2 + 3|a|^3), & 0 \le |a| < 1 \\ \frac{1}{6}(2 - |a|)^3, & 1 \le |a| < 2 \\ 0, & 2 \le |a| \end{cases}$$

这里 $a$ 是归一化的相对距离：$a = \frac{|X - P_i|}{l}$，其中 $l$ 是网格间距

##### 三维拓展与张量积

图像通常是二维或三维的，FFD 通过 张量积 (Tensor Product) 将一维的 B-样条函数拓展到高维。简单来说，就是把 x, y, z 三个方向的权重乘起来

三维公式：

$$Q_{local}(X) = \sum_{i} \sum_{j} \sum_{k} \phi_{ijk} \cdot \beta^{(3)}(u_i) \cdot \beta^{(3)}(v_j) \cdot \beta^{(3)}(w_k)$$

- $\phi_{ijk}$：索引为 $(i, j, k)$ 的控制点的位移向量
- $u_i, v_j, w_k$：像素点 $X$ 与该控制点在 x, y, z 三个轴向上的归一化距离。例如 $u_i = \frac{x - P_x}{l_x}$。
- 每一项 $\beta^{(3)}$ 都是一个一维的权重计算


##### 另一种表达方式：局部坐标系优化

在实际计算（特别是编程实现）中，直接计算所有控制点的距离效率很低，通常使用 局部坐标系 $u, v, w \in [0, 1)$ 来简化表达

假设点 $X$ 落在以控制点 $P_{ijk}$ 为左下角的网格单元中。设 $u$ 为 $X$ 在该网格内的相对坐标，$u \in [0, 1)$，此时，能影响点 $X$ 的只有它左边、左中、右中、右边这四个方位的邻居控制点。它们的权重可以预先推导为四个固定的多项式函数：

1. 左邻居 ($i-1$)，距离约为 $1+u$：
$$\beta_{-1}(u) = (1-u)^3 / 6$$
2. 当前基准点 ($i$)，距离约为 $u$：
$$\beta_{0}(u) = (3u^3 - 6u^2 + 4) / 6$$
3. 右邻居 ($i+1$)，距离约为 $1-u$：
$$\beta_{1}(u) = (-3u^3 + 3u^2 + 3u + 1) / 6$$
4. 远右邻居 ($i+2$)，距离约为 $2-u$：
$$\beta_{2}(u) = u^3 / 6$$

##### 代码实例解析

**1. 定位网格与局部坐标**
```C++
// iGrid 存储整数部分，即当前像素落在哪一个网格单元（对应公式中的索引 i, j, k）
iGrid[index] = static_cast<int>(floor(fFrom[index])); 

// fu 存储小数部分，即局部坐标 u, v, w，取值在 [0, 1) 之间
fu[index] = fFrom[index] - floor(fFrom[index]); 
```

这一步将物理坐标转化为了算法友好的局部坐标，`iGrid` 确定了我们要找哪一块 $4 \times 4 \times 4$ 的控制点，`fu` 确定了我们在格子内部的精确位置

**2. 预计算权重（利用另一种表达方式）**

```C++
// 循环 4 次，计算 u, v, w 三个方向上，周围 4 层邻居的权重
for (int ite = 0; ite < 4; ++ite) {
    fBSpline_u[ite] = BSplinei(ite - 1, fu[0]); // 对应 beta_{-1} 到 beta_{2}
    fBSpline_v[ite] = BSplinei(ite - 1, fu[1]);
    fBSpline_w[ite] = BSplinei(ite - 1, fu[2]);
}
```

这里调用的 `BSplinei` 函数，内部就是上一节提到的 $\beta_{-1}(u)$ 到 $\beta_{2}(u)$ 这四个多项式公式，代码预先算好了 x, y, z 三个方向的权重表，避免在三层循环内部重复计算。

**3. 卷积求和（计算局部位移场）**

```C++
// 三层循环遍历周围 4x4x4 = 64 个邻居控制点
for (int i = -1; i <= 2; i++)
  for (int j = -1; j <= 2; j++)
    for (int k = -1; k <= 2; k++) 
    {
       // 获取当前遍历到的控制点的位移向量 phi
       // iGrid 是基准索引，i, j, k 是相对偏移量
       float poff = GetCtrPnt(iGrid[0] + i, iGrid[1] + j, iGrid[2] + k);
       
       // 如果这个控制点有位移，则计算它的贡献
       if (poff != 0) {
           // 权重的张量积：总权重 = x权重 * y权重 * z权重
           // 注意数组下标 +1 是因为 C++ 数组从 0 开始，而循环从 -1 开始
           float w = fBSpline_u[i + 1] * fBSpline_v[j + 1] * fBSpline_w[k + 1];
           
           // 累加位移：位移 = Sum(权重 * 控制点位移)
           afOffset[idx] += poff[idx] * w;
       }
    }
```

这段循环完全对应了公式：$Q_{local}(X) = \sum \sum \sum \phi_{ijk} \cdot W$，它通过查表法快速获取了 $W$，并累加了所有相关控制点的贡献

**4. 应用变换**

```C++
// 最终坐标 = 原始坐标 + 计算出的局部位移偏移量
fTo[idx] = fFrom[idx] + afOffset[idx];
```

对应最开始的公式 $Y = X + Q_{local}(X)$


### 图像配准

##### 图像配准的数学优化框架

图像配准的本质是一个最优化问题，旨在寻找最优的空间变换参数 $\hat{T}$。其核心数学表达为：

$$\hat{T} = \text{argmax}_{T} ( S(I_{fixed}, I_{moving} \circ T) + \mathcal{R}(T) )$$

该公式包含两部分竞争项：

1. 相似性测度项 $S$：衡量参考图像 $I_{fixed}$ 与变换后的浮动图像 $I_{moving}(T(x))$ 之间的匹配程度。常见的度量包括灰度平方差和 (SSD)：
$$SSD(T) = \frac{1}{|\Omega|} \sum_{x \in \Omega} (I_{fixed}(x) - I_{moving}(T(x)))^2$$
2. 正则化项 $\mathcal{R}(T)$：用于约束变换 $T$ 的平滑性，防止出现物理上不可能的折叠或过度扭曲。例如在 TPS 中，$\mathcal{R}$ 为弯曲能量函数。
    
整个配准过程是一个迭代闭环：

初始化参数 $\theta$ $\rightarrow$ 空间变换 $\rightarrow$ 插值重采样 $\rightarrow$ 计算相似度 $S$ $\rightarrow$ 计算梯度 $\nabla S$ $\rightarrow$ 更新参数 $\theta \leftarrow \theta - \alpha \nabla S$。

##### 图像变换策略：前向与反向

在计算 $I_{moving}(T(x))$ 这一步时，涉及到离散网格的映射问题。

**前向变换 (Forward Warping)**

定义：对于源图像定义域 $\Omega_s$ 中的每一个整点坐标 $x$，计算其在目标定义域 $\Omega_t$ 中的位置 $x' = T(x)$，并将 $I_s(x)$ 赋值给 $I_t(x')$

数学描述：
$$I_t(T(x)) = I_s(x), \quad \forall x \in \Omega_s \cap \mathbb{Z}^2$$

存在的问题：

1. 空洞 (Holes)：变换 $T$ 通常是非线性的或者包含缩放，映射后的坐标集合 $\{T(x) | x \in \Omega_s\}$ 在 $\Omega_t$ 中是稀疏分布的，这意味着存在目标像素 $y \in \Omega_t \cap \mathbb{Z}^2$，使得没有任何源像素映射到它，导致 $I_t(y)$ 未定义（出现黑点或裂缝）
2. 重叠 (Overlaps)：可能存在 $x_1 \neq x_2$，使得 $T(x_1)$ 和 $T(x_2)$ 映射到同一个目标像素（或者非常接近），导致像素值的竞争冲突


**反向变换 (Backward Warping)**

定义：为了保证目标图像 $I_t$ 的完整性，我们要遍历目标定义域 $\Omega_t$ 中的每一个整点像素 $y$，寻找它在源图像中的对应位置 $x$

$$I_t(y) = I_s(T^{-1}(y)), \quad \forall y \in \Omega_t \cap \mathbb{Z}^2$$

或者定义一个从目标到源的逆映射 $F = T^{-1}$，则有 $x = F(y)$

优势：

由于我们是遍历目标图像的所有像素 $y$，可以保证每一个 $y$ 都有且仅有一个对应的灰度值，从数学上根除了空洞和重叠问题

##### 插值技术：解决非整数坐标映射

在反向变换中，计算出的源坐标 $x = T^{-1}(y)$ 通常是一个浮点数向量 $x \in \mathbb{R}^2$，而不属于整数网格 $\mathbb{Z}^2$，由于数字图像 $I_s$ 仅在整数坐标上有定义，我们无法直接读取 $I_s(x)$

插值 (Interpolation) 的目的就是基于离散的采样点构建连续的图像函数 $I_s(x)$，从而估算任意连续坐标处的灰度值

对于任意浮点坐标 $x = (x_1, x_2)^T$，可以分解为整数部分（基准点）和小数部分（相对偏移）：

基准索引：$i = \lfloor x_1 \rfloor, \quad j = \lfloor x_2 \rfloor$

相对偏移：$u = x_1 - i, \quad v = x_2 - j, \quad u, v \in [0, 1)$

双线性插值 (Bilinear Interpolation)

这是最常用的插值方法，通过在两个方向上分别进行线性插值来实现。它利用了浮点坐标 $x$ 周围的四个整数邻居：$I(i, j), I(i+1, j), I(i, j+1), I(i+1, j+1)$。

数学公式推导：

1. 先在 X 方向插值：
$$I(x_1, j) \approx (1-u)I(i, j) + uI(i+1, j)$$$$I(x_1, j+1) \approx (1-u)I(i, j+1) + uI(i+1, j+1)$$

2. 再在 Y 方向插值：
$$I(x_1, x_2) \approx (1-v)I(x_1, j) + vI(x_1, j+1)$$

综合展开式：

$$I(x) \approx (1-u)(1-v)I(i, j) + u(1-v)I(i+1, j) + (1-u)vI(i, j+1) + uvI(i+1, j+1)$$


### 代码实验

该实验实现了一个完整的图像配准闭环：

1. **准备**：将图像转化为坐标网格
2. **变换**：构建仿射变换矩阵，计算坐标的新位置
3. **插值**：利用反向变换和双线性插值，生成变形后的图像
4. **评估**：计算图像差异（SSD）
5. **优化**：通过有限差分法近似求导，并使用梯度下降更新参数。    

##### 1. 数据预处理：生成坐标网格 (Make Coordinate)

代码逻辑：利用 np.meshgrid 生成网格，coord 是一个 $2 \times H \times W$ 的矩阵

```Python
def make_coord(shape, flatten=False):
    # 输入 shape: (H, W) 或 (H, W, D)
    # 输出 coord: (2, H, W) 存储了每个像素的坐标索引
    coord_seqs = []
    for i, n in enumerate(shape):
        seq = np.arange(n).astype(float) # 生成 [0, 1, ..., n-1]
        coord_seqs.append(seq)
    
    # meshgrid 生成网格矩阵
    coord = np.stack(np.meshgrid(*coord_seqs, indexing='ij'), axis=0)
    
    if flatten:
        # 将 (2, H, W) 变为 (2, H*W)，方便矩阵乘法
        coord = coord.reshape(len(shape), -1)
    return coord
```

##### 2. 空间变换模型：仿射变换 (Affine Transformation)

通过矩阵运算将原始坐标 $x$ 映射为新坐标 $T(x)$，将变换矩阵分解为旋转、缩放、错切和平移

1. 从 `parameters` 字典中提取 $\theta$（角度）、缩放因子、错切因子和平移量
2. 分别构建旋转矩阵 $R$、缩放矩阵 $S$、错切矩阵 $H$
3. 通过矩阵乘法组合变换：$M = H \cdot S \cdot R$
4. 应用变换：`new_coord = M * old_coord + translation`

注意：这里的 `reshape` 操作是因为矩阵乘法需要二维矩阵（$2 \times N$），算完后再变回 $(2, H, W)$ 的图像形状

```Python
def Affine_transformation(img_coord, parameters):
    # 输入 img_coord: 原始坐标网格
    # 输入 parameters: 包含 theta, scale, shear, trans 等参数
    
    # 1. 构建旋转矩阵 (Rotation)
    theta = parameters['theta']
    Rotate = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
    
    # 2. 构建缩放矩阵 (Scaling)
    scale_a, scale_b = parameters['scale_a'], parameters['scale_b']
    Scaling = np.array([[scale_a, 0], [0, scale_b]])
    
    # 3. 构建错切矩阵 (Shear)
    shear_a, shear_b = parameters['shear_a'], parameters['shear_b']
    Shear = np.array([[1, shear_a], [shear_b, 1]])
    
    # 4. 构建平移向量 (Translation)
    trans_a, trans_b = parameters['trans_a'], parameters['trans_b']
    Translation = np.array([[trans_a], [trans_b]])
    
    # 5. 组合变换并计算 T(x)
    # 顺序：先旋转，再缩放，再错切，最后平移
    # reshape(-1) 是把图像展平进行矩阵运算
    matrix = np.dot(Shear, np.dot(Scaling, Rotate))
    new_img_coord = np.dot(matrix, img_coord.reshape(img_coord.shape[0], -1)) + Translation
    
    # 变回 (2, H, W) 的形状返回
    return new_img_coord.reshape(img_coord.shape)
```

##### 3. 插值与重采样：线性插值 (Linear Interpolation)

反向变换，上一步算出的 `new_img_coord` 是浮点数（例如坐标 10.4, 20.6），我们需要计算这个位置的像素值

1. 取整与求余：
    coord_floor 是整数部分（基准点 $i, j$）
    rel_coord 是小数部分（权重 $u, v$）
2. 边界处理：
    使用 np.clip 限制坐标范围，防止变换到图像外面的坐标导致程序崩溃
3. 双线性插值：
    利用公式加权平均。比如 (1-rel_coord\[0]) * (1-rel_coord\[1]) 就是左上角像素的权重。

```Python
def linear_interpolation(coord, img):
    # 输入 coord: 变换后的浮点坐标 T(x)
    # 输入 img: 源图像 (Source Image)
    
    # 1. 分离整数和小数部分
    coord_floor = np.floor(coord).astype(int) # 基准点 (i, j)
    rel_coord = coord - coord_floor           # 相对偏移 (u, v)
    
    # 2. 边界处理 (防止越界)
    # 限制坐标在 [0, H-1] 和 [0, W-1] 之间
    valid_x = np.clip(coord_floor[0], 0, img.shape[-2]-1)
    valid_y = np.clip(coord_floor[1], 0, img.shape[-1]-1)
    
    # 获取周围四个邻居的坐标：(i,j), (i,j+1), (i+1,j), (i+1,j+1)
    # 注意：这里为了简化代码逻辑，需要对 +1 的邻居也做 clip 处理
    
    # 3. 计算插值 (双线性公式)
    # 权重计算：(1-u)(1-v)*I_00 + ...
    output_img = (
        (1 - rel_coord[0]) * (1 - rel_coord[1]) * img[:, valid_x, valid_y] 
        +
        (1 - rel_coord[0]) * rel_coord[1] * img[:, valid_x, np.clip(valid_y+1, 0, img.shape[-1]-1)] 
        +
        rel_coord[0] * (1 - rel_coord[1]) * img[:, np.clip(valid_x+1, 0, img.shape[-2]-1), valid_y] 
        +
        rel_coord[0] * rel_coord[1] * img[:, np.clip(valid_x+1, 0, img.shape[-2]-1), np.clip(valid_y+1, 0, img.shape[-1]-1)]
    )
    
    return output_img
```

##### 4. 相似性测度：SSD (Sum of Squared Differences)

目标函数，通过计算变换后的浮动图像与参考图像的均方误差来衡量配准的好坏

```Python
def SSD_similarity(input_tensor, target_tensor):
    # 输入 input_tensor: 变形后的浮动图像
    # 输入 target_tensor: 参考图像
    
    # 计算差值的平方均值
    return ((input_tensor - target_tensor) ** 2).mean()
```

##### 5. 优化：梯度估计 (Compute Grads)

对于每一个参数（如 theta）：
1. 让参数增加一点点 (`+step`)，做一遍变换和插值，算出误差 `loss1`
2. 让参数减少一点点 (`-step`)，做一遍变换和插值，算出误差 `loss2`
3. 梯度 $\approx \frac{loss1 - loss2}{2 \times step}$

```Python
def compute_grads(input_tensor, target_tensor, transformation, similarity, parameters, parameters_step):
    grads = {}
    # 生成参考图像的坐标网格
    img_coord = make_coord(target_tensor.shape[1:])
    
    for key in parameters.keys():
        step = parameters_step[key]
        orig_val = parameters[key]
        
        # 1. 前向扰动 f(x + step)
        parameters[key] = orig_val + step
        out1 = linear_interpolation(transformation(img_coord, parameters), input_tensor)
        loss1 = similarity(out1, target_tensor)
        
        # 2. 后向扰动 f(x - step)
        parameters[key] = orig_val - step
        out2 = linear_interpolation(transformation(img_coord, parameters), input_tensor)
        loss2 = similarity(out2, target_tensor)
        
        # 3. 计算中心差分梯度
        grads[key] = (loss1 - loss2) / (2 * step)
        
        # 还原参数
        parameters[key] = orig_val
        
    return grads
```
##### 6. 主循环：迭代优化 (Iterative Optimization)

代码逻辑：
1. 初始化参数
2. 进入循环：
    - **变换**：计算坐标映射
    - **插值**：生成当前参数下的图像
    - **算梯度**：看看参数怎么改能减小误差
    - **更新参数**：`param = param - lr * grad` (梯度下降)
3. 重复上述过程直到收敛

```Python
# 初始化
parameters, parameters_step, lrs = Affine_initialize_parameters()
img_coord = make_coord(target_img_tensor.shape[1:]) 

for step in range(steps):
    # 1. 正向传播：生成变形图像
    deformed_img = linear_interpolation(
        Affine_transformation(img_coord, parameters), 
        float_img_tensor
    )
    
    # 2. 计算当前误差 (仅用于观察，不参与反向传播计算)
    ssd = SSD_similarity(deformed_img, target_img_tensor)
    
    # 3. 计算梯度 (核心步骤)
    grads = compute_grads(float_img_tensor, target_img_tensor, 
                          Affine_transformation, SSD_similarity, 
                          parameters, parameters_step)
                          
    # 4. 更新参数 (梯度下降)
    parameters = update_parameters(parameters, grads, lrs)
```