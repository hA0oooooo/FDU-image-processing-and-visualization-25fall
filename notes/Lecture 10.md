

##### 泛化大津算法：最大化类间方差

Otsu 算法是一种自适应阈值确定方法，既适用于二分类，也可泛化至多分类问题

目标是找到分类 $K=\{C_1, C_2, ..., C_k\}$，使得类间方差 $\sigma_{bet}^2$ 最大化

定义如下变量：

- $\pi_k = \frac{N_k}{N}$：第 $k$ 类像素占比
- $\mu_k$：第 $k$ 类的平均灰度
- $\mu_{glb}$：图像全局平均灰度（常数）

类间方差公式推导与展开：

$$\sigma_{bet}^2 = \sum_{k \in K} \pi_k (\mu_k - \mu_{glb})^2$$

将平方项展开并利用 $\sum \pi_k = 1$ 和 $\sum \pi_k \mu_k = \mu_{glb}$ 的性质：

$$\begin{aligned} \sigma_{bet}^2 &= \sum_{k \in K} \pi_k (\mu_k^2 - 2\mu_k \mu_{glb} + \mu_{glb}^2) \\ &= \sum_{k \in K} \pi_k \mu_k^2 - 2\mu_{glb} \sum_{k \in K} \pi_k \mu_k + \mu_{glb}^2 \sum_{k \in K} \pi_k \\ &= \sum_{k \in K} \pi_k \mu_k^2 - 2\mu_{glb}^2 + \mu_{glb}^2 \\ &= \sum_{k \in K} \pi_k \mu_k^2 - \mu_{glb}^2 \end{aligned}$$

由于 $\mu_{glb}$ 是全局统计量（常数），最大化 $\sigma_{bet}^2$ 等价于最大化 $\sum_{k \in K} \pi_k \mu_k^2$

##### 最小化类内方差

目标是使得每个类别内部的像素差异最小，即加权类内方差 $\sigma_{in}^2$ 最小化

类内方差公式推导：

$$\sigma_{in}^2 = \sum_{k \in K} \pi_k \sigma_k^2 = \sum_{k \in K} \pi_k \left( \frac{1}{N_k} \sum_{x \in \Omega_k} (I_x - \mu_k)^2 \right)$$

其中 $\sigma_k^2$ 是第 $k$ 类的方差

将公式展开，利用 $E[I^2]$ 表示图像二阶矩（常数）：

$$\begin{aligned} \sigma_{in}^2 &= \frac{1}{N} \sum_{k \in K} \sum_{x \in \Omega_k} (I_x^2 - 2I_x\mu_k + \mu_k^2) \\ &= \frac{1}{N} \sum_{x \in \Omega} I_x^2 - \frac{2}{N} \sum_{k \in K} \mu_k \sum_{x \in \Omega_k} I_x + \sum_{k \in K} \pi_k \mu_k^2 \\ &= E[I^2] - 2\sum_{k \in K} \pi_k \mu_k^2 + \sum_{k \in K} \pi_k \mu_k^2 \\ &= E[I^2] - \sum_{k \in K} \pi_k \mu_k^2 \end{aligned}$$

关于等价性的证明：

- 根据全方差公式，图像的总方差 $\sigma_{tot}^2$ 是固定的，且满足 $\sigma_{tot}^2 = \sigma_{in}^2 + \sigma_{bet}^2$
- 由于 $\sigma_{tot}^2$ 对特定图像为常数，因此 $\min \sigma_{in}^2 \iff \max \sigma_{bet}^2$
- 这证明了两种优化目标是完全等价的；在计算上，由于 $\sigma_{bet}^2$ 计算主要涉及均值操作，通常比计算二阶矩的 $\sigma_{in}^2$ 更高效

##### k-means 聚类算法

K-means 是一种迭代优化算法，旨在将 $N$ 个像素点划分为 $K$ 个簇，使得目标函数（误差平方和）最小化

目标函数：

最小化像素灰度值 $I_x$ 与其所属簇中心 $\mu_k$ 之间的欧氏距离平方和：

$$J(I, K) = \sum_{x=0}^{N-1} \sum_{k=0}^{K-1} \delta(c_x, k) (I_x - \mu_k)^2$$

其中 $\delta(c_x, k)$ 是指示函数，当像素 $x$ 属于类别 $k$ 时为1，否则为0

算法迭代步骤：

- 初始化：随机选取 $K$ 个数值作为初始簇中心 $\{\mu_0, ..., \mu_{K-1}\}$
- 第一步（E-step样行为）：固定中心，更新分类；对于每个像素 $x$，计算其到各个中心 $\mu_k$ 的距离，将其归类到最近的簇

$$c_x = \arg \min_k |I_x - \mu_k|^2$$

- 第二步（M-step样行为）：固定分类，更新中心；根据当前的分类结果，重新计算每个簇的中心（均值）

$$\mu_k = \frac{\sum_{x \in \Omega} \delta(c_x, k) I_x}{\sum_{x \in \Omega} \delta(c_x, k)}$$

- 终止条件：重复上述步骤直至收敛（簇中心不再变化）或达到最大迭代次数

理论性质：

固定 $\mu_k$ 时，最小化 $J$ 等价于寻找最近邻；固定 $c_x$ 时，最小化 $J$ 等价于求解均值；这保证了算法在每一步迭代中目标函数 $J$ 都是非递增的，从而保证收敛（但不保证全局最优）


