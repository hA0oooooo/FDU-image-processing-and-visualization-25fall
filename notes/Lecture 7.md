### Sampling & Reconstruction

##### (1) 采样函数的傅里叶变换

设 $f(t)$ 为时域信号，其傅里叶变换为 $F(u)$

设 $f_n$ 为离散样本 $f_n = f(n\Delta T)$ 

采样函数 $\overline{f}(t)$ 定义为 $f(t)$ 与冲击串 $S_{\Delta T}(t)$ 的乘积：

$\overline{f}(t) = f(t) S_{\Delta T}(t)$ 

对 $\overline{f}(t)$ 进行傅里叶变换，得到 $\overline{F}(u)$，根据连续卷积定理 ($f(t)h(t) \iff F(u)*H(u)$)：

$\overline{F}(u) = \mathcal{F}\{f(t) S_{\Delta T}(t)\} = F(u) * \mathcal{F}\{S_{\Delta T}(t)\}$

冲激串的傅里叶变换为 $\mathcal{F}\{S_{\Delta T}(t)\} = S(u) = \frac{1}{\Delta T} \sum_{n=-\infty}^{\infty} \delta(u - \frac{n}{\Delta T})$

代入 $\overline{F}(u)$ 的表达式：$\overline{F}(u) = F(u) * \left( \frac{1}{\Delta T} \sum_{n=-\infty}^{\infty} \delta(u - \frac{n}{\Delta T}) \right)$

利用卷积的线性性质：$\overline{F}(u) = \frac{1}{\Delta T} \sum_{n=-\infty}^{\infty} \left( F(u) * \delta(u - \frac{n}{\Delta T}) \right)$

利用 $\delta$ 函数的卷积特性 ($F(u) * \delta(u-u_0) = F(u-u_0)$)：

$\overline{F}(u) = \frac{1}{\Delta T} \sum_{n=-\infty}^{\infty} F(u - \frac{n}{\Delta T})$ 

- 回答：为什么没有 $\frac{1}{M}$ 因子？
- 这里的推导使用的是连续傅里叶变换 (FT)，卷积定理为 $f \cdot h \iff F * H$
- $\frac{1}{M}$ 因子是离散傅里叶变换 (DFT)在定义 $f \cdot h \iff \frac{1}{M} F * H$ 时，为保证变换对的归一化

##### (2) 奈奎斯特采样定律与信号恢复

**频域分析**

$\overline{F}(u)$ 是原信号 $F(u)$ 的无限周期性拷贝，拷贝间隔为 $\frac{1}{\Delta T}$ 

假设 $f(t)$ 是带限函数 (band-limited)，即 $F(u)$ 在 $[-u_{max}, u_{max}]$ 之外为 $0$

为使 $\overline{F}(u)$ 的拷贝不发生重叠 (混叠, Aliasing)，采样间隔 $\frac{1}{\Delta T}$ 必须大于 $F(u)$ 的总宽度 $2u_{max}$

即奈奎斯特采样定律： $\frac{1}{\Delta T} > 2u_{max}$ 

**频域恢复**

如果满足采样定律，我们可以通过一个理想低通滤波器 (盒状函数) $H(u)$ 来恢复 $F(u)$ 

定义 $H(u)$，其宽度覆盖一个周期 $[-\frac{1}{2\Delta T}, \frac{1}{2\Delta T}]$：
$$H(u) = \begin{cases} \Delta T & |u| \le \frac{1}{2\Delta T} \\ 0 & |u| > \frac{1}{2\Delta T} \end{cases}$$

(奈奎斯特定律保证了 $u_{max} < \frac{1}{2\Delta T}$，所以 $F(u)$ 的完整形态包含在 $H(u)$ 内)

通过 $F(u) = \overline{F}(u) H(u)$  进行恢复：

$F(u) = \left( \frac{1}{\Delta T} \sum_{n=-\infty}^{\infty} F(u - \frac{n}{\Delta T}) \right) H(u)$

$F(u) = \left( \frac{1}{\Delta T} F(u) \right) H(u) = \left( \frac{1}{\Delta T} F(u) \right) (\Delta T) = F(u)$


**时域恢复 (`SINC` 插值)**

从频域恢复 $F(u)$ 转换回时域 $f(t)$，计算 $\mathcal{F}^{-1}\{F(u)\}$：

$f(t) = \mathcal{F}^{-1}\{F(u)\} = \mathcal{F}^{-1}\{\overline{F}(u) H(u)\}$

根据连续卷积定理 $f(t) = \mathcal{F}^{-1}\{\overline{F}(u)\} * \mathcal{F}^{-1}\{H(u)\} = \overline{f}(t) * h(t)$

我们计算 $h(t) = \mathcal{F}^{-1}\{H(u)\}$：

$h(t) = \int_{-\infty}^{\infty} H(u) e^{j2\pi u t} du = \int_{-1/(2\Delta T)}^{1/(2\Delta T)} \Delta T e^{j2\pi u t} du$
$h(t) = \Delta T \left[ \frac{e^{j2\pi u t}}{j2\pi t} \right]_{-1/(2\Delta T)}^{1/(2\Delta T)} = \frac{\Delta T}{j2\pi t} \left( e^{j\pi t/\Delta T} - e^{-j\pi t/\Delta T} \right) = \frac{\Delta T}{j2\pi t} \left( 2j \sin(\pi t/\Delta T) \right) = \frac{\sin(\pi t/\Delta T)}{\pi t / \Delta T} = \text{sinc}\left(\frac{t}{\Delta T}\right)$
将 $h(t)$ 和 $\overline{f}(t)$ 代回卷积：$f(t) = \overline{f}(t) * h(t) = \left( \sum_{n=-\infty}^{\infty} f(n\Delta T) \delta(t-n\Delta T) \right) * \text{sinc}\left(\frac{t}{\Delta T}\right)$

利用 $\delta$ 函数的卷积特性 ($\delta(t-t_0) * h(t) = h(t-t_0)$):

$f(t) = \sum_{n=-\infty}^{\infty} f(n\Delta T) \text{sinc}\left(\frac{t - n\Delta T}{\Delta T}\right)$ 

##### (3) 细节说明

**频域信号的对称性**

实信号的傅里叶变换必须具有共轭对称性 (Conjugate Symmetry)：

$F(u) = F^*(-u)$

1. 幅度对称： $|F(u)| = |F^*(-u)| = |F(-u)|$
2. 相位反对称： $\phi(u) = -\phi(-u)$ 

如果信号在正频率 $u_{max}$ 处存在能量 (即 $|F(u_{max})| \ne 0$)，那么它必须在负频率 $-u_{max}$ 处也存在完全相等的能量 (即 $|F(-u_{max})| = |F(u_{max})|$)

* 进一步的一个性质：$f(t)$ 是实偶函数 $\implies$ $F(u)$ 是实偶函数


### 需要承认的定理

结论 1：$f(t)$有界, 则$\overline{F}(u)$会产生混淆 (也就是$F(u)$无界)，反之, $F(u)$有界 (带限), 则$f(t)$无界 

这个结论是傅里叶变换不确定性原理 (Uncertainty Principle) 的体现，即一个信号不能同时在时域和频域上都是有限的

- **$f(t)$ 有界 $\implies F(u)$ 无界 $\implies \overline{F}(u)$ 产生混淆**

    - 这里的“有界” (bounded) 指的是时限 (time-limited)，即 $f(t)$ 仅在一个有限的时间区间内非零，例如 $f(t)$ 是一个盒状函数
    - 不确定性原理指出：如果一个信号是时限的，那么它的傅里叶变换 $F(u)$ 必然是带通无限 (frequency-unlimited) 的，即 $F(u)$ 会延伸到无穷频率 (例如 $\text{sinc}$ 函数) 
    - $\overline{F}(u)$ 是 $F(u)$ 以 $\frac{1}{\Delta T}$ 为间隔的周期性拷贝。由于 $F(u)$ 本身是无限宽的，无论 $\frac{1}{\Delta T}$ 多大，这些无限宽的拷贝都必然会发生重叠，也就是混淆 (Aliasing)

- **反之, $F(u)$ 有界 (带限) $\implies f(t)$ 无界**

    - 如果一个信号是带限的 (band-limited)，即 $F(u)$ 仅在 $[-u_{max}, u_{max}]$ 区间内非零 
    - 那么它的傅里叶逆变换 $f(t)$ 必然是**时域无限 (time-unlimited)** 的 
    - 例如：$F(u)$ 是盒状函数 (带限)，则 $f(t)$ 是 $\text{sinc}$ 函数 (时域无限)


结论 2：当 $f(t)$ 其傅里叶变换 $F(u)$ 具有带限性质的时域信号时，$F(u)$ 的离散采样 $\{F_m\}$ 的逆变换一定是周期的，并可对一个周期采样 $\{f_n\}$，使得这两对采样相互决定 (一一对应) 

这个结论是离散傅里叶变换 (DFT) 存在的理论基础，它建立了 $\{f_n\}$ 和 $\{F_m\}$ 之间的桥梁

- **$F(u)$ 的离散采样 $\{F_m\}$ $\implies$ 逆变换结果是周期的**
    
    - 这与采样定理 (时域采样 $\to$ 频域周期) 具有对偶性
    - 采样定理：在时域进行采样 ($\overline{f}(t)$)，会导致频域 ($\overline{F}(u)$) 变为周期性
    - 对偶地：在频域进行采样 ($\{F_m\}$)，会导致时域 (其逆变换结果) 变为周期性 

- 两对采样相互决定 (一一对应)，存在以下事实
    
	1. 对带限的 $f(t)$ 进行时域采样，得到了 $\{f_n\}$ (根据奈奎斯特定律，$\overline{F}(u)$ 的拷贝不重叠)
	2. 对 $F(u)$ 在一个周期上进行频域采样，得到了 $\{F_m\}$，这对应一个周期的时域信号
	3. 有限的 $M$ 个时域样本 $\{f_n\}$ 与 频域样本 $\{F_m\}$ 间，存在一一对应 的关系 

### 离散傅里叶变换的定义与一一对应性质

1. 将 DFT (傅里叶变换) 写成一个矩阵 $A$    
2. 将 IDFT (逆傅里叶变换) 写成一个矩阵 $B$
3. 证明这两个矩阵互为逆矩阵，即 $A \times B = I$ (单位矩阵)
4. 如果一个变换存在逆变换 (逆矩阵)，那么它必定是一个一一对应的变换

- **DFT (1D-DFT):** $F_m = \sum_{n=0}^{M-1} f_n e^{-j2\pi mn/M}$
- 矩阵乘法：$\vec{F}_m = A \vec{f}_n$，  $A_{ij} = e^{-j2\pi ij/M}$

- **IDFT (1D-IDFT):** $f_n = \frac{1}{M} \sum_{m=0}^{M-1} F_m e^{j2\pi mn/M}$
- 矩阵乘法：$\vec{f}_n = B \vec{F}_m$，因此 $B_{ij} = \frac{1}{M} e^{j2\pi ij/M}$

为了简化书写，定义 $w = e^{j2\pi/M}$ (复平面上的 $M$ 次单位根)

- $A_{ij} = (e^{j2\pi/M})^{-ij} = w^{-ij}$
- $B_{ij} = \frac{1}{M} (e^{j2\pi/M})^{ij} = \frac{1}{M} w^{ij}$

想证明 $A$ 和 $B$ 的乘积  是单位矩阵 $I$

$C_{ij} = \sum_{k=0}^{M-1} A_{ik} B_{kj} = \sum_{k=0}^{M-1} \left( w^{-ik} \right) \left( \frac{1}{M} w^{kj} \right) = \frac{1}{M} \sum_{k=0}^{M-1} w^{-ik + kj} = \frac{1}{M} \sum_{k=0}^{M-1} w^{k(j-i)}$

- **情况 1：$i = j$ (矩阵的对角线元素)**
- $C_{ii} = \frac{1}{M} \sum_{k=0}^{M-1} w^{k(0)} = \frac{1}{M} \sum_{k=0}^{M-1} w^0 = \frac{1}{M} \sum_{k=0}^{M-1} 1 = 1$
- **情况 2：$i \neq j$ (矩阵的非对角线元素)**
- $C_{ij} = \frac{1}{M} \sum_{k=0}^{M-1} w^{kl} = \frac{1}{M} \sum_{k=0}^{M-1} (w^l)^k$
- $C_{ij} = \frac{1}{M} \left[ \frac{1 - (w^l)^M}{1 - w^l} \right]$
- $C_{ij} = \frac{1}{M} \left[ \frac{0}{1 - w^l} \right] = 0$

故 $C = A \times B$：
$C_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases} = I$

### 二维连续傅里叶变换与采样定理

- 傅里叶变换 (FT) 
$$F(u,v)=\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}f(t,z)e^{-j2\pi(ut+vz)}dtdz$$
- 傅里叶逆变换 (IFT) 
$$f(t,z)=\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}F(u,v)e^{j2\pi(ut+\nu z)}dudv$$

* 二维盒状函数 (2D Box Function) 的变换
$$f(t,z) = \begin{cases} A & -T/2 \le t \le T/2 \text{ and } -Z/2 \le z \le Z/2 \\ 0 & \text{else} \end{cases}$$

证明 $\mathcal{F}\{f(t,z)\} = ATZ[\frac{\sin(\pi\mu T)}{(\pi\mu T)}][\frac{\sin(\pi\nu Z)}{(\pi\nu Z)}]$

$\mathcal{F}\{f(t,z)\} = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty}f(t,z)e^{-j2\pi(\mu t+\nu z)}dt~dz$ 

$F(\mu,\nu) = \int_{-T/2}^{T/2}\int_{-Z/2}^{Z/2}Ae^{-j2\pi(\mu t+\nu z)}dt~dz$ 

由于积分核 $e^{-j2\pi(\mu t+\nu z)} = e^{-j2\pi\mu t} e^{-j2\pi\nu z} = A \left( \int_{-T/2}^{T/2} e^{-j2\pi\mu t} dt \right) \left( \int_{-Z/2}^{Z/2} e^{-j2\pi\nu z} dz \right)$ 

计算 $t$ 向的积分：$\int_{-T/2}^{T/2} e^{-j2\pi\mu t} dt = \left[ \frac{e^{-j2\pi\mu t}}{-j2\pi\mu} \right]_{-T/2}^{T/2} = \frac{e^{j\pi\mu T} - e^{-j\pi\mu T}}{j2\pi\mu}$

利用欧拉公式 $\sin(\theta) = \frac{e^{j\theta} - e^{-j\theta}}{2j} = \frac{2j \sin(\pi\mu T)}{j2\pi\mu} = \frac{\sin(\pi\mu T)}{\pi\mu}$：

$= T \left[ \frac{\sin(\pi\mu T)}{\pi\mu T} \right]$

同理，计算 $z$ 向的积分：$\int_{-Z/2}^{Z/2} e^{-j2\pi\nu z} dz = Z \left[ \frac{\sin(\pi\nu Z)}{\pi\nu Z} \right]$

将两部分积分结果代回 $F(\mu,\nu)$：

$F(\mu,\nu) = A \cdot T \left[ \frac{\sin(\pi\mu T)}{\pi\mu T} \right] \cdot Z \left[ \frac{\sin(\pi\nu Z)}{\pi\nu Z} \right]$

##### 二维采样定理 (2D Sampling Theorem)

- 二维冲击串 (Impulse Train) 
$$s_{\Delta T\Delta Z}(t,z)=\Sigma_{m=-\infty}^{\infty}\Sigma_{n=-\infty}^{\infty}\delta(t-m\Delta T,Z-n\Delta Z)$$    
- 采样函数 (Sampled Function) 
$$\overline{f}(t,z) = f(t,z) s_{\Delta T\Delta Z}(t,z)$$
* **(1) 获得冲击窜采样后函数的频域变换**

根据二维连续卷积定理：

$f(t,z)h(t,z) \iff F(\mu,\nu) * H(\mu,\nu)$

应用此定理，时域的乘积对应频域的卷积：

$\overline{F}(\mu,\nu) = \mathcal{F}\{f(t,z)\} * \mathcal{F}\{s_{\Delta T\Delta Z}(t,z)\} = F(\mu,\nu) * S(\mu,\nu)$

**步骤 1：** 求二维冲击串 $s_{\Delta T\Delta Z}(t,z)$ 的傅里叶变换 $S(\mu,\nu)$

$s_{\Delta T\Delta Z}(t,z)$ 是周期函数，可展开为二维傅里叶级数：

$s_{\Delta T\Delta Z}(t,z) = \sum_{k=-\infty}^{\infty} \sum_{l=-\infty}^{\infty} c_{kl} e^{j2\pi(\frac{kt}{\Delta T} + \frac{lz}{\Delta Z})}$

系数 $c_{kl}$ 为：

$c_{kl} = \frac{1}{\Delta T \Delta Z} \int_{-\Delta T/2}^{\Delta T/2} \int_{-\Delta Z/2}^{\Delta Z/2} s_{\Delta T\Delta Z}(t,z) e^{-j2\pi(\frac{kt}{\Delta T} + \frac{lz}{\Delta Z})} dt dz$

在一个周期内，$s_{\Delta T\Delta Z}(t,z) = \delta(t,z)$：

$c_{kl} = \frac{1}{\Delta T \Delta Z} \int_{-\Delta T/2}^{\Delta T/2} \int_{-\Delta Z/2}^{\Delta Z/2} \delta(t,z) e^{-j2\pi(\frac{kt}{\Delta T} + \frac{lz}{\Delta Z})} dt dz$

根据 $\delta$ 函数的二维取样特性：$c_{kl} = \frac{1}{\Delta T \Delta Z} e^0 = \frac{1}{\Delta T \Delta Z}$

代回傅里叶级数：$s_{\Delta T\Delta Z}(t,z) = \sum_{k=-\infty}^{\infty} \sum_{l=-\infty}^{\infty} \frac{1}{\Delta T \Delta Z} e^{j2\pi(\frac{kt}{\Delta T} + \frac{lz}{\Delta Z})}$

对上式两边取傅里叶变换：$S(\mu,\nu) = \mathcal{F}\{s_{\Delta T\Delta Z}(t,z)\} = \mathcal{F}\left\{ \sum_{k=-\infty}^{\infty} \sum_{l=-\infty}^{\infty} \frac{1}{\Delta T \Delta Z} e^{j2\pi(\frac{kt}{\Delta T} + \frac{lz}{\Delta Z})} \right\}$

$S(\mu,\nu) = \frac{1}{\Delta T \Delta Z} \sum_{k=-\infty}^{\infty} \sum_{l=-\infty}^{\infty} \mathcal{F}\left\{ e^{j2\pi(\frac{kt}{\Delta T} + \frac{lz}{\Delta Z})} \right\}$

利用变换对 $\mathcal{F}\{e^{j2\pi(\mu_0 t + \nu_0 z)}\} = \delta(\mu-\mu_0, \nu-\nu_0)$：

$S(\mu,\nu) = \frac{1}{\Delta T \Delta Z} \sum_{k=-\infty}^{\infty} \sum_{l=-\infty}^{\infty} \delta(\mu - \frac{k}{\Delta T}, \nu - \frac{l}{\Delta Z})$

**步骤 2：** 将 $S(\mu,\nu)$ 代回卷积表达式

$\overline{F}(\mu,\nu) = F(\mu,\nu) * S(\mu,\nu)$

$\overline{F}(\mu,\nu) = F(\mu,\nu) * \left( \frac{1}{\Delta T \Delta Z} \sum_{k=-\infty}^{\infty} \sum_{l=-\infty}^{\infty} \delta(\mu - \frac{k}{\Delta T}, \nu - \frac{l}{\Delta Z}) \right)$

利用卷积的线性性质：

$\overline{F}(\mu,\nu) = \frac{1}{\Delta T \Delta Z} \sum_{k=-\infty}^{\infty} \sum_{l=-\infty}^{\infty} \left( F(\mu,\nu) * \delta(\mu - \frac{k}{\Delta T}, \nu - \frac{l}{\Delta Z}) \right)$

利用 $\delta$ 函数的卷积特性 $F(\mu,\nu) * \delta(\mu-\mu_0, \nu-\nu_0) = F(\mu-\mu_0, \nu-\nu_0)$：

$\overline{F}(\mu,\nu) = \frac{1}{\Delta T \Delta Z} \sum_{k=-\infty}^{\infty} \sum_{l=-\infty}^{\infty} F(\mu - \frac{k}{\Delta T}, \nu - \frac{l}{\Delta Z})$


* **(2) 频域恢复 $F(\mu,\nu)$**

假设满足 2D 奈奎斯特定律 ($\frac{1}{\Delta T} > 2u_{max}, \frac{1}{\Delta Z} > 2v_{max}$)，$\overline{F}(\mu,\nu)$ 的拷贝不发生混叠

我们定义一个 2D 盒状函数 $H(\mu,\nu)$ 作为理想低通滤波器：

$$H(\mu,\nu) = \begin{cases} \Delta T \Delta Z & |\mu| \le \frac{1}{2\Delta T} \text{ and } |\nu| \le \frac{1}{2\Delta Z} \\ 0 & \text{else} \end{cases}$$

(高度设为 $\Delta T \Delta Z$ 以抵消 $\overline{F}$ 表达式中的 $\frac{1}{\Delta T \Delta Z}$ 因子)

用 $H(\mu,\nu)$ 乘以 $\overline{F}(\mu,\nu)$ 来恢复 $F(\mu,\nu)$：

$F(\mu,\nu) = \overline{F}(\mu,\nu) H(\mu,\nu)$

$F(\mu,\nu) = \left( \frac{1}{\Delta T \Delta Z} \sum_{k=-\infty}^{\infty} \sum_{l=-\infty}^{\infty} F(\mu - \frac{k}{\Delta T}, \nu - \frac{l}{\Delta Z}) \right) H(\mu,\nu)$

$H(\mu,\nu)$ 仅在 $|\mu| \le \frac{1}{2\Delta T}, |\nu| \le \frac{1}{2\Delta Z}$ 内非零，只有 $k=0, l=0$ 的中央拷贝项被保留：

$F(\mu,\nu) = \left( \frac{1}{\Delta T \Delta Z} F(\mu, \nu) \right) (\Delta T \Delta Z) = F(\mu,\nu)$


**(3) 时域恢复 $f(t,z)$ (Sinc 插值)**

对 $F(\mu,\nu) = \overline{F}(\mu,\nu) H(\mu,\nu)$ 两边取傅里叶逆变换 (IFT)：

$f(t,z) = \mathcal{F}^{-1}\{F(\mu,\nu)\} = \mathcal{F}^{-1}\{\overline{F}(\mu,\nu) H(\mu,\nu)\}$

根据二维连续卷积定理 ：$f(t,z) = \mathcal{F}^{-1}\{\overline{F}(\mu,\nu)\} * \mathcal{F}^{-1}\{H(\mu,\nu)\}$

我们知道 $\mathcal{F}^{-1}\{\overline{F}(\mu,\nu)\} = \overline{f}(t,z)$，即原始的采样函数：

$\overline{f}(t,z) = f(t,z) s_{\Delta T\Delta Z}(t,z) = \sum_{k=-\infty}^{\infty} \sum_{l=-\infty}^{\infty} f(k\Delta T, l\Delta Z) \delta(t-k\Delta T, z-l\Delta Z)$

计算 $h(t,z) = \mathcal{F}^{-1}\{H(\mu,\nu)\}$ (盒状函数的 IFT)：

$h(t,z) = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} H(\mu,\nu) e^{j2\pi(\mu t + \nu z)} d\mu d\nu$

$h(t,z) = \int_{-1/(2\Delta T)}^{1/(2\Delta T)} \int_{-1/(2\Delta Z)}^{1/(2\Delta Z)} (\Delta T \Delta Z) e^{j2\pi\mu t} e^{j2\pi\nu z} d\mu d\nu$

$h(t,z) = (\Delta T \Delta Z) \left[ \int_{-1/(2\Delta T)}^{1/(2\Delta T)} e^{j2\pi\mu t} d\mu \right] \left[ \int_{-1/(2\Delta Z)}^{1/(2\Delta Z)} e^{j2\pi\nu z} d\nu \right]$

计算 $\mu$ 向积分：

$\int_{-1/(2\Delta T)}^{1/(2\Delta T)} e^{j2\pi\mu t} d\mu = \left[ \frac{e^{j2\pi\mu t}}{j2\pi t} \right]_{-1/(2\Delta T)}^{1/(2\Delta T)} = \frac{e^{j\pi t/\Delta T} - e^{-j\pi t/\Delta T}}{j2\pi t}$

$= \frac{2j \sin(\pi t/\Delta T)}{j2\pi t} = \frac{\sin(\pi t/\Delta T)}{\pi t}$

将 $\Delta T$ 乘回：

$\Delta T \left( \frac{\sin(\pi t/\Delta T)}{\pi t} \right) = \frac{\sin(\pi t/\Delta T)}{\pi t / \Delta T} = \text{sinc}\left(\frac{t}{\Delta T}\right)$

同理，$\nu$ 向积分乘以 $\Delta Z$ 得到 $\text{sinc}\left(\frac{z}{\Delta Z}\right)$

因此 $h(t,z) = \text{sinc}\left(\frac{t}{\Delta T}\right) \text{sinc}\left(\frac{z}{\Delta Z}\right)$

代回时域卷积

$f(t,z) = \overline{f}(t,z) * h(t,z)$

$f(t,z) = \left( \sum_{k,l} f(k\Delta T, l\Delta Z) \delta(t-k\Delta T, z-l\Delta Z) \right) * \left( \text{sinc}\left(\frac{t}{\Delta T}\right) \text{sinc}\left(\frac{z}{\Delta Z}\right) \right)$

利用 $\delta$ 函数的卷积特性 ($\delta(t-t_0, z-z_0) * h(t,z) = h(t-t_0, z-z_0)$)：

$f(t,z) = \sum_{k=-\infty}^{\infty} \sum_{l=-\infty}^{\infty} f(k\Delta T, l\Delta Z) \text{sinc}\left(\frac{t-k\Delta T}{\Delta T}\right) \text{sinc}\left(\frac{z-l\Delta Z}{\Delta Z}\right)$


### 二维离散傅里叶变换

- **2D-DFT
$$F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})}$$

- **2D-IDFT**
$$f(x,y)=\frac{1}{MN}\sum_{u=0}^{M-1}\sum_{v=0}^{N-1}F(u,v)e^{j2\pi(\frac{ux}{M}+\frac{vy}{N})}$$

- **二维循环卷积定义** 
$$f(x,y)*h(x,y)=\sum_{m=0}^{M-1}\sum_{n=0}^{N-1}f(m,n)h(x-m,y-n)$$

* **2D-DFT 的性质**

- **① 平移 (Translation)**
    
    - **时域平移** ：$f(x-x_{0},y-y_{0})\Leftrightarrow F(u,v)e^{-j2\pi(\frac{x_{0}u}{M}+\frac{y_{0}v}{N})}$
	- $\mathcal{F}\{f(x-x_0, y-y_0)\} = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} f(x-x_0, y-y_0) e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})}$
	- 令 $p = x-x_0$，$q = y-y_0$，则 $x = p+x_0$，$y = q+y_0$：
	- $= e^{-j2\pi(\frac{ux_0}{M}+\frac{vy_0}{N})} \left( \sum_{p=0}^{M-1}\sum_{q=0}^{N-1} f(p,q) e^{-j2\pi(\frac{up}{M}+\frac{vq}{N})} \right)$
	- $= F(u,v) e^{-j2\pi(\frac{x_0 u}{M}+\frac{y_0 v}{N})}$
            
    - **频域平移** ：$f(x,y)e^{j2\pi(\frac{u_{0}x}{M}+\frac{v_{0}y}{N})}\Leftrightarrow F(u-u_{0},v-v_{0})$
	- $\mathcal{F}\{f(x,y) e^{j2\pi(\dots)}\} = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} f(x,y) e^{j2\pi(\frac{u_0 x}{M}+\frac{v_0 y}{N})} e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})}$
	- $= \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} f(x,y) e^{-j2\pi \left( \frac{(u-u_0)x}{M} + \frac{(v-v_0)y}{N} \right)}$
	- $= F(u-u_0, v-v_0)$

- ② 中心化 (Centering): $f(x,y)(-1)^{x+y}\Leftrightarrow F(u-M/2,v-N/2)$

    - $(-1)^{x+y} = (-1)^x (-1)^y$
    - 根据欧拉公式 $e^{j\pi} = -1$，则 $e^{j\pi x} = (-1)^x$ 且 $e^{j\pi y} = (-1)^y$
    - $e^{j\pi x} e^{j\pi y} = e^{j\pi(x+y)}$

    - 我们需要 $e^{j2\pi(\frac{u_{0}x}{M}+\frac{v_{0}y}{N})}$ 的形式 
    - $e^{j\pi(x+y)} = e^{j2\pi(\frac{x}{2} + \frac{y}{2})} = e^{j2\pi(\frac{(M/2)x}{M} + \frac{(N/2)y}{N})}$ (假设 M, N 为偶数)
    - 这符合频域平移的形式 ，其中 $u_0 = M/2$，$v_0 = N/2$

    - $\mathcal{F}\{f(x,y)(-1)^{x+y}\} = \mathcal{F}\{f(x,y) e^{j2\pi(\frac{(M/2)x}{M} + \frac{(N/2)y}{N})}\}$
    - $= F(u - M/2, v - N/2)$

- ③ 周期性 (Periodicity): $F(u,v) = F(u+k_1 M, v+k_2 N)$

    - $F(u+k_1 M, v+k_2 N) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)e^{-j2\pi(\frac{(u+k_1 M)x}{M}+\frac{(v+k_2 N)y}{N})}$
    - $= \sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y) e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})} e^{-j2\pi(k_1 x + k_2 y)}$
    - 根据欧拉公式 $e^{-j2\pi(\text{integer})} = \cos(2\pi \cdot \text{integer}) - j\sin(2\pi \cdot \text{integer}) = 1 - 0 = 1$
    - $= \sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})}$
    - $= F(u,v)$，证毕 ($f(x,y)$ 的周期性同理可证)

- ④ 对称性 (Symmetry) (实函数): 若 $f(x,y)$ 为实函数，则 $F^{*}(u,v)=F(-u,-v)$

    - $F(u,v) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})}$
    - $F^*(u,v) = \left( \sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})} \right)^*$
    - $= \sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f^*(x,y) \left( e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})} \right)^*$
    - 因为 $f(x,y)$ 是实函数，$f^*(x,y) = f(x,y)$
    - $F^*(u,v) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y) e^{+j2\pi(\frac{ux}{M}+\frac{vy}{N})}$

    - 现在计算 $F(-u, -v)$：
    - $F(-u,-v) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)e^{-j2\pi(\frac{(-u)x}{M}+\frac{(-v)y}{N})}$
	* $F^*(u,v) = F(-u,-v)$，证毕
    
    - 这意味着幅度对称 $|F(u,v)| = |F(-u,-v)|$ ，相角反对称 $\phi(u,v) = -\phi(-u,-v)$ 