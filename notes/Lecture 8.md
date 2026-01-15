### 频率域滤波 5 步骤

1. **填充 (Padding)**

    - 给定大小为 $M\times N$ 的图像 $f(x,y)$，填充 (padding) 图像为 $P\times Q$ (通常 $P \ge 2M, Q \ge 2N$)，得到 $f_p(x,y)$
    - 这一步是为了防止缠绕错误 (Wraparound Error)，这是 DFT 使用的循环卷积 (circular convolution) 相对于标准线性卷积的副作用

2. **中心化 (Centering) 与 DFT**

    - 用 $(-1)^{x+y}$ 乘以填充后的 $f_p(x,y)$
    - $f_p(x,y)(-1)^{x+y}$
    - 这一步是为了将频域的原点 $(0,0)$ 从图像的角落平移到中心 $(P/2, Q/2)$ (根据 $f(x,y)(-1)^{x+y}\Leftrightarrow F(u-P/2,v-Q/2)$)
    - 对中心化后的图像计算 DFT，得到中心化的傅里叶变换 $F(u,v)$

3. **滤波 (Filtering)**

    - 生成一个大小为 $P\times Q$、实对称且中心化的滤波函数 $H(u,v)$
    - 在频域中，根据卷积定理 $f*h \iff FH$，将图像的频谱 $F(u,v)$ 与滤波器 $H(u,v)$ 逐点相乘
    - $G(u,v)=H(u,v)F(u,v)$

4. **反变换与反中心化**
    
    - 计算 $G(u,v)$ 的傅里叶逆变换 (IDFT)，得到 $g_p'(x,y) = \mathcal{F}^{-1}[G(u,v)]$
    - 再次乘以 $(-1)^{x+y}$ 来撤销步骤 2 中的中心化操作
    - $g_p(x,y) = \mathcal{F}^{-1}[G(u,v)] (-1)^{x+y}$
    - 由于输入 $f(x,y)$ 是实函数，计算中可能产生微小的虚部，因此取其实部
    - $g_p(x,y) = \{\text{real}[\mathcal{F}^{-1}[G(u,v)]]\}(-1)^{x+y}$

5. **提取 (Cropping)**

    - 从 $g_p(x,y)$ 的左上象限提取出 $M\times N$ 大小的图像
    - 这对应于步骤 1 中填充的逆操作，得到最终的滤波后图像 $g(x,y)$


### 频域滤波器 - 平滑

平滑操作对应低通滤波 (Lowpass Filtering)，其目的是抑制高频分量 (如噪声、边缘)，保留低频分量 (如图像的平缓背景)

**1. 理想低通滤波器 (ILPF)**

- 定义： 理想低通滤波器是一个频域的“盒状函数”或“圆柱体”，它无损地通过 (乘以 1) 所有 $D_0$ 半径内的频率，并完全阻断 (乘以 0) $D_0$ 之外的所有频率

- 公式：
$$H(u,v)=\begin{cases}1,&D(u,v)\le D_{0}\\ 0,&D(u,v)>D_{0}\end{cases}$$
* 其中 $D(u,v)$ 是到频域中心 (P/2, Q/2) 的距离：$$D(u,v)=[(u-P/2)^{2}+(v-Q/2)^{2}]^{1/2}$$
- 问题 (振铃效应 Ringing Effect)：

    - 频域的盒状函数 $H(u,v)$ (ILPF)
    - 根据傅里叶变换的对偶性，对应的时域函数 $h(x,y) = \mathcal{F}^{-1}\{H(u,v)\}$ 是Sinc 函数
    - Sinc 函数具有**旁瓣 (sidelobes)**，即在主峰值周围有波动的振荡
    - 频域相乘 $G=FH$ 等价于时域卷积 $g=f*h$，这个卷积 $f * \text{sinc}$ 会将 Sinc 函数的振荡 (振铃) 引入到最终的图像 $g(x,y)$ 中，在边缘附近产生重影


**2. 高斯低通滤波器 (GLPF)**

- 定义： 为避免振铃效应，我们使用在频域和时域都平滑的高斯函数
- 公式： $$H(u,v)=e^{-D^{2}(u,v)/2\sigma^{2}}$$
* 其中 $D(u,v)$ 是到频域中心的距离，$\sigma$ 控制了高斯函数的宽度 ( 截止频率 )

- 高斯函数的傅里叶变换： (证明高斯 $\iff$ 高斯)

1. 已知 1D 高斯函数的傅里叶变换 (FT)

根据我们之前的推导，1D 高斯函数 $f(t) = e^{-at^2}$ 的傅里叶变换为：
$$F(u) = \mathcal{F}\{e^{-at^2}\} = \sqrt{\frac{\pi}{a}} e^{-\frac{\pi^2}{a} u^2}$$
2. 求解 1D 高斯函数的傅里叶逆变换 (IFT)

我们的目标是求 $h(x) = \mathcal{F}^{-1}\{H(u)\}$，其中 $H(u)$ 是 1D 高斯低通滤波器：

$H(u) = e^{-u^2 / (2\sigma^2)}$

我们利用步骤 1 的变换对，通过傅里叶逆变换的定义 $\mathcal{F}^{-1}\{F(u)\} = f(t)$ 来求解

$\mathcal{F}^{-1}\left\{ \sqrt{\frac{\pi}{a}} e^{-\frac{\pi^2}{a} u^2} \right\} = e^{-at^2}$

根据线性性质，将常数 $\sqrt{\frac{\pi}{a}}$ 移到另一侧：$\mathcal{F}^{-1}\left\{ e^{-\frac{\pi^2}{a} u^2} \right\} = \frac{1}{\sqrt{\pi/a}} e^{-at^2} = \sqrt{\frac{a}{\pi}} e^{-at^2}$

现在，我们将 $H(u) = e^{-u^2 / (2\sigma^2)}$ 与 $\mathcal{F}^{-1}$ 中的 $e^{-\frac{\pi^2}{a} u^2}$ 进行匹配：

$e^{-u^2 / (2\sigma^2)} = e^{-\frac{\pi^2}{a} u^2}$，通过比较指数项，可得：$\frac{u^2}{2\sigma^2} = \frac{\pi^2 u^2}{a}$

解得 $a = 2\pi^2\sigma^2$

我们将 $a = 2\pi^2\sigma^2$ 代入 $\mathcal{F}^{-1}$ 的结果表达式 $\sqrt{\frac{a}{\pi}} e^{-at^2}$ 中 (并将 $t$ 替换为 $x$)：

$h(x) = \mathcal{F}^{-1}\{ e^{-u^2 / (2\sigma^2)} \} = \sqrt{\frac{2\pi^2\sigma^2}{\pi}} e^{-(2\pi^2\sigma^2) x^2} = \sqrt{2\pi\sigma^2} e^{-2\pi^2\sigma^2 x^2}$

这是一个 $K \cdot e^{-C x^2}$ 的形式，证明了 1D 高斯滤波器的时域核函数 $h(x)$ 仍然是高斯函数

3. 扩展到 2D (Separability)

2D 高斯低通滤波器 $H(u,v)$ (为简化推导，暂不考虑中心平移) 为：

$H(u,v) = e^{-D^2(u,v)/2\sigma^2} = e^{-(u^2+v^2)/2\sigma^2}$

这个 2D 函数是可分离 (Separable) 的：

$H(u,v) = \left( e^{-u^2/2\sigma^2} \right) \left( e^{-v^2/2\sigma^2} \right) = H(u) \cdot H(v)$

由于 2D 傅里叶变换是可分离的，其逆变换也是可分离的，2D IFT 等于两个 1D IFT 的乘积：

$h(x,y) = \mathcal{F}^{-1}\{H(u,v)\} = \mathcal{F}^{-1}\{H(u)\} \cdot \mathcal{F}^{-1}\{H(v)\} = h(x) \cdot h(y)$

将步骤 2 中 $h(x)$ 的结果代入：

$h(x,y) = \left( \sqrt{2\pi\sigma^2} e^{-2\pi^2\sigma^2 x^2} \right) \cdot \left( \sqrt{2\pi\sigma^2} e^{-2\pi^2\sigma^2 y^2} \right) = 2\pi\sigma^2 e^{-2\pi^2\sigma^2 (x^2+y^2)}$

4. 结论：$h(x,y)$ 仍是一个 2D 高斯函数，高斯低通滤波器的傅里叶逆变换仍然是高斯函数


- 优势：时域核函数 $h(x,y)$ (高斯) 是平滑的，没有旁瓣，因此 $g=f*h$ 不会产生振铃效应


**3. 高通滤波器**

* 高通滤波器 = 1 - 低通滤波器

### 频域滤波器 - 锐化

##### 导数的傅里叶变换

**Approach 1** 分部积分法

$\mathcal{F}(\frac{\partial}{\partial x}f(x)) = \int_{-\infty}^{\infty} \frac{\partial}{\partial x}f(x) e^{-j2\pi ux} dx = \int_{-\infty}^{\infty} e^{-j2\pi ux} d(f(x))$ 

使用分部积分法 $\int u dv = uv - \int v du$：

$\int_{-\infty}^{\infty} e^{-j2\pi ux} d(f(x)) = \left[ e^{-j2\pi ux}f(x) \right]_{-\infty}^{\infty} - \int_{-\infty}^{\infty} f(x) d(e^{-j2\pi ux})$ 

因为 $f(x)$ 在 $\pm\infty$ 处必须为 $0$ (zero padding) ，且 $|e^{-j2\pi ux}|=1$ (有界) ，第一项 $\left[ \dots \right]_{-\infty}^{\infty}$ 为 0

$\mathcal{F}(\frac{\partial}{\partial x}f(x)) = - \int_{-\infty}^{\infty} f(x) d(e^{-j2\pi ux}) = - \int_{-\infty}^{\infty} f(x) (-j2\pi u) e^{-j2\pi ux} dx$ 

$= j2\pi u \int_{-\infty}^{\infty} f(x) e^{-j2\pi ux} dx = j2\pi u F(u)$   

**Approach 2** 反函数法

该方法从傅里叶逆变换 (IFT) 的定义出发 $f(x) = \int_{-\infty}^{\infty} F(u) e^{j2\pi ux} du$ 

对 $f(x)$ 两边求 $x$ 的偏导数：$\frac{\partial}{\partial x}f(x) = \frac{\partial}{\partial x} \int_{-\infty}^{\infty} F(u) e^{j2\pi ux} du$ 

将求导运算移入积分号内 (因为 $F(u)$ 与 $x$ 无关)：

$= \int_{-\infty}^{\infty} \frac{\partial}{\partial x} [F(u) e^{j2\pi ux}] du = \int_{-\infty}^{\infty} F(u) (j2\pi u) e^{j2\pi ux} du$ 

我们观察 $\frac{\partial}{\partial x}f(x) = \int_{-\infty}^{\infty} [(j2\pi u)F(u)] e^{j2\pi ux} du$，表明 $\frac{\partial}{\partial x}f(x)$ 是 $(j2\pi u)F(u)$ 的傅里叶逆变换

因此，$\mathcal{F}(\frac{\partial}{\partial x}f(x)) = (j2\pi u)F(u)$
	
##### 频率域的拉普拉斯 (Laplacian) 算子

时域定义：$\nabla^{2}f(x,y)=\frac{\partial^{2}f}{\partial x^{2}}+\frac{\partial^{2}f}{\partial y^{2}}$ 

频域推导：对 $\nabla^{2}f$ 取傅里叶变换，并应用 1D 导数结论 $\mathcal{F}\{\frac{\partial f}{\partial x}\} = j2\pi u F(u)$：

$\mathcal{F}\{\frac{\partial^{2}f}{\partial x^{2}}\} = j2\pi u \mathcal{F}\{\frac{\partial f}{\partial x}\} = (j2\pi u)(j2\pi u)F(u,v) = -4\pi^2 u^2 F(u,v)$

$\mathcal{F}\{\frac{\partial^{2}f}{\partial y^{2}}\} = j2\pi v \mathcal{F}\{\frac{\partial f}{\partial y}\} = (j2\pi v)(j2\pi v)F(u,v) = -4\pi^2 v^2 F(u,v)$

$\mathcal{F}\{\nabla^{2}f(x,y)\} = \mathcal{F}\{\frac{\partial^{2}f}{\partial x^{2}} + \frac{\partial^{2}f}{\partial y^{2}}\} = (-4\pi^2 u^2 - 4\pi^2 v^2) F(u,v)$

因此，拉普拉斯滤波器的频域表达式为：$H_{Lap}(u,v) = -4\pi^{2}(u^{2}+v^{2})$ 

时空域中的拉普拉斯滤波操作为：$\nabla^{2}f(x,y) = \mathcal{F}^{-1}\{H_{Lap}(u,v)F(u,v)\}$ 


##### 使用拉普拉斯算子锐化图像

1. 图像锐化
$$g(x, y) = f(x, y) - c\nabla^{2}f(x, y)$$
锐化操作可以直接在频域通过以下滤波器实现
$$g(x, y) = \mathcal{F}^{-1}\{[1 - cH_{Lap}(u, v)]F(u, v)\}$$

2. 高频强调滤波 (High Boosting)

第一步：获取低频分量，使用低通滤波器对图像进行平滑处理，得到模糊图像
$$f_{LP}(x, y) = \mathcal{F}^{-1}\{H_{LP}(u, v)F(u, v)\}$$

第二步：获取高频掩膜，用原图减去平滑后的低频图像，得到包含边缘和细节的高频掩膜
$$g_{mask}(x, y) = f(x, y) - f_{LP}(x, y)$$
在频域中，这对应于用 1 - 低通滤波器，从而得到高通滤波器
$$g_{mask}(x, y) = \mathcal{F}^{-1}\{(1 - H_{LP}(u, v))F(u, v)\} = \mathcal{F}^{-1}\{H_{HP}(u, v)F(u, v)\}$$

第三步：加权增强与合成
$$g(x, y) = f(x, y) + k \cdot g_{mask}(x, y)$$

整个过程可以合并为一个频域滤波器表达式
$$g(x, y) = \mathcal{F}^{-1}\{[1 + k(1 - H_{LP}(u, v))]F(u, v)\}$$
或者写作
$$g(x, y) = \mathcal{F}^{-1}\{[1 + kH_{HP}(u, v)]F(u, v)\}$$


### 图像复原：去噪

* 图像增强 v.s. 复原
* 退化模型和噪声模型


