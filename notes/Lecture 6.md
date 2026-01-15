### 傅里叶级数 频域变换与逆变换

傅里叶级数是一种将周期性函数 $f(t)$（周期为 $T$）分解为一系列不同频率的正弦和余弦函数（或等价的复指数函数）的无穷加权和的方法。

* **复指数形式**：
$$    f(t)=\sum_{n=-\infty}^{\infty}c_{n}e^{j\frac{2\pi n}{T}t}  $$
   
* **傅里叶系数 $c_n$**：$$c_{n}=\frac{1}{T}\int_{-T/2}^{T/2}f(t)e^{-j\frac{2\pi n}{T}t}dt$$
##### 频域变换 (Fourier Transform, FT)

频域变换（或傅里叶变换）是将傅里叶级数从周期函数推广到非周期函数（可视为周期 $T \to \infty$）的工具，它将一个在时空域 (Spatio-temporal Domain) 的连续函数 $f(t)$ 变换为频域 (Frequency Domain) 函数 $F(u)$

* **变换公式**：
$$    F(u) = \mathcal{F}\{f(t)\} = \int_{-\infty}^{\infty}f(t)e^{-j2\pi ut}dt  $$
   
##### 频域逆变换 (Inverse Fourier Transform, IFT)

频域逆变换是将频域函数 $F(u)$ 重新转换回时空域函数 $f(t)$ 的过程，傅里叶变换和逆变换构成一个可逆且唯一的“频域变换对”

* **逆变换公式**：
$$    f(t) = \mathcal{F}^{-1}\{F(u)\} = \int_{-\infty}^{\infty}F(u)e^{j2\pi ut}du  $$

##### 傅里叶反演定理 (Fourier Inversion Theorem)

**证明目标：** 证明 $\mathcal{F}^{-1}\{\mathcal{F}\{f(t)\}\} = f(t)$

$\mathcal{F}^{-1}\{\mathcal{F}\{f(t)\}\} = \int_{\mu=-\infty}^{\infty} F(\mu) e^{j2\pi \mu t} d\mu$

其中 $F(\mu) = \int_{\tau=-\infty}^{\infty} f(\tau) e^{-j2\pi \mu \tau} d\tau$

代入 IFT 表达式：

$\mathcal{F}^{-1}\{\mathcal{F}\{f(t)\}\} = \int_{\mu=-\infty}^{\infty} \left[ \int_{\tau=-\infty}^{\infty} f(\tau) e^{-j2\pi \mu \tau} d\tau \right] e^{j2\pi \mu t} d\mu$

交换积分顺序 (先对 $\mu$ 积分，再对 $\tau$ 积分)：

$= \int_{\tau=-\infty}^{\infty} f(\tau) \left[ \int_{\mu=-\infty}^{\infty} e^{-j2\pi \mu \tau} e^{j2\pi \mu t} d\mu \right] d\tau$

$= \int_{\tau=-\infty}^{\infty} f(\tau) \left[ \int_{\mu=-\infty}^{\infty} e^{j2\pi \mu (t - \tau)} d\mu \right] d\tau$

**关键步骤：** 识别括号内的积分

$\delta(x) = \mathcal{F}^{-1}{\{1\}} = \int_{-\infty}^{\infty} e^{j2\pi u x} du$

我们将 $k$ 替换为 $\mu$，$x$ 替换为 $(t - \tau)$，得到：

$\int_{\mu=-\infty}^{\infty} e^{j2\pi \mu (t - \tau)} d\mu = \delta(t - \tau)$

$= \int_{\tau=-\infty}^{\infty} f(\tau) \delta(t - \tau) d\tau = f(t)$

结论：$\mathcal{F}^{-1}\{\mathcal{F}\{f(t)\}\} = f(t)$ 证毕，同理可证 $\mathcal{F}\{\mathcal{F}^{-1}\{F(\mu)\}\} = F(\mu)$


##### 频域变换对图像处理的直觉

* **空间域 (Spatial Domain) $f(x,y)$**：$f(x,y)$ 描述的是图像在**空间坐标** $(x,y)$ 上的强度值

* **频域 (Frequency Domain) $F(u,v)$**：这是对图像 $f(x,y)$ 执行二维傅里叶变换后得到的维度，$F(u,v)$ 描述的是图像由哪些频率成分构成的
    * $(u,v)$：代表频率，$u$ 表示图像在 x 方向（水平）变化的频率，$v$ 表示图像在 y 方向（垂直）变化的频率
    * $F(u,v)$ 回答的问题是：整幅图像中，*频率为 (u, v) 的模式*（即特定方向和变化速率的模式）*占有多大比重*？

* **图像频率的含义**
    * **低频 (Low Frequency)**：$(u,v)$ 坐标值小，靠近频域图的中心 $(0,0)$，它们代表图像中缓慢变化的区域，如大片平滑的表面、天空、物体的基本轮廓和主体结构，频域中心的 $F(0,0)$ 点被称为“DC分量”，代表了整幅图像的平均亮度
    * **高频 (High Frequency)**：$(u,v)$ 坐标值大，远离频域图的中心，它们代表图像中剧烈变化的区域，如物体的边缘、精细的纹理、细节和噪声

* **核心应用：滤波 (Filtering)**
    频域处理的核心优势在于卷积定理 (Convolution Theorem)
    1.  空间域操作：对图像 $f(x,y)$ 进行模糊或锐化，需要执行一次计算量庞大的卷积操作：$g(x,y) = f(x,y) * h(x,y)$。
    2.  频域操作：根据卷积定理，空间域的卷积等价于频域中对应频谱的逐元素乘法：$G(u,v) = F(u,v) H(u,v)$。
    3.  流程：
        * 分解：将 $f(x,y)$ 变换到频域，得到 $F(u,v)$
        * 编辑：设计一个频域滤波器（掩码）$H(u,v)$，执行简单的乘法 $G(u,v) = F(u,v) H(u,v)$
        * 重建：将处理后的 $G(u,v)$ 通过逆变换 $\mathcal{F}^{-1}$ 合成回空间域图像 $g(x,y)$

### 频域变换对与常用变换

* **频域 (Frequency Domain)**
    * 变量 $u$ 决定了正弦/余弦项的频率，因此 $F(u)$ 所在的域被称为频域
    * $F(u)$ 通常是一个复函数，可表示为 $F(u) = R(F(u)) + j \cdot I(F(u))$
    * 频谱 (Frequency Spectrum)：$|F(u)|$ 是 $F(u)$ 的幅度
$$        |F(u)| = \sqrt{R(F(u))^2 + I(F(u))^2} \quad \text{}      $$
    * 相位 (Phase Angle)：$\theta(u)$ 是 $F(u)$ 的相角
$$       \theta(u) = \arctan\left[\frac{I(F(u))}{R(F(u))}\right] \quad \text{}     $$
* 时空域 (Spatio-temporal Domain)：原始函数 $f(t)$（或 $f(x,y)$）所在的域被称为时空域

##### 常用变换 (Commonly Used Transforms)

以下是一些基本函数及其傅里叶变换的结果

* **冲激函数 (Impulse Function, $\delta(t)$)**
    * 定义：一个在 $t=0$ 处为无穷大，在其他位置为0的函数，其积分为 1
$$   \delta(t)=\begin{cases}\infty&if~t=0\\ 0&if~t\ne0\end{cases} \quad \text{and} \quad \int_{-\infty}^{\infty}\delta(t)dt=1 \quad \text{}    $$
    * 取样特性 (Sifting Property)：冲激函数它能筛选出函数在某一点的值
$$        \int_{-\infty}^{\infty}f(t)\delta(t-t_{0})dt=f(t_{0}) \quad \text{}      $$
* **变换对**：
	1.  时域冲激：位于 $t_0$ 的冲激函数，其傅里叶变换是一个复指数函数（在频域中振荡）
$$          \mathcal{F}\{\delta(t-t_{0})\} = e^{-j2\pi ut_0} \quad \text{}         $$
	2.  频域冲激：一个恒为1的常数函数，其傅里叶变换是一个位于 $u=0$ 的冲激函数
$$            \mathcal{F}\{f(t)=1\} = \delta(u) \quad \text{}          $$
	3.  根据 $\delta$ 函数的取样特性， $f(t) = \int_{-\infty}^{\infty} \delta(u - u_0) e^{j2\pi u t} du = e^{j2\pi u_0 t}$ 比较 IFT 定义 $f(t) = \int_{-\infty}^{\infty} F(u) e^{j2\pi u t} du$，我们得到变换对：
$$\mathcal{F}\{e^{j2\pi u_0 t}\} = \delta(u - u_0)$$
	* 考虑冲激函数的傅里叶逆变换
	
* **冲激串 (Impulse Train, $S_{\Delta T}(t)$)**
    * 定义：一系列间隔为 $\Delta T$ 的冲激函数的总和
	$$  S_{\Delta T}(t)=\sum_{k=-\infty}^{\infty}\delta(t-k\Delta T) \quad \text{}  $$
    * 变换对：时域中的冲激串，其傅里叶变换在频域中也是一个冲激串。
$$        \mathcal{F}\{S_{\Delta T}(t)\} = \frac{1}{\Delta T}\sum_{n=-\infty}^{\infty}\delta(u-\frac{n}{\Delta T}) \quad \text{}      $$
* **盒状函数 (Box Function)**
    * **定义**：一个在 $[-W/2, W/2]$ 区间内值为 $A$，区间外为 0 的矩形脉冲
    * **变换对**：时域中的盒状函数（矩形）对应频域中的 `sinc` 函数（一个中央峰值高、两侧振荡衰减的函数）

##### 证明：周期性冲激串 (Impulse Train) $S_{\Delta T}(t)$ 的傅里叶变换 (FT)

* 见作业


##### 证明：时域盒状函数 (Box Function) $\to$ 频域 Sinc 函数

盒状函数 $f(t)$ 在 $[-W/2, W/2]$ 区间内值为 $A$，区间外为 $0$：
$$f(t) = \begin{cases} A & |t| \le W/2 \\ 0 & |t| > W/2 \end{cases}$$
$f(t)$ 的傅里叶变换 $F(\mu)$：
$$F(\mu) = \int_{-\infty}^{\infty} f(t) e^{-j2\pi \mu t} dt = \int_{-W/2}^{W/2} A e^{-j2\pi \mu t} dt$$
$$F(\mu) = A \left[ \frac{e^{-j2\pi \mu t}}{-j2\pi \mu} \right]_{-W/2}^{W/2}$$
代入积分上下限：
$$F(\mu) = \frac{A}{-j2\pi \mu} \left( e^{-j2\pi \mu (W/2)} - e^{-j2\pi \mu (-W/2)} \right)$$
整理指数项：
$$F(\mu) = \frac{A}{-j2\pi \mu} \left( e^{-j\pi \mu W} - e^{+j\pi \mu W} \right) = \frac{A}{j2\pi \mu} \left( e^{+j\pi \mu W} - e^{-j\pi \mu W} \right)$$
利用欧拉公式 $\sin(\theta) = \frac{e^{j\theta} - e^{-j\theta}}{2j}$：
$$F(\mu) = A \left( \frac{2j \sin(\pi \mu W)}{j2\pi \mu} \right) = \frac{A \sin(\pi \mu W)}{\pi \mu}$$
转换为 $\text{sinc}$ 函数的标准形式 $\text{sinc}(x) = \frac{\sin(\pi x)}{\pi x}$：
$$F(\mu) = AW \left( \frac{\sin(\pi \mu W)}{\pi \mu W} \right) = AW \cdot \text{sinc}(\mu W)$$

##### 证明：时域 Sinc 函数 $\to$ 频域盒状函数 使用傅里叶变换的对偶性

我们从傅里叶逆变换 (IFT) 的定义出发：
$$f(t) = \int_{-\infty}^{\infty} F(\mu) e^{j2\pi \mu t} d\mu$$
我们希望将上式变换为 $f(-\mu)$ 的形式。将 $t$ 替换为 $-\mu$：
$$f(-\mu) = \int_{-\infty}^{\infty} F(\mu') e^{j2\pi \mu' (-\mu)} d\mu'$$
$$\mathcal{F}\{F(t)\} = \int_{-\infty}^{\infty} F(t) e^{-j2\pi \mu t} dt$$
因此，$\mathcal{F}\{F(t)\} = f(-\mu)$，证毕

由于盒状函数 $f(t)$ 是一个偶函数， $f(-\mu) = f(\mu)$

$$\mathcal{F}\{AW \cdot \text{sinc}(tW)\} = \begin{cases} A & |\mu| \le W/2 \\ 0 & \text{else} \end{cases}$$

### 二维图像时空域变换至频域的具体细节

##### 二维离散傅里叶变换 (DFT for 2D)

对二维图像 $f(x,y)$ 进行傅里叶变换，在数学上是将其从空间域 (由像素坐标 $x, y$ 描述) 转换为 频域 (由频率 $u, v$ 描述)，首先，对图像的 每一行 分别执行一次一维傅里叶变换 (1D DFT)，然后，对上一步得到的结果的 每一列 再分别执行一次一维傅里叶变换 (1D DFT)，这个过程的最终产物是一个频域图像 $F(u,v)$，它的大小与原图相同，但图中的每个点 $(u,v)$ 不再代表空间位置，而是代表了原始图像中特定方向和变化速率的 频率成分

- 对于一个大小为 $M \times N$ 的数字图像 $f(x,y)$，其二维离散傅里叶变换 $F(u,v)$ 的定义如下：

$$ F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})}$$

- $f(x,y)$ 是原始图像在 空间域 的函数，其中 $(x,y)$ 是像素的坐标 (行和列)
- $F(u,v)$ 是变换后的 频域 函数，其中 $(u,v)$ 是频率变量 (水平频率和垂直频率)。
- $e^{-j\theta}$ 是复指数项，它代表了构成图像的基础“波纹” (正弦/余弦波)
- 这个公式的直观含义是：计算 $F(u,v)$ 在 $(u,v)$ 这一点的值，就是将原始图像 $f(x,y)$ 与一个特定频率 $(u,v)$ 的“波纹” $e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})}$ 进行“对比” (在复数上的点乘求和)，看原始图像中含有多少这个频率的成分

##### 变换的实际操作 (可分离性)

- 直接计算上述的 $F(u,v)$ 公式 (一个双重求和) 的计算量非常大，2D DFT 具有 可分离性 (Separability)，这使得我们可以将其分解为两次 1D DFT，大大提高了效率

- **操作步骤 1 (对行变换)**：取图像 $f(x,y)$ 的第一行 ($y=0$)，将其视为一个一维数组 $f(x, 0)$，对这个数组进行 1D DFT，得到 $F(u, 0)$，对所有 $N$ 行都重复此操作
$$ F_{rows}(u, y) = \sum_{x=0}^{M-1} f(x,y) e^{-j2\pi \frac{ux}{M}}$$
- **操作步骤 2 (对列变换)**：将上一步得到的中间结果 $F_{rows}(u, y)$，按列取出 (固定 $u$，变化 $y$)，对其进行 1D DFT
$$ F(u,v) = \sum_{y=0}^{N-1} F_{rows}(u,y) e^{-j2\pi \frac{vy}{N}}$$
- **操作步骤 3 (最终结果)**：$F(u,v)$ 就是 $f(x,y)$ 的二维傅里叶变换

### 证明：一维连续傅里叶变换的卷积定理

- 一维连续傅里叶变换 (FT):    $$F(u) = \mathcal{F}\{f(t)\} = \int_{-\infty}^{\infty} f(t) e^{-j2\pi u t} dt$$
- 一维连续傅里叶逆变换 (IFT):
$$f(t) = \mathcal{F}^{-1}\{F(u)\} = \int_{-\infty}^{\infty} F(u) e^{j2\pi u t} du$$    
- 一维连续卷积 (Convolution):
$$f(t)*h(t) = \int_{-\infty}^{\infty} f(\tau) h(t - \tau) d\tau$$

**① 证明：卷积变换为乘积 $f(t)*h(t) \iff F(u)H(u)$**

$\mathcal{F}\{f(t)*h(t)\} = \int_{t=-\infty}^{\infty} \left[ f(t)*h(t) \right] e^{-j2\pi u t} dt$

$= \int_{t=-\infty}^{\infty} \left[ \int_{\tau=-\infty}^{\infty} f(\tau) h(t - \tau) d\tau \right] e^{-j2\pi u t} dt$

交换积分顺序：

$= \int_{\tau=-\infty}^{\infty} f(\tau) \left[ \int_{t=-\infty}^{\infty} h(t - \tau) e^{-j2\pi u t} dt \right] d\tau$

对括号内的 $t$ 积分进行变量替换，令 $p = t - \tau$，则 $t = p + \tau$，$dt = dp$：

$= \int_{\tau=-\infty}^{\infty} f(\tau) \left[ \int_{p=-\infty}^{\infty} h(p) e^{-j2\pi u (p + \tau)} dp \right] d\tau$

$= \int_{\tau=-\infty}^{\infty} f(\tau) e^{-j2\pi u \tau} \left[ \int_{p=-\infty}^{\infty} h(p) e^{-j2\pi u p} dp \right] d\tau$

括号内即为 $H(u)$ 的定义：

$= \int_{\tau=-\infty}^{\infty} f(\tau) e^{-j2\pi u \tau} H(u) d\tau$

$= F(u)H(u)$，证毕


**② 证明：乘积变换为卷积 $f(t)h(t) \iff F(u)*H(u)$**

$\mathcal{F}\{f(t)h(t)\} = \int_{t=-\infty}^{\infty} f(t)h(t) e^{-j2\pi u t} dt$

将 $h(t)$ 用其 IFT 表达式代入: 

$= \int_{t=-\infty}^{\infty} f(t) \left[ \int_{v=-\infty}^{\infty} H(v) e^{j2\pi v t} dv \right] e^{-j2\pi u t} dt$

交换积分顺序：

$= \int_{v=-\infty}^{\infty} H(v) \left[ \int_{t=-\infty}^{\infty} f(t) e^{j2\pi v t} e^{-j2\pi u t} dt \right] dv$

合并括号内的指数项：

$= \int_{v=-\infty}^{\infty} H(v) \left[ \int_{t=-\infty}^{\infty} f(t) e^{-j2\pi (u - v) t} dt \right] dv$

括号内即为 $F(u-v)$ 的定义：

$= \int_{v=-\infty}^{\infty} H(v) F(u - v) dv$

$= (H*F)(u) = (F*H)(u)$，证毕


### 证明：一维离散傅里叶变换的卷积定理

设 $f(t)$ 和 $h(t)$ 为 $M$ 点的一维离散信号

- 一维离散傅里叶变换 
$$F(u)=\sum_{t=0}^{M-1}f(t)e^{-j2\pi ut/M}$$
- 一维离散傅里叶逆变换 (1D-IDFT):
$$f(t)=\frac{1}{M}\sum_{u=0}^{M-1}F(u)e^{j2\pi ut/M}$$
- 一维循环卷积
$$f(t)*h(t)=\Sigma_{m=0}^{M-1}f(m)h(t-m)$$

**① 证明： $f(t)*h(t) \iff F(u)H(u)$**

$\mathcal{F}\{f(t)*h(t)\} = \sum_{t=0}^{M-1} \left( f(t)*h(t) \right) e^{-j2\pi ut/M}$

$= \sum_{t=0}^{M-1} \left( \Sigma_{m=0}^{M-1}f(m)h(t-m) \right) e^{-j2\pi ut/M}$

交换求和顺序：

$= \Sigma_{m=0}^{M-1} f(m) \left( \sum_{t=0}^{M-1} h(t-m) e^{-j2\pi ut/M} \right)$

对括号内的项进行变量替换，令 $p = t-m$，则 $t = p+m$ (求和范围因周期性不变)：

$= \Sigma_{m=0}^{M-1} f(m) \left( \sum_{p=0}^{M-1} h(p) e^{-j2\pi u(p+m)/M} \right)$

$= \Sigma_{m=0}^{M-1} f(m) e^{-j2\pi um/M} \left( \sum_{p=0}^{M-1} h(p) e^{-j2\pi up/M} \right)$

$= \Sigma_{m=0}^{M-1} f(m) e^{-j2\pi um/M} H(u)$

$= F(u)H(u)$，证毕

**② 证明： $f(t)h(t) \iff \frac{1}{M}F(u)*H(u)$**

$\mathcal{F}\{f(t)h(t)\} = \sum_{t=0}^{M-1} f(t)h(t) e^{-j2\pi ut/M}$

将 $h(t)$ 用其 1D-IDFT 表达式代入

$= \sum_{t=0}^{M-1} f(t) \left( \frac{1}{M}\sum_{m=0}^{M-1}H(m)e^{j2\pi mt/M} \right) e^{-j2\pi ut/M}$

$= \frac{1}{M} \sum_{m=0}^{M-1} H(m) \left( \sum_{t=0}^{M-1} f(t) e^{j2\pi mt/M} e^{-j2\pi ut/M} \right)$

$= \frac{1}{M} \sum_{m=0}^{M-1} H(m) \left( \sum_{t=0}^{M-1} f(t) e^{-j2\pi (u-m)t/M} \right)$

括号内为 $F(u-m)$ 的定义：

$= \frac{1}{M} \sum_{m=0}^{M-1} H(m) F(u-m)$

$= \frac{1}{M} (H*F)(u) = \frac{1}{M} (F*H)(u)$


### 常用变换

**① 证明：时域平移 (Time Shift)**

**证明目标：** $f(t-t_0) \iff F(u) e^{-j2\pi u t_0}$

$\mathcal{F}\{f(t-t_0)\} = \int_{t=-\infty}^{\infty} f(t-t_0) e^{-j2\pi u t} dt$

对积分变量进行替换，令 $\tau = t - t_0$，则 $t = \tau + t_0$，$dt = d\tau$：

$= \int_{\tau=-\infty}^{\infty} f(\tau) e^{-j2\pi u (\tau + t_0)} d\tau$

$= \int_{\tau=-\infty}^{\infty} f(\tau) e^{-j2\pi u \tau} e^{-j2\pi u t_0} d\tau$

$= F(u) e^{-j2\pi u t_0}$，


**② 证明：频域平移 (Frequency Shift)**

**证明目标：** $f(t) e^{j2\pi u_0 t} \iff F(u-u_0)$

$\mathcal{F}\{f(t) e^{j2\pi u_0 t}\} = \int_{t=-\infty}^{\infty} \left[ f(t) e^{j2\pi u_0 t} \right] e^{-j2\pi u t} dt$

$= \int_{t=-\infty}^{\infty} f(t) e^{-j2\pi (u - u_0) t} dt$

$= F(u-u_0)$，证毕