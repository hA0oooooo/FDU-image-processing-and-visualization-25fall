### 噪声模型

- **基本假设**: 假设噪声独立于空间坐标, 并且与图像本身不相关 (即像素值与噪声值不相关) 

- **高斯/正态噪声** (Gaussian Noise):  $p(z)=\frac{1}{\sqrt{2\pi}\sigma}e^{-(z-\overline{z})^{2}/2\sigma^{2}}$ 

- **瑞利噪声** (Rayleigh Noise): $p(z)=\begin{cases}\frac{2}{b}(z-a)e^{\frac{-(z-a)^{2}}{b}},&z\ge a\\ 0,&z<a\end{cases}$ 

- **伽马(爱尔朗)噪声** (Erlang/Gamma Noise): $p(z)=\begin{cases}\frac{a^{b}z^{b-1}}{(b-1)!}e^{-az},&z\ge c\\ 0,&z<a\end{cases}$ 

- **指数噪声** (Exponential Noise): $p(z)=\begin{cases}ae^{-az},&z\ge0\\ 0,&z<0\end{cases}$ 

- **均匀噪声** (Uniform Noise):$p(z)=\begin{cases}\frac{1}{b-a},&a\le z\le b\\ 0,&else\end{cases}$ 

- **脉冲/椒盐噪声** (Impulse/Salt-and-pepper Noise):

$p(z)=\begin{cases}P_{a},& \text{像素灰度设为a (胡椒)} \\ P_{b},& \text{像素灰度设为b (盐粒)} \\ 1-P_{a}-P_{b},& \text{像素灰度不变} \end{cases}$ 

- **白噪声** (White Noise): 噪声的傅里叶谱 (Fourier spectrum) $|F(u,v)|$ 是一个常量 
- **周期噪声** (Periodic 
- Noise): 这是一种在频域中表现明显的噪声 

### 估计噪声

- **图像域估计 (空间域)**
    - 如果图像中存在灰度恒定的小区域 (small strips), 我们可以通过计算这些区域的**直方图 (histograms)** 来估计噪声的 PDF (概率密度函数) 
    - 例如, 可以通过观察直方图的形状来判断噪声是高斯噪声、瑞利噪声还是均匀噪声 

- **频域估计**
    - 周期性噪声 (Periodic noise) 在频域中非常容易识别 
    - 它在图像的频谱 (Spectrum) 中表现为集中的能量脉冲 (impulses) , 例如由正弦噪声 (sinusoidal noise) 引起的共轭脉冲


### 去噪 - 空间滤波

##### 均值滤波器


##### 统计排序滤波器



##### 自适应局部降噪滤波器

$$\hat{f}(x,y)=g(x,y)-\frac{\sigma_{\eta}^{2}}{\sigma_{L}^{2}}[g(x,y)-m_{L}]$$

- $\hat{f}(x,y)$: 滤波器输出的复原图像像素
- $g(x,y)$: 含有噪声的输入图像像素
- $m_{L}$: 像素 $(x,y)$ 邻域窗口 $S_{xy}$ 内的局部均值
- $\sigma_{\eta}^{2}$: 全局噪声方差 (Noise variance), 代表我们试图去除的噪声的强度, 这是一个在整幅图像上恒定的参数
- $\sigma_{L}^{2}$: 局部方差 (Local variance), 在邻域窗口 $S_{xy}$ 内计算得到的方差, 它随 $(x,y)$ 的位置变化

##### 自适应中值滤波器

Level A : 判断中值是否为脉冲，根据某些条件增加窗口尺寸

If $z_{min} < z_{med} < z_{max}$, go to Level B（当前中值不是极端噪音）
Else, increase the size of $S_{xy}$ （当前中值是极端噪音，扩大局部窗口面积）
If $S_{xy} \le S_{max}$, repeat level A 
Else, output $z_{med}$ （达到限制窗口时中值依旧是极端噪音，也只好保留）

Level B : 判断当前值是否为脉冲

If $z_{min} < z_{xy} < z_{max}$, output $z_{xy}$ （判断当前值是否是极端噪音，不是则保留）
Else output $z_{med}$ （用非极端噪音的中值代替当前值）

* 处理密度很高的盐椒噪声
* 修改：如果算法在最大窗口 $S_{max}$ 内都无法找到一个可信的中值 ($z_{med}$) 来替换噪声，那么宁愿放弃滤波，保留原始像素值 ($z_{xy}$)，也不愿用一个已知的噪声 ($z_{med}$) 来替换它


### 去噪 - 频域滤波

##### 陷波滤波器 ( Notch Filter )

##### 带阻滤波器

##### 带通滤波器

##### 最优陷波滤波

1. 提取干扰模式的主频率分量: $N(u,v) = H_{NP}(u,v)G(u,v)$
	- 通常需要显示的观察 $G(u,v)$ 的谱来交互创建陷波滤波器

2. 获得空间域噪声模式: $\eta(x,y) = \mathcal{F}^{-1}\{H_{NP}(u,v)G(u,v)\}$

3. 通过加权函数(调制函数)获得估计: $\hat{f}(x,y) = g(x,y) - w(x,y)\eta(x,y)$

    - 其中 $w(x,y)$ 称为加权或调制函数, 它的优化需要某种假设        

4. 如使得 $\hat{f}(x,y)$ 在每一个点 $(x,y)$ 的邻域区域 $S_{xy}$ 上的方差最小
$$ min~\sigma^2(x,y) = \frac{1}{|S_{xy}|} \sum_{(a,b) \in S_{xy}} (\hat{f}(a,b) - \bar{\hat{f}}(x,y))^2$$

	- 假设 $w$ 在 $S_{xy}$ 内一致, 解得
$$ w(x,y) = \frac{\mu_{g\eta|S_{xy}} - \mu_{g|S_{xy}}\mu_{\eta|S_{xy}}}{\sigma_{\eta|S_{xy}}^{2}}$$
$$ \mu_{g\eta|S_{xy}} = \frac{1}{|S_{xy}|} \sum_{(a,b) \in S_{xy}} (g(a,b)\eta(a,b))$$


### 估计退化函数

##### 观察法 (Image Observation)

- 目标: 从已经退化的图像 $g(x,y)$ 本身来反推出退化函数 $H(u,v)$

- 前提: 需要在 $g(x,y)$ 中找到一个强信号的子图像 $g_s(x,y)$ (此时可忽略噪声, 即 $g_s \approx h * f_s$); 并且必须能估计该子图像未退化(清晰)时的样子 $f_s(x,y)$

- 过程:
    
    - 从 $g_s(x,y)$ (模糊块) 出发, 用锐化滤波器或手工方法得到 $\hat{f}_s(x,y)$ (对清晰块的估计)
    - 对 $g_s(x,y)$ 进行傅里叶变换得到 $G_s(u,v)$
    - 对 $\hat{f}_s(x,y)$ 进行傅里叶变换得到 $\hat{F}_s(u,v)$
    - 根据 $G_s(u,v) \approx H(u,v) \hat{F}_s(u,v)$, 通过除法求解 $H(u,v)$:	$$H(u,v) \approx H_{s}(u,v)=\frac{G_{s}(u,v)}{\hat{F}_{s}(u,v)}$$
- 缺点: 手工方法比较费时费力

##### 试验法 (Experimentation)

- 目标: 通过对退化系统进行一次"可控的试验"来找出 $H(u,v)$

- 前提: 必须能够使用导致图像退化的同一套设备

- 过程:
    
    - 构造一个**冲激** $f(x,y)$ (Impulse, 即理想的点光源, 如黑背景上的小白点)

    - 使用有问题的退化系统拍摄这个冲激, 得到退化(模糊)了的图像 $g(x,y)$

    - 对 $g(x,y)$ (模糊的点) 进行傅里叶变换, 得到 $G(u,v)$

    - 冲激 $f(x,y)$ 的傅里叶变换 $F(u,v)$ 是一个**常量 $A$**

    - 根据模型 $G(u,v) = H(u,v)F(u,v)$, 即 $G(u,v) = H(u,v) \times A$

    - 求解 $H(u,v)$:
$$H(u,v) = \frac{G(u,v)}{A}$$

- 优点: 这是最准确的方法, $G(u,v)$ (模糊点的频谱) 直接反映了 $H(u,v)$

##### 数学建模法 (Modeling)

- 目标: 基于退化的物理原理 (如物理学或运动学), 直接写出 $H(u,v)$ 的数学公式

- 前提: 必须知道退化是_什么原因_造成的

- 示例1: 大气湍流 (Atmospheric turbulence)
    
    - 物理模型可表示为:
$$H(u,v)=exp(-k(u^{2}+v^{2})^{5/6})$$

    - 只需要估计参数 $k$ (湍流严重程度) 即可

- 示例2: 传感器运动模糊 (Sensor motion)
    
    - 假设曝光 $T$ 期间的运动轨迹为 $x_0(t), y_0(t)$
    - 物理模型可表示为:
$$H(u,v)=\int_{0}^{T}e^{-j2\pi[ux_{0}(t)+vy_{0}(t)]}dt$$
    - 只需要知道运动轨迹即可积分算出 $H(u,v)$


### 复原方法

##### 逆滤波

- 逆滤波 (Inverse Filtering) 是最直接的图像复原方法

- 退化模型是 $G(u,v) = H(u,v)F(u,v) + N(u,v)$

- 假设我们已经知道退化函数 $H(u,v)$, 并且暂时忽略噪声 $N(u,v)$, 那么模型就是 $G(u,v) \approx H(u,v)F(u,v)$；已知噪声的情况下也可以减去噪声

- 为了复原 $\hat{F}(u,v)$, 我们只需要在频率域做一个除法, "撤销" $H(u,v)$ 的影响:

$$ \hat{F}(u,v) = \frac{G(u,v)}{H(u,v)}$$
- 完整的复原公式应该是:

$$ \hat{F}(u,v) = \frac{G(u,v)}{H(u,v)} = \frac{H(u,v)F(u,v) + N(u,v)}{H(u,v)}$$$$ \hat{F}(u,v) = F(u,v) + \frac{N(u,v)}{H(u,v)}$$
- 问题就出在 $\frac{N(u,v)}{H(u,v)}$ 这一项，$H(u,v)$ (退化函数, 例如运动模糊或大气湍流) 通常是一个低通滤波器, 这意味着它在高频区域的值会很小，放大的噪声会完全淹没 (dominate) 原始的图像信号 $F(u,v)$
* 改进：修改退化函数的高频区域取值为 1 


##### 最小均方误差 ( 维纳 ) 滤波

- 维纳滤波(Wiener filtering)的目标是找到均方误差 $e^2 = E\{(f-\hat{f})^2\}$ 最小的估计 $\hat{f}$
- 维纳滤波的频域表达式为:
$$ \hat{F}(u,v) = \left[ \frac{1}{H(u,v)} \frac{|H(u,v)|^2}{|H(u,v)|^2 + K(u,v)} \right] G(u,v)$$
- 其中 $K(u,v) = S_{\eta}(u,v) / S_f(u,v)$ (噪声信号功率谱之比)

**在不同情况下的分析**

- **1. 低频 (信号远大于噪声)**

    - **条件**: $S_{\eta} \ll S_f$
    - **分析**: 噪声信号比 $K(u,v) \to 0$。此时, 括号中的调节项 $\frac{|H|^2}{|H|^2 + K} \to 1$
    - **结果**: $\hat{F}(u,v) \approx \frac{G(u,v)}{H(u,v)}$，维纳滤波等效于逆滤波, 因为此时逆滤波是安全的

- **2. 高频 (噪声远大于信号)**

    - **条件**: $S_{\eta} \gg S_f$
    - **分析**: 噪声信号比 $K(u,v) \to \infty$。同时, 在高频区 $H(u,v) \to 0$
    - **结果**: 括号中的调节项 $\frac{|H|^2}{|H|^2 + K} \to 0$，$\hat{F}(u,v) \to 0$，维纳滤波将该频率的输出置零, 有效抑制了噪声爆炸

- **3. 信号噪声相当 ($S_{\eta} \sim S_f$)

    - **条件**: $S_{\eta} \sim S_f$, 此时 $K(u,v) \approx C$
    - **分析 (a)**: 如果此时 $H(u,v) \to 0$ (如高频边缘), 调节项 $\frac{|H|^2}{|H|^2 + C} \approx \frac{0}{0 + C} = 0$, $\hat{F}(u,v) \to 0$
    - **分析 (b)**: 如果此时 $|H(u,v)| \gg 0$ (例如 $H \approx 1$, 退化不严重的中频), 调节项变为 $\frac{1}{1 + C}$
    - **结果 (b)**: $\hat{F}(u,v) \approx \left[ \frac{1}{H(u,v)} \frac{1}{1 + C} \right] G(u,v)$，这是一个**折衷/加权平均**的结果, 既不是完全的逆滤波, 也不是完全置零

##### 维纳滤波处理白噪声的完整逻辑与步骤

维纳滤波的目标是在最小均方误差 (MSE)准则下，从噪声图像 $g(x,y)$ 中恢复出最接近原始图像 $f(x,y)$ 的估计 $\hat{f}(x,y)$

当处理的噪声是白噪声 (White Noise) 且没有模糊 ($H(u,v)=1$) 时，其逻辑步骤如下：

**第一步：确定维纳滤波公式**

- 通用维纳滤波公式 (包含模糊 $H$)：
$$\hat{F}(u,v) = \left[ \frac{1}{H(u,v)} \frac{|H(u,v)|^2}{|H(u,v)|^2 + S_{\eta}(u,v) / S_f(u,v)} \right] G(u,v)$$

- **本题的简化**：
1. **无模糊**：降质函数 $H(u,v) = 1$
2. **白噪声**：白噪声的定义是其功率谱 $S_{\eta}(u,v)$ 在所有频率上都是一个常数，我们记为 $K$ (即噪声方差 $\sigma_\eta^2$)

- 将 $H=1$ 和 $S_{\eta}=K$ 代入通用公式：
$$\hat{F}(u,v) = \left[ \frac{1}{1} \frac{|1|^2}{|1|^2 + K / S_f(u,v)} \right] G(u,v)$$
- 整理得到最终的滤波器形式：
$$\hat{F}(u,v) = \left[ \frac{S_f(u,v)}{S_f(u,v) + K} \right] G(u,v)$$

- $S_f(u,v)$ 是原始信号 $f$ 的功率谱
- $K$ 是白噪声 $\eta$ 的功率谱（一个常数）

**第二步：估计未知的功率谱

我们无法直接知道 $S_f(u,v)$ (因为 $f$ 未知) 和 $K$。因此必须估计它们

- **1. 估计噪声功率谱 $K$**
    
    - **方法**：在带噪图像 $g(x,y)$ 中，找到一块在原始图像 $f$ 中本应是“平坦”的区域（如天空、墙壁，灰度变化很小）
    - **计算**：计算这个小区域内像素的 **方差 (Variance)**
    - **逻辑**：在这块平坦区域， $f(x,y) \approx \text{Constant}$，方差 $\sigma_f^2 \approx 0$。因此，观测到的方差 $\sigma_g^2$ 几乎完全由噪声贡献， $\sigma_g^2 \approx \sigma_\eta^2$
    - **结果**：$K = \sigma_\eta^2 \approx \sigma_g^2 \text{(from flat region)}$

- **2. 估计信号功率谱 $S_f(u,v)$**
    
    - **前提**：假设信号 $f$ 和噪声 $\eta$ 不相关
    - **理论**：带噪图像的功率谱 $S_g(u,v) = S_f(u,v) + S_{\eta}(u,v)$
    - **计算**：$S_f(u,v) = S_g(u,v) - S_{\eta}(u,v)$
    - **近似**：我们用 $g(x,y)$ 的傅里叶变换的模的平方 $|G(u,v)|^2$ 来近似 $S_g(u,v)$
    - **结果**：$S_f(u,v) \approx |G(u,v)|^2 - K$

**第三步：构建并应用滤波器**

将第二步的估计结果代入第一步的公式中：
$$\hat{F}(u,v) = \left[ \frac{S_f(u,v)}{S_f(u,v) + K} \right] G(u,v) \approx \left[ \frac{|G(u,v)|^2 - K}{|G(u,v)|^2 - K + K} \right] G(u,v)$$
$$\hat{F}(u,v) = \left[ \frac{|G(u,v)|^2 - K}{|G(u,v)|^2} \right] G(u,v)$$

**第四步：反变换**

- 通过傅里叶逆变换 (Inverse FT) 将估计的频域 $\hat{F}(u,v)$ 转换回空间域，得到最终的复原图像 $\hat{f}(x,y)$
$$\hat{f}(x,y) = \mathcal{F}^{-1}\{\hat{F}(u,v)\}$$

