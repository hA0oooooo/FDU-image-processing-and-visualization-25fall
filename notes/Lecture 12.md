
### 形态学：膨胀 腐蚀 开 闭

**1 基础定义与集合论基础**

形态学操作基于集合论，研究图像中像素集合的几何结构
* 结构元 (SE)：用于探测图像的小型集合，定义了运算的几何特征，具有明确的原点
* 平移：集合 $B$ 相对于点 $z$ 的平移记为 $(B)_z$，即 $(B)_z = \{c \mid c = b + z, b \in B\}$
* 反射：集合 $B$ 关于原点的反射记为 $\hat{B}$，即 $\hat{B}=\{w|w=-b, b\in B\}$

**2 腐蚀 (Erosion)**

* 定义：结构元平移后完全包含于前景集合 $A$ 的所有原点位置的集合
* 公式：$$A\ominus B=\{z|(B)_{z}\subseteq A\}$$
* 几何意义：收缩或细化物体，消除小于结构元的噪点，剥离边界

**3 膨胀 (Dilation)**

* 定义：结构元反射并平移后，与前景集合 $A$ 交集不为空的所有原点位置的集合
* 公式：$$A\oplus B=\{z|(\hat{B})_{z}\cap A\ne\emptyset\}$$
* 几何意义：粗化或增长物体，桥接裂缝，填充孔洞

**4 开操作 (Opening)**

* 定义：先腐蚀，后膨胀
* 公式：$$A\circ B=(A\ominus B)\oplus B$$
* 几何直观：结构元在目标内部滚动，平滑轮廓，断开狭颈，消除细微突出物
* 特性：具有幂等性，即多做一次开操作结果不变

**5 闭操作 (Closing)**

* 定义：先膨胀，后腐蚀
* 公式：$$A\bullet B=(A\oplus B)\ominus B$$
* 几何直观：结构元在目标外部滚动，弥合间断，填补孔洞和沟壑
* 特性：具有幂等性，通常用于平滑轮廓的凹陷部分


**膨胀与腐蚀的对偶性 (Duality)**

膨胀和腐蚀关于集合的补集 (Complement) 和反射 (Reflection) 是彼此对偶的。这意味着对前景做腐蚀，等效于对背景做膨胀（使用反射后的结构元），反之亦然

公式：
$$(A \ominus B)^c = A^c \oplus \hat{B}$$
$$(A \oplus B)^c = A^c \ominus \hat{B}$$

证明：

以第一个公式 $(A \ominus B)^c = A^c \oplus \hat{B}$ 为例进行证明
根据腐蚀的定义：$A \ominus B = \{z \mid (B)_z \subseteq A\}$

左边推导：
$z \in (A \ominus B)^c$
$\iff z \notin (A \ominus B)$
$\iff (B)_z \not\subseteq A$  (结构元平移后不完全包含于 A)
$\iff (B)_z \cap A^c \neq \emptyset$ (意味着 $(B)_z$ 中至少有一个元素落在 $A$ 的补集中)

右边推导：
根据膨胀定义 $X \oplus Y = \{z \mid (\hat{Y})_z \cap X \neq \emptyset\}$
设 $X = A^c$，$Y = \hat{B}$
注意 $\hat{B}$ 的反射是 $B$ (即 $\hat{\hat{B}} = B$)
$z \in A^c \oplus \hat{B}$
$\iff (\hat{\hat{B}})_z \cap A^c \neq \emptyset$
$\iff (B)_z \cap A^c \neq \emptyset$

结论：
左边 $\iff (B)_z \cap A^c \neq \emptyset \iff$ 右边
证毕

同理可证第二个公式


**开运算与闭运算：定义与几何理解**

开运算 (Opening)
定义：$A \circ B = (A \ominus B) \oplus B$
几何形式：$A \circ B = \cup \{(B)_z \mid (B)_z \subseteq A\}$
直观理解：结构元 $B$ 在 $A$ 内部滚动，能滚到的所有区域的并集，用于平滑轮廓、断开狭颈、去除毛刺

闭运算 (Closing)
定义：$A \bullet B = (A \oplus B) \ominus B$
几何形式：$A \bullet B = [\cup \{(B)_z \mid (B)_z \cap A = \varnothing \}]^c$
直观理解：结构元 $B$ 在 $A$ 外部滚动，滚不到的区域（背景中的孔洞、缝隙）被填充


**开闭运算的性质与证明**

**对偶性 (Duality)**

性质：
$$(A \circ B)^c = A^c \bullet \hat{B}$$
$$(A \bullet B)^c = A^c \circ \hat{B}$$

证明：
利用前面证明的"膨胀腐蚀对偶性"：$(X \ominus Y)^c = X^c \oplus \hat{Y}$ 和 $(X \oplus Y)^c = X^c \ominus \hat{Y}$

对于 $(A \circ B)^c$：
$$
\begin{aligned}
(A \circ B)^c &= ((A \ominus B) \oplus B)^c & \text{(展开开运算定义)} \\
&= (A \ominus B)^c \ominus \hat{B} & \text{(利用膨胀对偶性)} \\
&= (A^c \oplus \hat{B}) \ominus \hat{B} & \text{(利用腐蚀对偶性)} \\
&= A^c \bullet \hat{B} & \text{(闭运算定义: 先膨胀后腐蚀)}
\end{aligned}
$$

**子集/包含性质 (Subset Property)**

性质：
$$A \circ B \subseteq A$$
$$A \subseteq A \bullet B$$

证明：
以开运算 $A \circ B \subseteq A$ 为例。使用几何定义的证明最为直观

根据几何定义公式：
$$A \circ B = \bigcup \{(B)_z \mid (B)_z \subseteq A\}$$
设任意元素 $x \in A \circ B$
根据并集定义，必然存在某个位移 $z$，使得 $x \in (B)_z$ 且满足条件 $(B)_z \subseteq A$
因为 $x \in (B)_z$ 且 $(B)_z \subseteq A$，根据集合包含传递性，必然有 $x \in A$
所以 $A \circ B$ 中的任一元素都属于 $A$
结论：$A \circ B \subseteq A$

(闭运算证明同理，利用其对偶性或外部滚动定义可证 $A$ 被包含在结果中)


**幂等性/不变性 (Idempotence)**

性质：
$$(A \circ B) \circ B = A \circ B$$
$$(A \bullet B) \bullet B = A \bullet B$$
这表明多次重复进行开运算（或闭运算）不会改变结果

证明：
我们要证明 $(A \circ B) \circ B = A \circ B$
令 $C = A \circ B$，我们需要证明 $C \circ B = C$

证明 $C \circ B \subseteq C$：
由子集性质可知，对于任何集合 $X$，都有 $X \circ B \subseteq X$
将 $X$ 替换为 $C$，直接得到 $C \circ B \subseteq C$

证明 $C \subseteq C \circ B$：
根据几何定义，$C = A \circ B$ 是所有包含在 $A$ 中的 $(B)_z$ 的并集
也就是说，$C$ 是由许多个 $B$ 拼凑而成的
既然 $C$ 本身就是由 $B$ 组成的，那么结构元 $B$ 显然可以完全拟合于 $C$ 的每一个局部（因为那些局部本来就是 $B$）
严格来说：对于 $C$ 中的任意一点 $x$，它一定属于某个 $(B)_{z'}$ 且 $(B)_{z'} \subseteq A$
由于开运算只去除了不能拟合 $B$ 的部分，而保留下来的 $(B)_{z'}$ 显然完全拟合 $B$
因此，$C$ 中的所有点都在"$B$ 能完全拟合 $C$"的区域内
所以，再做一次开运算不会删除任何点，即 $C \subseteq C \circ B$

结论：
因为 $C \circ B \subseteq C$ 且 $C \subseteq C \circ B$，故 $C \circ B = C$
即 $(A \circ B) \circ B = A \circ B$


**其他证明目标**

已知 $C \subseteq D$，求证 $C \circ B \subseteq D \circ B$

证明所用定义
开运算的几何解释为 $A \circ B = \cup \{(B)_z \mid (B)_z \subseteq A\}$，即 $A \circ B$ 是所有能完全包含于 $A$ 的结构元 $B$ 的平移 $(B)_z$ 的并集

证明步骤
设任意元素 $x \in C \circ B$
根据几何定义，一定存在某个位移 $z$，使得 $x \in (B)_z$ 且 $(B)_z \subseteq C$
已知 $C \subseteq D$，由 $(B)_z \subseteq C$ 可推出 $(B)_z \subseteq D$
现在有 $x \in (B)_z$ 且 $(B)_z \subseteq D$，根据 $D \circ B$ 的几何定义，这等价于 $x \in D \circ B$
由于 $x$ 是 $C \circ B$ 中的任意元素，且 $x \in D \circ B$，因此 $C \circ B \subseteq D \circ B$


### Successive Operation

连续膨胀是指通过重复应用较小的结构元 $B$，来实现与较大结构元 $kB$ 进行膨胀等效的操作

**1 定义与递推公式**

假设 $A$ 为原始图像集合，$B$ 为基础结构元
连续膨胀 $(A \oplus kB)$ 定义为对 $A$ 进行 $k$ 次结构元为 $B$ 的膨胀迭代

$$(A \oplus kB) = \underbrace{(\dots((A \oplus B) \oplus B) \dots \oplus B)}_{k \text{ 次}}$$

其中初始条件规定为
$$(A \oplus kB) = A, \quad \text{当 } k=0$$

这一性质表明，大尺寸结构元 $kB$ 可以分解为 $k$ 个小尺寸结构元 $B$ 的闵可夫斯基和（Minkowski Sum）

**2 数学证明 (基于闵可夫斯基和)**

在欧几里得空间中，膨胀 $A \oplus B$ 等价于集合的向量加法（闵可夫斯基和）
$$A \oplus B = \{ a + b \mid a \in A, b \in B \}$$

证明连续膨胀等效于一次性大核膨胀，即证明膨胀运算满足结合律 (Associativity)
$$(A \oplus B) \oplus C = A \oplus (B \oplus C)$$

证明过程
设任意元素 $z \in (A \oplus B) \oplus C$
根据定义，存在 $x \in (A \oplus B)$ 和 $c \in C$，使得 $z = x + c$
进一步展开 $x$，存在 $a \in A$ 和 $b \in B$，使得 $x = a + b$
代入得 $z = (a + b) + c$
由于欧几里得空间中的向量加法满足结合律，即 $(a+b)+c = a+(b+c)$，因此 $z = a + (b + c)$
令 $y = b + c$，显然 $y \in \{ b + c \mid b \in B, c \in C \} = B \oplus C$
因此 $z = a + y, \quad \text{其中 } a \in A, y \in (B \oplus C)$
根据膨胀定义，这意味着 $z \in A \oplus (B \oplus C)$
同理可证反向包含关系，由此得证 $(A \oplus B) \oplus C = A \oplus (B \oplus C)$

推论
令 $C = B$，则有 $(A \oplus B) \oplus B = A \oplus (B \oplus B)$
根据定义，$kB$ 即为 $k$ 个 $B$ 的连续膨胀（即 $kB = B \oplus B \oplus \dots \oplus B$）
通过数学归纳法可知 $(\dots((A \oplus B) \oplus B) \dots \oplus B) = A \oplus (\underbrace{B \oplus B \oplus \dots \oplus B}_{k}) = A \oplus kB$

**3 几何分解实例**

该性质允许将复杂的几何形状分解为简单的基本单元，从而降低计算复杂度

方形分解 (Square)
一个边长较大的正方形结构元，可以分解为小的 $3 \times 3$ 正方形结构元的多次连续膨胀

圆形/圆盘分解 (Disk)
一个大半径的圆盘形结构元，可以通过小圆盘（或近似圆形的结构元）的多次连续膨胀来近似生成

**4 几何演变**

随着膨胀次数 $k$ 的增加，目标区域的边界 $F$ 会向外均匀扩展
若 $B$ 是圆形，则连续膨胀使得物体的角点变得圆滑；若 $B$ 是方形，则物体的轮廓倾向于变成矩形特征

### Distance Transform

**1 具体的边界获取方法**

在距离变换算法中，确定第一层即边界是启动算法的关键

对于二值图像中的每一个前景像素 P(x,y)（值为1），检查其周围的 8 个相邻像素（上、下、左、右、左上、左下、右上、右下）
条件：如果这 8 个邻居中，至少有一个是背景像素（值为0），那么 P(x,y) 就是边界点
操作：将该点在距离图中的值设为 1

**2 不同距离度量的向内传播机制**

**A D4 距离 (城市街区 / City-block)**

连接性：只看 4-邻域（上、下、左、右）
代价：每走一步，距离值 +1
如果一个像素是内部点，且它的上下左右邻居中最小的值是 k，那么它的值就是 k+1

**B D8 距离 (棋盘 / Chessboard)**

连接性：看 8-邻域（包括对角线）
代价：每走一步（无论是横着走还是斜着走），距离值都 +1

**C 欧氏距离 (Euclidean)**

连接性：几何真实距离
代价：水平/垂直移动代价为 1，对角线移动代价为 √2 ≈ 1.414
简单的整数迭代法无法精确计算欧氏距离，存在专门计算欧氏距离变换的算法


**3 距离变换的意义和作用**

骨架提取 / 中轴变换
现象：观察课件中的长方形或不规则形状的距离图，你会发现图像中间会出现高亮（数值大）的山脊线
作用：这些局部极大值点连成的线，构成了物体的拓扑骨架。距离变换是提取物体骨架的常用预处理步骤，用于描述物体的几何结构（如笔画分析）

寻找物体中心
现象：在距离变换图中，亮度最高（灰度值最大）的点，代表该点距离边界最远
作用：这个点通常被认为是形状的几何中心或最深内部点。对于粘连物体的分割（如细胞计数），找到局部最大值点是定位每个独立细胞核心的关键

形状度量与分析
现象：不同的形状会产生独特的距离梯度场
作用：通过统计距离值的分布，可以计算物体的平均宽度（最大距离值的2倍即为最大宽度），或者用于颗粒度分析

形态学侵蚀的快速实现
作用：如果你想对图像进行半径为 R 的腐蚀，只需要在距离变换图上通过阈值操作（保留距离值 > R 的像素），而不需要用巨大的结构元去卷积整张图


### 图像配准：空间变换

##### 线性变换 - 齐次坐标系

齐次坐标系是一种将 $N$ 维空间的点用 $N+1$ 个坐标数值来表示的方法。
对于平面上的有限点 $(x, y)$，其齐次坐标表示为 $(x_1, x_2, x_3)$，且满足以下关系：

$$
x = \frac{x_1}{x_3}, \quad y = \frac{x_2}{x_3}
$$

在图像配准和三维变换中，通常令最后一个分量 $x_3$ (或三维中的 $x_4$) 为 1，即从 $[x, y, z]$ 扩展为 $[x, y, z, 1]$

在笛卡尔坐标系下引入第四维（即“补1”）主要解决了以下两个数学和计算问题：

1. **将仿射变换统一为线性变换（矩阵乘法）**

* 问题：在普通坐标系中，线性变换（如旋转、缩放）可以用矩阵乘法 $x' = Ax$ 表示，但平移（Translation）是向量加法 $x' = x + t$，无法直接合并进同一个矩阵乘法中
* 解决：通过引入齐次坐标（补1），可以将平移操作也写入矩阵中，使其变成标准的矩阵乘法形式；旋转、缩放、平移等一系列变换就可以通过连乘同一个 $4 \times 4$ 矩阵来完成

变换公式如下，注意观察矩阵的最后一列承载了平移量：

$$
\begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix} = \begin{bmatrix} e_{11} & e_{12} & e_{13} & e_{14} \\ e_{21} & e_{22} & e_{23} & e_{24} \\ e_{31} & e_{32} & e_{33} & e_{34} \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
$$
$$M = \begin{bmatrix} \text{Rotation/Scale} & \text{Translation} \\ 0 & 1 \end{bmatrix}$$

左上角负责“乘法”（旋转缩放），最右列负责“加法”（平移）

2. **表示无穷远点**
* 当最后一个分量不为 0（通常为1）时，表示的是有限点 $(x, y)$
* 当最后一个分量为 0 时，即 $(x_1, x_2, 0)$，它不再对应具体的坐标点，而是描述了斜率为 $\lambda = x_2/x_1$ 的方向上的无穷远点

##### 线性变换 - 刚性变换与相似变换

- **平移 (Translations)**
$$ \begin{bmatrix} 1 & 0 & 0 & X_{trans} \\ 0 & 1 & 0 & Y_{trans} \\ 0 & 0 & 1 & Z_{trans} \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

- **Pitch (绕 X 轴旋转 $\Phi$)**
$$ \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos\Phi & \sin\Phi & 0 \\ 0 & -\sin\Phi & \cos\Phi & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

- **Roll (绕 Y 轴旋转 $\Theta$)**
$$ \begin{bmatrix} \cos\Theta & 0 & \sin\Theta & 0 \\ 0 & 1 & 0 & 0 \\ -\sin\Theta & 0 & \cos\Theta & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

- **Yaw (绕 Z 轴旋转 $\Omega$)**
$$ \begin{bmatrix} \cos\Omega & \sin\Omega & 0 & 0 \\ -\sin\Omega & \cos\Omega & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

* 相似变换在刚性变换的基础上增加了缩放
$$\begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix} = \begin{bmatrix} s_0 & 0 & 0 & 0 \\ 0 & s_1 & 0 & 0 \\ 0 & 0 & s_2 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \cdot T_{rigid} \cdot \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$$



### 非线性变换 / 形变变换

##### Local Affine

* 区域控制+全局影响 ，通过对图像不同区域应用独立的仿射变换矩阵，并利用距离加权插值平滑处理区域间的过渡 
##### FFD

* 点控制+局部影响 ，通过操纵覆盖在图像上的规则网格控制点，利用三次B-样条核函数计算像素位移，实现高自由度的局部形变


