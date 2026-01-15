##### 体光照模型核心原理

体渲染中的光照计算旨在增强图像的深度感和面结构信息，主流模型为 **Blinn-Phong** 模型，它通过综合环境光、漫反射和镜面反射来确定采样点的颜色
$$C = (k_a + k_d(N \cdot L)) \cdot C_{TF} + k_s(N \cdot H)^n$$
- **$C_{TF}$**：通过传输函数（Transfer Function）对采样点标量值进行映射得到的原始颜色 
- **$k_a, k_d, k_s$**：分别为环境光、漫反射和镜面反射系数 
- **$N$**：采样点处的法线方向（在体数据中通常由梯度矢量代替）
- **$L$**：光源方向
- **$H$**：半角向量，即视线方向 $V$ 和光线方向 $L$ 的平均方向
- **$n$**：高光系数，决定高光区域的集中程度
##### 法线（梯度）计算逻辑

在体渲染中，体素本身没有定义的几何面，因此使用 **梯度矢量（Gradient Vector）** 作为等值面的法矢 $N$，梯度代表了标量场中值变化最剧烈的方向。

对于正交网格，通常采用 **中心差分法（Central Difference）** 计算梯度：

$$\text{grad } I = \nabla I = \left( \frac{\partial I}{\partial x}, \frac{\partial I}{\partial y}, \frac{\partial I}{\partial z} \right)$$

分量计算公式：

- $G_x = \frac{F_{i+1,j,k} - F_{i-1,j,k}}{2\Delta x}$ 
- $G_y = \frac{F_{i,j+1,k} - F_{i,j-1,k}}{2\Delta y}$ 
- $G_z = \frac{F_{i,j,k+1} - F_{i,j,k-1}}{2\Delta z}$ 

计算得到的梯度矢量需进行归一化处理（变为单位向量）才能代入光照公式使用

以下展示包含光照计算的直接体渲染完整逻辑： 

```
Procedure TraceRay(Ray R):
    Initialize Color C_acc = 0, Alpha A_acc = 0  # 初始化累计颜色和不透明度
    Determine Entrance x1 and Exit x2            # 确定光线进入和离开体数据的范围
    For each sample point Sx from x1 to x2:      # 沿着光线等距离采样
        if A_acc >= 1.0: break                   # 如果不透明度已饱和则跳出循环
        # 1. 采样与插值
        ScalarValue s = Interpolate(Sx)      # 获取当前点的标量值（如三线性插值）
        # 2. 梯度（法线）计算
        Vector N = CalculateGradient(Sx)         # 使用中心差分法计算梯度并归一化

        # 3. 分类（映射）
        Color C_raw = TransferFunction_Color(s)  # 获取原始颜色 RGB
        Alpha a_s = TransferFunction_Alpha(s)    # 获取不透明度 alpha

        # 4. 光照效应计算 (Blinn-Phong)
        Color C_shaded = ComputeLighting(N, L, V, C_raw) # 应用光照公式调整颜色

        # 5. 合成（从前向后积分）
        C_acc = C_acc + (1 - A_acc) * C_shaded * a_s    # 累计颜色亮度贡献
        A_acc = A_acc + (1 - A_acc) * a_s            # 累计总吸收率（不透明度）

    Return C_acc                                 # 返回最终像素颜色
```


