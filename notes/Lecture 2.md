### 数字图像形成

* Radiometric Resolution：辐射分辨率 or 灰度分辨率，由比特深度决定
* Geometric Resolution：几何分辨率，由像素总数决定
* Volumetric：3-dimension of x-y-z

### 图像亮度与对比度

* brightness adaptation level：10e−6 mL - 10e4 mL
* Weber Ratio：在一个亮度为 I 的背景上，有一个亮度为 I+ΔI 的小圆圈 ，ΔI 是人眼能刚好分辨出来的最小亮度增量。韦伯比率就是 ΔI/I
-  Contrast：对比度
-  Weber Contrast：韦伯对比度 提供了一种衡量局部对比度的方法
    - 公式: $$\frac{I_b - I}{I_b}$$
-  Michelson Contrast：迈克尔逊对比度
    - 公式: $$\frac{I_{max}​ - I_{min}}{I_{max}​ + I_{min}}$$​​
    - 应用实例: 这个公式常用于评估整个图像的整体对比度，尤其是在评价一个包含明暗交替图案（例如，光学测试图卡上的条纹） 

### 坐标系与医学图像

*  Voxel/pixel size：注意像素/体素的物理尺寸 
*  图像坐标 v.s. 物理坐标
*  坐标转换：Origin (原点)，Spacing (间距)，Pixel Index (像素索引)
*  Orientation (方位) 与 世界坐标系：
	* Anterior (前) VS Posterior (后) 冠状面 (Coronal plane)
	- Left (左) VS Right (右) 矢状面 (Sagittal plane)
	- Superior (上) VS Inferior (下) 横断面 (Transverse plane
- 48 xyz/ijk axis schemes：$P_3^{3} \times 2^3$

### 图像像素操作

* Single-pixel operations
* Neighbourhood operations 平均模糊化 / 图像插值
* Geometric spatial transformations
* Image intensity as random variables, z

### 图像灰度值变换

* 连续变化 v.s. 阈值
* binarize：二元灰度值
* Negative (Linear)：黑白反转
* Linear(identity) v.s. Logarithmic(Log) v.s. Power law
* 见课件图片

* Contrast stretching transformation (对比度拉伸) : the simplest piecewise linear functions

* Intensity Level Slicing (灰度级分层)：to highlight a specific range of intensities in an image
* 二值映射（不尊重原图） 区域映射（尊重原图）

### 基于直方图的变换

* Histogram-based transform
* nD-joint histogram (e.g. 2D 联合直方图)
