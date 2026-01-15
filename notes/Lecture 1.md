### 数据介绍


### 图形概念

$$Size_{MB} = \frac{T \times FPS \times (W \times H \times D) \times C \times B}{8 \times 1024 \times 1024}$$

##### 符号说明与典型取值

- $T$：时间长度 (1s, 10s, 60s)
- $FPS$：时间轴采样率 (24, 30, 60 fps)
- $W \times H$：单张切片的分辨率 (512x512, 1024x1024 px)
- $D$：切片堆叠层数或深度 (128, 256, 512 slices)
- $C$：颜色或矢量通道数 (1 为标量灰度，3 为 RGB 或三维矢量)
- $B$：量化比特位深 (8, 12, 16 bit)
