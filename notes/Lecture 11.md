	
### 高斯混合模型分割算法

 1. 变量定义与概率模型

- 观测数据：图像含 $N$ 个像素点，记为 $X = \{x_1, \dots, x_N\}$，其中每个像素 $x_i$ 是观测变量
- 隐变量：$Z = \{z_1, \dots, z_N\}$，其中 $z_i \in \{0, \dots, K-1\}$ 表示像素 $x_i$ 对应的类别标签
- 模型参数：$\theta = \{\pi_k, \mu_k, \sigma_k\}_{k=0}^{K-1}$
- 先验概率 (Prior)：$P(z_i = k) = \pi_k$，满足 $\sum_{k=0}^{K-1} \pi_k = 1$
- 条件概率 (Likelihood)：$P(x_i | z_i = k) = \Phi(x_i | \mu_k, \sigma_k)$，服从高斯分布：
$$\Phi(x | \mu, \sigma) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

2. 极大似然估计与 Q 函数定义

- 似然函数 ：观测数据 $X$ 关于参数 $\theta$ 的似然函数为所有样本边缘概率的乘积：
$$L(\theta) = P(X|\theta) = \prod_{i=1}^N P(x_i|\theta) = \prod_{i=1}^N \sum_{k=0}^{K-1} P(x_i, z_i=k|\theta)$$
	- 对数似然函数 (Log-Likelihood Function)：
	$$l(\theta) = \log L(\theta) = \sum_{i=1}^N \log \left( \sum_{k=0}^{K-1} \pi_k \Phi(x_i | \mu_k, \sigma_k) \right)$$
	- 难点：由于求和符号 $\sum$ 位于 $\log$ 函数内部，直接求导令梯度为 0 无法得到解析解，因此引入 EM 算法
	- Q 函数 (Auxiliary Function)：Q 函数定义为，在给定观测数据 $X$ 和当前参数 $\theta^{[m]}$ 的条件下，完全数据 $(X, Z)$ 的对数似然函数关于隐变量 $Z$ 后验分布的数学期望：
$$Q(\theta | \theta^{[m]}) = E_{Z | X, \theta^{[m]}} [ \log P(X, Z | \theta) ]$$

**Q 函数怎么来的？**

- 引入关于隐变量 $z_i$ 的任意概率分布 $Q_i(z_i=k)$，简称 $Q_{ik}$，满足归一化条件 $\sum_{k=0}^{K-1} Q_{ik} = 1$
- 构造期望形式：
$$l(\theta) = \sum_{i=1}^N \log \left( \sum_{k=0}^{K-1} Q_{ik} \frac{\pi_k \Phi(x_i | \mu_k, \sigma_k)}{Q_{ik}} \right)$$
$$l(\theta) \ge \sum_{i=1}^N \sum_{k=0}^{K-1} Q_{ik} \log \left( \frac{\pi_k \Phi(x_i | \mu_k, \sigma_k)}{Q_{ik}} \right) \triangleq B(\theta, Q)$$
- 此时 $B(\theta, Q)$ 即为对数似然函数的下界 (ELBO)

- 为了使下界最紧，需要满足 Jensen 不等式的取等条件
- 取等条件为随机变量取值为常数，即对于任意 $k$，以下比值必须为常数 $c_i$（仅依赖于 $i$）：
$$\frac{\pi_k^{[m]} \Phi(x_i | \mu_k^{[m]}, \sigma_k^{[m]})}{Q_{ik}} = c_i$$
$$Q_{ik} = \frac{1}{c_i} \pi_k^{[m]} \Phi(x_i | \mu_k^{[m]}, \sigma_k^{[m]})$$
- 利用分布的归一化性质 $\sum_{k=0}^{K-1} Q_{ik} = 1$ 确定常数 $c_i$：
$$\sum_{k=0}^{K-1} Q_{ik} = \sum_{k=0}^{K-1} \frac{1}{c_i} \pi_k^{[m]} \Phi(x_i | \mu_k^{[m]}, \sigma_k^{[m]}) = 1$$
- 解得常数 $c_i$ 等于当前参数下的边缘概率：
$$c_i = \sum_{k=0}^{K-1} \pi_k^{[m]} \Phi(x_i | \mu_k^{[m]}, \sigma_k^{[m]}) = P(x_i | \theta^{[m]})$$
- 将 $c_i$ 代回 $Q_{ik}$ 的表达式：
$$Q_{ik} = \frac{\pi_k^{[m]} \Phi(x_i | \mu_k^{[m]}, \sigma_k^{[m]})}{\sum_{j=0}^{K-1} \pi_j^{[m]} \Phi(x_i | \mu_j^{[m]}, \sigma_j^{[m]})}$$
- 结论：最优的 $Q_{ik}$ 分布即为隐变量的后验概率 $P(z_i=k | x_i, \theta^{[m]})$

**Q 函数怎么来的？（结束）**


$$Q(\theta | \theta^{[m]}) = E_{Z | X, \theta^{[m]}} [ \log P(X, Z | \theta) ]$$
- 展开后得到 GMM 的 Q 函数形式：
$$Q(\theta | \theta^{[m]}) = \sum_{i=1}^N \sum_{k=0}^{K-1} P(z_i=k | x_i, \theta^{[m]}) \left( \log \pi_k + \log \Phi(x_i | \mu_k, \sigma_k) \right)$$
3. E-Step：后验概率推导 (Bayes Inference)

- 目标是计算 Q 函数中的权重项，即隐变量的后验概率 $P(z_i=k | x_i, \theta^{[m]})$，记为 $P_{ik}^{[m+1]}$
- 根据贝叶斯定理 (Bayes' Theorem)：
$$P(\text{类别}|\text{数据}) = \frac{P(\text{数据}|\text{类别}) \cdot P(\text{类别})}{P(\text{数据})}$$
- 对应到 GMM 模型中，
$$P_{ik}^{[m+1]} = \frac{\Phi(x_i | \mu_k^{[m]}, \sigma_k^{[m]}) \cdot \pi_k^{[m]}}{\sum_{j=0}^{K-1} \Phi(x_i | \mu_j^{[m]}, \sigma_j^{[m]}) \cdot \pi_j^{[m]}}$$

4. M-Step：参数更新推导 (Weighted MLE)

* 先直接产生新隐变量估计：
$$\pi_k^{[m+1]} = \frac{\sum_{i=1}^N P_{ik}^{[m+1]}}{N}$$


- 目标是寻找新的参数 $\theta^{[m+1]}$ 使得 Q 函数最大化：
$$\theta^{[m+1]} = \arg\max_{\theta} Q(\theta | \theta^{[m]})$$
- 更新均值 $\mu_k$：将 Q 函数中与 $\mu_k$ 相关的项对 $\mu_k$ 求偏导并令其为 0：
$$\frac{\partial Q}{\partial \mu_k} = \sum_{i=1}^N P_{ik}^{[m+1]} \frac{\partial}{\partial \mu_k} \left( -\frac{(x_i - \mu_k)^2}{2\sigma_k^2} \right) = \sum_{i=1}^N P_{ik}^{[m+1]} \frac{x_i - \mu_k}{\sigma_k^2} = 0$$$$\mu_k^{[m+1]} = \frac{\sum_{i=1}^N P_{ik}^{[m+1]} x_i}{\sum_{i=1}^N P_{ik}^{[m+1]}}$$
-  更新方差 $\sigma_k^2$：将 Q 函数中与 $\sigma_k$ 相关的项对 $\sigma_k^2$ 求偏导并令其为 0（注意 $\log \Phi$ 中包含 $-\log \sigma$）：
$$\frac{\partial Q}{\partial \sigma_k^2} = \sum_{i=1}^N P_{ik}^{[m+1]} \left( -\frac{1}{2\sigma_k^2} + \frac{(x_i - \mu_k)^2}{2(\sigma_k^2)^2} \right) = 0$$$$(\sigma_k^{[m+1]})^2 = \frac{\sum_{i=1}^N P_{ik}^{[m+1]} (x_i - \mu_k^{[m+1]})^2}{\sum_{i=1}^N P_{ik}^{[m+1]}}$$
-  更新混合系数 $\pi_k$：需要在约束条件 $\sum_{k=0}^{K-1} \pi_k = 1$ 下最大化 $\sum_{i,k} P_{ik} \log \pi_k$$$\mathcal{L}(\pi, \lambda) = \sum_{i=1}^N \sum_{k=0}^{K-1} P_{ik}^{[m+1]} \log \pi_k + \lambda \left( \sum_{k=0}^{K-1} \pi_k - 1 \right)$$$$\frac{\partial \mathcal{L}}{\partial \pi_k} = \frac{\sum_{i=1}^N P_{ik}^{[m+1]}}{\pi_k} + \lambda = 0 \implies \pi_k = -\frac{\sum_{i=1}^N P_{ik}^{[m+1]}}{\lambda}$$

5. 算法流程总结
- 1. 初始化：设定 $\theta^{[0]}$
- 2. E-Step：根据当前参数 $\theta^{[m]}$，利用贝叶斯公式计算属于各类别的后验概率 $P_{ik}^{[m+1]}$
- 3. M-Step：根据后验概率 $P_{ik}^{[m+1]}$，利用加权 MLE 更新参数得到 $\theta^{[m+1]}$
- 4. 迭代：重复 E 和 M 步骤直到对数似然函数收敛
- 5. 输出：根据最终的后验概率 $\hat{z}_i = \arg\max_k P_{ik}$ 确定像素类别


**GMM + MRF 加入空间正则化后的后验概率公式变为：**

$$P_{ik}^{[m+1]} = \frac{\Phi(x_i | \mu_k, \sigma_k) \cdot \exp\left( \beta \sum_{j \in N_i} P_{jk}^{[m]} \right)}{\sum_{l=0}^{K-1} \Phi(x_i | \mu_l, \sigma_l) \cdot \exp\left( \beta \sum_{j \in N_i} P_{jl}^{[m]} \right)}$$


**GMM + MRF + Altas  更好的估计：**
$$\pi_{kx}^{[m+1]} = \frac{f(k_x | P_{N_x}^{[m]}, \theta^{[m]}) \cdot p(A_{kx})}{NF}$$
之后
$$P_{ik}^{[m]} \propto \pi_{kx}^{[m]} \cdot \Phi(x_i | \mu_k^{[m]}, \sigma_k^{[m]})$$


**故障原因：** 单点分为一类
$$P(x_i | \mu_k=x_i, \sigma_k) = \frac{1}{\sqrt{2\pi}\sigma_k}$$
- 当 $\sigma_k \to 0$ 时，似然函数 $L \to \infty$



### 形态学 膨胀 腐蚀

* 开
* 闭