# Dimension Reduction

## 1 样本均值&样本方差

对于一个样本矩阵 $X=\begin{pmatrix}
    x_1&x_2&\dotsb&x_N
\end{pmatrix}^\top_{N\times p},x\in\mathbb{R}^p$，样本均值和方差可以用以下的方式表示：
$$
\bar{x}=\frac{1}{N}\sum_{i=1}^Nx_i=\frac{1}{N}\begin{pmatrix}
    x_1&x_2&\dotsb&x_N
\end{pmatrix}\begin{pmatrix}
    1\\1\\\vdots\\1
\end{pmatrix}_{N\times1}=\frac{1}{N}X^\top\mathbf{1}_{N}
$$

$$
\begin{aligned}
    \Sigma&=\frac{1}{N}\sum_{i=1}^N(x_i-\bar{x})(x_i-\bar{x})^\top\\
    &=\frac{1}{N}(X^\top-\frac{1}{N}X^\top\mathbf{1}_{N}\mathbf{1}_{N}^\top)(X^\top-\frac{1}{N}X^\top\mathbf{1}_{N}\mathbf{1}_{N}^\top)^\top\\
    &=\frac{1}{N}X^\top(I_N-\frac{1}{N}\mathbf{1}_{N}\mathbf{1}_{N}^\top)(I_N-\frac{1}{N}\mathbf{1}_{N}\mathbf{1}_{N}^\top)^\top X\\
    &=\frac{1}{N}X^\top HH^\top X\\
\end{aligned}
$$

上式中 $H=(I_N-\frac{1}{N}\mathbf{1}_{N}\mathbf{1}_{N}^\top)$ 称为中心化矩阵，实际上 $H=H^\top$ 且 $HH^\top=H$，因此上式可以简化为 $\Sigma=\frac{1}{N}X^\top H X$。

## 2 PCA

PCA 的基本思想是**最大化投影方差和最小化重构损失**。

我们可以定义投影之后的方差为目标函数：
$$
\begin{aligned}
    \mathcal{J}&=\frac{1}{N}\sum_{i=1}^N\left((x_i-\bar{x})^\top u\right)^2\\
    &=\frac{1}{N}\sum_{i=1}^Nu^\top(x_i-\bar{x})(x_i-\bar{x})^\top u\\
    &=u^\top\left(\frac{1}{N}\sum_{i=1}^N(x_i-\bar{x})(x_i-\bar{x})^\top\right) u\\
    &=u^\top \Sigma u~~~s.t.~u^\top u=1\\
\end{aligned}
$$

约束条件限制了投影向量 $u$ 为单位向量，以免拉伸投影。可以通过有约束的优化问题构建一个拉格朗日形式的损失函数：
$$
\ell(u,\lambda)=u^\top \Sigma u+\lambda(1-u^\top u)
$$

$$
\frac{\partial \ell}{\partial u}=2\Sigma u-2\lambda u
$$

令 $\frac{\partial \ell}{\partial u}=0$，可得 $\Sigma u=\lambda u$，表明主成因（投影方向）即是协方差矩阵的一组特征向量，因此主成因分析可以通过对协方差矩阵进行特征值分解的方式进行，**选择 $q$ 个特征向量作为投影方向，即可以将原始维度 $p$ 降维至 $q$**。

或者从最小化重构代价的角度，定义目标函数：
$$
x_i=\sum_{k=1}^p(x_i^\top u_k)u_k
$$

$$
\hat{x}_i=\sum_{k=1}^q(x_i^\top u_k)u_k
$$

$$
\begin{aligned}
    \mathcal{J}&=\frac{1}{N}\sum_{i=1}^N\|x_i-\hat{x}_i\|^2\\
    &=\frac{1}{N}\sum_{i=1}^N\|\sum_{k=q+1}^p(x_i^\top u_k)u_k\|^2\\
    &=\frac{1}{N}\sum_{i=1}^N\sum_{k=q+1}^p(x_i^\top u_k)^2\\
    &=\sum_{k=q+1}^p\frac{1}{N}\sum_{i=1}^N((x_i-\bar{x})^\top u_k)^2\\
    &=\sum_{k=q+1}^pu_k^\top\Sigma u_k~~~s.t.~u_k^\top u_k=1\\
\end{aligned}
$$

由于 $u$ 之间是线性无关的，因此以上最优化问题可以从求和的形式拆分成若干的独立的优化问题。那么两种视角下，**以投影方差最大化的视角来看，最终求出的特征向量对应较大的特征值，而以重构代价最小化的视角来看，最终求出的特征向量对应较小的特征值**。

## 3 PCA、SVD、PCoA

从上面的推导来看，PCA 的过程实际上是对协方差矩阵 $\Sigma$ 进行特征值分解的过程：
$$
\Sigma=G\Lambda G^\top~~~G^\top G=I~~~\Lambda=\mathrm{diag}(\lambda_1,\lambda_2,\dotsb,\lambda_q,\dotsb,\lambda_p)~~~\lambda_1\geq\lambda_2\geq\dotsb\lambda_p
$$

实际上，这一过程等价于对中心化的样本矩阵 $HX$ 进行奇异值分解：
$$
HX=U\Sigma_X V^\top
$$

这是因为：
$$
\Sigma=X^\top HX=(U\Sigma_X V^\top)^\top(U\Sigma_X V^\top)=V\Sigma_X^2V^\top
$$

以此方式进行奇异值分解可以得到特征向量矩阵 $V$，即代表 PCA 投影方向。

如果对矩阵 $T=HXX^\top H^\top$ 进行特征值分解,则可以直接得到投影坐标矩阵，这是因为：
$$
T=HXX^\top H^\top=(U\Sigma_X V^\top)(U\Sigma_X V^\top)^\top=U\Sigma_X^2U^\top
$$

而投影坐标矩阵可以表示为 $HXV=U\Sigma_X V^\top V=U\Sigma_X$。

这种分析方式称为主坐标分析（PCoA）。

## P-PCA

P-PCA 是对传统 PCA 的扩展，提供了 PCA 的概率视角。

P-PCA 中以隐变量的形式看待降维之后的特征，对于原始数据 $x\in\mathbb{R}^p$，降维之后的隐变量为 $z\in\mathbb{R}^q$，P-PCA 假设：
$$
z\in\mathcal{N}(\mathcal{O}_q,I_q)
$$

$$
x=Wz+\mu+\epsilon
$$

$$
\epsilon\in\mathcal{N}(\mathcal{O}_p,\sigma^2I)
$$

P-PCA 的推断过程需要用到 $P(z|x)$，以下是推导过程：

$$
\mathbb{E}[x]=\mathbb{E}[Wz+\mu]+\mathbb{E}[\epsilon]=\mu
$$

$$
\mathrm{Var}[x]=\mathrm{Var}[Wx+\mu]+\mathrm{Var}[\epsilon]=WW^\top+\sigma^2I
$$

$$
\mathbb{E}[x|z]=\mathbb{E}[Wz+\mu+\epsilon]=Wz+\mu
$$

$$
\mathrm{Var}[x|z]=\mathrm{Var}[Wx+\mu+\epsilon]=\sigma^2I
$$

因此，可以写出联合概率分布：
$$
\begin{pmatrix}
    x\\z
\end{pmatrix}=x\sim\mathcal{N}\left(\begin{bmatrix}
    \mu\\\mathcal{O}_q
\end{bmatrix},\begin{bmatrix}
    WW^\top+\sigma^2I&\Delta\\
    \Delta^\top&I\\
\end{bmatrix}\right)
$$

其中 $\Delta=W$，然后从联合概率分布中推导出 $P(z|x)$：
$$
\mathbb{E}[z|x]=\mathbb{E}[x_{z-x}]+\Sigma_{zx}\Sigma_{xx}^{-1}x=\mathcal{O}_q+\Sigma_{zx}\Sigma_{xx}^{-1}(x-\mu)=W^\top(WW^\top+\sigma I)^{-1}(x-\mu)
$$

$$
\mathrm{Var}[z|x]=\Sigma_{zz-x}=\Sigma_{zz}-\Sigma_{zx}\Sigma_{xx}^{-1}\Sigma_{xz}=I-W^\top(WW^\top+\sigma^2I)^{-1}W
$$

因此，$z|x\sim\mathcal{N}(W^\top(WW^\top+\sigma I)^{-1}(x-\mu),I-W^\top(WW^\top+\sigma^2I)^{-1}W)$，或者进一步通过矩阵求逆引理（Woodbury恒等式）化简为：
$$
z|x\sim\mathcal{N}((WW^\top+\sigma I)^{-1}W^\top(x-\mu),\sigma^2(W^\top W+\sigma^2I)^{-1})
$$

该算法的优化过程使用 EM 算法迭代式地极大似然，当 $\sigma^2\rightarrow 0$ 时，P-PCA 退化为 PCA。
