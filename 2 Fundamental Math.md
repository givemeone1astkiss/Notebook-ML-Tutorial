# Fundamental Math---Gaussian Distribution

## 1 Maximum Likelihood Estimate（MLE）

对于一系列数据点 $X=(x_1,\dotsb,x_N)^\top,x_i\in\mathbb{R}^p$，假设其服从高斯分布 $\mathcal{N}(\mu,\Sigma)$，即：
$$p(x)=\frac{1}{\sqrt{2\pi}|\Sigma|^\frac{1}{2}}\exp(-\frac{1}{2}(x-\mu)^\top\Sigma^{-1}(x-\mu))$$

首先考虑 $p=1$ 的情况，此时协方差矩阵 $\Sigma$ 退化为方差 $\sigma^2$：
$$p(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(x-\mu)^2}{2\sigma^2})$$

使用 MLE 对 $\mu$ 和 $\sigma$ 进行估计，则有：
$$\begin{aligned}
    \log P(X|\theta)&=\sum_{i=1}^N\log p(x_i|\theta)\\
    &=\sum_{i=1}^N\log\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(x-\mu)^2}{2\sigma^2})\\
    &=\sum_{i=1}^N\left[\log\frac{1}{\sqrt{2\pi}}+\log\frac{1}{\sigma}-\frac{(x-\mu)^2}{2\sigma^2}\right]\\
\end{aligned}$$

对于均值进行估计：

$$\begin{aligned}
    \mu_{MLE}&=\arg\max_\mu\log P(X|\theta)\\
    &=\arg\max_\mu\sum_{i=1}^N -\frac{(x-\mu)^2}{2\sigma^2}\\
    &=\arg\min_\mu\sum_{i=1}^N (x-\mu)^2\\
\end{aligned}$$

$$\begin{aligned}
    \frac{\partial}{\partial \mu}\sum_{i=1}^N (x-\mu)^2&=\sum_{i=1}^N -2(x_i-\mu)\\
\end{aligned}$$

令 $\frac{\partial}{\partial \mu}\sum_{i=1}^N (x-\mu)^2=0$ 可得 $\mu_{MLE}=\frac{1}{N}\Sigma x_i$ 此为对 $\mu$ 的无偏估计，因为:
$$\mathbb{E}[\mu_{MLE}]=\frac{1}{N}\sum_{i=1}^N \mathbb{E}[x_i]=\frac{1}{N}\sum_{i=1}^N\mu=\mu$$

而对于方差进行估计：
$$\begin{aligned}
    \sigma_{MLE}^2&=\arg\max_{\sigma^2}\log P(X|\theta)\\
    &=\arg\max_{\sigma^2}\sum_{i=1}^N\left[-\log\sigma-\frac{(x-\mu)^2}{2\sigma^2}\right]\\
    &=\arg\max_{\sigma^2}\mathcal{L}(\sigma^2)\\
\end{aligned}$$

$$\frac{\partial \mathcal{L}(\sigma^2)}{\partial \sigma^2}=\sum_{i=1}^N\left[-\frac{1}{\sigma}+(x_i-\mu)^2\sdot\sigma^{-3}\right]$$

令 $\frac{\partial \mathcal{L}(\sigma^2)}{\partial \sigma^2}=0$，得 $\sigma^2_{MLE}=\frac{\sum_{i=1}^N (x_i-\mu_{MLE})^2}{N}$ 此为对 $\sigma^2$ 的有偏估计，无偏估计为 $\hat{\sigma}=\frac{\sum_{i=1}^N (x_i-\mu_{MLE})^2}{N-1}$ 因为：

$$\begin{aligned}
    \mathbb{E}[\sigma_{MLE}^2]&=\mathbb{E}\left[\frac{\sum_{i=1}^N (x_i-\mu_{MLE})^2}{N}\right]\\
    &=\mathbb{E}\left[\frac{1}{N}\sum_{i=1}^Nx_i^2-\frac{2}{N}\sum_{i=1}^Nx_i\mu_{MLE}+\frac{1}{N}\sum_{i=1}^N\mu_{MLE}^2\right]\\
    &=\mathbb{E}\left[\frac{1}{N}\sum_{i=1}^Nx_i^2-2\mu_{MLE}^2+\mu_{MLE}^2\right]\\
    &=\mathbb{E}\left[\frac{1}{N}\sum_{i=1}^Nx_i^2-\mu_{MLE}^2\right]\\
    &=\mathbb{E}\left[\frac{1}{N}\sum_{i=1}^Nx_i^2-\mu_{MLE}^2\right]\\
    &=\mathbb{E}\left[(\frac{1}{N}\sum_{i=1}^Nx_i^2-\mu^2)-(\mu_{MLE}^2-\mu^2)\right]\\
    &=\mathbb{E}\left[\frac{1}{N}\sum_{i=1}^Nx_i^2-\mu^2\right]-\mathbb{E}\left[\mu_{MLE}^2-\mu^2\right]\\
    &=\mathbb{E}\left[\frac{1}{N}\sum_{i=1}^N(x_i^2-\mu^2)\right]-\left(\mathbb{E}\left[\mu_{MLE}^2\right]-\mu^2\right)\\
    &=\frac{1}{N}\sum_{i=1}^N\mathbb{E}\left[x_i^2-\mu^2\right]-\left(\mathbb{E}\left[\mu_{MLE}^2\right]-\mathbb{E}\left[\mu_{MLE}\right]^2\right)\\
    &=\frac{1}{N}\sum_{i=1}^N(\mathbb{E}\left[x_i^2\right]-\mu^2)-\mathrm{Var}[\mu_{MLE}]\\
    &=\sigma^2-\frac{1}{N}\sigma^2\\
    &=\frac{N-1}{N}\sigma^2
\end{aligned}$$

其中：
$$\begin{aligned}
    \mathrm{Var}[\mu_{MLE}]&=\mathrm{Var}[\frac{1}{N}\sum_{i=1}^N x_i]\\
    &=\frac{1}{N^2}\sum_{i=1}^N \mathrm{Var}[x_i]\\
    &=\frac{1}{N}\sigma^2\\
\end{aligned}$$

## 2 Multivariate Gaussian Distribution

现在，考虑多元高斯分布的情况，对于一个符合多元高斯分布的 $p$ 维变量，其分布可以用均值向量 $\mu=\begin{pmatrix}
    \mu_1\\\vdots\\\mu_p\\
\end{pmatrix}$ 和协方差矩阵 $\Sigma=\begin{pmatrix}
    \sigma_{11}&\dotsb&\sigma_{1p}\\\vdots&\ddots&\vdots\\\sigma_{p1}
    &\dotsb&\sigma_{pp}
\end{pmatrix}$ 描述。

协方差矩阵是一个半正定矩阵，通常是一个正定矩阵。

在多元高斯分布的概率密度函数中，$(x-\mu)^\top\Sigma^{-1}(x-\mu)$ 可以被看作 $x$ 与 $\mu$ 的**马氏距离**（的平方），当协方差矩阵是单位矩阵时，马氏距离就是欧氏距离。

当协方差矩阵是正定矩阵时，我们可以对其进行特征值分解：
$$\Sigma=U\Lambda U^\top=\sum_{i=1}^pu_i\lambda_iu_i^\top\\UU^\top=U^\top U=1,\Lambda=\mathrm{diag}(\lambda_1,\dotsb,\lambda_p)$$

同样地，我们也可以对其逆矩阵进行特征值分解：
$$\Sigma^{-1}=(U\Lambda U^\top)^{-1}=U\Lambda^{-1}U^\top=\sum_{i=1}^pu_i\frac{1}{\lambda_i}u_i^\top$$

因此，可以做以下代换：
$$(x-\mu)^\top\Sigma^{-1}(x-\mu)=\sum_{i=1}^p(x-\mu)^\top u_i\frac{1}{\lambda_i}u_i^\top(x-\mu)$$

这表明，在多元高斯分布中，概率值相等的一系列点在变量空间中分布于一个以均值为中心的超椭球面上（在 $p=2$ 时，即为椭圆）。

多元高斯分布有以下局限性：

- 对于一个 $p$ 维的变量，以多元高斯分布描述时参数量为 $O(p^2)$ 这对复杂问题不友好，因此有时会对协方差矩阵进行简化，例如假设其为一个对角矩阵；
- 单纯的高斯分布具有单峰值的特性，因此对于较复杂的数据分布会过度简化。

## 3 Marginal Probability Distribution & Conditional Probability Distribution

已知一个高斯分布，求其边缘概率分布和联合概率分布是一类常见的问题，例如：已知$X=\begin{pmatrix}X_a\\X_b\end{pmatrix}$，其中 $X_a\in\mathbb{R}^m$，$X_b\in\mathbb{R}^n$，且 $X\sim\mathcal{N}(\mu,\Sigma)$，其中 $\mu=\begin{pmatrix}\mu_a\\\mu_b\end{pmatrix}$，$\Sigma=\begin{pmatrix}\Sigma_{aa}&\Sigma_{ab}\\\Sigma_{ba}&\Sigma_{bb}\end{pmatrix}$，求 $P(X_a)$、$P(X_b)$、$P(X_a|X_b)$、$P(X_b|X_a)$。

有如下的定理：
$$X\sim\mathcal{N}(\mu,\Sigma),Y=AX+B\Rightarrow Y\sim\mathcal{N}(A\mu+B,A\Sigma A^\top)$$

那么由于：
$$X_a=\begin{pmatrix}I_m&O_n\end{pmatrix}\begin{pmatrix}X_a\\X_b\end{pmatrix}$$

因此：
$$\mathbb{E}[X_a]=\begin{pmatrix}I_m&O_n\end{pmatrix}\mu=\mu_a\\$$

$$\mathrm{Var}[X_a]=\begin{pmatrix}I_m&O_n\end{pmatrix}\Sigma\begin{pmatrix}I_m\\O_n\end{pmatrix}=\Sigma_{aa}$$

即 $X_a\sim\mathcal{N}(\mu_a,\Sigma_{aa})$，同理，$X_b\sim\mathcal{N}(\mu_b,\Sigma_{bb})$。

对于条件概率分布，我们定义：
$$X_{b-a}=X_b-\Sigma_{ba}\Sigma_{aa}^{-1}X_a=\begin{pmatrix}-\Sigma_{ba}\Sigma_{aa}&I\end{pmatrix}\begin{pmatrix}X_a\\X_b\end{pmatrix}$$

以类似上文的方式可以得到 $X_{b-a}$ 的均值及方差：

$$\mu_{b-a}=\mu_b-\Sigma_{ba}\Sigma_{aa}^{-1}\mu_a$$

$$\Sigma_{bb-a}=\Sigma_{bb}-\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}$$

$\Sigma_{bb-a}$ 的定义即块 $\Sigma_aa$ 在 $\Sigma$ 中的*舒尔补*（Schur complement）。

因此 $X_{b-a}\sim\mathcal{N}(\mu_{b-a},\Sigma_{bb-a})$，

那么由于 $X_b=X_{b-a}+\Sigma_{ba}\Sigma_{aa}^{-1}X_a$，因此：
$$E[X_b|X_a]=\mu_{b-a}+\Sigma_{ba}\Sigma_{aa}^{-1}X_a$$

$$\mathrm{Var}[X_b|X_a]=\mathrm{Var}[X_{b-a}]=\Sigma_{bb-a}$$

因此 $X_b|X_a\sim\mathcal{N}(\mu_{b-a}+\Sigma_{ba}\Sigma_{aa}^{-1}X_a,\Sigma_{bb-a})$，同理  $X_b|X_a\sim\mathcal{N}(\mu_{a-b}+\Sigma_{ab}\Sigma_{bb}^{-1}X_b,\Sigma_{aa-b})$。

## 4 Joint Probability Distribution

如果已知：
$$P(X)=\mathcal{N}(\mu,\Lambda^{-1})$$

$$P(Y|X)=\mathcal{N}(AX+b,L^{-1})$$

其中 $\Lambda$ 表示精度矩阵，其定义为协方差矩阵的逆。如果定义 $Z=\begin{pmatrix}X\\Y\end{pmatrix}$，如何表示 $Z$ 的概率分布（即 $X$、$Y$ 的条件概率分布）？

首先，可以对 $Y$ 进行重参数化：
$$Y=AX+B+\epsilon$$

$$\epsilon\sim\mathcal{N}(0,L^{-1})$$

则有：
$$\begin{aligned}
    \mathbb{E}[Y]&=\mathbb{E}[AX+B+\epsilon]\\
    &=\mathbb{E}[AX+B]+\mathbb{E}[\epsilon]\\
    &=A\mu+B\\
\end{aligned}$$

$$\begin{aligned}
    \mathrm{Var}[Y]&=\mathrm{Var}[AX+B+\epsilon]\\
    &=\mathrm{Var}[AX+B]+\mathrm{Var}[\epsilon]\\
    &=A\Lambda^{-1}A^\top+L^{-1}\\
\end{aligned}$$

那么可以写出；
$$Z\sim\mathcal{N}\left(\begin{bmatrix}\mu\\A\mu+B\end{bmatrix},\begin{bmatrix}\Lambda^{-1}&\Delta\\\Delta&A\Lambda^{-1}A^\top+L^{-1}\\\end{bmatrix}\right)$$

$$\begin{aligned}
    \Delta&=\mathrm{Cov}(X,Y)\\
    &=\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])^\top]\\
    &=\mathbb{E}[(X-\mu)(Y-A\mu-B)^\top]\\
    &=\mathbb{E}[(X-\mu)(AX+B+\epsilon-A\mu-B)^\top]\\
    &=\mathbb{E}[(X-\mu)(AX+\epsilon-A\mu)^\top]\\
    &=\mathbb{E}[(X-\mu)(AX-A\mu)^\top+(X-\mu)\epsilon^\top]\\
    &=\mathbb{E}[(X-\mu)(AX-A\mu)^\top]+\mathbb{E}[(X-\mu)\epsilon^\top]\\
    &=\mathbb{E}[(X-\mu)(AX-A\mu)^\top]\\
    &=\mathbb{E}[(X-\mu)(X-\mu)^\top]A^\top\\
    &=\Lambda^{-1} A^\top\\
\end{aligned}$$
