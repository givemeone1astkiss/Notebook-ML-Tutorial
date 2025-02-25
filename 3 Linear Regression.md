# Linear Regression

## 1 Least Square Estimate

对于一个数据集 $\mathcal{D}=\{\langle x_1,y_1\rangle\langle x_2,y_2\rangle\dotsb\langle x_N,y_N\rangle\}$，其中 $x_i\in\mathbb{R}^p$，$y_i\in\mathbb{R}$，$i=1,2,\dotsb,N$，以矩阵的形式表示样本和标签:
$$
X=\begin{pmatrix}
    x_1&x_2\dotsb&x_N
\end{pmatrix}^\top=\begin{pmatrix}
    x_{11}&x_{12}&\dotsb&x_{1p}\\
    x_{21}&x_{22}&\dotsb&x_{2p}\\
    \vdots&\vdots&\ddots&\vdots\\
    x_{N1}&x_{N2}&\dotsb&x_{Np}\\
\end{pmatrix}
$$

$$
Y=\begin{pmatrix}
    y_1\\y_2\\\dotsb\\y_N
\end{pmatrix}
$$

最小二乘估计（LSE）旨在寻找一个权重向量 $w$ 使之满足如下的损失函数最小化：
$$
\ell(w)=\sum_{i=1}^N\|w^\top x_i-y_i\|^2_2
$$

对损失函数进行如下变形：
$$
\begin{aligned}
    \ell(w)&=\sum_{i=1}^N(w^\top x_i-y_i)^2\\
    &=(w^\top X^\top-Y^\top)(Xw-Y)\\
    &=w^\top X^\top Xw-w^\top X^\top Y-Y^\top Xw+Y\top Y\\
    &=w^\top X^\top Xw-2w^\top X^\top Y+Y^\top Y\\
\end{aligned}
$$

优化目标可以写作：
$$
\begin{aligned}
    w^*=\arg\min_w \ell(w)
\end{aligned}
$$

对损失函数求偏导：
$$
\begin{aligned}
    \frac{\partial \ell(w)}{\partial w}&=2X^\top Xw-2X^\top Y
\end{aligned}
$$

因此，$w^*=(X^\top X)X^\top Y$，其中 $(X^\top X)X^\top$ 被称作矩阵 $X$ 的伪逆。

以上是对 $X$ 行空间的理解，还可以从列空间角度理解线性回归，即寻找向量 $Y$ 在 $X$ 列空间上的投影 $X\beta$，可以列方程：
$$
X^\top(Y-X\beta)=0
$$

即得 $\beta=(X^\top X)^{-1}X^\top=w$。

从概率论的视角理解 LSE，我们假定估计结果的噪声符合高斯分布：
$$
\hat{y}=f(w)+\epsilon=w^\top x+\epsilon,\epsilon\sim\mathcal{N}(0,\sigma^2)
$$

则有条件概率：
$$
y|x;w\sim\mathcal{N}(w^\top x,\sigma^2)
$$

以极大似然的角度来看：
$$
\begin{aligned}
    \mathcal{J}(w)&=\log P(Y|X;w)\\
    &=\sum_{i=1}^N \log p(y_i|x_i;w)\\
    &=\sum_{i=1}^N(\log \frac{1}{\sqrt{2\pi}\sigma}-\frac{1}{2\sigma^2}(y_i-w^\top x_i)^2)\\
\end{aligned}
$$

$$
\begin{aligned}
    w^*&=\arg\max_w \mathcal{J}(w)\\
    &=\arg\min_w (y_i-w^\top x_i)^2\\
    &=(X^\top X)X^\top Y\\
\end{aligned}
$$

因此，**最小二乘估计等价于噪声为高斯分布的极大似然估计**。

## 2 Regularization —— Ridge Regression

当输入的特征维度极大或样本量极小时，可能会出现所谓的“维度灾难”，在这种情况下，模型易发生过拟合，在数学上体现为 $w^*$ 表达式中 $X^\top X$ 不可逆，因此无法得到解析解在这种情况下有三种解决思路：

- 增加样本数量
- 降维/特征选择/特征提取
- 正则化

正则化的一般模式是在传统的损失函数上增加对参数空间的限制：
$$
w^*=\arg\min_w \ell(w)+\lambda P(w)
$$

当 $P(w)=\|w\|^2_2$ 时，称为岭回归（ridge regression）或 L2 正则化，而当 $P(w)=\|w\|$ 时，称为套索回归（Lasso regression）或 L1 正则化。

采用 L2 正则化时，优化的目标函数可以改写为：
$$
\begin{aligned}
    \ell(w)&=\sum_{i=1}^N \|w^\top x_i-y_i\|^2_2+\lambda\|w\|_2^2\\
    &=w^\top X^\top Xw-2w^\top X^\top Y+Y^\top Y+\lambda w^\top w\\
    &=w^\top (X^\top X+\lambda I)w-2w^\top X^\top Y+Y^\top Y\\
\end{aligned}
$$

$$
\begin{aligned}
    \frac{\partial \ell(w)}{\partial w}&=2(X^\top X+\lambda I)w-2w^\top Y\\
\end{aligned}
$$

因此，$w^*=(X^\top X+\lambda I)^{-1}X^\top Y$，在此情况下，$X^\top X+\lambda I$ 为一正定矩阵，因此一定可逆。

从贝叶斯角度来看 L2 正则化，我们同样认为预测噪声服从正态分布 $\mathcal{N}(0,\sigma^2)$，同时，认为权重服从正态分布 $w\sim\mathcal{N}(0,\sigma^2_w)$。

列贝叶斯方程：
$$
P(w|y)=\frac{P(y|w)P(w)}{P(y)}
$$

其中：
$$
P(y|w)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y-w^\top X)^2}{2\sigma^2})
$$

$$
P(w)=\frac{1}{\sqrt{2\pi}\sigma_w}\exp(-\frac{w^\top w}{2\sigma^2_w})
$$

从贝叶斯学派极大后验估计的方法来看，最小二乘可以转化成以下优化问题：
$$
\begin{aligned}
    w^*&=\arg\max_w P(w|y)\\
    &=\arg\max_w P(y|w)P(w)\\
    &=\arg\max_w \frac{1}{2\pi\sigma\sigma_w}\exp(-\frac{(y-w^\top X)^2}{2\sigma^2}-\frac{w^\top w}{2\sigma^2_w})\\
    &=\arg\min_w \frac{(y-w^\top X)^2}{2\sigma^2}+\frac{w^\top w}{2\sigma^2_w}\\
    &=\arg\max_w (y-w^\top X)^2+\frac{\sigma^2}{\sigma^2_w}w^\top w\\
\end{aligned}
$$

即与带有 L2 正则化的最小二乘优化目标一致，因此，**噪声和先验为高斯分布的极大后验估计等价于带有 L2 正则化的最小二乘估计**。
