# Kernel Method

由于 SVM 模型本身是基于线性可分的归纳偏置构建的，对于明显线性不可分的数据，SVM 是难以处理的，这种情况下*核方法*（kernel method）就可以发挥作用。

核函数 $K(x,x')$ 是定义在输入空间上的二元函数，它计算两个输入样本 $x$ 和 $x'$ 在某个特征空间中的内积，其可以表示为：
$$
K\lang\phi(x),\phi(x')\rang
$$

其中 $\phi$ 是一个由输入空间 $\mathbb{R}^p$ 向高维特征空间 $\mathbb{H}$ 的映射。核技巧的关键优势在于我们无需实际建模 $\phi$ 或显式地求出 $\phi(x)$，这避免了高维特征空间上的内积运算带来的巨大计算成本。

通常我们讨论的核函数均为**正定核函数**，其要求核函数$K:\mathcal{X}\times\mathcal{X}$ 满足：

- **对称性：**

$$
K(x_i,x_j)=K(x_j,x_i)
$$

- **Gram 矩阵 $K$ ($K_{ij}=K(x_i,x_j)$)正定或至少半正定：**

$$
\forall \alpha=\begin{pmatrix}
    \alpha_1&\dotsb&\alpha_n
\end{pmatrix}\not ={\mathbf{0}},\{x_1,\dotsb,x_n\}\subset\mathcal{X},\sum_{i=1}^n\sum_{j=1}^n\alpha_i\alpha_jK(x_i,x_j)\geq0
$$

而 **Mecer 定理**指出，对于任意一个正定的核函数 $K$，存在某个希尔伯特空间 $\mathcal{H}$ 及其对应的映射 $\phi:\mathcal{X}\rightarrow\mathcal{H}$ 使得 $\forall x,x'\in\mathcal{X}$，有：
$$
K(x,x')=\lang\phi(x),\phi(x')\rang_\mathcal{H}
$$

这里的希尔伯特空间是一个无限维的向量空间，即一个函数空间，被称为该核函数的**再生核希尔伯特空间（RKHS）**，该核函数被称为该希尔伯特空间的再生核函数。我们可以把该内积运算看作两个无限维向量，即函数之间的内积：
$$
K(x,x')=\lang K(x,\sdot),K(\sdot,x')\rang_{\mathcal{H}}
$$

因此，我们可以认为，在使用正定核函数时，内积操作所在的隐空间是一个无限维的空间。

当使用核技巧时，SVM 的优化目标可以改写为：
$$
\begin{cases}
    \min_\lambda\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_jK(x_i,x_j)+\sum_{i=1}^N\lambda_i\\
    \mathrm{s.t.}~\lambda_i\geq0
\end{cases}
$$

常用的正定核函数有：

- **线性核函数（Linear Kernel）：**

$$
K(x,x')=x^\top x'
$$

- **多项式核（Polynomial Kernel）：**

$$
K(x,x')=(x^\top x+c)^d,c\geq0,d>0
$$

- **高斯径向基核函数（RBF Kernel）：**

$$
K(x,x')=\exp(-\gamma\|x-x'\|_2^2),\gamma>0
$$

- **Sigmoid 核（Sigmoid Kernel）：**

$$
K(x,x')=\tanh(\kappa x\sdot x'+\theta)
$$

- **拉普拉斯核（Laplacian Kernel）：**

$$
K(x,x')=\exp(-\gamma\|x-x'\|_1),\gamma>0
$$
