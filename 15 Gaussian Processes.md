# Gaussian Processes

## 1 Kernel Method for Linear Regression

考虑一个经典的线性回归模型 $M$：
$$
y(x)=w^\top\phi(x)\\
p(w)=\mathcal N(w\mid 0,\alpha^{-1}I)
$$
它由一个超参数 $\alpha$ 控制，这个超参数表示分布的精度，在实际应用中，我们感兴趣的是计算函数 $y(x)$ 在某个具体的 $x$ 处的函数值，例如训练数据点处的函数值，因此我们感兴趣的实际上是函数值的概率分布。我们将函数值的集合记作 $\boldsymbol y$，其元素为 $y_n=y(x_n)$，因此：
$$
\boldsymbol y=\boldsymbol \Phi w
$$
其中 $\boldsymbol \Phi$ 表示设计矩阵，其元素为 $\Phi_{nk}=\Phi_k(x_n)$，由于 $\boldsymbol y$ 是由 $w$ 的元素给出的服从高斯分布的变量的线性组合，因此其本身是高斯分布，我们只需要确定其均值和方差，即可确定其具体的分布形式：
$$
\mathbb E[\boldsymbol y]=\boldsymbol \Phi\mathbb E[w]=0\\
\mathrm{cov}(\boldsymbol y)=\mathbb E[\boldsymbol y\boldsymbol y^\top]=\boldsymbol \Phi\mathbb E[\boldsymbol w\boldsymbol w^\top]\boldsymbol \Phi^\top=\frac{1}{\alpha}\boldsymbol\Phi\boldsymbol\Phi^\top=K
$$
其中 $K$ 是 Gram 矩阵：
$$
K_{nm}=k(x_n,x_m)=\frac1\alpha\phi(x_n^\top)\phi(x_m)
$$
$k(x,x')$ 表示核函数。

通常来说，高斯过程被定义为函数 $y(x)$ 上的一个概率分布，使得在任意点集 $x_1,\dotsb,x_N$ 处计算的 $y(x)$ 的值联合起来服从高斯分布。特别地，在输入向量为二维向量的情况下，被称为高斯随机场。

高斯随机过程的一个关键点是 $N$ 个变量 $y_1,\dotsb,y_N$ 上的联合概率分布完全由二阶统计量确定，在多数情况下我们对 $y$ 的均值没有任何先验知识，所以令其等于零，这等价于在一般的贝叶斯线性回归中令权重的均值为零，然后，$y$ 的协方差通过核函数确定：
$$
\mathbb E[y(x_n)y(x_m)]=k(x_n,x_m)
$$
高斯过程的优势在于我们可以绕过基函数的选择而直接定义核函数。

## 2 Gaussian Processes for Regression

为了把⾼斯过程模型应⽤于回归问题，我们需要考虑观测⽬标值的噪声，形式为：
$$
t_n=y_n+\epsilon_n
$$
其中 $y_n=y(x_n)$，$\epsilon_n$ 是一个随机噪声变量，它的值对于每个观测 $n$ 是独立的，这里我们要考虑服从高斯分布的噪声过程：
$$
p(t_n\mid y_n)=\mathcal{N}(t_n\mid y_n,\beta^{-1})
$$
其中 $\beta$ 是一个超参数，表示噪声的精度，由于噪声对于每个数据点是独立的，因此以 $\boldsymbol y=(y_1,\dotsb,y_n)^\top$ 为条件，目标值 $\boldsymbol t=(t_1,\dotsb,t_N)^\top$ 的联合概率分布是一个各向同性的高斯分布。形式为：
$$
p(\boldsymbol t\mid \boldsymbol y)=\mathcal N(\boldsymbol t\mid \boldsymbol y,\beta^{-1}I_N)
$$
根据高斯过程的定义，边缘概率分布 $p(\boldsymbol y)$ 是一个高斯分布，均值为零，协方差矩阵由 Gram 矩阵 K 定义：
$$
p(\boldsymbol y)=\mathcal N(\boldsymbol y\mid 0,K)
$$
核函数通常满足对于相似的点 $x_n$ 和 $x_m$，对应的值 $y(x_n)$ 和$y(x_m)$ 的相关性要大于不相似的点，而相似性的定义通常与实际应用有关。

为了找到以输入值 $x_1,\dotsb,x_N$ 为条件的边缘概率分布 $p(\boldsymbol t)$，我们对 $\boldsymbol y$ 积分，可以通过使用线性高斯模型的结果完成：
$$
p(\boldsymbol t)=\int p(\boldsymbol t\mid\boldsymbol y)p(\boldsymbol y)~\mathrm d\boldsymbol y=\mathcal N(\boldsymbol t\mid 0,\boldsymbol C)\\
\boldsymbol C(x_n,x_m)=k(x_n,x_m)+\beta^{-1}\delta_{nm}
$$
对于高斯过程回归，一个广泛使用的核函数的形式是指数项的二次型加上常数和线性项：
$$
k(x_n,x_m)=\theta_0\exp\left\{-\frac{\theta_1}{2}\|x_n-x_m\|^2\right\}+\theta_2+\theta_3x_n^\top x_m
$$
在给定一组训练数据的情况下，对新的输入变量 $x_{N+1}$ 预测目标变量 $t_{N+1}$，这要求我们计算预测分布 $p(t_{N+1}\mid \boldsymbol t_N,X_N,x_{N+1})$。我们首先写下联合概率分布 $p(\boldsymbol t_{N+1})$，其中 $\boldsymbol t_{N+1}$ 表示向量 $(t_1,\dotsb,t_N,t_{N+1})^\top$，这可以表示为；
$$
p(\boldsymbol t_{N+1})=\mathcal N(\boldsymbol t_{N+1}\mid 0,\boldsymbol C_{N+1})
$$
我们可以对 $C_{N+1}$ 进行分块：
$$
\boldsymbol C_{N+1}=\begin{pmatrix}\boldsymbol C_N&\boldsymbol k\\\boldsymbol k^\top&c\end{pmatrix}
$$
其中向量 $\boldsymbol k$ 的元素是 $k(x_n,x_{N+1}),n=1,\dotsb,N$，标量 $c=k(x_{N+1},x_{N+1})+\beta^{-1}$，然后我们可以从联合概率分布的表达式中推导出条件概率分布的表达式：
$$
p(t_{N+1}\mid \boldsymbol t)=\mathcal N(m(x_{N+1}),\sigma^2(x_{N+1}))\\
m(x_{N+1})=\boldsymbol k^\top \boldsymbol C_N^{-1}\boldsymbol t\\
\sigma^2(x_{N+1})=c-\boldsymbol k^\top\boldsymbol C^{-1}_N\boldsymbol k
$$
由于对于一个定义良好的高斯分布来说，协方差矩阵应当是正定的，即 $\boldsymbol C$ 一定是正定的，这就要求 $k(x_n,x_m)$ 一定是半正定的。

预测分布的均值可以写成；
$$
m(x_{N+1})=\sum_{n=1}^N a_nk(x_n,x_{N+1})
$$
其中，$a_n$ 表示 $\boldsymbol C_N^{-1}\boldsymbol t$ 的第 $n$ 个元素。

现在对比一下高斯过程与基函数线性回归的复杂度，二者均涉及矩阵求逆过程，对于高斯过程，其涉及求一个 $N\times N$ 的矩阵的逆，$N$ 表示样本点的数量，计算复杂度 $O(N^3)$，而对于线性基函数回归，其涉及求一个 $M\times M$ 的矩阵的逆，$M$ 表示基函数的数量，计算复杂度为 $O(M^3)$。两种方法均涉及向量与矩阵的乘法，在高斯过程中需进行 $O(N^2)$ 次，在基函数线性回归中需要进行 $O(M^2)$ 次，因此总而言之，如果基函数的数量比样本点的数量小，那么使用基函数框架是更高效的，但是高斯过程的优点是我们可以处理那些只能通过无穷多基函数表达的协方差函数。

## 3 Learning the Hyperparameters

⾼斯过程模型的预测部分依赖于协⽅差函数的选择。在实际应用中，我们不固定协方差函数，而是更喜欢使用一组带有参数的函数，然后从数据中推断参数的值，这下参数控制了相关性的长度缩放及噪声的精度等，对应于标准参数模型中的超参数。

学习超参数的方法基于计算似然函数 $p(\boldsymbol t\mid \boldsymbol \theta)$，其中 $\boldsymbol \theta$ 表示高斯过程模型的超参数，最简单的方式是通过最大化似然函数的方式进行 $\boldsymbol \theta$ 的点估计：
$$
\ln p(\boldsymbol t\mid \boldsymbol\theta)=-\frac12\ln|\boldsymbol C_N|-\frac12\boldsymbol t^\top\boldsymbol C_N^{-1}\boldsymbol t-\frac N2\ln(2\pi)\\
\frac{\partial}{\partial\theta_i}\ln p(\boldsymbol t\mid \boldsymbol\theta)=-\frac12\mathrm{Tr}\left(\boldsymbol C_n^{-1}\frac{\partial\boldsymbol C_N}{\partial \theta_i}\right)+\frac12\boldsymbol t^\top\boldsymbol C_N^{-1}\frac{\partial \boldsymbol C_N}{\partial\theta_i}\boldsymbol C_N^{-1}\boldsymbol t
$$
通常情况下 $\ln p(\boldsymbol t\mid \boldsymbol\theta)$ 是一个非凸的函数，因此其有多个极大值点。引入一个 $\boldsymbol \theta$ 上的先验然后使用基于梯度的方法最大化对数后验是容易的，在一个纯粹的贝叶斯方法中，我们需要计算 $\boldsymbol\theta$ 的边缘概率，乘以先验概率 $p(\boldsymbol\theta)$ 和似然函数 $p(\boldsymbol t\mid \boldsymbol\theta)$，然而精确的积分通常是不可行的，需要借助一些近似手段。

## 4 Automatic Relevance Determination

通过最大似然⽅法进⾏的参数最优化，能够将不同输入的相对重要性从数据中推断出来，这是高斯过程的自动相关性确定。

考虑二维输入空间 $\boldsymbol x=(x_1,x_2)$，有一个以下形式的核函数：
$$
k(x,x')=\theta_0\exp\left\{-\frac12\sum_{i=1}^2\eta_i(x_i-x_i')^2\right\}
$$
随着特定的 $\eta_i$ 的减小，函数逐渐对对应的输入变量 $x_i$ 不敏感，通过使用极大似然法按照数据集调整这些参数，它可以检测到对于预测分布几乎没有影响的输入变量，因为对应的 $\eta_i$ 会很小，这在实际应用中十分有效，因为它使得这些输入可以被遗弃。 

ARD 框架可以整合至指数-二次核中：
$$
k(\boldsymbol x_n,\boldsymbol x_m)=\theta_0\exp\left\{-\frac12\sum^D_{i=1}\eta_i(x_{ni}-x_{mi})^2\right\}+\theta_2+\theta_3\sum^D_{i=1}x_{ni}x_{mi}
$$

## 5 Gaussian Processes for Classification

对于一个二分类问题，我们可以定义函数 $a(\boldsymbol x)$ 上的高斯过程，然后使用 sigmoid 函数 $y=\sigma(a)$ 进行变换，那么我们就得到了函数 $y(\boldsymbol x)$ 上的一个非高斯随机过程，其中 $y\in(0,1)$。我们可以写出目标变量的条件分布形式：
$$
p(t\mid a)=\sigma(x)^t(1-\sigma(a))^{1-t}
$$
我们引入 $\boldsymbol a_{N+1}$ 上的高斯过程先验，它的分量为 $a(x_1),\dotsb,a(x_{N+1})$：
$$
p(\boldsymbol a_{N+1})=\mathcal N(\boldsymbol a_{N+1}\mid\boldsymbol 0,\boldsymbol C_{N+1})
$$
此时协方差矩阵不再包含噪声项，因为我们假设所有的训练数据点都被正确标注，但是为了计算的便利，我们可以人为地加入一个由参数 $v$ 控制的类似噪声的项，它可以确保协方差矩阵是正定的，因此 $\boldsymbol C_{N+1}$ 中的元素为：
$$
C(x_n,x_m)=k(x_m,x_m)+v\delta_{nm}
$$
其中 $k(x_n,x_m)$ 是一个任意的半正定核函数。对于二分类问题，有：
$$
p(t_{N+1}=1\mid \boldsymbol t_N)=\int p(t_{N+1}=1\mid a_{N+1})p(a_{N+1}\mid \boldsymbol t_N)~\mathrm da_{N+1}
$$
 其中 $p(t_{N+1}\mid a_{N+1})=\sigma(a_{N+1})$。这个积分无法解析求解，因此需要使用近似的方式求解。

一种近似方法基于变分推断，使用类似变分 logistic 回归中的局部变分近似，这使得 sigmoid 函数的乘积可以通过高斯的乘积近似，因此使得对 $a_N$ 的积分可以解析地计算。

另一种方法使用期望传播，这是由于真是后验概率是单峰分布，因此期望近似可以取得较好的效果。

还有一种近似方式基于拉普拉斯近似，这种框架通过二阶泰勒展开，在最大后验（MAP）估计点处用高斯分布近似后验。
