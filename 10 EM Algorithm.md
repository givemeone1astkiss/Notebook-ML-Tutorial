# EM Algorithm

EM 算法是一种专门用于潜在变量模型的极大似然估计算法。

## 1 K-Means

K 均值聚类算法是一种非概率的聚类算法。对于一个由 $D$ 维欧氏空间中的随机变量 $x$ 的 $N$ 次观测构成的数据集 $\{x_1,\dotsb,x_N\}$，算法的目标是将数据集划分为 $K$ 个类别。

对于聚类任务，一个直觉是聚类内部点之间的距离应当小于数据点与聚类外部的点之间的距离。为了度量这两个距离，我们引入一组向量 $\{\mu_k\}, k=1,2,\dotsb,K$，其代表了聚类的中心，我们的任务是使得每个数据点和与它最近的向量 $\mu_k$ 之间的距离的平方和最小。

以 “1-of-K” 的方式定义一组二值指示函数表示数据点的聚类情况，对于每个数据点 $x_n$，有 $r_{nk}\in\{0,1\},k=1,2,\dotsb,K$，如果算法将 $x_n$ 分配至类别 $i$，则 $r_{ni}=1,r_{nj}=0,i\not = {j}$。同时定义目标函数:
$$
\mathcal{J}=\sum_{n=1}^N\sum_{k=1}^Kr_{nk}\Vert x_n-\mu_k\Vert ^2
$$
它表示每个数据点与它被分配的向量 $\mu_k$ 之间距离的平方和，算法的目标可以被描述为找到 $\{r_{nk}\}$ 和 $\{\mu_k\}$ 的值，使 $\mathcal{J}$ 达到最小值。

在 K-means 算法中，我们通过两个独立的算法步骤进行优化：

- **E-step：**在 E 步中，我们固定 $\mu_k$ ，并对 $r_{nk}$ 进行优化
- **M-step：**在 M 步中，我们固定 $r_{nk}$，并对 $\mu_k$ 进行优化

首先考虑 E 步，我们可以通过将 $x_n$ 的类别设置为距离最近的聚类中心的类别完成这一点：
$$
r_{nk}=
\begin{cases}
1&k=\arg\min_j\Vert x_n-\mu_j\Vert ^2\\
0&otherwise
\end{cases}
$$
然后考虑 M 步，由于目标函数是一个关于 $\mu_k$ 的二次函数，令其关于 $\mu_k$ 的导数为零：
$$
2\sum_{n=1}^Nr_{nk}(x_n-\mu_k)=0
$$
可以解出：
$$
\mu_k=\frac{\sum_nr_{nk}x_n}{\sum_nr_{nk}}
$$
这个表达式的含义即是令 $\mu_k$ 为类别 $k$ 中所有数据点的均值。上述优化算法被称为 K-means 算法。

K 均值算法的基础是将平方欧几里得距离视为数据点与代表向量之间不相似度的度量。这不仅限制了能够处理的数据变量的类型（例如不能处理变量特征中含有类别标签的情况），而且聚类中心的确定对于异常点不具有鲁棒性。通过要引入一个更一般的不相似度的度量 $\mathcal{V}(x,x')$，然后将算法的目标更改为以下失真度量：
$$
\mathcal{J}=\sum_{n=1}^N\sum_{k=1}^Kr_{nk}\mathcal{V}(x_n,\mu_k)
$$
这被称为 K 中心点算法（K-medoids algorithm）。

在 K-means 算法中，可能会出现有数据点位于两个聚类中间位置的情况，在这种情况下，强行将数据点分配到距离最近的聚类是不合适的。

## 2 GMM

### 2.1 Introduction

高斯混合模型可以写作高斯分布的线性叠加的形式：
$$
p(x)=\sum_{k=1}^K\pi_k\mathcal{N}(x\mid \mu_k,\Sigma_k)
$$
此时引入一个 $K$ 元二值随机变量 $z$，其中一个特定的元素 $z_k=1$， 其余元素为 0， 其边缘概率分布根据混合系数进行赋值：
$$
p(z_k=1)=\pi_k,0\leq\pi_k\leq1,\sum_{k=1}^K\pi_k=1
$$

$$
p(z)=\prod_{k=1}^K\pi_k^{z_k}
$$

根据潜在变量的边缘概率分布 $p(z)$ 和条件概率分布 $p(x\mid z)$ 定义联合概率分布 $p(x,z)$ :
$$
p(x\mid z)=\prod_{k=1}^K\mathcal{N}(x\mid \mu_k,\Sigma_k)^{z_k}
$$

$$
p(x)=\sum_zp(x.z)=\sum_zp(z)p(x\mid z)=\sum_{k=1}^K\pi_k\mathcal{N}(x\mid \mu_k,\Sigma_k)
$$

另一个重要的量是隐变量的后验分布 $p(z\mid x)$，可以由贝叶斯定理推出：
$$
\gamma(z_k)=p(z=1\mid x)=\frac{p(z_k=1)p(x\mid z_k=1)}{\sum_{i=1}^Kp(z_i=1)p(x\mid z_i=1)}=\frac{\pi_k\mathcal{N}(x\mid \mu_k,\Sigma_k)}{\sum_{i=1}^K\pi_i\mathcal{N}(x\mid \mu_i,\Sigma_k)}
$$
$\gamma(z_k)$ 也可以被解释成分量 $k$ 对于“解释”观测值 $x$ 的“责任”。

假设我们有一个观测数据集 $\{x_1,\dotsb,x_N\}$，将其建模为一个 $N\times D$ 的矩阵 $X$。对应的隐变量表示成一个 $N\times D$ 的矩阵  $Z$，假定数据点独立地从概率分布中抽取，我们就可以将这个独立同分布数据集的高斯混合模型表示成这样的概率图：

![GMM](.\images\10.1.png)

对数似然函数表示为：
$$
\ln p(X\mid \pi,\mu,\Sigma)=\sum_{n=1}^N\ln \left[\sum_{k=1}^K\pi_k\mathcal{N}(x_n\mid \mu_k,\Sigma_k)\right]
$$
在 GMM 的框架下，奇异性的存在可能会引起一些问题，例如对于一个协方差矩阵形式为 $\sum_k=\sigma_k^2I$ 的高斯混合模型，假设模型的第 $j$ 个分量的均值 $\mu_j$ 与某个数据点完全一致，那么这个数据点对似然函数的贡献项是：
$$
\mathcal{N}(x_n\mid x_n,\sigma_j^2I)=\frac{1}{(2\pi)^\frac{1}{2}\sigma_j^2}
$$
如果考虑 $\sigma_j\rightarrow 0$，那么这一项会趋近于无穷大，因此，对数似然函数的极大化并非一个具有良好定义的问题。之所以单一的高斯分布不会出现这种问题，是因为如果单一的高斯分布退化到了一个具体的数据点上，那么它总会给由其他数据点产生的似然函数贡献可乘的因子 ，这些因子会以指数的速度趋于零，从而使整体的似然函数趋于零而不是无穷大。然而，由于混合模型中存在至少两个分量，其中一个分量会具有有限的方差，因此对所有数据点赋予一个有限的概率值，而另一个分量会收缩至一个具体的数据点，因此会给对数似然函数贡献一个不断增加的值。因此，在对此类模型进行极大似然估计时应该尽量避免出现这种病态解。

![GMM](.\images\10.2.png)

一种启发式的优化方法是如果检测到模型分量退化至一个样品点，就将其均值重新设置为一个随机的值，将其方差随机设置为一个较大的值，并继续优化。

另一个问题被称为模型的可区分（identifiability）问题，即对于任意的 $K$ 个分量的混合概率分布，总会有 $K!$ 个完全等价的解，表示相同的概率分布。

此外，由于对 $k$ 的求和出现在对数内部，这也使得直接令对数似然函数的导数为零并析出闭式解的方法不适用。

### 2.2 EM for GMM

使用 EM 算法优化 GMM 是可行的。

首先考虑似然函数优化的约束条件，令 $\ln p(X\mid \pi,\mu,\Sigma)$ 关于高斯分量的均值 $\mu_{k}$ 的导数为零：
$$
\sum_{n=1}^N\frac{\pi_k\mathcal{N}(x_n\mid \mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal{N}(x_n\mid \mu_j,\Sigma_j)}\Sigma_k^{-1}(x_n-\mu)=\sum_{n=1}^Nr(z_{nk})\Sigma_k^{-1}(x_n-\mu)=0
$$
然后整理可以得到：
$$
\mu_k=\frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})x_n\\
N_k=\sum_{n-1}^N\gamma(z_{nk})
$$
因此，第 $k$ 个高斯分量的均值可以通过对数据集中所有的数据点求加权平均的方式得到，权重为该分量对各变量的责任。

类似地，如果对协方差矩阵做相同的处理，我们可以得到：
$$
\Sigma_k=\frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})(x_n-\mu_k)(x-\mu_k)^\top
$$
这是类似的加权平均形式。

最后考虑混合系数，使用拉格朗日乘子的方式考虑归一化约束：
$$
L(\pi_k)=\ln(X\mid \pi,\mu,\Sigma)+\lambda\left(\sum_{k=1}^K\pi_k-1\right)\\
\frac{\partial L}{\partial \pi_k}=\sum_{n=1}^N\frac{\mathcal{N}(x_n\mid \mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal{N}(x_n\mid \mu_j,\Sigma_j)}+\lambda
$$
化简后可得：
$$
\pi_k=\frac{N_k}{N}
$$
这些公式并给出混合模型的解析解，因为责任本身以一种复杂的方式依赖于这些参数，然而这些公式使得我们可以通过一种简单的迭代方式寻找问题的极大似然解，具体来说，该迭代算法包含以下几个步骤：

- **初始化：**由于 EM 算法每轮迭代需要较大的计算量，因此为模型参数选择合适的初始值是十分重要的，通常先运行 K-means 算法寻找合适的初始值，均值和协方差矩阵被初始化为通过 K 均值算法找到的聚类的样本均值、协方差，混合系数被初始化为对应类别中数据点的占比。
- **E-step：**使用当前参数计算责任：

$$
\gamma(z_{nk})=\frac{\mathcal{N}(x_n\mid \mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal{N}(x_n\mid \mu_j,\Sigma_j)}
$$

- **M-step：**使用当前责任重新进行参数估计：

$$
\mu_k=\frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})x_n\\
\Sigma_k=\frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})(x_n-\mu_k)(x-\mu_k)^\top\\
\pi_k=\frac{N_k}{N}\\
N_k=\sum_{n-1}^N\gamma(z_{nk})
$$

- 计算对数似然函数，进行收敛判断，若为收敛则返回 E 步

## 3 Another Viewpoint

如果将模型的参数集合记作 $\theta$，将潜在变量集合记作 $Z$，将观测数据集合记作 $X$，则对数似然函数可以表示为：
$$
\ln p(X\mid \theta)=\ln\left[\sum_Zp(X,Z\mid \theta)\right]
$$
一个重要的观察是对潜在变量的求和位于对数内部，这使得即使联合概率分布 $p(X,Z\mid \theta)$ 属于指数族分布，由于这个求和式的存在，边缘概率分布 $p(X\mid \theta)$ 往往也不是指数族分布。求和的存在使得似然函数变得复杂。

假定对于每个观测，都有潜在变量的对应值，我们将 $\{X,Z\}$称为完整数据集，完整数据集的对数似然函数形式为 $\ln p(X,Z\mid \theta)$，我们假定对这个对数似然函数进行最大化是简单的。

然而实际上，我们并不能取得完整数据集，我们关于潜在变量 $Z$ 的知识仅仅来源于后验概率分布 $p(Z\mid X,\theta)$。由于我们不能使用完整数据的对数似然函数，因此我们反过来考虑它在后验概率分布下的期望，这对应 E 步骤，在接下来的 M 步骤中，我们对这个期望进行最大化。

在 E 步骤中，我们使用旧参数计算潜在变量的后验概率分布 $p(Z\mid X,\theta^{old})$，然后根据这个后验概率计算完整数据集似然函数的期望，这是一个关于一般参数的函数：
$$
\mathcal{Q}(\theta,\theta^{old})=\sum_Zp(Z\mid X,\theta^{old})\ln p(X,Z\mid \theta)
$$
在 M 步骤，我们进行参数更新以最大化该期望：
$$
\theta^{new}=\arg\max_\theta\mathcal{Q}(\theta,\theta^{old})
$$
EM 算法也可以用来寻找模型的极大后验估计解，在这种情况下，M 步中需要最大化的量为 $Q(\theta,\theta^{old})+\ln p(\theta)$，选择合适的先验分布可以消除模型参数的奇异性。

EM 算法同样适用于数据集中含有随机或非随机的缺失值的情况。

在 GMM 中，我们考虑对完整数据 $\{X,Z\}$ 进行极大似然估计，似然函数的形式是：
$$
p(X,Z\mid \mu,\Sigma,\pi)=\prod_{n=1}^N\prod_{k=1}^K[\pi_k\mathcal{N}(x_n\mid \mu_k,\Sigma^k)]^{z_{nk}}
$$
对数形式为：
$$
p(X,Z\mid \mu,\Sigma,\pi)=\sum_{n=1}^N\sum_{k=1}^Kz_{nk}[\ln\pi_k+\ln\mathcal{N}(x_n\mid \mu_k,\Sigma^k)]
$$
在这个形式下，对数运算直接作用于高斯分布，因此这样的形式是易于优化的。

然后我们考虑完整数据对数似然函数关于潜在变量后验概率分布的期望，这个后验概率分布的形式是：
$$
p(Z\mid X,\mu,\theta,\pi)\propto\prod_{n=1}^N\prod_{k=1}^K[\pi_k\mathcal{N}(x_n\mid \mu_k,\Sigma^k)]^{z_{nk}}
$$
在这个后验分布下，指示值的期望为：
$$
\mathbb{E}[z_{nk}]=\frac{\sum_{z_n}z_{nk}\prod_{k'}[\pi_{k'}\mathcal{N}(x_n\mid \mu_{k'},\Sigma_{k'})]^{z_{nk}}}{\sum_{z_n}\prod_j[\pi_j\mathcal{N}(x_n\mid \mu_j,\Sigma_j)]^{z_{nk}}}=\frac{\pi_k\mathcal{N}(x\mid \mu_k,\Sigma_k)}{\sum_{i=1}^K\pi_i\mathcal{N}(x\mid \mu_i,\Sigma_k)}=\gamma(z_{nk})
$$
于是完整数据的对数似然函数的期望是：
$$
\mathbb{E}_Z[\ln p(X,Z\mid \mu,\Sigma,\pi)]=\sum_{n=1}^N\sum_{k=1}^K\gamma(z_{nk})[\ln\pi_k+\ln\mathcal{N}(x_n\mid \mu_k,\Sigma_k)]
$$
以此方式得出的各参数的更新方式与之前推导得出的方式一致。

再来考察 K-means 算法与 GMM 的联系，实际上 K-means算法可以被看作 GMM 的一种极端情况。考虑这样一个 GMM，其中任意一个混合分量的协方差矩阵形式为 $\epsilon I$，$\epsilon$ 为一个在分量之间共享的常数。对于一个数据点 $x_n$，责任表示为：
$$
\gamma(z_{nk})=\frac{\pi_k\exp\left\{-\frac{\Vert x_n-\mu_k\Vert ^2}{2\epsilon}\right\}}{\sum_j\pi_j\exp\left\{-\frac{\Vert x_n-\mu_k\Vert ^2}{2\epsilon}\right\}}
$$
对于 $\epsilon\rightarrow 0$ 的极限情况，分类由软分类退化为硬分类，每个数据点都被分配为距离最近的均值的聚类。那么对均值的 EM 估计就简化为了 K-means 的形式。在这种情况下，完整数据的对数似然就变成了 ：
$$
\mathbb{E}_Z[\ln p(X,Z\mid \mu,\Sigma,\pi)]\rightarrow-\frac{1}{2}\sum_{n=1}^N\sum_{k=1}^Kr_{nk}\Vert x_n-\mu_k\Vert ^2+\mathrm{const}
$$
在经典的 K-means 算法中，聚类的协方差并没有被估计，一个带有协方差矩阵的硬分配版本的高斯混合模型被称为椭圆 $K$ 均值算法。

## 4 EM Algorithms in General Form

对于一个一般的潜在变量模型，其优化目标是最大化似然函数：
$$
p(X\mid \theta)=\sum_Zp(X,Z\mid \theta)
$$
直接对 $p(X\mid \theta)$ 进行优化是困难的，而优化 $p(X,Z\mid \theta)$ 则容易得多。接下来，引入一个定义在潜在变量空间上的分布 $q(Z)$，对于任意的 $q(Z)$，有以下的分解成立：
$$
\ln p(X\mid \theta)=\mathcal{L}(q,\theta)+KL(q(Z)\Vert p(Z\mid X))\\
\mathcal{L}(q,\theta)=\sum_Zq(Z)\ln\left[\frac{p(X,Z\mid \theta)}{q(Z)}\right]\\
KL(q(Z)\Vert p(Z\mid X))=-\sum_Zq(Z)\ln\left[\frac{p(Z\mid X,\theta)}{q(Z)}\right]
$$
$\mathcal{L}(q,\theta)$ 是一个关于 $\theta$ 的函数，也是一个关于 $q(Z)$ 的泛函，$\mathcal{L}(q,\theta)$ 包含了联合概率分布，而 $KL(q\Vert p)$ 包含了条件概率分布。由于 $KL(q\Vert p)=-\sum_Zq(Z)\ln\left[\frac{p(Z\mid X,\theta)}{q(Z)}\right]\geq0$ ，当且仅当 $q(Z)=p(Z\mid X)$ 时取等，因此可以认为 $\mathcal{L}(q,\theta)$ 是对数似然 $ln p(X\mid \theta)$ 的下界。

在 E 步中，下界 $\mathcal{L}(q,\theta)$ 关于 $q(Z)$ 被最大化，此时有 $\ln p(X\mid \theta)=\mathcal{L}(q,\theta)$，而 $\theta$ 保持不变，在接下来的 M 步中， $q(Z)$ 保持不变，下界关于 $\theta$ 进行最大化，这会使得对应的对数似然函数增大，由于概率分布 $q(Z)$ 由旧的参数值决定，并且在 M 步中保持固定，因此它不会与新的后验概率分布一致，同时 $KL(q\Vert p)\not ={0}$，因此，对数似然函数的增加量大于下界的增加量。

如果将 $q(Z)=p(Z\mid X,\theta^{old})$ 代入 $\mathcal{L}(q,\theta)$，在 E 步之后，$\mathcal{L}(q,\theta)$ 的形式是：
$$
\begin{aligned}
\mathcal{L}(q,\theta)&=\sum_Zp(Z\mid X,\theta^{old})\ln p(X,Z\mid \theta)-\sum_Zp(Z\mid X,\theta^{old})\ln p(Z\mid X,\theta^{old})\\
&=\mathcal{Q}(\theta,\theta^{old})+\mathrm{const}
\end{aligned}
$$
因此，对 $\mathcal{L}(q,\theta)$ 的优化实际上等价于对完整数据对数似然期望的优化。

![GMM](.\images\10.3.png)

那么我们可以看到，EM 算法的两个步骤均增大了对数似然函数的一个良好定义的下界值，并且完整的 EM 循环会使得模型的参数向对数似然函数最大化的方向改变。

对于使用 EM 算法进行极大后验估计的情况，我们可以对应地进行如下分解：
$$
\begin{aligned}
\ln p(\theta\mid X)&=\mathcal{L}(q,\theta)+KL(q(Z)\Vert p(Z\mid X))+\ln p(\theta)-\ln p(X)\\
&\geq\mathcal{L}(q,\theta)+\ln p(\theta)+\mathrm{const}\\
\end{aligned}
$$
类似地，我们可以对 $q$ 和 $\theta$ 两项轮流进行优化，关于 $q$ 的优化产生了与标准 EM 算法相同的 E 步骤，而 M 步中关于 $\theta$ 的优化则要求额外考虑 $\ln p(\theta)$。

## 5 Variants of the EM Algorithm

对于 M 步骤难以计算的问题，有一类推广 EM 算法（GEM algorithm），这类算法不关于 $\theta$ 对 $\mathcal{L}(q,\theta)$ 进行最大化，而是改变参数的值去增大 $\mathcal{L}(q,\theta)$ 的值。一种形式的 GEM 算法是在 M 步中使用某种非线性优化策略，例如共轭梯度法，另一种形式的 GEM 算法则是在每个 M 步骤中进行若干具有限制条件的最优化，被称为期望条件最大化算法（ECM algorithm）。

类似的我们可以对 E 步进行推广，对 $\mathcal{L}(q,\theta)$ 关于 $q$ 进行一个部分的最优化而不是完全的最优化。

在增量的 EM 算法中，每个循环只处理一个数据点。在 E 步骤中，我们只计算一个数据点的责任，而在 M 步中，如果混合的分量是指数族分布的成员，那么责任只会出现在简单的充分统计量中，这使得这些量可以高效地更新。例如，在 GMM 中，如果我们在某一个循环中选择了数据点 $x_m$，增量更新的形式可以被表示为：
$$
\mu_k^{new}=\mu_k^{old}+\left(\frac{\gamma^{new}(z_{nk})-\gamma^{old}(z_nk)}{N_k^{new}}\right)(x_m-\mu_k^{old})\\
N_k^{new}=N_k^{old}+\gamma^{new}(z_{nk})-\gamma^{old}(z_{nk})
$$
协方差矩阵和混合系数的更新公式与此类似。
