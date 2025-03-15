# Sampling Methods

采样算法，或蒙特卡洛算法，是一类随机的近似推断方法，其解决的基本问题是关于一个概率分布 $p(z)$ 寻找某个函数 $f(z)$ 的期望：
$$
\mathbb E[f]=\int f(z)p(z)~\mathrm dz
$$
采样方法的基本思想是从概率分布 $p(z)$ 中独立抽取一组变量 $z^{(i)}$，其中 $l=1,2\dotsb,L$，这使得期望可以通过有线求和的方式估计：
$$
\hat f=\frac1L\sum_{i=1}^L f(z^{(i)})
$$
只要样本是从概率分布  $p(z)$ 中抽取的，那么 $\mathbb E[\hat f]=\mathbb E[f]$，因此估计 $\hat f$ 具有正确的均值，估计样本的方差为：
$$
\mathrm{Var}[\hat f]=\frac1L\mathbb E[(f-\mathbb E[f])^2]
$$
但是问题在于样本 $\{z^{(i)}\}$ 可能并并不独立，因此有效样本数量会远小于表观的样本数量，并且如果 $f(z)$ 在 $p(z)$ 较大的区域的值较小，或者在 $p(z)$ 较小的区域取值较大，那么期望可能由小概率的区域控制，为了达到足够的精度，需要较大的样本大小。

对于许多模型，联合概率分布可以由图模型确定，在没有观测变量的情况下，从联合概率分布 $p(z)=\prod_{i=1}^Mp(z_i\mid\mathrm {pa_i})$ 中采样是容易的，可以使用祖先采样方法，按照 $z_1,z_2,\dotsb,z_M$ 的顺序遍历一遍变量集合，对图遍历一次之后，我么们会得到来自联合概率的一个样本。

对于部分节点被观测值初始化的有向图的情况，我们可以对上面的采样方法进行推广，在每一个步骤中，我们得到变量 $z_i$ 的一个采样值，它的值被观测，并且将采样值与观测值相比较，如果它们相符，那么采样值被保留，算法继续进行，处理下一个变量，否则从第一个节点重新初始化采样。

对于由无向图定义的概率分布，不存在从没有观测变量的先验概率中一遍采样的方法。

## 1 Basic Sampling Methods
### 1.1 Standard Distributions

假设我们有一个均匀分布的随机数来源 $z\sim\mathrm{Norm}(0,1)$，我们使用某个函数 $f(\sdot)$ 对 $z$ 的值进行变换 $y=f(z)$，则有：
$$
p(y)=p(z)\left|\frac{\mathrm d z}{\mathrm d y}\right|
$$
这种情况下，$p(z)\propto 1$，我们的目标是选择一个函数 $f(z)$ 使得产生出的 $y$ 值具有某种所需的分布形式：
$$
z=h(y)=\int_{-\infty}^{y}p(\hat y)~\mathrm d \hat y
$$
因此，$y=h^{-1}(z)$，我们需要的变换函数是所求分布的累积密度函数（CDF）的反函数。

指数分布是一种可以使用这种采样方法的分布，对于一个指数分布 $p(y)=\lambda\exp(-\lambda y),0\leq y\leq \infty$，这种情况下，$h(y)=\int_{0}^y \lambda\exp(-\lambda y)\mathrm dy=1-\exp(-\lambda y)$，逆变换可以表示为 $y=-\lambda^{-1}\ln(1-z)$。

另一种可以应用变换方法的分布是柯西分布：
$$
p(y)=\frac{1}{\pi}\frac{1}{1+y^2}
$$
这种情况下，不定积分的反函数可以用 $\tan$ 函数表示。

对于多个变量情况的推广是容易的，只需要将导数的绝对值转变为 Jacobian 行列式：
$$
p(y_1,\dotsb,y_M)=p(z_1,\dotsb,z_M)\left|\frac{\partial(z_1,\dotsb,z_M)}{\partial(y_1,\dotsb,y_N)}\right|
$$
Box-Muller 方法可以用于生成高斯分布的样本，我们首先对 $(0,1)$ 上的两个均匀分布的变量 $z_1$、$z_2$ 使用 $z\rightarrow 2z-1$ 的方式进行变换，，然后我们丢弃那些不满足 $z_1^2+z_2^2\leq1$ 的点对，这产生出一个单单位圆内部的均匀分布，且 $p(z_1,z_2)=\frac1\pi$，然后对于每个点对 $z_1$、$z_2$，我们计算：z
$$
y_1=z_1\left(\frac{-2\ln r^2}{r^2}\right)^{1/2}\\
y_1=z_2\left(\frac{-2\ln r^2}{r^2}\right)^{1/2}\\
r^2=z_1^2+z_2^2
$$
 这样，$y_1$ 和 $y_2$ 的联合概率分布可以表示为：
$$
p(y_1,y_2)=p(z_1,z_2)\left|\frac{\partial(z_1,z_2)}{\partial(y_1,y_@)}\right|=\left[\frac{1}{\sqrt{2\pi}}\exp\left(\frac{-y_1^2}{2}\right)\right]\left[\frac{1}{\sqrt{2\pi}}\exp\left(\frac{-y_2^2}{2}\right)\right]
$$
因此采样得到的 $y_1$ 和 $y_2$ 是独立且符合高斯分布的两个随机变量，均值为 0 ，方差为 1。

### 1.2 Rejection Sampling

假设我们希望从任意的概率分布 $p(z)$ 中采样，假定我们可以计算对于任意给定的 $z$ 值的未归一化的概率 $\widetilde p(z)$，但归一化因子是未知的：
$$
p(z)=\frac1{Z_p}\widetilde p(z)
$$
拒绝采样的框架首先建立一个简单的概率分布，称为提议分布（proposal distribution），并且我们已经可以从这个简单的分布中采样。然后我们引入一个常数 $k$，它的值满足这样的性质：
$$
\forall z,kq(z)\geq \widetilde p(z)
$$
函数 $kq(z)$ 被称为比较函数，拒绝采样的每个步骤涉及到生成两个随机数，首先，我们从 $q(z)$ 中生成一个随机数 $z_0$，然后我们在区间  $[0,kq(z_0)]$ 上的均匀分布中生成一个随机数 $u_0$，最后，如果 $u_0>\widetilde p(z_0)$，那么样本被拒绝，否则样本被保留。

![rejection sampling](./images/12-1.png)

$z$ 的原始值从概率分布 $q(z)$ 中生成，这些样本之后被接受的概率为 $\frac{\widetilde p(Z)}{kq(z)}$ ，因此一个样本被接受的概率为：
$$
p(\mathrm{accept})=\int\left\{\frac{\widetilde p(z)}{kq(z)}\right\}q(z)~\mathrm dz=\frac1k\int\widetilde p(z)~\mathrm dz
$$
因此，为了达到最好的采样效果，我们应该调整 $k$ 的值，使得其尽量小，同时满足 $\forall z,kq(z)\geq \widetilde p(z)$。

### 1.3 Adaptive rejection sampling

通常确定提议分布 $q(z)$ 的一个合适的解析形式是困难的，另一种确定其函数形式的方式是基于 $p(z)$ 直接构建。对于 $p(z)$ 是凸函数的情况，即 $\ln p(z)$ 的导数是 $z$ 的单调非增函数时，界限函数的构建是相当简单的。

函数 $\ln p(z)$ 和它的切线在某些初始的格点处计算，⽣成的切线的交点被⽤于构建界限函数，然后，我们从界限分布中抽取一个样本值，由于界限函数的对数是由一系列线性函数组合而成，因此界限函数本身由一个分段的指数函数组成，其形式为：
$$
q(z)=k_i\lambda_i\exp\{-\lambda_i(z-z_i)\},\hat z_{i-1,i}<z\leq \hat z_{i,i+!}
$$
其中：

- $\hat z_{i-1,i}$ 是在点 $z_{i-1}$ 和 $z_i$ 处的切线的交点
- $\lambda_i$ 是切线在 $z_i$ 处的斜率
- $k_i$ 表示对应的偏移量

一旦样品点被抽取完毕，我们就可以使用通常的拒绝准则了，如果样本被拒绝，那么它被并入格点的集合中，计算出一条新的切线，从而界限函数被优化，提议分布的近似效果变好，拒绝的概率就会减小。

这个算法可以拓展到不是对数凹函数的概率分布中，只需要在每个拒绝采样的步骤中使用 Metropolis-Hasting 阶梯函数即可。

虽然拒绝采样在一维或二维空间中是一个有效的方法，但它不适用于高维空间，这是因为随着维度的上升，接受率会以指数的级别下降，在对高维空间的概率分布的采样中，它起着子过程的作用。

### 1.4 Importance Sampling

重要性采样并不提供从指定的概率分布中采集样本的方法，而是提供直接估计期望的方法。

假设直接从 $p(z)$ 中采样无法完成，但是对于任意给定的 $z$ 值。我们可以很容易地计算出 $p(z)$，一种简单的计算期望的方法是将 $z$ 空间离散化为均匀的格点，将被积函数使用求和的方式计算，形式为：
$$
\mathbb E[f]\simeq\sum_{l=1}^Lp(z^{(l)})f(z(l))
$$
这种方式的缺陷在于求和式中的项的数目随 $z$ 的维度指数增长，此外，我们感兴趣的概率分布通常将它们的大部分质量限制在 $z$ 空间的一个很小的区域，因此均匀采样十分低效，因此我们希望从 $p(z)$ 的值较大的区域采样，或者更理想地，从 $p(z)f(Z)$ 的值较大的区域采样。

重要性采样类似于拒绝采样，从提议分布  $q(z)$ 中采样，然后以 $q(z)$ 中的样本的有限和的形式表示期望：
$$
\begin{aligned}
\mathbb E[f]&=\int f(Z)p(Z)~\mathrm dz\\
&=\int f(Z)\frac{p(z)}{q(z)}q(z)~\mathrm dz\\
&\simeq\frac1L\sum_{l=1}^L\widetilde r_lf(z^{(l)})
\end{aligned}
$$
其中  $r_l=\frac{p(z)}{q(z)}$，称为重要性权重（importance weights），修正了由于错误的概率分布中采样引入的偏差，与拒绝采样不同，所有生成的样本均被保留。

常见的情况是，概率分布 $p(z)$ 的计算结果没有归一化，即 $p(z)=\widetilde p(z)/Z_p$，其中 $\widetilde p(z)$ 可以很容易地计算出来，而归一化因子是未知的，因此，我们可以使用重要性采样分布 $q(z)=\frac{\widetilde q(z)}{Z_q}$，其具有类似的性质：
$$
\begin{aligned}
\mathbb E[f]&=\int f(Z)p(Z)~\mathrm dz\\
&=\frac{Z_q}{Z_p}\int f(Z)\frac{\widetilde p(z)}{\widetilde q(z)}q(z)~\mathrm dz\\
&\simeq\frac{Z_q}{Z_p}\frac1L\sum_{l=1}^L\widetilde r_lf(z^{(l)})
\end{aligned}
$$


其中  $\widetilde r_l=\frac{\widetilde p(z)}{\widetilde q(z)}$，我们可以使用同样的有限求和的方式来计算比值 $\frac{Z_q}{Z_p}$，结果为：
$$
\frac{Z_p}{Z_q}=\frac1{Z_q}\int\widetilde p(z)~\mathrm dz=\int\frac{\widetilde p(z)}{\widetilde q(z)}q(z)~\mathrm dz\simeq\frac1L\sum_{l=1}^L\widetilde r_l
$$
因此：
$$
\mathbb E[f]\simeq\sum_{l=1}^Lw_lf(z^{(l)})
$$
其中我们定义：
$$
w_l=\frac{\widetilde r_l}{\sum_m\widetilde r_m}=\frac{{\widetilde p(z^{(l)})}/{q(z^{(l)})}}{\sum_m{\widetilde p(z^{(m)})}/{q(z^{(m)})}}
$$
重要性采样的成功严重依赖采样分布与目标分布的匹配程度，当目标分布变化剧烈同时相当多的概率质量集中于小部分区域时，重要性权重由几个较大的权值控制，剩余权值相对较小，因此有效的样本集的大小会比表面上的样本集的大小小得多。如果没有足够多的样本落在 $p(z)f(z)$ 较大的区域中，期望的估计可能明显偏离真实值，同时表面的方差可能极小，这种错误同时无法检测。

因此重要性采样的一个重要缺点是它具有产生任意错误的结果的可能性。因此一个隐性的要求是采样分布 $q(z)$ 不应该在 $p(z)$ 可能较大的区域中取得较小的值或为零的值。

这种⽅法的⼀个重要改进被称为似然加权采样

### 1.5 Sampling-Importance-Resampling

采样-重要性-重采样方法类似于拒绝采样，但是避免了常数 $k$ 的选择。

第一阶段，从协议分布 $q(z)$ 中采样 $L$ 个样本 $z^{(1)},\dotsb,z^{(L)}$，第二个阶段权值 $w_1,\dotsb,w_L$ 由公式 $w_l=\frac{\widetilde r_l}{\sum_m\widetilde r_m}=\frac{\frac{\widetilde p(z^{(l)})}{\widetilde q(z^{(l)})}}{\sum_m\frac{\widetilde p(z^{(m)})}{\widetilde q(z^{(m)})}}$ 构建，最后 $L$ 个样本的第二个集合从离散概率分布 $(z^{(1)},\dotsb,z^{(L)})$ 中抽取，概率由权值 $(w_1,\dotsb,w_L)$ 给出。

生成的样本只是近似地服从 $p(z)$：
$$
\begin{aligned}
p(z\leq a)&=\sum_{l:z^{(l)}\leq a}w_l\\
&=\frac{\sum_lI(z^{(l)}\leq a){\widetilde p(z^{(l)})}/{q(z^{(l)})}}{\sum_l {\widetilde p(z^{(l)})}/{q(z^{(l)})}}\\
\end{aligned}
$$
但是在极限 $L\rightarrow\infty$ 的情况下，分布就逼近正确的分布：
$$
\begin{aligned}
p(z\leq a)&=\frac{\int I(z^{(l)}\leq a)[{\widetilde p(z^{(l)})}/{q(z^{(l)})}]q(z^{(l)})~\mathrm d z}{\int[{\widetilde p(z^{(l)})}/{q(z^{(l)})}]q(z^{(l)})~\mathrm d z}\\
&=\frac{\int I(z\leq a)\widetilde p(z)~\mathrm dz}{\int \widetilde p(z)~\mathrm dz}\\
&=\int I(z\leq a)p(z)~\mathrm dz\\
\end{aligned}
$$
如果我们需要求出概率分布 $p(z)$ 的各阶矩，那么可以直接使用原始样本和权值进行计算，因为：
$$
\begin{aligned}
\mathbb E[f(z)]&=\int f(z)(z)~\mathrm d z\\
&=\frac{\int f(z)\left[{\widetilde p(z)}/{q(z)}\right]q(z)~\mathrm dz}{\int\left[{\widetilde p(z)}/{q(z)}\right]q(z)~\mathrm dz}\\
&\simeq\sum_{l=1}^Lw_lf(z^{(l)})
\end{aligned}
$$

### 1.6 Sampling and the EM Algorithm

对于一个隐含变量为 $Z$，可见变量为 $X$，参数为 $\theta$，在 M 步中关于 $\theta$ 最大化的步骤为完整数据对数似然的期望：
$$
Q(\theta,\theta^{old})=\int p(Z\mid X,\theta^{old})\ln p(Z,X\mid \theta)\mathrm dZ
$$
我们使用采样的方式近似这个积分，方法是计算样本 $\{Z^{(l)}\}$ 上的有限和，这些样本是从后验概率估计中抽取的：
$$
Q(\theta,\theta^{old})\simeq\frac1L\sum_{l=1}^L\ln p(Z^{(l)},X\mid \theta)
$$
然后 $Q$ 函数在 M 步骤中使用通常的方式进行优化，这个步骤被称为蒙特卡洛 EM 算法。

假设我们希望从联合后验概率分布 $p(\theta,Z\mid X)$ 中抽取样本，但是如果直接抽样是困难的，此时数据增广算法（data augmentation algorithm）是适用的，这个算法在两个步骤之间交替进行：

- 归咎步骤（imputation step）：我们注意到下面的关系：
  $$
  p(Z\mid X)=\int p(Z\mid \theta,X)p(\theta\mid X)~\mathrm d\theta
  $$
  因此对于 $l=1,\dotsb,L$，我们首先从当前对 $p(\theta\mid X)$ 的估计中抽取样本 $\theta^{(i)}$，然后使用这个样本从概率分布 $p(Z\mid \theta^{(i)},X)$ 中抽取样本 $Z^{(i)}$

- 后验步骤（posterior step）此时我们对 $Z$ 边缘化：

$$
p(\theta\mid X)=\int p(\theta\mid Z,X)p(Z\mid X)~\mathrm dZ
$$

​	此时我们使用抽取的样本更新 $\theta$ 上的后验分布的估计：
$$
p(\theta\mid X)\simeq \frac1L \sum_{l=1}^Lp(\theta\mid Z^{(i)},X)
$$


## 2 Markov Chain Monte Carlo

马尔科夫蒙特卡洛方法可以从一大类概率分布中采样，同时应对维度增长。

我们同样假定目标分布为 $p(z)=\frac{\widetilde p(z)}{Z_p}$，对于任意的 $z$ 值都可以计算 $Z_p$。从提议分布 $q(z)$ 中采样，但现在我们记录下采样时的状态  $z^{(\tau)}$ 以及依赖于当前状态的提议分布 $q(z\mid z^{(\tau)})$，从而样本序列组成了一个马尔科夫链。

在基本的 Metropolis 算法中，我们假定提议分布是对称的，即 $q(z_A\mid z_B)=q(z_B\mid z_A)$，因此候选样本被接受的概率为：
$$
A(z^*,z^{(\tau)})=\min \left(1.\frac{\widetilde p(z^*)}{\widetilde p(z^{(\tau)})}\right)
$$
采样可以这样进行：从 $(0,1)$ 的均匀分布中随机采样一个数 $u$，从 $q(z\mid z^{(\tau)})$ 中采样 ，如果 $u>A(z,z^{(\tau)})$，就接受这个样本，$z^{(\tau+1)}=z^*$，否则候选样本点被丢弃，$z^{(\tau+1)}=z^{(\tau)}$，然后从 $q(z\mid z^{(\tau+1)})$ 中再次采样。

在 Metropolis 算法中，当一个候选点被拒绝时，前一个样本点会被再次包含到最终的样本列表中，从而产生样本点的多个副本。在实际实现的过程中，每个样本之只会有一个副本，以及一个记录状态出现次数的因子。

只要对于任意的 $z_A$ 和 $z_B$ 都有 $q(z_A\mid z_B)$ 为正，那么当 $\tau \rightarrow\infty$ 时，$z^{(\tau)}$ 趋近于 $p(z)$。但是采集得到的序列并不是来自 $p(z)$ 的一组独立的样本，它们之间是高度相关的。如果我们希望保证样本的独立性，我们可以选择每  $M$ 个样保留一个样本，然后丢弃大部分样本。

### 2.1 Markov Chain

一阶 Markov chain 被定义为一系列随机变量 $z^{(1)},\dotsb,z^{(M)}$ 使得下面的条件独性质对于 $m\in\{1,\dotsb,M-1\}$ 成立：
$$
p(z^{(m+1)}\mid z^{(1)},\dotsb,z^{(m)})=p(z^{(m+1)}\mid z^{(m)})
$$
这可以表示为链形的有向概率图，我们将变量之间的条件概率定义为转移概率：
$$
T_m(z^{(m)},z^{(m+1)})=p(z^{(m)}\mid z^{(m+1)})
$$
如果对应于所有的 $m$，转移概率相同，那么称这个马尔科夫链是同质的。

对于一个特定的变量，边缘概率可以根据前一个变量的边缘概率用链式乘积的方式表示出来：
$$
p(z^{(m+1)})=\sum_{z^{m}}p(z^{(m+1)}\mid z^{(m)})p(z^{(m)})
$$
对于一个概率分布来说，如果马尔科夫链中的每一步都让这个概率分布保持不变，那么这个概率分布对于马尔科夫链是不变的，或者静止的，因此对于一个同质的马尔科夫链，如果：
$$
p^*(z)=\sum_{z'}=\sum_{z'}T(z,z')p^*(z')
$$
那么概率分布 $p^*(z)$ 是不变的，一个马尔科夫链可以有多个不变的概率分布，例如如果转移概率由恒等概率给出，那么任意的概率分布都是不变的。

确保所求概率 $p(z)$ 不变的一个充分非必要条件是令概率转移满足细节平衡：
$$
p^*(z)T(z,z')=p^*(z')T(z',z)
$$
满足关于特定概率分布的细节平衡性质的转移概率会使得那个概率分布具有不变性：
$$
\sum_{z'}p^*(z')T(z',z)=\sum_{z'}p^*(z)T(z,z')=p^*(z)\sum_{z'}p(z'\mid z)=p^*(z)
$$
满足细节平衡性质的马尔科夫链被称为可翻转（reversible）的。

我们的目的是使用马尔科夫链从一个给定的概率分布中采样，如果我们构造一个马尔科夫链使得所求的概率分布是不变的，那么我们就可以达到这个目的。此外，我们还要求 $m\rightarrow\infty$，概率分布  $p(z^{(m)})$ 收敛至所求的不变的概率分布 $p^*(z)$，与初始概率分布 $p(z^{(0)})$ 无关，这种性质被称为各态经历性（ergodicity），这个不变的概率分布被称为均衡分布 （equilibrium distribution），显然，一个具有各态经历性的马尔科夫链只能有唯一的均衡分布。

可以证明，同质的马尔科夫链具有各态经历性，只需对不变的概率分布和转移概率做弱限制即可。

我们可以从一组基转移 $B_1,\dotsb,B_K$ 中构建转移概率，方法是将各个基转移表示为混合概率分布，形式为：
$$
T(z',z)=\sum_{k=1}^K\alpha_kB_k(z',z)
$$
混合系数 $\alpha_1,\dotsb,\alpha_K$ 满足 $\alpha_k>0$ 且 $\sum_k\alpha_k=1$，除了这种线性的混合，基转移还可以通过连续的应用组合到一起：
$$
T(z',z)=\sum_{z_1}\dotsb\sum_{z_{K-1}}B_1(z',z_1)\dotsb B_{K-1}(z_{K-2},z_{K-1})B_K(z_{K-1},z)
$$
如果对于一个概率分布关于每个基转移都是不变的，由这两种混合方式产生的基转移显然都是不变的。

如果每个基转移都满足细节平衡，那么使用线性方式混合的混合转移也满足细节平衡 。

### 2.2 Metropolis-Hastings Algorithm

Metropolis-Hastings 算法是 Metropolis 算法的一个推广，在这种情况下，提议分布不是参数的对称函数，特别地，在第 $\tau$ 步中，当前状态为 $z^{(\tau)}$，我们从概率分布 $q_k(z\mid z^{(\tau)})$ 中抽取一个样本 $z^*$，然后以概率 $A(z^*,z^{(\tau)})$ 接受它：
$$
A(z^*,z^{(\tau)})=\min \left(1,\frac{\widetilde p(z^*)q_k(z^{(\tau)}\mid z^*)}{\widetilde p(z^{(\tau)})q_k(z^*\mid z^{(\tau)})}\right)
$$
$k$ 标记出可能的转移集合中的成员，接受准则的计算不需要知道概率分布  $p(z)=\frac{\widetilde p(z)}{Z_p}$ 中的归一化常数，对于一个对称的提议分布，Metropolis-Hastings 算法会退化为 Metropolis算法。

$p(z)$ 对于由 Metropolis-Hastings 算法定义的马尔科夫链是一个不变的概率分布，因为：
$$
\begin{aligned}
p(z)q_k(z'\mid z)A_k(z',z)&=\min(p(z)q_k(z'\mid z),p(z')q_k(z\mid z'))\\
&=\min(p(z')q_k(z\mid z'),p(z)q_k(z'\mid z))\\
&=p(z')q_k(z\mid z')A_k(z,z')
\end{aligned}
$$
即满足细节平衡条件。

提议分布的具体选择对算法的表现有重大影响，对于连续状态空间来说，一个常见的选择是使用一个以当前状态为中心的高斯分布，这需要在确定分布的方差参数时进行一个重要的折衷：

- 方差过小，则接受转移的比例会上升，但是遍历状态空间的形式是一个较慢的随机游走过程，时间开销大
- 方差过大，则拒绝率上升

如果概率分布在不同方向上的差异非常大，那么 Metropolis-Hastings 的收敛速度会十分慢。

## 3 Gibbs Sampling

吉布斯采样的每个步骤涉及到将一个变量的值替换为以剩余变量为条件，从这个概率分布中抽取的那个变量的值。因此我们将 $z_i$ 替换为从概率分布 $p(z_i\mid z_{\backslash i})$ 中抽取的值，这个步骤要么按照某种特定的顺序在变量之间进行循环，要么在每一步按照某个概率分布随机地选择一个变量进行更新。

吉布斯采样能够从正确的分布中采集样本是由两个因素保障的：

- 采样的过程中 $p(z)$ 是不变的
- 采样过程满足各态经历性

为了完成算法，初始状态的概率分布也应该被指定，虽然在多轮迭代之后，样本与初始状态的分布无关。

为了得到近似独立的样本，需要对采样得到的样本进行下采样。

我们可以将吉布斯采样的步骤看作 Metropolis-Hastings 算法的一个特定的情况：考虑一个 Metropolis-Hastings 采样的步骤，它涉及到变量 $z_k$，同时保持剩余变量 $z_{\backslash k}$ 不变，并且对于这种情况来说，从 $z$ 到 $z^*$ 的转移概率为 $q_k(z^*\mid z)=p(z_k^*\mid z_{\backslash k})$，同时，有 $z^*_{\backslash k}=z_{\backslash k}$，因为在一个采样步中其余元素保持不变，因此接受概率的因子可以表示为：
$$
A(z^*,z)=\min \left(1,\frac{\widetilde p(z^*)q_k(z^{(\tau)}\mid z^*)}{\widetilde p(z)q_k(z^*\mid z^{(\tau)})}\right)=\min\left(1,\frac{p(z^*_k\mid z^*_{\backslash k})p(z^*_{\backslash k})q_k(z_k\mid z^*_{\backslash k})}{p(z_k\mid z_{\backslash k})p(z_{\backslash k})q_k(z_k^*\mid z_{\backslash k})}\right)=\min(1,1)=1
$$
因此采样步骤总是可以接受的。

传统的吉布斯采样有这样的缺陷：

- 逐个更新变量，若变量间高度相关，相邻样本会高度依赖，导致链混合缓慢；
- 在相关性强的分布中，链可能需要大量步骤才能覆盖高概率区域。

⼀种减小吉布斯采样过程中的随机游走行为的⽅法被称为过松弛（over-relaxation）。对于一个特定分量 $z_i$，条件概率分布具有均值  $z_i$ 和方差 $\sigma_i^2$，在过度松弛框架中，$z_i$ 被替换为：
$$
z_i'=\mu_i+\alpha(z_i-\mu)+\sigma_i(1-\alpha^2)^{\frac12}v
$$
其中 $v$ 是一个随机变量，均值为 $0$，方差为 $1$，$\alpha$ 是一个参数，满足 $-1<\alpha<1$。对于 $\alpha=0$ 的情形，方法等价于标准的吉布斯采样，对于 $\alpha <0$，步骤会偏向于与均值相反的一侧。这个步骤使得所求的概率分布具有不变性，其效果是当变量高度相关时，鼓励状态空间中的直接移动。

另一种提升吉布斯采样效果的方法是分块吉布斯（blocking Gibbs）采样，将变量集合分块（未必互斥），然后在每个块内部联合地采样，采样时以剩余的变量为条件。

## 4 Slice Sampling

Metropolis 算法的一个困难之处是对于步长的敏感性。如果步长过小，那么由于随机游走行为，算法会很慢，如果步长过大，那么由于较高的拒绝率，算法会相当低效。切片采样方法提供了一个可以自动调节步长来匹配分布特征的方法。与之前一样，它需要我们能够计算未归一化的概率分布 $\widetilde p(z)$。

首先考虑一元变量的情形，切片采样使用额外的变量 $u$ 对 $z$ 进行增广，然后从联合子空间中采样。例如从：
$$
\begin{aligned}
\hat p(z,u)=\begin{cases}
\frac{1}{Z_p}&if~0\leq u\leq \hat p(z)\\
0&otherwise
\end{cases}
\end{aligned}
$$
均匀地采样，其中 $Z_p=\int \hat p(z)\mathrm{d} z$，$z$ 上的边缘概率分布为：
$$
\int\hat p(u,z)\mathrm du=\int_0^{\hat p(z)}\frac1{Z_p}\mathrm du=\frac{\hat p(z)}{Z_p}=p(z)
$$
因此我们可以通过从 $\hat p(z,u)$ 中采样，然后忽略 $u$ 值的方式得到 $p(z)$ 的样本。这可以通过交替地对两个变量采样完成，给定 $z$ 的值，我们可以计算 $\hat p(z)$ 的值，然后再 $0\leq u\leq\hat p(z)$上均匀地对 $u$ 进行采样，然后我们固定 $u$，在由 $\{z:\hat p(z)>u\}$ 定义的分布的切片上对  $z$ 进行均匀地采样。

在实际的采样过程中，直接从穿过概率分布的切片中采样很困难，因此我们定义了一个采样方法，假设 $z$  的当前值记作 $z^{(\tau)}$，并且我们已经得到了一个对应的样本 $u$，$z$ 的下一个值可以通过考察包含 $z^{(\tau)}$ 的区域 $z_{\min}\leq z\leq z_{\max}$ 来获得，我们希望区域包含尽可能多的切片，从而使得 $z$ 空间中能进行较大的移动，同时希望切片外的区域尽可能小，因为切片外的区域会使得采样变得低效。

一个选择区域的方式是，从一个包含 $z^{(\tau)}$ 的具有某个宽度 $w$  的区域开始，然后测试每个端点，看它们是否位于切片内部，如果有端点没有在切片内部，那么区域在增加 $w$ 值的方向上进行拓展，直到端点位于区域外，然后一个样本 $z'$ 被从这个区域中均匀抽取，如果 $z'$ 位于区域内，那么它就构成了 $z^{(\tau+1)}$，否则区域收缩，使 $z'$ 成为一个端点，并且区域中仍包含 $z^{(\tau)}$，然后另一个样本从缩小之后的区域中抽取，以此类推。

切片采样同样可以用于多元分布，方式是按照吉布斯采样的方式对每个变量进行采样，这要求对于每个变量 $z_i$，我们能够计算一个正比于 $p(z_i\mid z_{\backslash i})$ 的函数。

## 5 The Hybrid Monte Carlo Algorithm

### 5.1 Dynamical Systems

随机采样的动态方法起源于模拟哈密顿动力学下进行变化的物理系统的行为。通过将 MCMC 中的概率仿真转化为哈密顿系统的方式，我们可以利用哈密顿动力学的框架。

我们考虑的动力学系统描述连续时间 $\tau$ 下状态变量 $z=\{z_i\}$ 的演化，经典的动力学由牛顿第二定律描述，即物体的加速度正比于施加的力，对应于关于时间的二阶微分方程。我们可以将一个二阶微分方程分解为两个相互偶合的一阶方程，方法是引入动量 $r$ 作为中间变量，对应于状态 $z$ 的变化率：
$$
r_i=\frac{\mathrm dz_i}{\mathrm d\tau}
$$
从动力学的角度，$z_i$ 可以被看作位置变量，因此对于每个位置变量，都存在一个对应的动量变量，位置和动量组成的联合空间被称为相空间（phase space）。我们可以将概率分布写作：
$$
p(z)=\frac{1}{Z_p}\exp(-E(z))
$$
其中 $E(z)$ 可以看作状态 $z$ 处的势能，系统的加速度是动量的变化率，通过施加力的方式确定，力本身是势能的负梯度，即：
$$
\frac{\mathrm dr_i}{\mathrm d\tau}=-\frac{\partial E(z)}{\partial z_i}
$$
以上是牛顿动力学的框架，而使用哈密顿框架重新写出这个动态系统的公式是比较方便的，我们首先将动能定义为：
$$
K(r)=\frac12\|r\|^2=\frac12\sum_i r_i^2
$$
系统的总能量是势能和动能的和：
$$
H(z,r)=E(z)+K(r)
$$
其中 $H(\sdot)$ 被称为哈密顿函数（Hamiltonian function），使用上面的公式，我们可以将系统的动力学用哈密顿方程的形式表现出来：
$$
\frac{\mathrm dz_i}{\mathrm d\tau}=\frac{\partial H}{\partial r_i}\\
\frac{\mathrm dr_i}{\mathrm d\tau}=-\frac{\partial H}{\partial z_i}
$$
在动态系统变化的过程中，哈密顿函数的值是一个常数，这一点可以通过求微分的方式看出：
$$
\frac{\mathrm dH}{\mathrm d\tau}=\sum_i\left\{\frac{\partial H}{\partial z_i}\frac{\partial z_i}{\partial\tau}+\frac{\partial H}{\partial r_i}\frac{\partial r_i}{\partial\tau}\right\}=\sum_i\left\{\frac{\partial H}{\partial z_i}\frac{\partial H}{\partial z_i}+\frac{\partial H}{\partial r_i}\frac{\partial r_i}{\partial\tau}\right\}=0
$$
哈密顿动态系统的第二个重要性质是动态系统在相空间中体积不变，这被称为 Liouville 定理，换句话说，如果我们考虑变量 $(z,\tau)$ 空间中的一个区域，那么当这个区域在哈密顿动态方程下变化时，它的形状可能会发生变化，但体积不变。我们注意到流场（位置在相空间的变化率）为：
$$
V=\left(\frac{\mathrm dz}{\mathrm d\tau},\frac{\mathrm dr}{\mathrm d\tau}\right)
$$
这个场的散度为 $0$，即
$$
\mathrm{div}V=\sum_i\left\{\frac{\partial}{\partial z_i}\frac{\mathrm d z_i}{\mathrm d\tau}+\frac{\partial}{\partial r_i}\frac{\mathrm d r_i}{\mathrm d\tau}\right\}=\sum_i\left\{\frac{\partial}{\partial z_i}\frac{\mathrm dH}{\mathrm dr_i}+\frac{\partial}{\partial r_i}\frac{\mathrm dH}{\mathrm dz_i}\right\}=0
$$
现在考虑相空间上的联合概率分布，它的总能量是哈密顿函数，即概率分布的形式为：
$$
p(z,r)=\frac{1}{Z_H}\exp(-H(z,r))
$$
使⽤体系的不变性和 $H$ 的守恒性，可以看到哈密顿动态系统会使得 $p(z, r)$ 保持不变。因此，在一段有限时间内，相空间的一个小区域中，哈密顿函数的值不变，因此概率密度也不会发生改变。

虽然哈密顿函数的值是不变的，但是 $z$ 和 $r$ 的值会发生变换，因此通过在一个有限的时间间隔上对哈密顿动态系统积分，我们就可以让 $z$ 以一种系统化的方式发生较大的变化，避免了随机游走的行为。

然而，哈密顿动态系统的变化对 $p(z,r)$ 的采样不具有各态经历性，因为 $H$ 的值是一个常数，为了得到一个具有各态历经性的采样⽅法，我们可以在相空间中引⼊额外的移动，这些移动会改变 $H$ 的值，同时保持了概率分布 $p(z,r)$ 的不变性，达到这个目的最简单的方式是将 $r$ 的值替换为一个从以 $z$ 为条件的概率分布中抽取的样本，这可以被看作吉布斯采样的步骤。$p(z,r)$ 中 $z$ 和 $r$ 的概率分布是独立的，概率分布 $p(r\mid z)$ 是高斯分布，从中采样是容易的。

在这种⽅法的⼀个实际应⽤中，我们必须解决计算哈密顿⽅程的数值积分的问题。这会引入一些数值误差。可以证明，能够在 Liouville 定理仍然精确成⽴的条件下，对积分⽅法进⾏修改。蛙跳（leapfrog）离散化是一个可行的解决方案，这种方案对位置变量和动量变量的离散时间近似 $\hat z$ 和 $\hat r$ 进行交替更新：
$$
\hat r_i\left(\tau+\frac{\epsilon}{2}\right)=\hat r_i(\tau)-\frac{\epsilon}{2}\frac{\partial E}{\partial z_i}(\hat z(\tau))\\
\hat z_i(\tau+\epsilon)=\hat z_i(\tau)+\epsilon\hat r_i\left(\tau+\frac{\epsilon}{2}\right)\\
\hat r_i\left(\tau\epsilon\right)=\hat r_i(\tau+\frac{\epsilon}{2})-\frac{\epsilon}{2}\frac{\partial E}{\partial z_i}(\hat z(\tau+\epsilon))\\
$$
这种方法对动量变量的更新方式是步长为 $\frac{\epsilon}{2}$ 的半步更新，对位置变量的更新是步长为 $\epsilon$ 的整步更新。

与基本的 Metropolis ⽅法不同，哈密顿动⼒学⽅法能够利⽤对数概率分布的梯度信息以及概率分布本⾝的信息。⼤多数可以得到梯度信息的情况下，使⽤哈密顿动⼒学⽅法是很有优势的。

### 5.2 Hybrid Monte Carlo

在哈密顿动力学中，一个非零的步长会在积分过程中引入误差，混合蒙特卡洛将哈密顿动态系统与 Metropolis 算法相结合，因此消除了与离散化过程关联的任何偏差。

算法使用了一个马尔科夫链，它由对动量变量 $r$ 的随机更新以及使用蛙跳算法对哈密顿动态系统的更新交替组成，在每次应用蛙跳算法后，基于哈密顿函数 $H$ 的值，确定 Metropolis 准则，确定生成的候选状态被接受或者拒绝。因此，如果 $(z,r)$ 是初始状态，$(z^*,r^*)$ 是积分之后的状态，那么候选状态被接受的概率为：
$$
\min(1,\exp\{H(z,r)-H(z^*,r^*)\})
$$
如果蛙跳积分完美地模拟了哈密顿动态系统，那么每个这种候选状态都会⾃动地被接受，因为 $H$ 的值会保持不变。由于数值误差，$H$ 的值可能减小，因此我们希望 Metropolis 准则将这种效果引发的任何偏差都消除，并且确保得到的样本确实是从所需的概率分布中抽取的。为了完成这件事，我们需要确保对应于蛙跳积分的更新⽅程满⾜细节平衡，因此对蛙跳积分方式进行这样的修改：

- 开始蛙跳积分序列之前，我们等概率地随机选择是沿着时间向前的方向积分（步长为 $\epsilon$），还是沿着时间向后的方向积分（步长为 $-\epsilon$）。我们首先注意到常规的蛙跳积分方法是可翻转的，因此 $L$ 步后向的积分会抵消 $L$ 步前向的积分；
- 然后我们可以证明蛙跳积分精确地保持了相空间的体积不变性；
- 我们使⽤这些结果证明细节平衡是成⽴的，考虑相空间的一个小区域 $\mathcal R$，它在 $L$ 次步长为 $\epsilon$ 的蛙跳迭代之后被映射到了区域 $\mathcal R'$，使用在蛙跳迭代下体积的不变性，我们看到如果 $\mathcal R$ 的体积是 $\delta V$，那么 $\mathcal R'$ 的体积也是，如果我们从概率分布中选择一个初始点，然后使用 $L$ 次蛙跳进行更新，那么从区域 $\mathcal R$ 转移到 $\mathcal R'$ 的概率为：

$$
\frac{1}{Z_H}\exp(-H(\mathcal R))\delta V\frac12\min\{1,\exp(H(\mathcal R)-H(\mathcal R'))\}
$$

​	类似地，从区域 $\mathcal R'$ 开始，沿着时间的反方向回到区域 $\mathcal R$ 的概率为：
$$
\frac{1}{Z_H}\exp(-H(\mathcal R'))\delta V\frac12\min\{1,\exp(H(\mathcal R')-H(\mathcal R))\}
$$
​	可以看出，两个概率是相等的，因此其满足细节平衡。

考虑具有独立分量的高斯分布 $p(z)$，它的哈密顿函数为：
$$
H(z,r)=\frac12\sum_i\frac{1}{\sigma_i^2}z_i^2+\frac12\sum_ir_i^2
$$
我们的结论对于分量之间具有相关性的高斯分布同样适用，因为混合蒙特卡洛算法具有旋转不变性。在积分阶段，它依赖于所有变量的值，因此，任何变量的一个较大的积分误差会产生一个较高的拒绝概率，为了让离散蛙跳积分对真实的连续时间动态系统产生一个较好的近似，有必要让蛙跳积分的标度 $\epsilon$ 小于势函数变化的最短的长度标度。这由 $\sigma_i$ 的最小值 $\sigma_{\min}$ 控制。为了使得状态在相空间中移动较大的距离产生与初始状态相对独立的新状态，并达到较高的接受率，蛙跳积分迭代的次数是 $\sigma_\max/\sigma_\max$ 的量级。

对于一个简单的 Metropolis 算法，假设其采用各向同性的高斯分布，方差为 $s^2$。为了避免高拒绝率，$s$ 的值应当设置为 $\sigma_\min$ 的量级，其对状态空间的探索通过随机游走进行，达到近似独立的状态所需的步骤数是 $(\sigma_\max^2/\sigma^2_\min)$ 的量级。
