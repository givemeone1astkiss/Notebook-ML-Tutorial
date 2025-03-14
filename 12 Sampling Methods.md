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

