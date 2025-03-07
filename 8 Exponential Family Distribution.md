# Exponential Family Distribution

## 1 Basic Concepts

指数族分布是一类重要的概率分布，包括高斯分布、泊松分布、狄利克雷分布、Beta 分布等，一个指数族分布应该满足以下的形式：
$$
p(x\mid \eta) = h(x)\exp\left(\eta^T T(x) - A(\eta)\right)
$$
其中：

- $x$ 是随机变量
- $\eta$ 是自然参数/规范参数，决定了分布的具体形式
- $T(x)$ 是充分统计量，是对样本数据信息的描述
- $A(\eta)$ 是对数配分函数，确保表达式满足概率的归一化性质
- $h(x)$ 是基度量（base measure），通常是不依赖 $\eta$ 的非负函数

由于充分统计量 $T(x)$ 的存在，指数族分布可以**无需保留所有样品点的信息，而只需对充分统计量进行维护**，极大便利了实际计算和在线的学习过程。

指数族分布的一个重要特性是它们具有对应的**共轭先验分布**。这意味着，如果我们选择一个属于指数族的先验分布，并且似然函数也属于同一指数族，则后验分布也将属于该指数族。这种性质极大地简化了贝叶斯推断的过程，因为它避免了复杂的积分运算，而只需对后验分布的参数进行拟合。

## 2 Standard Form of Gaussian Distribution

标准形式下的单变量高斯分布的概率密度函数（PDF）可以表示为：

$$
p(x\mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

其中，$x$ 是随机变量，$\mu$ 是均值，$\sigma^2$ 是方差。

通过以下变形可以将其转换为指数族分布的标准形式：

$$
\begin{aligned}
    p(x\mid \mu, \sigma^2) &= \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{x^2 - 2x\mu + \mu^2}{2\sigma^2}\right)\\
    &= \exp(-\frac{1}{2}\log 2\pi\sigma^2) \exp\left(-\frac{x^2}{2\sigma^2} + \frac{x\mu}{\sigma^2} - \frac{\mu^2}{2\sigma^2}\right)\\
    &=\exp\left(\begin{pmatrix}
        \frac{\mu}{\sigma^2}&-\frac{1}{2\sigma^2}
    \end{pmatrix}\begin{pmatrix}
        x\\x^2
    \end{pmatrix} - (\frac{\mu^2}{2\sigma^2}+\frac{1}{2}\log 2\pi\sigma^2)\right)\\
\end{aligned}
$$
接下来，定义自然参数 $\eta$ 和其他相关组件：

- **自然参数**：令 $\eta_1 = \frac{\mu}{\sigma^2}$ 和 $\eta_2 = -\frac{1}{2\sigma^2}$，因此 $\eta = \begin{pmatrix}
        \frac{\mu}{\sigma^2}&-\frac{1}{2\sigma^2}
    \end{pmatrix}$。
- **充分统计量**：选择 $T(x) = \begin{pmatrix}
        x\\x^2
    \end{pmatrix}$。
- **基度量**：$h(x) = 1$。
- **对数配分函数**：$A(\eta) = -\frac{\eta_1^2}{4\eta_2} - \frac{1}{2}\log(-2\eta_2)$。

现在，我们把所有这些代入到指数族的标准形式中：

$$
p(x\mid \eta) = \underbrace{1}_{h(x)} \cdot \exp\left(\underbrace{\begin{pmatrix}
        \frac{\mu}{\sigma^2}&-\frac{1}{2\sigma^2}
    \end{pmatrix}}_{\eta}\underbrace{\begin{pmatrix}
        x\\x^2
    \end{pmatrix}}_{T(x)} - \underbrace{(\frac{\mu^2}{2\sigma^2}+\frac{1}{2}\log 2\pi\sigma^2)}_{A(\eta)}\right)
$$

## 3 Logarithmic Partition Function

由于概率的归一化性质，指数族分布的形式中一定存在某种形式的约束，我们进行以下推导：

$$
\int h(x)\exp\left(\eta^T T(x) - A(\eta)\right)\,\mathrm{d}x = 1
$$
对两边关于自然参数 $\eta$ 求导，得到：
$$
\int h(x) \exp\left(\eta^T T(x) - A(\eta)\right) (T(x) - \nabla_{\eta} A(\eta)) \,\mathrm{d}x = 0
$$

由于 $p(x\mid \eta) = h(x)\exp\left(\eta^T T(x) - A(\eta)\right)$，可以将上式改写为：
$$
\int p(x\mid \eta) (T(x) - \nabla_{\eta} A(\eta)) \,\mathrm{d}x = 0
$$

这意味着：
$$
\int p(x\mid \eta) T(x) \,\mathrm{d}x - \nabla_{\eta} A(\eta) \int p(x\mid \eta) \,\mathrm{d}x = 0
$$

因为 $\int p(x\mid \eta) \,\mathrm{d}x = 1$，我们有：
$$
E_{p(x\mid \eta)}[T(x)] = \nabla_{\eta} A(\eta)
$$

我们知道 $E[T(x)] = \nabla_{\eta} A(\eta)$，接下来求其方差。方差的定义是：
$$
\text{Var}[T(x)] = E[T(x)^2] - (E[T(x)])^2
$$

首先计算 $E[T(x)^2]$：
$$
E[T(x)^2] = \int p(x\mid \eta) T(x)^2 \,\mathrm{d}x
$$

根据之前的结果，我们知道：
$$
\nabla_{\eta} A(\eta) = E[T(x)]
$$

对方程两边再次关于 $\eta$ 求导，得到：
$$
\nabla_{\eta} E[T(x)] = \nabla_{\eta} \left( \int p(x\mid \eta) T(x) \,\mathrm{d}x \right)
$$

由于积分和微分可以交换顺序（在一定条件下），我们得到：
$$
\nabla_{\eta} E[T(x)] = \int \left( \nabla_{\eta} p(x\mid \eta) \cdot T(x) + p(x\mid \eta) \cdot \nabla_{\eta} T(x) \right) \,\mathrm{d}x
$$

注意到 $\nabla_{\eta} p(x\mid \eta) = p(x\mid \eta) (T(x) - \nabla_{\eta} A(\eta))$（由归一化条件得出），因此：
$$
\nabla_{\eta} E[T(x)] = \int \left( p(x\mid \eta) (T(x) - \nabla_{\eta} A(\eta)) T(x) + p(x\mid \eta) \nabla_{\eta} T(x) \right) \,\mathrm{d}x
$$

简化上式：
$$
\nabla_{\eta} E[T(x)] = \int p(x\mid \eta) T(x)^2 \,\mathrm{d}x - \nabla_{\eta} A(\eta) \int p(x\mid \eta) T(x) \,\mathrm{d}x + \int p(x\mid \eta) \nabla_{\eta} T(x) \,\mathrm{d}x
$$

由于 $\int p(x\mid \eta) T(x) \,\mathrm{d}x = E[T(x)] = \nabla_{\eta} A(\eta)$，并且 $\int p(x\mid \eta) \nabla_{\eta} T(x) \,\mathrm{d}x = E[\nabla_{\eta} T(x)]$，所以：
$$
\nabla_{\eta} E[T(x)] = E[T(x)^2] - (\nabla_{\eta} A(\eta))^2 + E[\nabla_{\eta} T(x)]
$$

对于指数族分布，$T(x)$ 与自然参数 $\eta$ 无关，因此$\nabla_{\eta} T(x) = 0$，从而：

$$
\nabla_{\eta} E[T(x)] = E[T(x)^2] - (\nabla_{\eta} A(\eta))^2
$$
但是，因为 $E[T(x)] = \nabla_{\eta} A(\eta)$，所以：
$$
\nabla_{\eta}^2 A(\eta) = E[T(x)^2] - (E[T(x)])^2
$$

最终，我们得到了方差的表达式：
$$
\text{Var}_{p(x\mid \eta)}[T(x)] = E[T(x)^2] - (E[T(x)])^2 = \nabla_{\eta}^2 A(\eta)
$$

## 4 Maximum Likelyhood Estimate

给定一组独立同分布（i.i.d.）的数据样本 $\{x_1, x_2, ..., x_n\}$，进行极大似然估计：

$$
\mathcal{J}(\eta) = \sum_{i=1}^{n} \log p(x_i\mid \eta)
$$

我们的目标是找到使得对数似然函数最大的参数 $\eta$：
$$
\eta^*=\arg\max_\eta\mathcal{J}(\eta)
$$
将指数族分布的形式代入对数似然函数中：
$$
\begin{aligned}
    \mathcal{J}(\eta) &= \sum_{i=1}^{n} \log \left( h(x_i)\exp\left(\eta^T T(x_i) - A(\eta)\right) \right)\\
    &=\sum_{i=1}^{n} \left[ \log h(x_i) + \eta^T T(x_i) - A(\eta) \right]\\
    &=\sum_{i=1}^{n} \log h(x_i) + \eta^T \sum_{i=1}^{n} T(x_i) - nA(\eta)\\
\end{aligned}
$$

为了找到极大似然估计值 $\hat{\eta}$，我们需要对 $\mathcal{J}(\eta)$ 关于 $\eta$ 求导并令其等于零：
$$
\begin{aligned}
    \nabla_{\eta} \mathcal{J}(\eta) &= \nabla_{\eta} \left( \sum_{i=1}^{n} \log h(x_i) + \eta^T \sum_{i=1}^{n} T(x_i) - nA(\eta) \right)\\
    &=\sum_{i=1}^{n} T(x_i) - n \nabla_{\eta} A(\eta)\\
    &\triangleq0\\
\end{aligned}
$$

根据指数族分布的性质，我们知道 $\nabla_{\eta} A(\eta) = E[T(x)]$。因此，极大似然估计条件变为：
$$
\sum_{i=1}^{n} T(x_i) - n E[T(x)] = 0
$$

即：
$$
\frac{1}{n} \sum_{i=1}^{n} T(x_i) = E[T(x)]
$$

这表明，为了最大化对数似然函数，自然参数 $\eta$ 应该选择使得充分统计量 $T(x)$ 的样本均值等于其期望值。

同时，该式也说明，我们无需保留所有的样品点数据，而只需要维护 $\frac{1}{n} \sum_{i=1}^{n} T(x_i)$ 即可通过 $\nabla_{\eta} A(\eta)$ 的反函数恢复最优参数。

## 5 Maximum Entropy Principle

给定一个离散随机变量 $X$ 取值为 $x$ 的概率为 $p(x)$，则该事件的信息量定义为：
$$
I(x) = -\log p(x)
$$

这里，对数的底数决定了信息量的单位；例如，以2为底得到的是比特（bit），而以自然对数 $e$ 为底则得到的是奈特（nat）。信息量衡量的是一个事件发生所提供的信息的多少，通常与该事件发生的概率成反比。具体来说，如果一个事件的概率越低，则它发生时提供的信息量越大。

熵是对随机变量不确定性的度量，也可以理解为系统平均信息量的期望值。对于一个离散随机变量 $X$，其熵 $H(X)$ 定义为所有可能取值的信息量的加权平均：
$$
H(X) = -\sum_{i} p(x_i) \log p(x_i)
$$

熵越高，表示系统的不确定性越大；反之，熵越低，表示系统的确定性越高或信息更加集中。

根据最大熵原理，在所有满足已知约束条件的概率分布中，具有最大熵的分布是最无偏的，即它包含了最少的假设信息。给定一个离散随机变量 $X$ 可能取值于有限集合 $\{x_1, x_2, ..., x_n\}$，其概率分布为 $p(x_i)$。我们需要找到使熵最大的分布：
$$
H(p) = -\int p(x) \log p(x) \,\mathrm{d}x~~~\mathrm{s.t.}\int p(x)\,\mathrm{d}x = 1
$$

使用拉格朗日乘数法解决这个问题。定义拉格朗日函数：
$$
\ell(p, \lambda) = -\int p(x) \log p(x) \,\mathrm{d}x + \lambda \left(1 - \int p(x) \,\mathrm{d}x\right)
$$

对 $p(x)$ 求变分并令其等于零：
$$
\frac{\delta \ell}{\delta p(x)} = -\log p(x) - 1 - \lambda = 0
$$

解得：
$$
p(x) = e^{-1-\lambda}
$$

利用归一化条件：
$$
\int_{-\infty}^{\infty} e^{-1-\lambda} \,\mathrm{d}x = 1
$$

$$
e^{-1-\lambda} \int_{-\infty}^{\infty} \,\mathrm{d}x = 1
$$

对于有限区间 $[a, b]$，我们有：
$$
e^{-1-\lambda} (b - a) = 1
$$

因此：
$$
p(x) = \frac{1}{b-a}, \quad x \in [a, b]
$$

这表明，在没有任何先验知识的情况下，最大熵对应的分布是均匀分布。

假设我们知道一些充分统计量 $T_j(x)$ 的期望值 $E[T_j(x)] = t_j$，其中 $j = 1, 2, ..., m$。那么我们需要在满足以下约束的情况下最大化熵：

$$
H(p) = -\int p(x) \log p(x) \,\mathrm{d}x~~~\,\mathrm{s.t.}\,\begin{cases}
    \int p(x) \,\mathrm{d}x = 1\\
\int p(x) T_j(x) \,\mathrm{d}x = t_j&\quad j = 1, 2, ..., m\\
\end{cases}
$$

再次使用拉格朗日乘数法，定义拉格朗日函数：
$$
\ell(p, \lambda, \alpha_j) = -\int p(x) \log p(x) \,\mathrm{d}x + \lambda \left(1 - \int p(x) \,\mathrm{d}x\right) + \sum_{j=1}^{m} \alpha_j \left(t_j - \int p(x) T_j(x) \,\mathrm{d}x\right)
$$

对 $p(x)$ 求变分并令其等于零：
$$
\frac{\delta \ell}{\delta p(x)} = -\log p(x) - 1 - \lambda - \sum_{j=1}^{m} \alpha_j T_j(x) = 0
$$

解得：
$$
p(x) = \exp\left(-1-\lambda - \sum_{j=1}^{m} \alpha_j T_j(x)\right)
$$

为了简化表示，我们可以写成：
$$
p(x) = \exp\left(\eta^T T(x) - A(\eta)\right)
$$

这是指数族分布的标准形式，其中，$\eta$ 是拉格朗日乘数 $\alpha_j$ 的向量形式，$A(\eta)$ 是配分函数，确保分布归一化。

通过归一化条件计算 $A(\eta)$：
$$
\int \exp\left(\eta^T T(x) - A(\eta)\right) \,\mathrm{d}x = 1
$$

从而：
$$
A(\eta) = \log \int \exp(\eta^T T(x)) \,\mathrm{d}x
$$
