# Introduction —— Frequentists vs Bayesians

频率派和贝叶斯派提供了两种不同的将概率的概念引入机器学习模型的方法。

对于一系列采样自数据分布 $P_{data}$ 的样本数据 $X=(x_1,x_2,\dotsb,x_N)$，建模一个参数为 $\theta$ 的机器学习模型，频率派将 $\theta$ 看作一个未知的常量，其经典的参数估计方式是通过极大似然估计（MLE）：
$$
\theta_{MLE}=\arg\max_\theta\prod_i p(x_i|\theta)=\arg\max_\theta\sum_i\log p(x_i|\theta)
$$

因此**频率派的模型最终可以转化为一个优化问题**。

贝叶斯派模型将参数 $\theta$ 看作一个具有自身概率分布 $P(\theta)$ 的随机变量，其建模的基本原理是贝叶斯公式：
$$
P(\theta|x)=\frac{P(x|\theta)P(\theta)}{P(x)}
$$

贝叶斯公式中：

- $P(\theta)$ 称为先验分布
- $P(\theta|x)$ 称为后验分布
- $P(x|\theta)$ 称为似然分布
- $P(x)$ 称为边际分布

贝叶斯派使用两种基本的参数估计方法，一种称为极大后验估计（MAP），这种估计方式与极大似然估计相似：
$$
\theta_{MAP}=\arg\max_\theta P(\theta|x)=\arg\max_\theta \frac{P(x|\theta)P(\theta)}{P(x)}=\arg\max_\theta P(x|\theta)P(\theta)
$$

类似于一个带有先验的极大似然估计。

而贝叶斯估计与极大后验估计不同，贝叶斯估计不是寻找一个单一的最佳参数值，而是计算参数在整个可能取值范围上的完整后验分布。这意味着贝叶斯估计提供的是关于参数不确定性的全貌，而不是仅仅给出一个最优估计值。
$$
P(\theta|x)=\frac{P(x|\theta)P(\theta)}{P(x)}=\frac{P(x|\theta)P(\theta)}{\int_\theta P(x|\theta)P(\theta)\,\mathrm{d}\theta}
$$

这意味着贝叶斯估计的关键在于求解积分 $\int_\theta P(x|\theta)P(\theta)\,\mathrm{d}\theta$，也就是说**贝叶斯派的模型通常能够转化成一个求解积分的问题**。

如果我们能够建模后验分布 $P(\theta|x)$，那么我们就可以以之为桥梁，预测新的数据点产生自原数据分布的概率：
$$
P(\tilde{x}|x)=\int_\theta P(\tilde{x}|x,\theta)\,\mathrm{d}\theta=\int_\theta P(\tilde{x}|\theta)P(\theta|x)\,\mathrm{d}\theta
$$

此即贝叶斯预测的思想。
