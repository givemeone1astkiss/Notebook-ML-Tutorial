# Linear Classification

## 1 Background

线性分类模型是一类基于线性回归的模型，大致可以分为以下类型：

- **硬分类模型：** 这类模型直接以标签 $\{0,1\}$ 作为输出，典型代表是线性判别分析（LDA）和感知机
- **软分类模型：** 这类模型以概率作为输出
  - **概率判别模型：** 这类模型建模条件概率 $P\{Y|X\}$，典型代表是逻辑斯蒂回归
  - **概率生成模型：** 这类模型建模联合概率分布 $P\{XY\}$，典型代表是高斯判别分析（GDA）和朴素贝叶斯

## 2 Perceptron Algorithm

感知机模型是一类硬分类模型，可以简单地表示为：
$$f(x)=\mathrm{sign}(w^\top x),x\in\mathbb{R}^p,w\in\mathbb{R}^p$$

其中 $\mathrm{sign}(\sdot)$ 表示符号函数：
$$\mathrm{sign}(a)=\begin{cases}
    1&a\geq 0\\
    -1&a<0\\
\end{cases}$$

现在考虑应该使用何种形式的损失函数，一种想法是使用被错误分类的点的个数作为损失函数：
$$\mathcal{L}(w)=\sum_{i=1}^N I\{y_iw^\top x_i<0\}$$

$I$ 表示指示函数，当 $\{\}$ 中的条件满足时，其值为 1，否则为 0。这种表示方式的弊端是指示函数是一个简单的二值函数，不可微，因此难以优化。因此感知机采用以下的函数：
$$\mathcal{L}(w)=\sum_{x_i\in D}-y_iw^\top x_i\\D:\{被错分的样本\}$$

$$\nabla_w\mathcal{L}=-y_ix_i$$

然后使用常规的随机梯度下降进行优化。

## 3 Linear Discriminant Analysis (LDA)

LDA 是一类基于降维的分类模型，其将样本点投影至一维空间，然后通过投影点与阈值的大小关系进行判别。

假设样本矩阵为 $X=\begin{pmatrix}x_1&x_2&\dotsb&x_N\end{pmatrix}^\top$，标签矩阵为 $Y=\begin{pmatrix}y_1&y_2&\dotsb&y_N\end{pmatrix}^\top$，其中 $x_i\in\mathbb{R}^p$，$y_i\in\{1,-1\}$。

我们定义 $X_1=\{x_i|y_i=1\}$，$X_2=\{x_i|y_i=-1\}$，且 $|X_1|=N_1$，$|X_2|=N_2$，$N=N_1+N_2$。

定义经投影向量 $w$ 投影之后所得实数矩阵为 $Z$，其中 $Z_1=\{z_i=w^\top x_i|y_i=1\}$，$Z_2=\{z_i=w^\top x_i|y_i=2\}$，定义其样本均值分别为 $\bar{z}_1$、$\bar{z}_2$，样本方差分别为 $\sigma_1^2$、$\sigma_2^2$：
$$\bar{z}_k=\frac{1}{N_k}\sum_{z_i\in Z_k}z_i$$

$$\sigma^2_k=\frac{1}{N_k}\sum_{z_i\in Z_k}(z_i-\bar{z}_k)^2=\frac{1}{N_k}\sum_{x_i\in X_k}(w^\top x_i-\bar{z}_k)(w^\top x_i-\bar{z}_k)^\top$$

LDA 进行投影遵循“高内聚，低耦合”的规则，以 $(\bar{z}_1-\bar{z}_2)^2$ 衡量耦合程度，以 $\sigma^2_1+\sigma^2_2$ 衡量内聚程度，因此优化目标可以表示为：
$$\mathcal{J}(w)=\frac{(\bar{z}_1-\bar{z}_2)^2}{\sigma^2_1+\sigma^2_2}$$

对分子进行化简：
$$\begin{aligned}
    \left(\bar{z}_1-\bar{z}_2\right)^2&=\left(\frac{1}{N_1}\sum_{z_i\in Z_1}z_i-\frac{1}{N_2}\sum_{z_i\in Z_2}z_i\right)^2\\
    &=\left(\frac{1}{N_1}\sum_{x_i\in X_1}w^\top x_i-\frac{1}{N_2}\sum_{x_i\in X_2}w^\top x_i\right)^2\\
    &=\left(\frac{1}{N_1}\sum_{x_i\in X_1}w^\top x_i-\frac{1}{N_2}\sum_{x_i\in X_2}w^\top x_i\right)^2\\
    &=\left(w^\top(\bar{x}_1-\bar{x}_2)\right)^2\\
    &=w^\top(\bar{x}_1-\bar{x}_2)(\bar{x}_1-\bar{x}_2)^\top w\\
\end{aligned}$$

对分母进行化简，由于：
$$\begin{aligned}
    \sigma^2_k&=\frac{1}{N_k}\sum_{x_i\in X_k}(w^\top x_i-\bar{z}_k)(w^\top x_i-\bar{z}_k)^\top\\
    &=\frac{1}{N_k}\sum_{x_i\in X_k}(w^\top x_i-\frac{1}{N_k}\sum_{x_j\in X_k}w^\top x_j)(w^\top x_i-\frac{1}{N_k}\sum_{x_j\in X_k}w^\top x_j)^\top\\
    &=\frac{1}{N_k}\sum_{x_i\in X_k}w^\top(x_i-\bar{x}_k)(x_i-\bar{x}_k)^\top w\\
    &=w^\top\frac{1}{N_k}\sum_{x_i\in X_k}(x_i-\bar{x}_k)(x_i-\bar{x}_k)^\top w\\
    &=w^\top\sigma_{X_k}^2w\\
\end{aligned}$$

因此：
$$\sigma^2_1+\sigma^2_2=w^\top(\sigma_{X_1}^2+\sigma_{X_2}^2)w$$

优化目标可以简化为：
$$\mathcal{J}(w)=\frac{w^\top(\bar{x}_1-\bar{x}_2)(\bar{x}_1-\bar{x}_2)^\top w}{w^\top(\sigma_{X_1}^2+\sigma_{X_2}^2)w}$$

为了方便使用以下记号：
$$S_b=(\bar{x}_1-\bar{x}_2)(\bar{x}_1-\bar{x}_2)^\top$$

$$S_w=\sigma_{X_1}^2+\sigma_{X_2}^2$$

因此：
$$\mathcal{J}(w)=\frac{w^\top S_b w}{w^\top S_w w}=(w^\top S_b w)(w^\top S_w w)^{-1}$$

$$\frac{\partial \mathcal{J}(w)}{\partial w}=(2S_bw)(w^\top S_w w)^{-1}-(w^\top S_b w)(w^\top S_w w)^{-2}(2S_ww)$$

令 $\frac{\partial \mathcal{J}(w)}{\partial w}=0$ 得；
$$\begin{aligned}
    (2S_bw)(w^\top S_w w)^{-1}-(w^\top S_b w)(w^\top S_w w)^{-2}(2S_ww)&=0\\
    S_bww^\top S_w w&=w^\top S_b wS_ww\\
    S_ww&=\frac{w^\top S_w w}{w^\top S_b w}\sdot S_bw\\
    w&=\frac{w^\top S_w w}{w^\top S_b w}\sdot S_w^{-1}S_bw\\
    w&=\frac{w^\top S_w w}{w^\top S_b w}\sdot S_w^{-1}(\bar{x}_1-\bar{x}_2) [(\bar{x}_1-\bar{x}_2)^\top w]\\
    &=\lambda S_w^{-1}(\bar{x}_1-\bar{x}_2),\lambda\in\mathbb{R}\\
\end{aligned}$$

由于 w 为一个投影向量，我们实际上只关心其方向，所以 $\lambda$ 的值其实并不重要，我们可以得出结论：
$$w\propto S_w^{-1}(\bar{x}_1-\bar{x}_2)=(\sigma_{X_1}^2+\sigma_{X_2}^2)^{-1}(\bar{x}_1-\bar{x}_2)$$

当 $S_w=I$ 时，或各特征具有各向同性时，我们认为：$w\propto (\bar{x}_1-\bar{x}_2)$

## 4 Logistic Regression

逻辑斯蒂回归是一类软分类模型，通过 Sigmoid 激活函数将线性回归的输出映射至 $[0,1]$：
$$\sigma(x)=\frac{1}{1+e^{-x}}$$

条件概率可以表示为：
$$P(y|x)=P(y=1|x)^{y}P(y=0|x)^{1-y}$$

MLE 优化目标可以写作：
$$\begin{aligned}
    w^*&=\arg\max_w\log P(Y|X)\\
    &=\arg\max_w \sum_{i=1}^N p(y_i|x_i)\\
    &=\arg\max_w \sum_{i=1}^N [y_i\log p(y_i=1|x_i)+(1-y_i)p(y_i=0|x_i)]\\
    &=\arg\max_w \sum_{i=1}^N [y_i\log \phi(x_i;w)+(1-y_i)\log(1-\phi(x_i;w))]\\
\end{aligned}$$

由于实际优化过程中倾向于最小化而非最大化，对上式取负号即得**交叉熵损失**，然后通过随机梯度下降优化。

## 5 Gaussian Discriminant Analysis（GDA）

高斯判别分析是一类概率生成模型，相较于直接建模条件概率 $P(y|x)$，其建模联合概率分布 $P(xy)$，然后通过对 $P(x\{y=1\})$ 和 $P(x\{y=0\})$ 比较大小进行判别，从贝叶斯公式来看这和直接比较条件概率是等价的：
$$P(y|x)=\frac{P(x|y)P(y)}{P(x)}$$

由于 $P(xy)=P(x|y)P(y)$，GDA 对二者分别进行建模，首先假定 $y\sim B(\phi)$，即 $P(y=1)=\phi$，$P(y=0)=1-\phi$，同时假设 $x|y$ 服从混合高斯分布：
$$x|y=1\sim\mathcal{N}(\mu_1,\Sigma)$$

$$x|y=0\sim\mathcal{N}(\mu_2,\Sigma)$$

以对数似然为目标函数：
$$\begin{aligned}
    \mathcal{J}(\theta)&=\log P(XY)\\
    &=\sum_{i=1}^N\log p(x_i|y_i)p(y_i)\\
    &=\sum_{i=1}^N[\log p(x_i|y_i)+\log p(y_i)]\\
    &=\sum_{i=1}^N[\log p(x_i|y_i=1)^{y_i}p(x_i|y_i=0)^{1-y_i}+\log p(y_i=1)^{y_i}p(y_i=0)^{1-y_i}]\\
    &=\sum_{i=1}^N[y_i\log \frac{1}{\sqrt{2\pi}|\Sigma|^{\frac{1}{2}}}\exp(-\frac{(x_i-\mu_1)^\top\Sigma^{-1}(x_i-\mu_1)}{2})+(1-y_i)\log \frac{1}{\sqrt{2\pi}|\Sigma|^{\frac{1}{2}}}\exp(-\frac{(x_i-\mu_2)^\top\Sigma^{-1}(x_i-\mu_2)}{2})+\log \phi^{y_i}(1-\phi)^{1-y_i}]\\
\end{aligned}$$

对优化项 $\phi$ 求偏导：
$$\begin{aligned}
    \frac{\partial \mathcal{J}(\theta)}{\partial \phi} &=\sum_{i=1}^N\left(\frac{y_i}{\phi}-\frac{1-y_i}{1-\phi}\right)\\
\end{aligned}$$

令 $\frac{\partial \mathcal{J}(\theta)}{\partial \phi}=0$ 得：
$$\begin{aligned}
    \sum_{i=1}^N\left(\frac{y_i}{\phi^*}-\frac{1-y_i}{1-\phi^*}\right)&=0\\
    \sum_{i=1}^N\left(y_i(1-\phi^*)-(1-y_i)\phi^*\right)&=0\\
    \sum_{i=1}^N\left(y_i-\phi^*\right)&=0\\
    \sum_{i=1}^N y_i-N\phi^*&=0\\
    \phi^*&=\bar{y_i}\\
\end{aligned}$$

针对优化项目 $\mu$ 求偏导：
$$\begin{aligned}
    \frac{\partial \mathcal{J}(\theta)}{\partial \mu_1}&=\sum_{i=1}^N\left(y_i\log\frac{1}{\sqrt{2\pi}|\Sigma|^{\frac{1}{2}}}\sdot \Sigma^{-1}(x_i-\mu_1)\right)\\
\end{aligned}$$

令 $\frac{\partial \mathcal{J}(\theta)}{\partial \mu_1}=0$，得：
$$\begin{aligned}
    \frac{\partial \mathcal{J}(\theta)}{\partial \mu_1}&=0\\
    \sum_{i=1}^N\left(y_i\log\frac{1}{\sqrt{2\pi}|\Sigma|^{\frac{1}{2}}}\sdot \Sigma^{-1}(x_i-\mu_1^*)\right)&=0\\
    \sum_{i=1}^N\left(y_i\sdot(x_i-\mu_1^*)\right)&=0\\
    \sum_{i=1}^Ny_ix_i&=\sum_{i=1}^Ny_i\mu_1^*\\
    \mu_1^*&=\frac{\sum_{i=1}^Ny_ix_i}{\sum_{i=1}^Ny_i}
\end{aligned}$$

同理可得 $\mu_2^*=\frac{\sum_{i=1}^N(1-y_1)x_i}{\sum_{i=1}^N(1-y_1)}$。

针对优化项目 $\Sigma$，首先有以下事实：
$$\frac{\partial \mathrm{tr}(AB)}{\partial A}=B^{-1}$$

$$\frac{\partial \mathrm{tr}(|A|)}{\partial A}=|A|A^{-1}$$

$$\mathrm{tr}(AB)=\mathrm{tr}(BA)$$

$$\mathrm{tr}(ABC)=\mathrm{tr}(CAB)=\mathrm{tr}(BCA)$$

对条件概率的对数似然进行处理：
$$\begin{aligned}
    \sum_{i=1}^N\log \frac{1}{\sqrt{2\pi}|\Sigma|^\frac{1}{2}}\exp(-\frac{(x_i-\mu)^\top\Sigma^{-1}(x_i-\mu)}{2})&=\sum_{i=1}^N\log \frac{1}{\sqrt{2\pi}}-\frac{1}{2}\log|\Sigma|-\frac{(x_i-\mu)^\top\Sigma^{-1}(x_i-\mu)}{2}\\
    &=C-\frac{N}{2}\log|\Sigma|-\frac{1}{2}\sum_{i=1}^N(x-\mu)^\top\Sigma^{-1}(x-\mu)\\
    &=C-\frac{N}{2}\log|\Sigma|-\frac{1}{2}\sum_{i=1}^N\mathrm{tr}((x-\mu)(x-\mu)^\top\Sigma^{-1})\\
    &=C-\frac{N}{2}\log|\Sigma|-\frac{1}{2}\sum_{i=1}^N\mathrm{tr}(\Sigma_x\Sigma^{-1})\\
\end{aligned}$$

因此：
$$\begin{aligned}
    \frac{\partial \mathcal{J}(\theta)}{\partial \Sigma}&=\frac{\partial}{\partial \Sigma}\left[-\frac{N_1}{2}\log|\Sigma|-\frac{1}{2}\sum_{i=1}^{N_1}\mathrm{tr}(\Sigma_1\Sigma^{-1})-\frac{N_2}{2}\log|\Sigma|-\frac{1}{2}\sum_{i=1}^{N_2}\mathrm{tr}(\Sigma_2\Sigma^{-1})+C\right]\\
    &=\frac{\partial}{\partial \Sigma}\left[-\frac{N}{2}\log|\Sigma|-\frac{N_1}{2}\mathrm{tr}(\Sigma_1\Sigma^{-1})-\frac{N_2}{2}\mathrm{tr}(\Sigma_2\Sigma^{-1})\right]\\
    &=-\frac{1}{2}(N\Sigma^{-1}-N_1\Sigma_1\Sigma_{-2}-N_2\Sigma_2\Sigma_{-2})\\
\end{aligned}$$

令 $\frac{\partial \mathcal{J}(\theta)}{\partial \Sigma}=0$，得 $\Sigma^*=\frac{1}{N}(N_1\Sigma_1+N_2\Sigma_2)$

## 6 Naive Bayes

朴素贝叶斯也是一类概率生成模型，可以看作是一类最简单的概率无向图模型，其通过假定特征之间的条件独立性对模型进行简化。

类似 GDA，对于先验 $P(y)$，在二分类问题下近似为伯努利分布，在多分类问题下近似为分类分布（categorical distribution）。

对于似然 $P(x|y)$ 的处理，朴素贝叶斯假设各特征维度之间的条件概率独立：
$$x_i\perp x_j|y,i\not ={j}$$

因此可以将其拆解为乘积的形式：
$$P(x|y)=\prod_{i=1}^pP(x_i|y)$$

对于离散的特征，我们假设其服从分类分布，对于连续的特征，假设其服从高斯分布 $\mathcal{N}(\mu_i,\sigma_i)$。

然后通过极大似然估计得到参数即可（**朴素贝叶斯是频率派模型而不是贝叶斯派模型**）。
