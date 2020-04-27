Expectation Maximization Algorithm

论文地址：https://www.nature.com/articles/nbt1406.pdf

# 简介

EM算法（期望最大算法），是一种迭代算法，用于含有隐变量（Hidden Variable）的概率参数模型的最大似然估计或极大后验概率估计。

（这第一句话读下来，是不是就有点晕晕的？先抛开这些，来个例子感受一下。）

# 举例

直接引用[论文](https://www.nature.com/articles/nbt1406.pdf)中的例子。

假设有两枚硬币A和B，它们的随机抛掷的结果如下图所示：


![捕获.JPG](D:\研究生阶段\机器学习\Datawhale-12\img\EFOBj8zqPWiY97L.jpg)

（图中，H表示正面，T表示反面）

我们很容易计算出两枚硬币各自抛出正面的概率：
$$
\begin{gather}
\theta_A = \frac{24}{24+6}=0.8 \\
\theta_B = \frac{9}{9+11}=0.45 \notag
\end{gather}
$$
现在我们硬币盖住，即我们只能看到抛掷的结果，但你并不知道是哪枚硬币抛掷的：

|  Coin  | Statistics |
| :----: | :--------: |
| Coin\* |   5H,5T    |
| Coin\* |   9H,1T    |
| Coin\* |   8H,2T    |
| Coin\* |   4H,6T    |
| Coin\* |   7H,3T    |

这种情况下，我们该如何估计$\theta_A$和$\theta_B$的值呢？

假设，我们用一个变量$Z=(z_1,z_2,z_3,z_4,z_5)$代表每一轮使用的硬币（使用硬币A或硬币B），如果我们**明确知道**了$z_i,i=1,2,3,4,5$代表是硬币A还是硬币B，那么计算$\theta_A$和$\theta_B$就和之前一样了。问题就在于，我们并不能确定$z_i$的状态，这样的变量就称为隐变量。

另一方面，如果我们知道了$\theta_A$和$\theta_B$，反推每一个$z_i$也是可行的。可惜，两者都不知道。

此时，EM算法就出场了！它的解决方法就是：

- 先随机初始化$\theta_A$和$\theta_B$。

- 然后基于$\theta_A$和$\theta_B$，去估计$Z$。

- 然后基于$Z$按照最大似然概率去估计新的$\theta_A$和$\theta_B$。

- 不断循环上述过程，直至收敛。

（**提问！这算法一定能收敛吗？**别急，收敛性证明肯定得有，我们先看完例子）

# 计算

假设，先随机初始化得到$\theta_A=0.6$和$\theta_B=0.5$。

对于第一轮来说，如果是硬币A，得出$(5H,5T)$的概率为：$0.6^5\times0.4^5$；如果是硬币B，得出$(5H,5T)$的概率为：$0.5^5\times0.5^5$。我们可以算出使用硬币A和硬币B的概率分别是：
$$
\begin{gather}
P_A = \frac{0.6^5\times0.4^5}{(0.6^5\times0.4^5)+(0.5^5\times0.5^5)}=0.45 \\
P_B = \frac{0.5^5\times0.5^5}{(0.6^5\times0.4^5)+(0.5^5\times0.5^5)}=0.55 \notag
\end{gather}
$$
同理，我们可以分别算出其他每轮的$P_A$和$P_B$，得到下表：

|  No  | Coin A | Coin B |
| :--: | :----: | :----: |
|  1   |  0.45  |  0.55  |
|  2   |  0.80  |  0.20  |
|  3   |  0.73  |  0.27  |
|  4   |  0.35  |  0.65  |
|  5   |  0.65  |  0.35  |

从期望的角度来看，对于第一轮抛掷，使用硬币A的概率是0.45，使用硬币B的概率是0.55。其他轮同理，这一步我们就称为E-step，实际上是估计出了$Z$的概率分布。

结合计算出的期望以及最初的投掷结果，可以进行更新，以第二轮的硬币A为例子，计算方式为：
$$
\begin{gather}
H : 0.80\times9=7.2 \\
T:0.80\times1=0.8 \notag
\end{gather}
$$
得到：

|  No  |       Coin A        |       Coin B        |
| :--: | :-----------------: | :-----------------: |
|  1   | $\approx$2.2H,2.2T  | $\approx$2.8H,2.8T  |
|  2   | $\approx$7.2H,0.8T  | $\approx$1.8H,0.2T  |
|  3   | $\approx$5.9H,1.5T  | $\approx$2.1H,0.5T  |
|  4   | $\approx$1.4H,2.1T  | $\approx$2.6H,3.9T  |
|  5   | $\approx$4.5H,1.9T  | $\approx$2.5H,1.1T  |
|      | $\approx$21.3H,8.6T | $\approx$11.7H,8.4T |

然后，再计算（或者说更新）新的$\theta_A$和$\theta_B$：
$$
\begin{gather}
\theta_A = \frac{21.3}{21.3+8.6}=0.71 \\
\theta_B = \frac{11.7}{11.7+8.4}=0.58 \notag
\end{gather}
$$
这一步就对应M-step。

如此反复迭代，就可以算出最终的$\theta_A$和$\theta_B$。

上述过程，用图展示如下：

<img src="D:\研究生阶段\机器学习\Datawhale-12\img\MVaigrh6UDzeOKy.jpg" alt="捕获.JPG" style="zoom:50%;" />

现在，是不是稍微有点了解EM算法的一个大致流程了呢？那么，接下来我们就要从数学角度进行阐述了，注意保持头脑清醒。

# 总述思想

EM算法的核心思想说起来非常简单，分为两步：Expectation-Step和Maximization-Step。E-Step主要通过观察数据和现有的模型来估计参数，然后用这个估计的参数来计算上述对数似然函数的期望值；而M-Step是寻找似然函数最大化时对应的参数。由于算法会保证在每次迭代之后似然函数都会增加，所以函数最终都会收敛。

# 推导

给定数据集，**假设样本间相互独立**，我们想要拟合模型$p(x;\theta)$得到数据的参数。根据分布我们可以得到如下的似然函数：
$$
\begin{align*}
L(\theta)&= \sum\limits_{i=1}^{n}logp(x_i;\theta)\\
&= \sum\limits_{i=1}^{n}log\sum\limits_{z}p(x_i,z;\theta)
\end{align*}
$$
第一步是对极大似然函数取对数，第二步是对每个样本的每个可能的类别$z$求联合概率分布之和。如果这个$z$是已知的，那么使用极大似然发会很容易。但如果$z$是隐变量，我们就需要用EM算法来求。

事实上，隐变量估计问题也可以通过梯度下降等优化算法，但实际由于求和项将随着隐变量的数目以指数级上升，会给梯度计算带来麻烦；而EM算法则可看作一种非梯度优化方法。

对于每个样本$i$，我们用$Q_i(z)$表示样本$i$隐含变量$z$的某种分布，且$Q_i(z)$满足条件$(\sum\limits_z^ZQ_i(z)=1,Q_i(z))\ge0$。

我们将上面的式子做以下变化：
$$
\begin{align*}
\sum\limits_i^nlogp(x_i;\theta)&= \sum\limits_{i=1}^{n}log\sum\limits_{z}p(x_i,z;\theta)\\
&= \sum\limits_{i=1}^{n}log\sum\limits_{z}^{Z}Q_i(z)\frac{p(x_i,z;\theta)}{Q_i(z)}\\
&\ge\sum\limits_i^n\sum\limits_{z}^{Z}Q_i(z)log\frac{p(x_i,z;\theta)}{Q_i(z)}
\end{align*}
$$
上面式子中，第一步是求和每个样本的所有可能的类别$z$的联合概率，但是这一步直接求导非常困难，所以将其分子分母都乘以函数$Q_i(z)$，转换到第二步。从第二步到第三步是利用Jensen不等式。

简单证明一下Jensen不等式：

如果$f$是凹函数，$X$是随机变量，则$E[f(X)]\le f(E[X])$，当$f$严格是凹函数时，则$E[f(X)]<f(E[X])$，凸函数反之。当$X=E[X]$时，即为常数时等式成立。

我们把第一步中的$log$函数看成一个整体，由于$log(x)$的二阶导数小于0，所以原函数为凹函数。我们把$Q_i(z)$看成一个概率$p_z$，把$\frac{p(x_i,z;\theta)}{Q_i(z)}$看成$z$的函数$g(z)$。根据期望公式有：
$$
E(z)=p_zg(z)=\sum\limits_{z}^{Z}Q_i(z)(\frac{p(x_i,z;\theta)}{Q_i(z)})
$$
根据Jensen不等式的性质：
$$
f(\sum\limits_{z}^{Z}Q_i(z)(\frac{p(x_i,z;\theta)}{Q_i(z)}))=f(E[z])\ge E[f(z)]=\sum\limits_{z}^{Z}Q_i(z)f(\frac{p(x_i,z;\theta)}{Q_i(z)})
$$
把$f()$用$log()$代入即可。

通过上面，我们得到了：$L(\theta)\ge J(z,Q)$的形式（$z$为隐变量），那么我们就可以通过不断最大化$J(z,Q)$的下界来使得$L(\theta)$不断提高。

用下图更加形象的解释：

![EM.jpg](D:\研究生阶段\机器学习\Datawhale-12\img\v2-2f7fc5ca144d2f85f14d46e88055dd86_720w.jpg)

这张图的意思就是：**首先我们固定$\theta$，调整$Q(z)$使下界$J(z,Q)$上升至与$L(\theta)$在此点$\theta$处相等（浅蓝色曲线到深蓝色曲线），然后固定$Q(z)$，调整$\theta$使下界$J(z,Q)$达到最大值（$\theta_t$到$\theta_{t+1}$），然后再固定$\theta$，调整$Q(z)$，一直到收敛到似然函数$L(\theta)$的最大值$\theta$处。**

也就是说，EM 算法通过引入隐含变量，使用 MLE（极大似然估计）进行迭代求解参数。通常引入隐含变量后会有两个参数，EM 算法首先会固定其中的第一个参数，然后使用 MLE 计算第二个变量值；接着通过固定第二个变量，再使用 MLE 估测第一个变量值，依次迭代，直至收敛到局部最优解。

但是这里有两个问题：

- **什么时候下界**$J(z,Q)$**与**$L(\theta)$**相等？**

- **为什么一定会收敛？**

首先第一个问题，当$X=E(X)$时，即为常数时等式成立：
$$
\frac{p(x_i,z;\theta)}{Q_i(z)}=c
$$
做个变换：
$$
\sum\limits_{z}p(x_i,z;\theta)=\sum\limits_{z}Q_i(z)c
$$
其中$\sum\limits_{z}Q_i(z)=1$ ，所以可以推导出：
$$
\sum\limits_{z}p(x_i,z;\theta)=c
$$
因此得到了：
$$
\begin{align*}
Q_i(z)&=\frac{p(x_i,z;\theta)}{\sum\limits_{z}p(x_i,z;\theta)}\\
&=\frac{p(x_i,z;\theta)}{p(x_i;\theta)}\\
&=p(z|x_i;\theta)
\end{align*}
$$
至此我们推出了在固定参数下，使下界拉升的$Q(z)$的计算公式就是后验概率，同时解决了$Q(z)$如何选择的问题。这就是我们刚刚说的 EM 算法中的 E-Step，目的是建立$L(\theta)$的下界。接下来得到 M-Step 目的是在给定$Q(z)$后调整$\theta$，从而极大化似然函数$L(\theta)$的下界$J(z,Q)$。

对于第二个问题，为什么一定会收敛？

这边简单说一下，因为每次$\theta$更新时（每次迭代时），都可以得到更大的似然函数，也就是说极大似然函数是单调递增，那么我们最终就会得到极大似然估计的最大值。

但是要注意，迭代一定会收敛，但不一定会收敛到真实的参数值，因为可能会陷入局部最优。所以 EM 算法的结果很受初始值的影响。

