Conditional Random Field——Part C

By Xyao

# 前言

本系列包含三部分，[part-A](https://github.com/TrippleKing/Datawhale-12/blob/master/ConditionalRandomField-A.md)中简单地对概率图模型做了介绍，同时叙述了HMM以及MRF；[part-B](https://github.com/TrippleKing/Datawhale-12/blob/master/ConditionalRandomField-B.md)中叙述了MEMM及其标注偏置问题，同时引出了CRF及其参数化形式。

在上述基础上，我们开启最后一部分，仍以链式条件随机场为叙述对象，展开CRF的学习与推断。

# 参数化形式

条件随机场（如无特殊说明，代表链式条件随机场）的参数化形式及含义已经在[part-B](https://github.com/TrippleKing/Datawhale-12/blob/master/ConditionalRandomField-B.md)中给出，我们不再赘述，直接引过来，方便下文的叙述。
$$
P(Y|X)=\frac{1}{Z}exp(\sum\limits_j\sum\limits_{i=1}^{n-1}\lambda_jt_j(y_{i+1},y_i,X,i)+\sum\limits_k\sum\limits_{i=1}^{n}\mu_ks_k(y_i,X,i))
$$

$$
Z=\sum\limits_yexp(\sum\limits_j\sum\limits_{i=1}^{n-1}\lambda_jt_j(y_{i+1},y_i,X,i)+\sum\limits_k\sum\limits_{i=1}^{n}\mu_ks_k(y_i,X,i))
$$

CRF的参数化形式可以进行一定的简化，这里的简化并不是删去公式中的某一部分，而是以向量的形式把"$\sum$"隐藏起来，使公式看起来比较简洁，这里就不做推导，感兴趣的读者可以自行阅读李航的《统计学习方法》。

# CRF的学习算法

条件随机场模型实际上是定义在时序数据上的对数线性模型，其学习方法包括极大似然估计和正则化的极大似然估计。具体的优化实现算法有改进的迭代尺度法、梯度下降法以及拟牛顿法。

其实CRF的学习算法与常见的机器学习中的学习算法没有太多的区别，仍然是求梯度，再更新参数，只不过CRF的目标是极大似然估计（即最大化），所以在更新梯度的时候是沿着梯度上升的方向进行更新。

具体的推导，可以参考[B站的白板推导系列](https://www.bilibili.com/video/BV19t411R7QU?p=8)。

# CRF的推断算法

条件随机场的推断问题是给定条件随机场$P(Y|X)$和输入序列（观测序列）$x$，求条件概率最大的输出序列（标记序列）$y^\ast$，即对观测序列进行标注，其中最著名的是维特比算法。
$$
\begin{equation}
\begin{split}
y^\ast &=\arg\max\limits_yP_w(y|x)\\
&=\arg\max\limits_y\frac{\exp(w\cdot F(y,x))}{Z_w(x)}\\
&=\arg\max\limits_y\exp(w\cdot F(y,x))\\
&=\arg\max\limits_y(w\cdot F(y,x))
\end{split}
\end{equation}
$$

$$
w=(w_1,w_2,...,w_K)^T
$$

$$
\begin{equation}
w_k=
\begin{cases}
\lambda_j,&j=1,2,...,K_1\\
\mu_k,&k=K_1+1,...,K
\end{cases}
\end{equation}
$$

$$
F(y,x)=(f_1(y,x),f_2(y,x),...,f_K(y,x))^T
$$

$$
\begin{equation}
f_k(y,x)=\sum\limits_{i=1}^{n-1}f_k(y_{i+1},y_i,X,i),\quad k=1,2,...,K
\end{equation}
$$

$$
\begin{equation}
f_k(y_{i+1},y_i,X,i)=
\begin{cases}
t_j(y_{i+1},y_i,X,i),&k=1,2,...,K_1\\
s_k(y_i,X,i)),&k=K_1+1,...,K
\end{cases}
\end{equation}
$$

于是，条件随机场的推断问题成为求非规范化概率最大的最优路径问题：
$$
\max_{y}(w\cdot F(y,x))
$$
注意，这时只需计算非规范化概率，而不必计算概率，可以大大提高效率。将上式改下如下：
$$
\max_y\sum\limits_{i=1}^{n-1}w\cdot F_i(y_{i+1},y_i,x)
$$
首先，求出位置0的各个标记$j=1,2,...,m$的非规范化概率:
$$
\delta_1(j)=w\cdot F_1(y_0=start,y_1=j,x),\quad j=1,2,...,m
$$
一般地，由递推公式，求出到位置$i$的各个标记$l=1,2,...,m$的非规范化概率最大值，同时记录非规范化概率最大值的路径：
$$
\delta_i(l)=\max\limits_{1\le j\le m}\lbrace\delta_{j-1}(j)+w\cdot F_i(y_{i+1}=l,y_i=j,x)\rbrace,\quad l=1,2,...,m
$$

$$
\Psi_i(l)=\arg\max\limits_{1\le j\le m}\lbrace\delta_{j-1}(j)+w\cdot F_i(y_{i+1}=l,y_i=j,x)\rbrace,\quad l=1,2,...,m
$$

直到$i=n$时终止。这时求得非规范化概率的最大值为：
$$
\max\limits_{1\le j\le m}\delta_n(j)
$$
及最优路径的终点：
$$
y_n^\ast=\arg\max\limits_{1\le j\le m}\delta_n(j)
$$
由此最优路径终点返回：
$$
y_i^\ast=\Psi_{i+1(y_{i+1}^{\ast})},\quad i=n-1,n-2,...,1
$$
得到最优路径$y^\ast=(y_1^\ast,y_2^\ast,...,y_n^\ast)^T$





