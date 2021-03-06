Conditional Random Field——Part A

By Xyao

# 前言

CRF本身属于概率图模型，直接讲述CRF可能会引起不适，所以本文先简述概率图模型再慢慢过渡到CRF，内容会比较多，希望读者能有耐心，慢慢阅读。

由于内容比较多，计划分三个部分叙述，这一部分先对CRF前期知识做一定的铺垫。

# 概率图模型(Probabilistic Graphical Model)

概率图模型是**一类用图（Graph，一种数据结构）来表达变量相关关系的概率模型**。

它以图为表示工具，最常见的是用一个结点表示一个或一组随机变量，结点之间的边表示变量间的概率相关关系，称为"变量关系图"。根据边的性质不同，概率图模型大致**可分为两类**：

- 第一类是使用有向无环图表示变量间的依赖关系，称为**有向图模型或贝叶斯网络(Bayesian Network)**；
- 第二类是使用无向图表示变量间的相关关系，称为**无向图模型或马尔可夫网络(Markov Network)**。

# 隐马尔可夫模型(Hidden Markov Model, HMM)

隐马尔可夫模型是结构最简单的动态贝叶斯网络(Dynamic Bayes Network)，这是一种著名的有向图模型，主要用于时序数据建模，在语音识别、自然语言处理等领域有广泛应用。

HMM的图结构可以用下图进行表示：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="http://q9qozit0b.bkt.clouddn.com/HMM.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    padding: 2px;">图1. HMM图结构</div>
</center>

隐马尔可夫模型中的变量可分为两组：

- 第一组是**状态变量**：$(y_1,y_2,...,y_n)$，其中$y_i\in Y$表示第$i$时刻的系统状态。通常假定状态变量是隐藏的、不可被观测的，因此状态变量亦被称为隐变量(Hidden Variable)。
- 第二组是**观测变量**：$(x_1,x_2,...,x_n)$，其中$x_i\in X$表示第$i$时刻的观测值。

在隐马尔可夫模型中，系统通常在多个状态$(s_1,s_2,...,s_N)$之间转换，因此状态变量$y_i$的取值范围$Y$(通常称为状态空间)通常是有$N$个可能取值的离散空间。观测变量$x_i$可以是离散型也可以是连续型，为了便于讨论，我们仅考虑离散型观测变量，并假定其取值范围$X$为$(o_1,o_2,...,o_M)$。

图1中的箭头表示了变量间的依赖关系。**在任一时刻，观测变量的取值仅依赖于状态变量，即$x_t$仅有$y_t$确定，与其他状态变量及观测变量的取值无关**。**同时，$t$时刻的状态$y_t$仅依赖于$t-1$时刻的状态$y_{t-1}$，与此前$t-2$个状态无关**。这就是所谓的"马尔可夫链"(Markov Chain)，即：系统下一时刻的状态仅由当前状态决定，不依赖于以往的任何状态。

## 两个基本假设

上文叙述了隐马尔可夫模型的两个基本假设，整理如下：

- **齐次马尔可夫性假设**：假设隐藏的马尔可夫链在任意时刻$t$的状态$y_t$只依赖于$t-1$时刻的状态$y_{t-1}$，与其他时刻的状态及观测无关，也与$t$时刻无关：

$$
P(y_t|y_{t-1},x_{t-1},...,y_1,x_1)=P(y_t|y_{t-1})
$$

- **观测独立性假设**：假设任意时刻的观测变量只依赖于该时刻的状态变量，与其他观测变量及状态变量无关：

$$
P(x_t|y_t,x_t,...,y_1,x_1)=P(x_t|y_t)
$$

所以，所有变量的联合概率分布就可以写成：
$$
P(x_1,y_1,...,x_n,y_n)=P(y_1)P(x_1|y_1)\prod\limits_{i=2}^{n}P(y_i)P(x_i|y_i)
$$

## 三个概率矩阵

除了上述结构信息外，欲确定一个隐马尔可夫模型还需一下三组参数：

- **状态转移概率**：模型在各个状态间转换的概率，通常记为矩阵$A=[a_{ij}]_{N\times N}$，其中

$$
a_{ij}=P(y_{t+1}=s_j|y_t=s_i),\ 1\le i,j\le N
$$

​	表示在任意时刻$t$，若状态为$s_i$，则在下一时刻状态为$s_j$的概率。

- **输出观测概率**：模型根据当前状态获得各个观测值的概率，通常记为矩阵$B=[b_{ij}]_{N\times M}$，其中

$$
b_{ij}=P(x_t=o_j|y_t=s_i),\ 1\le i\le N,1\le j\le M
$$

​	表示在任意时刻$t$，若状态为$s_i$，则观测值$o_j$被获取的概率。

- **初始状态概率**：模型在初始时刻各状态出现的概率，通常记为$\pi=(\pi_1,\pi_2,...,\pi_N)$，其中

$$
\pi_i=P(y_1=s_i),\ 1\le i\le N
$$

​	表示模型的初始状态为$s_i$的概率。

通过指定状态空间$Y$、观测空间$X$和上述三组参数，就能确定一个隐马尔可夫模型，通常其参数$\lambda=[A,B,\pi]$来指代。给定隐马尔可夫模型$\lambda$，它按如下过程产生观测序列$(x_1,x_2,...,x_n)$：

1. 设置$t=1$，并根据初始状态概率$\pi$选择初始状态$y_1$；
2. 根据状态$y_t$和输出观测概率$B$选择观测变量取值$x_t$；
3. 根据状态$y_t$和状态转移概率$A$，确定$y_{t+1}$；
4. 若$t<n$，设置$t=t+1$，并转到第2.步，否则停止。

其中$y_t\in(s_1,s_2,...,s_N)$和$x_t\in(o_1,o_2,...,o_M)$分别为第$t$时刻的状态和观测值。

## 三个基本问题

在实际应用中，常关注隐马尔可夫模型的三个基本问题：

- 给定模型$\lambda=[A,B,\pi]$，如何有效计算其产生观测序列$x=(x_1,x_2,...,x_n)$的概率$p(x|\lambda)$？换言之，如何评估模型与观测序列之间的匹配程度？
- 给定模型$\lambda=[A,B,\pi]$和观测序列$x=(x_1,x_2,...,x_n )$，如何找到与此观测序列最匹配的状态序列$y=(y_1,y_2,...,y_n)$？换言之，如何根据观测序列推断出隐藏的模型状态？
- 给定观测序列$x=(x_1,x_2,...,x_n )$，如何调整模型参数$\lambda=[A,B,\pi]$使得该序列出现的概率$P(x|\lambda)$最大？换言之，如何训练模型使其能最好地描述观测数据？

本文的重点是CRF，因此不对HMM作过多的公式推导，而是简单地介绍一些概念，让阅读者能够对概率图模型、HMM有一定的认识，这有助于阅读者更好地理解CRF。

## 存在的问题

隐马尔可夫模型存在的问题正是它的两个基本假设，这两个基本假设直接产生非常大的限制，大多数实际应用都无法很好地满足这两个假设。例如词性标注问题中，一个词被标注为动词还是名词，不仅与它本身以及它前一个词的标注有关，还依赖于上下文中的其他词。

# 马尔可夫随机场(Markov Random Field, MRF)

马尔可夫随机场是一种著名的无向图模型，有一组势函数(Potential Functions)，亦称"因子"(Factor)，这是定义在变量子集上的非负实函数，主要是用于定于概率分布函数。

下图展示一个简单的马尔可夫随机场：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="http://q9qozit0b.bkt.clouddn.com/MRF.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    padding: 2px;">图2. MRF图结构</div>
</center>



对于图中结点的一个子集，若其中任意两个结点间都有边连接，则称该结点子集为一个"团"(Clique)。若在一个团中加入另外任何一个结点都不再形成团，则称该团为"极大团"(Maximal Clique)；即，极大团就是不能被其他团所包含的团。例如，在图2中，$(x_1,x_2),(x_1,x_3),(x_2,x_4),(x_2,x_5),(x_2,x_6),(x_3,x_5),(x_5,x_6)$和$(x_2,x_5,x_6)$都是团，并且除了  $(x_2,x_5),(x_2,x_6),(x_5,x_6)$ 之外都是极大团。

## 因子分解

在马尔可夫随机场中，多个变量之间的联合概率分布能基于极大团分解为多个因子的乘积，每个因子仅与一个团相关。具体来说，设其无向图$G$，$C$是$G$上的所有极大团构成的集合，$X_Q$表示$Q$对应的随机变量组，$Q\in C$，则联合概率分布$P(X)$为
$$
P(X) = \frac{1}{Z}\prod\limits_{Q\in C}\psi_Q(X_Q)
$$
其中，$Z$为规范化因子，$Z=\sum\limits_X\prod\limits_{Q\in C}\psi_Q(X_Q)$。规范化因子保证$P(X)$构成一个概率分布。

函数$\psi_Q(X_Q)$成为势函数，要求势函数是严格正的，通常定义为指数函数：
$$
\psi_Q(X_Q)=exp(-E(X_Q))
$$

## 全局马尔可夫性

设结点A, B是在无向图G中被结点集合C分开的任意结点集合，如图3所示：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="http://q9qozit0b.bkt.clouddn.com/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E6%80%A7.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    padding: 2px;">图3. 全局马尔可夫性</div>
</center>



结点A, B和C所对应的随机变量组分别是$X_A$, $X_B$和$X_C$。

全局马尔可夫性是指给定随机变量组$X_C$条件下，随机变量组$X_A$和$X_B$是条件独立的，即：
$$
P(X_A,X_B|X_C) = P(X_A|X_C)P(X_B|X_C)
$$
由全局马尔可夫性可以得到两个很有用的推论：

- **局部马尔可夫性**：给定某变量的邻接变量，则该变量条件独立于其他变量。设$v\in V$是无向图G中任意一个结点，$W$是与$v$有边连接的所有结点，$O$是$v$和$W$以外的其他所有结点。$v$表示的随机变量是$X_v$，$W$表示的随机变量组是$X_W$，$O$表示的随机变量组是$X_O$。局部马尔可夫性指在给定随机变量组$X_W$的条件下随机变量$X_v$与随机变量组$X_O$是独立的，即：

$$
P(X_v,X_O|X_W) = P(X_v|X_W)P(X_O|X_W)
$$

​	由全局马尔可夫性很容易推出局部马尔可夫性。

- **成对马尔可夫性**：给定所有其他变量，两个非邻接变量条件独立。设$u$和$v$是无向图G中任意两个没有边连接的结点，结点$u$和结点$v$分别对应随机变量$X_u$和$X_v$。其他所有结点为$O$，对应的随机变量组是$X_O$。成对马尔可夫性是指在给定随机变量组$X_O$的条件下随机变量$X_u$和$X_v$是条件独立的，即：

$$
P(X_u,X_v|X_O)=P(X_u|X_O)P(X_v|X_O)
$$

​	由全局马尔可夫性也很容易推出成对马尔可夫性。

