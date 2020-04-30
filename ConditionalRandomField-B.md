Conditional Random Field——Part B

By Xyao

# 前言

在[part-A](https://github.com/TrippleKing/Datawhale-12/blob/master/ConditionalRandomField-A.md)中，简单地对概率图模型做了介绍，同时叙述了隐马尔可夫模型(HMM)以及马尔可夫随机场(MRF)，相信读者对概率图模型中的一些概念已经有所了解。

在此基础上，我们开启part-B部分，先介绍最大熵马尔可夫模型(MEMM)，再引出条件随机场(CRF)。为什么按这样的顺序？且耐心阅读下文。

# 最大熵马尔可夫模型(Maximum Entropy Markov Model)

在[part-A](https://github.com/TrippleKing/Datawhale-12/blob/master/ConditionalRandomField-A.md)中，我们提到HMM的两大基本假设(忘记得赶紧再看一下[part-A](https://github.com/TrippleKing/Datawhale-12/blob/master/ConditionalRandomField-A.md))，这两大基本假设也成为了HMM的致命缺点（有种成也萧何败萧何的感觉）。

最大熵马尔可夫模型并没有像HMM通过联合概率进行建模，而是直接学习条件概率，即为判别模型。如下图所示：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://i.loli.net/2020/04/30/ZiuOsCceYTFqtMd.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    padding: 2px;">图1. MEMM图结构</div>
</center>

建立的条件概率为：
$$
P(y_{1:n}|x_{1:n})=\prod\limits_{i=1}^{n}P(y_i|y_{i-1},x_{1:n})=\prod\limits_{i=1}^{n}\frac{exp(w^Tf(y_i,y_{i-1},x_{1:n}))}{Z(y_{i-1},x_{1:n})}
$$
其中$Z$是局部归一化因子。

MEMM打破了观测变量独立假设，使得模型更加合理，同时直接学习条件概率，对于标注问题而言更符合实际应用。

## 标注偏置问题

但是，MEMM本身存在标注偏置问题(Label Bias Problem)，下面简单地解释一下：

<img src="https://i.loli.net/2020/04/30/BWGgKxEvRr7JUjs.png" alt="MEMM_1.png" style="zoom:50%;" />

假设有上图所示的概率分布，`State 1`可以通过`0.4->0.55->0.3`达到`State 2`，概率为$0.4\times 0.55\times 0.3=0.066$，同理其他路径的概率也可以计算得到，如`0.4->0.45->0.5`的概率为0.09。计算结果发现：

<img src="https://i.loli.net/2020/04/30/OU5QPJo4bhXcA6f.png" alt="MEMM_2.png" style="zoom:50%;" />

概率最大的路径是图中标红的路径，即`State 1`仍然保持为`State 1`。但是，从全局角度分析，`State 1`在每一次的观测中总是更倾向于转移到`State 2`，`State 2`在每一次观测中更倾向于转移到`State 2`。但为什么最终对于`State 1`而言，最大概率仍是保持为`State 1`。原因在于，`State 1`仅有两种转移状态而`State 2`有着5种转移状态，而MEMM中所做的是局部归一化，这使得`State 1`的平均转移概率普遍偏高，使得概率最大路径更容易出现在转移少的状态中。

# 条件随机场(Conditional Random Field, CRF)

终于轮到主角CRF出场了，根据前面的叙述，可以了解到HMM被自身的两个基本假设所约束，MEMM打破了观测变量独立假设，但是产生了标注偏置问题，为了解决上述种种问题，条件随机场被提出来了。

条件随机场可以看做给定观测值的马尔可夫随机场(马尔可夫随机场在[part-A](https://github.com/TrippleKing/Datawhale-12/blob/master/ConditionalRandomField-A.md)介绍)，也是一个判别式模型(这一点与MEMM相同)。

条件随机场试图对多个变量在给定观测值后的条件概率进行建模。具体来说，若令$X=(x_1,x_2,...,x_n)$为观测序列，$Y=(y_1,y_2,...,y_n)$为与之相应的标记序列，则条件随机场的目标是构建条件概率模型$P(Y|X)$。学习时，利用训练数据集通过极大似然估计或正则化的极大似然估计得到条件概率模型$\hat P(Y|X)$；预测时，对于给定的观测序列$x$，求出条件概率$\hat P(y|x)$最大的标记序列。

## 一般的条件随机场

令$G=(V,E)$表示节点与标记序列$Y$中元素一一对应的无向图，$y_v$表示与节点$v$对应的标记变量，$n(v)$表示节点$v$的邻接节点集合，若图$G$的每个变量$y_v$都满足马尔可夫性(参考[part-A](https://github.com/TrippleKing/Datawhale-12/blob/master/ConditionalRandomField-A.md)的讲解)，即：
$$
P(y_v|X,Y_w,w\ne v)=P(y_v|X,Y_{n(v)})
$$
则$(X,Y)$构成一个条件随机场。

也就是说，在给定观测序列$X$的条件下，节点$v$的标记变量仅与其邻接节点的标记变量有关。

理论上说，图$G$可具有任意结构，只要能表示标记变量之间的条件独立性关系即可。

## 链式条件随机场

在现实应用中，图$G$通常为链式结构，一般在说条件随机场时默认为链式条件随机场，另做说明的除外。

我们标题中的CRF，讲的也是链式条件随机场，图结构如下所示：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://i.loli.net/2020/04/30/rnejLd8Bio4gYa6.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    padding: 2px;">图2. CRF图结构</div>
</center>

是不是感觉和MEMM很像？CRF其实就是将MEMM中的有向图改成了无向图。

结合上图，CRF建立的条件概率就可以如下给出：
$$
P(y_i|y_1,y_2,...,y_{i-1},y_{i+1},...,y_n,X_{1:n})=P(y_i|y_{i-1},y_{i+1},X_{1:n})
$$

$$
i=1,2,...,n(在i=1和n时只考虑单边)
$$

## 参数化形式

与马尔可夫随机场定义联合概率的方式类似，条件随机场使用势函数和图结构上的团来定义条件概率$P(Y|X)$。在条件随机场中，通过选用指数势函数并引入特征函数(Feature Function)，条件概率被定义为：
$$
P(Y|X)=\frac{1}{Z}exp(\sum\limits_j\sum\limits_{i=1}^{n-1}\lambda_jt_j(y_{i+1},y_i,X,i)+\sum\limits_k\sum\limits_{i=1}^{n}\mu_ks_k(y_i,X,i))
$$
其中，$t_j(y_{i+1},y_i,X,i)$是定义在观测序列的两个相邻标记位置上的转移特征函数(Transition Feature Function)，用于刻画相邻标记变量之间的相关关系以及观测序列对它们的影响；$s_k(y_i,X,i)$是定义在观测序列的标记位置$i$上的状态特征函数(Status Feature Function)，用于刻画观测序列对标记变量的影响；$\lambda_j$和$\mu_k$为参数；$Z$为规范化因子，用于确保上式是正确定义的概率。

显然，要使用条件随机场，还需定义合适的特征函数。特征函数通常是实值函数，用来刻画数据的一些很可能成立或期望成立的经验特性。通常取值为1或0：当满足特征条件时取值为1，否则为0。

举个例子，给出$X$和$Y$两个序列，如下图所示：

![FF.JPG](https://i.loli.net/2020/04/30/4pA3LUqn5Ta8SzY.jpg)

转移特征函数可表示为：
$$
\begin{equation}
t_j(y_{i+1,y_i,X,i})=
\begin{cases}
1,&if \ y_{i+1}=[P],y_i=[V] and \ x_i="knock"\\
0,& otherwise
\end{cases}
\end{equation}
$$
表示第$i$个观测值$x_i$为单词"knock"时，相应的标记$y_i$和$y_{i+1}$很可能分别为$[V]$和$[P]$。

状态特征函数表示为：
$$
\begin{equation}
s_k(y_{y_i,X,i})=
\begin{cases}
1,&if \ y_i=[V] and \ x_i="knock"\\
0,& otherwise
\end{cases}
\end{equation}
$$
表示观测值$x_i$为单词"knock"时，它所对应的标记很可能是$[V]$。

注意：条件随机场和马尔可夫随机场均使用势函数定义概率，两者在形式上没有显著的区别；但是条件随机场构建的是条件概率，而马尔可夫随机场构建的是联合概率。