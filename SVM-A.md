Support Vector Machine（SVM）——Part A

By Xyao

# 前言

支持向量机是深度学习还未蓬勃发展之前，机器学习算法中最重要的支柱，学习支持向量机可以说是机器学习之路上重要而又困难的一个关卡。本文希望帮助读者能够清晰梳理支持向量机的"脉络"，学习起来可以清楚且轻松。

完整梳理支持向量机，内容也是相当多的，因此计划分为三部分进行叙述。

支持向量机有三宝：间隔(Margin)、对偶(Dual)、核技巧(Kernel Trick)，这"三宝"也是完整理解SVM的关键所在。

根据训练数据的特性，SVM可以分为三类：

1. **硬间隔支持向量机**（Hard Margin SVM）：面对的训练数据为**线性可分**。
2. **软间隔支持向量机**（Soft Margin SVM）：面对的训练数据为**近似线性可分**。
3. **核技巧支持向量机**（Kernel Trick SVM）：面对的训练数据为**线性不可分**。

我们从最基础的硬间隔支持向量机来开启SVM学习之路。

# 硬间隔支持向量机

硬间隔支持向量机面对的训练数据是线性可分的，我们先来看一个简单的例子：

<img src="http://q9qozit0b.bkt.clouddn.com/SVM_1.JPG" alt="SVM_1.JPG" style="zoom:50%;" />

如上图所示，在这样一个数据集中，存在多个可以将数据集正确划分为两类的超平面。那么。哪个超平面最好呢？或者说该如何去选择最优的超平面呢？我们直观上看，应该去找位于两类训练数据"正中间"的超平面（图中为加粗的黑线），因为这样的超平面对训练数据本身的局部扰动拥有最好的"容忍性"。

- 问：为什么这么说？或者如何理解"容忍性"？
- 答：由于训练数据的局限性或者噪声的影响，训练集外的数据可能比上图中的数据更接近两个类的分隔界面，这将使得许多超平面划分错误，而加粗的黑线所代表的超平面受影响最小，即具有最好的"容忍性"，或者说泛化能力最强。

这样的划分超平面可以通过如下线性方程来描述：
$$
w^Tx+b=0
$$
从而SVM模型可以定义为：
$$
f_w(x)=sign(w^Tx+b)
$$
其中，$sign()$函数称为分类决策函数，对于某样本$(x_i,y_i)$，若$w^Tx_i+b>0$，则$sign(w^Tx_i+b)=+1$；若$w^Tx_i+b<0$，则$sign(w^Tx_i+b)=-1$。

注意：在SVM中，我们的$y_i$取值为$+1$或$-1$，而不是$0$或$1$。

稍微总结一下，可以说，SVM的**核心问题**是如何找到"容忍性"最好的超平面。那么，怎样用数学语言来对<"容忍性"最好>这一概念进行描述呢？SVM就提出了**最大间隔分类器**的思想。

- **最大**：顾名思义，最大化，即Maximum。
- **间隔**：定义间隔函数，Margin Function。
- **分类器**：即如何决策分类。

最大化没什么好讲，就是字面意思。讲一讲"间隔"。

## 间隔

空间中每一个样本点都可以计算它到超平面的最短距离（即垂直距离），假设现在有$A,B,C$三个样本点，他们都被划分在超平面的同一侧，它们到超平面的最短距离分别为$0.1,1,100$，可以看出$C$点离得最远，那么我们很有理由相信$C$点是被正确划分的；而$A$点离超平面很近，我们就不那么确信$A$点的分类结果；对$B$点的分类确信度则介于$A$和$C$之间。那么，我们可以看出样本点到超平面的远近可以表示分类预测的确信程度。

因此，对于某一给定的超平面$w^Tx+b=0$的情况下，我们可以计算出所有样本点到超平面的距离，即：
$$
distance = \frac{|w^Tx+b|}{||w||}
$$
距离计算公式很简单，就是二维空间到高维空间的一个拓展（二维空间即点到直线的距离）。

我们的间隔函数就取为所有样本的距离的最小值，即：
$$
margin\ function=\min\limits_{i=1,...,N}distance=\min\limits_{i=1,...,N}\frac{|w^Tx_i+b|}{||w||}
$$
那么，我们在最大化间隔的时候，也就是在最大化"最小距离"，即使得最小确信度尽可能地大。（可能有些拗口，建议多读几遍。）

对于每个样本点，根据决策分类函数，可以有以下推导：
$$
\begin{equation}
\begin{cases}
w^Tx_i+b>0,&y_i=+1\\
w^Tx_i+b<0,&y_i=-1
\end{cases}
\end{equation}
$$

$$
\rightarrow |w^Tx_i+b|=y_i(w^Tx_i+b)
$$

$$
\rightarrow\max\ margin\ function=\max\min\limits_{i=1,2,...,N}\frac{y_i(w^Tx_i+b)}{||w||}
$$

## 分类器

说分类器倒不如说是一种约束，它对样本施加了约束，基于给定的超平面，应符合：
$$
\begin{equation}
\begin{cases}
w^Tx_i+b>0,&y_i=+1\\
w^Tx_i+b<0,&y_i=-1
\end{cases}
\end{equation}
$$

$$
\rightarrow s.t. \ y_i(w^Tx_i+b)>0
$$

对于该约束问题，我们可以理解为，必然存在一个大于0的数（记为$\gamma$），使得$\min\limits_{i=1,2,...,N}y_i(w^Tx_i+b)$等于$\gamma$，即：
$$
\begin{equation}
\exists\gamma>0,s.t.\min\limits_{i=1,2,...,N}y_i(w^Tx_i+b)=\gamma
\end{equation}
$$

## 小结

对上述进行总结整理，可以得到"最大间隔分类器"的约束最优化问题，即：
$$
\begin{equation}
\begin{cases}
\max\limits_{w,b}\min\limits_{i=1,2,...,N}\frac{y_i(w^Tx_i+b)}{||w||}\\
s.t.\ y_i(w^Tx_i+b)>0,&i=1,2,...N
\end{cases}
\end{equation}
$$

$$
\rightarrow\begin{equation}
\begin{cases}
\max\limits_{w,b}\min\limits_{i=1,2,...,N}\frac{y_i(w^Tx_i+b)}{||w||}\\
s.t.\min\limits_{i=1,2,...,N}y_i(w^Tx_i+b)=\gamma,&i=1,2,...N
\end{cases}
\end{equation}
$$

$$
\rightarrow\begin{equation}
\begin{cases}
\max\limits_{w,b}\frac{\gamma}{||w||}\\
s.t.\min\limits_{i=1,2,...,N}y_i(w^Tx_i+b)=\gamma,&i=1,2,...N
\end{cases}
\end{equation}
$$

事实上，$\gamma$的取值并不影响最优化问题的求解，例如，对于超平面$w_1^Tx_i+b_1=0$而言，取$w_2=2w_1$和$b_2=2b_1$，超平面$w_2^Tx_2+b_2=0$仍然是同一个超平面，即$w$和$b$的同比缩放并不改变超平面本身，所以我们完全有理由取到一组$w,b$使得$\gamma=1$，即：
$$
\rightarrow\begin{equation}
\begin{cases}
\max\limits_{w,b}\frac{1}{||w||}\\
s.t.\min\limits_{i=1,2,...,N}y_i(w^Tx_i+b)=1,&i=1,2,...N
\end{cases}
\end{equation}
$$
我们又注意到，最大化$\frac{1}{||w||}$和最小化$\frac{1}{2}||w||^2$是等价的，从而推出：
$$
\rightarrow\begin{equation}
\begin{cases}
\min\limits_{w,b}\frac{1}{2}||w||^2\\
s.t.\min\limits_{i=1,2,...,N}y_i(w^Tx_i+b)=1,&i=1,2,...N
\end{cases}
\end{equation}
$$

$$
\rightarrow\begin{equation}
\begin{cases}
\min\limits_{w,b}\frac{1}{2}||w||^2\\
s.t.\ y_i(w^Tx_i+b)\ge1,&i=1,2,...N
\end{cases}
\end{equation}
$$

这是一个凸二次规划问题（Convex Quadratic Programming Problem）。

至此，我们推导出了硬间隔支持向量机的约束最优化问题。

## 补充：存在性、唯一性问题

可能有人会问，在上述约束最优化问题中，最终一定有解吗？即使有解，解一定是唯一的吗？

下面，对上述问题进行证明：

（1）首先，对于**存在性问题**进行证明：

我们回顾一下上文（重点看一下那张图），硬间隔支持向量机面对的训练数据集是**线性可分**的，所以，必然存在超平面能够对数据集进行正确划分（而且存在多个可行的超平面），硬间隔支持向量机希望做到的事情是从这些可行超平面中求得"容忍性"最好的超平面，因此，必然存在满足条件的最优解。

（2）对于**唯一性问题**进行证明：

假设满足约束最优化问题的最优解有两个 $(w_1^{\ast},b_1^{\ast}) $ 和 $(w_2^{\ast},b_2^{\ast})$ 。那么，首先，它们需要满足最小化条件$\min\limits_{w,b}\frac{1}{2}||w||^2$，可以得出：
$$
||w_1^{\ast}||=||w_2^{\ast}||=min
$$
（总不能认为最小值还能有两个不一样的吧？？）

现在，我们取一组非最优解$w_3,b_3$，令
$$
w_3=\frac{w_1^{\ast}+w_2^{\ast}}{2},b_3=\frac{b_1^{\ast}+b_2^{\ast}}{2}
$$
易知$w_3,b_3$也是一组可行解，则有：
$$
min\le||w_3||\le\frac{1}{2}||w_1^{\ast}||+\frac{1}{2}||w_2^{\ast}||=min
$$
从而，得到$||w_3||=\frac{1}{2}||w_1^{\ast}||+\frac{1}{2}||w_2^{\ast}||=||\frac{w_1^{\ast}+w_2^{\ast}}{2}||$，当且仅当$w_1^{\ast}$与$w_2^{\ast}$两向量共线同向时取等，即：
$$
w_1^{\ast}=\lambda w_2^{\ast}
$$
又因为$||w_1^{\ast}||=||w_2^{\ast}||=min$，则有$||w_1^{\ast}||=|\lambda|\cdot||w_2^{\ast}||$，可以推出：
$$
|\lambda|=1\rightarrow\lambda=\pm1
$$
分两种情况，分别推导：

1. 若$\lambda=-1$，则$w_1^{\ast}=w_2^{\ast}=0$，代入约束条件则有$y_i(0\cdot x+b)-1\ge0,i=1,2,...,N$，显然$b$无解，该情况不可取。
2. 若$\lambda=1$，则$w_1^{\ast}=w_2^{\ast}$，可以用同一个符号来表示，记为$w^{\ast}$。

下面，再来证明$b_1^{\ast}=b_2^{\ast}$:

假设$x_1'$和$x_2'$是集合$\lbrace x_i|y_i=+1\rbrace$中分别对应于$(w^{\ast},b_1^{\ast})$和$(w^{\ast},b_2^{\ast})$使得优化问题不等式等号成立的点，$x_1''$和$x_2''$是集合$\lbrace x_i|y_1=-1\rbrace$中分别对应于$(w^{\ast},b_1^{\ast})$和$(w^{\ast},b_2^{\ast})$使得优化问题不等式等号成立的点，即：
$$
w^{\ast}\cdot x_1'+b_1^{\ast}=1
$$

$$
w^{\ast}\cdot x_2'+b_2^{\ast}=1
$$

$$
w^{\ast}\cdot x_1''+b_1^{\ast}=-1
$$

$$
w^{\ast}\cdot x_2''+b_2^{\ast}=-1
$$

可以得到，
$$
b_1^{\ast}=-\frac{1}{2}(w^{\ast}\cdot x_1'+w^{\ast}\cdot x_1'')
$$

$$
b_2^{\ast}=-\frac{1}{2}(w^{\ast}\cdot x_2'+w^{\ast}\cdot x_2'')
$$

$$
\rightarrow b_1^{\ast}-b_2^{\ast}=-\frac{1}{2}[w^{\ast}\cdot(x_1'-x_2')+w^{\ast}\cdot(x_1''-x_2'')]
$$

我们将$x_1'$和$x_2'$分别代入到$(w^{\ast},b_2^{\ast})$和$(w^{\ast},b_1^{\ast})$组成的超平面中，则有
$$
w^{\ast}\cdot x_2'+b_1^{\ast}\ge1=w^{\ast}\cdot x_1'+b_1^{\ast}
$$

$$
w^{\ast}\cdot x_1'+b_2^{\ast}\ge1=w^{\ast}\cdot x_2'+b_2^{\ast}
$$

可以得出:
$$
w^{\ast}\cdot(x_2'-x_1')\ge0
$$

$$
w^{\ast}\cdot(x_1'-x_2')\ge0
$$

$$
\rightarrow w^{\ast}\cdot(x_1'-x_2')=0
$$

同理，可以得到
$$
w^{\ast}\cdot(x_1''-x_2'')=0
$$
因此，
$$
b_1^{\ast}=b_2^{\ast}
$$
最终，$w_1^{\ast}=w_2^{\ast};b_1^{\ast}=b_2^{\ast}$，即最优解唯一。

