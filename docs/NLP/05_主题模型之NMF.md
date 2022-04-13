# 主题模型之NMF

在LSI主题模型中使用了奇异值分解，面临着高维度计算量太大的问题，于是出现了非负矩阵分解(NMF)模型，它同样使用矩阵分解但是计算量和处理速度比LSI快。



## 非负矩阵分解算法原理

非负矩阵分解(non-negative matrix factorization,简称NMF)是一种常用的矩阵分解方法，它可以适用于很多领域，如图像特征识别、语音识别等等。

奇异值分解将一个矩阵分解为三个矩阵：

$$
A = U\Sigma V^T
$$

如果降维到$k$维，则表达式为：

$$
A_{m\times n} \approx U_{m\times k}\Sigma_{k\times k} V^T_{k\times n}
$$

虽然NMF也是矩阵分解，但却使用了不同的思路，其目标是将矩阵分解为两个矩阵：

$$
A_{m\times n}=W_{m\times k}H_{k\times n}
$$

同时要求$W_{m\times k}\ge 0 ,H_{k\times n}\ge 0$。这个方法和协同过滤中矩阵分解的FunkSVD思路是一致的。



## NMF算法优化求解

为了定量比较矩阵$A_{m\times n}$和分解后$W_{m\times k}H_{k\times n}$的近似程度，可以采用以下两种损失函数定义方法：

(1)平方距离

$$
||A-B||^2=\sum_{i,j}(A_{ij}-B_{ij})^2
$$

(2)KL散度

$$
D(A||B) =\sum_{ij}(A_{ij}log\frac{A_ij}{B_{ij}}-A_{ij}+B_{ij})
$$

于是NFM算法针对平方距离的优化目标可以写成：

$$
\arg\min_{W,H}||V-WH||^2
\\
s.t. W \ge 0, H \ge 0
$$

针对KL散度的优化目标可以写成：

$$
\arg\min_{W,H}D(V||WH)
\\
s.t. W \ge 0, H \ge 0
$$

下面以平方损失函数为例，进行推导。平方损失函数可以写成如下形式。

$$
J(W,H)=\frac{1}{2}||V-WH||^2=\frac{1}{2}\sum_{ij}||V_{ij}-(WH)_{ij}||^2
$$

分别对$W,H$求偏导可以得到：

$$
\begin{align}
\frac{\partial J(W,H)}{\partial W_{ik}}
&=\sum_{j}(V_{ij}-(WH)_{ij})\cdot \frac{\partial (WH)_{ij}}{\partial W_{ik}}\\
&=\sum_{j}H_{kj}(V_{ij}-(WH)_{ij})\\
&=\sum_{j}V_{ij}H_{kj}-\sum_{j}(WH)_{ij}H_{kj}\\
&=(VH^T)_{ik}-(WHH^T)_{ik}\\
\end{align}
$$

同理可以得到：

$$
\frac{\partial J(W,H)}{\partial H_{kj}}=(W^TV)_{kj}-(W^TWH)_{kj}
$$

使用梯度下降算法进行迭代有：

$$
\begin{align}
W_{ik}=W_{ik}-\alpha_1\cdot[(VH^T)_{ik}-(WHH^T)_{ik}]\\
H_{kj}=H_{kj}-\alpha_2\cdot [(W^TV)_{kj}-(W^TWH)_{kj}]
\end{align}
$$

如果选取

$$
\alpha_1 = \frac{W_{ik}}{(W^TWH)_{ik}} \qquad  \alpha_2=\frac{H_{kj}}{(WHH^T)_{kj}}
$$

那么最终的迭代式：

$$
\begin{align}
W_{ik}=W_{ik}\cdot \frac{(VH^T)_{ik}}{(WHH^T)_{ik}}\\
H_{kj}=H_{kj}\cdot \frac{(W^TV)_{kj}}{(W^TWH)_{kj}}
\end{align}
$$


## NMF算法应用

假设输入有$m$个文本，$n$个词，而$A_{ij}$对应第$i$个文本的第$j$个词的特征值，这里最常用的就是预处理后标准化的TF-IDF值。设主题数为$k$，一般要比文本数小。NMF分解后，$W_{ik}$对应的第$i$个文本和第$k$个主题的概率相关度，而$H_{kj}$对应第$j$个词和第$k$个主题的概率相关度。

由于NMF算法是非负的矩阵分解，得到的$W,H$矩阵值的大小是可以用概率值的角度去看，从而得到文本和主题的概率分布关系。

NMF除了应用在出题模型中，还可以应用在图像处理、语音识别、信号处理和医药工程等，是一个普适的方法。在相关领域只需要将NMF套入一个合适的模型，使得$W,H$矩阵都可以有明确的意义。



