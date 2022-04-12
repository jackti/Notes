# 聚类算法之Mean Shift

MeanShift算法，又称为均值漂移算法，和K-Means算法一样，都是基于聚类中心的聚类算法，但是MeanShift算法不需要事先指定类别个数$k$。在MeanShift算法中，聚类中心是通过在给定区域中的样本的均值来确定的，通过不断更新聚类中心，直到最终的聚类中心不再改变为止。

MeanShift算法在聚类、图像平滑、分割和视频跟踪等方面有广泛应用。



## MeanShift向量

对于给定的$n$维向量空间$\mathbb{R}^n$中的$m$个样本点$(x^{(1)},x^{(2)},...,x^{(m)})$，对于其中一个样本$x^{(i)}$，其MeanShift向量为：

$$
M_h(x)=\frac{1}{k}\sum_{x^{(i)}\in S_h}(x^{(i)}-x)
$$

其中$S_h$值的是一个半径为$h$的高纬球区域。$S_h$的定义为：

$$
S_h(x)=\{y|(y-x)^T(y-x)\le h^2\}
$$

在$S_h$的区域内，每一个样本点$x^{(i)}$对样本$x$的贡献是一样。而在实际中，每一个样本点$x^{(i)}$对于样本$x$的贡献是不同的，这种贡献与$x$到每一个点之间的距离是相关的。

基于以上的考虑，对基本的MeanShift向量形式中增加核函数和样本权重，得到如下的改进MeanShift向量形式：

$$
\begin{align}
M_h(x)
&=\frac{\sum_{x^{(i)}\in S_h} K(\frac{x^{(i)}-x}{h})(x^{(i)}-x)}{\sum_{x^{(i)}\in S_h} K(\frac{x^{(i)}-x}{h})}\\
&=\frac{\sum_{x^{(i)}\in S_h} K(\frac{x^{(i)}-x}{h})x^{(i)}}{\sum_{x^{(i)}\in S_h} K(\frac{x^{(i)}-x}{h})}-x\\
&=m_h(x)-x
\end{align}
$$

其中$m_h(x)\overset{def}=\frac{\sum_{x^{(i)}\in S_h} K(\frac{x^{(i)}-x}{h})x^{(i)}}{\sum_{x^{(i)}\in S_h} K(\frac{x^{(i)}-x}{h})},K(\frac{x^{(i)}-x}{h})$是高斯核函数。

















































