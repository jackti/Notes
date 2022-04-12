# 集成学习算法之Adaboost

boosting提升方法的基本思路：对于一个复杂任务，将多个专家的判断进行适当的综合所得出的判断，要比其中任何一个专家单独的判断要好。

对于分类问题而言，提升方法就是从弱学习器算法出发，反复学习，得到一系列弱分类器（又称为基本分类器），然后组合这些弱分类器，构成一个强分类器。大多数的提升方法都是改变训练数据的概率分布（训练数据的权重分布），针对不同的训练数据分布调用弱学习算法学习一系列弱分类器。

## Adaboost算法原理

> 输入：训练数据集$T=\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(N)},y^{(N)})\}$ 和弱学习器算法，其中$x^{(i)}\in\mathbb{R}^n,y^{(i)}\in\{+1,-1\}$
>
> 输出：最终的分类器$G(x)$
>
> (1)初始化训练数据的分布
>
> $$
> D_1=(w_{11},...,w_{1i},...,w_{1N}),w_{1i}=\frac{1}{N}\quad i=1,2,...,N \notag
> $$
>
> (2)对于$m=1,2,...,M$
>
> ​	①使用具有权值分布$D_m$的训练数据集学习，得到基本分类器
>
> $$
> G_m(x):\mathcal{X}\Rightarrow\{-1,+1\}\notag
> $$
>
> ​	②计算$G_m(x)$在训练数据集上的分类误差率
>
> $$
> e_m = P(G_m(x^{(i)})\ne y^{(i)})=\sum_{i=1}^Nw_{mi}I(G_m(x^{(i)})\ne y^{(i)})=\sum_{G_m(x^{(i)})\ne y^{(i)}}w_{mi}\notag
> $$
>
> ​	这里$w_{mi}$表示第$m$轮中第$i$个实例的权值，且$\sum_{i=1}^N w_{mi}=1$。这表明$G_m(x)$在加权的训练数据集上的分类误差率是$G_m(x)$误分类样本的权值之和。
>
> ​	③计算$G_m(x)$的系数
>
> $$
> \alpha_m=\frac{1}{2}\ln\frac{1-e_m}{e_m}\notag
> $$
>
> $\alpha_m$表示$G_m(x)$在最终分类器中的重要性。当$e_m\le\frac{1}{2}$时，$\alpha_m\ge0$并且$\alpha_m$随着$e_m$的减小而增大，所以误分类率越小的基本分类器在最终的分类器中的作用越大。
>
> ​	④跟新训练数据集的权值分布
>
> $$
> \begin{array}{}
> D_{m+1}=(w_{m+1,1},w_{m+1,2},...,w_{m+1,N}) \notag \\
> w_{m+1,i}=\frac{w_{mi}}{Z_m}\exp(-\alpha_my^{(i)}G_m(x^{(i)})) \qquad i=1,2,...,N
> \end{array}
> $$
>
> 这里$Z_m$是规范化因子
>
> $$
> Z_m=\sum_{i=1}^N w_{mi}\exp(-\alpha_my^{(i)}G_m(x^{(i)}))\notag
> $$
>
> 它使$D_{m+1}$称为一个概率分布。可知被基本分类器$G_m(x)$误分类的样本权重得以扩大，而被正确分类样本的权重得以缩小。
>
> (3)构建基本分类器的线性组合
>
> $$
> f(x)=\sum_{i=1}^M\alpha_mG_m(x) \notag
> $$
>
> 线性组合$f(x)$实现$M$个基本分类器的加权表决。系数$\alpha_m$表示基本分类器$G_m(x)$的重要性(但是这里所有$\alpha_m$之和并不为1)，$f(x)$的符号决定实例$x$的类，$f(x)$的绝对值表示分类的确信度。



## Adaboost算法的解释

在Adaboost算法中直接给出了弱学习器系数公式和样本权重更新迭代公式，并没有给出推导。至于推导过程则可以从Adaboost算法的另一种解释得到：**Adaboost算法是模型为加法模型，损失函数为指数函数，学习算法为前向分步算法的分类问题。**

模型为加法模型很好理解：最终的强分类器是若干个弱分类器加权平均而得到的。

### 前向分步算法

考虑加法模型

$$
f(x)=\sum_{i=1}^M\beta_mb(x;\gamma_m)
$$

其中$b(x;\gamma_m)$为基函数，$\gamma_m$是基函数的参数，$\beta_m$为基函数的系数。式(1)是一个明显的加法模型。

在给定训练数据以及损失函数$L(y,f(x))$的条件下，学习加法模型$f(x)$可以看成是经验风险极小化即损失函数极小化问题：

$$
\begin{align}
&\min_{\beta_m,\gamma_m}\sum_{i=1}^NL(y^{(i)},f(x^{(i)}))\\
\Rightarrow &\min_{\beta_m,\gamma_m} \sum_{i=1}^NL(y^{(i)},\sum_{m=1}^M\beta_mb(x^{(i)},\gamma_m))
\end{align}
$$

通常式(3)是一个复杂的优化问题，需要同时求解出从$m=1$到$M$的所有参数$\beta_m,\gamma_m$。前向分布算法求解这一优化问题的思想是：由于是加法模型，如果能够从前往后，每一步只学习一个基函数及其系数，逐步逼近优化目标函数，那么就可以简化优化的复杂度。即每步只需优化损失函数：

$$
\min_{\beta,\gamma}\sum_{i=1}^N L(y^{(i)},\beta b(x^{(i)};\gamma))
$$

前向分步算法步骤如下：

> 输入：训练数据集$T=\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(N)},y^{(N)})\}$ ；损失函数$L(y,f(x))$；基函数集$\{ b(x;\gamma)\}$
>
> 输出：加法模型$f(x)$
>
> (1)初始化$f_0(x)=0$
>
> (2)对$m=1,2,...,M$
>
> ​	①极小化损失函数
>
> $$
> (\beta_m,\gamma_m)= \arg \min_{\beta,\gamma}\sum_{i=1}^N L(y^{(i)},f_{m-1}(x)+\beta b(x^{(i)};\gamma)) \notag
> $$
>
> ​	    得到参数$\beta_m,\gamma_m$
>
> ​	②更新
>
> $$
> f_m(x)=f_{m-1}(x)+\beta_mb(x;\gamma_m)\notag
> $$
>
> ​	③得到加法模型
>
> $$
> f(x)=f_M(x)=\sum_{m=1}^M \beta_mb(x;\gamma_m) \notag
> $$
>

前向分步算法的的优势在将同时求解出从$m=1$到$M$的所有参数$\beta_m,\gamma_m$的优化问题简化为逐次求解各个$\beta_m,\gamma_m$的优化问题。



### 前向分步算法与AdaBoost

通过前向分步算法是可以推导出AdaBoost算法的。其中加法模型为

$$
f(x)=\sum_{m=1}^M(\alpha_mG_m(x))
$$

损失函数为指数函数

$$
L(y,f(x))=\exp(-yf(x))
$$

假设经过$m-1$轮迭代前向分步算法已经得到$f_{m-1}(x)$：

$$
\begin{align}
f_{m-1}(x)
&=f_{m-2}(x)+\alpha_{m-1}G_{m-1}(x)\\
&=f_{m-3}(x)+\alpha_{m-2}G_{m-2}(x)+\alpha_{m-1}G_{m-1}(x)\\
&=\alpha_{1}G_{1}(x)+\alpha_{2}G_{2}(x)+...+\alpha_{m-1}G_{m-1}(x)\\
\end{align}
$$

在第$m$轮迭代得到$\alpha_m,G_m(x)$和$f(x)$

$$
f_{m}(x)=f_{m-1}(x)+\alpha_mG_m(x)
$$

目标是使前向分步算法得到$\alpha_m$和$G_m(x)$使$f_m(x)$在训练数据集上的指数损失最小，即

$$
\begin{align}
(\alpha_m,G_m(x))
&=\arg\min_{\alpha,G}\sum_{i=1}^N\exp(-y^{(i)}f_m(x^{(i)}))\\
&=\arg\min_{\alpha,G}\sum_{i=1}^N\exp[-y^{(i)}(f_{m-1}(x^{(i)})+\alpha G(x^{(i)}))]\\
&=\arg\min_{\alpha,G}\sum_{i=1}^N\exp[-y^{(i)}f_{m-1}(x^{(i)}) - y^{(i)}\alpha G(x^{(i)})]\\
&=\arg\min_{\alpha,G}\sum_{i=1}^N\{\exp(-y^{(i)}f_{m-1}(x^{(i)})   \cdot  \exp(-y^{(i)}\alpha G(x^{(i)}))\}\\
\end{align}
$$

令$\bar{w}_{mi}=\exp[-y^{(i)}f_{m-1}(x^{(i)})]$ ，由于$\bar{w}_{mi}$ 既不依赖$\alpha$也不依赖$G$，是一个与最小化无关，但是$\bar{w}_{mi}$是依赖于$f_{m-1}(x)$，随着每一轮迭代而发生改变。

由于$y^{(i)},G(x^{(i)})\in \{+1,-1\}$，所以$y\alpha G(x)$可以写成另一种形式

$$
y\alpha G(x)=
\left\{
\begin{array}{}
    \alpha &\quad y\cdot G(x)=1 \\
    -\alpha &\quad y\cdot G(x)=-1
\end{array}
\right.
$$

损失函数可以写成

$$
\begin{align}
L(\alpha,G)
&=\sum_{i=1}^N\{\exp(-y^{(i)}f_{m-1}(x^{(i)})   \cdot  \exp(-y^{(i)}\alpha G(x^{(i)}))\}\\
&=\sum_{i=1}^N\bar{w}_{mi}\cdot\exp(-y^{(i)}\alpha G(x^{(i)}))\\
&=\sum_{i=1}^N\bar{w}_{mi}\cdot\exp(-y^{(i)}\alpha G(x^{(i)}))\\
&=\sum_{y^{(i)}=G_m(x^{(i)})}\bar{w}_{mi} e^{-\alpha}+\sum_{y^{(i)}\ne G_m(x^{(i)})}\bar{w}_{mi} e^{\alpha}\\
&= e^{-\alpha} \sum_{y^{(i)}=G_m(x^{(i)})}\bar{w}_{mi} + e^{\alpha} \sum_{y^{(i)}\ne G_m(x^{(i)})}\bar{w}_{mi} \\
&= e^{-\alpha} \sum_{i=1}^N \bar{w}_{mi} I(y^{(i)}=G_{m}(x^{(i)})) + e^{\alpha} \sum_{i=1}^N \bar{w}_{mi} I(y^{(i)}\ne G_{m}(x^{(i)}))\\
&=e^{-\alpha} \sum_{i=1}^N \bar{w}_{mi} +(e^{\alpha}+e^{-\alpha}) \sum_{i=1}^N \bar{w}_{mi} I(y^{(i)}\ne G_m(x^{(i)}))\\
\end{align}
$$

对$L$求导，得到：

$$
\frac{\partial L}{\partial \alpha}=-e^{-\alpha}\sum_{i=1}^N\bar{w}_{mi}+(e^{\alpha}+e^{-\alpha})\sum_{i=1}^N \bar{w}_{mi}I(y^{(i)}\ne G_m(x^{(i)}))
$$

令式(24)等于0，且定义$e_m=\sum_{i=1}^N \bar{w}_{mi}I(y^{(i)}\ne G_m(x^{(i)}))$，那么有：

$$
\begin{align}
(e^{\alpha}+e^{-\alpha})\sum_{i=1}^N \bar{w}_{mi}I(y^{(i)}\ne G_m(x^{(i)}))&=e^{-\alpha}\sum_{i=1}^N\bar{w}_{mi}\\
(e^{\alpha}+e^{-\alpha})e_m&=e^{-\alpha}\sum_{i=1}^N\bar{w}_{mi}\\
(e^{\alpha}+e^{-\alpha})e_m&=e^{-\alpha}\\
\end{align}
$$

式(25)进行移项操作，式(25)-(26)利用了定义$e_m=\sum_{i=1}^N \bar{w}_{mi}I(y^{(i)}\ne G_m(x^{(i)}))$ ，式(26)-(27)使用$\sum_{i=1}^N\bar{w}_{mi}=1$。最终可以求解得到：

$$
\alpha = \frac{1}{2}\ln \frac{1-e_m}{e_m}
$$

这里得到的$\alpha$和Adaboost算法中给出的公式完全一致。

最后来看每一轮样本权值的更新，由于$\bar{w}_{mi}=\exp(-y^{(i)}f_{m-1}(x^{(i)}))$，于是得到：

$$
\begin{align}
\bar{w}_{m+1,i}
&=\exp(-y^{(i)}f_{m}(x^{(i)}))\\
&=\exp[-y^{(i)}(f_{m-1}(x^{(i)})+\alpha_m G_m(x^{(i)}))]\\
&=\bar{w}_{mi}\exp(-y^{(i)}\alpha_m G_m(x^{(i)}))\\
\end{align}
$$

式(31)和Adaboost算法中样本权值的更新，只是相差了规范化因子。



## Adaboost算法误差分析

Adaboost最基本的性质是在学习过程中不断减少训练误差，即在训练数据集上的分类误差率。

>  定理：Adaboost算法最终分类器的训练误差界为
>
> $$
> \frac{1}{N}\sum_{i=1}^N I(G(x^{(i)})\ne y^{(i)})\le \frac{1}{N}\sum_{i}\exp(-y^{(i)}f(x^{(i)}))=\prod_m Z_m
> $$
>

证明：首先证明第一个不等式：当$G(x^{(i)})\ne y^{(i)}$时，$y^{(i)}f(x^{(i)})\lt 0$，因而$\exp(-y^{(i)}f(x^{(i)}))\ge 1$。

得到

$$
\sum_i \exp(-y^{(i)}f(x^{(i)}))\ge N \\
\Rightarrow \quad
\frac{1}{N}\sum_i \exp(-y^{(i)}f(x^{(i)}))\ge \frac{1}{N}\cdot N =1 \ge  \frac{1}{N}\sum_{i=1}^N I(G(x^{(i)})\ne y^{(i)})\\
$$

证明第二个等式需要使用到$Z_m$的定义和样本权重初始值定义：

$$
\begin{align}
w_{1i}&=\frac{1}{N}\\
Z_m w_{m+1,i}&=w_{mi}\exp(-\alpha_m y^{(i)}G_m(x^{(i)}))
\end{align}
$$

那么有：

$$
\begin{align}
\frac{1}{N}&\sum_i \exp(y^{(i)}f(x^{(i)}))\\
&=\frac{1}{N}\sum_i \exp(  \sum_{m=1}^M\alpha_m y^{(i)}G_m(x^{(i)}) )\\
&=\sum_i \frac{1}{N} \exp(  \sum_{m=1}^M\alpha_m y^{(i)}G_m(x^{(i)}) )\\
&=\sum_i w_{1i} \exp(  \sum_{m=1}^M\alpha_m y^{(i)}G_m(x^{(i)}) )\\
&=\sum_i [ w_{1i}  \alpha_m y^{(1)}G_m(x^{(1)}) ]  \exp(  \sum_{i=2}^M\alpha_m y^{(i)}G_m(x^{(i)}) )\\
&=\sum_i [ w_{1i}  \alpha_m y^{(1)}G_m(x^{(1)}) ]  \exp(  \sum_{i=2}^M\alpha_m y^{(i)}G_m(x^{(i)}) )\\
&=\sum_i [ Z_1 w_{2i} ]  \exp(  \sum_{i=2}^M\alpha_m y^{(i)}G_m(x^{(i)}) )\\
&=Z_1\sum_i w_{2i}   \exp(  \sum_{i=2}^M\alpha_m y^{(i)}G_m(x^{(i)}) )\\
&=Z_1Z_2\sum_i w_{3i}   \exp(  \sum_{i=3}^M\alpha_m y^{(i)}G_m(x^{(i)}) )\\
&=...\\
&=Z_1Z_2 ... Z_{M-1}\sum_i w_{Mi}   \exp(  \sum_{i}\alpha_M y^{(i)}G_M(x^{(i)}) )\\
&=Z_1Z_2 ... Z_{M-1}Z_{M}\\
&=\prod_{i=1}^M Z_i\\
\end{align}
$$

这一定理说明，可以在每一轮选取适当的$G_m$使得$Z_m$最小，从而使训练误差下降最快。

> 对于$\alpha$的推导方式有两种：第一种是最小化损失函数进行推导（上面的式(28)的推导方式）；第二种是最小化训练误差界进行推导。现在对第二种方式进行推导：
>
> $$
> \begin{array}{}
> Z_m
> &=\sum_{i=1}^N w_{mi}\exp(-\alpha_my^{(i)}G_m(x^{(i)}))\\
> &=\sum_\limits{y^{(i)}=G_m(x^{(i)})} w_{mi} e^{-\alpha}+\sum_\limits{y^{(i)}\ne G_m(x^{(i)})}w_{mi} e^{\alpha}\\
> &=(1-e_m)\cdot e^{-\alpha_m}+e_m\cdot e^{\alpha_m}
> \end{array}
> $$
>
> 对式(48)求导等于0，可以得到：
>
> $$
> \frac{\partial Z_m}{\partial \alpha_m}=-(1-e_m)\cdot e^{-\alpha_m}+e_m\cdot e^{\alpha_m}=0
> $$
>
> 求得
>
> $$
> \alpha_m = \frac{1}{2}\ln \frac{1-e_m}{e_m}
> $$
>







### Adaboost算法扩展

对于Adaboost多元分类算法，原理和二元分类类似，主要的区别在于弱分类器的系数上。比如Adaboost SAMME算法，它的弱分类器系数
$$
\alpha_m =\frac{1}{2}\ln\frac{1-e_m}{e_m}+\ln(C-1)
$$
其中$C$是类别数。如果是二元分类则$C=2$，和二元分类算法中的系数是一致的。

Adaboost算法也可以处理回归问题，常见的有Adaboost R2回归等都类似。
























