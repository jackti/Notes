# 词向量之word2vec(三)

在基于Hierarchical Softmax算法中使用了Huffman树来代替传统的神经网络，提高了模型训练的效率。但是也存在一些问题，如果训练样本的中心词$w$是一个很生僻的词，那么就会导致在Huffman树路径很长。Negative Sampling是另一种求解word2vec模型的方法，摒弃了Huffman树，采用了Negative Sampling(负采样)的方法来求解。

假设现有一个训练样本，中心词是$w$，它的周围上下文共有$2c$个词，记做$Context(w)$。由于中心词$w$的确和$Context(w)$相关存在，因此是一个真实的正例。通过负采样得到neg个和$w$不同的中心词$w_i,i=1,2,...,neg$。这样$Context(w)$和$w_i$组成了neg个并不真实存在的负例。利用这一个正例和neg个负例，可以进行二元逻辑回归，得到负采样的每个词$w_i$对应的模型$\theta_i$和每次词的词向量。

## Negative Sampling负采样方法

为了得到neg个负例，word2vec采用了基于词频的采样方法。如果词汇表大小为$V$，那么可以将一段长度为1的线段分成$V$份，每份对应词汇表中的一个词，且每个词对应的线段长度是不一样的，高频词对应的线段长，低频词对应的线段短。每次词$w$的线段长度有下式决定

$$
len(w)=\frac{count(w)}{\sum_\limits{u \in vocab}count(u)} \notag
$$

在word2vec中，分子和分母都取了3/4次幂如下

$$
len(w)=\frac{count(w)^{3/4}}{\sum_\limits{u \in vocab}count(u)^{3/4}} \notag
$$

![neg_sample](assets/neg_sample.png)

在采样前将这段长度为1的线段划分成$M$等份这里$M \gg V$ 这样就可以保证每个词对应的线段都会划分成对应的小块。而$M$份中的每一份都会落在某一个词对应的线段上。在采样的时候只需要从$M$个位置中采样出neg个位置，此时采样得到的每一个位置对应的线段所属的词就是负采样词。在word2vec中$M$取值默认为$10^8$。



## 梯度计算

Negative Sampling也采用二元逻辑回归来求解模型参数，通过负采样可以得到neg个负例$(Context(w),w_i)，i=1,2,...,neg$。为了统一描述，这里将正例定义为$w_0$。

在逻辑回归中正例应该满足：

$$
p(Context(w_0),w_0)=\sigma(x_{w_0}^T \theta^{w_0}), \quad y_0=1
$$

负例应该满足：

$$
p(Context(w_0),w_i)=1-\sigma(x_{w_i}^T \theta^{w_i}), \quad y_i=0 ,\quad i=1,2,...,neg
$$

所以由式(1)(2)此时模型的似然函数可以写成：

$$
L(\theta^{w_i},x_{w_i})=\sigma(x_{w_0}^T \theta^{w_0})\prod_{i=1}^{neg} \bigg(1-\sigma(x_{w_i}^T \theta^{w_i})\bigg)
=\prod_{i=0}^{neg} \sigma(x_{w_i}^T\theta^{w_i})^{y_i}(1-\sigma(x_{w_i}^T\theta^{w_i}))^{1-y_i} 
$$


从形式上看最大化$L$相当于最大化样本$\sigma(x_{w_0}^T \theta^{w_0})$，同时最小化$\sigma(x_{w_i}^T \theta^{w_i}),i=1,2,...,neg$ 背后的含义就是增大正样本的概率同时降低负样本的概率。

此时对应的对数似然函数为

$$
\mathcal{L}(\theta^{w_i},x_{w_i})
= \sum_{i=0}^{neg} \bigg(  y_i \log  \sigma(x_{w_i}^T\theta^{w_i})+(1-y_i)\log(1- \sigma(x_{w_i}^T\theta^{w_i})) \bigg)
$$

和Hierarchical Softmax类似，采用随机梯度上升法，每次只用一个样本更新梯度，来进行迭代更新参数$\theta^{w_i},x_{w_i},i=0,1,...,neg$ 。

首先计算$\mathcal{L}$关于$\theta^{w_i}$的偏导数有

$$
\begin{align}
\frac{\partial L}{\partial \theta^{w_i}}
& =y_i (1- \sigma(x_{w_i}^T\theta^{w_i}))x_{w_i}-(1-y_i)\sigma(x_{w_i}^T\theta^{w_i})x_{w_i}
\\
&=(y_i-\sigma(x_{w_i}^T\theta^{w_i}))x_{w_i}
\end{align}
$$

同理可以得到$\mathcal{L}$关于$x_{w_i}$的偏导数有

$$
\frac{\partial L}{\partial x_{w_i}}=(y_i-\sigma(x_{w_i}^T\theta^{w_i}))\theta^{w_i} 
$$

有了梯度就可以采用梯度上升法进行迭代来一步步求解参数$x_{w_i},\theta^{w_i},i=0,1,2,...,neg$。



## 基于Negative Sampling的CBOW模型

基于Negative Sampling的CBOW模型，使用随机梯度上升算法的流程

>输入：基于CBOW语料训练样本，词向量维度$M$，CBOW上下文大小$2c$，步长$\eta$
>
>输出：词汇表每个词的模型参数$\theta$，所有的词向量$x_w$
>
>（1）随机初始化所有的模型参数$\theta$，所有的词向量$w$
>
>（2）对于每个训练样本$(Context(w_0),w_0)$负采样出neg个负例中心词$w_i,i=1,2,...,neg$
>
>（3）进行梯度上升迭代过程，对于训练集中没一个样本$(Context(w_0),w_0,w_1,..,w_{neg})$进行如下处理
>
>​	① $e=0$，计算$x_{w_0}=\frac{1}{2c}\sum_{i=1}^{2c}x_i$
>
>​	② for i = 0 to neg 计算
>$$
>\begin{aligned}
>f &=\sigma(x_{w_i}^T \theta^{w_i})
>\\
>g &= \eta(y_i-f)
>\\
>e &= e+ g\theta^{w_i}
>\\
>\theta^{w_i} &= \theta^{w_i}+ gx_{w_i}
>\end{aligned}
>$$
>​	③ 对于$Context(w)$中的每一个词向量$x_k$(共2c个)进行更新：
>$$
>x_k = x_k+e \notag
>$$
>（4）如果梯度收敛，则结束梯度迭代，否则返回步骤（3）继续迭代





## 基于Negative Sampling的Skig-gram模型

基于Negative Sampling的Skig-gram模型，使用随机梯度上升算法的流程

> 输入：基于Skig-gram语料训练样本，词向量维度$M$，Skig-gram上下文大小$2c$，负采样个数neg，步长$\eta$
>
> 输出：词汇表每个词的模型参数$\theta$，所有的词向量$x_w$
>
> （1）随机初始化所有的模型参数$\theta$，所有的词向量$w$
>
> （2）对于每个训练样本$(Context(w_0),w_0)$负采样出neg个负例中心词$w_i,i=1,2,...,neg$
>
> （3）进行梯度上升迭代过程，对于训练集中没一个样本$(Context(w_0),w_0,w_1,..,w_{neg})$进行如下处理
>
> ​	① for i =1 to 2c:
>
> ​		(a) $e=0$
>
> ​		(b) for j = 0 to neg 计算
> $$
> \begin{aligned}
> f &=\sigma(x_{w_i}^T \theta^{w_j})
> \\
> g &= \eta(y_j-f)
> \\
> e &= e+ g\theta^{w_j}
> \\
> \theta^{w_j} &= \theta^{w_j}+ gx_{w_i}
> \end{aligned}
> $$
> ​	③ 对于$Context(w)$中的每一个词向量$x_k$(共2c个)进行更新：
> $$
> x_k = x_k+e \notag
> $$
> （4）如果梯度收敛，则结束梯度迭代，否则返回步骤（3）继续迭代



