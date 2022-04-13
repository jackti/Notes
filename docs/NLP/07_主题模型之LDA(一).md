# 主题模型之LDA(一)

隐含狄利克雷分布(Latent Dirichlet Allocation,简称LDA)是一种广泛使用的主题模型。LDA是基于贝叶斯模型的，涉及到贝叶斯模型往往都离不开"先验分布"、"数据(似然)"和"后验分布"。在贝叶斯学派中有：
$$
先验分布+数据(似然)=后验分布  \notag
$$


## 共轭分布

共轭分布(conjugate distribution)的概率中一共涉及到三个分布：先验、似然和后验。如果先验分布和似然分布所确定的后验分布与该先验分布是属于同一类型的分布，则该先验分布和似然分布是共轭分布，也称为共轭先验。

假设随机变量$X$服从分布$p(X|\theta)$，其观测样本为$X=\{x_1,x_2,...,x_m\}$，参数$\theta$服从先验分布$\pi(\theta)$，那么后验分布为：
$$
p(\theta|X)=\frac{\pi(\theta)p(X|\theta)}{p(X)} \notag
$$
如果后验分布$p(\theta|X)$和先验分布$\pi(\theta)$是同种类型的分布，则称先验分布$\pi(\theta)$为似然分布$p(X|theta)$的共轭分布。比较常见的例子有：高斯分布是高斯分布是共轭分布，Beta分布是二项分布的共轭分布，Dirichlet分布是多项分布的共轭分布。



## 二项分布和Beta分布

对于数据似然，常常可以使用二项分布，其分布为$X\sim b(n,p)$：

$$
Binom(k|n,p)=\dbinom{k}{n} p^k(1-p)^{n-k}
$$

表示$n$个独立是/非试验中，成功$k$次的概率分布。其中$p$为试验成功的概率。当$n=1$时，二项分布就是伯努利分布。

现在希望找到和二项分布共轭的分布，也就是Beta分布。其表达式为：

$$
Beta(p|\alpha,\beta)=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}p^{\alpha-1}(1-p)^{\beta-1}
$$

其中$\Gamma(\cdot)$是Gamma函数，满足$\Gamma(x)=(x-1)!$

为了证明二项分布和Beta分布是共轭分布，只需要证明后验分布$P(p|n,k,\alpha,\beta)$也是Beta分布即可，推导如下：

$$
\begin{align}
P(p|n,k,\alpha,\beta) 
&\propto P(k|n,p)P(p|\alpha,\beta)\\
&=\dbinom{k}{n} p^k(1-p)^{n-k} \cdot \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}p^{\alpha-1}(1-p)^{\beta-1}\\
&\propto p^{k+\alpha-1}(1-p)^{n-k+\beta-1}
\end{align}
$$

将上面的式子归一化后，得到后验概率：

$$
P(p|n,k,\alpha,\beta) =\frac{\Gamma(\alpha+\beta+n)}{\Gamma(\alpha+k)\Gamma(\beta+n-k)} p^{k+\alpha-1}(1-p)^{n-k+\beta-1}
$$

后验分布是一个Beta分布，且有：

$$
Beta(p|\alpha,\beta)+BinomCount(k,n-k)=Beta(p|\alpha+k,\beta+n-k)
$$

Beta分布的期望计算：

$$
\begin{align}
\mathbb{E}(Beta(p|\alpha,\beta))
&=\int_0^1 p\cdot Beta(p|\alpha,\beta) dp\\
&=\int_0^1 p \cdot\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}p^{\alpha-1}(1-p)^{\beta-1}dp\\
&=\int_0^1 \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}p^{\alpha}(1-p)^{\beta-1}dp\\
&=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \cdot \frac{\Gamma(\alpha+1)\Gamma(\beta)}{\Gamma(\alpha+\beta+1)} \int_0^1 \frac{\Gamma(\alpha+\beta+1)}{\Gamma(\alpha+1)\Gamma(\beta)}p^{\alpha}(1-p)^{\beta-1}dp\\
&=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \cdot \frac{\Gamma(\alpha+1)\Gamma(\beta)}{\Gamma(\alpha+\beta+1)} \int_0^1 Beta(p|\alpha+1,\beta) dp\\
&=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \cdot \frac{\Gamma(\alpha+1)\Gamma(\beta)}{\Gamma(\alpha+\beta+1)} \cdot 1\\
&=\frac{\alpha}{\alpha+\beta}
\end{align}
$$



## 多项式分布和Dirchlet分布

多项式分布是一种重要的多维离散分布，它是二项式分布的推广。假设随机试验有$k$个可能的结果$A_1,A_2,...,A_k$每个结果出现的次数为随机变量$X_1,X_2,...,X_n$，每个结果出现的概率为$p_1,p_2,...,p_k$，$n$次独立重复试验中随机事件出现的次数分别为$n_1,n_2,...,n_k$的概率符合多项式分布，即：

$$
P(X_1=n_1,X_2=n_2,...,X_k=n_k)=\frac{n!}{n_1!n_2!...n_k!}p_1^{n_1}p_2^{n_2}...p_k^{n_k}
$$

超过二维的Beta分布一般称为狄利克雷(简称Dirichlet)分布。可以说Beta分布是Dirichlet分布在二维时的特殊形式，一般意义上的$k$维Dirichlet分布可以表示为：

$$
Dir(\vec{p}|\vec{\alpha})=\frac{\Gamma(\alpha_1+\alpha_2+...+\alpha_K)}{\Gamma(\alpha_1)\Gamma(\alpha_2)...\Gamma(\alpha_K)}p_1^{\alpha_1-1}p_2^{\alpha_2-1}...p_K^{\alpha_K-1}
=\frac{\Gamma(\sum_{k=1}^K \alpha_k)}{\prod_{k=1}^K \Gamma(\alpha_k)}\prod_{k=1}^K p_k^{\alpha_k-1}
$$

Dirichlet分布可以表示为$Dirichlet(\vec{p}|\vec{\alpha})$，而多项式分布可以表示为$multi(\vec{m}|n,\vec{p})$，可以得到如下结论：

$$
Dirichlet(\vec p|\vec \alpha) + MultiCount(\vec m) = Dirichlet(\vec p|\vec \alpha + \vec m)
$$

对于Dirichlet分布的期望，也有和Beta分布类似的性质：

$$
\mathbb{E}(Dirichlet(\vec{p}|\vec{\alpha}))=(\frac{\alpha_1}{\sum_{k=1}^K\alpha_k},\frac{\alpha_2}{\sum_{k=1}^K\alpha_k},...,\frac{\alpha_K}{\sum_{k=1}^K\alpha_k})
$$


## LDA主题模型

在LDA主题模型中首先需要假定一个主题数目$K$，这样所有的主题就都基于$K$个主题展开，具体如图所示，灰色圆圈表示可观察变量，白色圆圈是隐藏变量，矩形框表示重复采样。

![topic_lda](assets/topic_lda.png)

LDA假设文档主题的先验分布是Dirichlet分布，对于任一文档$d$，其主题分布$\theta_d$为：
$$
\theta_d = Dirichlet(\vec{\alpha})
$$
其中$\theta_d$是一个$K$维向量，$\alpha$是分布的超参数。

LDA假设主题中词的先验分布是Dirichlet分布，对于任一主题$k$，其词分布$\beta_k$为：
$$
\beta_k = Dirichlet(\vec{\eta})
$$
其中$\eta$是一个$V$维向量，$\eta$是分布的超参数。$V$代表词汇表里所有词的个数。

对于数据中任一一个篇文档$d$中的第$n$个词，可以从主题分布$\theta_d$得到它的主题编号$z_{dn}$的分布为：
$$
z_{dn}=multi(\theta_d)
$$
而对于该主题编号$z_{dn}$，得到词$w_{dn}$的概率分布为：
$$
w_{dn}=multi(\beta_{z_{dn}})
$$
在这个LDA模型里，有$M$个文档主题的Dirichlet分布，而对应的数据有$M$个主题编号的多项式分布，这样$(\alpha\rightarrow \theta_d \rightarrow \vec{z}_d)$就组成了Dirichlet-Multi共轭，根据上文提到的贝叶斯推断方法就可得到Dirichlet分布的文档主题后验分布。

设在第$d$个文档中，第$k$个主题的词的个数为$n_d^{(k)}$，则对应的多项式分布计数可以表示为：

$$
\vec{n}_d = (n_d^{(1)},n_d^{(2)},...,n_d^{(K)})
$$

利用Dirichlet-Multi共轭，得到$\theta_d$的后验分布为：

$$
Dirichlet(\theta_d| \vec{\alpha}+\vec{n}_d)
$$

同理对于主题与词的分布，有$K$个主题与词的Dirichlet分布，而对应的数据有$K$个主题编号的多项式分布，这样$(\eta \rightarrow \beta_k \rightarrow \vec{w}_k)$就组成了Dirichlet-Multi共轭共轭，根据上文提到的贝叶斯推断方法就可得到Dirichlet分布的文档主题后验分布。

设在第$k$个主题中，第$v$个词的个数为$n_k^{(v)}$，则对应的多项式分布的计数可以表示为：

$$
\vec{n}_k = (n_k^{(1)},n_k^{(2)},...,n_k^{(K)})
$$

利用Dirichlet-Multi共轭，得到$\beta_k$的后验分布为：

$$
Dirichlet(\beta_k| \vec{\eta}+\vec{n}_k)
$$

由于主题产生词不依赖具体某一个文档，因此文档主题分布和主题词分布是独立的。LDA模型的求解方法一般有两种：第一种是基于Gibbs采样算法求解；第二种是基于变分推断算法求解。



















