---

# Lil'Log 全文中英对照：对比表示学习

> 原文：[Contrastive Representation Learning (Lilian Weng, May 2021)](https://lilianweng.github.io/posts/2021-05-31-contrastive/)
> 说明：中文为完整译文；英文原文紧随其后以引用块给出。公式与插图保持原文。

对比表示学习的目标，是学习一个嵌入空间：其中相似的样本对在空间中彼此靠近，而不相似的样本对则彼此远离。对比学习既可用于监督设置，也可用于无监督设置。在处理无监督数据时，对比学习是自监督学习中最强大的方法之一。

> The goal of contrastive representation learning is to learn such an embedding space in which similar sample pairs stay close to each other while dissimilar ones are far apart. Contrastive learning can be applied to both supervised and unsupervised settings. When working with unsupervised data, contrastive learning is one of the most powerful approaches in self-supervised learning.

## 对比训练目标
> **EN** Contrastive Training Objectives

在对比学习损失函数的早期版本中，每次只涉及一个正样本和一个负样本。近年来训练目标的趋势，是在一个 batch 中纳入多个正样本对与负样本对。

> In early versions of loss functions for contrastive learning, only one positive and one negative sample are involved. The trend in recent training objectives is to include multiple positive and negative pairs in one batch.

### 对比损失
> **EN** Contrastive Loss

对比损失（Contrastive loss；Chopra et al. 2005）是最早用于以对比方式做深度度量学习的训练目标之一。

> Contrastive loss (Chopra et al. 2005) is one of the earliest training objectives used for deep metric learning in a contrastive fashion.

给定输入样本列表 $\{ \mathbf{x}_i \}$，每个样本有对应标签 $y_i \in \{1, \dots, L\}$，共 $L$ 个类别。我们希望学习函数 $f_\theta(.): \mathcal{X}\to\mathbb{R}^d$，将 $x_i$ 编码为嵌入向量，使得同类样本的嵌入相似，而不同类样本的嵌入差异很大。因此，对比损失接收一对输入 $(x_i, x_j)$：当它们属于同一类时最小化嵌入距离，否则最大化距离。

> Given a list of input samples $\{ \mathbf{x}_i \}$, each has a corresponding label $y_i \in \{1, \dots, L\}$ among $L$ classes. We would like to learn a function $f_\theta(.): \mathcal{X}\to\mathbb{R}^d$ that encodes $x_i$ into an embedding vector such that examples from the same class have similar embeddings and samples from different classes have very different ones. Thus, contrastive loss takes a pair of inputs $(x_i, x_j)$ and minimizes the embedding distance when they are from the same class but maximizes the distance otherwise.

$$ \mathcal{L}_\text{cont}(\mathbf{x}_i, \mathbf{x}_j, \theta) = \mathbb{1}[y_i=y_j] \| f_\theta(\mathbf{x}_i) - f_\theta(\mathbf{x}_j) \|^2_2 + \mathbb{1}[y_i\neq y_j]\max(0, \epsilon - \|f_\theta(\mathbf{x}_i) - f_\theta(\mathbf{x}_j)\|_2)^2 $$

其中 $\epsilon$ 是超参数，定义不同类样本之间距离的下界。

> where $\epsilon$ is a hyperparameter, defining the lower bound distance between samples of different classes.

### Triplet 损失
> **EN** Triplet Loss

Triplet 损失最早在 FaceNet（Schroff et al. 2015）论文中提出，用于学习同一个人在不同姿态与角度下的人脸识别。

> Triplet loss was originally proposed in the FaceNet (Schroff et al. 2015) paper and was used to learn face recognition of the same person at different poses and angles.

![triplet-loss](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/triplet-loss.png)

给定一个 anchor 与一个正样本时的 triplet loss 示意。（图片来源：Schroff et al. 2015）

> Illustration of triplet loss given one positive and one negative per anchor. (Image source: Schroff et al. 2015)

给定一个 anchor 输入 $\mathbf{x}$，我们选取一个正样本 $\mathbf{x}^+$ 与一个负样本 $\mathbf{x}^-$，即 $\mathbf{x}^+$ 与 $\mathbf{x}$ 属于同一类，而 $\mathbf{x}^-$ 从另一类中采样。Triplet 损失同时学习：最小化 anchor $\mathbf{x}$ 与正样本 $\mathbf{x}^+$ 之间的距离，并最大化 anchor $\mathbf{x}$ 与负样本 $\mathbf{x}^-$ 之间的距离，其形式为：

> Given one anchor input $\mathbf{x}$, we select one positive sample $\mathbf{x}^+$ and one negative $\mathbf{x}^-$, meaning that $\mathbf{x}^+$ and $\mathbf{x}$ belong to the same class and $\mathbf{x}^-$ is sampled from another different class. Triplet loss learns to minimize the distance between the anchor $\mathbf{x}$ and positive $\mathbf{x}^+$ and maximize the distance between the anchor $\mathbf{x}$ and negative $\mathbf{x}^-$ at the same time with the following equation:

$$ \mathcal{L}_\text{triplet}(\mathbf{x}, \mathbf{x}^+, \mathbf{x}^-) = \sum_{\mathbf{x} \in \mathcal{X}} \max\big( 0, \|f(\mathbf{x}) - f(\mathbf{x}^+)\|^2_2 - \|f(\mathbf{x}) - f(\mathbf{x}^-)\|^2_2 + \epsilon \big) $$

其中 margin 参数 $\epsilon$ 配置为相似对与不相似对之间距离的最小间隔。

> where the margin parameter $\epsilon$ is configured as the minimum offset between distances of similar vs dissimilar pairs.

选择具有挑战性的 $\mathbf{x}^-$ 对真正提升模型至关重要。

> It is crucial to select challenging $\mathbf{x}^-$ to truly improve the model.

### Lifted Structured 损失
> **EN** Lifted Structured Loss

Lifted Structured Loss（Song et al. 2015）在一个训练 batch 内利用所有样本对之间的边，以获得更好的计算效率。

> Lifted Structured Loss (Song et al. 2015) utilizes all the pairwise edges within one training batch for better computational efficiency.

![lifted-structured-loss](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/lifted-structured-loss.png)

对比损失、triplet 损失与 lifted structured 损失的对比示意。红色与蓝色边分别连接相似与不相似的样本对。（图片来源：Song et al. 2015）

> Illustration compares contrastive loss, triplet loss and lifted structured loss. Red and blue edges connect similar and dissimilar sample pairs respectively. (Image source: Song et al. 2015)

令 $D_{ij} = | f(\mathbf{x}_i) - f(\mathbf{x}_j) |_2$，结构化损失函数定义为

> Let $D_{ij} = | f(\mathbf{x}_i) - f(\mathbf{x}_j) |_2$, a structured loss function is defined as

$$ \begin{aligned} \mathcal{L}_\text{struct} &= \frac{1}{2\vert \mathcal{P} \vert} \sum_{(i,j) \in \mathcal{P}} \max(0, \mathcal{L}_\text{struct}^{(ij)})^2 \\ \text{where } \mathcal{L}_\text{struct}^{(ij)} &= D_{ij} + \color{red}{\max \big( \max_{(i,k)\in \mathcal{N}} \epsilon - D_{ik}, \max_{(j,l)\in \mathcal{N}} \epsilon - D_{jl} \big)} \end{aligned} $$

其中 $\mathcal{P}$ 为正样本对集合，$\mathcal{N}$ 为负样本对集合。注意，稠密的成对平方距离矩阵可以按每个训练 batch 轻松计算。

> where $\mathcal{P}$ contains the set of positive pairs and $\mathcal{N}$ is the set of negative pairs. Note that the dense pairwise squared distance matrix can be easily computed per training batch.

$\mathcal{L}_\text{struct}^{(ij)}$ 中的红色部分用于难负例挖掘。然而它并不光滑，实践中可能导致收敛到较差的局部最优。因此将其松弛为：

> The red part in $\mathcal{L}_\text{struct}^{(ij)}$ is used for mining hard negatives. However, it is not smooth and may cause the convergence to a bad local optimum in practice. Thus, it is relaxed to be:

$$ \mathcal{L}_\text{struct}^{(ij)} = D_{ij} + \log \Big( \sum_{(i,k)\in\mathcal{N}} \exp(\epsilon - D_{ik}) + \sum_{(j,l)\in\mathcal{N}} \exp(\epsilon - D_{jl}) \Big) $$

在该论文中，他们还提出：给定少量随机正样本对，通过主动纳入难负样本来提升每个 batch 中负样本的质量。

> In the paper, they also proposed to enhance the quality of negative samples in each batch by actively incorporating difficult negative samples given a few random positive pairs.

### N-pair 损失
> **EN** N-pair Loss

Multi-Class N-pair loss（Sohn 2016）将 triplet 损失推广为与多个负样本进行比较。

> Multi-Class N-pair loss (Sohn 2016) generalizes triplet loss to include comparison with multiple negative samples.

给定一个 $(N + 1)$-元组训练样本 $\{ \mathbf{x}, \mathbf{x}^+, \mathbf{x}^-_1, \dots, \mathbf{x}^-_{N-1} \}$，其中包含一个正样本与 $N-1$ 个负样本，N-pair 损失定义为：

> Given a $(N + 1)$-tuplet of training samples, $\{ \mathbf{x}, \mathbf{x}^+, \mathbf{x}^-_1, \dots, \mathbf{x}^-_{N-1} \}$, including one positive and $N-1$ negative ones, N-pair loss is defined as:

$$ \begin{aligned} \mathcal{L}_\text{N-pair}(\mathbf{x}, \mathbf{x}^+, \{\mathbf{x}^-_i\}^{N-1}_{i=1}) &= \log\big(1 + \sum_{i=1}^{N-1} \exp(f(\mathbf{x})^\top f(\mathbf{x}^-_i) - f(\mathbf{x})^\top f(\mathbf{x}^+))\big) \\ &= -\log\frac{\exp(f(\mathbf{x})^\top f(\mathbf{x}^+))}{\exp(f(\mathbf{x})^\top f(\mathbf{x}^+)) + \sum_{i=1}^{N-1} \exp(f(\mathbf{x})^\top f(\mathbf{x}^-_i))} \end{aligned} $$

若每个类只采样一个负样本，则它等价于多类分类的 softmax 损失。

> If we only sample one negative sample per class, it is equivalent to the softmax loss for multi-class classification.

### NCE
> **EN** NCE

Noise Contrastive Estimation（NCE）是一种估计统计模型参数的方法，由 Gutmann & Hyvarinen 于 2010 年提出。其思路是通过逻辑回归区分目标数据与噪声。可进一步阅读 NCE 如何用于学习词嵌入。

> Noise Contrastive Estimation, short for NCE, is a method for estimating parameters of a statistical model, proposed by Gutmann & Hyvarinen in 2010. The idea is to run logistic regression to tell apart the target data from noise. Read more on how NCE is used for learning word embedding here.

令 $\mathbf{x}$ 为目标样本 $\sim P(\mathbf{x} \vert C=1; \theta) = p_\theta(\mathbf{x})$，$\tilde{\mathbf{x}}$ 为噪声样本 $\sim P(\tilde{\mathbf{x}} \vert C=0) = q(\tilde{\mathbf{x}})$。注意，逻辑回归建模的是 logit（即 log-odds）；此处我们希望建模来自目标数据分布而非噪声分布的样本 $u$ 的 logit：

> Let $\mathbf{x}$ be the target sample $\sim P(\mathbf{x} \vert C=1; \theta) = p_\theta(\mathbf{x})$ and $\tilde{\mathbf{x}}$ be the noise sample $\sim P(\tilde{\mathbf{x}} \vert C=0) = q(\tilde{\mathbf{x}})$. Note that the logistic regression models the logit (i.e. log-odds) and in this case we would like to model the logit of a sample $u$ from the target data distribution instead of the noise distribution:

$$ \ell_\theta(\mathbf{u}) = \log \frac{p_\theta(\mathbf{u})}{q(\mathbf{u})} = \log p_\theta(\mathbf{u}) - \log q(\mathbf{u}) $$

经 sigmoid $\sigma(.)$ 将 logit 转为概率后，可应用交叉熵损失：

> After converting logits into probabilities with sigmoid $\sigma(.)$, we can apply cross entropy loss:

$$ \begin{aligned} \mathcal{L}_\text{NCE} &= - \frac{1}{N} \sum_{i=1}^N \big[ \log \sigma (\ell_\theta(\mathbf{x}_i)) + \log (1 - \sigma (\ell_\theta(\tilde{\mathbf{x}}_i))) \big] \\ \text{ where }\sigma(\ell) &= \frac{1}{1 + \exp(-\ell)} = \frac{p_\theta}{p_\theta + q} \end{aligned} $$

此处列出的是仅含一个正样本与一个噪声样本的 NCE 损失原始形式。在许多后续工作中，纳入多个负样本的对比损失也被广泛称为 NCE。

> Here I listed the original form of NCE loss which works with only one positive and one noise sample. In many follow-up works, contrastive loss incorporating multiple negative samples is also broadly referred to as NCE.

### InfoNCE
> **EN** InfoNCE

CPC（Contrastive Predictive Coding；van den Oord, et al. 2018）中的 InfoNCE 损失受 NCE 启发，使用 categorical 交叉熵损失，在一组无关噪声样本中识别正样本。

> The InfoNCE loss in CPC (Contrastive Predictive Coding; van den Oord, et al. 2018), inspired by NCE, uses categorical cross-entropy loss to identify the positive sample amongst a set of unrelated noise samples.

给定上下文向量 $\mathbf{c}$，正样本应从条件分布 $p(\mathbf{x} \vert \mathbf{c})$ 中抽取，而 $N-1$ 个负样本从 proposal 分布 $p(\mathbf{x})$ 中抽取，且与上下文 $\mathbf{c}$ 独立。为简洁起见，将所有样本记为 $X=\{ \mathbf{x}_i \}^N_{i=1}$，其中仅有一个 $\mathbf{x}_\texttt{pos}$ 为正样本。正确识别正样本的概率为：

> Given a context vector $\mathbf{c}$, the positive sample should be drawn from the conditional distribution $p(\mathbf{x} \vert \mathbf{c})$, while $N-1$ negative samples are drawn from the proposal distribution $p(\mathbf{x})$, independent from the context $\mathbf{c}$. For brevity, let us label all the samples as $X=\{ \mathbf{x}_i \}^N_{i=1}$ among which only one of them $\mathbf{x}_\texttt{pos}$ is a positive sample. The probability of we detecting the positive sample correctly is:

$$ p(C=\texttt{pos} \vert X, \mathbf{c}) = \frac{p(x_\texttt{pos} \vert \mathbf{c}) \prod_{i=1,\dots,N; i \neq \texttt{pos}} p(\mathbf{x}_i)}{\sum_{j=1}^N \big[ p(\mathbf{x}_j \vert \mathbf{c}) \prod_{i=1,\dots,N; i \neq j} p(\mathbf{x}_i) \big]} = \frac{ \frac{p(\mathbf{x}_\texttt{pos}\vert c)}{p(\mathbf{x}_\texttt{pos})} }{ \sum_{j=1}^N \frac{p(\mathbf{x}_j\vert \mathbf{c})}{p(\mathbf{x}_j)} } = \frac{f(\mathbf{x}_\texttt{pos}, \mathbf{c})}{ \sum_{j=1}^N f(\mathbf{x}_j, \mathbf{c}) } $$

其中评分函数为 $f(\mathbf{x}, \mathbf{c}) \propto \frac{p(\mathbf{x}\vert\mathbf{c})}{p(\mathbf{x})}$。

> where the scoring function is $f(\mathbf{x}, \mathbf{c}) \propto \frac{p(\mathbf{x}\vert\mathbf{c})}{p(\mathbf{x})}$.

InfoNCE 损失优化正确分类正样本的负对数概率：

> The InfoNCE loss optimizes the negative log probability of classifying the positive sample correctly:

$$ \mathcal{L}_\text{InfoNCE} = - \mathbb{E} \Big[\log \frac{f(\mathbf{x}, \mathbf{c})}{\sum_{\mathbf{x}' \in X} f(\mathbf{x}', \mathbf{c})} \Big] $$

$f(x, c)$ 估计密度比 $\frac{p(x\vert c)}{p(x)}$ 这一事实，与互信息优化存在联系。为最大化输入 $x$ 与上下文向量 $c$ 之间的互信息，有：

> The fact that $f(x, c)$ estimates the density ratio $\frac{p(x\vert c)}{p(x)}$ has a connection with mutual information optimization. To maximize the the mutual information between input $x$ and context vector $c$, we have:

$$ I(\mathbf{x}; \mathbf{c}) = \sum_{\mathbf{x}, \mathbf{c}} p(\mathbf{x}, \mathbf{c}) \log\frac{p(\mathbf{x}, \mathbf{c})}{p(\mathbf{x})p(\mathbf{c})} = \sum_{\mathbf{x}, \mathbf{c}} p(\mathbf{x}, \mathbf{c})\log\color{blue}{\frac{p(\mathbf{x}|\mathbf{c})}{p(\mathbf{x})}} $$

其中蓝色对数项由 $f$ 估计。

> where the logarithmic term in blue is estimated by $f$.

对于序列预测任务，直接建模未来观测 $p_k(\mathbf{x}_{t+k} \vert \mathbf{c}_t)$ 可能相当昂贵；CPC 转而建模密度函数，以保留 $\mathbf{x}_{t+k}$ 与 $\mathbf{c}_t$ 之间的互信息：

> For sequence prediction tasks, rather than modeling the future observations $p_k(\mathbf{x}_{t+k} \vert \mathbf{c}_t)$ directly (which could be fairly expensive), CPC models a density function to preserve the mutual information between $\mathbf{x}_{t+k}$ and $\mathbf{c}_t$:

$$ f_k(\mathbf{x}_{t+k}, \mathbf{c}_t) = \exp(\mathbf{z}_{t+k}^\top \mathbf{W}_k \mathbf{c}_t) \propto \frac{p(\mathbf{x}_{t+k}\vert\mathbf{c}_t)}{p(\mathbf{x}_{t+k})} $$

其中 $\mathbf{z}_{t+k}$ 为编码后的输入，$\mathbf{W}_k$ 为可训练权重矩阵。

> where $\mathbf{z}_{t+k}$ is the encoded input and $\mathbf{W}_k$ is a trainable weight matrix.

### Soft-Nearest Neighbors 损失
> **EN** Soft-Nearest Neighbors Loss

Soft-Nearest Neighbors Loss（Salakhutdinov & Hinton 2007, Frosst et al. 2019）将其扩展为包含多个正样本。

> Soft-Nearest Neighbors Loss (Salakhutdinov & Hinton 2007, Frosst et al. 2019) extends it to include multiple positive samples.

给定一个 batch 的样本 $\{\mathbf{x}_i, y_i)\}^B_{i=1}$，其中 $y_i$ 为 $\mathbf{x}_i$ 的类标签，以及衡量两个输入相似度的函数 $f(.,.)$，温度 $\tau$ 下的 soft nearest neighbor 损失定义为：

> Given a batch of samples, $\{\mathbf{x}_i, y_i)\}^B_{i=1}$ where $y_i$ is the class label of $\mathbf{x}_i$ and a function $f(.,.)$ for measuring similarity between two inputs, the soft nearest neighbor loss at temperature $\tau$ is defined as:

$$ \mathcal{L}_\text{snn} = -\frac{1}{B}\sum_{i=1}^B \log \frac{\sum_{i\neq j, y_i = y_j, j=1,\dots,B} \exp(- f(\mathbf{x}_i, \mathbf{x}_j) / \tau)}{\sum_{i\neq k, k=1,\dots,B} \exp(- f(\mathbf{x}_i, \mathbf{x}_k) /\tau)} $$

温度 $\tau$ 用于调节表示空间中特征的集中程度。例如，当温度较低时，损失由小距离主导，彼此相距较远的表示贡献很小并变得无关紧要。

> The temperature $\tau$ is used for tuning how concentrated the features are in the representation space. For example, when at low temperature, the loss is dominated by the small distances and widely separated representations cannot contribute much and become irrelevant.

### 常见设定
> **EN** Common Setup

我们可以放宽 soft nearest-neighbor 损失中「类」与「标签」的定义：例如通过对原始样本做数据增强以构造噪声版本，从而从无监督数据中构造正样本对与负样本对。

> We can loosen the definition of "classes" and "labels" in soft nearest-neighbor loss to create positive and negative sample pairs out of unsupervised data by, for example, applying data augmentation to create noise versions of original samples.

大多数近期研究遵循如下对比学习目标的定义，以纳入多个正样本与负样本。根据（Wang & Isola 2020）中的设定，令 $p_\texttt{data}(.)$ 为 $\mathbb{R}^n$ 上的数据分布，$p_\texttt{pos}(., .)$ 为 $\mathbb{R}^{n \times n}$ 上的正样本对分布。这两个分布应满足：

> Most recent studies follow the following definition of contrastive learning objective to incorporate multiple positive and negative samples. According to the setup in (Wang & Isola 2020), let $p_\texttt{data}(.)$ be the data distribution over $\mathbb{R}^n$ and $p_\texttt{pos}(., .)$ be the distribution of positive pairs over $\mathbb{R}^{n \times n}$. These two distributions should satisfy:

- 对称性：$\forall \mathbf{x}, \mathbf{x}^+, p_\texttt{pos}(\mathbf{x}, \mathbf{x}^+) = p_\texttt{pos}(\mathbf{x}^+, \mathbf{x})$
- 边缘匹配：$\forall \mathbf{x}, \int p_\texttt{pos}(\mathbf{x}, \mathbf{x}^+) d\mathbf{x}^+ = p_\texttt{data}(\mathbf{x})$

> - Symmetry: $\forall \mathbf{x}, \mathbf{x}^+, p_\texttt{pos}(\mathbf{x}, \mathbf{x}^+) = p_\texttt{pos}(\mathbf{x}^+, \mathbf{x})$
> - Matching marginal: $\forall \mathbf{x}, \int p_\texttt{pos}(\mathbf{x}, \mathbf{x}^+) d\mathbf{x}^+ = p_\texttt{data}(\mathbf{x})$

为学习编码器 $f(\mathbf{x})$ 以得到 L2 归一化的特征向量，对比学习目标为：

> To learn an encoder $f(\mathbf{x})$ to learn a L2-normalized feature vector, the contrastive learning objective is:

$$ \begin{aligned} \mathcal{L}_\text{contrastive} &= \mathbb{E}_{(\mathbf{x},\mathbf{x}^+)\sim p_\texttt{pos}, \{\mathbf{x}^-_i\}^M_{i=1} \overset{\text{i.i.d}}{\sim} p_\texttt{data} } \Big[ -\log\frac{\exp(f(\mathbf{x})^\top f(\mathbf{x}^+) / \tau)}{ \exp(f(\mathbf{x})^\top f(\mathbf{x}^+) / \tau) + \sum_{i=1}^M \exp(f(\mathbf{x})^\top f(\mathbf{x}_i^-) / \tau)} \Big] & \\ &\approx \mathbb{E}_{(\mathbf{x},\mathbf{x}^+)\sim p_\texttt{pos}, \{\mathbf{x}^-_i\}^M_{i=1} \overset{\text{i.i.d}}{\sim} p_\texttt{data} }\Big[ - f(\mathbf{x})^\top f(\mathbf{x}^+) / \tau + \log\big(\sum_{i=1}^M \exp(f(\mathbf{x})^\top f(\mathbf{x}_i^-) / \tau)\big) \Big] & \scriptstyle{\text{; Assuming infinite negatives}} \\ &= -\frac{1}{\tau}\mathbb{E}_{(\mathbf{x},\mathbf{x}^+)\sim p_\texttt{pos}}f(\mathbf{x})^\top f(\mathbf{x}^+) + \mathbb{E}_{ \mathbf{x} \sim p_\texttt{data}} \Big[ \log \mathbb{E}_{\mathbf{x}^- \sim p_\texttt{data}} \big[ \sum_{i=1}^M \exp(f(\mathbf{x})^\top f(\mathbf{x}_i^-) / \tau)\big] \Big] & \end{aligned} $$
