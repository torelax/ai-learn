## 语言：句子嵌入

> **EN** Language: Sentence Embedding

本节聚焦于如何学习句子嵌入。

> In this section, we focus on how to learn sentence embedding.

### 文本增强

> **EN** Text Augmentation

视觉应用中的大多数对比学习方法都依赖于为每张图像创建增强版本。然而，构造不改变句子语义的文本增强更具挑战性。本节我们将探讨三种增强文本序列的方法，包括词法编辑、回译以及应用截断或 dropout。

> Most contrastive methods in vision applications depend on creating an augmented version of each image. However, it is more challenging to construct text augmentation which does not alter the semantics of a sentence. In this section we look into three approaches for augmenting text sequences, including lexical edits, back-translation and applying cutoff or dropout.

#### 词法编辑

> **EN** Lexical Edits

EDA（Easy Data Augmentation；Wei & Zou 2019）定义了一组简单但强大的文本增强操作。给定一个句子，EDA 随机选择并应用以下四种简单操作之一：

> EDA (Easy Data Augmentation; Wei & Zou 2019) defines a set of simple but powerful operations for text augmentation. Given a sentence, EDA randomly chooses and applies one of four simple operations:

1. 同义词替换（SR）：将 $n$ 个随机非停用词替换为其同义词。
2. 随机插入（RI）：在句子的随机位置插入一个从随机选定的非停用词的同义词。
3. 随机交换（RS）：随机交换两个词，并重复 $n$ 次。
4. 随机删除（RD）：以概率 $p$ 随机删除句子中的每个词。

> 1. Synonym replacement (SR): Replace $n$ random non-stop words with their synonyms.
> 2. Random insertion (RI): Place a random synonym of a randomly selected non-stop word in the sentence at a random position.
> 3. Random swap (RS): Randomly swap two words and repeat $n$ times.
> 4. Random deletion (RD): Randomly delete each word in the sentence with probability $p$.

其中 $p=\alpha$ 且 $n=\alpha \times \text{sentence_length}$，其直觉是：更长的句子在保持原始标签的同时可以吸收更多噪声。超参数 $\alpha$ 大致表示一个句子中可能被一次增强改变的词的百分比。

> where $p=\alpha$ and $n=\alpha \times \text{sentence_length}$, with the intuition that longer sentences can absorb more noise while maintaining the original label. The hyperparameter $\alpha$ roughly indicates the percent of words in one sentence that may be changed by one augmentation.

EDA 在多个分类基准数据集上被证明能提升分类准确率，相比未使用 EDA 的基线。在较小的训练集上，性能提升更为显著。EDA 中的四种操作都有助于提高分类准确率，但在不同的 $\alpha$ 值下达到最优。

> EDA is shown to improve the classification accuracy on several classification benchmark datasets compared to baseline without EDA. The performance lift is more significant on a smaller training set. All the four operations in EDA help improve the classification accuracy, but get to optimal at different $\alpha$'s.

![EDA-exp1](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/EDA-exp1.png)

EDA 在多个分类基准上带来性能提升。

> EDA leads to performance improvement on several classification benchmarks. (Image source: Wei & Zou 2019)

在 Contextual Augmentation（Sosuke Kobayashi, 2018）中，位置 $i$ 处词 $w_i$ 的新替换可以从给定概率分布 $p(.\mid S\setminus\{w_i\})$ 中平滑采样，该分布由类似 BERT 的双向语言模型预测。

> In Contextual Augmentation (Sosuke Kobayashi, 2018), new substitutes for word $w_i$ at position $i$ can be smoothly sampled from a given probability distribution, $p(.\mid S\setminus\{w_i\})$, which is predicted by a bidirectional LM like BERT.

#### 回译

> **EN** Back-translation

CERT（Contrastive self-supervised Encoder Representations from Transformers；Fang et al. (2020)；code）通过回译生成增强句子。可以使用针对不同语言的不同翻译模型来创建不同版本的增强。一旦我们有了文本样本的噪声版本，就可以使用上文介绍的许多对比学习框架（例如 MoCo）来学习句子嵌入。

> CERT (Contrastive self-supervised Encoder Representations from Transformers; Fang et al. (2020); code) generates augmented sentences via back-translation. Various translation models for different languages can be employed for creating different versions of augmentations. Once we have a noise version of text samples, many contrastive learning frameworks introduced above, such as MoCo, can be used to learn sentence embedding.

#### Dropout 与截断

> **EN** Dropout and Cutoff

Shen 等人（2020）提出将截断（Cutoff）应用于文本增强，灵感来自 cross-view training。他们提出了三种截断增强策略：

> Shen et al. (2020) proposed to apply Cutoff to text augmentation, inspired by cross-view training. They proposed three cutoff augmentation strategies:

1. Token cutoff 移除若干选定 token 的信息。为确保没有数据泄漏，输入、位置编码及其他相关嵌入矩阵中对应的 token 都应置零。
2. Feature cutoff 移除若干特征列。
3. Span cutoff 移除一段连续的文本块。

> 1. Token cutoff removes the information of a few selected tokens. To make sure there is no data leakage, corresponding tokens in the input, positional and other relevant embedding matrices should all be zeroed out.,
> 2. Feature cutoff removes a few feature columns.
> 3. Span cutoff removes a continuous chunk of texts.

![text-cutoff](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/text-cutoff.png)

token、feature 与 span 截断增强策略示意图。

> Schematic illustration of token, feature and span cutoff augmentation strategies. (Image source: Shen et al. 2020)

可以为一个样本创建多个增强版本。训练时，Shen 等人（2020）应用额外的 KL 散度项来衡量不同增强样本预测之间的一致性。

> Multiple augmented versions of one sample can be created. When training, Shen et al. (2020) applied an additional KL-divergence term to measure the consensus between predictions from different augmented samples.

SimCSE（Gao et al. 2021；code）从无监督数据中学习，仅通过 dropout 噪声预测句子自身。换言之，他们将 dropout 作为文本序列的数据增强。一个样本简单地以不同的 dropout mask 两次输入编码器，这两个版本构成正样本对，批次内其他样本则视为负样本对。这与截断增强颇为相似，但 dropout 更灵活，对被 mask 掉的内容缺乏明确定义的语义含义。

> SimCSE (Gao et al. 2021; code) learns from unsupervised data by predicting a sentence from itself with only dropout noise. In other words, they treat dropout as data augmentation for text sequences. A sample is simply fed into the encoder twice with different dropout masks and these two versions are the positive pair where the other in-batch samples are considered as negative pairs. It feels quite similar to the cutoff augmentation, but dropout is more flexible with less well-defined semantic meaning of what content can be masked off.

![SimCSE](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/SimCSE.png)

SimCSE 通过应用不同的 dropout mask 创建增强样本。监督版本利用自然语言推断（NLI）数据集，给定句子对预测正样本（蕴含）或负样本（矛盾）。

> SimCSE creates augmented samples by applying different dropout masks. The supervised version leverages NLI datasets to predict positive (entailment) or negative (contradiction) given a pair of sentences. (Image source: Gao et al. 2021)

他们在 7 个 STS（Semantic Text Similarity，语义文本相似度）数据集上进行了实验，计算句子嵌入之间的余弦相似度。他们还尝试了可选的 MLM 辅助目标损失，以帮助避免灾难性遗忘 token 级知识。该辅助损失被发现有助于提升迁移任务上的性能，但在主要 STS 任务上有一致的下降。

> They ran experiments on 7 STS (Semantic Text Similarity) datasets and computed cosine similarity between sentence embeddings. They also tried out an optional MLM auxiliary objective loss to help avoid catastrophic forgetting of token-level knowledge. This aux loss was found to help improve performance on transfer tasks, but a consistent drop on the main STS tasks.

![SimCSE-STS-exp](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/SimCSE-STS-exp.png)

SimCSE 在一组 STS 基准上的实验结果。

> Experiment numbers on a collection of STS benchmarks with SimCES. (Image source: Gao et al. 2021)

### 来自 NLI 的监督

> **EN** Supervision from NLI

未经任何微调时，预训练 BERT 的句子嵌入在语义相似度任务上表现不佳。我们不能直接使用原始嵌入，而需要通过进一步微调来改进嵌入。

> The pre-trained BERT sentence embedding without any fine-tuning has been found to have poor performance for semantic similarity tasks. Instead of using the raw embeddings directly, we need to refine the embedding with further fine-tuning.

自然语言推断（NLI）任务是提供监督信号以学习句子嵌入的主要数据来源；例如 SNLI、MNLI 和 QQP。

> Natural Language Inference (NLI) tasks are the main data sources to provide supervised signals for learning sentence embedding; such as SNLI, MNLI, and QQP.

#### Sentence-BERT

> **EN** Sentence-BERT

SBERT（Sentence-BERT）（Reimers & Gurevych, 2019）依赖孪生网络与三元组网络架构来学习句子嵌入，使得句子相似度可以通过嵌入对之间的余弦相似度来估计。注意，学习 SBERT 依赖监督数据，因为它在多个 NLI 数据集上进行了微调。

> SBERT (Sentence-BERT) (Reimers & Gurevych, 2019) relies on siamese and triplet network architectures to learn sentence embeddings such that the sentence similarity can be estimated by cosine similarity between pairs of embeddings. Note that learning SBERT depends on supervised data, as it is fine-tuned on several NLI datasets.

他们在 BERT 模型之上实验了若干不同的预测头：

> They experimented with a few different prediction heads on top of BERT model:

- Softmax 分类目标：孪生网络的分类头建立在两个嵌入 $f(\mathbf{x}), f(\mathbf{x}')$ 与 $\vert f(\mathbf{x}) - f(\mathbf{x}') \vert$ 的拼接之上。预测输出为 $\hat{y}=\text{softmax}(\mathbf{W}_t [f(\mathbf{x}); f(\mathbf{x}'); \vert f(\mathbf{x}) - f(\mathbf{x}') \vert])$。他们表明最重要的组件是逐元素差 $\vert f(\mathbf{x}) - f(\mathbf{x}') \vert$。
- 回归目标：这是对 $\cos(f(\mathbf{x}), f(\mathbf{x}'))$ 的回归损失，其中池化策略影响较大。实验中他们观察到 `max` 远差于 `mean` 和 `CLS`-token。
- 三元组目标：$\max(0, |f(\mathbf{x}) - f(\mathbf{x}^+)|- |f(\mathbf{x}) - f(\mathbf{x}^-)| + \epsilon)$，其中 $\mathbf{x}, \mathbf{x}^+, \mathbf{x}^-$ 分别为锚点、正样本与负样本句子的嵌入。

> - Softmax classification objective: The classification head of the siamese network is built on the concatenation of two embeddings $f(\mathbf{x}), f(\mathbf{x}')$ and $\vert f(\mathbf{x}) - f(\mathbf{x}') \vert$. The predicted output is $\hat{y}=\text{softmax}(\mathbf{W}_t [f(\mathbf{x}); f(\mathbf{x}'); \vert f(\mathbf{x}) - f(\mathbf{x}') \vert])$. They showed that the most important component is the element-wise difference $\vert f(\mathbf{x}) - f(\mathbf{x}') \vert$.
> - Regression objective: This is the regression loss on $\cos(f(\mathbf{x}), f(\mathbf{x}'))$, in which the pooling strategy has a big impact. In the experiments, they observed that`max` performs much worse than`mean` and`CLS`-token.
> - Triplet objective: $\max(0, |f(\mathbf{x}) - f(\mathbf{x}^+)|- |f(\mathbf{x}) - f(\mathbf{x}^-)| + \epsilon)$, where $\mathbf{x}, \mathbf{x}^+, \mathbf{x}^-$ are embeddings of the anchor, positive and negative sentences.

实验中哪种目标函数效果最好取决于数据集，因此没有普适的最优选择。

> In the experiments, which objective function works the best depends on the datasets, so there is no universal winner.

![SBERT](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/SBERT.png)

Sentence-BERT 训练框架示意图，含 softmax 分类头与回归头。

> Illustration of Sentence-BERT training framework with softmax classification head and regression head. (Image source: Reimers & Gurevych, 2019)

SentEval 库（Conneau and Kiela, 2018）常用于评估所学句子嵌入的质量。SBERT 在当时（2019 年 8 月）的 7 项任务中有 5 项优于其他基线。

> The SentEval library (Conneau and Kiela, 2018) is commonly used for evaluating the quality of learned sentence embedding. SBERT outperformed other baselines at that time (Aug 2019) on 5 out of 7 tasks.

![SBERT-SentEval](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/SBERT-SentEval.png)

Sentence-BERT 在 SentEval 基准上的性能。

> The performance of Sentence-BERT on the SentEval benchmark. (Image source: Reimers & Gurevych, 2019)

#### BERT-flow

> **EN** BERT-flow

若嵌入在各维度上均匀分布，则称嵌入表示空间为各向同性（isotropic）；否则为各向异性（anisotropic）。Li 等人（2020）表明，预训练 BERT 学习到的是非平滑的各向异性句子嵌入语义空间，因此在未经微调时文本相似度任务表现较差。经验上，他们观察到 BERT 句子嵌入存在两个问题：词频使嵌入空间产生偏置——高频词靠近原点，低频词远离原点；低频词稀疏散布——低频词嵌入往往离其 $k$-NN 邻居更远，而高频词嵌入更密集地聚集。

> The embedding representation space is deemed isotropic if embeddings are uniformly distributed on each dimension; otherwise, it is anisotropic. Li et al, (2020) showed that a pre-trained BERT learns a non-smooth anisotropic semantic space of sentence embeddings and thus leads to poor performance for text similarity tasks without fine-tuning. Empirically, they observed two issues with BERT sentence embedding: Word frequency biases the embedding space. High-frequency words are close to the origin, but low-frequency ones are far away from the origin. Low-frequency words scatter sparsely. The embeddings of low-frequency words tend to be farther to their $k$-NN neighbors, while the embeddings of high-frequency words concentrate more densely.

BERT-flow（Li et al, 2020；code）通过归一化流（normalizing flows）将嵌入变换为平滑且各向同性的高斯分布。

> BERT-flow (Li et al, 2020; code) was proposed to transform the embedding to a smooth and isotropic Gaussian distribution via normalizing flows.

![BERT-flow](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/BERT-flow.png)

BERT-flow 中对原始句子嵌入空间进行基于流的校准示意图。

> Illustration of the flow-based calibration over the original sentence embedding space in BERT-flow. (Image source: Li et al, 2020)

令 $\mathcal{U}$ 为观测到的 BERT 句子嵌入空间，$\mathcal{Z}$ 为期望的潜空间（标准高斯）。于是 $p_\mathcal{Z}$ 为高斯密度函数，$f_\phi: \mathcal{Z}\to\mathcal{U}$ 为可逆变换：

> Let $\mathcal{U}$ be the observed BERT sentence embedding space and $\mathcal{Z}$ be the desired latent space which is a standard Gaussian. Thus, $p_\mathcal{Z}$ is a Gaussian density function and $f_\phi: \mathcal{Z}\to\mathcal{U}$ is an invertible transformation:

$$ \mathbf{z}\sim p_\mathcal{Z}(\mathbf{z}) \quad \mathbf{u}=f_\phi(\mathbf{z}) \quad \mathbf{z}=f^{-1}_\phi(\mathbf{u}) $$

基于流的生成模型通过最大化 $\mathcal{U}$ 的边缘似然来学习可逆映射函数：

> A flow-based generative model learns the invertible mapping function by maximizing the likelihood of $\mathcal{U}$'s marginal:

$$ \max_\phi\mathbb{E}_{\mathbf{u}=\text{BERT}(s), s\sim\mathcal{D}} \Big[ \log p_\mathcal{Z}(f^{-1}_\phi(\mathbf{u})) + \log\big\vert\det\frac{\partial f^{-1}_\phi(\mathbf{u})}{\partial\mathbf{u}}\big\vert \Big] $$

其中 $s$ 为从文本语料 $\mathcal{D}$ 中采样的句子。仅优化流参数 $\phi$，预训练 BERT 的参数保持不变。

> where $s$ is a sentence sampled from the text corpus $\mathcal{D}$. Only the flow parameters $\phi$ are optimized while parameters in the pretrained BERT stay unchanged.

BERT-flow 被证明在大多数 STS 任务上提升性能，无论是否使用 NLI 数据集的监督。由于学习归一化流进行校准不需要标签，它可以利用包括验证集与测试集在内的整个数据集。

> BERT-flow was shown to improve the performance on most STS tasks either with or without supervision from NLI datasets. Because learning normalizing flows for calibration does not require labels, it can utilize the entire dataset including validation and test sets.

#### 白化操作

> **EN** Whitening Operation

Su 等人（2021）应用白化（whitening）操作以改善所学表示的各向同性，并降低句子嵌入的维度。

> Su et al. (2021) applied whitening operation to improve the isotropy of the learned representation and also to reduce the dimensionality of sentence embedding.

他们将句子向量的均值变换为 0，协方差矩阵变换为单位矩阵。给定样本集 $\{\mathbf{x}_i\}_{i=1}^N$，令 $\tilde{\mathbf{x}}_i$ 与 $\tilde{\Sigma}$ 为变换后的样本及对应协方差矩阵：

> They transform the mean value of the sentence vectors to 0 and the covariance matrix to the identity matrix. Given a set of samples $\{\mathbf{x}_i\}_{i=1}^N$, let $\tilde{\mathbf{x}}_i$ and $\tilde{\Sigma}$ be the transformed samples and corresponding covariance matrix:

$$ \begin{aligned} \mu &= \frac{1}{N}\sum_{i=1}^N \mathbf{x}_i \quad \Sigma = \frac{1}{N}\sum_{i=1}^N (\mathbf{x}_i - \mu)^\top (\mathbf{x}_i - \mu) \\ \tilde{\mathbf{x}}_i &= (\mathbf{x}_i - \mu)W \quad \tilde{\Sigma} = W^\top\Sigma W = I \text{ thus } \Sigma = (W^{-1})^\top W^{-1} \end{aligned} $$

若对 $\Sigma$ 做 SVD 分解 $\Sigma = U\Lambda U^\top$，则有 $W^{-1}=\sqrt{\Lambda} U^\top$ 且 $W=U\sqrt{\Lambda^{-1}}$。在 SVD 中，$U$ 为正交矩阵，列向量为特征向量；$\Lambda$ 为对角矩阵，对角元素为排序后的正特征值。

> If we get SVD decomposition of $\Sigma = U\Lambda U^\top$, we will have $W^{-1}=\sqrt{\Lambda} U^\top$ and $W=U\sqrt{\Lambda^{-1}}$. Note that within SVD, $U$ is an orthogonal matrix with column vectors as eigenvectors and $\Lambda$ is a diagonal matrix with all positive elements as sorted eigenvalues.

可通过只取 $W$ 的前 $k$ 列来应用降维策略，称为 `Whitening`-$k$。

> A dimensionality reduction strategy can be applied by only taking the first $k$ columns of $W$, named`Whitening`-$k$.

![whitening-SBERT](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/whitening-SBERT.png)

whitening-$k$ 操作的伪代码。

> Pseudo code of the whitening-$k$ operation. (Image source: Su et al. 2021)

白化操作被证明优于 BERT-flow，在 256 维句子嵌入下于许多 STS 基准上达到 SOTA，无论是否使用 NLI 监督。

> Whitening operations were shown to outperform BERT-flow and achieve SOTA with 256 sentence dimensionality on many STS benchmarks, either with or without NLI supervision.

### 无监督句子嵌入学习

> **EN** Unsupervised Sentence Embedding Learning

#### 上下文预测

> **EN** Context Prediction

Quick-Thought（QT）向量（Logeswaran & Lee, 2018）将句子表示学习表述为分类问题：给定一个句子及其上下文，分类器根据向量表示区分上下文句子与其他对比句子（“完形填空”测试）。这种表述移除了导致训练变慢的 softmax 输出层。

> Quick-Thought (QT) vectors (Logeswaran & Lee, 2018) formulate sentence representation learning as a classification problem: Given a sentence and its context, a classifier distinguishes context sentences from other contrastive sentences based on their vector representations ("cloze test"). Such a formulation removes the softmax output layer which causes training slowdown.

![quick-thought](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/quick-thought.png)

Quick-Thought 句子嵌入向量学习示意图。

> Illustration of how Quick-Thought sentence embedding vectors are learned. (Image source: Logeswaran & Lee, 2018)

令 $f(.)$ 与 $g(.)$ 为将句子 $s$ 编码为定长向量的两个函数。令 $C(s)$ 为 $s$ 的上下文句子集合，$S(s)$ 为候选句子集合，其中仅包含一个上下文句子 $s_c \in C(s)$ 及许多非上下文负样本句子。Quick Thoughts 模型学习优化预测唯一真实上下文句子 $s_c \in S(s)$ 的概率。当将句子对 $(s, s_c)$ 视为正样本对、而将其他对 $(s, s')$（$s' \in S(s), s'\neq s_c$）视为负样本时，这本质上就是 NCE 损失。

> Let $f(.)$ and $g(.)$ be two functions that encode a sentence $s$ into a fixed-length vector. Let $C(s)$ be the set of sentences in the context of $s$ and $S(s)$ be the set of candidate sentences including only one sentence $s_c \in C(s)$ and many other non-context negative sentences. Quick Thoughts model learns to optimize the probability of predicting the only true context sentence $s_c \in S(s)$. It is essentially NCE loss when considering the sentence $(s, s_c)$ as the positive pairs while other pairs $(s, s')$ where $s' \in S(s), s'\neq s_c$ as negatives.

$$ \mathcal{L}_\text{QT} = - \sum_{s \in \mathcal{D}} \sum_{s_c \in C(s)} \log p(s_c \vert s, S(s)) = - \sum_{s \in \mathcal{D}} \sum_{s_c \in C(s)}\frac{\exp(f(s)^\top g(s_c))}{\sum_{s'\in S(s)} \exp(f(s)^\top g(s'))} $$

#### 互信息最大化

> **EN** Mutual Information Maximization

IS-BERT（Info-Sentence BERT）（Zhang et al. 2020；code）采用基于互信息最大化的自监督学习目标，以无监督方式学习良好的句子嵌入。

> IS-BERT (Info-Sentence BERT) (Zhang et al. 2020; code) adopts a self-supervised learning objective based on mutual information maximization to learn good sentence embeddings in the unsupervised manners.

![IS-BERT](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/IS-BERT.png)

Info-Sentence BERT 示意图。

> Illustration of Info-Sentence BERT. (Image source: Zhang et al. 2020)

IS-BERT 工作流程如下：

> IS-BERT works as follows:

使用 BERT 将输入句子 $s$ 编码为长度为 $l$ 的 token 嵌入 $\mathbf{h}_{1:l}$。

> Use BERT to encode an input sentence $s$ to a token embedding of length $l$, $\mathbf{h}_{1:l}$.

然后应用具有不同核大小（例如 1、3、5）的一维卷积网络处理 token 嵌入序列，以捕获 n-gram 局部上下文依赖：$\mathbf{c}_i = \text{ReLU}(\mathbf{w} \cdot \mathbf{h}_{i:i+k-1} + \mathbf{b})$。输出序列经填充以保持与输入相同尺寸。

> Then apply 1-D conv net with different kernel sizes (e.g. 1, 3, 5) to process the token embedding sequence to capture the n-gram local contextual dependencies: $\mathbf{c}_i = \text{ReLU}(\mathbf{w} \cdot \mathbf{h}_{i:i+k-1} + \mathbf{b})$. The output sequences are padded to stay the same sizes of the inputs.

第 $i$ 个 token 的最终局部表示 $\mathcal{F}_\theta^{(i)} (\mathbf{x})$ 为不同核大小表示的拼接。

> The final local representation of the $i$-th token $\mathcal{F}_\theta^{(i)} (\mathbf{x})$ is the concatenation of representations of different kernel sizes.

全局句子表示 $\mathcal{E}_\theta(\mathbf{x})$ 通过对 token 表示 $\mathcal{F}_\theta(\mathbf{x}) = \{\mathcal{F}_\theta^{(i)} (\mathbf{x}) \in \mathbb{R}^d\}_{i=1}^l$ 做 mean-over-time 池化得到。

> The global sentence representation $\mathcal{E}_\theta(\mathbf{x})$ is computed by applying a mean-over-time pooling layer on the token representations $\mathcal{F}_\theta(\mathbf{x}) = \{\mathcal{F}_\theta^{(i)} (\mathbf{x}) \in \mathbb{R}^d\}_{i=1}^l$.

由于互信息估计对连续高维随机变量通常不可 tract，IS-BERT 依赖 Jensen-Shannon 估计器（Nowozin et al., 2016, Hjelm et al., 2019）来最大化 $\mathcal{E}_\theta(\mathbf{x})$ 与 $\mathcal{F}_\theta^{(i)} (\mathbf{x})$ 之间的互信息。

> Since the mutual information estimation is generally intractable for continuous and high-dimensional random variables, IS-BERT relies on the Jensen-Shannon estimator (Nowozin et al., 2016, Hjelm et al., 2019) to maximize the mutual information between $\mathcal{E}_\theta(\mathbf{x})$ and $\mathcal{F}_\theta^{(i)} (\mathbf{x})$.

$$ I^\text{JSD}_\omega(\mathcal{F}_\theta^{(i)} (\mathbf{x}); \mathcal{E}_\theta(\mathbf{x})) = \mathbb{E}_{\mathbf{x}\sim P} [-\text{sp}(-T_\omega(\mathcal{F}_\theta^{(i)} (\mathbf{x}); \mathcal{E}_\theta(\mathbf{x})))] \\ - \mathbb{E}_{\mathbf{x}\sim P, \mathbf{x}' \sim\tilde{P}} [\text{sp}(T_\omega(\mathcal{F}_\theta^{(i)} (\mathbf{x}'); \mathcal{E}_\theta(\mathbf{x})))] $$

其中 $T_\omega: \mathcal{F}\times\mathcal{E} \to \mathbb{R}$ 为可学习网络，参数为 $\omega$，生成分类器分数。负样本 $\mathbf{x}'$ 从分布 $\tilde{P}=P$ 采样。$\text{sp}(x)=\log(1+e^x)$ 为 softplus 激活函数。

> where $T_\omega: \mathcal{F}\times\mathcal{E} \to \mathbb{R}$ is a learnable network with parameters $\omega$, generating discriminator scores. The negative sample $\mathbf{x}'$ is sampled from the distribution $\tilde{P}=P$. And $\text{sp}(x)=\log(1+e^x)$ is the softplus activation function.

IS-BERT 在 SentEval 上的无监督结果优于大多数无监督基线（2020 年 9 月），但 unsurprisingly 弱于监督运行。使用带标签的 NLI 数据集时，IS-BERT 产生与 SBERT 相当的结果（见图 25 与 30）。

> The unsupervised numbers on SentEval with IS-BERT outperforms most of the unsupervised baselines (Sep 2020), but unsurprisingly weaker than supervised runs. When using labelled NLI datasets, IS-BERT produces results comparable with SBERT (See Fig. 25 & 30).

![IS-BERT-SentEval](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/IS-BERT-SentEval.png)

IS-BERT 在 SentEval 基准上的性能。

> The performance of IS-BERT on the SentEval benchmark. (Image source: Zhang et al. 2020)

## 引用

> **EN** Citation

引用格式如下：

> Cited as:

Weng, Lilian. (May 2021). Contrastive representation learning. Lil'Log. https://lilianweng.github.io/posts/2021-05-31-contrastive/.

> Weng, Lilian. (May 2021). Contrastive representation learning. Lil'Log. https://lilianweng.github.io/posts/2021-05-31-contrastive/.

或

> Or

```
@article{weng2021contrastive,
  title   = "Contrastive Representation Learning",
  author  = "Weng, Lilian",
  journal = "lilianweng.github.io",
  year    = "2021",
  month   = "May",
  url     = "https://lilianweng.github.io/posts/2021-05-31-contrastive/"
}

```

（中文说明：以上为 Lilian Weng 博客《Contrastive Representation Learning》的 BibTeX 引用条目，可直接用于 LaTeX 参考文献。）

## 参考文献

> **EN** References

[1] Sumit Chopra, Raia Hadsell and Yann LeCun.「Learning a similarity metric discriminatively, with application to face verification.」/「判别式学习相似度度量及其在人脸验证中的应用。」CVPR 2005.

> [1] Sumit Chopra, Raia Hadsell and Yann LeCun."Learning a similarity metric discriminatively, with application to face verification." CVPR 2005.

[2] Florian Schroff, Dmitry Kalenichenko and James Philbin.「FaceNet: A Unified Embedding for Face Recognition and Clustering.」/「FaceNet：用于人脸识别与聚类的统一嵌入。」CVPR 2015.

> [2] Florian Schroff, Dmitry Kalenichenko and James Philbin."FaceNet: A Unified Embedding for Face Recognition and Clustering." CVPR 2015.

[3] Hyun Oh Song et al.「Deep Metric Learning via Lifted Structured Feature Embedding.」/「通过提升结构化特征嵌入进行深度度量学习。」CVPR 2016. [code]

> [3] Hyun Oh Song et al."Deep Metric Learning via Lifted Structured Feature Embedding." CVPR 2016. [code]

[4] Ruslan Salakhutdinov and Geoff Hinton.「Learning a Nonlinear Embedding by Preserving Class Neighbourhood Structure」/「通过保持类邻域结构学习非线性嵌入」AISTATS 2007.

> [4] Ruslan Salakhutdinov and Geoff Hinton."Learning a Nonlinear Embedding by Preserving Class Neighbourhood Structure" AISTATS 2007.

[5] Michael Gutmann and Aapo Hyvärinen.「Noise-contrastive estimation: A new estimation principle for unnormalized statistical models.」/「噪声对比估计：非归一化统计模型的新估计原理。」AISTATS 2010.

> [5] Michael Gutmann and Aapo Hyvärinen."Noise-contrastive estimation: A new estimation principle for unnormalized statistical models." AISTATS 2010.

[6] Kihyuk Sohn et al.「Improved Deep Metric Learning with Multi-class N-pair Loss Objective」/「使用多类 N-pair 损失目标改进深度度量学习」NIPS 2016.

> [6] Kihyuk Sohn et al."Improved Deep Metric Learning with Multi-class N-pair Loss Objective" NIPS 2016.

[7] Nicholas Frosst, Nicolas Papernot and Geoffrey Hinton.「Analyzing and Improving Representations with the Soft Nearest Neighbor Loss.」/「用 Soft 最近邻损失分析与改进表示。」ICML 2019

> [7] Nicholas Frosst, Nicolas Papernot and Geoffrey Hinton."Analyzing and Improving Representations with the Soft Nearest Neighbor Loss." ICML 2019

[8] Tongzhou Wang and Phillip Isola.「Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.」/「通过超球面上的对齐与均匀性理解对比表示学习。」ICML 2020. [code]

> [8] Tongzhou Wang and Phillip Isola."Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere." ICML 2020. [code]

[9] Zhirong Wu et al.「Unsupervised feature learning via non-parametric instance-level discrimination.」/「通过非参数实例级判别进行无监督特征学习。」CVPR 2018.

> [9] Zhirong Wu et al."Unsupervised feature learning via non-parametric instance-level discrimination." CVPR 2018.

[10] Ekin D. Cubuk et al.「AutoAugment: Learning augmentation policies from data.」/「AutoAugment：从数据中学习增强策略。」arXiv preprint arXiv:1805.09501 (2018).

> [10] Ekin D. Cubuk et al."AutoAugment: Learning augmentation policies from data." arXiv preprint arXiv:1805.09501 (2018).

[11] Daniel Ho et al.「Population Based Augmentation: Efficient Learning of Augmentation Policy Schedules.」/「基于种群的增强：高效学习增强策略调度。」ICML 2019.

> [11] Daniel Ho et al."Population Based Augmentation: Efficient Learning of Augmentation Policy Schedules." ICML 2019.

[12] Ekin D. Cubuk & Barret Zoph et al.「RandAugment: Practical automated data augmentation with a reduced search space.」/「RandAugment：搜索空间缩减的实用自动化数据增强。」arXiv preprint arXiv:1909.13719 (2019).

> [12] Ekin D. Cubuk & Barret Zoph et al."RandAugment: Practical automated data augmentation with a reduced search space." arXiv preprint arXiv:1909.13719 (2019).

[13] Hongyi Zhang et al.「mixup: Beyond Empirical Risk Minimization.」/「mixup：超越经验风险最小化。」ICLR 2017.

> [13] Hongyi Zhang et al."mixup: Beyond Empirical Risk Minimization." ICLR 2017.

[14] Sangdoo Yun et al.「CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.」/「CutMix：训练具有可定位特征的强分类器的正则化策略。」ICCV 2019.

> [14] Sangdoo Yun et al."CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features." ICCV 2019.

[15] Yannis Kalantidis et al.「Mixing of Contrastive Hard Negatives」/「对比硬负样本的混合」NeuriPS 2020.

> [15] Yannis Kalantidis et al."Mixing of Contrastive Hard Negatives" NeuriPS 2020.

[16] Ashish Jaiswal et al.「A Survey on Contrastive Self-Supervised Learning.」/「对比自监督学习综述。」arXiv preprint arXiv:2011.00362 (2021)

> [16] Ashish Jaiswal et al."A Survey on Contrastive Self-Supervised Learning." arXiv preprint arXiv:2011.00362 (2021)

[17] Jure Zbontar et al.「Barlow Twins: Self-Supervised Learning via Redundancy Reduction.」/「Barlow Twins：通过冗余削减的自监督学习。」arXiv preprint arXiv:2103.03230 (2021) [code]

> [17] Jure Zbontar et al."Barlow Twins: Self-Supervised Learning via Redundancy Reduction." arXiv preprint arXiv:2103.03230 (2021) [code]

[18] Alec Radford, et al.「Learning Transferable Visual Models From Natural Language Supervision」/「从自然语言监督学习可迁移视觉模型」arXiv preprint arXiv:2103.00020 (2021)

> [18] Alec Radford, et al."Learning Transferable Visual Models From Natural Language Supervision" arXiv preprint arXiv:2103.00020 (2021)

[19] Mathilde Caron et al.「Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (SwAV).」/「通过对比聚类分配无监督学习视觉特征（SwAV）。」NeuriPS 2020.

> [19] Mathilde Caron et al."Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (SwAV)." NeuriPS 2020.

[20] Mathilde Caron et al.「Deep Clustering for Unsupervised Learning of Visual Features.」/「用于无监督视觉特征学习的深度聚类。」ECCV 2018.

> [20] Mathilde Caron et al."Deep Clustering for Unsupervised Learning of Visual Features." ECCV 2018.

[21] Prannay Khosla et al.「Supervised Contrastive Learning.」/「监督对比学习。」NeurIPS 2020.

> [21] Prannay Khosla et al."Supervised Contrastive Learning." NeurIPS 2020.

[22] Aaron van den Oord, Yazhe Li & Oriol Vinyals.「Representation Learning with Contrastive Predictive Coding」/「对比预测编码的表示学习」arXiv preprint arXiv:1807.03748 (2018).

> [22] Aaron van den Oord, Yazhe Li & Oriol Vinyals."Representation Learning with Contrastive Predictive Coding" arXiv preprint arXiv:1807.03748 (2018).

[23] Jason Wei and Kai Zou.「EDA: Easy data augmentation techniques for boosting performance on text classification tasks.」/「EDA：提升文本分类任务性能的简易数据增强技术。」EMNLP-IJCNLP 2019.

> [23] Jason Wei and Kai Zou."EDA: Easy data augmentation techniques for boosting performance on text classification tasks." EMNLP-IJCNLP 2019.

[24] Sosuke Kobayashi.「Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations.」/「Contextual Augmentation：基于范式关系的词级数据增强。」NAACL 2018

> [24] Sosuke Kobayashi."Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations." NAACL 2018

[25] Hongchao Fang et al.「CERT: Contrastive self-supervised learning for language understanding.」/「CERT：用于语言理解的对比自监督学习。」arXiv preprint arXiv:2005.12766 (2020).

> [25] Hongchao Fang et al."CERT: Contrastive self-supervised learning for language understanding." arXiv preprint arXiv:2005.12766 (2020).

[26] Dinghan Shen et al.「A Simple but Tough-to-Beat Data Augmentation Approach for Natural Language Understanding and Generation.」/「一种简单但难以击败的自然语言理解与生成数据增强方法。」arXiv preprint arXiv:2009.13818 (2020) [code]

> [26] Dinghan Shen et al."A Simple but Tough-to-Beat Data Augmentation Approach for Natural Language Understanding and Generation." arXiv preprint arXiv:2009.13818 (2020) [code]

[27] Tianyu Gao et al.「SimCSE: Simple Contrastive Learning of Sentence Embeddings.」/「SimCSE：句子嵌入的简单对比学习。」arXiv preprint arXiv:2104.08821 (2020). [code]

> [27] Tianyu Gao et al."SimCSE: Simple Contrastive Learning of Sentence Embeddings." arXiv preprint arXiv:2104.08821 (2020). [code]

[28] Nils Reimers and Iryna Gurevych.「Sentence-BERT: Sentence embeddings using Siamese BERT-networks.」/「Sentence-BERT：使用孪生 BERT 网络的句子嵌入。」EMNLP 2019.

> [28] Nils Reimers and Iryna Gurevych."Sentence-BERT: Sentence embeddings using Siamese BERT-networks." EMNLP 2019.

[29] Jianlin Su et al.「Whitening sentence representations for better semantics and faster retrieval.」/「白化句子表示以获得更好语义与更快检索。」arXiv preprint arXiv:2103.15316 (2021). [code]

> [29] Jianlin Su et al."Whitening sentence representations for better semantics and faster retrieval." arXiv preprint arXiv:2103.15316 (2021). [code]

[30] Yan Zhang et al.「An unsupervised sentence embedding method by mutual information maximization.」/「一种通过互信息最大化实现的无监督句子嵌入方法。」EMNLP 2020. [code]

> [30] Yan Zhang et al."An unsupervised sentence embedding method by mutual information maximization." EMNLP 2020. [code]

[31] Bohan Li et al.「On the sentence embeddings from pre-trained language models.」/「论预训练语言模型的句子嵌入。」EMNLP 2020.

> [31] Bohan Li et al."On the sentence embeddings from pre-trained language models." EMNLP 2020.

[32] Lajanugen Logeswaran and Honglak Lee.「An efficient framework for learning sentence representations.」/「学习句子表示的高效框架。」ICLR 2018.

> [32] Lajanugen Logeswaran and Honglak Lee."An efficient framework for learning sentence representations." ICLR 2018.

[33] Joshua Robinson, et al.「Contrastive Learning with Hard Negative Samples.」/「带硬负样本的对比学习。」ICLR 2021.

> [33] Joshua Robinson, et al."Contrastive Learning with Hard Negative Samples." ICLR 2021.

[34] Ching-Yao Chuang et al.「Debiased Contrastive Learning.」/「去偏对比学习。」NeuriPS 2020.

> [34] Ching-Yao Chuang et al."Debiased Contrastive Learning." NeuriPS 2020.
