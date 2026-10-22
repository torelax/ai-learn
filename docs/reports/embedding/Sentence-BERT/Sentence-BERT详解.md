# Sentence-BERT：可独立编码的语义句向量

> 用共享权重的 BERT 分别编码句子，再通过句对监督把向量空间训练成可直接用余弦相似度检索的空间。
>
> Nils Reimers、Iryna Gurevych · EMNLP-IJCNLP 2019 · [论文](https://aclanthology.org/D19-1410/) · [arXiv v1](https://arxiv.org/abs/1908.10084v1) · [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)

## 论文简介

BERT 擅长把两个句子一起读完再判断关系，但每个新句对都要重新运行 Transformer。对于语义搜索、聚类和大规模两两比较，这种交叉编码器的成本很高；直接拿未经句向量训练的 BERT 输出求平均，语义相似度又不好。SBERT 保留预训练语言模型，用共享参数的孪生／三元组结构，结合池化与 NLI、相似度回归或三元组监督，得到可离线计算、在线比较的句向量。

论文分别考察通用语义相似度、论证相似度、Wikipedia 章节匹配和分类迁移。它显著优于直接池化 BERT，且大幅降低大量句对比较的成本；但双塔并非在所有任务上达到交叉编码器精度，跨话题迁移仍有明显差距。本文数值采用原论文设定，不能当作今天所有 Sentence Transformers 模型的成绩。

## 背景与相关工作

交叉编码器输入 `[CLS] A [SEP] B [SEP]`，两个句子的 token 从第一层起互相注意；它输出的是**一对句子的判断**，没有可复用的独立句向量。对 10,000 句寻找最相似句对，需要约 5,000 万次网络推理。作者在当时的 V100 上估算约 65 小时。

另一条路线将每句映射为固定向量。Skip-Thought 用上下文预测训练，InferSent 用 SNLI/MultiNLI 训练 BiLSTM，Universal Sentence Encoder 混合无监督目标和 NLI。SBERT 沿用句对监督思路，但从预训练 BERT/RoBERTa 出发，降低训练成本。论文还讨论 Poly-Encoder：它能缓存候选表示，但额外交互及非对称打分不适合所有聚类用法。

## 模型：池化与共享参数

对 token 输出 $h_1,\ldots,h_T$，默认用 mean pooling 得到句向量。实际批处理只对有效 token 求均值：

$$
u=\frac{\sum_{i=1}^{T}m_i h_i}{\sum_{i=1}^{T}m_i}.
$$

$m_i$ 是 padding mask。论文还比较 `[CLS]` 和逐维最大值池化。两条分支共享同一个编码器参数，因此任意句子都能用同一映射独立编码；“孪生”不是两套独立训练的 BERT。

![原论文图 1：NLI 分类训练结构](figs/S3-F1-01.png)

图中两个 BERT 共享权重，池化后的 $u,v$ 与逐元素绝对差共同进入分类头。句对关系通过损失反传到编码器，推理时不需要保留这个分类头。

### 三种监督目标

**分类目标。** 对 NLI 的蕴含、矛盾、中立三类，使用

$$
p=\operatorname{softmax}\bigl(W[u;v;|u-v|]+b\bigr),\qquad
\mathcal L_{\rm cls}=-\log p_y.
$$

其中 $u,v\in\mathbb R^d$，拼接特征为 $3d$ 维。这个目标不是现代检索训练中“批内其他文档都是负例”的 InfoNCE。

**回归目标。** 对有连续相似度标注的句对，用余弦相似度和目标分数的均方误差训练；推理时直接计算同一个余弦相似度。

**三元组目标。** 给定 anchor、正例、负例：

$$
\mathcal L_{\rm triplet}=\max\{\|s_a-s_p\|_2-\|s_a-s_n\|_2+\epsilon,0\}.
$$

论文使用欧氏距离、$\epsilon=1$。它约束相对距离，不要求正例与 anchor 的向量完全重合。

![原论文图 2：回归训练与相似度推理](figs/S3-F2-01.png)

图 2 中两个句向量直接进入余弦函数。候选语料的向量能预先算好，查询到来时只需编码查询和执行向量搜索。

## 训练数据与设置

通用模型联合使用 **SNLI 约 57 万句对**与 **MultiNLI 约 43 万句对**。前者主要来自图像描述句，后者扩展到口语、书面语等多种体裁；监督是句对逻辑关系，而不是搜索点击或 query–passage relevance。

默认训练一个 epoch，batch size 16，Adam 学习率 $2\times10^{-5}$，前 10% 步数做线性 warmup，默认 mean pooling。BERT-base/large 与 RoBERTa-base/large 分别形成 SBERT/SRoBERTa。针对 STS、论证相似度和章节匹配，作者另外用目标任务数据继续训练，评测时区分这些模型。

## 语义相似度实验

### 不使用 STS 标注进行训练

STS12–16、STS benchmark 与 SICK-R 提供句对语义相关性标注；论文用向量余弦与人工标注的 **Spearman 相关系数乘 100**。这里所谓“unsupervised STS”只表示没有使用 STS 训练集，SBERT 本身已经接受 NLI 监督。

| 方法 | STS benchmark | 七项平均 |
|---|---:|---:|
| GloVe 均值 | 58.02 | 61.32 |
| 原始 BERT 均值 | 46.35 | 54.81 |
| 原始 BERT CLS | 16.50 | 29.19 |
| InferSent | 68.03 | 65.01 |
| Universal Sentence Encoder | 74.92 | 71.22 |
| SBERT-NLI-base | 77.03 | 74.89 |
| SBERT-NLI-large | 79.23 | 76.55 |
| SRoBERTa-NLI-large | 79.10 | 76.68 |

结果支持的核心结论是：**适合语言建模的 token 表示，不自动等于适合余弦比较的句向量**。SBERT 的收益来自针对句级几何结构的训练；RoBERTa 替换带来的平均增益反而很小。SICK-R 上 USE 仍有优势，说明训练领域和任务组成影响结果。

### 使用 STS benchmark 微调

STS benchmark 共 8,628 句对，训练／开发／测试为 5,749／1,500／1,379，覆盖描述、新闻和论坛。作者采用 10 个随机种子，报告测试集均值和方差。

| 模型与训练顺序 | Spearman ×100 |
|---|---:|
| BERT-base，仅 STSb | $84.30\pm0.76$ |
| SBERT-base，仅 STSb | $84.67\pm0.19$ |
| BERT-large，NLI → STSb | $88.77\pm0.46$ |
| SBERT-large，NLI → STSb | $86.10\pm0.13$ |

NLI 预训练再做 STS 微调通常有益，但交叉编码器仍可更准。因此“保持 BERT 准确率”应理解为若干设定下接近，不能写成无损替代。

## 论证、章节与迁移任务

**Argument Facet Similarity（AFS）**要求区分同一议题下的不同论证角度，比“是否谈同一主题”更细。10 折验证中 SBERT-large 的 Spearman 为 75.93，接近 BERT-large 的 76.38；改为两个话题训练、第三个话题测试后，两者分别降到 53.10 和 60.34。领域内成绩不能替代跨话题泛化证据。

**Wikipedia 章节三元组**以同章节句子为正例、不同章节句子为负例。三元组目标训练一个 epoch 后，SBERT-base/large 准确率为 80.42%/80.78%，对照方法约 74%。这里学的是章节主题关系，和 NLI 逻辑关系不是同一个标签。

**SentEval**冻结句向量并训练逻辑回归分类器，覆盖情感、主观性、问题类型和释义识别等七项任务。SBERT-NLI-large 平均 87.69，InferSent 为 85.59，USE 为 85.10。原始 BERT 在这组任务并不像 STS 那样差：下游分类器可以重新利用特征，而直接余弦没有这个补偿环节。

## 消融：哪些结论有条件

原论文表 6 在 **STSb 开发集**评估，不要与前面的测试集数字混用。

| 池化 | NLI 分类训练 | STSb 回归训练 |
|---|---:|---:|
| Mean | 80.78 | 87.44 |
| Max | 79.07 | 69.92 |
| CLS | 79.80 | 86.62 |

“NLI 下池化差别较小”有数据支持；“任何训练目标下池化都不重要”则不成立，Max 在回归训练中大幅落后。

NLI 分类头仅拼接 $(u,v)$ 时为 66.04，加入 $|u-v|$ 后为 80.78；再加入逐元素乘积 $u*v$ 为 80.44，没有进一步改善。差分特征对这套训练特别重要，但分类头拼接形式只影响训练，线上余弦检索不会拼接所有候选再运行分类器。

## 计算效率与结论

长度相近句子组成 batch 的 smart batching 减少 padding。在作者硬件与 STSb 句长分布下，SBERT-base 吞吐由 GPU 1,378 句/秒增至 2,042；CPU 从 44 增至 83。原文“65 小时到约 5 秒”的大幅收益来自把大量 **句对 Transformer 前向**变成 **每句一次前向加便宜的向量比较**，不是声称 BERT 本体快了四万倍。

论文建立了可缓存语义句向量的训练范式，适用于相似句搜索、聚类和迁移特征。实验重点是英文短句及当时的基准；它没有证明模型具备长文档、复杂约束、所有语言或高难度检索能力，也没有把向量索引构建和端到端服务开销纳入所有时间数字。
