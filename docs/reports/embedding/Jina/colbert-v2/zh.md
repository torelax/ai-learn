> 原文: [arXiv:2408.16672](https://arxiv.org/abs/2408.16672)（2024-08-29；v4 2024-09-14）
> 说明: 中英对照供阅读。英文为论文原文（HTML 转写，公式按论文整理）；中文为对应翻译。图表编号与原文一致。caption 保留英文并附中译。数值原样，未改动。

**预印本：** arXiv:2408.16672v4 [cs.IR]，2024 年 9 月 14 日。CC BY 4.0。

**权重：** https://huggingface.co/jinaai/jina-colbert-v2

---

# Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever

# Jina-ColBERT-v2：通用多语迟交互检索器

**Authors / 作者：** Rohan Jha<sup>†</sup>, Bo Wang, Michael Günther, Georgios Mastrapas, Saba Sturua, Isabelle Mohr, Andreas Koukounas, Mohammad Kalim Akram, Nan Wang, Han Xiao

<sup>†</sup> Work done while at Jina AI. / 于 Jina AI 期间完成。

**Affiliation / 单位：** The University of Texas at Austin；Jina AI GmbH, Berlin · research@jina.ai

---

## Abstract / 摘要

**EN.** Multi-vector dense models, such as ColBERT, have proven highly effective in information retrieval. ColBERT’s late interaction scoring approximates the joint query-document attention seen in cross-encoders while maintaining inference efficiency closer to traditional dense retrieval models, thanks to its bi-encoder architecture and recent optimizations in indexing and search. In this work we propose a number of incremental improvements to the ColBERT model architecture and training pipeline, using methods shown to work in the more mature single-vector embedding model training paradigm, particularly those that apply to heterogeneous multilingual data or boost efficiency with little tradeoff. Our new model, Jina-ColBERT-v2, demonstrates strong performance across a range of English and multilingual retrieval tasks.

**中.** 多向量稠密模型（如 ColBERT）在信息检索中已被证明非常有效。ColBERT 的迟交互打分在近似 cross-encoder 式联合 query–文档注意力的同时，凭借双编码器架构以及近年索引与搜索优化，把推理效率保持在更接近传统稠密检索的水平。本文对 ColBERT 的模型架构与训练流水线提出若干增量改进，借鉴更成熟的单向量嵌入训练范式中已验证有效的方法——尤其是面向异构多语数据、或以很小代价提升效率的那些。新模型 **Jina-ColBERT-v2** 在一系列英文与多语检索任务上表现强劲。

---

## 1 Introduction / 引言

**EN.** Neural retrieval has gained popularity in recent years following the arrival of capable pre-trained language models (PLMs) (Devlin et al., 2019; Liu et al., 2019; Clark et al., 2020). Two types of approaches have been employed to apply PLMs to retrieval. Sparse neural retrieval systems, such as SPLADE (Formal et al., 2021), represent texts as weighted bags of words that are interpreted as sparse high-dimensional vectors for maximum inner product search (MIPS). Dense retrievers similarly encode queries and documents as dense vectors, capturing relevance signals through spatial relationships extending beyond exact term matching.

**中.** 随着有能力的预训练语言模型（PLM）出现，神经检索近年流行起来。将 PLM 用于检索大致有两类：稀疏神经检索（如 SPLADE）把文本表示为加权词袋，当作稀疏高维向量做最大内积搜索（MIPS）；稠密检索则把查询与文档编码为稠密向量，用超越精确词匹配的空间关系捕捉相关性。

**EN.** Most dense retrievers encode a query or document as a single vector, commonly the result of mean-pooling or the [CLS]-embedding over the transformer’s final layer token embeddings. In contrast, recent multi-vector retrievers like ColBERT (Khattab and Zaharia, 2020) generalize this embedding process to maintain an embedding for each token, computing relevance scores as a function of the similarities of query and document tokens instead. To make the ColBERT usable in practice, the output dimensionality is restricted to be much smaller than the single-vector models. This approach has the benefit of remaining compatible with much of the vector similarity infrastructure that makes single-vector methods efficient, but requires more space to store even a smaller embedding per token and compute at inference time to aggregate token interactions into a single score. This late interaction over token embeddings achieves greater in-domain performance and tends to be more robust out-of-domain than single-vector similarity. While ColBERTv2 is trained only on English MSMARCO triplets (Bajaj et al., 2016) and has a monolingual BERT backbone, making it incapable of multilingual retrieval, some previous works extend the model to multilingual retrieval.

**中.** 多数稠密检索器用单个向量表示查询或文档（常见为末层 token 嵌入的均值池化或 [CLS]）。相比之下，ColBERT 这类多向量检索器为每个 token 保留一个嵌入，相关性由查询 token 与文档 token 的相似度函数给出。为实用起见，输出维远小于单向量模型：仍可复用大量向量相似度基础设施，但每个 token 都要存一个（虽更小的）向量，推理时还要把 token 交互聚成一个分数。这种对 token 嵌入的**迟交互**通常比单向量相似度有更好的域内表现，且域外更稳健。ColBERTv2 仅在英文 MSMARCO 三元组上训练、骨干为单语 BERT，无法做多语检索；已有工作把它扩展到多语。

**EN.** ColBERT-XM (Louis et al., 2024) does this by using parameter extensions for each additional language, and (Lawrie et al., 2023) trains solely on machine-translated English MSMARCO data to get effective heterogeneous multilingual performance. These approaches, however, come with trade-offs in terms of model usability and training data diversity. Other multilingual multi-vector models like BGE-M3 (Chen et al., 2024) produce extremely large token representations that limit their practical utility for first-stage retrieval.

**中.** ColBERT-XM 用每语参数扩展；Lawrie et al. (2023) 仅用机翻 MSMARCO 获得异构多语表现。二者在可用性与训练数据多样性上各有代价。BGE-M3 等模型的 token 表示极大，限制其一阶段检索的实用性。

**EN.** In this work, we propose Jina-ColBERT-v2, which introduces an improved training recipe for ColBERT models with the following features:

- **Training with diverse weakly-supervised data:** We additionally pretrain our modified PLM with rotary position embedding and train on large-scale unlabeled text pairs from various corpus with a weakly-supervised single-vector contrastive objective. A second-stage of ColBERT finetuning with labeled triplet data and supervised distillation is used to further boost its performance.
- **General multilingual performance:** We train with data from a variety of high- and low-resource languages using both labeled and unlabeled data, including human- and machine-translated training data, and show that this improves even out-of-domain multilingual performance.
- **Inference-agnostic efficiency:** We introduce multiple sizes of linear projection heads, jointly trained using the non-weight tying variant of Matryoshka Representation Loss (Kusupati et al., 2022), enabling the selection of token embedding size at inference time with minimal performance degradation. We demonstrate that reducing the embedding dimensionality in half from 128 to 64 yields only a minor performance tradeoff. Additionally, our flash-attention optimized backbone, Jina-XLM-RoBERTa, provides further free performance improvement during inference.

**中.** 本文提出 **Jina-ColBERT-v2**，改进 ColBERT 训练配方，要点如下：

- **多样弱监督数据训练：** 对改造后的 PLM 用旋转位置编码继续预训练，并在大规模无标注文本对上以弱监督单向量对比目标训练；第二阶段用带标注三元组与监督蒸馏做 ColBERT 微调。
- **通用多语表现：** 在高/低资源多种语言上，用有标与无标数据（含人工与机翻），并显示这甚至能提升域外多语表现。
- **与推理解耦的效率：** 联合训练多种尺寸的线性投影头（Matryoshka Representation Loss 的非权值共享变体），推理时可选择 token 嵌入维数且性能损失很小；维数从 128 减半到 64 仅有轻微代价。Flash-attention 优化骨干 **Jina-XLM-RoBERTa** 在推理时再带来几乎「白捡」的性能提升。

**EN.** Our experimental results show competitive retrieval performance across both English and multilingual benchmarks. We also present controlled experiments demonstrating the benefits, or lack thereof, of the training modifications we consider in developing our training recipe.

**中.** 实验在英文与多语基准上表现有竞争力；并对训练配方中的各项改动做了受控实验，说明其收益或缺失。

---

## 2 Related Work / 相关工作

### 2.1 Single-Vector Retrieval / 单向量检索

**EN.** Single-vector encoder models have demonstrated their potential as general-purpose embedding models across a number of downstream tasks (Muennighoff et al., 2023). When used in a bi-encoder retrieval model, they asymmetrically encode queries and documents as separate dense vectors, and measure their pairwise relevance as the cosine similarity between the vectors. Owing to their strong in-domain performance and straightforward inference scheme, there has been a growing focus on improving their training. Studies demonstrate that large-scale unsupervised pair training utilizing in-batch negatives, followed by a small-scale triplet finetuning stage, significantly improves performance compared to a dense retriever trained solely on triplet data (Li et al., 2023; Günther et al., 2023). Other works have incorporated asymmetric task-specific instructions for queries and documents to further enhance performance (Wang et al., 2024) and demonstrated the efficacy of using synthetically generated training data, including using diverse task instructions and machine translations, to further improve model representations (Wang et al., 2023; Lee et al., 2024).

**中.** 单向量编码器已在多种下游任务上展现通用嵌入潜力。双编码器设定下，查询与文档分别编码，用余弦相似度衡量相关性。因域内强、推理直接，训练改进受到关注。研究表明：大规模无监督成对训练（批内负例）再接小规模三元组微调，显著优于只在三元组上训的稠密检索器。另有工作为查询/文档加非对称任务指令，以及用合成数据（多样指令、机翻）改善表示。

### 2.2 Multi-Vector Retrieval / 多向量检索

**EN.** Multi-vector retrievers like ColBERT also employ a bi-encoder structure, but queries and passages are represented by a collection of smaller token embeddings rather than one large vector. As such, ColBERTv2’s training uses many of the same techniques as state-of-the-art single-vector models: cross-encoder distillation, multiple negatives per query, and self-mined hard negatives. Recent models have continued to improve on this training recipe, particularly for multilingual or non-English training. BGE-M3 (Chen et al., 2024) adopts the two-stage pairs-to-triplets training pipeline, and does self-knowledge distillation, treating the combination of its sparse, dense, and multi-vector scores as the teacher score.

**中.** ColBERT 类多向量检索器同样是双编码器，但用一组较小的 token 嵌入而非一个大向量。ColBERTv2 因此沿用了许多单向量 SOTA 技巧：cross-encoder 蒸馏、每查询多负例、自挖难负例。近年配方继续改进，尤其面向多语；BGE-M3 采用「成对→三元组」两阶段，并以稀疏/稠密/多向量分数组合做自蒸馏老师分。

### 2.3 Multilingual Retrieval / 多语检索

**EN.** Owing to the quality of English-based pre-trained models (BERT) and annotated data (MSMARCO), many advances in neural retrieval have been applied first to the monolingual English setting (Karpukhin et al., 2020; Xiong et al., 2020; Khattab and Zaharia, 2020). Researchers, however, have also made advances in non-English capabilities. On the modeling front, multilingual PLMs like mBERT (Devlin et al., 2019) and later XLM-RoBERTa (Conneau et al., 2020) have expanded pre-training to include text in up to 100 languages, including in cross-language contexts. For multilingual retrieval data, there are two approaches: natural and translated. Datasets like Mr-Tydi and MIRACL (Zhang et al., 2021; Zhang et al., 2023b) are built from human-generated and annotated queries, whereas mMARCO (Bonifacio et al., 2022) is a collection of machine-translated copies of MSMARCO which inherit their judgments from the original dataset. The former method tends to be of higher quality and lacks the subtle distributional/idiomatic errors, dubbed "translationese", that the latter sometimes exhibits. Naturally, however, human generation costs more per example. Recent multi-vector work has also proposed further modifications along the dimensions of architecture and data. ColBERT-XM (Louis et al., 2024) addresses the so-called curse of multilinguality (Conneau et al., 2020), the performance degradation of models pre-trained on too many tasks, with shared- and per-language parameters that allow for more robust zero-shot language transfer and post-hoc language extension. On the data approach, ColBERT-X (Nair et al., 2022; Lawrie et al., 2023; Yang et al., 2024) uses language-mixed batches of machine-translated English data, and BGE-M3 (Chen et al., 2024) curates unsupervised and high-quality supervised corpora of diverse multilingual training data.

**中.** 因英文 PLM（BERT）与标注（MSMARCO）质量高，许多神经检索进展先落在单语英文。建模上，mBERT、XLM-RoBERTa 等把预训练扩到最多约 100 种语言及跨语语境。多语检索数据有两条路：自然与翻译。Mr. TyDi、MIRACL 等人造查询与标注；mMARCO 是 MSMARCO 的机翻副本并继承原判定。前者质量更高、少「翻译腔」，但人工成本更高。架构上 ColBERT-XM 用共享+分语参数应对「多语诅咒」；数据上 ColBERT-X 用语言混合的机翻批次，BGE-M3 精选无监督与高质量监督多语语料。

---

## 3 Training Overview / 训练总览

**EN.** Jina-ColBERT-v2’s training paradigm has three parts:

1. **Modified Encoder Architecture:** We use a modified encoder backbone, derived from XLM-RoBERTa with improvements made to its architecture and pre-training regime. We further extend ColBERT’s linear projection head by jointly training a collection of different-size heads for embedding size reduction.
2. **Pair Training:** To learn from the semantic structure of large quantities of diverse data in many languages, we first train our encoder model on weakly supervised text pairs from a variety of embedding datasets.
3. **Triplet Training:** Our model is further finetuned using retrieval examples in many languages with both positives and hard negatives, supervised by a highly-capable multilingual cross-encoder.

**中.** 训练范式分三部分：（1）改造编码器（源自 XLM-RoBERTa，改进架构与预训练；并联合训练多种尺寸的线性投影头）；（2）成对训练（弱监督多语文本对）；（3）三元组训练（多语正例+难负例，由强多语 cross-encoder 监督蒸馏）。

---

## 4 Architecture / 架构

### 4.1 Backbone Improvements / 骨干改进

**EN.** Following many prior single- and multi-vector multilingual training efforts, we adopted XLM-RoBERTa as our backbone model due to its strong performance across various downstream tasks (Nair et al., 2022; Louis et al., 2024; Chen et al., 2024). To improve the efficiency, we enhance the XLM-RoBERTa architecture with flash attention (Dao, 2024). We replace the absolute positional embeddings with rotary positional embeddings (RoPE, Su et al., 2023), which are empirically understood to be better. They also have the advantage of supporting context lengths far longer than 512 tokens, although we do not explicitly focus on long-context in this work. To warm up its new positional embeddings, we continued pre-training the modified backbone with the same masked language modeling objective for 160,000 steps on the RefinedWeb dataset (Penedo et al., 2023), a modern, high-quality corpus, under the masked language modeling objective. During this pre-training phase, we set the maximum sequence length to 8,192 tokens with a rotary base of 10,000 and employed whole-word-masking (Devlin et al., 2019), masking out 30% of the tokens. We call this modified language model Jina-XLM-RoBERTa.

**中.** 沿用多语单/多向量训练惯例，选用 **XLM-RoBERTa** 为骨干。为效率加入 **flash attention**；用 **RoPE** 替换绝对位置嵌入（经验上更好，且可支持远超 512 token 的上下文；本文不专门做长上下文）。为预热新位置编码，在 **RefinedWeb** 上以同样的掩码语言建模目标继续预训练 **160,000** 步；最大序列长度 **8,192**，rotary base **10,000**，整词掩码，掩 **30%** token。称该改造 LM 为 **Jina-XLM-RoBERTa**。

### 4.2 Multiple Linear Heads / 多线性投影头

**EN.** To reduce index sizes, ColBERT includes a linear head that projects its token embeddings from the hidden dimension of its language model down to a lower dimension ($768\to 128$). As a notable exception, BGE-M3’s multi-vector retrieval does not take this step, keeping its token embeddings at a full 1024 dimensions. We jointly train six linear heads with dimensions $d\in\{64,96,128,256,512,768\}$ using Matryoshka Representation Loss (MRL, Kusupati et al., 2022). This allows users to choose greater or lesser space efficiency, with an associated performance trade-off. Figure 1 quantifies this tradeoff, showing the strong performance preservation of our reduced-dimension linear heads. Halving the token dimension ($128\to 64$) only causes its nDCG@10 to drop by 0.01 (1.59%). We unfortunately find that MRL’s weight-tying efficient variant (MRL-E), where losses are computed on truncations of the same token vector, does not preserve performance well, which we hypothesize is a consequence of the already-low projected dimension of the original ColBERT formulation.

**中.** 为减小索引，ColBERT 用线性头把 LM 隐维投影到更低维（$768\to 128$）。BGE-M3 的多向量检索是例外，保持 **1024** 维。本文用 MRL **联合训练六个头**：$d\in\{64,96,128,256,512,768\}$，推理时可在空间效率与性能间权衡。图 1 显示低维头仍能很好保分：维数减半（$128\to 64$）仅使 nDCG@10 下降 **0.01（1.59%）**。权值共享的高效变体 **MRL-E**（在同一向量截断上算损失）保分不佳；作者猜测因原 ColBERT 投影维已经很低。

![Figure 1: nDCG@10 scores for BEIR datasets using 64-, 96-, and 128-dimension linear projection heads for token embeddings.](figs/fig01.png)

**Caption EN.** Figure 1: nDCG@10 scores for BEIR datasets using 64-, 96-, and 128-dimension linear projection heads for token embeddings.

**Caption 中.** 图 1：令牌向量维度取 64、96、128 时，若干 BEIR 数据集上的 nDCG@10。

---

## 5 Pair Training / 成对训练

**EN.** To leverage an abundance of text pairs with varying richness of semantic structure, we draw inspiration from common practices in single-vector embedding model training and begin with training on these text pairs, focusing on optimizing the embedding model’s performance on general semantic similarity and relatedness tasks. This weakly-supervised stage is in contrast to previous ColBERT works, which typically start directly from a PLM like BERT with triplet training on 32-way or 64-way retrieval triplets consisting of a query, a positive passage, and multiple mined negatives.

**中.** 为利用语义结构丰富程度不一的大量文本对，借鉴单向量嵌入惯例，先做成对训练，优化一般语义相似/相关。这与先前 ColBERT 工作不同：后者通常直接从 BERT 类 PLM 出发，在 32-way 或 64-way 检索三元组（查询、正段落、多个挖出的负例）上训练。

### 5.1 Data Composition / 数据构成

**EN.** Our pair training data consists of a broad range of weakly supervised datasets harvested from the web. We adjusted sampling rates across different languages and domains based on intuition, resulting in a set of 450 million weakly supervised, semantically related sentence pairs, question-answer pairs, and query-document pairs. Of these 450 million pairs, 50.0% are in English. Our non-English pair-wise datasets contain a diverse collection of 29 major languages, including 3.0% code data, with 4.3% representing cross-lingual data.

**中.** 成对数据来自网络收获的弱监督集；按直觉调语言与领域采样率，得到 **4.5 亿** 弱监督语义相关句对、问答对、查询–文档对。其中 **50.0%** 为英文；非英文覆盖 **29** 种主要语言，含 **3.0%** 代码、**4.3%** 跨语。

### 5.2 Contrastive Loss / 对比损失

**EN.** We utilize the same single-vector pair-training loss function as described in Günther et al. (2023). Due to the often symmetric nature of our text pairs, the loss is calculated in both directions. During the pair training stage, we set the temperature $\tau=0.02$ and used a peak learning rate of $5\times 10^{-5}$ with a warm-up period of 1,000 steps. The model was trained using the Adam optimizer for 100,000 steps with a global batch size of 16,384.

**中.** 采用与 Günther et al. (2023) 相同的单向量成对对比损失；因文本对常对称，损失双向计算。温度 $\tau=0.02$，峰值学习率 $5\times 10^{-5}$，warmup **1,000** 步；Adam，**100,000** 步，全局 batch **16,384**。

---

## 6 Triplet Training / 三元组训练

### 6.1 Data Composition / 数据构成

**EN.** Our triplet dataset consists of (1) high-quality, human-annotated research datasets such as MSMARCO, DuReader, and MIRACL (Bajaj et al., 2016; He et al., 2018; Zhang et al., 2023b) with diversely mined negatives; (2) high-quality datasets like MSMARCO and NQ translated from English into Chinese, French, German, Japanese, Russian and Spanish, following our previous work (Mohr et al., 2024); and (3) synthetically generated datasets to address common failure modes of dense vector models such as negation and to cover niche domains like legal IR. The triplet dataset covers 14 widely used languages, with a strong emphasis on Arabic, Chinese, English, French, German, Japanese, Russian, and Spanish. We sample the datasets to create a language distribution similar to that used in pair training. English accounts for 45.9% of the triplets, with 52.1% roughly evenly split between the mentioned high-resource non-English languages and a small 2.0% share for lower-resource languages. Notably, owing to the limitations of our various sources of data, we train on triplets with only 7 negatives per example, in contrast to the 32- or 64-way triplets of ColBERTv2.

**中.** 三元组含：（1）MSMARCO、DuReader、MIRACL 等高质人工标注集及多样挖出的负例；（2）将 MSMARCO、NQ 等从英文译到中、法、德、日、俄、西（沿 Mohr et al., 2024）；（3）合成数据，针对否定等稠密模型常见失败模式，并覆盖法律 IR 等小众域。覆盖 **14** 种常用语言，重点阿、中、英、法、德、日、俄、西；采样使语言分布接近成对阶段。英文占 **45.9%**，上述高资源非英约占 **52.1%**（大致均分），低资源约 **2.0%**。受数据源限制，每例仅 **7** 个负例，对比 ColBERTv2 的 32-/64-way。

### 6.2 Supervision Loss / 监督损失

**EN.** Following ColBERTv2, we finetune our pair-trained checkpoint on samples with hard negatives using a KL divergence loss function to distill soft labels from the teacher model. For the teacher model, we use jina-reranker-v2-base-multilingual, a highly capable multilingual cross encoder. This stage trains for 100,000 steps with a batch size of 32 and a cosine decay learning rate schedule with 5% warm-up that peaks at $1\times 10^{-5}$. We use pure BFLOAT-16 precision, and apply magnitude-based gradient clipping with a threshold of 1 for stability.

**中.** 沿 ColBERTv2，在带难负例的样本上对成对阶段 checkpoint 做 KL 散度蒸馏，老师为 **jina-reranker-v2-base-multilingual**（强多语 cross-encoder）。本阶段 **100,000** 步，batch **32**，余弦衰减学习率、**5%** warmup、峰值 $1\times 10^{-5}$；纯 BFLOAT-16；基于幅度的梯度裁剪阈值 **1**。

---

## 7 Results / 结果

**EN.** We evaluate Jina-ColBERT-v2 on four widely used benchmarks, BEIR, LoTTE, and MIRACL and mMARCO. For general English performance, we use the same subset of 14 retrieval and text-similarity tasks from the BEIR benchmark as in Santhanam et al. (2022). Additionally, we assess performance on the LoTTE benchmark, which focuses on long-tail queries, and the MIRACL and mMARCO benchmarks (Zhang et al., 2023b; Bonifacio et al., 2022), which assess non-English retrieval performance. We report nDCG@10 for the BEIR and MIRACL collections, MRR@10 for mMARCO, and Success@5 for LoTTE. Scores are reported on the test split for BEIR, development split for MIRACL and mMARCO, and search test split for LoTTE. We use the same maximum query/document lengths as reported in Santhanam et al. (2022), and use the default (32/300) for MIRACL and mMARCO.

**中.** 在 BEIR、LoTTE、MIRACL、mMARCO 上评测。英文用与 Santhanam et al. (2022) 相同的 BEIR 14 任务子集；LoTTE 偏长尾查询；MIRACL 与 mMARCO 评非英。指标：BEIR/MIRACL 为 **nDCG@10**，mMARCO 为 **MRR@10**，LoTTE 为 **Success@5**。划分：BEIR 测集；MIRACL/mMARCO 开发集；LoTTE search 测集。最大 query/文档长度同 Santhanam et al. (2022)；MIRACL/mMARCO 用默认 **32/300**。

**Table 1.** Comparison of nDCG@10 scores between BM25, ColBERTv2, answer-colbert-small and Jina-ColBERT-v1 and Jina-ColBERT-v2 on the BEIR test set. Abbreviations: nf NFCorpus, fi FIQA, tc TREC-COVID, ar ArguAna, qu Quora, sd SciDocs, sf SciFact, to Webis-Touche, db DBpedia-Entity, fe FEVER, cf Climate-FEVER, hp HotpotQA, nq Natural Questions. Row label “Ours” denotes Jina-ColBERT-v2 as used in the paper body; “answerai-v1” denotes answerai-colbert-small-v1.

| BEIR | avg | nf | fi | tc | ar | qu | sd | sf | to | db | fe | cf | hp | nq |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 44.0 | 32.5 | 23.6 | 65.6 | 31.5 | 78.9 | 15.8 | 66.5 | 36.7 | 31.3 | 75.3 | 21.3 | 60.3 | 32.9 |
| ColBERTv2 | 49.6 | 33.7 | 35.4 | 72.6 | 46.5 | 85.5 | 15.4 | 68.9 | 26.0 | 45.2 | 78.5 | 17.6 | 67.5 | 52.4 |
| answerai-v1 | 55.7 | 37.3 | 41.2 | 84.6 | 50.1 | 87.7 | 18.4 | 74.8 | 25.7 | 45.6 | 91.0 | 33.1 | 76.1 | 59.1 |
| Ours | 53.1 | 34.6 | 40.8 | 83.4 | 36.6 | 88.7 | 18.6 | 67.8 | 27.4 | 47.1 | 80.5 | 23.9 | 76.6 | 64.0 |

**中.** 表 1：BEIR 测集 nDCG@10。相对 ColBERTv2，Jina-ColBERT-v2 英文整体更强（avg **53.1** vs **49.6**），但仍落后于单语 answerai-colbert-small-v1（**55.7**）。在 ArguAna（ar）上明显低于 ColBERTv2（**36.6** vs **46.5**），作者归因于反论检索任务与检索向三元组分布不合，或更强的 query 增强注意力在很长（300 token）查询上的局限。

**Table 2.** Comparison of Success@5 across LoTTE search query subsets.

| LoTTE | avg | Life. | Rec. | Wri. | Sci. | Tech. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 67.8 | 80.2 | 68.5 | 74.7 | 53.6 | 61.9 |
| ColBERTv2 | 72.0 | 84.7 | 72.3 | 80.1 | 56.7 | 66.1 |
| Ours | 76.4 | 87.0 | 77.6 | 83.8 | 60.5 | 73.0 |

**中.** 表 2：LoTTE Success@5；Ours avg **76.4**，高于 ColBERTv2 的 **72.0**。

**Table 3.** Comparison of nDCG@10 on the MIRACL dev set. mDPR-ZS = zero-shot; mDPR-FT = fine-tuned (per-language). “-” indicates missing in the source table.

| MIRACL | avg | ar | bn | de | es | en | fa | fi | fr | hi | id | ja | ko | ru | sw | te | th | yo | zh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 38.5 | 48.1 | 50.8 | 22.6 | 31.9 | 35.1 | 33.3 | 55.1 | 18.3 | 45.8 | 44.9 | 36.9 | 41.9 | 33.4 | 38.3 | 49.4 | 48.4 | 40.6 | 18.0 |
| mDPR-ZS | 41.8 | 49.9 | 44.3 | 49.0 | 47.8 | 39.4 | 48.0 | 47.2 | 43.5 | 38.3 | 27.2 | 43.9 | 41.9 | 40.7 | 29.9 | 35.6 | 35.8 | 39.6 | 51.2 |
| mDPR-FT | 62.7 | 72.5 | 68.4 | - | 48.8 | 56.5 | 59.3 | 71.4 | 58.9 | 51.6 | 49.6 | 64.2 | 59.0 | 59.7 | 68.5 | 80.4 | 69.5 | - | 65.0 |
| Ours | 62.3 | 75.3 | 75.0 | 50.4 | 53.8 | 57.0 | 56.3 | 74.0 | 54.1 | 60.0 | 54.7 | 63.2 | 67.1 | 64.3 | 49.9 | 74.2 | 77.2 | 62.3 | 52.3 |

**中.** 表 3：MIRACL 开发集 nDCG@10。明显超过 BM25 与零样本 mDPR；avg **62.3**，略低于按语微调的 mDPR-FT（**62.7**）。正文指出每个 mDPR-FT 只在一种语言上微调，而本模型多语同训，可能受多语诅咒影响。

**Table 4.** Comparison of MRR@10 (table caption in source: mRR@10) on the mMARCO dev set.

| mMARCO | avg | ar | de | nl | es | fr | hi | id | it | ja | pt | ru | vi | zh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM-25 | 13.9 | 11.1 | 13.6 | 14.0 | 15.8 | 15.5 | 13.4 | 14.9 | 15.3 | 14.1 | 15.2 | 12.4 | 13.6 | 11.6 |
| ColBERT-XM | 25.4 | 19.5 | 27.0 | 27.5 | 28.5 | 26.9 | 23.8 | 26.3 | 26.5 | 24.1 | 27.6 | 25.1 | 22.6 | 24.6 |
| Ours | 31.3 | 27.2 | 33.1 | 33.0 | 34.1 | 33.5 | 30.9 | 31.9 | 33.7 | 27.6 | 33.7 | 29.8 | 28.7 | 30.2 |

**中.** 表 4：相对 ColBERT-XM 的零样本 mMARCO 评测全面提升（avg **31.3** vs **25.4**），包括成对/三元组训练中未出现其 mMARCO 训练集的语言（正文写作 dt, hi, id, it, pt, vi；表内对应列为 nl 等）。

---

## 8 Ablation Studies / 消融实验

**EN.** In this section we present short ablation studies on modifications to three various aspects of ColBERT modeling and training.

**中.** 本节对 ColBERT 建模与训练的若干改动做简短消融。

### 8.1 Efficient Evaluation / 高效评测

**EN.** Due to the compute and time costs of indexing corpora containing tens of millions of documents, evaluating every model checkpoint and ablation on every task is not feasible. Therefore, we follow recent works (Clavié, 2024; Merrick et al., 2024) by comparing models’ quality on smaller sampled-corpus versions of HotpotQA, NQ, MS MARCO, and MIRACL (Chinese, French, German, Japanese, Spanish). These sampled corpora are constructed by combining the top 250 BM25-retrieved passages with all judged passages. We observe good agreement between the sampled-corpus evaluation scores and the full-fidelity ones when used to make binary or ranking-based model comparisons, but we leave a more rigorous analysis of this observation to future work. We only use the sampled corpora for ablation studies. For the final model, we evaluate on the full version of every dataset.

**中.** 因对数千万文档建索引昂贵，无法对每个 checkpoint/消融跑全任务。故在 HotpotQA、NQ、MS MARCO 与 MIRACL（中、法、德、日、西）的**采样语料**上比较：top **250** BM25 命中 ∪ 全部已判定段落。用于二值或排序式模型比较时与全量一致较好，严格分析留待未来。消融用采样；最终模型用全量。

### 8.2 Task Instructions / 任务指令

**EN.** Inspired by the use of instruction prefixes in single-vector works like Su et al. (2022), we experimented with adding task-specific natural language instructions for retrieval (RET), question answering (QA), and semantic text similarity (STS). However, results in Table 5 show a generally negative effect across most BEIR datasets. We hypothesize that this is because instructions are not well-suited for late interaction models, which operate at the token level. Any embedding conditioning that the instructions might provide likely becomes less effective when aggregated at the token similarity level. Furthermore, these instructions occupy valuable space within the system’s fixed token capacity.

**中.** 受单向量指令前缀启发，尝试为 RET / QA / STS 加自然语言任务指令。表 5 显示多数 BEIR 数据集上效果偏负。假设：迟交互在 token 级运作，指令带来的嵌入条件在 MaxSim 聚合后变弱；且指令占用固定 token 容量。

**Table 5.** nDCG@10 on BEIR: natural language instructions vs. query/document marker tokens (default). Datasets marked * use the BM25-sampled corpus (Section 8.1). Columns grouped by task type: RET (nf, tc, sf, to, db, fe, cf, ms\*), QA (fq, hp\*, nq\*), STS (ar, qu, sd).

|  | nf | tc | sf | to | db | fe | cf | ms\* | fq | hp\* | nq\* | ar | qu | sd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mark. | 32.4 | 59.3 | 67.9 | 19.3 | 35.3 | 67.1 | 18.3 | 34.4 | 37.5 | 25.9 | 40.8 | 37.5 | 86.1 | 18.4 |
| Inst. | 32.9 | 63.2 | 67.5 | 18.8 | 33.9 | 64.4 | 16.7 | 34.0 | 37.1 | 24.9 | 42.9 | 34.2 | 86.0 | 17.9 |

**中.** 表 5：标记 token（默认）相对自然语言指令，多数任务上更好或接近；指令并非迟交互的有效替代。

### 8.3 Score Normalization / 分数归一化

**EN.** Recently, Clavié (2024) applied min-max normalization to both the student and teacher scores before computing the KL loss. This adjustment brings the score distributions of the ColBERT model and its CE teacher into closer alignment, as the original score distribution for ColBERT theoretically ranges from zero to the number of query tokens, and is model-dependent for the teacher CE. Our experiment presented in Table 6, however, shows this method to have inconclusive benefit to nDCG@10 on the BEIR and MIRACL datasets when applied to our model. We consider this result to be understandable given Clavié (2024)’s very small observed effect.

**中.** Clavié (2024) 在算 KL 前对学生与老师分数做 min-max 归一化，使 ColBERT（理论范围约 0 到查询 token 数）与 CE 老师的分数量级对齐。表 6 显示对本模型 nDCG@10 收益不确定；与 Clavié 观测到的很小效应一致，可理解。

**Table 6.** nDCG@10 with/without score normalization on a retrieval-oriented BEIR & MIRACL subset. All except TREC-COVID (tc) use BM25-sampled corpora.

|  | tc | hp | nq | ms | de | es | fr | ja | zh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 78.7 | 36.6 | 58.0 | 45.4 | 57.3 | 40.6 | 50.7 | 63.4 | 63.2 |
| + Score Norm. | 80.1 | 36.4 | 56.6 | 45.1 | 57.7 | 39.3 | 51.3 | 61.8 | 62.5 |

### 8.4 Query Augmentation Attention / 查询增强注意力

**EN.** An important feature of ColBERT’s implementation is its query augmentation mechanism. By padding queries with [MASK] tokens to a uniform length, ColBERT uses BERT’s masked language modeling ability to produce additional soft term embeddings which interact with document token embeddings during MaxSim scoring. However, prior ColBERT models do not modify the attention mask to allow query tokens to attend to the mask tokens, which some hypothesize might harm generalization by making this augmentation feature too integral to the embedding process. Our controlled triplet training experiment in Table 7, however, demonstrates a positive effect across a variety of tasks, with particular benefit to non-English tasks in MIRACL. We therefore allow this attention in our training and inference.

**中.** ColBERT 的重要实现细节是查询增强：用 [MASK] 把查询 pad 到统一长度，利用 MLM 能力产生额外「软词项」嵌入，在 MaxSim 中与文档 token 交互。先前 ColBERT **不**改注意力 mask，不允许查询 token attend 到这些 mask token；有人担心若允许会让增强过度嵌入表示、伤害泛化。表 7 的受控三元组实验显示跨任务正向效果，对 MIRACL 非英任务尤其明显。故训练与推理均允许该注意力。

**Table 7.** nDCG@10 with/without query-augmentation [MASK] token attention (full-fidelity scores).

|  | tc | hp | nq | ms | de | es | fr | ja | zh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 77.2 | 70.4 | 54.6 | 37.6 | 33.3 | 40.3 | 35.9 | 54.9 | 34.4 |
| + [MASK] attn. | 80.2 | 71.5 | 58.8 | 44.3 | 45.6 | 49.8 | 44.8 | 58.8 | 52.9 |

---

## 9 Conclusion / 结论

**EN.** This work presents Jina-ColBERT-v2, a capable multilingual ColBERT model that is the result of improvements to its architecture and training process. We implement modifications to the model architecture that yield efficiency gains with effectively no downside, and subsequently train it on a heterogeneous mix of data of varying tasks, languages, and supervision structures in order to bolster its performance as a general purpose retriever. Our ablation experiments demonstrate the sensitivity of ColBERT to modifications to its representations. We hope that our work will support future multilingual ColBERT development, and prompt further exploration into the properties and optimal configuration of its query augmentation mechanism. We are also encouraged by the many inference-only optimization works on ColBERT representations, and suggest further effort be invested in tying these methods more closely with the model’s training objective.

**中.** 本文给出 **Jina-ColBERT-v2**：有能力的多语 ColBERT，来自架构与训练改进。架构改动带来几乎无代价的效率收益；再在任务、语言、监督结构异构的混合数据上训练，强化通用检索。消融显示 ColBERT 对其表示改动敏感。希望支持未来多语 ColBERT，并推动对查询增强机制性质与最优配置的探索；亦鼓励把大量「仅推理」的 ColBERT 表示优化与训练目标更紧地结合。

---

## 10 Acknowledgement / 致谢

**EN.** We thank Qi Liu and Jiaxin Mao from Renmin University of China for the contributions to Jina-ColBERT-v1 and offer the insights about MRL over MRL-E for ColBERT models.

**中.** 感谢中国人民大学刘琦、毛佳鑫对 Jina-ColBERT-v1 的贡献，以及关于 ColBERT 上 MRL 优于 MRL-E 的见解。

---

## References / 参考文献（主要条目）

- Bajaj et al., 2016. [MS MARCO](https://microsoft.github.io/msmarco/)
- Bonifacio et al., 2022. [mMARCO, arXiv:2108.13897](https://arxiv.org/abs/2108.13897)
- Chen et al., 2024. [BGE M3-Embedding, arXiv:2402.03216](https://arxiv.org/abs/2402.03216)
- Clavié, 2024. [JaColBERTv2.5, arXiv:2407.20750](https://arxiv.org/abs/2407.20750)
- Conneau et al., 2020. [XLM-R, arXiv:1911.02116](https://arxiv.org/abs/1911.02116)
- Dao, 2024. FlashAttention-2 (ICLR)
- Devlin et al., 2019. [BERT, arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Formal et al., 2021. [SPLADE, arXiv:2107.05720](https://arxiv.org/abs/2107.05720)
- Günther et al., 2023. [Jina Embeddings 2, arXiv:2310.19923](https://arxiv.org/abs/2310.19923)
- He et al., 2018. DuReader
- Karpukhin et al., 2020. DPR (EMNLP)
- Khattab and Zaharia, 2020. [ColBERT, arXiv:2004.12832](https://arxiv.org/abs/2004.12832)
- Kusupati et al., 2022. Matryoshka Representation Learning (NeurIPS)
- Lawrie et al., 2023. [Neural Approaches to Multilingual IR, arXiv:2209.01335](https://arxiv.org/abs/2209.01335)
- Lee et al., 2024. [Gecko, arXiv:2403.20327](https://arxiv.org/abs/2403.20327)
- Li et al., 2023. Towards general text embeddings with multi-stage contrastive learning
- Louis et al., 2024. [ColBERT-XM, arXiv:2402.15059](https://arxiv.org/abs/2402.15059)
- Mohr et al., 2024. [Multi-task contrastive bilingual embeddings, arXiv:2402.17016](https://arxiv.org/abs/2402.17016)
- Muennighoff et al., 2023. MTEB (EACL)
- Nair et al., 2022. Transfer learning for cross-language dense retrieval (ECIR)
- Penedo et al., 2023. [RefinedWeb, arXiv:2306.01116](https://arxiv.org/abs/2306.01116)
- Santhanam et al., 2022. [ColBERTv2, arXiv:2112.01488](https://arxiv.org/abs/2112.01488)
- Su et al., 2022. [One Embedder, Any Task, arXiv:2212.09741](https://arxiv.org/abs/2212.09741)
- Su et al., 2023. RoFormer / RoPE
- Wang et al., 2023. [Improving text embeddings with LLMs, arXiv:2401.00368](https://arxiv.org/abs/2401.00368)
- Wang et al., 2024. [Multilingual E5, arXiv:2402.05672](https://arxiv.org/abs/2402.05672)
- Zhang et al., 2021. [Mr. TyDi, arXiv:2108.08787](https://arxiv.org/abs/2108.08787)
- Zhang et al., 2023a. Toward best practices for training multilingual dense retrieval (TOIS)
- Zhang et al., 2023b. MIRACL (TACL)
- Teacher model: [jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)
