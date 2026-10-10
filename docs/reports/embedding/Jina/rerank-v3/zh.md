> 原文: [arXiv:2509.25085](https://arxiv.org/abs/2509.25085)（2025-09；HTML 版本标注 2025-10-06）
> 说明: 中英对照供阅读。英文为论文原文（HTML 转写，公式按论文整理）；中文为对应翻译。图表编号与原文一致。caption 保留英文并附中译。数值原样。

**预印本：** arXiv:2509.25085v4 [cs.CL]，2025 年 10 月 6 日。License: CC BY-NC-SA 4.0。

**权重：** https://huggingface.co/jinaai/jina-reranker-v3

---

# jina-reranker-v3: Last but Not Late Interaction for Listwise Document Reranking

# jina-reranker-v3：列表级文档精排中的「最后但非迟」交互

**Authors / 作者：** Feng Wang, Yuqing Li, Han Xiao

**Affiliation / 单位：** Jina AI GmbH · University of Pittsburgh（Yuqing Li）· Prinzessinnenstraße 19, 10969, Berlin, Germany · research@jina.ai

---

## Abstract / 摘要

**EN.** jina-reranker-v3 is a 0.6B-parameter multilingual listwise reranker that introduces a novel last but not late interaction. Unlike late interaction models like ColBERT that encode documents separately before multi-vector matching, our approach applies causal attention between the query and all candidate documents in the same context window, enabling rich interactions before extracting contextual embeddings from each document’s final token. The new model achieves state-of-the-art BEIR performance with 61.85 nDCG@10 while being significantly smaller than other models with comparable performance.

**中.** jina-reranker-v3 是一个 0.6B 参数的多语列表级精排器，提出新颖的 **last but not late interaction（最后但非迟交互，LBNL）**。与 ColBERT 等先分别编码文档再做多向量匹配的迟交互模型不同，本方法在同一上下文窗口内对查询与全部候选文档施加因果注意力，在从每篇文档的最终 token 抽取上下文嵌入之前就能发生丰富交互。新模型在 BEIR 上以 61.85 nDCG@10 达到当时最强表现，同时显著小于性能相当的其他模型。

---

## 1 Introduction / 引言

**EN.** Neural document retrieval faces a fundamental efficiency-effectiveness tradeoff. Cross-encoders achieve strong performance through joint query-document processing but require separate forward passes for each pair, while embedding models enable efficient similarity computation but lose fine-grained interaction signals. Recent models have attempted to bridge this gap through different interaction approaches. Late interaction models like ColBERT (Khattab and Zaharia, 2020) and their variants (Liu et al., 2024; Jha et al., 2024) separately encode queries and documents into multi-vector representations, then perform interaction through token-level similarity operations.

**中.** 神经文档检索面临根本的效率–效果权衡。Cross-encoder 通过联合处理查询与文档取得强表现，但每对都需要单独前向；嵌入模型能高效算相似度，却丢掉细粒度交互信号。近期模型用不同交互方式试图弥合这一鸿沟。ColBERT（Khattab and Zaharia, 2020）及其变体（Liu et al., 2024；Jha et al., 2024）等迟交互模型分别把查询与文档编成多向量表示，再通过 token 级相似度运算做交互。

**EN.** We introduce jina-reranker-v3, which features a novel last but not late interaction (LBNL) that takes a fundamentally different approach from existing methods. While late interaction models delay attention until after encoding documents separately, our method applies causal attention between the query and all documents within the context window, enabling cross-document interactions before extracting contextual embeddings from each document’s last token. Unlike late interaction models that interact after encoding, we enable interactions during encoding—making our approach not late. This “listwise” processing is not possible with separate encoding or bi-encoder approaches and represents our core innovation.

**中.** 我们提出 jina-reranker-v3，其核心是与既有方法根本不同的 **LBNL（last but not late interaction）**。迟交互模型把注意力推迟到文档分别编码之后；我们的方法在上下文窗口内对查询与全部文档施加因果注意力，在抽取每篇文档末 token 的上下文嵌入之前就允许跨文档交互。交互发生在编码过程中，因而「不迟」。这种「列表级」处理在分别编码或双塔方案中不可行，是本文的核心创新。

**EN.** Evaluation shows jina-reranker-v3 achieves 61.85 nDCG@10 on BEIR (Thakur et al., 2021), representing the highest score among all evaluated rerankers and a 4.79% improvement over our previous jina-reranker-v2. The model excels particularly in multi-hop retrieval with HotpotQA reaching 78.58, fact verification achieving 94.01 on FEVER, competitive multilingual performance across 18 languages at 66.83 on MIRACL (Zhang et al., 2023a) and crosslingual retrieval with 67.92 Recall@10 on MKQA (Longpre et al., 2020) across 26 languages, and code retrieval reaching 70.64 on CoIR (Li et al., 2024).

**中.** 评测显示 jina-reranker-v3 在 BEIR（Thakur et al., 2021）上达到 61.85 nDCG@10，为所评精排中最高，相对前作 jina-reranker-v2 提升 4.79%。多跳检索 HotpotQA 达 78.58，事实验证 FEVER 达 94.01；18 语 MIRACL（Zhang et al., 2023a）为 66.83；跨语 MKQA（Longpre et al., 2020，26 语）Recall@10 为 67.92；代码检索 CoIR（Li et al., 2024）为 70.64。

---

## 2 Related Work / 相关工作

**EN.** Document reranking approaches can be categorized by their interactions and learning objectives. Traditional learning-to-rank methods (Bruch et al., 2023) include pointwise approaches that predict relevance scores independently, pairwise methods like RankNet (Burges et al., 2005) that compare document pairs, and listwise techniques that optimize global ranking objectives. Cross-encoders like BERT-based rerankers (Nogueira and Cho, 2019) achieve strong performance through full query-document interaction but require separate forward passes for each pair, creating computational bottlenecks for large-scale retrieval. Recent comparative studies (Déjean et al., 2024) demonstrate that while LLM-based rerankers show impressive zero-shot capabilities, traditional cross-encoders remain highly competitive across diverse retrieval scenarios.

**中.** 文档精排可按交互方式与学习目标分类。传统学习排序（Bruch et al., 2023）包括独立预测相关性的 pointwise、如 RankNet（Burges et al., 2005）的 pairwise，以及优化全局排序目标的 listwise。基于 BERT 的 cross-encoder（Nogueira and Cho, 2019）通过完整查询–文档交互表现强，但每对需单独前向，大规模检索形成瓶颈。近期对比研究（Déjean et al., 2024）表明：虽 LLM 精排零样本能力突出，传统 cross-encoder 在多样检索场景中仍极具竞争力。

**EN.** Late interaction models represent a significant approach that balances efficiency with expressiveness. ColBERT (Khattab and Zaharia, 2020) exemplifies this approach by independently encoding queries and documents into multi-vector representations, then computing similarity through MaxSim operations over token-level embeddings. This design enables pre-computation of document representations while preserving fine-grained matching signals. Recent developments have expanded this approach: analysis of matching mechanisms and token pruning strategies (Liu et al., 2024) provides theoretical foundations, LITE (Ji et al., 2024) introduces learnable late interactions, and Jina-ColBERT-v2 (Jha et al., 2024) extends the approach to multilingual settings. PyLate (Chaffin and Sourty, 2025) provides flexible frameworks for training and deployment of such models. The late chunking method (Günther et al., 2024) processes complete documents through transformers before applying chunking boundaries, extracting chunk-level embeddings that preserve contextual relationships. This approach demonstrates how leveraging broader document context can improve embedding quality, though it focuses primarily on retrieval rather than reranking applications.

**中.** 迟交互模型在效率与表达力之间取得重要平衡。ColBERT（Khattab and Zaharia, 2020）独立编码查询与文档为多向量，再用 token 级嵌入上的 MaxSim 算相似度；文档表示可预计算，同时保留细粒度匹配。后续包括匹配机制与 token 剪枝分析（Liu et al., 2024）、可学习迟交互 LITE（Ji et al., 2024）、多语扩展 Jina-ColBERT-v2（Jha et al., 2024），以及训练部署框架 PyLate（Chaffin and Sourty, 2025）。late chunking（Günther et al., 2024）先整篇过 Transformer 再切块，抽取保留语境关系的块级嵌入；主要面向检索而非精排。

**EN.** LLM-powered reranker has emerged as a powerful family with diverse implementations. These approaches can be categorized into discriminative and generative methods. Generative approaches like RankGPT (Qin et al., 2023) prompt LLMs to generate ranked lists, leveraging their reasoning capabilities for relevance assessment, but typically require large models for competitive performance. Fine-tuning methods like RankVicuna (Pradeep et al., 2023) adapt existing models for relevance scoring tasks. Efficiency-focused innovations include FIRST (Reddy et al., 2024), which accelerates inference through single-token decoding, and PE-Rank (Qin et al., 2024), which leverages passage embeddings to reduce computational latency by $4.5\times$. Recent advances in training methodology include ERank (Cai et al., 2025), which combines supervised fine-tuning with reinforcement learning for improved ranking quality, and the Qwen3 Embedding series (Zhang et al., 2025), which demonstrates sophisticated multi-stage training pipelines. DeAR (Abdallah et al., 2025) introduces dual-stage reasoning with LLM distillation for enhanced cross-document analysis.

**中.** 基于 LLM 的精排亦可分判别式与生成式。RankGPT（Qin et al., 2023）等生成式让 LLM 产出排序列表，通常需大模型才有竞争力。RankVicuna（Pradeep et al., 2023）等微调做相关性打分。效率向创新包括单 token 解码加速的 FIRST（Reddy et al., 2024），以及用段落嵌入把延迟降约 $4.5\times$ 的 PE-Rank（Qin et al., 2024）。训练方法上有 SFT 与强化学习结合的 ERank（Cai et al., 2025）、多阶段管线的 Qwen3 Embedding 系列（Zhang et al., 2025），以及双阶段推理与 LLM 蒸馏的 DeAR（Abdallah et al., 2025）。

---

## 3 Model Architecture / 模型架构

**EN.** jina-reranker-v3 implements a new interaction that fundamentally differs from existing approaches. Built upon Qwen3-0.6B (Yang et al., 2025) with 28 transformer layers, 1024 hidden dimensions, 16 attention heads, and 131K token context capacity, our approach processes queries and multiple documents simultaneously within shared context windows. We add a lightweight MLP projector (1024 → 512 → 512 dimensions) to transform contextual representations into ranking-optimized embeddings. Table 5 provides complete architectural specifications.

**中.** jina-reranker-v3 实现与既有方法根本不同的新交互。骨干为 Qwen3-0.6B（Yang et al., 2025）：28 层 Transformer、隐维 1024、16 个注意力头、上下文容量 131K token；查询与多篇文档在共享上下文窗口中同时处理。另加轻量 MLP 投影器（1024 → 512 → 512）把上下文表示映射为面向排序的嵌入。完整架构规格见 Table 5。

### 3.1 Architecture / 架构

![Figure 1: Architecture of jina-reranker-v3 showing the transformer backbone with special token positions for embedding extraction. The model processes multiple documents and query in one context window, extracting contextual embeddings at designated token positions for similarity computation.](figs/fig01.png)

**Caption EN.** Figure 1: Architecture of jina-reranker-v3 showing the transformer backbone with special token positions for embedding extraction. The model processes multiple documents and query in one context window, extracting contextual embeddings at designated token positions for similarity computation.

**Caption 中.** 图 1：jina-reranker-v3 架构——Transformer 骨干与用于嵌入抽取的特殊 token 位置。模型在同一上下文窗口中处理多篇文档与查询，在指定 token 位置抽取上下文嵌入再算相似度。

**EN.** Figure 1 illustrates our architecture that addresses fundamental limitations in existing interactions. ColBERT (Khattab and Zaharia, 2020) achieves efficiency through separate encoding followed by multi-vector interaction, but cannot capture early query-document interactions during encoding or enable cross-document interactions within the attention mechanism.

**中.** 图 1 说明本架构如何应对既有交互的根本局限。ColBERT 靠分别编码再多向量交互换效率，但无法在编码阶段捕获早期查询–文档交互，也无法在注意力机制内做跨文档交互。

**EN.** Our LBNL approach enables causal self-attention interaction within the transformer architecture: instead of delaying interaction until after separate encoding as in late interaction models, we process all documents and the query simultaneously within shared context windows. This allows each document to attend to other documents and observe their content, enabling contextual embeddings that capture not just query-document relevance but also inter-document relationships and comparative context. Such cross-document interactions are impossible in separate encoding approaches and represent a fundamental advancement in jina-reranker-v3 architecture.

**中.** LBNL 在 Transformer 内做因果自注意力交互：不再像迟交互那样把交互推迟到分别编码之后，而是在共享上下文中同时处理全部文档与查询。每篇文档可 attend 其他文档并观察其内容，上下文嵌入不仅刻画查询–文档相关性，也刻画文档间关系与对照语境。分别编码方案无法做到这种跨文档交互，是 jina-reranker-v3 架构的根本推进。

**EN.** We extract contextual embeddings at designated special token positions: $\tilde{\mathbf{q}}=\mathbf{H}_{t_{q}}$ and $\tilde{\mathbf{d}}_{i}=\mathbf{H}_{t_{i}}$ where $t_{q}$ and $t_{i}$ are positions of the special tokens and $\mathbf{H}$ represents the transformer’s final layer hidden states after causal self-attention. These embeddings capture both local document semantics and global cross-document context through the shared attention mechanism, enabling rich inter-document interactions unavailable in separate encoding approaches.

**中.** 在指定特殊 token 位置抽取上下文嵌入：$\tilde{\mathbf{q}}=\mathbf{H}_{t_{q}}$，$\tilde{\mathbf{d}}_{i}=\mathbf{H}_{t_{i}}$，其中 $t_{q}$、$t_{i}$ 为特殊 token 位置，$\mathbf{H}$ 为因果自注意力后 Transformer 最后一层隐状态。共享注意力使这些嵌入同时携带局部文档语义与全局跨文档语境。

**EN.** A two-layer projection network with ReLU activation maps the 1024-dimensional hidden states to 256-dimensional embedding space: $\mathbf{q}=P_{\phi}(\tilde{\mathbf{q}})$ and $\mathbf{d}_{i}=P_{\phi}(\tilde{\mathbf{d}}_{i})$. Relevance scores are computed via cosine similarity: $s_{i}=\cos(\mathbf{q},\mathbf{d}_{i})$. This architecture combines the expressiveness of joint encoding with efficient similarity computation.

**中.** 带 ReLU 的两层投影网络将 1024 维隐状态映射到 **256 维**嵌入空间：$\mathbf{q}=P_{\phi}(\tilde{\mathbf{q}})$，$\mathbf{d}_{i}=P_{\phi}(\tilde{\mathbf{d}}_{i})$；相关性用余弦 $s_{i}=\cos(\mathbf{q},\mathbf{d}_{i})$。该表述与正文开篇及 Table 5 给出的投影器规格 **1024 → 512 → 512** 不一致；本对照译文按原文保留「256 维」一句，规格以 Table 5 为准（见附录）。

**EN.** For document collections exceeding the 131K token context limit, we process documents in batches of up to 64 documents per forward pass, with query embeddings maintained consistently across batches to ensure ranking coherence.

**中.** 当文档集合超过 131K token 上下文上限时，按每批最多 64 篇文档做前向，并在各批之间保持查询嵌入一致，以保证排序连贯。

### 3.2 Prompt Template / 提示模板

**EN.** jina-reranker-v3 processes structured prompts following Qwen3’s instruction format with system / user / assistant roles to leverage existing instruction-following capabilities. As shown in Table 1, the system prompt establishes a search relevance expert persona, while the user prompt provides clear ranking instructions with dual query placement.

**中.** 模型按 Qwen3 的 system / user / assistant 指令格式处理结构化提示。如表 1：system 设定检索相关性专家角色；user 给出清晰排序指令，并采用双重查询放置。

**Table 1 / 表 1：** Complete prompt template structure used by jina-reranker-v3. Special tokens `<|doc_emb|>` and `<|query_emb|>` mark positions for embedding extraction from transformer hidden states. / jina-reranker-v3 完整提示模板。特殊 token `<|doc_emb|>` 与 `<|query_emb|>` 标记从 Transformer 隐状态抽取嵌入的位置。

```
<|im_start|>system
You are a search relevance expert who can determine
a ranking of passages based on their relevance to the query.
<|im_end|>
<|im_start|>user
I will provide you with k passages, each indicated by a numerical identifier.
Rank the passages based on their relevance to query: [QUERY]
<passage id="1">
[DOCUMENT_1] <|doc_emb|>
</passage>
<passage id="2">
[DOCUMENT_2] <|doc_emb|>
</passage>
...
<passage id="k">
[DOCUMENT_k] <|doc_emb|>
</passage>
<query>
[QUERY] <|query_emb|>
</query>
<|im_end|>
```

**EN.** The template strategically places the query both at the beginning for instructions and at the end for final attention, sandwiching all documents in between. This design enables the final query position to attend to all preceding documents through causal attention while maintaining clear task instructions. Special tokens `<|doc_emb|>` after each document and `<|query_emb|>` after the final query mark specific positions for embedding extraction from transformer hidden states.

**中.** 查询既出现在开头指令中，又出现在末尾供最终注意力使用，全部文档夹在中间。因果注意力下，末尾查询位置可 attend 到此前所有文档，同时指令仍清晰。每篇文档后的 `<|doc_emb|>` 与末尾查询后的 `<|query_emb|>` 标记嵌入抽取位置。

---

## 4 Training / 训练

### 4.1 Loss Functions / 损失函数

**EN.** jina-reranker-v3 employs a comprehensive multi-objective training approach combining InfoNCE loss with specialized auxiliary losses to optimize ranking performance across diverse domains.

**中.** 训练采用多目标：以 InfoNCE 为主，辅以专门辅助损失，优化跨域排序。

**EN.** The core training objective integrates multiple loss components, each addressing distinct aspects of the ranking problem:

**中.** 核心目标整合多项损失，各应对排序问题的不同侧面：

$$
\ell=\ell_{\mathrm{rank}}+0.45\cdot\ell_{\mathrm{disperse}}+0.85\cdot\ell_{\mathrm{dual}}+0.85\cdot\ell_{\mathrm{similar}}
\tag{1}
$$

**EN.** The primary component is the InfoNCE loss $\ell_{\mathrm{rank}}$ (van den Oord et al., 2019), which generates the core ranking signal through contrastive learning with hard negatives:

**中.** 主项为 InfoNCE 损失 $\ell_{\mathrm{rank}}$（van den Oord et al., 2019）：用难负例对比学习产生核心排序信号。

$$
\ell_{\mathrm{rank}}=-\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{s(\mathbf{q}_{i},\mathbf{d}_{i}^{+})/\tau}}{Z_{i}}
\quad\text{where}\quad
Z_{i}=e^{s(\mathbf{q}_{i},\mathbf{d}_{i}^{+})/\tau}+\sum_{k=1}^{K}e^{s(\mathbf{q}_{i},\mathbf{d}_{i,k}^{-})/\tau}
\tag{2}
$$

**EN.** Here, $\mathbf{q}_{i}$ denotes the query embedding, $\mathbf{d}_{i}^{+}$ represents the positive document embedding, $\mathbf{d}_{i,k}^{-}$ denotes one of $K$ negative document embeddings, $s(\cdot,\cdot)$ is the cosine similarity function, $\tau$ is the temperature parameter, and $N$ is the batch size.

**中.** 其中 $\mathbf{q}_{i}$ 为查询嵌入，$\mathbf{d}_{i}^{+}$ 为正文档嵌入，$\mathbf{d}_{i,k}^{-}$ 为 $K$ 个负文档之一，$s(\cdot,\cdot)$ 为余弦相似度，$\tau$ 为温度，$N$ 为 batch size。

**EN.** To prevent representation collapse, we incorporate the dispersive loss $\ell_{\mathrm{disperse}}$ (Wang et al., 2024), which enhances embedding diversity by maximizing the average pairwise cosine distance between document embeddings:

**中.** 为防止表示坍缩，引入分散损失 $\ell_{\mathrm{disperse}}$（Wang et al., 2024）：通过最大化文档嵌入间平均成对余弦距离增强多样性。

$$
\ell_{\mathrm{disperse}}=\frac{1}{N}\sum_{i=1}^{N}\log\frac{1}{K}\sum_{k=1}^{K}\left(e^{s(\mathbf{d}_{i}^{+},\mathbf{d}_{i,k}^{-})/\tau}+\sum_{k'=k}^{K-1}e^{s(\mathbf{d}_{i,k}^{-},\mathbf{d}_{i,k'+1}^{-})/\tau}\right)
\tag{3}
$$

**EN.** The dual matching loss $\ell_{\mathrm{dual}}$ follows the same formulation as Eq. 2 but computes the query embedding from the query tokens at the sequence start. (Footnote: During training, the special token `<|query_emb|>` is inserted at the end of the query at the beginning of the input sequence.) This enforces bidirectional consistency between query-to-document and document-to-query similarity scores, enhancing ranking robustness.

**中.** 双匹配损失 $\ell_{\mathrm{dual}}$ 与式 (2) 同形，但从序列**开头**的查询 token 取查询嵌入。（脚注：训练时在输入序列开头的查询末尾也插入特殊 token `<|query_emb|>`。）由此强制查询→文档与文档→查询相似度双向一致，增强排序稳健性。

**EN.** Finally, the similarity loss $\ell_{\mathrm{similar}}$ (Huang et al., 2024) maintains semantic coherence at the document level. For each document in the input set, we create an augmented duplicate $\mathbf{d}_{i}^{*}$ through text augmentation techniques. The loss then treats the original document and its augmented version as a positive pair, while other documents serve as negatives. This encourages consistent embedding representations for semantically equivalent documents, even when their surface forms differ due to augmentation.

**中.** 相似度损失 $\ell_{\mathrm{similar}}$（Huang et al., 2024）保持文档级语义一致性：对输入集合中每篇文档做文本增强得到副本 $\mathbf{d}_{i}^{*}$，原文档与增强版为正对，其余为负，促使语义等价、表面形式不同的文档嵌入一致。

### 4.2 Multi-Stage Training / 多阶段训练

**EN.** The training methodology follows a progressive three-stage approach designed for systematic complexity scaling:

**中.** 训练采用递进三阶段，系统放大复杂度：

**EN.** **Stage 1: Foundation Specialization.** Starting from pretrained Qwen3-0.6B, we simultaneously train domain-specific configurations using LoRA fine-tuning with $r$=16 and $\alpha$=32 targeting all attention and FFN layers while freezing the backbone. The model processes training sequences containing 16 documents per query (one positive and 15 negative examples), with each document truncated or padded to 768 tokens, yielding a maximal total sequence length of 12,288 tokens. Training data is drawn from diverse datasets including BGE-M3 (Chen et al., 2024) for multilingual coverage across 15 languages, Cornstack (Suresh et al., 2025) for code retrieval, as well as specialized datasets for biomedical (Xu et al., 2024) and instruction following (Weller et al., 2024) configurations.

**中.** **阶段 1：基础特化。** 从预训练 Qwen3-0.6B 出发，同时训练多个领域配置：LoRA（$r$=16，$\alpha$=32）作用于全部注意力与 FFN 层，骨干冻结。每查询 16 篇文档（1 正 + 15 负），每篇截断/填充至 768 token，最大总序列长 12,288。数据含 BGE-M3（Chen et al., 2024，15 语）、Cornstack（Suresh et al., 2025，代码）、生物医学（Xu et al., 2024）与指令跟随（Weller et al., 2024）等。

**EN.** **Stage 2: Context and Hard Negative Mining.** This stage combines context extension and comprehensive robustness optimization. Context extension is implemented in two ways: (1) extending individual document length to 8,192 tokens through datasets like MLDR (Chen et al., 2024) for long-document understanding, and (2) increasing the number of negative documents from 15 to 45 per query while maintaining the total sequence length under 131K tokens. Simultaneously, cross-system hard negative mining ensures robustness through specialized optimizations including jina-en-v2 for English performance, miracl-v2 for multilingual retrieval, cornstack-v2 for code understanding, and context-chunk-v3 for long-document processing. Training systematically mines hard negatives across multiple retrieval systems including BGE, Jina, GTE, and E5-Large with up to 25 negatives per query and very low temperature of 0.05, using key datasets including MS-MARCO (Campos et al., 2016), mMARCO (Bonifacio et al., 2021), and domain-specific synthetic question-answer pairs.

**中.** **阶段 2：上下文与难负例挖掘。** 结合上下文扩展与稳健性优化。扩展两条路：（1）用 MLDR（Chen et al., 2024）等把单篇文档长度扩至 8,192 token；（2）每查询负文档数由 15 增至 **45**，总序列长仍低于 131K。同时跨系统难负例挖掘，并有 jina-en-v2、miracl-v2、cornstack-v2、context-chunk-v3 等特化优化。训练从 BGE、Jina、GTE、E5-Large 等多检索系统系统挖难负例，每查询最多 **25** 负、温度低至 0.05；关键数据含 MS-MARCO、mMARCO 与领域合成问答对。（注：正文写负例 15→45，附录 Table 6 写 Stage 2 Number of Negatives 为 [9, 25]；两处均按原文保留，不作调和。）

**EN.** **Stage 3: Model Ensemble and Optimization.** The final stage combines multiple specialized models trained in previous stages through linear model merging. Each domain-specific model contributes weighted expertise, with merge weights ranging from 0.25 to 0.65 based on domain importance and performance. This approach enables the final model to leverage diverse domain knowledge while maintaining architectural efficiency.

**中.** **阶段 3：模型集成与优化。** 对前两阶段多个特化模型做线性合并；各域权重按重要性与表现取 **0.25–0.65**，在保持架构效率的同时融合多域知识。

**EN.** Detailed hyperparameter evolution across stages demonstrates multi-objective optimization with stage-tailored configurations (see Appendix Table 6). Foundation stages use aggressive learning rates of 5e-5 with substantial negative sampling of 15 negatives. Context scaling stages reduce batch sizes dramatically from 60 to 6 to accommodate 8K sequences while employing conservative learning rates of 6e-6. Loss weight adaptation varies across different domain specializations, with dispersive loss typically set to 0.45, dual-matching loss ranging from 0.65 to 0.85, and similarity loss stabilizing around 0.75 to 0.85 depending on the specific domain requirements.

**中.** 各阶段超参见附录 Table 6。基础阶段学习率 5e-5、15 负例；上下文扩展阶段 batch 由 60 大幅降至 6 以容纳 8K 序列，学习率保守至 6e-6。损失权重随域特化变化：分散损失通常 0.45，双匹配 0.65–0.85，相似度约 0.75–0.85。

---

## 5 Evaluation / 评测

### 5.1 Experimental Setup / 实验设置

**EN.** Our evaluation spans four challenging benchmarks that test different aspects of ranking capability. BEIR (Thakur et al., 2021) represents the gold standard for English retrieval evaluation, encompassing 13 heterogeneous tasks from question answering on Natural Questions to fact verification on FEVER, testing the model’s ability to generalize across domains without task-specific optimization. MIRACL (Zhang et al., 2023b) pushes multilingual boundaries with 18 languages spanning diverse linguistic families, from Arabic and Chinese to Finnish and Thai, requiring deep cross-lingual understanding. MKQA (Longpre et al., 2021) specifically challenges cross-lingual question answering capabilities, while CoIR (Li et al., 2025) focuses on the specialized domain of code retrieval, where semantic understanding of programming constructs becomes crucial.

**中.** 评测覆盖四个基准：BEIR（13 个异构英语检索任务）、MIRACL（18 语）、MKQA（跨语问答）、CoIR（代码检索）。

**EN.** The first-stage dense retriever is jina-embeddings-v3, providing the foundation top-100 candidates that all rerankers process. Second-stage rerankers encompass our previous jina-reranker-v2, the multilingual bge-reranker-v2-m3, the mxbai-rerank variants at different scales, and Qwen3-Reranker-0.6B and Qwen3-Reranker-4B models.

**中.** 一阶段稠密检索为 jina-embeddings-v3，给出全体精排共用的 top-100 候选。（代码检索一阶段为 jina-code-embeddings-0.5b，见 Table 2。）二阶段精排包括 jina-reranker-v2、bge-reranker-v2-m3、mxbai-rerank 各尺度，以及 Qwen3-Reranker-0.6B 与 Qwen3-Reranker-4B。

### 5.2 Overall Performance Across Benchmarks / 跨基准总体表现

**EN.** Table 2 demonstrates jina-reranker-v3’s exceptional performance density across diverse evaluation scenarios. On BEIR, our model achieves the highest score among all rerankers at 61.85, establishing new state-of-the-art performance for English retrieval. This represents a 4.79% improvement over our previous jina-reranker-v2 at 57.06, directly attributable to LBNL interaction mechanism where query and document embeddings are extracted from shared forward passes rather than separate encoding pipelines.

**中.** Table 2 显示跨场景表现密度。BEIR 上 61.85 为所评精排最高；相对 jina-reranker-v2 的 57.06 提升 4.79%，归因于 LBNL：查询与文档嵌入来自共享前向而非分别编码管线。

**EN.** Parameter efficiency analysis reveals striking advantages compared to larger alternatives. Against the 1.5B parameter mxbai-rerank-large-v2, jina-reranker-v3 achieves superior BEIR performance with 61.85 versus 61.44 using $2.5\times$ fewer parameters, while providing specialized domain coverage unavailable in competing models reaching 70.64 on CoIR. This efficiency derives from architectural innovations: Qwen3’s optimized transformer backbone combined with our specialized 512-dimensional projector network that concentrates ranking signals without requiring massive parameter scaling.

**中.** 相对 1.5B 的 mxbai-rerank-large-v2（BEIR 61.44），本模型以约 $2.5\times$ 更少参数达到 61.85，并在 CoIR 上达 70.64。效率来自 Qwen3 骨干与 **512 维投影网络**（与 §3.1「256 维」表述并存于原文；此处按正文参数效率段译）。

**EN.** Multilingual evaluation reveals strong cross-lingual capabilities despite the model’s compact architecture. The 66.83 score on MIRACL, while 2.49 points below the multilingual-specialized bge-reranker-v2-m3 at 69.32, demonstrates effective knowledge transfer from our progressive training methodology. The 67.92 MKQA performance closely approaches jina-reranker-m0’s 68.19, indicating that architectural sophistication can partially offset parameter differences in multilingual scenarios.

**中.** MIRACL 66.83，比多语特化的 bge-reranker-v2-m3（69.32）低 2.49 分；MKQA 67.92，接近 jina-reranker-m0 的 68.19。

**Table 2 / 表 2：** Evaluation results for all rerankers. All scores are from our runs based on the top-100 retrieval results from the first row. For MKQA, we used Recall@10; for all other benchmarks, we used NDCG@10. / 全体精排评测。分数基于一阶段 top-100（首行）。MKQA 用 Recall@10，其余用 nDCG@10。

| Models | # Param | BEIR | MIRACL | MKQA | CoIR |
| --- | --- | ---: | ---: | ---: | ---: |
| **First-stage Retriever** | | | | | |
| jina-embeddings-v3 | 0.5B | 55.81 | 58.90 | 65.63 | - |
| jina-code-embeddings-0.5b | - | - | - | - | 73.94 |
| **Second-stage Reranker** | | | | | |
| jina-reranker-v3 | 0.6B | **61.85** | 66.83 | 67.92 | 70.64 |
| jina-reranker-v2 | 0.3B | 57.06 | 63.65 | 67.90 | 58.35 |
| jina-reranker-m0 | 2.4B | 58.95 | 66.75 | 68.19 | 66.89 |
| bge-reranker-v2-m3 | 0.6B | 56.51 | **69.32** | 67.88 | 36.28 |
| mxbai-rerank-base-v2 | 0.5B | 58.40 | 55.32 | 64.24 | 65.71 |
| mxbai-rerank-large-v2 | 1.5B | 61.44 | 57.94 | 67.06 | 70.87 |
| Qwen3-Reranker-0.6B | 0.6B | 56.28 | 57.70 | 65.34 | 65.18 |
| Qwen3-Reranker-4B | 4.0B | 61.16 | 67.52 | **69.25** | **73.91** |

### 5.3 English Retrieval Performance on BEIR / BEIR 英语检索

**EN.** Table 3 provides granular analysis across BEIR’s heterogeneous tasks, revealing specific architectural advantages. The model achieves consistent excellence across diverse reasoning tasks, with particularly strong performance on complex multi-hop reasoning reaching 78.58 on HotpotQA and fact verification achieving 94.01 on FEVER. These results highlight how LBNL interaction enables sophisticated query-document self-attention during encoding, capturing evidence relationships that separate encoding approaches miss.

**中.** Table 3 给出 BEIR 各任务细粒度结果。多跳 HotpotQA 78.58、事实验证 FEVER 94.01 尤为突出，体现 LBNL 在编码阶段的查询–文档自注意力能捕获分别编码易漏的证据关系。

**EN.** Within the same scale category, jina-reranker-v3 reveals significant advantages. Against bge-reranker-v2-m3 with the same 0.6B parameters, jina-reranker-v3 delivers a substantial 5.34% improvement from 56.51 to 61.85, demonstrating architectural innovation over simple parameter scaling. The specialized 512-dimensional projector network effectively concentrates ranking signals while preserving contextual representations from the Qwen3 backbone. Remarkably, our model surpasses mxbai-rerank-large-v2’s 61.44 performance while using $2.5\times$ fewer parameters, establishing that sophisticated architecture can surpass brute-force scaling approaches.

**中.** 同为 0.6B 时，相对 bge-reranker-v2-m3（56.51→61.85）提升 5.34%。以约 $2.5\times$ 更少参数超过 mxbai-rerank-large-v2 的 61.44。

**EN.** Since jina-reranker-v3 processes all documents simultaneously in a listwise manner within shared context windows, we investigate the sensitivity to document ordering. We evaluate three variants: documents ordered by descending relevance scores (D), ascending scores (A), and random permutation (R). The results show modest variations across orderings, with random ordering (R) achieving the highest average of 62.24, followed by descending (D) at 61.85 and ascending (A) at 61.45. While the differences are not conclusive, this analysis reveals that the LBNL interaction maintains relatively stable performance across different input orderings, suggesting robust self-attention mechanisms that can effectively process documents regardless of their initial arrangement.

**中.** 因列表级共享上下文，考察文档顺序敏感性：按相关性降序 (D)、升序 (A)、随机 (R)。平均分 R 62.24、D 61.85、A 61.45，差异不大，表明 LBNL 对输入顺序较稳健。

**EN.** The model’s dominance extends particularly to question-answering scenarios, where Natural Questions achieves 74.28 and argumentative retrieval on ArguAna reaches 73.43, showcasing the benefit of contextual embeddings. These tasks require understanding complex query intent and matching it against nuanced document semantics, precisely the scenario where our dual embedding extraction approach provides maximum advantage over traditional cross-encoder scoring.

**中.** 问答向任务亦强：Natural Questions 74.28，ArguAna 73.43，体现上下文嵌入与双位置嵌入抽取的优势。

**Table 3 / 表 3：** Performances of different rerankers (nDCG@10 in %) on BEIR. Top-100 retrieval results from jina-embeddings-v3 are passed as input. The best results are marked in bold. Avg. represents the averaged result of the 13 BEIR datasets. For jina-reranker-v3, (D)/(A)/(R) denote document ordering variants: Descending, Ascending, and Random relevance score ordering, respectively. / BEIR 上各精排 nDCG@10（%）。输入为一阶段 jina-embeddings-v3 的 top-100。(D)/(A)/(R) 分别为降序 / 升序 / 随机相关性顺序。

| Models | Size | Avg. | TC | NFC | NQ | HQA | FQA | AA | TCH | DBP | SD | FVR | CFV | SF | QRA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **First-stage Retriever** | | | | | | | | | | | | | | | |
| jina-embeddings-v3 | 0.5B | 55.81 | 77.81 | 36.65 | 64.31 | 64.63 | 47.47 | 54.31 | 26.55 | 41.07 | 19.91 | 89.00 | 42.33 | 72.4 | 89.06 |
| **Second-stage Reranker** | | | | | | | | | | | | | | | |
| jina-reranker-v3 (D) | 0.6B | 61.85 | 84.75 | 37.66 | **74.28** | **78.58** | 49.16 | 73.43 | 32.24 | 47.98 | 23.23 | **94.01** | 41.63 | 76.51 | **90.63** |
| jina-reranker-v3 (A) | 0.6B | 61.45 | 85.90 | 39.14 | 72.34 | 77.48 | 50.99 | 69.36 | 29.73 | 48.30 | 23.90 | 93.46 | 41.72 | 76.75 | 89.73 |
| jina-reranker-v3 (R) | 0.6B | **62.24** | 86.59 | 38.92 | 72.90 | 78.03 | 51.81 | **74.12** | 30.12 | 48.37 | 24.26 | 93.84 | 43.05 | 76.84 | 90.24 |
| jina-reranker-m0 | 2.4B | 58.95 | 84.17 | 41.03 | 72.25 | 76.99 | 51.62 | 40.69 | 31.79 | 49.34 | 22.91 | 91.14 | 36.42 | **79.94** | 88.01 |
| jina-colbert-v2 | 0.6B | 54.49 | 81.94 | 35.88 | 66.01 | 74.36 | 43.62 | 35.46 | 29.11 | 47.14 | 19.40 | 87.92 | 29.20 | 70.13 | 88.25 |
| jina-reranker-v2 | 0.3B | 57.06 | 80.53 | 37.17 | 67.39 | 76.17 | 46.48 | 39.28 | 32.35 | 47.81 | 20.03 | 93.02 | 37.17 | 76.50 | 87.83 |
| bge-reranker-v2-m3 | 0.6B | 56.51 | 82.19 | 34.33 | 69.52 | 77.89 | 45.45 | 36.21 | 33.12 | 46.72 | 17.79 | 91.03 | 38.69 | 72.64 | 89.10 |
| mxbai-rerank-base-v2 | 0.5B | 58.40 | 82.75 | 37.57 | 67.74 | 77.35 | 47.33 | 47.33 | 30.71 | 48.00 | 18.09 | 93.30 | 42.93 | 77.76 | 88.33 |
| mxbai-rerank-large-v2 | 1.5B | 61.44 | 81.51 | 37.76 | 72.46 | 78.10 | **52.75** | 74.55 | 29.81 | 49.07 | 18.58 | 93.94 | 42.03 | 78.86 | 89.36 |
| Qwen3-Reranker-0.6B | 0.6B | 56.28 | **87.08** | 38.37 | 56.54 | 74.41 | 43.45 | 56.53 | 27.26 | 43.54 | 20.98 | 86.19 | 44.11 | 74.89 | 78.32 |
| Qwen3-Reranker-4B | 4.0B | 61.16 | **87.08** | **41.56** | 69.06 | 77.03 | 52.29 | 58.82 | **33.73** | **50.81** | **26.01** | 87.80 | **47.59** | 78.41 | 84.83 |

### 5.4 Multilingual Performance on MIRACL / MIRACL 多语表现

**EN.** MIRACL evaluation across 18 diverse languages demonstrates jina-reranker-v3’s cross-lingual consistency despite its compact architecture. The 66.50 average performance reveals sophisticated multilingual understanding, with particularly strong results in morphologically complex languages like Arabic achieving 78.69 and challenging contexts like Thai reaching 81.06. These results reflect the effectiveness of our progressive multilingual training strategy, where architectural advantages help compensate for reduced multilingual specialization.

**中.** MIRACL 18 语评测显示紧凑架构下的跨语一致性。正文此处写平均 **66.50**，阿拉伯语 **78.69**、泰语 **81.06**（与 Table 4 中 Avg. 66.83、AR 78.85、TH 81.57 等单元格不完全一致；译文按各处原文分别保留）。

**EN.** Perhaps most significantly, jina-reranker-v3 exhibits minimal performance degradation across linguistic families, from Indo-European languages like Russian at 65.20 to Sino-Tibetan languages like Thai at 81.06. This consistency stems from our progressive multilingual training strategy that incorporates diverse datasets including MIRACL, mMARCO, and domain-specific multilingual corpora during the three-stage training progression. The architectural advantage becomes particularly evident in Korean achieving 73.83, where the model’s LBNL interaction enables effective handling of complex agglutinative morphology that traditional cross-encoders struggle to process efficiently.

**中.** 跨语系退化较小：俄语 65.20，泰语 81.06；韩语正文写 73.83（Table 4 为 KO 73.60）。渐进多语训练含 MIRACL、mMARCO 与领域多语语料。

**EN.** Compared to bge-reranker-v2-m3’s dedicated multilingual optimization averaging 69.32, jina-reranker-v3 accepts a 2.82-point gap while achieving superior English performance and maintaining architectural efficiency. This trade-off reflects our design philosophy: contextual embedding extraction provides competitive multilingual capabilities without massive multilingual scaling, creating an optimal balance for applications requiring both English excellence and cross-lingual competency.

**中.** 相对 bge-reranker-v2-m3 的 69.32，正文此处写接受 **2.82** 分差距（§5.2 相对同一对照写的是 2.49 分，基于 Table 2 的 66.83）；在英语更强与架构更省之间取平衡。

**Table 4 / 表 4：** Multilingual retrieval performance on the MIRACL (measured by nDCG@10). / MIRACL 多语检索（nDCG@10）。

| Models | Avg. | AR | BN | EN | ES | FA | FI | FR | HI | ID | JA | KO | RU | SW | TE | TH | ZH | DE | YO |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **First-stage Retriever** | | | | | | | | | | | | | | | | | | | |
| jina-embeddings-v3 | 58.90 | 71.53 | 69.86 | 48.37 | 46.91 | 54.13 | 71.15 | 50.90 | 55.05 | 47.83 | 56.46 | 64.76 | 55.63 | 54.07 | 70.48 | 73.56 | 55.29 | 49.18 | 65.01 |
| **Second-stage Reranker** | | | | | | | | | | | | | | | | | | | |
| jina-reranker-v3 | 66.83 | 78.85 | 79.47 | 59.45 | 54.57 | 57.70 | 76.03 | 55.74 | 61.52 | 57.43 | 65.94 | 73.60 | 65.50 | 64.54 | 74.53 | 81.57 | 65.60 | 56.74 | 74.07 |
| jina-reranker-m0 | 66.75 | 79.78 | 78.01 | 59.21 | 53.55 | 58.90 | 70.00 | 56.66 | 62.83 | 54.92 | 66.51 | 72.86 | 67.37 | 59.04 | 70.19 | 80.37 | 64.51 | 58.50 | 80.44 |
| jina-reranker-v2 | 63.65 | 72.50 | 79.42 | 46.66 | 51.54 | 57.81 | 73.05 | 50.90 | 60.94 | 56.66 | 59.15 | 72.60 | 53.43 | 66.47 | 74.62 | 77.75 | 62.49 | 53.06 | 76.69 |
| bge-reranker-v2-m3 | **69.32** | **80.51** | **81.85** | 57.67 | **57.64** | **61.92** | **80.38** | **59.60** | 67.66 | **58.86** | **67.37** | **75.14** | **67.61** | **68.92** | **76.69** | **82.29** | 64.46 | **58.32** | 80.85 |
| mxbai-rerank-base-v2 | 55.32 | 71.08 | 58.21 | 56.61 | 48.89 | 46.59 | 64.92 | 50.47 | 44.75 | 49.48 | 57.99 | 64.88 | 54.16 | 48.40 | 55.15 | 72.71 | 58.44 | 20.33 | 72.66 |
| mxbai-rerank-large-v2 | 57.94 | 71.38 | 63.48 | 57.55 | 49.14 | 48.38 | 66.70 | 51.61 | 45.12 | 49.05 | 56.61 | 64.98 | 54.80 | 51.79 | 62.41 | 74.51 | 62.29 | 38.66 | 74.42 |
| Qwen3-Reranker-0.6B | 56.16 | 67.44 | 66.67 | 50.91 | 45.77 | 52.07 | 65.50 | 43.28 | 60.36 | 49.66 | 51.56 | 61.03 | 48.88 | 46.72 | 69.86 | 72.95 | 45.14 | 43.00 | 70.04 |
| Qwen3-Reranker-4B | 67.52 | 78.32 | 81.51 | 59.37 | 53.07 | 61.63 | 78.70 | 55.02 | **68.71** | 54.90 | 65.32 | 71.80 | 64.66 | 66.50 | 75.60 | 82.00 | 59.35 | 57.56 | **81.39** |

---

## 6 Conclusion / 结论

**EN.** We present jina-reranker-v3, a 0.6B-parameter multilingual listwise reranker that introduces last but not late interaction for efficient document reranking. Our approach enables cross-document interactions during encoding by processing queries and multiple documents simultaneously within shared context windows, then extracting contextual embeddings from designated special token positions. By adapting long-context generative LLMs into a discriminative model, jina-reranker-v3 bridges the efficiency-effectiveness gap while maintaining significant computational advantages over generative rerankers. Future work includes studying the robustness of ranking against prompt injections and deduplication within the context window using submodularity optimization.

**中.** 我们提出 0.6B 多语列表精排 jina-reranker-v3，引入 LBNL 以高效精排：在共享上下文中同时处理查询与多文档，实现编码期跨文档交互，再从指定特殊 token 抽取上下文嵌入。把长上下文生成式 LLM 改造成判别式模型，弥合效率–效果鸿沟，并相对生成式精排保持算力优势。未来工作包括抗提示注入的排序稳健性，以及用子模优化在上下文窗口内做去重。

---

## Appendix A / 附录 A：Model Configuration and Training Details

**Table 5 / 表 5：** Model architecture configuration for jina-reranker-v3. / jina-reranker-v3 架构配置。

| Parameter | Value |
| --- | --- |
| Total Parameters | 0.6B |
| Non-Embedding Parameters | 0.44B |
| Hidden Size | 1,024 |
| Number of Layers | 28 |
| Attention Heads (Q/KV) | 16/8 (GQA) |
| Context Length | 131,072 |
| Effective Sequence Length | 8,192 |
| Projector Architecture | 1024 → 512 → 512 |
| Projector Activation | ReLU |

**Table 6 / 表 6：** Multi-stage supervised fine-tuning hyperparameters showing ranges across 47 training configurations. / 多阶段监督微调超参（跨 47 个训练配置的范围）。

| Hyperparameter | Stage 1 Foundation | Stage 2 Context & Hard Mining | Stage 3 Model Ensemble |
| --- | --- | --- | --- |
| Learning Rate | 5e-5 | [5e-5, 6e-6] | - |
| Batch Size (per device) | 60 | [6, 60] | - |
| Max Sequence Length | [768, 2048] | [2048, 8192] | - |
| Max Query Length | - | [256, 512] | - |
| Max Doc Length | - | [512, 2048] | - |
| Number of Negatives | 15 | [9, 25] | - |
| In-batch Negatives | 3 | [0, 3] | - |
| Temperature | 0.25 | [0.05, 0.25] | - |
| Training Mode | LoRA | LoRA/Full | Linear Merging |
| LoRA Rank | 16 | 16 | - |
| Word Embeddings | Tuned | Frozen/Tuned | - |
| Backbone | Frozen | Frozen/Tuned | - |
| Dispersive Loss $\alpha$ | 0.45 | [0.25, 0.45] | - |
| Dual Matching $\alpha$ | 0.85 | [0.65, 0.85] | - |
| Similarity Loss $\alpha$ | 0.85 | [0.75, 0.85] | - |

---

## Key References / 主要参考文献

- Khattab and Zaharia (2020). ColBERT. https://arxiv.org/abs/2004.12832
- van den Oord et al. (2019). Representation Learning with Contrastive Predictive Coding (InfoNCE). https://arxiv.org/abs/1807.03748
- Thakur et al. (2021). BEIR. https://arxiv.org/abs/2104.08663
- Zhang et al. (2023). MIRACL. https://doi.org/10.1162/tacl_a_00595
- Longpre et al. (2021). MKQA. https://arxiv.org/abs/2007.15207
- Li et al. (2024/2025). CoIR. https://arxiv.org/abs/2407.02883
- Yang et al. (2025). Qwen3 Technical Report. https://arxiv.org/abs/2505.09388
- Nogueira and Cho (2019). Passage Re-ranking with BERT. https://arxiv.org/abs/1901.04085
- Chen et al. (2024). BGE M3-Embedding.
- Jha et al. (2024). Jina-ColBERT-v2. https://arxiv.org/abs/2408.16672
- Zhang et al. (2025). Qwen3 Embedding. https://arxiv.org/abs/2506.05176
- Günther et al. (2024). Late Chunking. https://arxiv.org/abs/2409.04701
