# Embedding 调研报告

> Embedding 的关键是把任务相关性变成可检索的表示；本报告串起表示、交互、训练、评测和部署，具体论文与工程细节分别回查专题。
>
> 内容整理：2026-10-28。历史模型规格与成绩保留原报告语境，不作为当前排行榜；统一来源见[资料清单](资料清单_论文与博客.md)，各篇入口见[阅读索引](README.md)。

本文是一份跨论文的技术总览。先读“理论基础与表示范式”，区分骨干、读出、表示与交互，再按任务读训练、评测和部署。历史时间线用于理解方法出现的原因；需要复现某一篇时，应进入其独立研读核对数据、版本和完整实验条件。

核心问题的现成入口如下，不必按模型发布顺序逐个读完：

| 问题 | 学习入口 |
| --- | --- |
| 为什么对比学习能形成检索空间，温度和负例如何影响梯度 | [对比学习与 InfoNCE](对比学习与InfoNCE精讲.md) |
| 如何构造数据、过滤假负例、使用教师 | [难负例实践](难负例挖掘工业实践.md)、[蒸馏专题](Embedding蒸馏技术详解.md) |
| 图搜图、文搜图的小模型如何训练 | [表示学习与小模型蒸馏](图搜图与文搜图表示学习.md) |
| 稀疏、多向量、指令检索分别增加什么能力 | 本文“理论基础与表示范式”和“训练方法与数据工程” |
| 怎样检验泛化、避免污染、分离 ANN 与模型误差 | 本文“评测体系与 Benchmark”和“向量数据库与部署” |

---

## 执行摘要

**Embedding（嵌入）** 是将离散符号（词、句、文档、图像）映射到连续向量空间的技术，使语义相近的对象在空间中距离更近。它是现代 NLP、信息检索、推荐系统与 RAG 的**语义基础设施**。

### 主要表示路线

| 路线                  | 时期       | 代表                               | 核心特征              | 局限                 |
| --------------------- | ---------- | ---------------------------------- | --------------------- | -------------------- |
| **词项统计** | 长期使用 | BoW、TF-IDF、BM25 | 词项匹配、倒排友好、可解释 | 词汇不匹配与语义泛化受限 |
| **静态稠密**    | 2013–2018 | Word2Vec, GloVe, FastText          | 低维稠密、语义相似    | 一词一向量，无上下文 |
| **上下文/序列** | 2018– | ELMo、BERT、SBERT、E5、BGE | 上下文表示与任务训练 | 预训练隐藏状态未必适合直接计算句间距离 |
| **LLM 骨干** | 2023– | E5-Mistral、NV-Embed、Qwen3-Embedding | 用 Decoder 骨干学习检索表示 | 编码与部署成本需单独评估 |
| **图文及多模态** | 2021– | CLIP、SigLIP、GME、ColPali | 图文对齐、融合输入或页面多向量 | 任务、输入预算和索引形态不同 |

### 发展趋势与代表性结果

模型演进同时发生在骨干、监督、表示与部署四个层面。下面结合具体模型和实验说明这些变化；数值保留对应的时间、任务与比较条件，便于回查。

#### LLM 骨干扩大了能力上限，Encoder 与小模型仍有重要位置

E5-Mistral 将指令和 LLM 合成任务用于检索表示；NV-Embed 改进双向注意力、读出和训练课表；Qwen3-Embedding 把多语、多任务合成、监督训练与模型合并组织成完整配方。它们说明生成式预训练可为表示学习提供强底座，但最终模型仍需针对检索训练，不能把任意 LLM 的最后一个隐藏状态直接当成成熟 embedding。

| 模型 | 发布材料中的结果 | 时间与口径 | 能说明什么 |
| --- | --- | --- | --- |
| NV-Embed-v2 | MTEB 总分 **72.31**，检索子类 **62.65** | 官方模型卡记录截至 **2024-08-30** 的 56 项文本任务、其中 15 项检索任务 | 读出、训练课表与 positive-aware 难负例共同构成较强的通用表示配方 |
| Qwen3-Embedding-8B | MTEB Multilingual **70.58** | 技术报告和 **2025-06-05** 发布材料中的多语言评测 | 开源模型可在当时的多语言协议下与领先 API 模型竞争；0.6B／4B／8B 提供不同容量选择 |

来源：[NV-Embed-v2 官方模型卡](https://huggingface.co/nvidia/NV-Embed-v2)、[Qwen3-Embedding 技术报告](https://arxiv.org/abs/2506.05176)。两行来自不同任务集合和时间快照，**72.31 与 70.58 不能直接比较大小来排列模型能力**。同一协议内的具体对照仍有价值，应连同基线、语言和输入预算保留。

Encoder 路线也持续发展：Arctic-Embed 用数据和训练配方提高小模型检索质量，Jina v3 以任务 LoRA 分化空间，Jasper／Stella 及 Jina v5-text 用蒸馏迁移教师能力。因此更准确的趋势是“大骨干提高上限，数据与训练决定能力如何落到不同容量”，而非“Encoder 已被淘汰”。

#### BGE-M3 将三种表示放进同一模型，价值在于可组合

[BGE-M3](BGE/M3/BGE-M3三功能统一详解报告.md) 用约 **568M** 参数、**8192 token** 输入窗口，统一多语言的 Dense、Sparse 和 ColBERT 式多向量表示，并利用多头评分提供自知识蒸馏信号。它让同一个文本骨干同时服务向量召回、词项匹配和细粒度交互，减少维护独立表示模型的负担。

这使其适合作为多语、长文及混合检索的实验基线，但“支持三种头”不表示线上必须全部启用。只建 Dense 索引不会自动获得 Sparse 或 MaxSim 的收益；三路索引、融合、存储和延迟也有额外成本。是否成为某个业务的优先方案，应由同库对照决定，而不是从参数量和功能数量直接推出。

#### Hybrid 与重排解决互补问题，组合要对应失败类型

Dense 可缓解同义表述和词汇不匹配，BM25／学习式 Sparse 可补实体、型号与词面证据；Cross-Encoder 或多向量评分进一步检查候选中的细粒度匹配。由此形成的“多路召回 → 融合 → 精排”是重要工程路线，其意义是把不同证据放到合适的计算阶段。

但融合并不保证提升：低质量稀疏通道会引入噪声，领域错配的 reranker 可能把正确候选排低；增加候选还会提高延迟。应比较 Dense、词法、融合及加入精排后的增量，并分别报告 Recall、排序指标和成本。机制见本文检索 Pipeline，实验背景可对照 [DPR](DPR/DPR详解.md)、[SPLADE v2](SPLADE/SPLADE-v2/SPLADE-v2详解.md)、[BGE-M3](BGE/M3/BGE-M3三功能统一详解报告.md) 与 [BEIR](https://arxiv.org/abs/2104.08663)。

#### MRL 将输出维度变成可调预算，不能与模型压缩混为一谈

[MRL](MRL/MRL详解.md) 同时优化多个嵌套前缀，使同一模型的短向量也能执行任务。Nomic v1.5、Jasper／Stella、Qwen3 等将它用于表示训练；OpenAI embedding-3 通过 `dimensions` 暴露缩短输出的接口，例如 large 的默认 3072 维可按接口生成 1024 维。参见 [Nomic 官方说明](https://www.nomic.ai/news/nomic-embed-matryoshka) 和 [OpenAI 发布说明](https://openai.com/index/new-embedding-models-and-api-updates/)。

收益主要落在向量存储、传输和相似度计算。它不意味着编码骨干参数同步减少，也不意味着任何未经嵌套训练的向量都能任意截断而保持质量。短向量召回后再用长向量评分时，候选上限与额外存储仍要计入。

#### 多模态正在分化为图文对齐、通用输入和页面细粒度匹配

CLIP／SigLIP 双塔学习图像与文本的可比较表示；GME、VLM2Vec、MegaPairs 等进一步组织单模态、跨模态、融合输入及任务指令；ColPali／ColQwen 保留页面局部视觉 token，以 MaxSim 匹配文本查询。Jina v4 和 Qwen3-VL-Embedding 则展示了 VLM 骨干与不同检索表示、精排器的结合方式。

ColPali 可以绕过显式 OCR 管道，但仍需核算图像编码、页面分辨率、多向量存储和评分开销；单向量多模态模型较易接 ANN，也可能压缩细粒度证据。Cohere Embed v4、Gemini Embedding 2 和 Seed1.6 等 API 扩展了多模态服务选择；其输入覆盖、上下文和评测不能用一个“支持多模态”标签替代。

#### 小模型蒸馏与兼容空间让部署出现新的组合方式

Jasper／Stella 的多教师训练、Jina v5-text 的任务靶向蒸馏，说明小学生可以学习大模型的排序与空间结构，但容量、任务覆盖和教师错误仍限制可迁移程度。**参数应按实际导出模型核算**：例如 Jina v5-text 的论文总参数约为 nano 239M、small 677M，不能只因后者使用 0.6B 底座就将整模认定为不超过 0.6B。

另一条路线是让不同编码器保持兼容：Jina v5-omni 用锁定文本塔保留既有文本路径；Voyage 4 官方明确支持同系列大小模型的共享空间，可用小查询模型检索大文档模型的库。两种兼容机制不同，但都把“是否需要重建库”变成技术问题。它们不是所有模型天然具备的性质，任意两个同维度 checkpoint 仍不能混用。参见 [Jina v5-text](Jina/v5-text/Jina-embeddings-v5-text详解.md)、[Jina v5-omni](https://arxiv.org/abs/2605.08384) 与 [Voyage 4 官方说明](https://blog.voyageai.com/2026/01/15/voyage-4/)。

### 贯穿各章节的结论

1. **表示服务于相关性定义**：STS、问答检索、实例图搜图与文搜图并不共享同一套标签。更大骨干不能替代合适的监督和评测。
2. **模型质量与系统质量分别测量**：先用精确搜索观察表示，再测 ANN、Hybrid、精排和下游生成。候选阶段漏掉的证据无法由后续精排找回。
3. **互补通道要有证据**：词法检索可补实体和编号，多向量可保留局部匹配，Cross-Encoder 可联合判断约束；是否值得加入取决于错误类型与成本。
4. **压缩有不同作用对象**：蒸馏改变学生学习到的能力，MRL 改维度，向量量化改存储精度，权重量化改编码器成本。它们可以组合，但误差与资源收益需要分别测量。
5. **榜单结论必须带协议**：任务集合、版本、语言、候选库与输入长度共同决定分数。不能从两个不同平均分推出跨模型的普遍优劣。

---

## 发展历程：从 BoW 到 LLM Embedding

### 完整时间线

本节分三层：**总览时间轴**定位范式，**逐项时间线**保留具体工作、先后和技术定位，**model card** 对照骨干、模态、规模、维度、窗口与交互。短表用于导读，不能替代详细记录。
「BERT 之后」是重点：从句向量、Dense Retrieval，到指令化、LLM 骨干、系列化产品线（BGE / Jina / GTE / E5…）与多模态通用嵌入（Seed / Qwen3-VL-Emb / ColPali…）。表中参数/分数为公开文档与论文报道的**量级参考**，以厂商更新为准。

#### 总览时间轴

按首次提出的时期理解方法，预印本和正式发表可能跨年；精确版本与出处见[资料清单](资料清单_论文与博客.md)。这里概括范式变化，后面的逐项时间线给出具体节点与来源，model card 再集中比较规格。

| 时期 | 代表工作 | 解决或暴露的问题 |
| --- | --- | --- |
| 1954 | 分布假说 | 用上下文共现理解词义，为统计语义提供出发点 |
| 1980s–1990s | WordNet；LSI（1990）、BM25 | 词典知识、低秩语义空间和词项概率检索是不同路线；LSI 已是稠密表示 |
| 2003 | 神经概率语言模型 | 联合学习连续词表示与语言模型，代表论文发表于 JMLR |
| 2013–2016 | Word2Vec、GloVe、FastText；DSSM | 静态词向量、子词泛化，以及用点击监督训练查询—文档语义匹配 |
| 2017–2018 | Transformer、InferSent、ELMo、BERT、USE | 上下文表示与句子级监督发展；预训练隐藏状态不自动具有合适的句间距离 |
| 2019 | Sentence-BERT、ORQA | 可预计算句向量与潜变量开放域检索，分别改变表示和问答训练方式 |
| 2020–2021 | DPR、ANCE、REALM、RAG、FiD | 监督双塔、全局难例、检索预训练与多证据生成形成不同系统范式 |
| 2020–2022 | ColBERT、ColBERTv2、PLAID | token 多向量细粒度交互，随后解决监督、索引体积与执行成本 |
| 2021 | CLIP、ALIGN、SimCSE、Contriever、SPLADE | 图文配对、无监督句向量／检索与学习式稀疏表示并行发展 |
| 2022–2023 | MRL、INSTRUCTOR、E5、GTE、BGE、Jina v2 | 嵌套维度、任务指令、多阶段数据训练和长上下文编码 |
| 2023–2024 | E5-Mistral、GritLM、LLM2Vec、NV-Embed、Qwen 系骨干 | 如何把 Decoder 预训练能力转为高质量检索空间，涉及注意力、读出、数据与蒸馏 |
| 2024 | BGE-M3、ColPali、VLM2Vec、GME、MegaPairs | 三种表示共同训练、页面视觉检索、通用任务转换与关系数据合成 |
| 2025–2026 | Qwen3-Embedding、SigLIP 2、DINOv3、Jina v4/v5、Qwen3-VL-Embedding 等 | 指令、多语、多模态、小学生与部署压缩继续交汇；各系列并不采用同一配方 |

#### 逐项时间线：方法、系列与发布节点

表中“预印本”采用 arXiv 首次提交时间，“发布”采用官方公告或项目记录；正式会议年份另注。只确认年份的系列保留年份，不把模型卡更新时间或编号后缀当成首发日期。每个节点保留方法定位，参数与完整实验继续在后面的卡片及独立研读中回查。

##### 统计语义与神经词表示的前史

| 时间及口径 | 工作与入口 | 方法增量与后续联系 |
| --- | --- | --- |
| 1954，论文 | [Harris：Distributional Structure](https://doi.org/10.1080/00437956.1954.11659520) | 以分布关系研究语言结构，为“上下文提供语义线索”建立基础；它不是一个已经训练好的向量检索模型。 |
| 1980 年代起，项目 | [WordNet](https://wordnet.princeton.edu/) | 人工词义集合及词汇关系构成另一种语义资源；它提供知识结构，与统计共现学习并行发展。 |
| 1990，论文 | LSA／LSI：Indexing by Latent Semantic Analysis | 对词项—文档矩阵做低秩分解，在神经网络普及前已用稠密潜在空间缓解词汇不匹配；不能归为“无语义的高维稀疏表示”。 |
| 1990 年代，检索系统 | [BM25／Okapi](https://www.microsoft.com/en-us/research/publication/okapi-at-trec-3/) | 概率检索中的词频饱和、文档长度与逆文档频率加权；后来的 Dense 并未使词法通道失去价值。 |
| 2003，JMLR | [Bengio 等：A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html) | 联合学习连续词表示和语言模型，以共享统计强度缓解离散组合问题；早于 Word2Vec 的神经表示起点之一。 |

##### 静态词向量到上下文与句子表示

| 时间及口径 | 工作与入口 | 方法增量与后续联系 |
| --- | --- | --- |
| 2013-01，预印本；同年后续论文 | [Word2Vec](https://arxiv.org/abs/1301.3781) | CBOW／Skip-gram 提升大语料词向量训练效率；负采样与短语组合见[后续工作](https://arxiv.org/abs/1310.4546)，不能把两篇方法细节都归给首篇。 |
| 2013，CIKM | [DSSM](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/) | 用点击监督学习查询—文档语义匹配，说明双塔搜索早于 BERT；监督来源和文本表示与后来的 DPR 不同。 |
| 2014，EMNLP | [GloVe](https://aclanthology.org/D14-1162/) | 用全局词共现统计学习词表示，与局部上下文预测形成对照；仍是一词一向量。 |
| 2016-07，预印本 | [FastText](https://arxiv.org/abs/1607.04606) | 将词分解成字符 n-gram 组合，缓解未登录词和形态变化；不是句子级检索训练的替代品。 |
| 2017-05，预印本；EMNLP 2017 | [InferSent](https://aclanthology.org/D17-1070/) | 用自然语言推断监督训练可迁移句向量，连接 NLI、句子表示与后来的 SBERT。 |
| 2017-06，预印本 | [Transformer](https://arxiv.org/abs/1706.03762) | 以注意力构建序列模型，成为后续 Encoder、Decoder 与 VLM 骨干的共同构件；本身不规定检索表示与损失。 |
| 2018，论文 | [ELMo](https://aclanthology.org/N18-1202/) | 用双向语言模型产生随上下文变化的词表示，突破静态词向量的一词一义限制。 |
| 2018-03，预印本 | [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175) | Transformer／DAN 句编码路线，把可迁移的句子表示做成通用接口；与后来 Sentence-T5 是不同工作。 |
| 2018-10，预印本；NAACL 2019 | [BERT](https://aclanthology.org/N19-1423/) | MLM 等预训练提供通用上下文特征；原始 CLS／mean 并不因此就适合用余弦距离做句子检索。 |
| 2019-08，预印本；EMNLP 2019 | [Sentence-BERT](Sentence-BERT/Sentence-BERT详解.md) | 共享编码器配合池化和句子级监督，将联合句对计算转为可预计算的句向量；速度优势取决于候选规模与比较任务。 |
| 2019-10，预印本 | [DistilBERT](https://arxiv.org/abs/1910.01108) | 将 BERT 的语言建模与中间表示知识迁入较小学生，是骨干蒸馏案例；它没有直接定义检索排序蒸馏。 |

##### 开放域检索、多向量与跨模态双塔

| 时间及口径 | 工作与入口 | 方法增量与后续联系 |
| --- | --- | --- |
| 2020-02，预印本 | [REALM](https://arxiv.org/abs/2002.08909) | 将潜变量知识检索纳入语言模型预训练，研究外部文档与参数知识的配合，也引入索引更新问题。 |
| 2020-04，预印本；EMNLP 2020 | [DPR](DPR/DPR详解.md) | 查询／段落双编码器、同批负例和 BM25 难负例形成开放域问答的可复用稠密检索基线。 |
| 2020-04，预印本 | [SPECTER](SPECTER/SPECTER系列详解.md) | 用科学论文引用图构造相关性监督，说明“文本相似”可由领域关系重新定义。 |
| 2020-04，预印本；SIGIR 2020 | [ColBERT](ColBERT/ColBERT详解.md) | 保留 token 多向量并用 MaxSim 后交互，兼顾文档预计算和细粒度匹配；索引与评分代价随之上升。 |
| 2020-05，预印本 | [RAG](https://arxiv.org/abs/2005.11401) | 将文档检索与生成概率结合；原论文的 RAG-Sequence／Token 比今天泛称的“检索再拼 prompt”更具体。 |
| 2020-07，预印本；ICLR 2021 | [ANCE](ANCE/ANCE详解.md) | 从全库 ANN 召回难负例并异步刷新索引，补充局部 batch 负例的覆盖不足。 |
| 2020，预印本；EACL 2021 | [FiD](https://aclanthology.org/2021.eacl-main.74/) | 分别编码检索段落，再在解码器融合多段证据；回答的是如何利用候选，而非怎样编码检索库。 |
| 2021-01，预印本 | [GradCache](GradCache/GradCache详解.md) | 缓存表示梯度并分块重算，让大对比候选集合与编码器激活内存解耦；与普通梯度累积不同。 |
| 2021-01，官方发布；同年预印本 | [CLIP](CLIP/CLIP详解.md) | 大规模图文配对和对称对比目标学习共同空间，支持零样本分类与跨模态匹配；不是实例图搜图的统一答案。 |
| 2021-02，预印本 | [ALIGN](https://arxiv.org/abs/2102.05918) | 用约 18 亿噪声图文对训练图像／文本双塔，展示弱配对规模与数据处理的价值；“十亿级”描述的是数据，不是参数。 |
| 2021-04，预印本 | [SimCSE](无监督对比检索三部曲_SimCSE-Contriever-Condenser.md) | 用 dropout 产生句子正视图，并研究 NLI 监督版本；STS 与查询—段落检索的目标不能直接等同。 |
| 2021-07／09，预印本 | [SPLADE](SPLADE/SPLADE/SPLADE详解.md)／[SPLADE v2](SPLADE/SPLADE-v2/SPLADE-v2详解.md) | MLM 词表上的学习式扩展、稀疏约束、聚合及改进训练，为神经表示保留倒排检索形态。 |
| 2021-08，预印本；Findings ACL 2022 | [Sentence-T5](https://aclanthology.org/2022.findings-acl.146/) | 系统研究如何从 T5 得到句子向量及扩大编码器规模；与 2018 年的 Universal Sentence Encoder 分开理解。 |
| 2021-12，预印本；NAACL 2022 | [ColBERTv2](ColBERTv2/ColBERTv2详解.md) | 去噪监督与残差压缩提高后交互的质量和存储效率；引擎执行仍由后续工作继续优化。 |
| 2021-12，预印本；EMNLP 2022 | [GTR](https://arxiv.org/abs/2112.07899) | 用 T5 Encoder 扩展双编码器规模并研究跨域检索；属于 Encoder-Decoder 家族的检索化，不等于推理时生成文本。 |
| 2021-12，预印本 | [Contriever](无监督对比检索三部曲_SimCSE-Contriever-Condenser.md) | 无监督对比训练用于稠密检索与跨域迁移，为依赖人工检索标注的路线提供对照。 |

##### 指令、维度弹性与检索预训练

| 时间及口径 | 工作与入口 | 方法增量与后续联系 |
| --- | --- | --- |
| 2022-02，预印本 | [SGPT](https://arxiv.org/abs/2202.08904) | 探索 GPT 系模型用于语义搜索；其双编码器的重要设计是位置加权 mean pooling，不能简单概括为所有版本都用 last-token。 |
| 2022-04，预印本 | [DiffCSE](https://arxiv.org/abs/2204.10298) | 引入对文本改动的差异感知目标，补充 dropout 视图的句向量对比学习。 |
| 2022-05，预印本；CIKM 2022 | [PLAID](PLAID/PLAID详解.md) | 利用质心近似和多阶段剪枝减少残差读取、解压及完整评分，是执行引擎改进，不是新骨干。 |
| 2022-05，预印本 | [RetroMAE](RetroMAE/RetroMAE与DupMAE详解.md) | 通过非对称掩码重建训练检索导向编码器，连接语言预训练与后续对比微调。 |
| 2022-05，预印本；NeurIPS 2022 | [MRL](MRL/MRL详解.md) | 联合优化嵌套维度，让短前缀也携带任务信息；后续 Nomic、Jasper、Qwen 等采用相关训练方式。 |
| 2022-12，预印本 | [E5](E5/E5详解.md) | 大规模弱监督文本对预训练，再做监督适配；query／passage 前缀成为模型输入契约的一部分。 |
| 2022-12，官方发布 | [OpenAI text-embedding-ada-002](https://openai.com/index/new-and-improved-embedding-model/) | 以统一 API 模型覆盖多类文本与代码匹配用途，提供较长输入窗口；闭源发布不公开全部骨干与训练细节。 |
| 2022-12，预印本 | [INSTRUCTOR](INSTRUCTOR/INSTRUCTOR详解.md) | 将任务指令与文本一起编码，用 MEDI 多任务数据学习按任务产生表示；不是只加一个字符串就自动获得指令能力。 |

##### 开源系列化与 LLM 嵌入的起步

| 时间及口径 | 工作与入口 | 方法增量与后续联系 |
| --- | --- | --- |
| 2023-03，预印本 | [SigLIP](SigLIP/SigLIP/SigLIP详解.md) | 用逐对 sigmoid 取代全 batch softmax 归一化，改变图文配对损失及分块计算方式。 |
| 2023-04，预印本 | [DINOv2](DINO/DINOv2/DINOv2详解.md) | 视觉自监督与数据筛选产生可迁移全局／局部特征，补充依赖图注的 CLIP 视觉语义。 |
| 2023-07，预印本 | [Jina embeddings v1](Jina/v1/Jina-embeddings-v1详解.md) | 从句向量数据、训练与模型规模建立系列起点；后续 v2 的长上下文和 v3 的任务适配是不同增量。 |
| 2023-08，模型发布；09，C-Pack 预印本 | [BGE v1／C-Pack](BGE/C-Pack/BGE-CPack详解.md) | 将中文数据、C-MTEB 与中英模型系列组织成开放资源；模型发布与论文日期分开记录。 |
| 2023-08，预印本 | [GTE](GTE/GTE系列详解.md) | 多阶段对比训练整合不同质量的文本对，为后续长上下文、多语及 Qwen 骨干系列打基础。 |
| 2023-09，项目发布 | [BGE v1.5](https://github.com/FlagOpen/FlagEmbedding) | 改进相似度分布与可用性；small／base／large、中英文权重各有适用输入约定。 |
| 2023，系列发布 | [BGE reranker base／large](BGE/BGE-Reranker详解.md) | 联合编码查询和候选输出标量相关性，与可建库的 embedding 分工；不是再增加一条单向量召回通道。 |
| 2023-10，预印本 | [Jina embeddings v2](Jina/v2/Jina-embeddings-v2详解.md) | JinaBERT 与位置机制支持 8192 token，随后发展分语言模型；需区分窗口长度与实际长文检索质量。 |
| 2023-11，官方发布 | [Cohere Embed v3](https://cohere.com/blog/introducing-embed-v3) | 英文／多语及轻量版本，通过 `search_query`／`search_document` 等输入类型区分用途；并非 2023 年 3 月发布。 |
| 2023–2024，早期产品系列 | [Voyage 早期 lite／large、voyage-2 系列](https://docs.voyageai.com/docs/embeddings) | 保留商业检索向 API 的演进支线；`voyage-lite-02-instruct` 等型号有特定用途，不能统称一个无版本的“Voyage-lite”。 |
| 2023-12-31，首次提交；arXiv 编号属 2024-01 | [E5-Mistral](E5-Mistral/zh.md) | Mistral-7B、任务指令与 LLM 合成检索数据结合，以较短微调把生成式底座转为强句向量；解释了论文年、编号月和模型发布可跨年。 |

##### LLM、混合表示和视觉文档共同发展

| 时间及口径 | 工作与入口 | 方法增量与后续联系 |
| --- | --- | --- |
| 2024-01，官方发布 | [OpenAI embedding-3-small／large](https://openai.com/index/new-embedding-models-and-api-updates/) | 改进文本与多语表示，并通过 `dimensions` 提供可缩短输出；small 默认 1536 维、large 默认 3072 维。 |
| 2024-01-30，项目发布；02，预印本 | [BGE-M3](BGE/M3/BGE-M3三功能统一详解报告.md) | 统一 Dense／Sparse／Multi-Vector、多语与 8192 token，通过跨头监督提升表示；三路收益需配套索引与融合。 |
| 2024-02，发布与预印本 | [Nomic Embed Text v1](Nomic-Embed/Nomic-Embed详解.md) | 强调开放数据、训练可复现与 8K 输入，使数据和训练条件可以回查，而非只发布一个分数。 |
| 2024-02，官方发布 | [Nomic Embed Text v1.5](https://www.nomic.ai/news/nomic-embed-matryoshka) | 在 v1 基础上引入 MRL，支持可变维度；后来 Vision 分支再研究与文本空间的对齐。 |
| 2024-02，预印本 | [GritLM](LLM-Embedding冲榜路线_E5Mistral-NVEmbed-GritLM-SFR-Arctic-Stella.md) | 在同一模型里联合生成与表示任务；“统一用途”不等于两种目标没有权衡。 |
| 2024-03，项目发布 | [BGE reranker v2](BGE/BGE-Reranker详解.md) | M3、Gemma、MiniCPM 等不同骨干形成多语和 LLM 精排器；需按模型区分输入长度、层数与计算成本。 |
| 2024-04，预印本 | [LLM2Vec](LLM2Vec/LLM2Vec详解.md) | 双向注意力、掩码下一 token 预测与无监督对比三个步骤，研究 Decoder 如何转为文本编码器。 |
| 2024，系列模型；05，v1 预印本 | [Arctic-Embed v1／v1.5](https://arxiv.org/abs/2405.05374) | 用数据、负例和训练细节提高小 Encoder 的检索质量；v1.5 的维度／量化增强应与原始 v1 分开看。 |
| 2024-05，预印本 | [NV-Embed v1](https://arxiv.org/abs/2405.17428) | 双向编码、latent attention 读出和分阶段指令训练，展示 LLM embedding 的改进不只是换大骨干。 |
| 2024-05，预印本 | [Jina-CLIP v1](Jina/clip/v1/Jina-CLIP详解.md) | 将图文对齐与文本检索训练结合，研究同一文本塔如何兼顾两类用途。 |
| 2024-06-27，首次提交；ICLR 2025 | [ColPali](ColPali/ColPali详解.md) | 页面渲染图经 PaliGemma 产生局部向量，与查询 token 做后交互，减少显式 OCR／切块的信息损失。 |
| 2024-07，模型发布；09，预印本 | [BGE-EN-ICL](BGE/EN-ICL/BGE-EN-ICL详解.md) | 在查询编码中加入任务相关示例，令文本嵌入器具备 few-shot 输入适配能力；示例占用编码预算。 |
| 2024-07，项目发布 | [BGE multilingual Gemma2](BGE/BGE-multilingual-gemma2详解.md) | 用 Gemma-2-9B 做多语检索表示，与 BGE-M3 的共享多头 Encoder 是不同路线。 |
| 2024-07，项目发布 | [BGE reranker v2.5 Gemma2 lightweight](BGE/BGE-Reranker详解.md) | 在精排中研究层数与 token 压缩，目标是减少联合编码成本；不能与向量截维混为一谈。 |
| 2024-07，mGTE 预印本；同年模型系列 | [mGTE／gte-v1.5／gte-Qwen2](GTE/GTE系列详解.md) | 将长上下文、多语编码、重排与 Decoder 骨干分化成产品系列；gte-Qwen2 是同年明确的 LLM 双编码器支线。 |
| 2024-08，模型卡结果快照 | [NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2) | 延续读出与课表设计，结合 positive-aware 难负例处理；56 任务 MTEB 72.31 对应 2024-08-30 快照。 |
| 2024，模型系列；12，合写预印本 | [Stella／Jasper](Jasper/Jasper-Token-Compression-600M详解.md) | 多教师蒸馏、MRL 与高效表示强调容量约束下的质量；两个系列具体骨干和压缩机制仍须分别回查，不能由合写日期推断同时发布。 |
| 2024-09，预印本 | [Jina embeddings v3](Jina/v3/Jina-embeddings-v3详解.md) | 多语共享骨干、任务 LoRA 与 MRL，让 retrieval、classification、clustering 等用途有不同适配。 |
| 2024-09，官方发布 | [Voyage-3／3-lite](https://blog.voyageai.com/2024/09/18/voyage-3/) | 将检索质量、上下文和成本分档；模型发布中的自建评测应保留候选和基线条件。 |
| 2024，模型卡 | [ColQwen2](ColQwen/ColQwen系列详解.md) | 将页面后交互接到 Qwen2-VL；属于 ColPali 路线的模型迭代，不应另造一篇独立源论文。 |
| 2024-10，预印本 | [VLM2Vec／MMEB](VLM2Vec/VLM2Vec详解.md) | 将分类、问答、定位等转为候选匹配，指令化对比训练 VLM，并用 MMEB 研究跨任务迁移。 |
| 2024-10，官方功能发布 | [Cohere Embed v3 图像扩展](https://docs.cohere.com/changelog/embed-v3-is-multimodal) | 既有 v3 文本空间增加图像输入能力；这是模型功能扩展节点，与 v3 文本模型首发分开记录。 |
| 2024-12，官方发布 | [Cohere Rerank 3.5](https://docs.cohere.com/changelog/rerank-v3.5) | 对候选输出相关性排序。它属于 Rerank 精排产品线，不是 Embed v3 的新版本。 |
| 2024-12，预印本 | [Arctic-Embed 2.0](https://arxiv.org/abs/2412.04506) | 从英文检索向多语能力扩展，关注跨语质量与英文能力之间的权衡。 |
| 2024-12，预印本 | [Jina-CLIP v2](Jina/clip/v2/Jina-CLIP-v2详解.md) | 多语文本、视觉表示与 MRL 组合；仍是双塔图文对齐，并非 Jina v4 式 VLM 多向量模型。 |
| 2024-12，预印本 | [MegaPairs／MMRet](MegaPairs/MegaPairs详解.md) | 在多个特征空间挖图像对并生成关系指令，规模化构造检索关系数据；与后续 BGE-VL checkpoint 按模型卡对应。 |
| 2024-12，预印本 | [GME](GME/GME详解.md) | 用 Qwen2-VL 编码文本、图像和融合输入，混合任务数据与合成融合样本，并用 UMRB 评测覆盖。 |

##### 多语、代码、视觉文档与通用多模态的继续扩展

| 时间及口径 | 工作与入口 | 方法增量与后续联系 |
| --- | --- | --- |
| 2025-01／06，v1／v2 预印本 | [KaLM-Embedding](https://arxiv.org/abs/2501.01028)／[V2](https://arxiv.org/abs/2506.20923) | 从较干净、多样的数据和小 Qwen 底座出发，再研究训练技巧；不能把 KaLM 一概归成 EuroBERT 模型。 |
| 2025-02，预印本 | [SigLIP 2](SigLIP/SigLIP2/SigLIP2详解.md) | 在 sigmoid 训练基础上加入多语、定位与自蒸馏等改进，并提供 NaFlex 变体；既服务图文匹配，也改善局部视觉特征。 |
| 2025，模型卡 | [ColQwen2.5／社区 ColQwen3](ColQwen/ColQwen系列详解.md) | 延续页面多向量，更新骨干与模型训练；官方系列与社区换底座版本分别标明来源及许可。 |
| 2025-03，技术报告 | [Gemini Embedding](https://arxiv.org/abs/2503.07891) | 把 Gemini 能力用于通用文本表示；报告日期、API 预览与正式可用日期不是同一事件，不能仅记为 2025 年 10 月。 |
| 2025-04，官方发布 | [Cohere Embed v4](https://docs.cohere.com/changelog/embed-multimodal-v4) | 支持文本、图像及混合输入，面向企业多模态检索；其输入预算、输出维度和比较条件按官方文档核对。 |
| 2025，模型系列 | [BGE-VL](https://github.com/VectorSpaceLab/MegaPairs) | 将通用视觉检索及关系合成数据落到公开权重，连接 CLIP／MLLM 两类骨干；不是把纯文本 BGE-M3 直接用于图片。 |
| 2025-05，技术报告 | [BGE-Code-v1／CodeR](https://arxiv.org/abs/2505.12697) | 用大规模代码任务合成训练通用代码表示，关注自然语言—代码、代码—代码等不同检索关系。 |
| 2025-06，官方发布与预印本 | [Qwen3-Embedding／Reranker](Qwen3-Embedding/zh.md) | 同底座提供 0.6B／4B／8B 的独立编码与联合精排模型，结合指令、MRL、多阶段数据训练和合并。 |
| 2025，前代产品线 | [Seed1.5-Embedding](https://seed.bytedance.com/en/blog/introducing-seed1-5-embedding) | 保留豆包文本向量产品的前代入口；不能将它、Seed1.6 基础模型与 Seed1.6-Embedding 的能力和版本号混用。 |
| 2025-06，官方发布 | [Seed1.6-Embedding](https://seed1-6-embedding.github.io/) | 基于 Seed1.6-Flash，支持文本、图像、视频及组合输入检索；维度、时长和序列预算依接口与版本确认。 |
| 2025-06，预印本 | [Jina embeddings v4](Jina/v4/Jina-embeddings-v4详解.md) | VLM 骨干结合任务适配及单／多向量输出，连接通用多模态空间与后交互检索。 |
| 2025-08，预印本 | [DINOv3](DINO/DINOv3/DINOv3详解.md) | 扩大视觉自监督训练并稳定局部特征，继续提供图搜图和视觉任务可用的表示；它不单独完成图文空间对齐。 |
| 2025-09，预印本；论文另述此前模型快照 | [Conan-embedding-v2](Conan-embedding/v2/Conan-embedding-v2详解.md) | 从零训练约 1.4B 的嵌入向 LLM，并结合数据、soft-masking 与动态难例；不能只概括成一次中文 hard-negative 续训。 |
| 2025-09，技术报告 | [EmbeddingGemma](https://arxiv.org/abs/2509.20354) | 约 300M 的多语文本表示路线，补充蒸馏、数据与小模型部署的比较；具体代码与评测仍以报告为准。 |

##### 全模态、任务蒸馏与兼容空间

| 时间及口径 | 工作与入口 | 方法增量与后续联系 |
| --- | --- | --- |
| 2026-01，预印本 | [Qwen3-VL-Embedding／Reranker](QAT/Qwen3-VL-Embedding/Qwen3-VL-Embedding详解.md) | 多模态 embedding 与精排统一组织，结合数据合成、分阶段训练、MRL／向量量化；查询编码与候选联合精排分别承担不同计算阶段。 |
| 2026-01，官方发布 | [Voyage 4](https://blog.voyageai.com/2026/01/15/voyage-4/) | 同系列共享表示空间，支持大小查询／文档模型组合；large 使用 MoE，nano 提供开放权重，不能把全系列都称为闭源或都称为 MoE。 |
| 2026-02，预印本 | [Jina embeddings v5-text](Jina/v5-text/Jina-embeddings-v5-text详解.md) | 任务靶向蒸馏、任务适配与几何正则；nano 与 small 的骨干、总参数和读出分别核算。 |
| 2026-03，公开预览；04，GA；05，报告 | [Gemini Embedding 2](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/) | 将文本、图像、音频、视频及文档映到统一空间；[技术报告](https://arxiv.org/abs/2605.27295)晚于产品预览，产品首发与论文发表需分别记录。 |
| 2026-05，预印本 | [Jina embeddings v5-omni](Jina/v5-omni/Jina-embeddings-v5-omni详解.md) | GELATO 锁定文本和模态塔，只训练轻量对齐部分，扩展图像／视频／音频同时保留原文本路径；与 v5-text 的二月报告分开记录。 |

这些节点反映几条并行演进线：**表示与交互**（单向量／词项／后交互）、**监督与数据**（弱配对／指令／合成／难例／蒸馏）、**骨干与模态**（Encoder／Decoder／VLM）和**部署**（MRL／量化／兼容空间）。时间更晚不表示所有维度都更好；同一年也会同时存在适合不同任务和预算的方案。

#### 前深度学习与静态词向量（简卡）

| 模型 / 方法  | 年份       | 架构要点                       | 模态     | 典型维度   | 备注                   |
| ------------ | ---------- | ------------------------------ | -------- | ---------- | ---------------------- |
| BoW / TF-IDF | 1950s–    | 计数 / 加权                    | 文本     | \|V\| 稀疏 | 仍作 Hybrid 组件       |
| LSA / LSI    | 1990 | SVD 降维                       | 文本     | ~100–300  | 早期「语义空间」       |
| Word2Vec     | 2013       | CBOW / Skip-gram + NegSampling | 词       | 100–300   | 类比运算里程碑         |
| GloVe        | 2014       | 全局共现 + 最小二乘            | 词       | 50–300    | 统计 + 神经折中        |
| FastText     | 2016       | 子词 n-gram                    | 词       | 300        | OOV / 形态丰富语言更强 |
| ELMo         | 2018       | 双向 LSTM 层加权               | 上下文词 | ~1024      | 一词多义破冰           |
| USE          | 2018       | Transformer / DAN              | 句       | 512        | 通用句嵌入工业开端     |

#### 从 BERT 到句向量与 Dense Retrieval（2018–2022）

| 模型                    | 年份  | 骨干 / 架构                   | 模态            | 参数量级               | 维度          | 上下文         | 交互范式          | Model Card 要点                                                   |
| ----------------------- | ----- | ----------------------------- | --------------- | ---------------------- | ------------- | -------------- | ----------------- | ----------------------------------------------------------------- |
| **BERT**          | 2018 预印本／2019 正式发表 | Encoder Transformer；MLM+NSP  | 文本            | Base 110M / Large 340M | 隐层 768/1024 | 512            | —（预训练底座）  | 开启「预训练→微调」；原生非句向量，靠`[CLS]`/mean 勉强抽句向量 |
| **Sentence-BERT** | 2019  | Siamese BERT + pooling        | 文本            | ≈BERT                 | 768/1024      | 512            | Bi-Encoder        | NLI/三元组；句相似度速度 ×1000+；专用 Embedding 产业起点         |
| **DistilBERT**    | 2019  | 6 层学生蒸馏 BERT             | 文本            | ~66M                   | 768           | 512            | Encoder / 可做 Bi | 语言预训练骨干蒸馏；不是检索排序蒸馏                                   |
| **DPR**           | 2020  | 双 BERT（Q/D 塔）             | 文本（Passage） | ~2×BERT-base          | 768           | 512            | Bi-Encoder        | Facebook；FAISS 稠密段落检索标准化                                |
| **REALM**         | 2020  | 检索增强预训练                | 文本            | —                     | —            | —             | Bi + 检索         | 检索进预训练循环                                                  |
| **ColBERT (v1)**  | 2020  | BERT + 每 token 向量          | 文本            | ~BERT                  | token×128    | 512            | Late Interaction  | MaxSim；精度/速度折中路线开创                                     |
| **SPECTER**       | 2020  | SciBERT + 引用图对比          | 科学文本        | ~110M                  | 768           | 512            | Bi-Encoder        | 论文检索 / 推荐专用                                               |
| **CLIP**          | 2021  | ViT + 文本 Transformer 双塔   | 图+文           | ViT-L/14 ≈400M 级     | 512/768       | 文本 77 tok 级 | Dual-/Bi-Encoder  | OpenAI；对比学习跨模态对齐；零样本分类                            |
| **ALIGN**         | 2021  | EfficientNet + BERT           | 图+文           | — | —            | —             | Dual-Encoder      | Google；约 18 亿弱配对，数据规模与参数量分开                                        |
| **SimCSE**        | 2021  | BERT/RoBERTa + Dropout 自监督 | 文本            | ≈BERT                 | 768           | 512            | Bi-Encoder        | 无监督对比学习句向量里程碑                                        |
| **Contriever**    | 2021  | BERT 无监督对比               | 文本            | ~110M                  | 768           | 512            | Bi-Encoder        | Meta；无标注稠密检索                                              |
| **ANCE**          | 2021  | RoBERTa + ANN 动态负例        | 文本            | ~125M                  | 768           | 512            | Bi-Encoder        | 异步 hard negative 更新                                           |
| **ColBERT v2**    | 2021 预印本／2022 正式发表 | BERT 多向量 + 蒸馏            | 文本            | ~110M                  | token×128    | 512            | Late Interaction  | 压缩索引 + 教师蒸馏，工程可用性上升                               |
| **RetroMAE**      | 2022* | 编码器-解码重建预训练         | 文本            | Encoder 级             | —            | 512+           | 预训练底座        | BGE 系 Stage-0；提升表示质量                                      |

\*RetroMAE 论文 2022，作为后续 BGE 预训练组件收录于此。

#### 指令化、稀疏与多维自适应（2021–2023）

| 模型                    | 年份     | 骨干 / 架构              | 模态             | 参数量级                | 维度       | 上下文 | 交互范式                   | Model Card 要点                                 |
| ----------------------- | -------- | ------------------------ | ---------------- | ----------------------- | ---------- | ------ | -------------------------- | ----------------------------------------------- |
| **Instructor**    | 2022     | GTR/T5 系 + 任务指令前缀 | 文本             | 中等 Encoder            | 768        | 512    | Bi-Encoder                 | 330+ 任务统一指令嵌入；「Instruct: …」范式先导 |
| **GTR**           | 2021–22 | T5 Encoder               | 文本             | Base→XXL               | 768        | 512    | Bi-Encoder                 | Google；Encoder-Decoder 家族的检索变体          |
| **SGPT**          | 2022     | GPT 系；双编码器采用位置加权 mean        | 文本             | 125M–5.8B              | 随模型     | 2K 级  | Bi-Encoder（Decoder 骨干） | 证明 Decoder-only 也可做句向量                  |
| **SPLADE v2**     | 2021 预印本 | BERT → 词表稀疏权重     | 文本             | ~BERT                   | \|V\| 稀疏 | 512    | Lexical/Sparse             | 可学习稀疏检索；与 Dense Hybrid                 |
| **MRL**           | 2022     | 训练框架（非单模型）     | 任意             | —                      | 可截断     | —     | —                         | Matryoshka；一向量多尺度（64…4096）            |
| **DiffCSE**       | 2022     | ELECTRA 式差异对比       | 文本             | ≈BERT                  | 768        | 512    | Bi-Encoder                 | 无监督句向量增强变体                            |
| **E5-base/large** | 2022–23 | BERT/XLM-R               | 文本（多语大版） | Base~110M / Large~330M | 768/1024   | 512    | Bi-Encoder                 | MS：「弱监督十亿对 → 监督微调」流水线定型      |

#### 开源文本系列与商业 API、LLM 骨干（2023–2024）

| 模型                                               | 年份       | 骨干 / 架构             | 模态                  | 参数量级                | 维度                | 上下文        | 交互范式                | Model Card 要点                                   |
| -------------------------------------------------- | ---------- | ----------------------- | --------------------- | ----------------------- | ------------------- | ------------- | ----------------------- | ------------------------------------------------- |
| **OpenAI text-embedding-ada-002**            | 2022末–23 | 未公开       | 文本                  | API                     | 1536                | 8K            | Bi-Encoder              | 长上下文 API Embedding 事实标准                   |
| **OpenAI text-embedding-3-small/large**      | 2024.01*   | 未公开；支持缩短输出维度        | 文本                  | API                     | 1536 / 3072（可截） | 8K            | Bi-Encoder              | 性价比与多语显著提升；`dimensions` API          |
| **Cohere embed-english / multilingual-v3.0** | 2023       | 私有 Encoder            | 文本                  | API                     | 1024                | 512           | Bi-Encoder              | 输入类型`search_query` / `search_document`    |
| **E5-mistral-7b-instruct**                   | 2023       | Mistral-7B + 指令       | 文本                  | 7B                      | 4096                | 4K–32K 级    | Bi-Encoder（LLM）       | LLM 骨干嵌入早期标杆；Query 侧加 instruct         |
| **GritLM-7B**                                | 2024*      | Mistral + 统一 Gen/Emb  | 文本                  | 7B                      | 4096                | 4K+           | Bi / 生成一体           | 「同一模型生成+表示」                             |
| **GTE-large / GTE-base**（Alibaba）          | 2023–24   | BERT / RoBERTa 系       | 文本                  | Base~0.1B / Large~0.3B | 768–1024           | 512→8K(v1.5) | Bi-Encoder              | 多阶段对比；后续**gte-Qwen2** 切入 LLM 骨干 |
| **gte-Qwen2-1.5B/7B-instruct**               | 2024       | Qwen2 Decoder           | 文本                  | 1.5B / 7B               | 1536+               | 32K           | Bi-Encoder              | 开源中文/多语强势                                 |
| **Nomic Embed Text v1 / v1.5**               | 2024       | 自定义 Encoder + MRL    | 文本                  | ~137M                   | 768（可截）         | 8K            | Bi-Encoder              | 长上下文开放权重；配套 Vision                     |
| **Snowflake arctic-embed-m/l**               | 2024       | BERT 系压缩训练         | 文本                  | M~110M / L~330M        | 768/1024            | 512           | Bi-Encoder              | 小参数检索强；Apache-2.0                          |
| **Voyage-2 / voyage-large-2**                | 2023–24   | API 专有                | 文本 / 代码分支       | API                     | 1024–1536          | 4K–16K       | Bi-Encoder              | 检索向优化；后有 voyage-code                      |
| **jina-embeddings-v2-base-***                | 2023–24   | JinaBERT / RoBERTa 变体 | 文本（en/de/es/zh…） | ~137–161M              | 768                 | **8K**  | Bi-Encoder              | 早期主打长上下文与分语言版                        |
| **LLM2Vec**                                  | 2024       | 改 LLM→双向 + 对比     | 文本                  | 随底座                  | 随底座              | 随底座        | Bi-Encoder              | 把任意 Decoder LLM 改造成编码器配方               |
| **UAE-Large-V1 / mxbai-embed**               | 2023–24   | BERT-large 系           | 文本                  | ~335M                   | 1024                | 512           | Bi-Encoder              | 社区蒸馏 / AnglE 损失路线                         |
| **bge-reranker-base/large**                  | 2023       | Cross-Encoder           | 文本                  | Base~0.1B / Large~0.5B | 标量分数            | 512           | **Cross-Encoder** | 与 BGE 嵌入配套的精排                             |

同表按系列与架构组织，包含跨年迭代；各版本的精确日期见上面的逐项时间线，不用所在分组代替发布年份。

#### BGE 全家桶（勿只记 M3）

BGE（BAAI General Embedding / FlagEmbedding）是**系列化**产品：中英尺寸档、多语三功能、ICL、多模态、代码、重排器齐全。

| 模型                                       | 发布    | 骨干                       | 模态          | 参数              | 维度         | 上下文       | 范式                 | 要点                                                                                                  |
| ------------------------------------------ | ------- | -------------------------- | ------------- | ----------------- | ------------ | ------------ | -------------------- | ----------------------------------------------------------------------------------------------------- |
| **bge-small/base/large-zh-v1.5**     | 2023    | BERT/RoBERTa 中文          | 文本-中       | 24M / 102M / 326M | 512/768/1024 | 512          | Bi                   | 中文检索工程默认档之一                                                                                |
| **bge-small/base/large-en-v1.5**     | 2023    | BERT 英                    | 文本-英       | 33M / 109M / 335M | 384/768/1024 | 512          | Bi                   | 相似度分布校准；HF 高下载                                                                             |
| **bge-m3**                           | 2024.01 | XLM-R 扩展                 | 文本·100+ 语 | **568M**    | 1024         | **8K** | Dense+Sparse+ColBERT | ★ Multi-Function / Lingual / Granularity；详见《[BGE-M3三功能统一详解](BGE/M3/BGE-M3三功能统一详解报告.md)》 |
| **bge-m3-retromae / unsupervised**   | 2024    | 同系中间 ckpt              | 文本          | —                | —           | 8K           | 预训练/对比中间态    | 复现与二阶段训练用                                                                                    |
| **bge-multilingual-gemma2**          | 2024.07 | Gemma-2-9B                 | 文本·多语    | ~9B               | 3584         | 4K+          | Bi（LLM）            | 大参数多语开源档                                                                                      |
| **bge-en-icl**                       | 2024    | LLM + few-shot 示例        | 文本-英       | LLM 级            | 高维         | 长           | Bi + ICL             | 提示中塞示例提升检索                                                                                  |
| **bge-reranker-v2-m3**               | 2024.03 | XLM-RoBERTa 骨干 Cross-Encoder | 文本·多语 | 约 568M        | 分数         | 8K           | Cross                | 多语精排配套                                                                                          |
| **bge-reranker-v2-gemma / minicpm**  | 2024    | LLM Cross-Encoder          | 文本          | 2B–9B 级         | 分数         | 长           | Cross                | 高精度重排                                                                                            |
| **BGE-VL**（bge-vl-base / llava 系） | 2025    | CLIP / LLaVA-NeXT          | 图+文         | 0.4B–8B 级       | —           | —           | Dual / MLLM-Emb      | 视觉检索、截图搜                                                                                      |
| **BGE-Code-v1**                      | 2025    | 代码专用骨干               | 代码+文       | —                | —           | —           | Bi                   | Code IR / CoIR 向                                                                                     |

> **选型提示**：短中文句对优先 `bge-*-zh-v1.5`；多语长文 + Hybrid → `bge-m3`；要 ICL/指令 → `bge-en-icl` / LLM 档；图文 → `BGE-VL`，不要用纯文本 M3 硬撑跨模态。
>
> **专题深读**：若只熟悉 BERT / 稠密句向量、尚未建立 Sparse 直觉，请先读仓库专题《[BGE-M3三功能统一详解报告](BGE/M3/BGE-M3三功能统一详解报告.md)》（含 Sparse 从零讲解、三 Head 公式、自知识蒸馏与 Hybrid 工程用法）。

#### Jina Embeddings 迭代链（v1 → v5，含 CLIP / Omni）

| 模型                                            | 发布     | 骨干                      | 模态                  | 参数                                        | 维度（MRL）           | 上下文        | 范式                       | 要点                                                           |
| ----------------------------------------------- | -------- | ------------------------- | --------------------- | ------------------------------------------- | --------------------- | ------------- | -------------------------- | -------------------------------------------------------------- |
| **jina-embeddings-v1** | 2023.07 预印本 | T5 Encoder | 文本·英语 | 35M／110M／330M | 按规模 | 按模型卡 | Bi | 清洗文本对、三元组与否定样本，两阶段训练 |
| **jina-embeddings-v2-base-en/de/es/zh…** | 2023–24 | JinaBERT                  | 文本（分语言）        | ~137–161M                                  | 768                   | 8K            | Bi                         | 长上下文 + 分语种卡                                            |
| **jina-clip-v1 / v2** | 2024.05／2024.12 预印本 | v1：JinaBERT + EVA02-base；v2：多语 Jina-XLM-R + EVA02-L/14 | 图+文 | v2 约 0.87B | v1 固定；v2 1024、支持 MRL | 按版本确认 | Dual-Encoder | v1 联合文文／图文；v2 增加多语、文档视觉与嵌套维度 |
| **jina-embeddings-v3**                    | 2024.09  | jina-XLM-RoBERTa-24L      | 文本·89 语           | **570M**                              | 1024→32              | **8K**  | Bi +**任务 LoRA**    | retrieval / classification / clustering / text-matching 适配器 |
| **jina-embeddings-v4**                    | 2025.06  | **Qwen2.5-VL-3B**   | 文+图+PDF             | **3.8B**                              | 2048（亦可多向量）    | **32K** | Bi + Late Interaction 可选 | 统一多模态；task LoRA；Qwen Research License                   |
| **jina-embeddings-v5-text-nano**          | 2026.02  | EuroBERT-210M             | 文本·多语            | **239M**                              | 768→32               | 8K            | Bi + 任务适配              | 蒸馏自 Qwen3-Embedding-4B；小模型 SOTA 档                      |
| **jina-embeddings-v5-text-small**         | 2026.02  | **Qwen3-0.6B-Base** | 文本·119+ 语         | **677M**                              | 1024→32              | **32K** | Bi + 任务适配              | MTEB/MMTEB 小模型顶尖；last-token；Query/Doc 前缀              |
| **jina-embeddings-v5-omni-nano/small**    | 2026.05 预印本 | 锁定文本塔 + 视听投影     | **文/图/视/音** | nano≈**0.95B**；small≈**1.57B**（全模态加载） | 与 text 对齐 768/1024 | 32K           | Locked Aligned Towers      | 文本索引、任意模态查询；几何保持对齐（arXiv:2605.08384）       |

> Jina 线索：**v2 长上下文 → v3 任务 LoRA → v4 MLLM 多模态 → v5 蒸馏小钢炮 + Omni 全模态**。其中 CLIP 是图文双塔，v4 是 VLM 单／多向量，v5-omni 是锁定塔的几何保持对齐，三者计算路径不同。

#### 2024–2026：LLM / MLLM 通用嵌入与 API、文档检索

| 模型                                                               | 年份              | 骨干 / 架构                                       | 模态                          | 参数量级        | 维度                  | 上下文                   | 交互范式         | Model Card 要点                                                                                           |
| ------------------------------------------------------------------ | ----------------- | ------------------------------------------------- | ----------------------------- | --------------- | --------------------- | ------------------------ | ---------------- | --------------------------------------------------------------------------------------------------------- |
| **NV-Embed-v1 / v2**                                         | 2024              | Mistral-7B；双向注意力 + Latent Attention pooling | 文本                          | 7B              | 4096                  | 512–4K+                 | Bi（LLM）        | v2 曾英 MTEB 领先（~72.31）；Hard Negative 工程强                                                         |
| **Stella / Jasper**                                          | 2024              | 多 Teacher 蒸馏 + MRL                             | 文本                          | 0.4B–1.5B 级   | 可变                  | 8K                       | Bi               | 小尺寸高分；社区蒸馏范式                                                                                  |
| **Snowflake arctic-embed-m-v1.5 等**                         | 2024–25          | 持续小模型迭代                                    | 文本                          | ≤330M          | 768+                  | 512                      | Bi               | 检索专用小模型持续刷分                                                                                    |
| **Voyage-3 / 3-large / 4** | 2024–26 | 文本模型系列；Voyage 4-large 为 MoE | 文本；多模态另有产品线 | API／4-nano 开放权重 | 按型号，Voyage 4 支持 256–2048 | 按型号 | Bi | Voyage 4 同系列共享空间，大小编码器可组合；非所有版本均为 MoE |
| **Cohere Embed v4**                                          | 2025              | 统一 Transformer                                  | **图+文**（原生）       | API             | 1024+                 | 长（官方称 128K 级能力） | Bi / 多模态      | 企业多模态检索 API                                                                                        |
| **SigLIP / SigLIP 2**                                        | 2023 / 2025       | ViT + 文本塔；Sigmoid loss                        | 图+文                         | 视觉塔~0.4–2B  | —                    | —                       | Dual-Encoder     | CLIP 改进；SigLIP2 自蒸馏+多语，常作 VLM 视觉塔                                                           |
| **ColPali**                                                  | 2024 预印本／ICLR 2025         | PaliGemma-3B + ColBERT MaxSim                     | **页面图像**            | ~3B             | patch×128            | 页级                     | Late Interaction | 免 OCR 文档页检索；ViDoRe                                                                                 |
| **ColQwen2 / ColQwen2.5 / 社区 ColQwen3** | 2024–25 及后续模型卡 | Qwen2-VL／Qwen2.5-VL／社区 Qwen3-VL 等 | 页面图像 | 随 checkpoint | 多向量 | 页级，分辨率与 token 预算按版本 | Late Interaction | 官方系列与社区版本分开，逐一核对骨干、训练与许可 |
| **VLM2Vec / V2**                                             | 2024–25          | LLaVA / Qwen2-VL 等                               | 图文交错                      | 2B–7B          | —                    | —                       | MLLM Bi-Encoder  | MMEB 路线开创者之一                                                                                       |
| **GME / BGE-VL / E5-V**                                      | 2024–25          | Qwen2-VL / LLaVA-NeXT                             | 图+文                         | 2B–8B          | —                    | —                       | MLLM-Emb         | 通用多模态嵌入族                                                                                          |
| **Qwen3-Embedding-0.6B/4B/8B**                               | 2025              | **Qwen3** Decoder                           | 文本·100+ 语                 | 0.6–8B         | 32–4096 MRL          | 32K                      | Bi（LLM）        | ★ 开源综合榜前列；指令前缀                                                                               |
| **Qwen3-VL-Embedding**                                       | 2026.01 预印本          | Qwen3-VL                                          | 图文/视频文档                 | VL 级           | —                    | 长                       | MLLM Bi          | 通义多模态嵌入产品线                                                                                      |
| **Seed-1.5-Embedding**                                       | 2025              | Seed 系                                           | 文为主（及演变）              | API             | —                    | —                       | Bi               | 豆包向量前代；C-MTEB 竞争                                                                                 |
| **Seed-1.6-Embedding**（`doubao-embedding-vision-250615`） | **2025.06** | **Seed1.6-Flash**；双塔取 **[EOS]**   | **文+图+视频** 混合检索 | API（火山引擎） | **2048 / 1024** | 随底座                   | Dual/Bi（MLLM）  | ★ C-MTEB ~75.6 SOTA；MMEB-v2 Image 77.78 / Video 领先；见[官方说明](https://seed1-6-embedding.github.io/) |
| **Conan-embedding-v2** | 2025.09 报告；含此前结果快照 | 从零训练嵌入向 LLM + soft-masking | 文本／跨语检索 | 约 1.4B | 3584 | 32K | Bi | 数据与动态难例共同训练                                                                           |
| **KaLM-Embedding / V2** | 2025 | Qwen 系小 LLM，按版本区分 | 文本·多语 | 小于 1B 的系列版本 | 按 checkpoint | 按 checkpoint | Bi | 数据清洗、多样性与训练技巧；不是 EuroBERT 系                                                                                |
| **Gemini Embedding / Embedding 2**                           | 2025 报告／2026.03 二代预览          | Google 专有                                       | 一代文本；二代图／文／音／视频／文档          | API             | —                    | —                       | Bi               | 检索子集高分；生态绑定 Google                                                                             |
| **OpenAI / Google / 平台多模态 Embedding**                   | 2025–26          | 闭源                                              | 图文等                        | API             | 可变                  | 长                       | Bi               | 与开源 MLLM-Emb 并行                                                                                      |

#### 读时间线的正确方式

1. **短表定位，逐项回查，卡片比较**：先看范式，随后沿具体模型与方法理解演进；窗口、维度、规模和实验条件再到卡片及独立研读核对。
2. **系列和任务分别展开**：BGE 的文本、M3、ICL、VL、Code、Reranker 不是同一个模型；Jina 的 CLIP、单向量、多向量、任务适配和全模态也有不同计算路径。
3. **区分三种日期**：论文首发、模型／API 发布、会议正式发表都可独立成为节点。例如 ColPali 有 2024 年预印本与 ICLR 2025，E5-Mistral 的首次提交与 arXiv 编号跨年。
4. **骨干不决定交互**：Qwen3-Embedding 用 Decoder 骨干做 Bi-Encoder，ColPali 用 VLM 做 Late Interaction；不能把“用了大模型”都归成 Cross-Encoder。
5. **开放程度逐个版本确认**：API 提供托管服务，开放权重便于自行部署，但训练代码、训练数据和商用许可仍需分别检查。Voyage 4-nano 等开放版本也不能因同系列有 API 就归为全闭源。
6. **并行路线按问题比较**：图文双塔、MLLM 单向量与页面多向量并存；质量结论需要任务、输入预算、候选库和评测版本一致，不能仅按发布时间排序。

### 各阶段详解（演进因果，而不只是年表）

读本节时盯住一条链：**上一时代解决了什么 → 留下什么硬伤 → 下一时代用什么机制替换**。型号名单仍以 “完整时间线” 的 model card 为准。

#### 稀疏统计时代（1954–2012）

| 方法              | 核心思想           | 向量特点                  | 典型应用         |
| ----------------- | ------------------ | ------------------------- | ---------------- |
| **BoW**     | 词频计数           | 维度 = 词表大小，极度稀疏 | 文本分类基线     |
| **TF-IDF**  | 词频 × 逆文档频率 | 抑制常见词权重            | 搜索引擎、LSH    |
| **LSA/LSI** | SVD 分解共现矩阵   | 约 100–300 维稠密        | 主题、去重       |
| **BM25**    | 概率检索打分       | 非稠密向量                | 至今 Hybrid 基线 |

**动机**：用可计算的词统计近似「相关性」。
**机制贡献**：可解释、可倒排、工程极便宜。
**硬伤**：同义/上下位无法对齐（「糖尿病」≠「血糖」字面）；维度随词表爆炸。
**为何持续使用**：精确关键词、SKU、错误码、法规条款号可能更依赖词法通道。Dense + BM25/Sparse 是否有互补收益，应以同一候选库上的相关性和成本实验判断。

#### 静态词嵌入时代（2013–2018）

分布假说（Harris）被神经方法做成低维稠密：Word2Vec 的 Skip-gram / CBOW + Negative Sampling，把「上下文预测」变成可反向传播的目标。

$$
\mathcal{L}_{\text{SkipGram}} \approx -\sum_{(w,c)}\log \sigma(v_w^\top v_c) - \sum_{(w,c^-)}\log \sigma(-v_w^\top v_{c^-})
$$

| 模型     | 创新                         | 维度     | 训练数据        |
| -------- | ---------------------------- | -------- | --------------- |
| Word2Vec | Skip-gram/CBOW + NegSampling | 100–300 | Google News 等  |
| GloVe    | 全局共现 + 加权最小二乘      | 50–300  | Wiki + Gigaword |
| FastText | 子词 n-gram                  | 300      | Common Crawl    |

**解决了什么**：语义类比、迁移到分类/NER 的初始化、OOV（FastText）缓解。
**硬伤**：一词一向量——「bank」银行/河岸共用一点；句子级只能平均词向量，句法与否定脆弱。
**如何被替代**：上下文模型（ELMo/BERT）让同一词型在不同句子中落到不同点。

#### 上下文嵌入与专用句向量（2018–2022）

**ELMo**：双向 LSTM 层加权 → 动态词向量，证明「上下文条件化」有效。
**BERT**：Encoder + MLM，成为通用编码器底座；但原生 `[CLS]`/mean **不是**为句相似与检索优化的。

**矛盾**：用 BERT 做句对相似度，若走 Cross-Encoder（两句拼进模型），精度高但无法预计算索引；若天真取 `[CLS]` 做 Bi-Encoder，精度塌方。

**Sentence-BERT（2019）** 的因果位置：用 Siamese/Triplet + 对比/回归损失，把 BERT 压成可 ANN 的句向量，推理相对 Cross-Encoder 快数量级，**专用 Embedding 产业从这里起步**。
```

[Sentence A] ── BERT ── Pooling ── u ──┐
                                        ├── CosineSimilarity(u, v)
[Sentence B] ── BERT ── Pooling ── v ──┘

```

随后分叉三条至今仍在用的线：

1. **Dense Retrieval**（DPR, 2020）：双塔 + 标注段落 + in-batch 负例，FAISS 进开放域 QA；
2. **Late Interaction**（ColBERT, 2020）：用多向量换回细粒度交互，精度逼近 CE、索引仍可预计算；
3. **无监督/弱监督对比**（SimCSE, Contriever）：降低对标注三元组的依赖。

**硬伤（进入 2022 前）**：512 上下文、跨语言与指令任务分裂、单向量对专有名词不稳、多模态另起炉灶（CLIP 2021 已开一条平行线）。

#### 指令化、稀疏神经检索与课表化训练（2022–2023）

- **Instructor / E5**：把「任务描述」写进 query 侧，一套权重服务多任务；弱监督十亿对 → 监督微调成为可复制课表。
- **SPLADE**：词法匹配神经化，Dense 丢关键词时的补丁变成可学习 Sparse。
- **MRL**：一向量多截断，部署侧按内存换精度。
- **商业 API**（ada-002 等）把 Embedding 变成基础设施计费项。

**因果**：问题从「有没有句向量」变成「**怎么规模化数据与接口契约**」。

#### LLM 骨干与系列化产品（2023–）

| 维度   | BERT 系 Embedding         | LLM 系 Embedding               |
| ------ | ------------------------- | ------------------------------ |
| 骨干   | Encoder 110M–560M        | Decoder 0.6B–8B+              |
| 预训练 | MLM                       | CLM（再改编码/双向）           |
| 池化   | CLS / mean                | last token / EOS / latent attn |
| 数据   | 百万～亿对                | 十亿弱监督 + 合成 + 难负例     |
| 指令   | 可选                      | 多数成为契约                   |
| 上下文 | 512 → 8K（部分 Encoder） | 8K–32K 常见                   |

**为何换骨干**：生成式 LLM 的语义与长上下文迁移到表示任务；合成 query 便宜。
**并未改变的事实**：主流仍是 **Bi-Encoder Dense**；LLM 只是更强的 $f_\theta$，不是自动变成 Cross-Encoder。
**系列化**：BGE（v1.5→M3→ICL→VL/Code）、Jina（v2→v5）、GTE→gte-Qwen、Qwen3-Emb、Seed API——竞争从「单点刷榜」转为「产品线 + 可私有化 + 多模态」。

代表机制钩子（名单细节见 “完整时间线”）：E5-mistral / NV-Embed（双向注意力与池化创新）、BGE-M3（三功能统一）、ColPali（视觉文档多向量）、Qwen3 / Seed（开源与 API 两端的多模态嵌入）。

#### 一张因果简图
```

BoW/BM25 ──(语义不足)──► Word2Vec ──(一词多义)──► ELMo/BERT
                              │
                              ▼
                     SBERT/DPR Dense 双塔
                      │           │
         (精度不够)    │           │ (关键词漏)
                      ▼           ▼
                 ColBERT/CE    SPLADE/Hybrid
                      │
         (长上下文/多任务/合成数据)
                      ▼
              Instruct + E5 课表 + LLM 骨干
                      │
         (文档版式 / 图文统一)
                      ▼
         CLIP 双塔 · MLLM-Emb · ColPali 多向量

```

---

## 理论基础与表示范式

本章回答三个问题：**向量空间在优化什么**、**表示长什么样（Dense / Sparse / Multi-Vector）**、**Query 与 Document 何时交互（Bi / Cross / Late）**。后文训练、评测与选型都建立在这三套坐标系上。

### Embedding 的数学定义

给定输入空间 $\mathcal{X}$（词、句、段落、图像、页面截图等），参数化映射

$$
f_\theta: \mathcal{X} \rightarrow \mathbb{R}^d, \quad v = f_\theta(x)
$$

把离散对象变成连续向量。写全一点，$f_\theta$ 通常是两段：

1. **骨干**产出 token（或 patch）级隐状态 $H\in\mathbb{R}^{n\times h}$；
2. **池化 / 读出头**把 $H$ 收成固定维 $v\in\mathbb{R}^d$（必要时再投影）。

检索与相似度任务通常在**归一化后的内积空间**上工作：先令 $\hat{v}=v/\|v\|_2$，再定义

$$
\text{sim}(x,y)=\hat{v}_x^\top \hat{v}_y=\cos(\theta_{xy}).
$$

此时余弦与点积数值相同；ANN 的 `IP` 索引与「先 L2 归一化再搜」是同一契约。训练未归一化、评测却用余弦（或反过来），会出现 loss 好看、线上召回崩——这是最常见的工程错位。

#### 池化：从 token 状态到一条向量（常被跳过的环节）

| 读出方式                       | 做法                              | 常见于                        | 要注意                                       |
| ------------------------------ | --------------------------------- | ----------------------------- | -------------------------------------------- |
| **[CLS] / 专用标记**     | 取位置 0 或特殊 token             | 早期 BERT 句向量              | 未为检索微调时表征弱                         |
| **Mean pooling**         | 对非 padding 位置取均值           | SBERT、多数 Encoder Embedding | 长文会被稀释；需正确 mask                    |
| **Last token / EOS**     | 取序列末隐状态                    | 多数 Decoder LLM Embedding    | 依赖因果掩码下「末位汇聚」；改双向后语义会变 |
| **Latent attention**     | 可学习 query 对$H$ 做注意力聚合 | NV-Embed                      | 多一套参数，常强于死板 mean                  |
| **多向量（不池化成 1）** | 保留每 token/ patch 向量          | ColBERT / ColPali             | 表示范式已变，见 “Multi-Vector：为何是 MaxSim”                     |

**断环警告**：只说「用了 Qwen3 骨干」却不说读出方式，复现实验会对不齐。换 mean ↔ last-token 等于换了一个模型头。

#### 对比学习在优化什么（InfoNCE 在算什么）

对每个 query $q$，有正样本 $d^+$ 与负样本 $\{d^-_k\}$。令 $s_i=\text{sim}(q,d_i)/\tau$，InfoNCE 就是把「正确文档」当成类别的交叉熵：

$$
\mathcal{L}_{\text{InfoNCE}}=-\log\frac{e^{s_+}}{e^{s_+}+\sum_k e^{s_k^-}}=-\log\,p(d^+\!\mid q;\text{候选集}).
$$

要点：

- **优化的是相对排序，不是向量绝对值**。换负例池，同一正对的「最优嵌入」可以变。
- **$\tau$**：越小分布越尖，难负例梯度更大；过大则「谁都差不多」，细粒度分不开。
- **in-batch 负例**：把同一 batch 里别人的正文档当自己的负例，等价于用很大的候选集做上述 CE，几乎零额外编码成本。代价是：batch 里若混进近重复 / 同文档切块，会制造**假负例**（“Hard Negative Mining”）。
- SigLIP 改成**逐对 sigmoid**，不做「全体负例上的 softmax」，大规模多模态时更稳；$\tau$ 与 InfoNCE **不可直接照搬**。

**错误用法**：把 Embedding 训练理解成「回归到某个固定教师向量」。可以做向量蒸馏（“蒸馏信号” / “Embedding 蒸馏与压缩”），但那是另一条损失；主路径对比学习并不要求绝对值对齐。

#### 非对称编码

生产检索里，Query 与 Document 往往**不应共用同一前缀**：

| 侧       | 常见处理                    | 原因                   |
| -------- | --------------------------- | ---------------------- |
| Query    | 任务指令 /`search_query`  | 短、意图强、要对齐任务 |
| Document | 纯正文 /`search_document` | 长、必须能离线预计算   |
| 两塔参数 | 共享骨干或 DPR 式双塔       | 共享省参；独立可专化   |

E5-instruct、Cohere embed-v3、多数 LLM Embedding 把非对称写成接口契约。模型若**按指令训练**，评测/线上漏 instruct，分数会明显掉——掉多少取决于该权重对指令的依赖，不是魔法常数。对称任务（STS、去重）通常**两侧同模板或都无指令**，与检索课表不要混用同一套字符串。

### 三种表示范式

表示范式回答：**一条文档在索引里存成什么**。它与「交互架构」（“三种交互架构”）正交：同一套骨干既可以读出 Dense，也可以接 Sparse Head 或保留多向量。
```

Dense        → 1 × d 稠密向量     （ANN 友好，信息瓶颈）
Sparse       → |V| 维词项权重     （倒排友好，精确匹配）
Multi-Vector → n × d' token 向量 （MaxSim，细粒度）

```

#### Dense：单向量信息瓶颈

$$
\mathrm{TopK}(q)=\underset{d\in\mathcal{C}}{\operatorname{arg\,top\text{-}k}}\;\text{sim}(f(q),f(d)).
$$

**够用**：开放域段落、FAQ、多数 RAG chunk、聚类/去重。
**不够**：查询词必须命中（合规/SKU/条款号）、版式敏感扫描件、超长文档内定位 → Hybrid 或 Multi-Vector。

**反例**：384 维通用句向量搜「合同第 12.3 条」——近邻常是同主题错条款。加大 $d$ 往往解决不了，要加词法通道。

#### Sparse：神经化的词法匹配

经典 BM25 用统计公式给词项加权。SPLADE 一类方法让 Encoder 对每个输入预测**词表上的权重**（常见实现：在 MLM 词表 logits 上做 $\mathrm{ReLU}(\cdot)$ / $\log(1+\mathrm{ReLU})$，再对 token 维聚合），得到稀疏向量 $w(d)\in\mathbb{R}^{|V|}$，分数为

$$
s_{\text{lex}}(q,d)=\sum_{t\in V} w_t(q)\,w_t(d).
$$

SPLADE 可以为未在原文出现的词预测权重，起到学习式词项扩展作用。稀疏正则约束激活范围和倒排访问成本，不能只看向量里有多少非零元素：高频词项的长倒排表仍可能使查询变慢。

BGE-M3 Sparse 与 Dense、ColBERT 头共享 Encoder，但不是 SPLADE 的完整词表预测：它对输入中实际出现的 token 产生权重，再按词项聚合，因此没有同样的隐式词表扩展机制。三条路线应分别理解：

| 路线 | 词项从哪里来 | 权重怎样产生 | 索引与代价 |
| --- | --- | --- | --- |
| BM25 | 分词后的原文词项 | 词频、逆文档频率和长度归一化 | 倒排；提供词法匹配基线 |
| SPLADE / v2 | MLM 词表，可激活原文未出现的词 | 神经词项权重、聚合与稀疏约束 | 倒排；扩展能力换取额外计算和存储 |
| BGE-M3 Sparse | 输入 token 对应的词项 | 共享骨干上的非负权重头 | 倒排；可与同模型 Dense 和多向量融合 |

详见 [SPLADE](SPLADE/SPLADE/SPLADE详解.md)、[SPLADE v2](SPLADE/SPLADE-v2/SPLADE-v2详解.md) 与 [BGE-M3](BGE/M3/BGE-M3三功能统一详解报告.md)。字符、子词和词项的粒度由具体 tokenizer 与索引实现决定，不能拿普通中文分词后的词项 ID 直接替代模型词表。

**不是**「另一种 1024 维 Dense」。**错误用法**：把 $|V|$ 维稀疏向量丢进 FAISS 做 L2。

#### Multi-Vector：为何是 MaxSim

Query / Doc 产出 $E_q\in\mathbb{R}^{|q|\times d'},\;E_d\in\mathbb{R}^{|d|\times d'}$。若对所有 token 对做全连接再求和，既贵又易被文档长度淹没。ColBERT 的 MaxSim：

$$
s(q,d)=\sum_{i=1}^{|q|}\max_{j=1}^{|d|} \hat{E}_q[i]^\top \hat{E}_d[j]
$$

含义：**每个 query token 在文档里找最匹配的一处**，再累加。这保留「谁对上了哪」的细粒度，又比 Cross-Encoder 便宜（文档向量可离线存）。ColPali 把 token 换成页面图像 patch，思想同一类。

相对 Dense：更贵的存储与检索算子（PLAID 等）。相对 Cross：文档仍可预计算。

### 多向量的表示、训练与检索执行

MaxSim 是非对称的集合相似度：每个查询向量选择文档中的最佳匹配，同一文档位置可以被多个查询 token 复用。这保留局部证据，但不是一对一对齐或逻辑约束验证；高分也不自动证明数量、否定和关系均得到满足。

![PLAID 原文图：图 1（原文 Figure 1）：后期交互架构](PLAID/figs/fig01.png)

图中查询与段落各自编码，查询中的每个 token 在段落里寻找最佳匹配，最后聚合为相关性分数。它说明的是后期交互计算图；PLAID 随后用分阶段筛选减少进入完整评分的候选，具体执行流程见独立研读。

| 工作 | 主要改变什么 | 与其他层的区别 |
| --- | --- | --- |
| [ColBERT](ColBERT/ColBERT详解.md) | 独立编码 token 表示，再用 MaxSim 算相关性 | 定义表示与交互，文档可离线编码 |
| [ColBERTv2](ColBERTv2/ColBERTv2详解.md) | 更强监督与残差压缩 | 既涉及训练，也减少多向量存储 |
| [PLAID](PLAID/PLAID详解.md) | 利用质心交互、剪枝与分阶段评分执行检索 | 属于检索引擎优化，不是新的编码骨干 |
| [ColPali](ColPali/ColPali详解.md) / [ColQwen](ColQwen/ColQwen系列详解.md) | 将文档表示扩展为页面视觉 token | 任务和输入变化；仍须解决存储、候选召回与评分 |

上述原论文的关键方法图、完整实验和设置集中在各篇研读。三种优化轴要分开：减少文档向量个数、减少单向量维度、压缩每维位数；引擎还可以减少需要解压或评分的候选。若保留全量向量给后续精排，最终总存储并不等于短表示索引的体积。

多向量既可以用专用引擎做全库召回，也可对已有候选评分。只评测 BM25 Top-1000 上的 MaxSim 不能证明全库检索能力；候选进入精排之前的漏召回必须单独测量。页面检索还应记录分辨率、视觉 token 数和是否合并 patch，因为它们同时改变质量、编码成本与索引大小。

XTR、WARP 及 MUVERA 是此问题的进一步阅读，尚未在本仓库完成独立研读，具体出处见[资料清单](资料清单_论文与博客.md#方法评测与工程补充阅读)。它们分别从 token 召回及估分、引擎执行、固定维近似等角度研究成本，不能统称为“又一种向量压缩”。

### 三种交互架构

交互架构回答：**算分时，注意力有没有同时看见 q 与 d**。

| 架构                       | 编码           | 交互           | 速度       | 精度       | 代表               |
| -------------------------- | -------------- | -------------- | ---------- | ---------- | ------------------ |
| **Bi-Encoder**       | q、d 各算一次  | 向量点积/余弦  | ★★★★★ | ★★★     | E5, BGE, DPR, CLIP |
| **Cross-Encoder**    | `[q;d]` 联合 | 全层交叉注意力 | ★         | ★★★★★ | BGE-Reranker       |
| **Late Interaction** | 各算多向量     | MaxSim         | ★★★     | ★★★★   | ColBERT, ColPali   |

#### 为什么 Bi 通常打不过 Cross（机制，不是玄学）

Cross 在每一层都可以让「查询里的否定/数量/约束」直接 attend 到「文档里的对应片段」。Bi 必须在**互不看见对方**的前提下，把全部检索相关信息压进一个（或一组）向量，再靠点积碰运气。

具体失败型：query「不含糖的酸奶」vs 文档大谈「酸奶、蔗糖」——Cross 容易靠 token 级交互判不相关；Bi 的单向量常仍因「酸奶」主题靠近而排前。这就是生产上 **Bi/Hybrid 召回 + Cross 精排** 的根由，不是「多堆一个模型好看」。

**复杂度**：Cross 对全库打分是 $O(|\mathcal{C}|)$，不可扩展；只对召回 Top-K 精排才合理。

#### FAQ：骨干 ≠ 表示 ≠ 交互

| 概念 | 问什么            | 例子                       |
| ---- | ----------------- | -------------------------- |
| 骨干 | 预训练谁          | BERT / Qwen / SigLIP       |
| 表示 | 索引存什么        | Dense / Sparse / Multi-Vec |
| 交互 | q–d 何时注意力   | Bi / Cross / Late          |
| 读出 | $H\to v$ 怎么做 | mean / EOS / latent        |

「上了 LLM Embedding」≠「变成了 Cross-Encoder」。Qwen3-Embedding 仍是 Bi+Dense；ColPali 骨干是 VLM，交互是 Late。

#### Fusion ≈ Rerank

多模态 **Fusion Encoder**（图文 token 进同一 Transformer 出标量分）与文本 **Cross-Encoder Rerank** 是同一计算图：**联合编码 → 标量相关性**。差别在模态与数据，不在「算不算 Embedding」。可 ANN 召回 → 必须 Bi 或 Late；只打候选 → Cross/Fusion。

### 相似度度量与索引约定

| 度量              | 适用                       |
| ----------------- | -------------------------- |
| 余弦 / 归一化点积 | 默认检索                   |
| 未归一化点积      | 须与训练一致；幅度会进分数 |
| 欧氏距离          | 聚类等；与余弦空间不要混用 |
| MaxSim            | 多向量                     |
| 稀疏点积          | SPLADE / BGE-M3 Sparse     |

训练、评测、向量库三处契约必须一致。

### 小结

1. 先定任务：对称相似 vs 非对称检索 → 再定损失与模板（“训练方法与数据工程”）。
2. 默认：Dense Bi +（可选）BM25/Sparse + Cross 精排。
3. 关键词/合规差 → 加 Sparse，而不是只加维。
4. 版式/无 OCR → Multi-Vector（ColPali 类），而不是迷信「先 OCR 再 Dense」。
5. 写清：骨干、读出、表示、交互四件套；缺一不可复现。

---

## 架构分类体系

“理论基础与表示范式” 已把**表示**与**交互**讲清。本章只补一张「选型坐标」：同一模型要在四个轴上同时定点，避免只报参数量。

### 四轴坐标（比单表有用）

| 轴             | 选项                                  | 决定什么                       |
| -------------- | ------------------------------------- | ------------------------------ |
| **骨干** | Encoder / Decoder-LLM / Enc-Dec / VLM | 上下文长度、语义上限、微调成本 |
| **读出** | CLS / mean / EOS / latent / 多向量    | 向量怎么从$H$ 来（“池化：从 token 状态到一条向量（常被跳过的环节）”）  |
| **表示** | Dense / Sparse / Multi-Vec（可组合）  | 索引形态与存储                 |
| **交互** | Bi / Cross / Late                     | 能否预计算、精排位置           |

**正交举例**：

| 模型            | 骨干        | 读出              | 表示                | 交互                    |
| --------------- | ----------- | ----------------- | ------------------- | ----------------------- |
| BGE-base        | Encoder     | CLS/mean 系       | Dense               | Bi                      |
| BGE-M3          | Encoder     | 多 Head           | Dense+Sparse+Multi  | Bi + Late（ColBERT 头） |
| Qwen3-Embedding | Decoder LLM | 指令 + 末位类读出 | Dense（常带 MRL）   | Bi                      |
| BGE-Reranker    | Encoder     | 分类头标量        | —（不算库向量）    | Cross                   |
| ColPali         | VLM         | patch 多向量      | Multi-Vec           | Late                    |
| CLIP            | ViT+Text    | 塔顶投影          | Dense（两模态对齐） | Bi（双塔）              |

### 骨干怎么选（机制，不是品牌）

| 骨干                           | 适合                                 | 不适合硬扛                                       |
| ------------------------------ | ------------------------------------ | ------------------------------------------------ |
| **Encoder ≤0.6B**       | 私有化 RAG、CPU/边缘、要 Sparse/多头 | 极长上下文、复杂指令跟随                         |
| **Decoder LLM 1.5B–8B** | 长上下文、指令式多任务、合成数据友好 | 极致 QPS / 端侧（除非蒸馏到学生）                |
| **VLM**                  | 图文统一、视觉文档                   | 纯文本检索的性价比（通常不如专用文本 Embedding） |

部署形态（API / 自托管 / 量化）是**运维轴**，不改变上面四轴；量化只动精度与吞吐，不把 Bi 变成 Cross。

---

## 训练方法与数据工程

本章目标：读完后能搭出一条**可照着做的课表**——弱监督 → 监督 + hard negative →（可选）多任务 / 蒸馏 / MRL——并知道每一步在优化什么、常见翻车点在哪。难负例的工业细节（刷新节奏、假负例审计、评测回归）下沉专题《[难负例挖掘工业实践](难负例挖掘工业实践.md)》；此处给主文级深度。

### 训练 Pipeline 总览
```

Stage 0: 骨干预训练 (CLM/MLM/图文对比, 通用语料)
    ↓
Stage 1: 弱监督对比预训练 (十亿级 pairs, InfoNCE / Sigmoid)
    ↓
Stage 2: 监督微调 (MS MARCO, NLI, QA + Hard Negatives)
    ↓
Stage 3: 多任务混合 (Retrieval + STS + Classification + Clustering)
    ↓
Stage 4 (可选): 蒸馏 / MRL / 领域适配 / 指令对齐

```

| 阶段    | 数据噪声     | 负例策略               | 典型步数/规模 | 失败模式               |
| ------- | ------------ | ---------------------- | ------------- | ---------------------- |
| Stage 1 | 高（弱监督） | 以 in-batch 为主       | 亿～十亿对    | 假正例把空间学糊       |
| Stage 2 | 低           | BM25/CE/Teacher 难负例 | 百万级标注    | 假负例 → 过推、泛化差 |
| Stage 3 | 中           | 任务混合采样           | 视任务表      | 检索被 STS 稀释        |
| Stage 4 | —           | 教师分布 / 截维损失    | 较短          | 蒸馏温度与教师错配     |

**课表原则**：先用脏数据把空间「撑开」，再用干净数据 + 难负例「削尖」；领域适配尽量在 Stage 2/4 做，而不是一上来就用小领域数据从头训。

### 核心损失函数

先分清任务，再选损失——这是最容易选错的一步。

| 任务形态                        | 标签长什么样                      | 常用损失                          | 说明                                  |
| ------------------------------- | --------------------------------- | --------------------------------- | ------------------------------------- |
| **检索 / 「这一对是正」** | $(q,d^+)$，负例 implicit 或显式 | **MNRL / InfoNCE**          | 生产检索主路径                        |
| **相似度回归**            | 句对 + 连续分（0–1 / 0–5）      | **CosineSimilarityLoss** 等 | STS；**不是**开放域检索默认损失 |
| **三元组**                | $(q,d^+,d^-)$                   | Triplet / Margin                  | 负例质量决定上限                      |
| **教师蒸馏**              | 教师分数 / 序 / 向量              | KL / MSE / cosine                 | “蒸馏信号”、“Embedding 蒸馏与压缩”                         |

**常见错配**：做 RAG 却用 CosineSimilarityLoss、且 batch 里没有检索负例 → 验证集 STS 涨、Recall 不涨；或只有连续相似度标签却硬上 MNRL。

#### InfoNCE

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(q, d^+) / \tau)}{\exp(\text{sim}(q, d^+) / \tau) + \sum_{d^-} \exp(\text{sim}(q, d^-) / \tau)}
$$

即 “对比学习在优化什么（InfoNCE 在算什么）”「候选集上的 CE」。$\tau$ 文本检索常见 $0.01$–$0.05$。

#### MNRL（in-batch 负例）

Batch 内配对 $(q_i,d_i)$：对 $q_i$，除 $d_i$ 外的 $\{d_j\}$ 皆为负。弱监督与检索微调主力。同文档切块 / 近重复进同一 batch → 假负例；要去重或分桶。

#### Sigmoid（SigLIP 风格）

逐对 logistic，不做全体负例 softmax。多模态大 batch 更稳；$\tau$ 与 InfoNCE 不通用。

#### MRL

$$
\mathcal{L}_{\text{MRL}} = \sum_{m \in \{64, 128, 256, 512, \ldots\}} w_m \cdot \mathcal{L}(v_{1:m})
$$

截断维是否够用，以自有检索集为准，不要背外部「保留百分之几」的传闻。

#### 蒸馏信号

$\mathrm{KL}(P_T\|P_S)$ / $\mathrm{MSE}(s_T,s_S)$ / $1-\cos(v_T,v_S)$：蒸排序、蒸分数、蒸向量。与 LLM token-KD 不是一套（“Embedding 蒸馏与压缩”）。

### Hard Negative Mining

**难负例**：模型容易排高、业务判定不相关。
**假负例**：其实相关（或应相关），却被当负例推开。

没有难负例，Stage 2 往往只是在推「显而易见的不相关」。假负例过多：loss 降、邻域被撕碎，近义召回或 STS 变差。

| 方法                | 机制             | 主要风险      |
| ------------------- | ---------------- | ------------- |
| Random / BM25       | 易或偏词法       | 语义难例不足  |
| In-batch            | 便宜             | 假负例        |
| Teacher top-k       | 语义难例         | 混假负例      |
| CE / Positive-aware | 过滤更准         | 贵 / 阈值要调 |
| Dynamic / Cross-GPU | 训练中加难、扩池 | 工程复杂      |

**可操作一层（positive-aware）**：Teacher 候选里，若分数 ≥ 正样本分 − margin，则不当负例。每 query 留 2–4 个真正难负例通常够用。

**例子**：query「如何重置路由器」；正文档=官方重置步骤。「如何重置手机」主题近、答案错 → 好难负例。同型号另一篇官方 FAQ 变体被标负 → 假负例，推开会伤召回。

刷新节奏、假负例治理与评测回归 → 《[难负例挖掘工业实践](难负例挖掘工业实践.md)》。

### 数据

| 策略        | 来源             | 主要风险                             |
| ----------- | ---------------- | ------------------------------------ |
| Natural     | NLI / QA / 复述  | 与「检索相关」不同构                 |
| Weak        | 标题–正文、点击 | **假正例**（标题党）把空间学糊 |
| Synthetic   | LLM 写 query     | 模板塌缩；污染测试集                 |
| Self-mining | 自检索           | 自我确认                             |
| Multi-task  | 指令任务表       | 检索被 STS 采样稀释                  |

E5 压缩课表：从大规模弱配对池筛选训练对 → 弱监督对比预训练 → 监督 + 负例；query／passage 角色前缀与后续 instruct 版本分别确认。BGE-M3：RetroMAE → 对比 → 三头 + Self-KD（见专题）。

合成数据应核对任务语义与模板、实体及长短查询覆盖、多语言分层，并排除测试查询／标签进入生成提示。是否允许使用评测语料作无标签适配，应按后文评测协议声明。

### 指令编码与查询文档非对称

查询和文档扮演不同角色；非对称可以来自参数、模板、输入长度和训练目标，不等于必须训练两套完全独立的模型。DPR 使用独立的查询／段落编码器，E5 等可以共享骨干并用角色前缀区分输入，部分指令模型只在查询侧添加自然语言任务描述。

| 做法 | 改变哪一层 | 需要确认什么 |
| --- | --- | --- |
| `query:` / `passage:` 等角色前缀 | 输入格式与角色识别 | 是否是该 checkpoint 实际训练约定 |
| 自然语言任务指令 | 编码时需要关注的相关性标准 | 是否用这类指令训练，改写后是否稳定 |
| 两套编码器参数 | 查询和候选的表示函数 | 输出空间是否联合对齐，换一侧是否仍兼容 |
| ICL 检索示例 | 指令输入中的任务演示 | 示例开销、顺序、长度预算与数据泄漏 |

以 E5-Mistral 类查询模板说明接口形态，其他模型不能照抄：

```text
Query:    Instruct: {task_description}
          Query: {query_text}
Document: {document_text}
```

指令需要进入模型实际接受训练的格式，并遵循对应的 tokenizer、特殊 token 和池化约定。BGE-EN-ICL 的示例也是编码输入，不是调用生成式 LLM 回答问题。见 [E5-Mistral](E5-Mistral/zh.md)、[INSTRUCTOR](INSTRUCTOR/INSTRUCTOR详解.md)、[BGE-EN-ICL](BGE/EN-ICL/BGE-EN-ICL详解.md) 与 [Qwen3-Embedding](Qwen3-Embedding/zh.md)。

**相关性改变与主题相似不同。** 查询“关于电动车电池的文章”可以命中很多主题相近文档；指令改成“仅保留讨论回收而非制造的文章”后，排序应随约束变化。[FollowIR](FollowIR/FollowIR详解.md) 专门检验这样的指令变化；在常规检索榜上分数高，并不保证模型遵守否定或排除条件。

训练与线上须遵循各自 checkpoint 的同一输入契约，但教师和学生不必使用完全相同的字符串：它们可能有不同的官方模板，要求的是同一任务语义和候选定义。文档模板、读出或骨干变化后，原库向量通常也要重算。对称任务能否去掉指令同样依训练设置决定，不能把所有检索模型机械改成两侧裸文本。

### 旋钮与判据

| 旋钮       | 起点                         | 怎么看                       |
| ---------- | ---------------------------- | ---------------------------- |
| $\tau$   | 0.01–0.05                   | 难例分不开略降；噪声过敏略升 |
| batch      | 尽量大                       | 更多 in-batch 负例           |
| hard neg/q | 2–4                         | 未过滤时宁少勿滥             |
| lr         | Encoder ~1e-5；LLM 更小/LoRA | 冲掉预训练则降               |
| 阶段切换   | 领域验证平台期               | 不要死磕固定 epoch           |

**最小可行**：开源 Embedding 续训 → 领域弱配对 → 正对 + 过滤难负例至验证饱和 → 可选 MRL/蒸馏。垂直域案例与文搜图清单见《[领域专用 Embedding 适配实践](领域专用Embedding适配实践.md)》。

**训练中看什么**：检索看 Recall@K / nDCG；只有 STS 标签才用相似度 Spearman。两者不能互相代替当「成功」判据。

### 进阶技巧（索引）

双向注意力（NV-Embed / LLM2Vec）、LoRA、RetroMAE、Latent Attention 池化、Query 扩展、Self-KD——都是在 “理论基础”到“训练与数据工程” 主路径上的加成，不替代「损失选对 + 负例干净」。

---

## 评测体系与 Benchmark

### MTEB（Massive Text Embedding Benchmark）

| 属性     | 详情                                            |
| -------- | ----------------------------------------------- |
| 任务范围 | 原始 MTEB 含 56 个数据集；语言覆盖、任务集合随版本与所选 benchmark 变化 |
| 任务类型 | 8 类（见下表）                                  |
| 指标     | 各任务特定（MAP, nDCG, Accuracy, v-measure 等） |
| 平台     | HuggingFace MTEB Leaderboard                    |
| 版本 | 记录代码版本、benchmark 名称、数据 revision 和评测配置；不能只写 v1/v2 |

**8 类任务各自在测什么**（平均分之前先看这个）：

| 任务类型                      | 测的能力                     | 和 RAG 检索的关系               | 典型翻车               |
| ----------------------------- | ---------------------------- | ------------------------------- | ---------------------- |
| **Retrieval**           | 非对称 q→d 排序（nDCG/MRR） | **最直接相关**            | 指令漏加、归一化不一致 |
| **Reranking** | 对已给候选排序 | 可使用 embedding 相似度或联合打分，取决于协议；不等于全库召回 | 把候选池成绩当全库检索成绩 |
| **STS**                 | 对称句对相似度（Spearman）   | 相关但不等价；STS 涨≠Recall 涨 | 用 STS 早停做检索模型  |
| **Classification** | 通常在冻结向量上训练分类器 | 测任务可分性；依具体 evaluator 确认分类器 | 把分类成绩直接当检索成绩 |
| **Clustering**          | 无监督簇纯度                 | 测空间几何                      | 检索强、聚类弱很常见   |
| **Pair Classification** | 句对关系（如重复）           | 偏对称匹配                      | —                     |
| **Summarization**       | 摘要与原文相关               | 窄                              | 权重低，别主导选型     |
| **Bitext Mining**       | 跨语对齐                     | 只在多语场景关键                | —                     |

公开榜的「平均分」会把上述能力揉在一起；业务若是 RAG，应主看 Retrieval（及自有集），其余当诊断。

### BEIR（Zero-Shot Retrieval）

- 18 个检索数据集，零样本评估
- 涵盖 Bio, Finance, News, QA 等 9 个领域
- 指标：nDCG@10, Recall@100
- BEIR 的一些数据集也被 MTEB 等框架采用；具体集合与切分依 benchmark 配置，不能把 BEIR 与 MTEB Retrieval 当作完全相同的集合

### 其他 Benchmark

| Benchmark           | 范围         | 特点             |
| ------------------- | ------------ | ---------------- |
| **MMTEB**     | 250+ 语言    | 多语言综合       |
| **C-MTEB**    | 35 中文任务  | 中文专用         |
| **MTEB-Code** | 代码检索     | CodeSearchNet 等 |
| **ViDoRe**    | 视觉文档检索 | ColPali 主战场   |
| **AIR-Bench** | 自动构造的检索评测 | 降低标注成本与污染风险；合成质量及是否见过测试材料仍需检查 |
| **CoIR**      | 代码 IR      | 代码搜索专用     |

### MTEB 局限性（必读）

1. **与实际 RAG 性能不完全相关**：MTEB 高 ≠ 你的领域 RAG 好
2. **任务平均掩盖短板**：检索强 ≠ 聚类强；汇报时拆开 Retrieval / STS / Clustering
3. **版本不可比**：v1 vs v2 分数体系不同，禁止跨版本排行榜叙事
4. **数据污染风险**：合成数据与爬虫对若扫过测试文档，分数虚高
5. **缺少长文档/多模态**：需 BEIR 子集之外的业务集、ViDoRe、MMEB 等补充

### 评测协议与污染防控

首先说明在检验哪一种泛化。未见查询、未见文档、未见领域、未见语言是不同协议；查询留出不自动构成跨域零样本。固定语料检索中，训练过程接触未标注语料并非一律违规，但必须公开这一条件，不能与完全未见语料设置混称。

| 协议组成 | 必须能回查的信息 |
| --- | --- |
| 数据边界 | 训练／验证／测试查询、qrels、语料快照；实体、来源文档、时间或图像实例的分组单位 |
| 训练可见性 | 哪些语料参与预训练、挖负例、伪 query 合成、教师打分；是否为固定语料设置 |
| 输入 | checkpoint 和 tokenizer revision、查询／文档模板、截断长度、切块与预处理 |
| 检索 | 相似度、归一化、ANN 参数、候选数、metadata 过滤；精排器及其候选来源 |
| 结果 | 每个任务和重要子群的指标、宏平均口径、成本；不只给一个总分 |

泄漏检查至少覆盖查询和答案的精确／近重复、同源文档的不同版本、同一图像的裁剪，以及用测试 query 或 qrels 合成训练数据的路径。只检查 `doc_id` 不够：同一内容可能换 URL、文件名和切块边界。模型未知预训练语料无法彻底排除污染时，应说明可检查的范围，而不是宣称“零污染”。

**检索库和训练样本要分别管理。** 将测试标签用于挖负例或调教师阈值会泄漏；按事先声明的协议索引测试语料是评测的必要步骤。若在该语料上生成训练 query，这属于额外的语料适配条件，需与不适配的对照分开报告。验证集用于模板和超参选择，测试集留到方案确定后评估。

指标也要对应相关性定义。检索 Recall@K 通常统计前 K 个候选覆盖的已标注相关文档比例，开放域问答常用的“Top-K 命中率”则统计前 K 段是否至少有一段含答案；两者不能同名混报。MRR 关注首个相关结果，nDCG 支持位置折损与分级相关性；mAP 适合有多个相关项的排序，但都受 qrels 完整性影响。未标注不自动等于不相关。

模型对比可对同一查询做成对 bootstrap，报告指标差值的区间；同实体或同源查询高度相关时按实体／来源重采样。小样本的几分提升不能自动视为稳定收益。人工对比应隐藏模型来源、混排去重后标注，再仲裁分歧，避免只看本模型的 Top-K。

最后分开做三种比较：全精度精确搜索衡量表示；相同表示下的 ANN 衡量索引近似误差；加入 Hybrid／精排／生成器后衡量系统收益。它们的候选范围、延迟与失败来源不同。需要检验复杂约束或长文本时，分别参考 [FollowIR](FollowIR/FollowIR详解.md)、[LongEmbed](LongEmbed/LongEmbed详解.md)、[ViDoRe v2](ViDoRev2/ViDoRe-v2详解.md)；BEIR、MTEB、MMTEB 的原始来源见[资料清单](资料清单_论文与博客.md#方法评测与工程补充阅读)。

---

## 应用场景全景

Embedding 不是「万能相似度」：不同任务对**对称性、粒度、索引形态、失败代价**的要求不同。先定任务，再定 “理论基础与表示范式” 的四轴与 “训练方法与数据工程” 的课表。

### 任务 × 表示 × 交互（比模型名单有用）

| 任务                        | 输入/输出      | 对称？          | 常用表示+交互               | 索引里存什么                | 典型失败                         |
| --------------------------- | -------------- | --------------- | --------------------------- | --------------------------- | -------------------------------- |
| **RAG 段落检索**      | query→chunk   | 否（q 短 d 长） | Dense Bi + Hybrid + CE 精排 | chunk 向量（+ 可选 sparse） | 切块切断语义；漏关键词；精排未开 |
| **FAQ / 问句匹配**    | q↔q'          | 是              | Dense Bi                    | 问题向量                    | 用检索非对称模板训 FAQ           |
| **语义搜索（站内）**  | query→doc     | 否              | Dense+BM25 Hybrid           | 全文 chunk                  | 只 Dense 漏 SKU/型号             |
| **去重 / 近重复**     | doc↔doc       | 是              | Dense Bi，阈值              | 全库或 LSH                  | 阈值过低误杀改版；过高漏洗稿     |
| **聚类 / 主题**       | doc→簇        | 是              | Dense                       | 向量                        | 用检索榜模型却不看 v-measure     |
| **推荐 Two-Tower**    | user, item→分 | 否              | Dense Bi（+ 特征塔）        | item 向量；user 在线算      | 冷启动；实时行为未进 user 塔     |
| **图搜图**            | image→image   | 是              | 视觉塔 Dense Bi             | 图像向量                    | 用 CLIP 文塔搜图；颜色/背景主导  |
| **文搜图**            | text→image    | 否              | 双塔对齐空间                | 图像向量                    | 训练图文对噪声大；指令未对齐     |
| **扫描件 / 版式文档** | text→page     | 否              | Multi-Vec Late              | 页级 patch 向量             | OCR→chunk→Dense 丢表格/图      |
| **代码搜索**          | nl/code↔code  | 可对称可非对称  | 代码专用 Bi                 | code snippet 向量           | 通用文本 Embedding 不懂 API 名   |
| **Agent 记忆**        | query→memory  | 否              | Dense Bi                    | 记忆条目向量                | 只存摘要丢细节；无 TTL/版本      |

### RAG：Embedding 管哪一段、不管哪一段

```

离线:  Raw docs → 清洗 → Chunking → [可选 metadata] → Embed(doc) → 向量库 (+ 可选 BM25 索引)
在线:  Query → [可选改写/扩展] → Embed(q) → ANN/Hybrid 召回 Top-K
       → [可选 Rerank] → Top-N chunks → Prompt 拼装 → LLM

```

**Embedding 只保证「候选 chunk 在语义上像相关」**，不保证：事实正确、最新、可引用、无冲突。下游幻觉、过期、多跳推理失败，不能单靠换 Embedding 模型解决。

#### Chunking 与向量的耦合

| 策略                         | 机制                       | 何时用           | 失败模式                                   |
| ---------------------------- | -------------------------- | ---------------- | ------------------------------------------ |
| 固定长度（如 512 tok）       | 实现简单                   | 通用 prose       | 法条/表格/步骤被拦腰切断 → 召回「半句话」 |
| 语义边界                     | 按段/标题切                | 结构化文档       | 切太碎丢上下文                             |
| Parent–Child                | 小块检索、大块进 prompt    | 长文 RAG         | 子块命中但父块未进上下文                   |
| Contextual chunk（dsRAG 等） | LLM 给每块加「文档级前缀」 | 强依赖块内自洽   | 前缀与训练分布不一致时 Embedding 漂移      |
| 页级视觉（ColPali）          | 不 OCR，整页多向量         | 扫描 PDF、幻灯片 | 纯文本 API 管道接不上                      |

**断环**：chunk 策略与 Embedding 训练粒度不一致（训练看整段、线上 128 tok 小块）时，Recall 上限会被 chunk 天花板卡住，换 8B 模型也无效。

#### RAG 常见失败与归因

| 现象                    | 先查什么                | 常见根因                         |
| ----------------------- | ----------------------- | -------------------------------- |
| 召回空 / 全不相关       | Top-K、阈值、库是否更新 | 索引 stale；query 未加 instruct  |
| 召回有主题但答非所问    | chunk 内容、精排        | 切块错；无 Rerank；Bi 单向量瓶颈 |
| 专有名词搜不到          | Hybrid 分数             | 纯 Dense；未建 BM25/Sparse       |
| 英文 query 中文 doc 差  | 多语模型与模板          | 单语 Embedding；跨语未训         |
| MTEB 高、业务 Recall 低 | 自有验证集              | 榜与领域分布脱节；数据污染       |
| 加了 Rerank 更差        | Rerank 训练域           | CE 与领域不匹配；候选太少        |

### FAQ、去重、推荐（短节）

**FAQ**：对称相似；两侧同一模板。误用检索的 `Instruct + Query` 非对称格式会把 FAQ 当「q 找长 doc」，Top-1 抖动。

**去重**：阈值在验证集上扫 Precision–Recall；近重复（改写、翻译）需要 harder 负例或更大模型，不是单纯调低阈值。

**推荐 Two-Tower**：与 Bi-Encoder 同构，但多了**特征工程、负采样（in-batch 曝光未点击）、实时 user 状态**。Embedding 文献里的 MNRL 课表要改成「点击为正、曝光未点击为负」，且假负例（用户其实喜欢但未点）更严重。

### 多模态检索在应用层的分叉

与 “多模态与专用 Embedding” 四类路线对齐（详述见《[图文 Embedding 模型技术综述](图文Embedding模型技术综述.md)》）：

| 场景                | 应走的路线                          | 别走弯路                              |
| ------------------- | ----------------------------------- | ------------------------------------- |
| 商品图搜图          | ① 双塔（CLIP/SigLIP 或电商视觉塔） | 用文本 RAG 模型 encode 图像路径字符串 |
| 以文搜图            | ① 双塔，共享对齐空间               | 只用 DINO 视觉塔（无文本塔）          |
| 图文混合 Agent 记忆 | ③ MLLM 通用嵌入                    | 强 OCR 再当纯文本                     |
| PDF 报表 / 扫描合同 | ④ ColPali 类 Late                  | OCR→chunk→Dense 丢表头/合并单元格   |

**图搜图 vs 文搜图**：图搜图只需视觉塔在同一度量空间；文搜图必须**文本 query 与图像 doc 共空间**（CLIP/SigLIP/MLLM-Emb）。DINO 类自监督视觉塔语义强，但不自带文本对齐，不能直接文搜图，需另训对齐头或换 CLIP 系。

---

## 检索 Pipeline 工程实践

Pipeline 设计 = 在**延迟预算**内最大化 **Recall@K（给 Rerank/LLM 的候选质量）**，不是堆模型个数。

### 三段式各自解决什么

| 阶段               | 算什么                                | 复杂度                    | 典型 K       | 失败时表现                                |
| ------------------ | ------------------------------------- | ------------------------- | ------------ | ----------------------------------------- |
| **Recall**   | Bi / Hybrid / Late 在**全库**   | $O(\log N)$ ANN 或倒排  | 50–200      | relevant 根本进不了候选 → 后面全白搭     |
| **Rerank**   | Cross / Late / 小 CE 在**候选** | $O(K \cdot L)$          | 5–20        | 召回够但 Top-N 仍含大量「主题对、答案错」 |
| **Generate** | LLM 读 context                        | $O(N \cdot \text{tok})$ | 3–10 chunks | context 太长/冲突/过时                    |

**原则**：Recall 阶段宁可多给（K 偏大），Rerank 收窄；省 Rerank 省的是延迟，丢的是 Precision。全库 Cross-Encoder 在百万级以上不可行（“为什么 Bi 通常打不过 Cross（机制，不是玄学）”）。

### Hybrid：为何有效、怎么融合

Dense 擅语义泛化；BM25/Sparse 擅**词面命中**（型号、法条号、错误码）。二者分数尺度不同，不能直接 $0.5 s_1 + 0.5 s_2$ 除非在验证集上标定。

#### RRF（Reciprocal Rank Fusion）

$$
\text{RRF}(d) = \sum_{r \in \{\text{dense},\text{sparse},\ldots\}} \frac{1}{k + \text{rank}_r(d)}
$$

- 只用**名次**，不用原始分 → 不怕 Dense 余弦与 BM25 量纲不同。
- $k$ 常取 60（经验）；$k$ 越大，头部名次差异被抹平。
- **适用**：多路召回源（Dense + BM25 + 多模型）快速合并。
- **局限**：不利用「强相关 doc 分数领先幅度」；两路都排第 50 的 doc 可能被抬上来。

#### 加权分融合（BGE-M3 等）

$$
s = w_1 s_{\text{dense}} + w_2 s_{\text{lex}} + w_3 s_{\text{mul}}
$$

权重必须在**自有验证集**上网格搜索或学习排序；论文示例权重不可直接抄。Lexical 与 Dense 分数需各自归一化或校准后再加。

**工程路径**（BGE-M3 常见）：Dense ∪ Sparse 并集召回 → RRF 或加权 → 可选 ColBERT 头 / Cross-Encoder 精排。细节见《[BGE-M3三功能统一详解报告](BGE/M3/BGE-M3三功能统一详解报告.md)》的混合检索部分。

#### 何时 Hybrid 仍不够

- 表格/图在 PDF 内，文本通道完全丢失 → 上 ColPali（“多模态与专用 Embedding”）。
- Query 极短且专有名词为主 → Sparse 权重要高，甚至 Sparse-only 兜底。
- 多语言 query 与单语 BM25 词表不匹配 → 靠多语 Dense，Sparse 仅作辅助。

### Reranker 放哪、选谁

| 类型                              | 与 Recall 关系             | 何时选                             |
| --------------------------------- | -------------------------- | ---------------------------------- |
| **Cross-Encoder**           | 全新算$s(q,d)$，看全交互 | 文本候选 ≤200，延迟允许百 ms 级   |
| **Late（ColBERT/ColPali）** | 用预存多向量 MaxSim        | 已存多向量索引；要 Bi 与 CE 中间档 |
| **LLM Rerank（4B 级）**     | 大模型打分                 | 质量优先、候选很少                 |

Reranker 训练域与业务域不一致时，会出现「CE 把正确 doc 打下去」。此时优先**领域微调 Reranker** 或**领域 Bi 召回**，而不是换更大的通用 CE。

### ANN 索引与召回质量

| 算法             | 机制             | 调什么                            | 典型坑                                 |
| ---------------- | ---------------- | --------------------------------- | -------------------------------------- |
| **HNSW**   | 图导航近似最近邻 | `efConstruction` / `efSearch` | `efSearch` 太小 → Recall 掉但延迟低 |
| **IVF**    | 先聚类再搜簇     | `nlist` / `nprobe`            | `nprobe` 太小漏簇                    |
| **PQ/OPQ** | 向量量化         | 码本数                            | 量化过猛伤精度；与 MRL 截维叠加要测    |
| **Flat**   | 暴力精确         | —                                | 仅小库或离线评测 gold                  |

**契约**：库内 metric（IP/L2）与 Embedding 是否 L2 归一化一致（“相似度度量与索引约定”）。换库不改归一化，会出现「离线评测 0.9、线上 0.6」的假象。

**增量更新**：doc 版本变更必须**同 id 覆盖或删后重插**；chunk 与 parent 版本不一致会导致 RAG 引用旧段落。

### 延迟与成本粗算（用于选型，非报价）

单次请求延迟 ≈ $T_{\text{embed}}(q) + T_{\text{ANN}}(K) + T_{\text{rerank}}(K') + T_{\text{LLM}}$。

- $T_{\text{embed}}(q)$：小 Encoder ~5–20ms GPU；7B LLM-Emb 更高，常 batch 摊薄。
- $T_{\text{ANN}}$：HNSW 毫秒级（百万级）；与 $K$ 弱相关。
- $T_{\text{rerank}}$：Cross 对 $K=100$ 常占 ** tens of ms–百 ms**，往往是瓶颈。

降本顺序：MRL 截维 → 小 Bi → 缩小 Rerank 输入 K → 量化 Embedding；**不要**先砍 Hybrid 再换 8B 模型。

---

## 文本 Embedding：机制族谱（非排行榜）

选型应问：**要什么交互、什么数据课表、能否私有化**——MTEB 总分只做粗筛。分数随版本变化，下文**不列具体榜分**；需对照时查 HuggingFace MTEB Leaderboard 与模型卡。

### 机制族 A：Encoder + 对比学习（SBERT 血脉）

- **骨干**：BERT/RoBERTa/XLM-R（110M–560M）。
- **训练**：NLI/STS 起步 → MS MARCO 检索 + in-batch / BM25 负例。
- **读出**：mean / CLS。
- **代表**：SBERT、BGE v1.5 系、E5-base/large、GTE-base。
- **适合**：私有化 RAG、CPU/边缘、要快、要微调。
- **上限**：512–8K 上下文（视版本）；复杂 instruct 跟随弱于 LLM 系。

### 机制族 B：弱监督课表 + 指令（E5 / BGE 主流）

- **Stage 1**：十亿级弱对（标题–正文、QA）撑空间。
- **Stage 2**：MARCO/NLI + **hard negatives** + Query 侧 instruction。
- **代表**：multilingual-e5-instruct、BGE-M3（再叠加 Sparse/Late 头）。
- **关键差异**：BGE-M3 在族 B 上增加**三功能头 + RetroMAE 打底**（见专题）；不是「又一个 BERT」。
- **适合**：多语 Hybrid 生产、要 Sparse 兜底。
- **翻车点**：漏 instruct；弱监督假正例未用 Stage2 纠偏。

### 机制族 C：Decoder LLM 作 Bi-Encoder

- **骨干**：Mistral/Qwen/Llama 1.5B–8B。
- **改造**：双向注意力 / LLM2Vec 式改造 + last-token 或 latent pooling。
- **数据**：更大弱监督 + 合成 query + 长上下文。
- **代表**：E5-mistral、NV-Embed、Qwen3-Embedding、GTE-Qwen2。
- **适合**：长文档、多任务 instruct、追求榜上限。
- **代价**：推理与显存；必须蒸馏/量化才适合边缘（“Embedding 蒸馏与压缩”）。

### 机制族 D：蒸馏得到的小模型（Students）

- **机制**：大 Teacher（族 C 或 CE）→ 小 Student（0.6B–2B）对齐**排序/分数/向量**（“Embedding 蒸馏与压缩”）。
- **代表**：Jasper、Stella、KaLM、Qwen3-Embedding-0.6B 路线。
- **适合**：要接近大模型检索质量、但部署预算有限。
- **注意**：蒸馏目标须与线上一致（同一 instruct、同一 Rerank 链路）；只蒸 STS 不蒸检索会偏。

### 机制族 E：API 闭源 Embedding

- **机制黑盒**，强在运维与长上下文 SLA。
- **代表**：OpenAI v3、Cohere v3/v4、Voyage、Gemini Embedding。
- **适合**：快速验证、无 GPU、数据可出境。
- **不适合**：强领域微调、离线合规、要 Hybrid 三头可控。

### 族谱 → 场景（对应 “实践选型指南”，此处不堆品牌）

| 你的约束             | 优先机制族                                   |
| -------------------- | -------------------------------------------- |
| 边缘 / CPU / 低成本  | A 或 D（蒸馏小模型）                         |
| 多语 Hybrid 生产     | B（+ Reranker）                              |
| 长上下文 + 高质量    | C，或 D 蒸出来                               |
| 无 GPU、先验证       | E                                            |
| ≤0.6B 自训（图+文） | D + 多模态 “多模态与专用 Embedding” ③④；Teacher 用大 CLIP/MLLM |

**附录式对照**：若需参数/许可/维度，查 “完整时间线” model card 与 HuggingFace；**勿用本章做选型唯一依据**。

---

## 多模态与专用 Embedding

术语与《[图文 Embedding 模型技术综述](图文Embedding模型技术综述.md)》一致：**① 双塔 ② Fusion ③ MLLM 通用嵌入 ④ 多向量后交互**。四类可组合（如 ③ 骨干 + ④ 文档头）。

### 四类路线对照（能力边界）

| 维度               | ① 双塔 CLIP/SigLIP | ② Fusion  | ③ MLLM-Emb         | ④ ColPali 类 Late     |
| ------------------ | ------------------- | ---------- | ------------------- | ---------------------- |
| **交互**     | Bi，图文各算一次    | 联合注意力 | Bi（last-token 等） | Late MaxSim            |
| **预计算**   | 图/文均可           | 仅候选打分 | 图/文均可           | 页向量可预计算         |
| **图搜图**   | ✅ 视觉塔           | ❌         | ✅（视觉编码）      | ✅（页/image patches） |
| **文搜图**   | ✅ 对齐空间         | ✅ 精排    | ✅ 指令化           | ✅ text query→page    |
| **扫描 PDF** | 弱（当普通图）      | 中         | 中                  | **强**           |
| **延迟**     | 低                  | 高         | 中–高              | 中（索引大）           |

### 图搜图 vs 文搜图（训练与索引）

**图搜图**：query 与 gallery 都是图像 → 同一视觉编码器 $f_v(\cdot)$，$s=\cos(f_v(q), f_v(d))$。可用 DINO/DINOv2 特征 + 小投影，或 CLIP 视觉塔。失败常见原因：域偏移（商品图 vs 用户手机拍）、颜色纹理主导、未做 hard negative（同款不同色）。

**文搜图**：query 是文本 → 需要 $f_t(\text{text})$ 与 $f_v(\text{image})$ **共空间**。必须走 CLIP/SigLIP 对比训练或 MLLM 图文对齐；不能直接把文本 Embedding 模型与 DINO 视觉特征拼在一起搜。

**≤0.6B 自训配方**（数据、完整损失、蒸馏与评测见[图搜图与文搜图表示学习](图搜图与文搜图表示学习.md)）：

1. 大 Teacher：CLIP/SigLIP 或 2B+ MLLM-Emb 产生伪标签或分数；
2. 小 Student：0.3B–0.6B 双塔或共享视觉塔；
3. 损失：图文 InfoNCE/Sigmoid +（可选）图搜图同塔对比；
4. 数据：商品图+标题弱对 + 经任务标签确认的 hard 负（是否允许同款不同色，按业务定义）；
5. 评测：**分开报** Image→Image 与 Text→Image，不要只报一个「多模态平均」。

### 视觉文档（④ 类）

ColPali / ColQwen：**页面渲染图 → patch 多向量**，query 侧 text token 多向量，MaxSim 聚合。绕过 OCR 流水线，对表格、图表、多栏排版更稳。代价是索引体积（每页数百向量）与专用检索器。

**错误路径**：扫描 PDF → OCR 错字 → chunk → 文本 Embedding → 问「图 3 销售额」类问题。

### 代码与领域专用

独立深读与文搜图对照见《[领域专用 Embedding 适配实践](领域专用Embedding适配实践.md)》。课表仍是 “训练方法与数据工程”：**领域适配放 Stage 2/4，不要用小领域数据从头训**。

| 类型 | 机制要点 | 与通用文本差异 | 深读 |
| --- | --- | --- | --- |
| **代码** | 标识符、AST、合成多任务三元组 | 通用文本对 API 名 OOV；CSN 会虚高 | [代码IR合写](Code-IR/代码检索Embedding合写.md) |
| **法律** | 领域 MLM + 文书结构；条号 Sparse | 「事实像」≠ 判例相关 | [法律合写](Legal/法律检索Embedding合写.md) |
| **医疗** | PubMed 点击对比 + PubMedBERT | 通用大双塔仍会域偏移 | [MedCPT](MedCPT/MedCPT详解.md) |
| **科学文献** | 引用图三元组；SPECTER2 按任务 adapter | 一篇论文一个向量不够打所有格式 | [SPECTER系列](SPECTER/SPECTER系列详解.md) |
| **推荐召回** | Two-Tower + 全库采样 softmax | 与 Bi-Encoder 同构，负例/曝光不同 | [YouTube DNN](YouTube-DNN/YouTube-DNN双塔详解.md) |

专用域**优先**领域数据微调 + 难负例，而不是换更大的通用 8B。文搜图把上表的「边 / 点击 / 结构 / Hybrid」换成标题–主图、同款不同色、SKU 倒排即可。

---

## Embedding 蒸馏与压缩

主文只讲清**版图与何时用**；损失组合、多教师课表、多模态 Teacher 细节见《[Embedding蒸馏技术详解](Embedding蒸馏技术详解.md)》。与 [`../distillation/知识蒸馏技术深度调研报告.md`](../distillation/知识蒸馏技术深度调研报告.md) 的边界：**那边**以 LLM/VLM **生成式** KD 为主；**本章**只谈**表示与排序**。

### 三种蒸馏信号（不要混用）

| 信号                  | 监督什么                      | 典型损失           | 适用                              |
| --------------------- | ----------------------------- | ------------------ | --------------------------------- |
| **排序 / 分数** | Teacher 对$(q,d)$ 相关度    | KL / MSE on scores | 检索主任务；Teacher 是 CE 或强 Bi |
| **向量**        | $v_T(x)$ 与 $v_S(x)$ 对齐 | MSE /$1-\cos$    | 维数相同或线性投影后              |
| **关系**        | 样本间距离结构                | triplet / pairwise | 数据少、要保流形                  |

**与 LLM KD 的区别**：LLM KD 蒸 **token 分布**（logits）；Embedding KD 蒸 **几何与排序**。把 MiniLLM/OPD 当 Embedding 主损失通常不对题。

### 典型场景

| 场景                      | 做法                                        | 注意                                           |
| ------------------------- | ------------------------------------------- | ---------------------------------------------- |
| **8B → 0.6B 部署** | 大 Bi Teacher + 检索损失 + 可选 CE 分数蒸馏 | 任务语义与候选一致；两侧分别遵循自己的模型模板 |
| **CE → Bi**        | CE 打候选分 → KL 到 Bi                     | 候选集要含 hard neg                            |
| **多 Teacher**      | 拼接分数或加权（Jasper/Stella 思路）        | Teachers 冲突时先统一口径再训                  |
| **Self-KD**         | EMA/ checkpoint 自身                        | BGE-M3 Stage3；无外部 Teacher                  |
| **无 Teacher 压缩** | **MRL 截维**、INT8/INT4 量化          | 不是蒸馏，但常一起做                           |

### MRL 与量化（部署压缩，非知识迁移）

- **MRL**：训练时惩罚前缀维，推理截断 $d$ → 索引变小、ANN 更快；精度损失在自有集上扫 $d\in\{256,512,\ldots\}$。
- **量化**：编码器权重量化与输出向量量化作用于不同对象。误差取决于模型、校准和任务，不能假设 embedding 总比生成模型耐受；需对同一 checkpoint 做精确搜索和实际索引对照。

Qwen3-Emb-8B INT4 等案例说明「大模型可压到消费级 GPU」，但 **0.6B 自训目标**更常是「Student + 蒸馏」而不是量化 8B。

### 蒸馏实验最小协议

1. 固定验证集与候选池：Recall@K / nDCG；教师、学生使用各自正确模板表达同一任务。
2. 报告：Student 参数量、维度、是否 MRL/量化。
3. 对照：Student 无蒸馏、Student 只蒸 STS（说明检索会偏）。
4. 失败信号：STS 升、Recall 不升 → 蒸馏目标错或假负例未控。

---

## 向量数据库与部署

### 库 vs 库：按能力选，不按营销选

| 需求                  | 优先                                       |
| --------------------- | ------------------------------------------ |
| 纯研究 / 单进程       | FAISS（库，非服务）                        |
| 要 Hybrid BM25 + 向量 | Elasticsearch、Vespa、Milvus 2.x、Weaviate |
| 已有 PostgreSQL       | pgvector + 全文（Hybrid 需自拼）           |
| 原型 / 小团队         | Qdrant、Chroma                             |
| 十亿级、磁盘索引      | Milvus DiskANN、Vespa                      |
| 强 metadata 过滤      | 带 filter 的 ANN（多数商业库支持）         |

**Hybrid 在库里的含义**：是「同一引擎内 BM25+向量」，还是「应用层 RRF 两路索引」——二者运维复杂度差很多。

### 部署检查清单（与 “相似度度量与索引约定”、“ANN 索引与召回质量” 闭环）

1. **Metric**：IP + L2 归一化 是否一致。
2. **维度**：MRL 截断后重新建库还是多集合。
3. **Metadata**：版本号、source、acl——过滤与 Embedding 正交但 RAG 必备。
4. **Batch 编码**：离线建库 batch 够大；在线 query 单条延迟单独测。
5. **服务**：TEI/vLLM/FlagEmbedding 用于 BGE/Qwen；Reranker 常单独进程防阻塞 ANN。
6. **监控**：Recall@K 抽样、索引 lag、embedding 模型版本号。

### MRL、向量量化与 ANN 误差

MRL 训练多个嵌套前缀具有任务能力，而非推理时任意裁去尾部维度。对于余弦空间，取前 $m$ 维后应重新归一化：

$$
z^{(m)}=\frac{v_{1:m}}{\|v_{1:m}\|_2}.
$$

查询和库端采用同一维度及处理约定。若 128 维取候选再用 1024 维重排，128 维阶段的召回限定最终上限；需要联合改变前缀维度、候选数和 ANN 搜索预算，而不只比较全维重排得分。

| 压缩对象 | 主要收益 | 保留的成本与潜在误差 |
| --- | --- | --- |
| 学生模型／权重 int8、int4 | 编码器的权重内存与部分计算带宽 | 激活、算子支持和吞吐单独测；参数个数未变 |
| MRL 前缀维度 | 每条向量更短，点积与索引成本可能降低 | 编码骨干通常仍运行；未经训练的截断无质量保证 |
| 向量 int8／binary | 每维位数减少，便于压缩存储或粗召回 | 标定、距离近似和精排原向量存储仍需计入 |
| PQ／残差压缩 | 用码本近似向量，减少存储和距离计算 | 码本训练、重构误差、候选召回和解压开销 |
| 多向量 token 合并／剪枝 | 每篇文档保留的向量更少 | 可能丢失稀有但重要的局部证据，需验证 MaxSim 相关性 |

裸向量大小为 $Ndb$ 字节，其中 $N$ 是条数、$d$ 是维度、$b$ 是每维字节数。百万条 1024 维 float32 约 4.096GB，128 维约 0.512GB；若为了全维重排另存 1024 维向量，这部分并没有消失。实际索引还包含图、码本、ID、metadata 与副本。

精确搜索可先固定同一相似度与浮点精度，得到表示的相关性基线。再分别加入截维、量化与 ANN，对比它们各自及组合后的任务 Recall／nDCG、近邻召回、内存和延迟。ANN 的近邻召回衡量是否找回该表示下的近邻，不能替代业务相关性指标。

二值召回后可以在候选上用 float 或 int8 重新评分，但代价是额外存储和访问；标量量化的范围应由训练／校准样本确定，不能在查看测试分数后反复调范围。QAT 是训练中模拟量化，PTQ 是训练后量化；GOR 是几何正则，不能当作 QAT 的另一个名字。详见 [MRL 解读](MRL/MRL解读.md)、[QAT 专题](QAT/量化感知训练QAT详解.md) 与 [Jina v5](Jina/v5-text/Jina-embeddings-v5-text详解.md)。

换模型、投影头、文档模板或预处理时，应重新核对与旧索引的兼容性；未明确训练并验证跨版本对齐时，重新编码并切换配套索引。仅在查询服务更换 checkpoint、继续使用旧库向量，是常见的空间不匹配来源。

### 参考拓扑

```

Client → Gateway → [Embed query] → Vector DB (ANN ± BM25)
                 → [Rerank Top-K] → LLM
        离线: Doc pipeline → Embed batch → Upsert

```

---

## 前沿方向与开放问题

每条附「可做什么实验」，避免空泛 trend 列表。

| 方向                         | 未解什么                | 可验证实验                                      |
| ---------------------------- | ----------------------- | ----------------------------------------------- |
| **MTEB ↔ RAG**        | 榜与业务 Recall 脱节    | 同一模型：MTEB Retrieval vs 自有 RAG 集相关系数 |
| **Unified Gen+Emb**    | 生成与嵌入是否互相伤    | GritLM/LLM2Vec：调 Gen/Emb loss 权重看 Recall   |
| **Visual Doc RAG**     | 何时胜过 OCR 管道       | 同 corpus：ColPali vs OCR+chunk+BGE 端到端      |
| **Long Context Emb**   | 32K 一次 embed vs chunk | 长文单向量 vs hierarchical chunk Recall         |
| **On-policy hard neg** | 动态负例是否稳          | 对比静态 mined neg：训练曲线与 STS 漂移         |
| **Agent Memory**       | 存 raw vs 摘要 vs 向量  | 多轮任务成功率 vs 记忆条数                      |
| **Graph+Vector**       | 何时 GraphRAG 值回票价  | 多跳 QA：纯向量 vs GraphRAG 成本/准确           |

**开放问题**（仍无共识）：嵌入反演与隐私、低资源语言公平性、多跳推理单向量瓶颈、增量索引与版本一致、端到端 RAG 评测标准缺失。

---

## 实践选型指南

### 决策：任务 → 表示 → 交互 → 训练（不是先选品牌）

```

1) 对称还是非对称？
   对称（FAQ/去重/STS）→ 两侧同模板；损失可用 STS 或 MNRL
   非对称（RAG/搜索）→ 按 checkpoint 约定设置两侧模板；对比目标 + 过滤难例
2) 索引里存什么？
   单向量 Dense → 默认
   要词面命中 → + Sparse/BM25 Hybrid
   扫描件/版式 → Multi-Vec Late（ColPali）或 MLLM-Emb
3) 算分方式？
   全库 → Bi 或 Late；Cross 只打 Top-K
4) 资源？
   无 GPU → 族 A 小模型或 API
   要 ≤0.6B 自训 → 族 D + “多模态与专用 Embedding” 多模态 Teacher
5) 验证？
   自有 Recall/nDCG → 再参考 MTEB Retrieval 子集

```

### 微调最小闭环（与 “训练方法与数据工程” 一致）

1. 数据：$(q, d^+)$ + 过滤后 hard neg；隔离测试查询与标签，并声明语料是否参与适配。
2. 损失：**检索用 MNRL/InfoNCE**，不是 CosineSimilarityLoss（除非纯 STS）。
3. 模板：与线上一致。
4. 验证：**InformationRetrievalEvaluator** 类指标，不是只跑 STS Spearman。
5. 上线：Hybrid + 可选 Rerank；记录 embedding 与索引版本。

### 常见陷阱（机制版）

| 陷阱              | 机制原因                         |
| ----------------- | -------------------------------- |
| STS 高、Recall 低 | 优化对称相似，非 q→d 排序       |
| 换 8B 无提升      | chunk/负例/instruct 瓶颈在前     |
| Hybrid 无效       | 权重未标定；Sparse 索引未建      |
| Rerank 反效果     | CE 域不匹配；K 太小              |
| 文搜图失败        | 未对齐图文空间；只用 DINO 视觉塔 |
| MTEB 选型         | 平均掩盖 Retrieval 短板          |

---

## 原始来源

论文、官方仓库、模型卡和实现博客统一在[资料清单](资料清单_论文与博客.md)维护。本报告不再复制另一份容易过期的书目；具体数字与方法条件优先回查相应原论文及[独立研读](README.md#论文精炼版与补充解读)。

---

## 术语表

| 术语                       | 解释                                                           |
| -------------------------- | -------------------------------------------------------------- |
| Bi-Encoder                 | Query/Doc 独立编码，点积相似度                                 |
| Cross-Encoder              | Query+Doc 联合编码，精度高但慢                                 |
| Late Interaction           | 多向量 + MaxSim，ColBERT 范式（BGE-M3 mul 头同族，见专题报告） |
| Sparse / Lexical Embedding | 词表维稀疏权重；可学习版 ≈ 神经 BM25（BGE-M3 / SPLADE）       |
| InfoNCE                    | 对比学习损失，Info Noise-Contrastive Estimation                |
| MRL                        | Matryoshka Representation Learning，多维度嵌套                 |
| ANN                        | Approximate Nearest Neighbor，近似最近邻                       |
| HNSW                       | Hierarchical Navigable Small World，图索引                     |
| RRF                        | Reciprocal Rank Fusion，倒数排名融合                           |
| Hard Negative              | 与 anchor 相似但标签不同的困难负样本                           |
| MTEB                       | Massive Text Embedding Benchmark                               |
| BEIR                       | Benchmarking IR zero-shot                                      |
| RAG                        | Retrieval-Augmented Generation                                 |

## 按问题回查原始工作

| 要理解的问题 | 已有学习入口 | 尚无独立研读的衔接材料 |
| --- | --- | --- |
| 句向量为什么不能直接用于所有检索任务 | [Sentence-BERT](Sentence-BERT/Sentence-BERT详解.md) → [DPR](DPR/DPR详解.md)，对照 [SimCSE／Contriever](无监督对比检索三部曲_SimCSE-Contriever-Condenser.md) | DSSM、InferSent 补监督与双塔前史 |
| 找到证据后怎样参与生成 | 本文 RAG 流水线和 DPR 阅读器 | RAG 的文档概率边缘化、FiD 的多段编码与解码融合；DrQA／ORQA／REALM 补任务和训练范式 |
| 词法与细粒度证据怎样保留 | [SPLADE](SPLADE/SPLADE/SPLADE详解.md)、[ColBERT](ColBERT/ColBERT详解.md) → [v2](ColBERTv2/ColBERTv2详解.md) → [PLAID](PLAID/PLAID详解.md) | XTR／WARP／MUVERA 比较进一步的召回与执行近似 |
| 有效上下文与有效 batch 是什么 | [LongEmbed](LongEmbed/LongEmbed详解.md)、[Late Chunking 合写](前沿短深读_LateChunking-Vec2Vec-ModernBERT-DINOv2v3-Qwen3-Seed-ViDoRev2.md)、[GradCache](GradCache/GradCache详解.md) | CDE、ConTEB／InSeNT 补语料与文档上下文；TAS-B 补 batch 采样 |
| 查询时增强和离线造训练数据有何不同 | [HyDE](HyDE/HyDE详解.md)、[InPars](InPars/InPars详解.md)、[Promptagator](Promptagator/Promptagator详解.md)、[E5-Mistral](E5-Mistral/zh.md) | 比较生成成本、查询漂移与域迁移，不把两种计算位置混为一谈 |
| 高分能否迁移到真实任务 | 本文评测协议、[FollowIR](FollowIR/FollowIR详解.md)、[ViDoRe v2](ViDoRev2/ViDoRe-v2详解.md) | BEIR／MTEB／MMTEB 原论文仍需独立研读；BRIGHT／ReasonIR 补推理相关性 |
| 小图文模型怎样利用大教师 | [图搜图与文搜图表示学习](图搜图与文搜图表示学习.md)、[MRL](MRL/MRL详解.md) | DataComp、MobileCLIP 系列补数据筛选和离线强化 |

这张表按机制连接现有内容，不把一条箭头当作已证实的模型继承关系，也不把“已引用”当作完整研读。未收录论文的原始出处及覆盖形态集中在[资料清单](资料清单_论文与博客.md#奠基性论文与经典阅读顺序)，现有精炼版、补充解读和全文对照以[阅读索引](README.md)为准。
