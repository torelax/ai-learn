> 原文: [arXiv:2506.05176](https://arxiv.org/abs/2506.05176)（Technical Report, 2025-06）
> 说明: 本文为 Qwen3-Embedding 技术报告的全文中文技术展开，公式/图表编号与原文一致；图片自 PDF 抽取，caption 中译；数值原样保留。

**预印本信息：** arXiv:2506.05176v3 [cs.CL]，首发 2025 年 6 月，最新版 2025-06-11。

**开源：**
- 模型：https://huggingface.co/Qwen ／ https://modelscope.cn/organization/qwen
- 代码：https://github.com/QwenLM/Qwen3-Embedding
- 协议：Apache 2.0

---

# Qwen3 Embedding：基于基础模型推进文本嵌入与重排（Advancing Text Embedding and Reranking Through Foundation Models）

**作者：** Yanzhao Zhang\*、Mingxin Li\*、Dingkun Long\*、Xin Zhang\*、Huan Lin、Baosong Yang、Pengjun Xie、An Yang、Dayiheng Liu、Junyang Lin、Fei Huang、Jingren Zhou

**单位：** Alibaba Group，Tongyi Lab（阿里巴巴通义实验室）

\* 共同一作。

**发布信息：**

| 项 | 值 |
| :--- | :--- |
| 类型 | Technical Report |
| 日期 | 2025 年 6 月 |
| 骨干 | Qwen3 Foundation Models（dense 版本） |
| 规模 | Embedding：0.6B / 4B / 8B；Reranker：0.6B / 4B / 8B |
| 上下文 | 32K tokens |
| 语言 | 100+ 语言（多语通用嵌入） |
| 协议 | Apache 2.0 |

---

## 摘要（Abstract）

作者提出 **Qwen3 Embedding 系列**——建立在 **Qwen3 基础模型** 之上，是前作 **GTE-Qwen 系列** 的显著升级。核心思路：**充分利用 Qwen3 LLM 在多语文本理解与生成上的能力**，把 LLM 既当作嵌入 / 重排模型的**骨干（backbone）**，又当作**训练数据合成器**。

**训练配方**结合三件事：

1. **大规模无监督预训练**（用 Qwen3-32B 合成的弱监督对比数据）；
2. **高质量数据集上的有监督微调**（公开数据集 + 筛选后的合成数据）；
3. **模型合并（model merging）** 策略——基于球面线性插值（SLERP）——进一步保证鲁棒性和泛化。

**产出**：两条产品线（嵌入 + 重排）× 三档参数（0.6B / 4B / 8B），共 6 个模型，全部开源。

**结果**：在 **MTEB 多语言**、**MTEB 英文 v2**、**CMTEB 中文**、**MTEB Code**、**MLDR**、**FollowIR** 等一整套 benchmark 上均取得 SOTA。旗舰 **Qwen3-Embedding-8B** 在 MTEB Multilingual 上达到 **70.58 分**、MTEB Code 上达到 **80.68 分**，**超过 Gemini-Embedding**；Qwen3-Reranker-0.6B 已经胜过多个既有 top 重排模型，Qwen3-Reranker-8B 相较 0.6B 再提升约 3.0 点。

---

## 引言（Introduction）

**背景**：文本嵌入与重排是搜索、问答、推荐等自然语言处理与信息检索应用的基础组件；随着 RAG（Retrieval-Augmented Generation）与 Agent 系统兴起，对嵌入 / 重排提出了新的要求——多语、多领域、指令跟随、代码检索、长文档检索等。

**LLM 的推动作用**：早期用 BERT 之类的 encoder-only 模型（Reimers & Gurevych, 2019）；后来 LLM 提供的更丰富的世界知识、更强的文本理解和推理能力，让基于 LLM 训练的嵌入 / 重排模型显著跃升。同时，LLM 还可以**参与训练数据合成与质量过滤**（Wang et al., 2024；Lee et al., 2024, 2025b），催生新的训练范式：例如按任务类型 / 领域 / 语言引入差异化指令训练（Su et al., 2023），或在重排上结合 zero-shot 提示与有监督微调（Ma et al., 2023；Pradeep et al., 2023；Zhuang et al., 2024）。

**本文贡献**：

1. **基座切换到 Qwen3**：同时利用 Qwen3 的 base 与 instruct 版本——base 做骨干，instruct 合成数据。
2. **嵌入模型采用多阶段训练管线**：
   - 大规模弱监督对比预训练（合成数据）；
   - 高质量数据 SFT；
   - **模型合并** 稳定分布外性能。
3. **重排模型采用两阶段训练**：高质量 SFT + 模型合并（**没有第一阶段弱监督**）。
4. **实用特性**：Embedding 支持 **MRL（Matryoshka Representation Learning）灵活维度**、**可定制指令**；Reranker 也支持任务化 instruction。
5. **规模全覆盖**：0.6B / 4B / 8B 各配一个 Embedding 与一个 Reranker，兼顾效率与效果。

**评测头条数据**：Qwen3-8B-Embedding 在 MTEB Multilingual 70.58、MTEB Code 80.68，超过此前专有 SOTA（Gemini-Embedding）；Qwen3-Reranker-8B 相比 0.6B 在多任务上再提升 3.0 点。

---

## 2 模型架构（Model Architecture）

**任务形式**：给定 query $q$ 与 document $d$，嵌入 / 重排模型基于一条指令 $I$ 所定义的相似度准则来评估相关性。训练数据组织成 $\{I_i, q_i, d^+_i, d^-_{i,1}, \dots, d^-_{i,n}\}$：$d^+_i$ 是与 $q_i$ 相关的正文档，$d^-_{i,j}$ 是若干负文档。用这种带指令的三元组训练，可以让同一个模型覆盖 retrieval、STS、classification、clustering 等下游任务。

**通用配置**：Qwen3 嵌入 / 重排都基于 **Qwen3 基础模型的 dense 版本**初始化，出 0.6B / 4B / 8B 三档；层数、hidden size、上下文长度详见表 1。

![图 1（原文 Figure 1）：Qwen3-Embedding（左）与 Qwen3-Reranker（右）的模型架构](figs/fig01.png)

**图 1（原文 Figure 1）：** **左侧 Qwen3-Embedding**——把 instruction 与 query 拼成上下文（**instruction 只加在 query 端，document 保持原样**），用 causal attention 通过 Qwen3 LLM，在输入末尾追加一个 `[EOS]` token；**最后一层对应 `[EOS]` 位置的 hidden state 即为句向量**。**右侧 Qwen3-Reranker**——把 instruction、query、document 一并塞进 LLM chat 模板，模型以**逐点（point-wise）** 的方式判断相关性——形式化为「yes / no」二分类，用「yes」token 与「no」token 的下一个 token 概率归一化得到相关性分数。

### 2.1 嵌入模型（Embedding Models）

- **注意力**：保持 **causal attention**——不像 NV-Embed / LLM2Vec 强行移除 causal mask。作者认为 Qwen3 本身的表达力足够，配合 `[EOS]` 池化即可获得强表征。
- **池化**：末位 `[EOS]` token 的最后一层 hidden state。
- **输入模板**（query 端）：

  ```
  {Instruction} {Query}<|endoftext|>
  ```

- **文档端**不加 instruction 前缀——保证文档 index 与具体下游任务解耦、可离线预建。

### 2.2 重排模型（Reranking Models）

- **形式**：**逐点二分类**——在同一段上下文中，让 LLM 判断该文档是否满足 query + instruction 的要求。
- **模板**（LLM chat）：

  ```
  <|im_start|>system
  Judge whether the Document meets the requirements based on the Query
  and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
  <|im_start|>user
  <Instruct>: {Instruction}
  <Query>: {Query}
  <Document>: {Document}<|im_end|>
  <|im_start|>assistant
  <think>\n\n</think>\n\n
  ```

- **打分公式**——用「yes」与「no」两个 token 在 assistant 首个位置的对数概率做 softmax：

$$
\text{score}(q, d) = \frac{e^{P(\text{yes}\mid I,q,d)}}{e^{P(\text{yes}\mid I,q,d)} + e^{P(\text{no}\mid I,q,d)}}
$$

### 2.3 六个模型的规格（表 1）

| 模型类型 | 模型 | Size | 层数 | Sequence Length | Embedding Dim | MRL Support | Instruction Aware |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Text Embedding | Qwen3-Embedding-0.6B | 0.6B | 28 | 32K | 1024 | Yes | Yes |
|  | Qwen3-Embedding-4B | 4B | 36 | 32K | 2560 | Yes | Yes |
|  | Qwen3-Embedding-8B | 8B | 36 | 32K | 4096 | Yes | Yes |
| Text Reranking | Qwen3-Reranker-0.6B | 0.6B | 28 | 32K | — | — | Yes |
|  | Qwen3-Reranker-4B | 4B | 36 | 32K | — | — | Yes |
|  | Qwen3-Reranker-8B | 8B | 36 | 32K | — | — | Yes |

**表 1：** Qwen3 Embedding / Reranker 模型架构。**MRL Support** 表示嵌入模型支持自定义输出维度（Matryoshka Representation Learning）；**Instruction Aware** 表示可根据不同任务自定义输入 instruction。

---

## 3 模型训练（Models Training）

### 3.1 训练目标（Training Objective）

#### 嵌入模型：改进版 InfoNCE

给定 batch 中的 $N$ 个训练样本，嵌入模型的对比损失定义为：

$$
\mathcal{L}_{\text{embedding}} = -\frac{1}{N} \sum_{i}^{N} \log \frac{e^{s(q_i, d^+_i)/\tau}}{Z_i} \tag{1}
$$

其中 $s(\cdot,\cdot)$ 是相似度（作者用余弦相似度），$\tau$ 是温度系数；归一化项 $Z_i$ 汇总了正对与多种负对：

$$
Z_i = e^{s(q_i, d^+_i)/\tau}
  + \sum_{k}^{K} m_{ik}\, e^{s(q_i, d^-_{i,k})/\tau}
  + \sum_{j\neq i} m_{ij}\, e^{s(q_i, q_j)/\tau}
  + \sum_{j\neq i} m_{ij}\, e^{s(d^+_i, d_j)/\tau}
  + \sum_{j\neq i} m_{ij}\, e^{s(q_i, d_j)/\tau}
$$

分母中依次是：（1）正文档 $d^+_i$；（2）$K$ 个 hard negative $d^-_{i,k}$；（3）batch 内其它 query $q_j$；（4）batch 内其它文档 $d_j$ 与正文档 $d^+_i$ 的相似度；（5）batch 内其它文档 $d_j$ 与当前 query $q_i$ 的相似度。

**关键机制：mask 因子 $m_{ij}$——用于缓解 false negative**：

$$
m_{ij} = \begin{cases}
0 & \text{if } s_{ij} > s(q_i, d^+_i) + 0.1 \text{ or } d_j == d^+_i, \\
1 & \text{otherwise},
\end{cases}
$$

其中 $s_{ij}$ 是 $(q_i, d_j)$ 或 $(q_i, q_j)$ 的相似度。**含义**：如果 in-batch 中的某个 $d_j$ 或 $q_j$ 与当前 query 的相似度**超过正样本相似度 + 0.1**，或该 $d_j$ 恰好就是 $d^+_i$，就把它当成潜在的 false negative **从负样本集合里排除**。这个 mask 的做法与「positive-aware hard-negative mining」精神一致——用相对得分而非绝对 top-k 来判断谁可以当负例。

#### 重排模型：SFT 交叉熵

$$
\mathcal{L}_{\text{reranking}} = -\log p(l \mid P(q, d)) \tag{2}
$$

其中 $p(\cdot \mid *)$ 是 LLM 输出的 token 概率，$l$ 是标签——正样本为「yes」、负样本为「no」。该损失鼓励模型对正确标签打更高概率，从而提升排序性能。

### 3.2 多阶段训练（Multi-stage Training）

多阶段训练在嵌入模型上已有先例（Li et al., 2023；Wang et al., 2022；Chen et al., 2024）：先在包含噪声的大规模弱监督数据上预训，再用小规模高质量数据 SFT。**Qwen3-Embedding 在这个范式之上引入三项关键创新：**

- **合成数据驱动的弱监督预训练**：不同于 GTE / E5 / BGE 从 Q&A 论坛、学术论文等开放来源收集弱监督对，Qwen3 直接**用基础模型合成对**——通过合成 prompt 精确控制任务、语言、长度、难度等维度；对低资源语言与场景尤其有效。
- **有监督微调中选用高质量合成子集**：由于 Qwen3-32B 合成质量已经很高，作者会**再筛一遍**（余弦相似度阈值 0.7）挑出 ~12M 高质量对，加进 SFT 阶段一起训。
- **模型合并（Model Merging）**：SFT 结束后，作者对训练过程中保留的多个 checkpoint 做**基于 SLERP（Spherical Linear Interpolation）** 的合并，参考 Li et al. (2024)。目的是提升在不同数据分布下的鲁棒性和泛化性，避免过拟合到某一批次的分布。

**重要说明**：**重排模型不做第一阶段的弱监督预训练**——只有「高质量 SFT + 模型合并」两阶段。作者的判断是重排的信号（yes/no 二元判定）已经足够密集，弱监督对二元判定的贡献不如对嵌入模型那么关键。

![图 2（原文 Figure 2）：Qwen3 Embedding 与 Reranker 的训练流水线](figs/fig02.png)

**图 2（原文 Figure 2）：** **上方 Qwen3-Embedding 的三阶段管线**——Stage 1「Weakly Supervised Training」用 ~150M 合成对，Stage 2「Supervised Training」用 ~19M 高质量对（~7M 公开有标注 + ~12M 筛选后的合成），Stage 3「Model Merging」把多个 SFT checkpoint 用 SLERP 合成一份最终权重；**下方 Qwen3-Reranker 只有两阶段**——SFT + Model Merging，跳过第一阶段的合成弱监督。整条管线里 Qwen3 LLM 既是**骨干**也是**数据合成器**（图中虚线指向合成数据的作用路径）。

### 3.3 合成数据集（Synthetic Dataset）

作者用 **Qwen3-32B** 作为数据合成器，覆盖四类相似度任务：**Retrieval、Bitext Mining、Classification、STS**——目标是让预训练阶段的嵌入模型对各种下游任务都有先验适配。

**Retrieval 数据合成（重点）——document-to-query，两阶段管线**：

1. **配置阶段（Configuration）**：给定一段文档，让 LLM 决定「Question Type」「Difficulty」「Character（提问人身份）」：
   - Character 从 **Persona Hub**（Ge et al., 2024）里检索出对该文档最相关的 top-5 候选，塞进 prompt；
   - Question Type 分为 keyword / factual / summary / judgment / background / acquire_knowledge / yes_or_no 等；
   - Difficulty 分为 high_school / university / phd。
   
   这一步的目的是**用「用户视角」注入多样性和真实感**——同一份文档不同角色的提问方式差异很大。

2. **Query 生成阶段（Query Generation）**：把第一阶段选好的 Character、Question Type、Difficulty，加上目标 query 的 length 和 language，让 LLM 生成最终 query。

**规模**：合成对总量约 **150M**——多任务、多语言的弱监督训练素材。

**筛选**：SFT 阶段挑一批高质量子集加入：**用余弦相似度 > 0.7 作为门槛，从随机采样中过滤**，最终保留约 **12M 高质量合成 SFT 数据**。

**训练数据统计（表 6）：**

| 阶段 | 数据集 | 规模 |
| :--- | :--- | :---: |
| Weakly Supervised Pre-Training | Synthetic Data（Qwen3-32B 合成） | ~150M |
| Supervised Fine Tuning | MS MARCO、NQ、HotpotQA、NLI、Dureader、T2-Ranking、SimCLUE、MIRACL、MLDR、Mr.TyDi、Multi-CPR、CodeSearchNet 等 + 高质量合成数据 | Labeled ~7M、Synthetic ~12M |

**表 6：** 每个阶段的数据规模。SFT 阶段的公开数据集覆盖英文、中文、多语与代码，配合 12M 高质量合成对一起训练。

---

## 4 评测（Evaluation）

### 4.1 设置

**嵌入评测**——统一在 **MMTEB**（Massive Multilingual Text Embedding Benchmark，Enevoldsen et al., 2025）上评。MMTEB 是 MTEB（Muennighoff et al., 2023）的社区扩展，覆盖 250+ 语言、500+ 任务，包括 retrieval、classification、STS 等经典任务，也覆盖 instruction following、long-document retrieval、code retrieval 等新任务。作者最终评了 **216 个任务**：MTEB Multilingual 131、MTEB English v2 41、CMTEB 32、MTEB Code 12。

**重排评测**——挑一批 retrieval 任务：

1. **基础相关性检索**：MTEB（英文）、CMTEB（中文）、MMTEB（多语）、MLDR（Chen et al., 2024）；
2. **代码检索**：MTEB-Code；
3. **复杂指令检索**：FollowIR（Weller et al., 2024）。

**对比方法**——开源：GTE、E5、BGE、NV-Embed-v2、GritLM-7B；商用 API：OpenAI text-embedding-3-large、Google Gemini-Embedding、Cohere-embed-multilingual-v3.0；重排对比 jina-reranker-v2、mGTE、BGE-m3。

### 4.2 主要结果

#### 4.2.1 MTEB Multilingual（表 2）

| 模型 | Size | Mean(Task) | Mean(Type) | BitextMining | Class. | Cluster. | InstRetrieval | MultiClass | PairClass. | Rerank | Retrieval | STS |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NV-Embed-v2 | 7B | 56.29 | 49.58 | 57.84 | 57.29 | 40.80 | 1.04 | 18.63 | 78.94 | 63.82 | 56.72 | 71.10 |
| GritLM-7B | 7B | 60.92 | 53.74 | 70.53 | 61.83 | 49.75 | 3.45 | 22.77 | 79.94 | 63.78 | 58.31 | 73.33 |
| BGE-M3 | 0.6B | 59.56 | 52.18 | 79.11 | 60.35 | 40.88 | -3.11 | 20.1 | 80.76 | 62.79 | 54.60 | 74.12 |
| multilingual-e5-large-instruct | 0.6B | 63.22 | 55.08 | 80.13 | 64.94 | 50.75 | -0.40 | 22.91 | 80.86 | 62.61 | 57.12 | 76.81 |
| gte-Qwen2-1.5B-instruct | 1.5B | 59.45 | 52.69 | 62.51 | 58.32 | 52.05 | 0.74 | 24.02 | 81.58 | 62.58 | 60.78 | 71.61 |
| gte-Qwen2-7b-instruct | 7B | 62.51 | 55.93 | 73.92 | 61.55 | 52.77 | 4.94 | 25.48 | 85.13 | 65.55 | 60.08 | 73.98 |
| text-embedding-3-large | — | 58.93 | 51.41 | 62.17 | 60.27 | 46.89 | -2.68 | 22.03 | 79.17 | 63.89 | 59.27 | 71.68 |
| Cohere-embed-multilingual-v3.0 | — | 61.12 | 53.23 | 70.50 | 62.95 | 46.89 | -1.89 | 22.74 | 79.88 | 64.07 | 59.16 | 74.80 |
| Gemini Embedding | — | 68.37 | 59.59 | 79.28 | 71.82 | 54.59 | 5.18 | 29.16 | 83.63 | 65.58 | 67.71 | 79.40 |
| **Qwen3-Embedding-0.6B** | 0.6B | 64.33 | 56.00 | 72.22 | 66.83 | 52.33 | 5.09 | 24.59 | 80.83 | 61.41 | 64.64 | 76.17 |
| **Qwen3-Embedding-4B** | 4B | 69.45 | 60.86 | 79.36 | 72.33 | 57.15 | 11.56 | 26.77 | 85.05 | 65.08 | 69.60 | 80.86 |
| **Qwen3-Embedding-8B** | 8B | **70.58** | **61.69** | **80.89** | **74.00** | **57.65** | 10.06 | 28.66 | **86.40** | 65.63 | **70.88** | **81.08** |

**表 2：** MTEB Multilingual（Enevoldsen et al., 2025）结果，分数取自 2025-06-04 的在线榜单。

**要点**：Qwen3-8B **70.58 分**超过 Gemini-Embedding（68.37）；Qwen3-0.6B **64.33** 已经追平甚至超过大多数 7B 开源模型（NV-Embed-v2、GritLM-7B），性价比突出。

#### 4.2.2 MTEB English v2 / CMTEB / MTEB Code（表 3）

| 模型 | Size | Dim | MTEB(Eng, v2) Task | MTEB(Eng, v2) Type | CMTEB Task | CMTEB Type | MTEB Code |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| NV-Embed-v2 | 7B | 4096 | 69.81 | 65.00 | 63.0 | 62.0 | — |
| GritLM-7B | 7B | 4096 | 67.07 | 63.22 | — | — | 73.6 |
| multilingual-e5-large-instruct | 0.6B | 1024 | 65.53 | 61.21 | — | — | 65.0 |
| gte-Qwen2-1.5B-instruct | 1.5B | 1536 | 67.20 | 63.26 | 67.12 | 67.79 | — |
| gte-Qwen2-7B-instruct | 7B | 3584 | 70.72 | 65.77 | 71.62 | 72.19 | 56.41 |
| text-embedding-3-large | — | 3072 | 66.43 | 62.15 | — | — | 58.95 |
| Cohere-embed-multilingual-v3.0 | — | 1024 | 66.01 | 61.43 | — | — | 51.94 |
| Gemini Embedding | — | 3072 | 73.30 | 67.67 | — | — | 74.66 |
| **Qwen3-Embedding-0.6B** | 0.6B | 1024 | 70.70 | 64.88 | 66.33 | 67.44 | 75.41 |
| **Qwen3-Embedding-4B** | 4B | 2560 | 74.60 | 68.09 | 72.26 | 73.50 | 80.06 |
| **Qwen3-Embedding-8B** | 8B | 4096 | **75.22** | **68.70** | **73.83** | **75.00** | **80.68** |

**表 3：** MTEB(Eng, v2) / CMTEB / MTEB(Code) 结果。CMTEB 上 Qwen3-8B 73.83 > gte-Qwen2-7B 71.62；MTEB Code 上 Qwen3-8B **80.68** 大幅领先 Gemini（74.66），是 Qwen3 系列在代码检索上尤其显眼的一项。

#### 4.2.3 重排模型（表 4）

评测协议：**先用 Qwen3-Embedding-0.6B 检索 top-100**，再让不同重排模型对同一份候选做二次排序——公平对齐首轮召回。

| 模型 | Param | MTEB-R | CMTEB-R | MMTEB-R | MLDR | MTEB-Code | FollowIR |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Embedding-0.6B（首轮） | 0.6B | 61.82 | 71.02 | 64.64 | 50.26 | 75.41 | 5.09 |
| Jina-multilingual-reranker-v2-base | 0.3B | 58.22 | 63.37 | 63.73 | 39.66 | 58.98 | -0.68 |
| gte-multilingual-reranker-base | 0.3B | 59.51 | 74.08 | 59.44 | 66.33 | 54.18 | -1.64 |
| BGE-reranker-v2-m3 | 0.6B | 57.03 | 72.16 | 58.36 | 59.51 | 41.38 | -0.01 |
| **Qwen3-Reranker-0.6B** | 0.6B | 65.80 | 71.31 | 66.36 | 67.28 | 73.42 | 5.41 |
| **Qwen3-Reranker-4B** | 4B | **69.76** | 75.94 | 72.74 | 69.97 | 81.20 | **14.84** |
| **Qwen3-Reranker-8B** | 8B | 69.02 | **77.45** | **72.94** | **70.19** | **81.22** | 8.05 |

**表 4：** 重排模型结果。所有分数为作者在同一份 top-100 候选上重排后的结果。三档 Qwen3-Reranker 均**优于原始首轮嵌入检索** + **超过所有基线重排**；Qwen3-Reranker-4B 在 FollowIR（复杂指令检索）上拿到 14.84 的高分——比 8B 更强，说明该任务上 4B 的规模 / 训练组合已足够。

### 4.3 消融分析

作者围绕两条主线做消融——大规模弱监督预训练是否有用、模型合并是否有用——都以 Qwen3-Embedding-0.6B 作为载体（表 5）：

| 变体 | MMTEB | MTEB(Eng, v2) | CMTEB | MTEB(Code, v1) |
| :--- | ---: | ---: | ---: | ---: |
| Qwen3-Embedding-0.6B w/ only synthetic data（**仅** Stage 1） | 58.49 | 60.63 | 59.78 | 66.79 |
| Qwen3-Embedding-0.6B w/o synthetic data（**跳过** Stage 1） | 61.21 | 65.59 | 63.37 | 74.58 |
| Qwen3-Embedding-0.6B w/o model merge（**跳过** Stage 3） | 62.56 | 68.18 | 64.76 | 74.89 |
| **Qwen3-Embedding-0.6B（完整三阶段）** | **64.33** | **70.70** | **66.33** | **75.41** |

**表 5：** 不同训练配置下 Qwen3-Embedding-0.6B 的性能（各基准的 mean-task 分数）。

**观察 1：大规模弱监督预训练很关键。** 只跑第一阶段合成数据、什么都不干（第一行）就能拿到 58.49 / 60.63 / 59.78 / 66.79 的合理分数——说明合成数据本身信号密度已经不低；而**跳过第一阶段**（第二行）后，SFT + merge 的结果显著低于完整管线——例如 MTEB(Eng, v2) 从 70.70 掉到 65.59，MTEB Code 从 75.41 掉到 74.58。

**观察 2：模型合并也是关键一环。** 去掉 Stage 3（第三行）仍然是各任务均衡采样的强 SFT 结果，但相比完整管线在 MMTEB 上少 1.77 分、在 MTEB(Eng, v2) 上少 2.52 分——**SLERP 合并对最终结果的贡献是稳定且明显的**。

**观察 3：合成数据在 SFT 阶段的价值。** 「仅合成」（58.49）和「无合成」（61.21）之间差距其实不大——大部分能力可以从公开 SFT 数据里学到；但**完整管线**（64.33）比「无合成」（61.21）高 3.12——说明**合成数据的价值不在替代公开数据，而在提供跨语言 / 跨领域的补充信号**。这与作者的定位一致：合成数据主要用于 Stage 1 无监督预训练，Stage 2 只挑高质量子集做补充。

---

## 5 结论（Conclusion）

作者提出 **Qwen3-Embedding 系列**——建立在 Qwen3 基础模型之上的一套完整嵌入 / 重排模型，覆盖多语检索、代码检索、复杂指令跟随等场景。**技术核心**：

1. 一条**多阶段训练管线**——大规模弱监督预训练 + 高质量 SFT + 模型合并；
2. **Qwen3 LLM 双重身份**——既做骨干，又做数据合成器（Qwen3-32B 合成了 ~150M 多任务多语对）；
3. **SLERP 模型合并**是最终稳定跨分布性能的关键一步；
4. 三档规模全覆盖，Embedding 支持 **MRL** 灵活维度，Embedding 与 Reranker 都支持**任务化 instruction**。

**评测结论**：在 MTEB Multilingual / MTEB Eng v2 / CMTEB / MTEB Code 上全面 SOTA；Qwen3-8B-Embedding 70.58 MTEB Multilingual、80.68 MTEB Code，超过 Gemini-Embedding。全系列以 **Apache 2.0** 开源，方便社区二次开发。

---

## 附录（Appendix）

### A.1 合成数据 Prompt 模板

**Retrieval 数据合成——两阶段管线（部分模板）**：

**Stage 1（Configuration）——选定 Character / Question Type / Difficulty**：

- 输入：Passage + 5 个 Character 候选（来自 Persona Hub）；
- 让 LLM 从候选中挑最合适的 Character；
- 让 LLM 从 keywords / acquire_knowledge / summary / yes_or_no / background 中挑 Question Type；
- 让 LLM 从 high_school / university / phd 中挑 Difficulty；
- 输出 JSON。

**Stage 2（Query Generation）**：

- 输入：Passage、Character、Requirement（Type、Difficulty、Length、Language）；
- LLM 以选定 Character 视角生成一条能召回该 Passage 的 Query；
- 输出 JSON——**value 使用目标 query 语言、key 保持英文**。

### A.2 各 benchmark 详细分数（表 7-9 摘要）

**表 7：MTEB(eng, v2) 各任务分数**——Qwen3-Embedding-8B 在 Classification（90.43）、STS（88.58）、Retrieval（69.44）、PairClass.（87.52）、Rerank（51.56）等大多数子任务上领先，Mean(Task) 75.22、Mean(Type) 68.70，都在开源与商用模型中最强。

**表 8：C-MTEB（MTEB cmn, v1）**——Qwen3-Embedding-8B 各子任务：Classification 76.97、Clustering 80.08、PairClass. 84.23、Rerank 66.99、Retrieval 78.21、STS 63.53，Mean(Task) 73.84、Mean(Type) 75.00，全面领先 gte-Qwen2-7B。

**表 9：MTEB(Code, v1) 12 项代码检索任务** nDCG@10：

- Qwen3-Embedding-8B：**Avg 80.68**——CosQA 38.04、CodeSearchNet 96.35、Apps 91.07、CodeEditSearch 76.97、CodeFeedback-MT 93.70、SyntheticText2SQL 78.75；
- Qwen3-Reranker-8B：**Avg 81.22**——Apps 94.55、CodeSearchNet-CCR 95.67、Code-Trans-Ocean-Contest 90.83、StackOverflowQA 97.3；
- 相比 gte-Qwen2-7B-instruct 62.17、NV-Embed-v2 63.74、BGE-M3 (dense) 58.22，**代码检索维度的绝对优势最明显**（+15 到 +20 分）。

---

*翻译约定：多阶段训练（multi-stage training）、弱监督预训练（weakly supervised pre-training）、有监督微调（supervised fine tuning，SFT）、模型合并（model merging）、球面线性插值（Spherical Linear Interpolation，SLERP）、逐点重排（point-wise reranking）、指令感知（instruction aware）、灵活维度（Matryoshka Representation Learning，MRL）、假负样本（false negative）、困难负样本（hard negative）、in-batch negatives、Persona Hub、任务感知（task-aware）、余弦相似度阈值（cosine similarity threshold）、代码检索（code retrieval）、复杂指令检索（complex instruction retrieval）、长文档检索（long-document retrieval）、跨语言检索（cross-lingual retrieval）、召回 / 首轮检索（first-stage retrieval）、重排（reranking）。Qwen3 / Qwen3-Embedding / Qwen3-Reranker / GTE-Qwen / E5 / BGE / BGE-M3 / NV-Embed / GritLM / Gemini-Embedding / Cohere-embed / OpenAI text-embedding-3 / Jina-reranker / mGTE / MTEB / CMTEB / MMTEB / MTEB-Code / MLDR / FollowIR / MS MARCO / NQ / HotpotQA / NLI / Dureader / T2-Ranking / SimCLUE / MIRACL / Mr.TyDi / Multi-CPR / CodeSearchNet / Persona Hub / SLERP / InfoNCE / Apache 2.0 按惯例不译。*
