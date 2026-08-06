# QZhou-Embedding 技术详解

> **paper**: [arXiv:2508.21632](https://arxiv.org/abs/2508.21632) · [PDF](https://arxiv.org/pdf/2508.21632)  
> **code / weights**: [GitHub](https://github.com/Kingsoft-LLM/QZhou-Embedding) · [HF Kingsoft-LLM/QZhou-Embedding](https://huggingface.co/Kingsoft-LLM/QZhou-Embedding)（Apache-2.0）  
> **refs**: [LLM2Vec](https://arxiv.org/abs/2404.05961) · [Echo](https://arxiv.org/abs/2402.15449) · [gte / gte-Qwen2](https://arxiv.org/abs/2308.03281) · [Token Prepending](Token-Prepending详解.md)  
> **backbone**: Qwen2.5-7B-Instruct → 双向注意力 + mean pooling + L2  
> **date**: 2025-08（Kingsoft AI / 金山）  
> **modality**: 文本  
> **languages**: 中英为主（冲榜 MTEB eng + CMTEB cmn）  
> **local PDF**: `docs/papers/embedding/QZhou-Embedding_2508.21632.pdf`  
> 本文写全 **多任务框架、训练/评测数据集清单、LLM 合成、Data Grouping、两阶段训练与 MTEB/CMTEB 结果**。

---

## 1. 一句话定位

**QZhou-Embedding** 以 **Qwen2.5-7B-Instruct** 为底座，改成 **双向注意力 + mean pooling**，再用 **检索 / NLI / 分类** 统一多任务框架 + **LLM 合成数据** + **两阶段全参微调**，在 2025-08-27 前后的 **MTEB（英）与 CMTEB（中）** 榜上宣称双榜第一。

| 项 | 内容 |
|----|------|
| 底座 | Qwen2.5-7B-Instruct |
| 架构改动 | causal → **bi-directional**；**mean pooling** + L2 Norm |
| 训练 | Stage1 纯检索 32k step → Stage2 全任务 8k step（$\eta_{\mathrm{RET}}=0.72$） |
| 微调方式 | **全参**（非 LoRA） |
| 数据 | 约 **11M** 四元组 `(query, pos, neg, instruction)`；合成约 5M 高质量样本后汇入 |
| 硬负例 | 每样本 **4** 个；$\tau=0.02$ |
| 长度 | 训练 query/passage **256 / 1536**；RoPE 外推可用到 ~8K |
| 开源 | Apache-2.0 |

与 Jasper 蒸馏叙事的关系：Jasper-Token-Compression 把 **QZhou（7B）+ Qwen3-Embedding-8B** 当作互补教师；QZhou 在 STS/细粒度语义上极强（见 §8 注）。

---

## 2. 背景：Decoder Embedding 的三条岔路

报告 §1 把「因果注意力限制上下文」的解法分成几类（此处补全对照）：

| 路线 | 代表 | 做法 |
|------|------|------|
| 重复输入 | Echo Embedding | 句子看两遍，第二遍前缀可见后缀 |
| **免训练层间注入** | **[Token Prepending](Token-Prepending详解.md)** | 上一层 SET 写回句首 `<PST>` |
| 改双向 + 训 | LLM2Vec、Conan-v2 soft-mask、**QZhou**、gte-Qwen2 | 训练期变成真双向编码器 |
| 蒸馏 | Jasper / Stella、Debater | 对齐强教师向量或思维链状态 |

**QZhou 的选择**：不做零训练榨干，而是 **双向 + 全参多任务对比**，用数据合成与两阶段课表冲榜。Token-Prepending 解释的是同一物理问题的「轻量补丁」；读 QZhou 前建议先读 TP，便于理解「为什么一定要动注意力」。

---

## 3. 模型架构

1. 加载 Qwen2.5-7B-Instruct；  
2. 所有层注意力改为 **双向**（去掉 causal mask）；  
3. 对最后一层隐状态做 **mean pooling**，再 L2 归一化得到句向量。

与 last-token pooling（E5-Mistral / gte-Qwen2 常见）不同：mean 更偏「全句平均语义」，对 STS/分类友好；检索侧靠指令与难负例拉分。

---

## 4. 统一多任务框架

把几乎所有文本数据映射到三类：**Retrieval / NLI / Classification**，每类有 **数据变换** + **专用损失**。

### 4.1 数据变换

#### Retrieval-oriented

| 源形态 | 变换 |
|--------|------|
| Title–Body/Abstract | title→query，body/abstract→pos（超长截断） |
| Claim–Evidence | claim→query；Supports→pos，Refutes→neg |
| Question–Answer | Question→query，Answer→pos |

#### NLI-oriented → `(x, y, score)` + Cosent

- **STS**：二元标签映到 $\{0,1\}$，或保留连续分；对称任务可 **互换 query/doc 翻倍**。  
- **Textual Entailment**：entailment/neutral/contradiction → 分数 **2 / 1 / 0**；同样可对称翻倍。

#### CLS-oriented（example-based）

对 `(text, label)`：text 作 query；**同标签**其它文本作 pos；**异标签**作 neg（见图 2）。相对「label 文本当 pos」更稳（呼应 NV-Embed 结论）。

---

### 4.2 损失函数（公式全）

#### Retrieval — InfoNCE + query–query 负例（式 1）

受 gte 启发：分母除 hard/in-batch document 负例外，再把 **batch 内其它 query** 当作负例：

$$
\mathcal{L}_{\mathrm{Retrieval}}
=
-\frac{1}{n}\sum_{i}
\log
\frac{
\exp\!\big(\mathrm{sim}(q_i,d_i^{+})/\tau\big)
}{
\exp\!\big(\mathrm{sim}(q_i,d_i^{+})/\tau\big)
+
\sum_j\exp\!\big(\mathrm{sim}(q_i,d_j^{-})/\tau\big)
+
\sum_{j\neq i}\exp\!\big(\mathrm{sim}(q_i,q_j)/\tau\big)
}.
$$

$\mathrm{sim}$ 为余弦（向量已归一化时即点积）；论文训练 $\tau=0.02$。

#### NLI — Cosent（式 2）

基于 Circle loss 思想的排序敏感损失，只需序关系 $\mathrm{sim}(i,j)>\mathrm{sim}(k,l)$：

$$
\mathcal{L}_{\mathrm{NLI}}
=
\log\!\Bigg(
1
+
\sum_{\mathrm{sim}(i,j)>\mathrm{sim}(k,l)}
\exp\!\Big(
\frac{\mathrm{sim}(x_k,x_l)-\mathrm{sim}(x_i,x_j)}{\tau}
\Big)
\Bigg).
$$

无硬负例前向时显存更省 → Stage2 上 Cosent batch **768**（InfoNCE 为 **256**）。

#### CLS — 带类别 MASK 的 InfoNCE（式 3）

example-based 下，in-batch 易把 **同类** 误当负例。预处理给每条样本挂上类别 $C_t$（不写进原文），负例项乘 MASK：

$$
\mathrm{MASK}(t_i,t_j)
=
\begin{cases}
0 & C_{t_i}=C_{t_j}\\
1 & \text{otherwise}
\end{cases}
$$

$$
\mathcal{L}_{\mathrm{CLS}}
=
-\frac{1}{n}\sum_i
\log
\frac{\exp(\mathrm{sim}(t_i,t_i^{+})/\tau)}{Z_i},
$$

$$
\begin{aligned}
Z_i
&=
\exp(\mathrm{sim}(t_i,t_i^{+})/\tau)
\\
&\quad+
\sum_n \mathrm{MASK}(t_i,t_{i,n}^{-})\,\exp(\mathrm{sim}(t_i,t_{i,n}^{-})/\tau)
\\
&\quad+
\sum_{j\neq i}\mathrm{MASK}(t_i,t_j)\,\exp(\mathrm{sim}(t_i,t_j)/\tau)
\\
&\quad+
\sum_{j\neq i}\sum_n \mathrm{MASK}(t_i,t_{j,n}^{-})\,\exp(\mathrm{sim}(t_i,t_{j,n}^{-})/\tau).
\end{aligned}
$$

约束 $C_{t_i}=C_{t_i^{+}}$（正例同类）。

---

## 5. 数据合成（LLM API）

三维增强（附录约束见表 4/5）：

| 维度 | 方法 | 作用 |
|------|------|------|
| 结构多样 | **Paraphrasing** | 改写 query 与 pos，语义不变、句法/语态变 |
| 语义多样 | **Augmentation** | 同域扩展话题/视角，禁止纯同义复述 |
| 难度 | **Hard NEG 生成** | 仿 POS 句法但答偏、掺无关、同题不同侧面 |

任务策略差异：

- 检索：改写/扩展 `(q,pos)` 并入原集；可合成 hard neg。  
- NLI：只改写单句并替换，**不做 Augmentation**（防标签糊）。  
- 分类：改写句、保留标签，再走 example-based；不做 Augmentation。

规模：小集（如 <60k）重点合成；最终约 **5M** API 高质量样本，汇入后总训练 **11M** 四元组。用 **gte-Qwen2-7B-instruct** 滤低分 query–pos；缺 hard neg 时 **30% API 生成**，其余用 stella-large-zh-v3 取 top-10～30。

---

## 6. 实验用到的数据集

论文 §6「Training Data」给出训练语料拼盘与过滤流程；评测则是 **MTEB (eng, v2)** 与 **CMTEB (cmn, v1)**。下表按原文整理（名称以报告英文为准）。

### 6.1 规模与流水线

| 项 | 内容 |
|----|------|
| 最终训练 | 约 **11M** 四元组 `(query, pos, neg, instruction)` |
| LLM 合成 | 约 **5M** 高质量样本并入（小集优先 Paraphrase / Augment） |
| 去重与过滤 | 全量去重；**gte-Qwen2-7B-instruct** 滤低分 query–pos |
| Hard neg | 缺负例时：约 **30%** API 合成；其余 **stella-large-zh-v3** 取 top-10～30 |
| 污染控制 | 对 MTEB 相关训练集做 **contamination exclusion**（去掉与测试集高度相似样本） |
| Instruction | 原集保留；MTEB 训练 split 对齐 **Qwen3-Embedding** 评测指令；Huatuo / Reddit / Law-GPT / GLUE 等外域自写指令（附录 A.2） |

### 6.2 数据来源总览（集合级）

| 来源集合 | 说明 |
|----------|------|
| **bge-en-icl / bge-m3-data / bge-multilingual-gemma2-data** | FlagEmbedding 开源拼盘（[FlagEmbedding/dataset](https://github.com/FlagOpen/FlagEmbedding/tree/master/dataset)） |
| **E5 公开数据约 1.5M** | 与 E5-Mistral / Echo / LLM2Vec 同系公开检索监督集 |
| **Stella `retrieval_data_llm`** | 高质量 `(query, pos, neg)` 三元组 |
| **zpoint 相关** | 含 Huatuo 医疗 QA 等 |
| **sentence-transformers / HF 其它** | reddit、hover、mr-tydi、law-gpt、s2orc 等 |
| **杂项开源** | web_questions、BioASQ、cmrc、CSL、nli_for_simcse、MLDR、GLUE、Yelp、Weibo Sentiment 等 |
| **MTEB 评测相关训练 split** | Imdb / MassiveIntent / MassiveScenario / STS12 / LCQMC / PAWSX / STSB 等（仅 train，已做污染剔除） |

异构源经 §4 三类变换后入训；对 bge/e5 子集与小规模 CLS/STS 等做合成增强。

### 6.3 主要具名数据集（训练）

#### 检索 / QA / 事实核查（Retrieval 向）

| 数据集 | 类型简介 |
|--------|----------|
| **MS MARCO**（passage + document） | 大规模网络搜索式 QA；passage/document 检索主粮 |
| **Natural Questions (NQ)** | Google 真实查询 → Wikipedia 答案段落 |
| **ELI5** | 长答案开放式 QA |
| **HotpotQA** | 多跳推理 QA |
| **MIRACL** | 多语种检索（18 语） |
| **SQuAD** | 阅读理解式段落问答 |
| **FEVER** | 声明–证据；Supports→pos，Refutes→neg（Claim–Evidence 变换） |
| **HoVer** | 多跳事实抽取与声明验证 |
| **Quora Question Pairs (QQP)** | 问句对是否同义 / 检索式用法 |
| **DuReader** | 中文真实场景机器阅读理解 |
| **Mr. TyDi** | 多语种稠密检索基准相关数据 |
| **S2ORC** | Semantic Scholar 学术语料（标题–摘要等 Title–Body 源） |
| **BioASQ** | 生物医学语义索引与 QA |
| **CMRC** | 中文跨度抽取式阅读理解 |
| **MLDR** | 多语长文档检索相关 |
| **Stella retrieval_data_llm** | 已挖好难负例的三元组 |
| **Huatuo / Law-GPT / Reddit** | 医疗 QA、法律、论坛语义检索；报告强调自写领域 instruction |

另含新闻 / arXiv / Wikipedia 等 **Title–Body/Abstract** 与论坛 **Question–Answer** 类数据（§3.2.1），报告未逐条点名者经统一变换并入。

#### NLI / STS / 分类（NLI & CLS 向）

| 数据集 | 类型简介 |
|--------|----------|
| **nli_for_simcse**（含 SNLI/MNLI 等） | 蕴含类句对 → Cosent 分数（entail/neutral/contradiction → 2/1/0） |
| **GLUE** | 多任务 NLU 拼盘；外域 instruction |
| **STS12 / STSB** | 语义相似度连续分；可对称翻倍 |
| **LCQMC** | 大规模中文问句匹配 |
| **PAWS-X** | 跨语对抗释义识别 |
| **Imdb-Classification** | 影评情感分类（example-based） |
| **MassiveIntent / MassiveScenario** | Massive NLU 意图/场景分类 |
| **Yelp Reviews** | 商户评论分类 |
| **Weibo Sentiment** | 微博情感分类 |
| **CSL** | 中文科学文献相关分类/检索用法 |

### 6.4 关键训练集一句话介绍

- **MS MARCO**：工业检索标配；短查询对长 passage，撑起 Stage1「纯检索」。  
- **NQ / HotpotQA / FEVER / HoVer**：真实查询、多跳与事实验证，补难负例与推理型检索。  
- **DuReader / CMRC / Huatuo**：中文与垂直域，服务 CMTEB 与领域 instruction。  
- **MIRACL / Mr.TyDi / MLDR / PAWS-X**：多语与长文档覆盖，虽主榜为 eng/cmn，语料更广。  
- **SNLI·MNLI / STS·LCQMC**：Stage2 Cosent 主战场；对称翻倍放大 STS。  
- **Imdb / Massive* / Yelp / Weibo**：example-based 分类，配合同类 MASK InfoNCE。  
- **E5 1.5M + BGE 系列拼盘**：社区已整理好的检索监督底座；报告称主要只更新了更难的负例。

### 6.5 评测基准（实验对比用）

| 基准 | 口径 | 任务类型 | 对比对象（报告口径） |
|------|------|----------|----------------------|
| **MTEB (eng, v2)** | 2025-08-27 榜 | Class. / Clust. / PairCLS / Rerank / STS / Retr. / Summ. | 榜前模型：Qwen3-Emb、Seed1.5/1.6、gemini-embedding、Jasper、Linq/SFR-Mistral、NV-Embed-v2、LGAI-Embedding 等 |
| **CMTEB (cmn, v1)** | 同上 | Class. / Clust. / PairCLS / Rerank / STS / Retr. | Seed、Conan v1/v2、Qwen3-Emb、xiaobu、piccolo、zpoint、ritrieve_zh 等 |

官方排名协议下双榜第一（报告宣称）。子任务分数见 §8。

---

## 7. 训练优化

### 7.1 Data Grouping（式 4）

比「按任务同质 batch」更细：按 **数据集文件** 分组，**每个 batch 只来自同一数据集**（同域 in-batch neg 更难）。采样权重：

$$
p_i
=
\frac{l_i^{\alpha}}{\sum_{j=1}^{m} l_j^{\alpha}},
$$

$l_i$ 为数据集大小，$\alpha$ 同 gte/mgte 设定。

### 7.2 两阶段训练

| | Stage 1 | Stage 2 |
|--|---------|---------|
| 数据 | 仅检索 | 检索 + NLI + CLS |
| Steps | **32k** | **8k** |
| LR | $3\times 10^{-5}$ | $2\times 10^{-5}$ |
| Warm-up | 300 | （同表；Stage2 沿用配置） |
| $\eta_{\mathrm{RET}}$ | 1 | **0.72** |
| 微调 | 全参 | 全参 |

Stage2 用 $\eta$ 控制检索算力占比，避免「一加 STS/CLS 数据检索就塌」：

$$
S_{\mathrm{ret}}=\sum_i M_i\,l_i^{\alpha},\quad
S_{\mathrm{non}}=\sum_i(1-M_i)\,l_i^{\alpha},
$$

$$
l_i^{\mathrm{samp}}
=
\begin{cases}
\dfrac{\eta_{\mathrm{RET}}\,l_i^{\alpha}}{S_{\mathrm{ret}}} & d_i\in\mathrm{RET}\\[8pt]
\dfrac{(1-\eta_{\mathrm{RET}})\,l_i^{\alpha}}{S_{\mathrm{non}}} & \text{else.}
\end{cases}
$$

（$M_i$ 为检索集指示；与论文记号一致处按 RET 集合定义。）

### 7.3 其它超参

- Hard neg：**4** / 样本；bf16；Adam wd **0.01**；  
- InfoNCE batch **256**，Cosent **768**；  
- Instruction：保留原数据指令；MTEB 相关集对齐 Qwen3-Embedding 评测指令；外域自写领域指令（附录 A.2）。

---

## 8. 实验结果（2025-08-27 口径）

### 8.1 MTEB English (eng, v2) — Table 2

| Model | Class. | Clust. | PairCls | Rerank | STS | Retr. | Summ. | Mean(Task) | Mean(Type) |
|-------|--------|--------|---------|--------|-----|-------|-------|------------|------------|
| Qwen3-Embedding-8B | **90.43** | 58.57 | 87.52 | 51.56 | 69.44 | 88.58 | 34.83 | 75.22 | 68.71 |
| Seed1.5-Embedding | 89.88 | 60.83 | 87.39 | 50.67 | 67.45 | 87.23 | 36.44 | 74.76 | 68.56 |
| **QZhou-Embedding** | 88.97 | **61.65** | **92.43** | **51.77** | 67.12 | **91.65** | 33.05 | **75.97** | **69.52** |

### 8.2 CMTEB Chinese (cmn, v1) — Table 3

| Model | Class. | Clust. | PairCls | Rerank | STS | Retr. | Mean(Task) | Mean(Type) |
|-------|--------|--------|---------|--------|-----|-------|------------|------------|
| Seed1.6-embedding | 77.98 | 73.11 | 88.71 | 71.65 | **79.69** | 68.94 | 75.63 | 76.68 |
| Conan-embedding-v2 | 76.47 | 68.84 | 92.44 | 74.41 | 78.31 | 65.48 | 74.24 | 75.99 |
| **QZhou-Embedding** | **79.99** | 70.91 | **95.07** | **74.85** | 78.80 | **71.89** | **76.99** | **78.58** |

**注（读表）**：英文 Table 2 中「STS=67 / Retr=91」量级与常见 MTEB 任务型均值习惯及 Jasper 文「QZhou STS≈91.65」不完全一致，**疑似 STS/Retr 列名对调**；引用分数时建议对照 [官方 HF / MTEB 榜](https://huggingface.co/Kingsoft-LLM/QZhou-Embedding) 实时列。中文表 Retr/STS 量级更符合惯例。PairCls 中英均极强（92.43 / 95.07），与 Cosent + 对称 NLI 数据一致。

---

## 9. 可迁移实践（对自研 Embedding）

1. **先检索、后全能**：Stage1 打底检索，Stage2 用 $\eta_{\mathrm{RET}}$ 保护检索不被 STS/CLS 淹没。  
2. **三类损失不要混成一个 InfoNCE**：NLI 用 Cosent；CLS 必须 **同类 MASK**。  
3. **Data Grouping**：同数据集组 batch，比随机混域更能产真 hard in-batch neg。  
4. **合成三板斧**：Paraphrase（结构）/ Augment（语义）/ HardNEG（难度）；小数据优先。  
5. **全参 vs LoRA**：冲榜选全参；资源紧可对标 gte/E5-Mistral 的 LoRA，但报告认为全参更满血。  
6. **因果问题**：若暂不能双向微调，可用 [Token Prepending](Token-Prepending详解.md) 做零训练基线；要上检索 SOTA 仍建议走 QZhou 式双向。

---

## 10. 局限与后续

- 报告自评：数据质量/多样性是主因；将做多模态、多语与 Agent 记忆场景。  
- 训练成本：7B 全参 + 11M 对 + API 合成，复现门槛高。  
- 未强调 Matryoshka / 弹性维；部署维数与 Qwen3-Embedding 系需自行确认模型卡。  
- Summarization 等子任务并非全面第一（英文 Summ. 33.05），全能≠每格 SOTA。

---

## 11. 公式速查

| 编号 | 名称 | 要点 |
|------|------|------|
| (1) | $\mathcal{L}_{\mathrm{Retrieval}}$ | InfoNCE + $\sum_{j\neq i}\mathrm{sim}(q_i,q_j)$ |
| (2) | $\mathcal{L}_{\mathrm{NLI}}$ | Cosent 序对指数和 |
| (3) | $\mathcal{L}_{\mathrm{CLS}}$ | InfoNCE × 类别 MASK |
| (4) | $p_i$ | 数据集大小幂加权采样 |
| — | $\eta_{\mathrm{RET}}$ | Stage2 检索占比（0.72） |

---

## 12. 结论

QZhou-Embedding 的配方可以压成一句话：

> **双向 Qwen2.5-7B + 三类任务专用损失与数据变换 + LLM 结构/语义/难负例合成 + 数据集级 Grouping + 先检索后全能的两阶段全参训练。**

它与 Token-Prepending 共享「Decoder 看不全上下文」的问题意识，但解法落在 **训练期架构与数据课表**；TP 则是 **推理期免费补丁**。做教师蒸馏（如 Jasper）时，QZhou 适合贡献 **细粒度语义 / PairCLS / 中文** 侧的互补信号。

## 参考文献

1. Yu et al. *QZhou-Embedding Technical Report*. arXiv:2508.21632, 2025.  
2. Fu et al. *Token Prepending*. ACL 2025. https://aclanthology.org/2025.acl-long.159/  
3. Li et al. GTE / gte-Qwen2；Wang et al. E5-Mistral；BehnamGhader et al. LLM2Vec；Springer et al. Echo.  
4. Zhang et al. Jasper & Stella；Qwen3-Embedding.  
