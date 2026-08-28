# Nomic-Embed 技术详解

> 基于技术报告 [Nomic Embed: Training a Reproducible Long Context Text Embedder](https://arxiv.org/abs/2402.01613)（Nussbaum et al., Nomic AI）与训练代码/数据 [nomic-ai/contrastors](https://github.com/nomic-ai/contrastors)。
> 本文把 **nomic-bert-2048 长上下文 MLM、235M 开放对比对、contrastors 训练栈、Dynamic NTK 推到 8192、以及 MTEB/LoCo/JinaLC 评测** 写全。

---

## 1. 一句话定位

**nomic-embed-text-v1** 是 Nomic AI 发布的 **可完全复现** 的英文长上下文文本嵌入模型：


| 项 | 内容 |
| --- | --- |
| 参数量 | **137M**（BERT-base 量级改编） |
| 上下文 | 训练 **2048**，推理 **Dynamic NTK → 8192** |
| 数据 | 开源 **~235M** 过滤后弱监督对 + 1.6M 级监督微调 |
| 代码 | Apache-2.0：**contrastors** |
| 宣称 | 同档开源长上下文中，**同时在短上下文 MTEB 与长上下文 LoCo 上超过** OpenAI `text-embedding-ada-002` 与 `text-embedding-3-small` |

与多数「只开权重」的嵌入模型不同，Nomic 强调 **weights + code + data** 三开，便于审计与复现。

---

## 2. 问题背景

2023–2024 开源 MTEB 前列多为 **512 上下文**（E5 / GTE / BGE）。长上下文需求（整页文档 RAG、专利、会议记录）却大量依赖闭源 API（Ada-002、embedding-3、Voyage 等）。开源长上下文选项中：

- **jina-embeddings-v2-base**：8192，但短上下文 MTEB 未稳定超过 Ada；
- **E5-Mistral-7B**：分数高，但 **7B 推理重**，且官方不鼓励超过 4K。

Nomic 目标：**~100M 参数级、8192、开源可审计，并且短+长榜都打过 OpenAI small/ada**。

---

## 3. 三阶段训练范式（总览）

与 E5/GTE 同构的经典三段：

```text
① Masked Language Modeling  →  nomic-bert-2048
② Weakly-supervised contrastive pretraining  →  大 batch InfoNCE
③ Supervised contrastive finetuning  →  hard negatives
```

### 3.1 弱监督 InfoNCE

对 batch $B=\{(q_i,d_i)\}_{i=1}^{n}$：

$$
\mathcal{L}_{C}
=
-\frac{1}{n}\sum_{i}
\log
\frac{
e^{s(q_i,d_i)/\tau}
}{
e^{s(q_i,d_i)/\tau}
+
\sum_{j\neq i}
e^{s(q_i,d_j)/\tau}
}.

$$

$s$ 取 **余弦相似度**。Nomic 采用 **单向** query→document 对比（不像部分工作再加 document→query 对称项）。

### 3.2 监督阶段加入 hard negatives

$$
\mathcal{L}_{C}
=
-\frac{1}{n}\sum_{i}
\log
\frac{
e^{s(q_i,d_i)/\tau}
}{
e^{s(q_i,d_i)/\tau}
+
\sum_{j\neq i}e^{s(q_i,d_j)/\tau}
+
\sum_{m=1}^{H}
e^{s(q_i,d_{\mathrm{hn}}(i,m))/\tau}
}.

$$

实践取 **$H=7$**；再多难负例收益有限。多 epoch 有害，故 **只训 1 epoch**。

---

## 4. 阶段①：nomic-bert-2048（长上下文 MLM）

### 4.1 数据

BooksCorpus + **2023 Wikipedia dump**；`bert-base-uncased` 分词后 **pack 到 2048 tokens**（短拼长、超长切分）。去掉 NSP（RoBERTa/MosaicBERT 经验）。

### 4.2 架构相对 BERT-base 的修改


| 改动 | 动机 |
| --- | --- |
| **RoPE** 替换绝对位置编码 | 长上下文外推 |
| **SwiGLU** 替换 GeLU | 质量；相对 GeGLU 在 FA 实现上约快 25% |
| **FlashAttention** | 内存/速度 |
| Dropout $=0$ | Cramming 等经验 |
| Vocab size 对齐 64 倍数 | 吞吐 |

得到 **137M** 编码器。掩码率 **30%**（非 15%）。AdamW：$5\times10^{-4}$，$\beta_2=0.98$，batch **4096**，DeepSpeed ZeRO-2，bfloat16。

### 4.3 RoPE 长上下文外推

训练长度 $L=2048$，推理目标 $L'=8192$。采用 **Dynamic NTK scaling**（相对静态 NTK / Position Interpolation）：

静态 NTK 思想：按长度比缩放 RoPE base $b$：

$$
b' = b\cdot s^{\frac{|D|}{|D|-2}},
\qquad
s=\frac{L'}{L}.

$$

Dynamic NTK 引入 $\alpha$，使短序列几乎不变、长序列平滑放大：

$$
b' = b\cdot\big((\alpha\cdot s)-(\alpha-1)\big)^{\frac{|D|}{|D|-2}}.

$$

评测时长文使用 $\alpha=2$。也可无微调直接外推（Peng/YaRN、emozilla 等社区结论）；Nomic 全阶段仍以 2048 训练，**8192 纯推理外推**。

### 4.4 GLUE  sanity check

nomic-bert-2048 在 GLUE 八任务均值约 **0.84**，与 MosaicBERT-2k（0.85）、JinaBERT（0.83）同档——说明长上下文改装 **未毁掉** 短句 NLU 能力（Cola 略弱可归因语料/位置编码差异）。

---

## 5. 阶段②：弱监督对比预训练

### 5.1 数据规模与一致性过滤

从 29 个公开源收集约 **470M** 原始对，用 **gte-base** 做一致性过滤（不用 all-MiniLM：易丢掉「相关但词汇重叠低」的检索对）：

1. 对每源 subsample 约 1M；
2. 分别嵌入 query / document；
3. 对每个 query 取余弦 top-$k$（$k=2$）；
4. 若正例 document **不在** top-$k$，丢弃。

过滤后约 **235M** 对（精确统计表：234,553,344）。曾尝试「余弦阈值过滤」，但易杀伤低相似度真检索对，且下游检索更差，故弃用。

### 5.2 长上下文对的专门构造

多数公开对 $<2048$。额外加入：

- Wikipedia **title ↔ full body**；
- S2ORC **abstract ↔ full paper**（同文）。

以学习跨段依赖。

### 5.3 数据源分布（过滤后，节选）


| Dataset | 点数 | 约占 |
| --- | --- | --- |
| Reddit title-body | 65.0M | 28% |
| PAQ | 53.0M | 23% |
| Amazon Reviews | 38.7M | 16% |
| S2ORC Title-Abstract | 35.4M | 15% |
| WikiAnswers | 9.9M | 4% |
| 其它（Wiki title-body、Codesearch、News…） | — | ~14% |
| **Total** | **~234.6M** | 100% |

### 5.4 训练技巧

| 项 | 设定 |
| --- | --- |
| 初始化 | nomic-bert-2048 |
| Global batch | **16,384**（极大 in-batch 负例） |
| 序列长 | 2048 |
| 优化 | AdamW $2\times10^{-4}$，warmup 700，inverse sqrt decay |
| 内存 | **GradCache** + 混合精度 |
| Batch 构成 | **单源填满整个 batch**，减少「靠数据源捷径」作弊 |

### 5.5 任务前缀（打破双塔对称）

与 E5 同思想，但前缀词表更任务化：

- `search_query` / `search_document` —— 非对称检索；
- `classification` —— STS / 复述等对称；
- `clustering` —— 聚类式语义聚合。

对称任务两侧同前缀；检索 query/document 不同前缀。经典例子：「What is the capital of France?」贴近改写问句还是贴近「Paris is…」——**无前缀则 STS 与 QA 奖励冲突**。

---

## 6. 阶段③：监督对比微调

### 6.1 数据（约 1.6M）


| Dataset | Samples |
| --- | --- |
| MSMarco | 484,864 |
| NLI | 275,200 |
| Reddit | 199,680 |
| MEDI SuperNLI | 177,408 |
| HotpotQA | 169,728 |
| FEVER | 139,776 |
| MEDI StackExchange / Flickr / Wiki | ~176K |
| NQ | 69,888 |

检索集用 **gte-base** 挖 top-20 难负例（排除正例）；非检索集随机负例（挖难负例无收益）。每 pair **随机采样** 已挖负例而非固定前 $N$ 个，以降低假负例固化。

### 6.2 超参

Batch **256**，$H=7$，LR $2\times10^{-5}$，warmup 400，**1 epoch**。同样使用任务前缀。

### 6.3 Ablated 变体

因 BGE/GTE/E5-Mistral 等会吃 BEIR 训练集（FEVER、HotpotQA），Nomic 另训 **nomic-embed-text-v1-ablated**（去掉 FEVER、HotpotQA、MEDI）。MTEB 约降 **1** 分（62.39→61.36），但 LoCo 反而有时更高——说明 BEIR 监督与长文检索任务 **非完全同向**。

---

## 7. 评测结果

### 7.1 总表（论文 Table 1）


| Model | Params | Seq | MTEB | LoCo | Jina LC | 开源数据 |
| --- | --- | --- | --- | --- | --- | --- |
| **nomic-embed-text-v1** | 137M | 8192 | **62.39** | **85.53** | 54.16 | Yes |
| nomic-embed-text-v1-ablated | 137M | 8192 | 61.36 | **86.89** | 53.53 | Yes |
| jina-base-v2 | 137M | 8192 | 60.39 | 85.45 | 51.90 | No |
| text-embedding-ada-002 | — | 8192 | 60.99 | 52.70 | 55.25 | No |
| text-embedding-3-small | — | 8192 | 62.26 | 82.4 | 58.21 | No |
| E5-Mistral-7B | 7B | 4096 | 66.6 | 87.8 | — | No data |

要点：在 **~100M 开源可复现** 约束下，Nomic 是少数 **MTEB≥Ada/3-small 且 LoCo 远超 Ada** 的模型；绝对分数仍低于 7B 级 E5-Mistral，但推理成本差两个数量级。

### 7.2 MTEB 分任务（监督后）

nomic-embed-text-v1：Cls 74.1 / Clust 43.9 / PairCls 85.2 / Rerank 55.7 / Retr **52.8** / STS 82.1 / Summ 30.1 → **Avg 62.4**。  
无监督对比检查点已达 Avg **59.9**（Retr 48.0），说明弱监督阶段已很强。

评测约定：分类类任务用 `classification` 前缀且 **可不 L2 norm**；检索用 search_* 前缀并 L2；文本截断 **512**（短评测）。

### 7.3 Jina Long Context

在 128 / 512 / 8191 三档上，Nomic 全面优于 jina-base-v2；8K 档 Avg **54.2** vs Jina **51.9**，接近 Ada **55.3**，仍低于 embedding-3-small/large。WikiCities 随长度上升分数下降（多模型共性）——该集聚类可能不适合作为「长文能力」唯一指标。

### 7.4 LoCo

8192 时 nomic-embed **85.5**，远超 Ada **52.7** 与 3-small **82.4**；与 Jina 持平附近，ablated 达 **86.9**。4096 档已可接近 E5-Mistral（87.8）在部分子集上的表现，同时参数约为其 **1/50**。

---

## 8. contrastors：可复现训练栈

### 8.1 资源

单机 **8×H100** 约一周：

- MLM ≈ 4 天；
- 弱监督对比 ≈ 3.5 天；
- 监督微调 ≈ 1 小时。

官方建议也可从已发布的 nomic-bert-2048 或 unsupervised checkpoint 热启。

### 8.2 审计意义

论文强调：闭源与「半开源」嵌入在高合规场景存在供应链风险（引用 sleeper agents 等讨论）。Nomic 通过开放 **数据 loader（235M 对）+ 代码 + 权重**，使第三方可：

1. 复现分数；
2. 审计污染与偏见；
3. 在私有域继续预训练而不依赖不可见配方。

5M 样本可视化见 Atlas 地图（报告链接）。

---

## 9. 与 E5 / Jina / OpenAI 的设计对照


| 维度 | E5 | Jina v2 | Nomic v1 |
| --- | --- | --- | --- |
| 上下文 | 512 | 8192 | **8192（外推）** |
| 位置 | 绝对/冻结 | ALiBi 等 | **RoPE + Dynamic NTK** |
| 弱监督过滤 | 自训过滤模型 | 未完整开数据 | **gte-base top-k，数据开源** |
| 前缀 | query:/passage: | 自有 | search_*/classification/clustering |
| 开源完整度 | 权重为主 | 权重 | **权重+代码+数据** |
| 参数 | 33–330M | 137M | **137M** |

---

## 10. 实现对照清单

```text
1. BERT-base → RoPE + SwiGLU + FA + drop0 → MLM@2048, mask 30%
2. 收集弱对 → gte-base consistency top-2 → ~235M
3. 加 Wiki/S2ORC 长对；单源 batch；GradCache；batch 16k InfoNCE
4. 任务前缀 search_query/document, classification, clustering
5. 监督集 + 7 hard neg；1 epoch
6. 推理 >2048 用 Dynamic NTK (α=2) 至 8192
```

可验证目标：

1. GLUE 均值 ≈0.84 → MLM 改装健康；
2. unsup checkpoint MTEB ≈60 → 弱监督有效；
3. 8192 LoCo ≫ Ada → 长上下文外推成立；
4. ablated vs full：MTEB/LoCo 此消彼长 → 理解 BEIR 泄漏权衡。

---

## 11. 小结

Nomic-Embed 的贡献不只是「又一个 62 分 MTEB 模型」，而是证明：**开放数据 + 经典三阶段对比 + 认真的长上下文 Encoder 改装**，可以在 **137M** 级别同时打赢闭源 small/ada 的短评测与长检索，并把复现门槛降到「一台 8×H100 + contrastors」。对工业界：需要合规审计或私有域继续训时，它是比 7B LLM 嵌入更务实的长文基线；对研究界：其过滤与前缀设计可直接迁到其它骨干。

同目录对照：《E5详解.md》（弱监督过滤鼻祖）、《GTE系列详解.md》（LLM 长上下文另一极）、《难负例挖掘工业实践.md》（监督阶段负例）。

---

## 参考

1. Nussbaum et al. (2024). Nomic Embed. [arXiv:2402.01613](https://arxiv.org/abs/2402.01613)
2. https://github.com/nomic-ai/contrastors
3. Portes et al. MosaicBERT; Su et al. RoFormer; Peng et al. YaRN
4. Wang et al. E5; Li et al. GTE; Günther et al. Jina Embeddings 2
