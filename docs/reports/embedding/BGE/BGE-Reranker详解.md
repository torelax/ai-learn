# BGE-Reranker v2 / v2.5：从 XLM-R 到 LLM-as-Reranker

> **paper**：附录 D of [Making Text Embedders Few-Shot Learners (ICLR 2025)](https://arxiv.org/abs/2409.15700) · [BGE-M3 (Chen 2024)](https://arxiv.org/abs/2402.03216)（v2-m3 出处）
> **code / model**：[FlagOpen/FlagEmbedding · rerankers](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_reranker) · [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) · [BAAI/bge-reranker-v2-gemma](https://huggingface.co/BAAI/bge-reranker-v2-gemma) · [BAAI/bge-reranker-v2-minicpm-layerwise](https://huggingface.co/BAAI/bge-reranker-v2-minicpm-layerwise) · [BAAI/bge-reranker-v2.5-gemma2-lightweight](https://huggingface.co/BAAI/bge-reranker-v2.5-gemma2-lightweight)
> **refs**：[BGE-EN-ICL (2024)](https://arxiv.org/abs/2409.15700) · [Cross-Encoder (Reimers & Gurevych 2019)](https://arxiv.org/abs/1908.10084) · [ColBERT (Khattab 2020)](https://arxiv.org/abs/2004.12832) · [monoT5 (Nogueira 2020)](https://arxiv.org/abs/2003.06713) · [RankLLaMA (Ma 2023)](https://arxiv.org/abs/2310.08319) · [RankT5 (Zhuang 2023)](https://arxiv.org/abs/2210.10634) · [MiniCPM (Hu 2024)](https://arxiv.org/abs/2404.06395) · [Gemma-2 (Team 2024)](https://arxiv.org/abs/2408.00118)
> **backbone**：v2-m3：**XLM-R Large (560M)**；v2-gemma：**Gemma-2B**；v2-minicpm-layerwise：**MiniCPM-2B**；**v2.5-gemma2-lightweight：Gemma-2-9B**
> **date**：v2-m3 2024-02（BGE-M3 同期）；v2-gemma / minicpm-layerwise 2024-04；**v2.5-gemma2-lightweight 2024-09**（与 BGE-EN-ICL / bge-multilingual-gemma2 同期）
> **modality**：文本；query + passage 联合编码打分
> **languages**：全部支持多语；v2-m3、v2.5-gemma2 主打多语；v2-gemma 主打英文
>
> 本文写全 **BGE-Reranker 家族的四个版本**如何从「XLM-R Cross-Encoder」演进到「LLM-as-Reranker + 深度/宽度双压缩 + 自蒸馏 + 4 种 prompt 类型」。核心贡献是 **v2.5-gemma2-lightweight 的两级压缩机制**——在保住精度前提下把 FLOPs 砍到 40%，让 LLM 级精排也能上线。这份精排也是 [BGE-EN-ICL](EN-ICL/BGE-EN-ICL详解.md) 与 [bge-multilingual-gemma2](BGE-multilingual-gemma2详解.md) 的蒸馏教师。

---

## 一句话定位

BGE-Reranker 家族 = **给 BGE 嵌入配套的 Cross-Encoder 精排**。核心问题是双塔嵌入的天花板——**query 和 doc 独立编码后只算内积**，无法捕捉细粒度交互；Cross-Encoder 把 query 与 doc 联合塞进同一 Transformer 做深度融合，代价是**必须 online 编码**（无法预建索引），因此只能用于**打候选 top-K**。

四个版本演进：

| 版本                                | 骨干            | 参数  | 输出方式              | 亮点                          |
| ----------------------------------- | --------------- | :---: | --------------------- | ----------------------------- |
| **bge-reranker-v2-m3**              | XLM-R Large     | 568M  | 单头 sigmoid          | 多语精排的标配，同期 BGE-M3 发布 |
| **bge-reranker-v2-gemma**           | Gemma-2B        | 2.5B  | LLM head 取 "Yes" logit | 首次 LLM-as-Reranker，英文最强  |
| **bge-reranker-v2-minicpm-layerwise** | MiniCPM-2B    | 2.5B  | 逐层可出分数           | **分层输出** 支持提前退出        |
| **bge-reranker-v2.5-gemma2-lightweight** | **Gemma-2-9B** | 9B  | LLM head + 双压缩       | **可调深度 + 可调宽度**，60% FLOPs 保 99% 精度 |

| 项              | v2.5-gemma2-lightweight                              |
| --------------- | ---------------------------------------------------- |
| Backbone        | **Gemma-2-9B**（256k tokenizer，多语覆盖强）          |
| 输入模板        | `A: {query} B: {passage} {prompt}`                   |
| 输出            | LLM head 对 "Yes" token 的 logit 作 relevance score  |
| Prompt 类型     | **4 种**：q→p / q→q / p→p / argument→counter        |
| 深度压缩        | **8–42 层可调**（每层都能出分）                        |
| 宽度压缩        | **1/2/4/8 压缩比**（在 8/16/24/32/40 层做 token merge）|
| Self-KD         | 最后一层做 teacher，前面各层做 student，KL loss       |
| LoRA rank / α   | 64 / 32                                               |
| Batch           | 128                                                   |
| Hard neg / query | 15                                                    |
| BEIR Mean       | **63.67**（无压缩）/ **63.10**（60% FLOPs）           |

## 谱系与位置

```text
BERT Cross-Encoder (2019) ── monoBERT / monoT5 (2019-20) ──┐
                                                            │
ColBERTv2 / TAS-B / RocketQA CE 教师系列 (2020-22)          │
                                                            │
                                        ┌───────────────────┴──── BGE-Reranker v2 家族 (2024)
                                        │                          │
                                        │            v2-m3 (XLM-R, 多语默认)
                                        │            v2-gemma (Gemma-2B, 英文强)
                                        │            v2-minicpm-layerwise (MiniCPM-2B, 分层)
                                        │            v2.5-gemma2-lightweight (Gemma-2-9B, 双压缩)
                                        │
                                        │
RankLLaMA / RankT5 / RankZephyr (2022-24) 一批 LLM-as-Reranker 探索
        │
        └── 都是 sequence-level 打分；BGE-Reranker v2-gemma / v2.5 沿此路线做工程化
```

---

## 问题背景：为什么要用 LLM 做 Reranker

Cross-Encoder Reranker 从 monoBERT (2019) 开始就是「double dip」——先 embedder 召回，再 CE 精排。这条路线两个长期瓶颈：

1. **精度上限受 CE 骨干限制**：BERT-large (330M) 或 XLM-R (560M) 是常见起点，进一步扩就掉训练稳定性。
2. **多语能力不均衡**：XLM-R 多语好，但英文单语能力弱于同期 BERT-large；反过来 monoBERT 单语强、多语差。

BGE-Reranker v2-m3 (2024-02) 是**多语精排的经典 CE**（配 BGE-M3 检索使用）；但作者们观察到 **RankLLaMA / RankT5** 已经证明「LLM-as-Reranker」的精度天花板更高。于是从 v2-gemma 开始，BGE-Reranker 转向 **decoder-only LLM 骨干 + LM head 出 "Yes" logit** 的路线。

到 v2.5-gemma2-lightweight，这条路线遇到新问题：**Gemma-2-9B 推理成本高，工业上线 P99 延迟压不住**。作者的解法是**深度压缩 + 宽度压缩 + 自蒸馏** 三件套，让 9B 模型在推理时**动态选层 + 动态合并 token**，达到 40% FLOPs 的同时保住 99% 精度。

---

## LLM-as-Reranker 的输入模板与打分

### 模板

给定 (query, passage) 对，构造 LLM 输入：

```
A: {query}
B: {passage}
{prompt}
```

`{prompt}` 是**任务模板**（4 种，见下节）。整个序列被输入到 Gemma-2 LM，得到 last token 位置的 LM head logits，**取 "Yes" token 的 logit 作为 relevance score**：

$$
s(q, p) = \mathrm{logit}\bigl(\text{Yes} \mid \text{"A: q, B: p, {prompt}"}\bigr)
$$

推理时对候选 top-K 每个 (q, p_i) 独立打分，按分数排序。

### 4 种 prompt

论文 Table 18 列出：

| 任务类型                | Prompt 模板                                                             | 典型任务                       |
| ----------------------- | ----------------------------------------------------------------------- | ------------------------------ |
| **query → passage**     | `Predict whether passage B contains an answer to query A.`               | MSMARCO / NQ / FEVER / TREC-COVID |
| **query → query**       | `Predict whether queries A and B are asking the same thing.`             | CQADupStack / QuoraRetrieval    |
| **passage → passage**   | `Predict whether passages A and B have the same meaning.`                | SciDocs / STS 类似任务          |
| **argument → counter-arg** | `Predict whether argument A and counterargument B express contradictory opinions.` | ArguAna 类辩论反驳检索         |

**关键设计**：训练时每个数据集用一条固定 prompt；推理时按任务类型选对应 prompt。**跟 embedding 端只 query 加指令不同，Reranker 是 (q, p) 联合编码，两侧都在 LLM 上下文里**。

---

## v2.5-gemma2-lightweight 的两级压缩

### 深度压缩：任意层都能出分

Gemma-2-9B 有 42 层。作者的核心 trick：**把 LM head 的 "Yes" prediction 线性层从最后一层「复制粘贴」到每一层**——每层都能独立算出 relevance 分数。

推理时用户可以选 `output_layer` ∈ [8, 42]：只跑到第 8 层就得到粗略分数、跑到第 42 层得到最精细分数。**精度 vs 延迟可无级调节**。

训练时：**自蒸馏（Self-KD）**。最后一层是 teacher，前面各层是 student，用 KL 散度对齐：

$$
\mathcal{L}_{\text{self-KD}}^{(\ell)} = \mathrm{KL}\bigl(P^{(42)}_{\text{Yes}} \,\big\|\, P^{(\ell)}_{\text{Yes}}\bigr)
$$

其中 $P^{(\ell)}_{\text{Yes}} = \mathrm{softmax}(\text{logits}^{(\ell)})[\text{Yes}]$。这样早期层学会「像最后一层一样评分」。

### 宽度压缩：token merging

在指定层（8/16/24/32/40）做 **token merging**：把 $n$ 个 token 合并成 1 个（论文实测支持 1/2/4/8 压缩比）。这类似 ViT 的 Token Merging (ToMe) 思路，但用在 LLM 语言 token 上。

合并规则**在训练时随机选**——每 step 从 4 种压缩比、5 个可选层里各选一个组合。这让模型学会「任意压缩策略下都能出好分」。

### 组合：60% FLOPs / 40% FLOPs saved

推理时用户选一种组合：

- **完整**（output_layer=42、compress=1）：100% FLOPs，最高精度
- **中档**（output_layer=25、compress=2、compress_layer=8）：约 **40% FLOPs 节省**（论文报的 "60% FLOPs" 是保留量）
- **激进**（output_layer=16、compress=4）：更多 FLOPs 节省，精度略降

Table 20/21 显示：40% FLOPs 节省后 BEIR Mean 从 63.67 掉到 63.10（**-0.57**）——损失极小；对高吞吐场景 (RAG 大规模精排) 是决定性收益。

---

## 训练配置

| 项              | 值                                                    |
| --------------- | ----------------------------------------------------- |
| 骨干            | Gemma-2-9B                                             |
| 训练方式        | LoRA rank=64、α=32                                    |
| Learning rate   | 1e-4                                                   |
| Loss            | 对比 InfoNCE + Self-KD (KL)                            |
| Batch           | 128                                                    |
| Hard neg / query| **15**（比 embedder 的 7 更多）                        |
| Prompt          | 训练时按数据集类型随机选 4 种之一                       |
| 数据            | BGE-M3 训练集 + Arguana / HotpotQA / FEVER            |
| 深度压缩        | 训练时**所有 layer 都出分**并计算 loss                  |
| 宽度压缩        | 训练时**随机采样**压缩比与压缩层                        |
| Optimizer       | AdamW                                                  |

**为什么 Reranker 用 15 hard neg 而 embedder 用 7**：

- Reranker 是**联合编码**，每对独立算 score；不吃 in-batch neg（除非做 listwise 蒸馏）。
- Embedder 靠 in-batch neg 撑负例池，hard neg 只是补充。
- 因此 Reranker 需要**每个 query 自带更多 hard neg**才能有效对比学习。

---

## 结果：BEIR & MIRACL 全面 SOTA

### BEIR (基于 bge-large-en-v1.5 top-100 重排)

| 数据集         | bge-large-en-v1.5 | v2-m3 | jina-reranker-v2 | v2-gemma | **v2.5-gemma2-lightweight (60% FLOPs)** | **v2.5-gemma2-lightweight (0% saved)** |
| -------------- | :---------------: | :---: | :--------------: | :------: | :-------------------------------------: | :------------------------------------: |
| ArguAna        | 63.54             | 37.70 | 52.23            | 78.68    | **86.04**                                | **86.16**                              |
| ClimateFEVER   | 36.49             | 37.99 | 34.65            | 39.07    | **48.41**                                | **48.48**                              |
| CQADupStack    | 42.23             | 38.24 | 40.21            | 45.85    | **49.18**                                | 48.90                                  |
| DBPedia        | 44.16             | 48.15 | 49.31            | 49.92    | 51.98                                    | **52.11**                              |
| FEVER          | 87.17             | 90.15 | 92.44            | 90.15    | **94.71**                                | 94.69                                  |
| FiQA2018       | 44.97             | 49.32 | 45.88            | 49.32    | 60.48                                    | **60.95**                              |
| HotpotQA       | 74.11             | 84.51 | 81.81            | 86.15    | 87.84                                    | **87.89**                              |
| MSMARCO        | 42.48             | 47.79 | 47.83            | 48.07    | 47.23                                    | 47.26                                  |
| NFCorpus       | 38.12             | 34.85 | 37.73            | 39.73    | 41.40                                    | **41.64**                              |
| NQ             | 55.04             | 69.37 | 67.35            | 72.60    | 75.37                                    | **75.58**                              |
| QuoraRetrieval | 89.06             | 89.13 | 87.81            | 90.37    | **91.25**                                | 91.18                                  |
| SCIDOCS        | 22.62             | 18.25 | 20.21            | 21.65    | 23.71                                    | **23.87**                              |
| SciFact        | 74.64             | 73.08 | 76.93            | 77.22    | **80.50**                                | 80.38                                  |
| Touche2020     | 25.08             | 35.68 | 32.45            | 35.68    | 30.64                                    | 31.09                                  |
| TREC-COVID     | 74.89             | 83.39 | 80.89            | 85.51    | 84.26                                    | 84.85                                  |
| **Mean**       | 54.31             | 55.36 | 56.52            | 60.71    | **63.10**                                | **63.67**                              |

**读法**：

- **v2.5-gemma2-lightweight 无压缩**：63.67，比 v2-gemma (60.71) **+2.96**，比 jina-reranker-v2 (56.52) **+7.15**。
- **60% FLOPs 版**：63.10，只掉 0.57。**这个"精度-成本"曲线是当前开源精排的甜点**。
- **ArguAna 断层领先**（86 vs 52-78）——argument→counter prompt 立功。
- 只有 Touche2020 略输 v2-gemma，其它 14 个数据集全部领先。

### BEIR（基于 E5-Mistral-7b top-100 重排）

同样 v2.5-gemma2-lightweight 领先：无压缩 64.04，60% FLOPs 63.36。**「更强的初检 + BGE-Reranker v2.5」组合能把 nDCG@10 从 56.85 拉到 64.04（+7.19）**。这一 7 分是 RAG 端到端质量最重要的可控收益之一。

### MIRACL 多语精排

论文 Table 22 未完整显示，但从摘录数字看：

- bge-m3 (Dense) 初检 + v2.5-gemma2-lightweight 重排：**MIRACL 所有 18 语平均分显著超过 v2-gemma 与 v2-m3**。
- Gemma-2 的 256k 多语 vocab 是关键。

---

## v2 家族其它版本要点速读

### bge-reranker-v2-m3（XLM-R 骨干，多语默认）

- **无 arXiv 单独论文**，作为 BGE-M3 的姊妹发布。
- Backbone: **XLM-R Large** (568M)，多语强。
- 输出：**Cross-Encoder 头**（linear + sigmoid），单头 score。
- 数据：BGE-M3 数据集。
- **BEIR Mean 55.36**——比 bge-large-en-v1.5 单塔（54.31）略高，但比 v2-gemma / v2.5 差。
- **优势**：轻量（568M vs 9B）、多语原生、部署简单。
- **短板**：单语精度不及 LLM 系。
- **典型用途**：多语 RAG 的默认精排；配 bge-m3 embedder 用起来最自然。

### bge-reranker-v2-gemma（Gemma-2B 骨干）

- Backbone: **Gemma-2B**（首代 Gemma，非 Gemma-2）。
- 输出：**LM head 取 "Yes" logit**，与 v2.5 同架构。
- 数据：主要英文数据 + 少量多语补充。
- **BEIR Mean 60.71**——首次证明 LLM-as-Reranker 相对 XLM-R CE 显著涨（+5.35）。
- **典型用途**：英文单语精排；对多语要求不高但要 LLM 精度。

### bge-reranker-v2-minicpm-layerwise（MiniCPM-2B，分层输出）

- Backbone: **MiniCPM-2B**（面壁智能）。
- **首创分层输出**：预先定义几个「出口层」（如 8/16/24/32/40），推理时可选。
- 训练时**每一出口都单独用 CE 训**（v2.5 的自蒸馏是这之上的改进）。
- **BEIR Mean 未在 BGE-EN-ICL 论文完整报**，但 HF card 显示与 v2-gemma 同档。
- **典型用途**：需要**层级式精度-延迟权衡**的场景，但精度上限不如 v2.5-gemma2。

### 演进逻辑

```text
v2-m3 (2024-02)：XLM-R Cross-Encoder，多语默认，BEIR 55.36
    │
    └─ v2-gemma (2024-04)：换 Gemma-2B + LM head Yes-logit，BEIR 60.71 (+5.35)
              │
              └─ v2-minicpm-layerwise (2024-04)：加分层输出，同精度、更灵活
                          │
                          └─ v2.5-gemma2-lightweight (2024-09)：换 Gemma-2-9B + 深度+宽度压缩 + Self-KD
                                    BEIR 63.67 (+2.96), 60% FLOPs 保 63.10
```

---

## 常见错误用法

1. **精排池 K 设太大或太小**：K 太大 → LLM Reranker 推理慢；K 太小 → recall 不够，精排也救不回来。**经验值 K=100（BEIR/MIRACL 标配）**；生产 RAG 可以调到 20–50。
2. **忘选对 prompt**：4 种 prompt 各有场景，用错会掉 1–3 分。**FAQ / 去重用 q→q**；**RAG 检索用 q→p**；**辩论/反驳用 arg→counter**。
3. **v2-m3 与 bge-large-en-v1.5 配对，v2.5-gemma2 与 E5-Mistral 配对**：初检强 → 精排更能发挥；初检弱（bge-large-en-v1.5 BEIR 54.31 vs E5-Mistral 56.85），精排后差距进一步放大。**能力级别要匹配**：小 embedder 配轻量 CE，LLM embedder 配 LLM Reranker。
4. **一律用 v2.5-gemma2-lightweight 无压缩**：9B LLM 每对 (q, p) 一次 forward，K=100 就是 100 次 9B 前向；生产环境要设 SLA。**默认从 60% FLOPs 起步**，精度掉不到关键决策阈值再放开。
5. **拿 v2.5-gemma2-lightweight 当嵌入用**：**不是 embedder**。它输出的是 score，不是向量；换用当嵌入需要重训（LM head 换成 pooling head）。
6. **蒸馏 Bi-encoder 时用 v2.5 直接给分数**：v2.5 输出是「Yes-logit」的**绝对值**，不是概率或排序位。蒸馏时应转成 softmax 概率或做 pairwise margin，否则学生学不好。BGE-EN-ICL 与 bge-multilingual-gemma2 就是这么蒸的。
7. **忘归一化 Reranker 分数**：v2.5-gemma2-lightweight 输出是 raw logit（可能范围 -20~+20）；直接和 embedder 的 cosine（-1~+1）加权融合会失衡。**先各自 minmax 或 softmax 再融合**。

---

## Cross-Encoder vs Late Interaction vs Bi-Encoder：Pipeline 中的位置

| 阶段     | 计算量          | 精度         | 索引形态           | 代表方法                                |
| -------- | --------------- | ------------ | ------------------ | --------------------------------------- |
| **召回 (Bi-Encoder)** | $O(N \cdot d)$（预建索引） | 中           | 向量 + ANN         | BGE-M3 / bge-large / bge-en-icl / mE5   |
| **精排 (Late Interaction)** | $O(K \cdot L_q \cdot L_d)$ | 高          | 多向量 + PLAID     | ColBERTv2 / ColPali / JinaColBERT       |
| **精排 (Cross-Encoder)** | $O(K \cdot (L_q + L_d))$（现算） | **最高**    | 无（只算候选）      | **BGE-Reranker v2/v2.5** / monoT5       |

**BGE-Reranker 定位**：**在 Bi-Encoder 召回 + Cross-Encoder 精排** 的两段式 pipeline 里担任第二段。Late Interaction（ColBERTv2）夹在中间，但工业上仍以「Bi + CE」两段式为主流——CE 的精度上限更高、且 K=100 的量级完全能承受。

见主文 §8「检索 Pipeline 工程实践」的三段式讨论。

---

## 与本仓库既有报告的挂接

- 姊妹版本：[BGE-EN-ICL 详解](EN-ICL/BGE-EN-ICL详解.md)（v2.5 是 EN-ICL 的蒸馏教师）· [bge-multilingual-gemma2 详解](BGE-multilingual-gemma2详解.md)（同样用 v2.5 蒸馏）
- 前置：[BGE-CPack 详解](C-Pack/BGE-CPack详解.md)（BGE 全家桶起点）· [BGE-M3 三功能统一详解报告](M3/BGE-M3三功能统一详解报告.md)（v2-m3 出处）
- Late Interaction 对照：[ColBERT 详解](../ColBERT/ColBERT详解.md) · [ColBERTv2 详解](../ColBERTv2/ColBERTv2详解.md) · [ColPali 详解](../ColPali/ColPali详解.md) · [ColQwen 系列详解](../ColQwen/ColQwen系列详解.md)
- 蒸馏配套：[Embedding 蒸馏技术详解](../Embedding蒸馏技术详解.md)（v2.5 的自蒸馏 + 教师蒸馏是核心信号）
- 训练损失：[对比学习与 InfoNCE 精讲](../对比学习与InfoNCE精讲.md) 的「蒸馏损失家族」章节
- 主文：[Embedding 调研报告](../Embedding调研报告.md) §3.3「三种交互架构」（Bi/Cross/Late 对比）与 §8.3「Reranker 放哪、选谁」

---

*本报告基于 BGE-EN-ICL 论文附录 D（arXiv 2409.15700）与 [FlagEmbedding 官方 rerankers 页](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_reranker) 整理。分数为 2024-09 BEIR / MIRACL；BGE-Reranker v2.5-gemma2-lightweight 是当前开源多语精排的事实 SOTA，同时兼顾工业延迟约束。*
