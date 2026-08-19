# bge-multilingual-gemma2：Gemma-2 骨干的多语嵌入

> **paper**：附录 C of [Making Text Embedders Few-Shot Learners (ICLR 2025)](https://arxiv.org/abs/2409.15700)（与 [BGE-EN-ICL 详解](BGE-EN-ICL详解.md) 同篇论文的姊妹发布）
> **code / model**：[FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) · [BAAI/bge-multilingual-gemma2](https://huggingface.co/BAAI/bge-multilingual-gemma2)
> **refs**：[BGE-EN-ICL (2024)](https://arxiv.org/abs/2409.15700) · [BGE-M3 (Chen 2024)](https://arxiv.org/abs/2402.03216) · [Gemma-2 (Team 2024)](https://arxiv.org/abs/2408.00118) · [MIRACL (Zhang 2023)](https://arxiv.org/abs/2210.09984) · [XLM-R (Conneau 2020)](https://arxiv.org/abs/1911.02116) · [mE5 (Wang 2024)](https://arxiv.org/abs/2402.05672)
> **backbone**：**Gemma-2-9B**（256k vocab，多语原生更好），causal attention + [EOS] pooling
> **date**：2024-09（与 BGE-EN-ICL / BGE-Reranker v2.5 同期）
> **modality**：文本
> **languages**：18 语核心（MIRACL 覆盖）+ 60+ 语被 Gemma-2 tokenizer 覆盖
>
> 本文写清 bge-multilingual-gemma2 是 BGE-EN-ICL 的**多语姊妹版**：**沿用同一训练框架，把 Mistral-7B 换成 Gemma-2-9B**，为的是那个 **256k 多语 vocab**。核心结论：**MIRACL 18 语 nDCG@10 = 74.1**（BGE-M3 Dense 69.2 + 4.9），**FR-MTEB / PL-MTEB / MIRACL 全部 SOTA**，MTEB 英文 69.88，唯一一个明显低于 gte-Qwen2 的场景是 C-MTEB（因为 Gemma-2 的中文能力弱于 Qwen2）。这份姊妹版说明：**LLM-Embedding 冲榜路线里，多语能力的决定性因素是 backbone tokenizer 与预训练语料的语言覆盖，而不是训练配方**。

---

## 一句话定位

bge-multilingual-gemma2 = **BGE-EN-ICL 配方 - ICL 训练 + Gemma-2-9B backbone**：

- 训练目标：同 BGE-EN-ICL 的 InfoNCE + hard neg + 蒸馏。
- 数据：英文（同 BGE-EN-ICL 的 public+full）+ 中文（BGE-M3 + Multi-CPR + 中文分类/聚类）+ 多语（MIRACL + Mr.TyDi）。
- Backbone：**Gemma-2-9B**（256k tokenizer，多语覆盖 60+ 语）。
- **暂时不加 ICL 训练**（future work）——作者明确说这是「初步尝试，ICL 探索留给未来」。

| 项              | 值                                                    |
| --------------- | ----------------------------------------------------- |
| Backbone        | **Gemma-2-9B**                                        |
| Tokenizer vocab | **256k**（比 Mistral 32k、Qwen2 152k、Llama-3 128k 都大） |
| 参数            | 9B                                                    |
| Attention       | causal（同 BGE-EN-ICL 保留原架构）                    |
| Pooling         | [EOS] token last hidden                              |
| LoRA rank / α   | 64 / 32                                                |
| Batch (Retrieval)| 512（7 hard neg + in-batch）                          |
| Batch (其它任务)| 256（7 hard neg，无 in-batch）                        |
| Max seq         | 512（无 ICL 示例，比 bge-en-icl 的 2048 短）           |
| 温度            | 0.02                                                   |
| 蒸馏教师        | bge-reranker（同期发布）                              |
| **MIRACL 18 语 nDCG@10** | **74.1**（全部 18 语 SOTA）                  |
| MTEB Avg (英文) | 69.88                                                 |
| FR-MTEB Avg     | **70.08**（SOTA）                                     |
| PL-MTEB Avg     | **70.00**（SOTA）                                     |
| C-MTEB Avg      | 68.44（略输 gte-Qwen2-7B 的 72.05）                  |

## 谱系与位置

```text
BGE-M3 (2024-02, XLM-R 骨干)  ──┐  BAAI 多语传统一条线
                                 │
                                 ├─→ bge-multilingual-gemma2 (2024-09, Gemma-2-9B)
                                 │           │
                                 │           ├── 训练框架：完全同 BGE-EN-ICL（无 ICL）
                                 │           └── 定位：把 BGE-EN-ICL 的英文强项挪到多语场景
                                 │
BGE-EN-ICL (2024-09, Mistral-7B) ─┘  BAAI 英文 ICL 一条线
```

**理解顺序**：先读 [BGE-EN-ICL 详解](BGE-EN-ICL详解.md)（拿到训练框架）→ 本篇（把英文换成多语场景）→ [BGE-M3 详解](BGE-M3三功能统一详解报告.md)（多语的另一条路：多头统一 + 长上下文）。

---

## 为什么选 Gemma-2 而不是 Qwen2 / Llama-3

作者在附录 C.1 明确说明选 Gemma-2 的唯一理由：**vocabulary size**。

| Backbone     | Vocab size | 多语原生表现（论文观察）  |
| ------------ | :--------: | ------------------------- |
| Mistral-7B   | 32k        | 英文强，其它语差            |
| Llama-3 8B    | 128k       | 英文优秀，中日韩较弱         |
| Qwen2-7B     | 152k       | **中英双强**，其它语一般     |
| **Gemma-2-9B** | **256k**  | **全面均衡**，60+ 语覆盖   |

XLM-R 的经典发现「更大的 vocabulary 显著改善多语性能」在 Gemma-2 上被再次验证。**256k vocab 让稀有语言的 token 不再被切成 3-4 个 sub-token**，语义完整性显著改善。这直接反映到 MIRACL 的低资源语（yo、sw、te、bn）分数上——bge-multilingual-gemma2 在这些语言上比 BGE-M3 Dense 高出 5–15 nDCG@10。

**权衡**：Gemma-2 的中文语料相对 Qwen2 少，所以 C-MTEB 分数低约 3.6 点；但**在 60+ 语的均衡表现上**是当前最好的开源选择。

---

## 训练数据组成

| 语种组          | 数据集                                                                 | 用途                       |
| --------------- | ---------------------------------------------------------------------- | -------------------------- |
| **英文**        | 除去 MSMARCO document ranking，与 BGE-EN-ICL full 版一致：            |                            |
|                 | ELI5 / HotpotQA / FEVER / MSMARCO passage / NQ / NLI / SQuAD / TriviaQA / Quora / Arguana / FiQA | Retrieval             |
|                 | SciDocsRR / StackOverFlowDupQuestions                                 | Reranking                  |
|                 | 8 个（AmazonReviews / Banking77 / Emotion / TweetSentiment / MTOP / IMDB / Toxic 等） | Classification |
|                 | Arxiv/Biorxiv/Medrxiv/Reddit/StackExchange × S2S/P2P + TwentyNewsgroups | Clustering                 |
|                 | STS12 / STS22 / STS-B                                                  | STS                         |
| **中文**        | 复用 BGE-M3 的中文数据，另加：                                          |                            |
|                 | Multi-CPR 三个 domain-specific 检索集                                   | Retrieval                  |
|                 | AmazonReviews-Cn + MultilingualSentiment                              | Classification              |
|                 | CSL-Clustering-S2S / P2P                                              | Clustering                 |
| **多语**        | **MIRACL**（18 语 Wikipedia QA）+ **Mr.TyDi**（11 语 QA）             | Retrieval（多语核心）      |

关键设计：**指令模板对多语数据也是英文**——例如 MIRACL 18 语用同一条 `Given a question, retrieve Wikipedia passages that answer the question.`。这让模型在**指令层面复用英文的语义能力**，在**内容层面处理任意语言**。

## 训练配置

| 项              | 值                                                    |
| --------------- | ----------------------------------------------------- |
| Backbone        | Gemma-2-9B                                             |
| 训练            | LoRA rank=64、α=32、lr=1e-4                            |
| Optimizer       | AdamW                                                  |
| 温度            | 0.02                                                   |
| Batch (Retrieval)| 512（7 hard neg + in-batch neg）                       |
| Batch (其它)    | 256（7 hard neg，无 in-batch）                        |
| Epochs          | 1                                                      |
| Max seq         | 512（q / p 各 512）                                    |
| 蒸馏教师        | bge-reranker（Retrieval 任务用）                      |
| 硬件            | 多语训练 GPU 需求略高于 bge-en-icl（Gemma-2-9B > Mistral-7B）|

**与 BGE-EN-ICL 的差异**：

1. **无 ICL 训练**：不加 few-shot 示例；作者留待未来做多语 ICL。
2. **Max seq 从 2048 降到 512**：因为没有示例段。
3. **数据规模更大**：多加了中文与多语数据（合并起来约多 100 万+）。
4. **Backbone 更大**：9B vs 7B。

---

## MIRACL：多语检索的核心验收

MIRACL 是 18 语 Wikipedia QA 数据集，是当前**多语检索的事实基准**。bge-multilingual-gemma2 在这里取得最重要的收益：

### nDCG@10

| 模型                        | Avg  | ar   | bn   | en   | es   | fa   | fi   | fr   | hi   | id   | ja   | ko   | ru   | sw   | te   | th   | zh   | de   | yo   |
| --------------------------- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| BM25                        | 31.9 | 39.5 | 48.2 | 26.7 | 7.7  | 28.7 | 45.8 | 11.5 | 35.0 | 29.7 | 31.2 | 37.1 | 25.6 | 35.1 | 38.3 | 49.1 | 17.5 | 12.0 | 56.1 |
| mDPR                        | 41.8 | 49.9 | 44.3 | 39.4 | 47.8 | 48.0 | 47.2 | 43.5 | 38.3 | 27.2 | 43.9 | 41.9 | 40.7 | 29.9 | 35.6 | 35.8 | 51.2 | 49.0 | 39.6 |
| mContriever                 | 43.1 | 52.5 | 50.1 | 36.4 | 41.8 | 21.5 | 60.2 | 31.4 | 28.6 | 39.2 | 42.4 | 48.3 | 39.1 | 56.0 | 52.8 | 51.7 | 41.0 | 40.8 | 41.5 |
| mE5-large                   | 66.6 | 76.0 | 75.9 | 52.9 | 52.9 | 59.0 | 77.8 | 54.5 | 62.0 | 52.9 | 70.6 | 66.5 | 67.4 | 74.9 | 84.6 | 80.2 | 56.0 | 56.4 | 78.3 |
| E5-Mistral-7B               | 63.4 | 73.3 | 70.3 | 57.3 | 52.2 | 52.1 | 74.7 | 55.2 | 52.1 | 52.7 | 66.8 | 61.8 | 67.7 | 68.4 | 73.9 | 74.0 | 54.0 | 54.1 | 79.7 |
| BGE-M3 (Dense)              | 69.2 | 78.4 | 80.0 | 56.9 | 56.1 | 60.9 | 78.6 | 58.3 | 59.5 | 56.1 | 72.8 | 69.9 | 70.1 | 78.7 | 86.2 | 82.6 | 62.7 | 56.7 | 81.8 |
| mGTE-TRM (Dense)            | 62.1 | 71.4 | 72.7 | 54.1 | 51.4 | 51.2 | 73.5 | 53.9 | 51.6 | 50.3 | 65.8 | 62.7 | 63.2 | 69.9 | 83.0 | 74.0 | 60.8 | 49.7 | 58.3 |
| **bge-multilingual-gemma2** | **74.1** | **81.0** | **82.3** | **64.5** | **64.2** | **64.0** | **81.2** | **64.2** | **68.2** | **61.5** | **79.1** | **69.7** | **77.0** | **81.9** | **88.1** | **84.6** | **68.0** | **63.5** | **90.3** |

**关键读法**：

- **18 语全部 SOTA**：没有一种语言输给其它开源模型。
- **相对 BGE-M3 Dense +4.9**：这是 XLM-R (0.56B) → Gemma-2 (9B) 的规模效应加上更好 tokenizer 的合力。
- **低资源语提升尤为显著**：Yoruba (yo) +8.5、Hindi (hi) +8.7、Swahili (sw) +3.2、Russian (ru) +6.9。
- **英文（en）从 56.9 → 64.5**：MIRACL 的英文题目本身有挑战性（比 BEIR NQ 更长），bge-multilingual-gemma2 在英文上也比 BGE-M3 大幅提升。

### Recall@100

| 模型                        | Avg  |
| --------------------------- | :--: |
| BM25                        | 67.3 |
| mE5-large                   | 94.1 |
| BGE-M3 (Dense)              | 95.5 |
| **bge-multilingual-gemma2** | **97.2** |

Recall@100 是**召回层核心指标**，bge-multilingual-gemma2 达到 97.2，意味着**97% 的正例都能被前 100 个候选覆盖**。作为 RAG / 搜索的召回层已经无短板。

---

## FR-MTEB / PL-MTEB / C-MTEB：单语深评

除了 MIRACL 这种"18 语跨语"，作者还在**单语场景**（法语 / 波兰语 / 中文）评估。

### FR-MTEB（法语 26 数据集）

| 模型                        | Retr | Rerank | Clust | PairClass | Class | STS | Summ | **Avg** |
| --------------------------- | :--: | :----: | :---: | :-------: | :---: | :-: | :--: | :-----: |
| mistral-embed               | 46.81 | 80.46 | 44.74 | 77.32 | 68.61 | 79.56 | 31.47 | 59.41 |
| gte-multilingual-base       | 52.97 | 76.47 | 41.66 | 79.46 | 68.72 | 81.36 | 29.74 | 59.79 |
| voyage-multilingual-2       | 54.56 | 82.59 | 46.57 | 78.66 | 68.56 | 80.13 | 29.96 | 61.65 |
| gte-Qwen2-1.5B-instruct     | 52.56 | 83.76 | 55.01 | 86.88 | 78.02 | 81.26 | 30.50 | 66.60 |
| gte-Qwen2-7B-instruct       | 55.65 | 78.70 | 55.56 | 90.43 | 81.76 | 82.31 | 31.45 | 68.25 |
| **bge-multilingual-gemma2** | **63.47** | **85.22** | **56.48** | 85.07 | 81.62 | **82.59** | 31.26 | **70.08** |

**Retr 63.47 是断崖式领先**（+7.82 vs gte-Qwen2-7B）——法语检索基本没对手。

### PL-MTEB（波兰语 26 数据集）

| 模型                        | Retr | Clust | PairClass | Class | STS | **Avg** |
| --------------------------- | :--: | :---: | :-------: | :---: | :-: | :-----: |
| gte-multilingual-base       | 46.40 | 33.67 | 85.45 | 60.15 | 68.92 | 58.22 |
| multilingual-e5-large       | 48.98 | 33.88 | 85.50 | 63.82 | 66.91 | 60.08 |
| mmlw-roberta-large          | 52.71 | 31.16 | 89.13 | 66.39 | 70.59 | 63.23 |
| gte-Qwen2-1.5B-instruct     | 51.88 | 44.59 | 84.87 | 72.29 | 68.12 | 64.04 |
| gte-Qwen2-7B-instruct       | 54.69 | 51.36 | 88.48 | 77.84 | 70.86 | 67.86 |
| **bge-multilingual-gemma2** | **59.41** | 50.29 | **89.62** | 77.99 | 70.64 | **70.00** |

同样 SOTA，Retr +4.72 vs gte-Qwen2-7B。

### C-MTEB（中文 35 数据集）

| 模型                        | Retr | Rerank | Clust | PairClass | Class | STS | **Avg** |
| --------------------------- | :--: | :----: | :---: | :-------: | :---: | :-: | :-----: |
| multilingual-e5-large       | 63.66 | 56.00 | 48.23 | 69.89 | 67.34 | 48.29 | 58.81 |
| e5-mistral-7b-instruct      | 61.75 | 61.86 | 52.30 | 72.19 | 70.17 | 50.22 | 60.81 |
| gte-multilingual-base       | 71.95 | 68.17 | 47.48 | 78.34 | 64.27 | 52.73 | 62.72 |
| bge-large-zh-v1.5           | 70.46 | 65.84 | 48.99 | 81.60 | 69.13 | 56.25 | 64.53 |
| gte-Qwen2-1.5B-instruct     | 71.86 | 68.21 | 54.61 | 86.91 | 71.12 | 60.96 | 67.65 |
| **gte-Qwen2-7B-instruct**   | **76.03** | **68.92** | **66.06** | **87.48** | **75.09** | **65.33** | **72.05** |
| bge-multilingual-gemma2     | 73.73 | 68.28 | 59.30 | 86.67 | 74.11 | 56.87 | 68.44 |

**唯一败给 gte-Qwen2-7B 的场景**：C-MTEB 总分 68.44 vs gte-Qwen2-7B 的 72.05。作者的坦诚解释：**Gemma-2 的中文预训练语料量级不如 Qwen2**——backbone 决定 ceiling。这也说明 **backbone 语言覆盖是选型的第一考量**。

**中文场景优先级建议**：中文场景选 gte-Qwen2 / Conan-v2 / QZhou-Embedding / bge-large-zh-v1.5；纯多语（含中文但不主打）选 bge-multilingual-gemma2 / BGE-M3。

---

## MTEB（英文 56 数据集）

作者在 MTEB 主表也报了 bge-multilingual-gemma2 的分数（69.88），仅次于 NV-Embed-v2 (72.31) / bge-en-icl (71.24) / Stella 1.5B (71.19) / SFR-2R (70.31) / gte-Qwen2 (70.24)。

值得注意：**bge-multilingual-gemma2 是所有 top-10 中唯一优先做多语的模型**——它在保住英文 top-10 的同时，多语能力全面 SOTA。这个「多语不牺牲英文」的定位是 Arctic-Embed 2.0 同期一直在追但没完全做到的目标。

**详细分数**（挑选 Retrieval 类，展示英文 top-1 数据集情况）：

| 数据集             | bge-multilingual-gemma2 | NV-Embed-v1 | gte-Qwen2-7B | SFR-2R | bge-en-icl few-shot |
| ------------------ | :---------------------: | :---------: | :----------: | :----: | :-----------------: |
| ArguAna            | 77.37                    | 68.21       | 64.27        | 62.34  | 83.08               |
| HotpotQA           | 83.26                    | 79.92       | 73.08        | 81.36  | 85.14               |
| FEVER              | 90.38                    | 87.77       | 95.11        | 92.16  | 92.83               |
| Natural Question   | 71.45                    | 71.22       | 67.00        | 73.96  | 73.88               |
| MSMARCO            | 45.71                    | 46.49       | 45.98        | 42.18  | 46.79               |
| TREC-COVID         | 64.27                    | 85.88       | 82.26        | 87.28  | 79.08               |

**观察**：Retrieval 上 bge-multilingual-gemma2 略弱于同期最强（NV-Embed-v2 / bge-en-icl），但**没有短板任务**——所有数据集都在 top-5 分位之内。

---

## 常见错误用法

1. **拿 bge-multilingual-gemma2 想跑 ICL few-shot**：**它没训 ICL**，加示例可能反而掉分（同 E5-Mistral / NV-Embed-v1 的模式）。要做多语 ICL 等未来版本，或用 bge-en-icl（仅英文有效）。
2. **在中文主导场景选 gemma2**：C-MTEB 68.44 < gte-Qwen2-7B 72.05；纯中文场景**优先 Qwen2 骨干**（gte-Qwen2 / QZhou / Conan-v2）。
3. **忘记指令**：与所有 BGE 系一样是**指令感知**——线上必须加对应任务的指令。MIRACL 场景用 `Given a question, retrieve Wikipedia passages that answer the question.`；AIR-Bench 用 `Given a question, retrieve passages that answer the question.`；MTEB 场景查论文 Table 7。
4. **只 query 加指令、doc 不加**：与 BGE-EN-ICL 一致的非对称约定——**passage 侧不加**。
5. **拿 512 max_seq 直接跑长文档**：max_seq=512 的训练意味着**长于 512 的输入会被截断**，长文档场景应先做 chunk（配合 [Late Chunking 详解]（Batch 4））或换 BGE-M3（支持 8K）。
6. **多语场景把不同语言的指令翻译过去**：论文实测「同一条英文指令用在 18 语」比「每种语言各译一条」更稳，因为**训练时英文指令 + 多语内容**是主导模式。翻译指令反而让模型看到未见过的组合。

---

## 与 BGE-M3 的选择对比

| 场景                          | 推荐                                             |
| ----------------------------- | ------------------------------------------------ |
| **短文档 / 一般检索 / 大 batch 服务** | bge-multilingual-gemma2                          |
| **长文档（>512 token）**        | BGE-M3（支持 8K；Dense + Sparse + Multi-vec 三头）|
| **需要稀疏检索 / 混合检索**     | BGE-M3（自带 sparse head）                       |
| **需要 late interaction / 多向量** | BGE-M3（自带 ColBERT-style head）                |
| **纯中文主导**                 | gte-Qwen2 / bge-large-zh-v1.5 / Conan-v2         |
| **纯英文主导 + 需要 ICL**       | bge-en-icl                                       |
| **多语 + 短文档 + 单向量**      | **bge-multilingual-gemma2**                      |

**核心结论**：bge-multilingual-gemma2 是**单向量密向量多语检索的当前首选**；BGE-M3 是**多头 + 长上下文场景**的选择。二者互补而非竞争。

---

## 与本仓库既有报告的挂接

- 姊妹版本：[BGE-EN-ICL 详解](BGE-EN-ICL详解.md)（训练框架完全同源，本篇是多语版）
- 前置：[BGE-CPack 详解](BGE-CPack详解.md)（BGE 全家桶起点）· [BGE-M3 三功能统一详解报告](BGE-M3三功能统一详解报告.md)（BAAI 多语传统）
- Backbone：Gemma-2 的 256k tokenizer 与其它多语骨干对比见 [无监督对比检索三部曲](无监督对比检索三部曲_SimCSE-Contriever-Condenser.md) 的 Contriever/mContriever 章节
- 配套精排：[BGE-Reranker 详解](BGE-Reranker详解.md)（Batch 3-4 已写）
- 对照多语 SOTA：[LLM-Embedding 冲榜路线](LLM-Embedding冲榜路线_E5Mistral-NVEmbed-GritLM-SFR-Arctic-Stella.md) 里的 Arctic-Embed 2.0（同期多语路线的对手）
- 训练损失：[对比学习与 InfoNCE 精讲](对比学习与InfoNCE精讲.md)
- 主文：[Embedding 调研报告](Embedding调研报告.md) §9.2「弱监督课表 + 指令」与 §12「向量数据库与部署」（多语落地考虑）

---

*本报告基于 BGE-EN-ICL 论文（arXiv 2409.15700 附录 C）与 [BAAI/bge-multilingual-gemma2 HF card](https://huggingface.co/BAAI/bge-multilingual-gemma2) 整理。分数为 2024-09 榜单；bge-multilingual-gemma2 是目前**多语单向量密向量检索的开源 SOTA**（尤其在 MIRACL / FR-MTEB / PL-MTEB 上）。*
