# ANCE 技术详解

> paper: [arXiv:2007.00808](https://arxiv.org/abs/2007.00808)（ICLR 2021）
> code: [https://aka.ms/ance](https://aka.ms/ance)
> local PDF: `docs/papers/embedding/ANCE_2007.00808.pdf`
> 中译: `docs/papers/embedding/ANCE_2007.00808_zh.md`
> backbone: RoBERTa / BERT Siamese（双塔共享）
> date: 2020–2021
> modality: 文本检索（web / OpenQA）
> languages: 英文主评测

> 本文把 **局部负例为何失效的方差分析、全局 ANN 难负例、异步索引刷新、与 DPR / BM25 对照、工业刷新节奏** 写全。工程闭环另见《[难负例挖掘工业实践](难负例挖掘工业实践.md)》；假负例过滤的「正例锚定」见《[NV-Retriever详解](NV-Retriever详解.md)》。

---

## 一句话定位

**ANCE**（**A**pproximate nearest neighbor **N**egative **C**ontrastive **E**stimation）用**正在训练的双塔模型**在全库 ANN 上挖难负例，并用**异步 Inferencer** 周期性刷新索引，解决 in-batch / BM25 负例「太易、与测试分布错位」导致的梯度塌缩。


| 项       | 内容                                                                       |
| ---------- | ---------------------------------------------------------------------------- |
| 问题     | 一阶段稠密检索要把相关文档从**整库**无关文档中分开                         |
| 诊断     | 局部负例梯度范数趋近 0 → 随机梯度方差大 → 收敛慢                         |
| 解法     | $D^-_{\mathrm{ANCE}}=\mathrm{ANN}_f(q)\setminus D^+$，异步刷新             |
| 宣称效果 | TREC DL / OpenQA 上明显超 DPR 式 BM25+随机负例；商业检索相对增益约 14–18% |

谱系位置：

```text
BM25 / 随机 / in-batch 负例
        → DPR（BM25 hard + in-batch）
        → ANCE（全局 ANN + 异步刷新）   ← 本文
        → RocketQA / ColBERTv2（CE 去噪 / 蒸馏）
        → NV-Retriever（正例感知过滤假负）
        → Conan DHNM（训中动态替换）
```

---

## 问题背景

稠密检索（Dense Retrieval）把 query / doc 编成向量，用点积或余弦做 ANN：

$$
f(q,d)=\mathrm{sim}\big(g(q;\theta),\,g(d;\theta)\big)

$$

一阶段目标是从语料 $C$ 中找回 $D^+$。训练形式上是学习排序：

$$
\theta^*=\arg\min_\theta\sum_q\sum_{d^+\in D^+}\sum_{d^-\in D^-}l\big(f(q,d^+),f(q,d^-)\big)

$$

但 $D^-=C\setminus D^+$ 可达百万～十亿，必须采样。常见采样：


| 来源           | 优点             | 缺陷                    |
| ---------------- | ------------------ | ------------------------- |
| BM25 top       | 便宜、像稀疏检索 | 易把 DR 训成「学 BM25」 |
| in-batch / NCE | 复用 forward     | 相对$q$ 往往太易        |
| 随机全库       | 无偏             | 几乎无信息              |

ANCE 主张：**训练负例分布应逼近测试时「模型最容易排错」的无关文档**。

---

## 收敛分析：为何局部负例不够

### 方差与梯度范数

对负例做重要性采样的一步 SGD：

$$
\theta_{t+1}=\theta_t-\eta\frac{1}{N p_{d^-}}\nabla_{\theta_t}l(d^+,d^-)

$$

收敛速度与梯度估计方差相关。最优采样近似满足：

$$
p^*_{d^-}\propto\big\|\nabla_{\theta_t}l(d^+,d^-)\big\|_2

$$

即：**梯度范数大的负例更值得采**。

对常见 BCE / hinge，有：

$$
l(d^+,d^-)\to 0 \;\Rightarrow\; \big\|\nabla_{\phi_L}l\big\|_2\to 0 \;\Rightarrow\; \big\|\nabla_\theta l\big\|_2\to 0

$$

「已经分得很开」的易负例几乎不贡献更新。

### 局部 batch 的概率论证

记 $D^{-*}$ 为真正有信息的难负例集合，$b$ 为 batch size。检索常见：

1. $b\ll|C|$
2. $|D^{-*}|\ll|C|$

则随机 mini-batch 撞上难负例的概率极低。这解释了为何视觉 / 词向量里好用的 in-batch hard 在 DR 上提升有限。

---

## 方法：全局 ANN 难负例

### 目标

$$
\theta^*=\arg\min_\theta\sum_q\sum_{d^+}\sum_{d^-\in D^-_{\mathrm{ANCE}}}l\big(f(q,d^+),f(q,d^-)\big)

$$

$$
D^-_{\mathrm{ANCE}}=\mathrm{ANN}_{f}(q)\setminus D^+

$$

实现上常用：BERT / RoBERTa **共享双塔**、点积、NLL / InfoNCE。

### 异步索引刷新（工程核心）

每步更新 $\theta$ 后若立刻全库重编码，Inference 成本不可承受。ANCE 拆成：

```text
Trainer ──用 ANN_{f_{k-1}} 挖负例──▶ 继续 SGD
                ▲
Inferencer ──用 checkpoint f_k 全库编码──▶ 重建 ANN_{f_k} ──┐
                └──────── 完成后交给 Trainer ◀───────────────┘
```

要点：

- **索引滞后**：负例来自「稍旧」的学生，而非当前 batch 权重。
- **资源配比**：文中 1:1 Trainer:Inferencer GPU 可接受；附录表明刷新过稀会伤效果。
- **实现数字（原文）**：约每 10k batch 刷新；每正例从 ANN top-200 **均匀采 1 个负例**；Faiss `IndexFlatIP`。

文档过长时：FirstP（前 512 token）或 MaxP（分段 max-pool，ANN 原生支持）。

---

## 实验要点

### TREC 2019 Deep Learning

相对 Rand / NCE / BM25 / DPR（BM25+Rand）负例，ANCE（FirstP / MaxP）在 passage / document **检索** NDCG@10 全面领先；文档检索上是少数能**稳定超过稀疏基线**的 BERT-Siamese 配方。常先用 BM25 负例 warm-up（`BM25 → ANCE`）。

### OpenQA（NQ / TriviaQA，DPR 设定）


| Retriever | NQ Top-20/100   | TQA Top-20/100  |
| ----------- | ----------------- | ----------------- |
| BM25      | 59.1 / 73.7     | 66.9 / 76.7     |
| DPR       | 78.4 / 85.4     | 79.4 / 85.0     |
| ANCE      | **81.9 / 87.5** | **80.3 / 85.3** |

同 reader（RAG-Token / DPR Reader）换 ANCE 检索，答案准确率也升——说明增益传到下游。

### 商业搜索

生产 DR 仅改训练负例为 ANCE：语料 2.5 亿～80 亿、768-d / 64-d、KNN / ANN，相对增益约 **+14%～+18%**。

### 与理论对照的经验验证

论文报告：ANCE 负例上的 **梯度范数显著大于** 局部负例，训练收敛更快——与 §3 分析一致。

---

## 局限与后续工作位置


| 局限       | 说明                                                                    |
| ------------ | ------------------------------------------------------------------------- |
| 假负例     | 全库 top 难负例里常混「其实相关但未标注」；ANCE**不**做正例锚定过滤     |
| 算力       | Inferencer 全库编码贵；大库需 ANN 近似 + 刷新调度                       |
| 索引陈旧   | 滞后过大 → 挖到的已不再难；过勤 → 吞吐崩                              |
| 实现复杂度 | 比「训前挖一次」重；故后续多数 MTEB 冲榜模型改用**强教师离线挖 + 过滤** |

后续：

- **RocketQA**：CE 去噪假负例
- **NV-Retriever**：TopK-MarginPos / PercPos，正例分作锚
- **Conan DHNM**：训练中按分数差动态换池内负例
- 工业默认：见《难负例挖掘工业实践》刷新节奏与每 query 2–4 干净负例

---

## 对本仓库的可迁移实践

1. **cloud_emb / 领域适配**：Stage2 难负例应用「当前或 lagged 学生 ANN」或强教师挖，而不是永久冻结 BM25 top。
2. **刷新**：大数据按 step（ANCE 风格）；中小库可每 epoch 全量重挖。
3. **必接假负过滤**：ANCE 挖完后加 NV-Retriever 式 `score_neg < score_pos - margin` 或 `score_neg < α·score_pos`。
4. **流式落盘**：挖负例脚本边写 jsonl + `--resume`（modelforge 约定）。

---

## 同目录对照


| 文档                                                   | 关系                            |
| -------------------------------------------------------- | --------------------------------- |
| [难负例挖掘工业实践.md](难负例挖掘工业实践.md)         | 挖→滤→训→回归→刷新闭环      |
| [NV-Retriever详解.md](NV-Retriever详解.md)             | 正例感知去假负；ANCE 的直接后继 |
| [Conan-embedding-v2详解.md](Conan-embedding-v2详解.md) | 动态池内替换                    |
| [E5详解.md](E5详解.md)                                 | 对比学习与监督微调公共祖先      |
| [ColBERTv2详解.md](ColBERTv2详解.md)                   | 难负例 + CE 蒸馏另一路线        |

---

## 参考文献

1. Xiong et al. (2021). Approximate Nearest Neighbor Negative Contrastive Learning for Dense Retrieval. ICLR 2021. [arXiv:2007.00808](https://arxiv.org/abs/2007.00808)
2. Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.
3. Moreira et al. (2024). NV-Retriever. [arXiv:2407.15831](https://arxiv.org/abs/2407.15831)
