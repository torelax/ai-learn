# RocketQA 技术详解

> **paper**: [arXiv:2010.08191](https://arxiv.org/abs/2010.08191)（NAACL 2021；百度）  
> **v2**: [arXiv:2110.07367](https://arxiv.org/abs/2110.07367)（RocketQAv2：双塔 + 重排联合训练）  
> **code**: [PaddlePaddle/RocketQA](https://github.com/PaddlePaddle/RocketQA)  
> **local PDF**: `docs/papers/embedding/RocketQA_2010.08191.pdf` · `docs/papers/embedding/RocketQAv2_2110.07367.pdf`  
> **backbone**: 双塔 DE（ERNIE 2.0 base）+ 交叉编码器 CE（ERNIE 2.0 large）  
> **date**: 2020–2021  
> **modality**: 文本稠密检索（OpenQA / Passage Ranking）  
> **languages**: 英文主评测（MS MARCO / NQ）  

> 本文把 **三大训练挑战、cross-batch 负例、CE 去噪难负例、伪标数据增强、四级火箭式课表、MS MARCO/NQ 结果，以及 v2 的动态 listwise 蒸馏** 写全。假负例后续演进见 [NV-Retriever详解.md](NV-Retriever详解.md)；训中动态换负见 [Conan-embedding详解.md](Conan-embedding详解.md)（DHNM）；异步全库 ANN 见 [ANCE详解.md](ANCE详解.md)。

---

## 一句话定位

**RocketQA** 不是新骨干，而是 **把双塔稠密检索训「够难、够干净、够大」** 的四级课表：跨卡扩负例 → 训 CE → **CE 过滤假负**再挖 hard → CE 伪标无监督问再训双塔。核心洞察：MS MARCO 上人工抽查发现，检索 top 里约 **70%「未标注正例」其实是相关**——naive hard negative 噪声极大。

| 项 | 内容 |
|----|------|
| 目标 | 训好 DE 检索器（OpenQA 一阶段） |
| 三板斧 | **Cross-batch negatives** · **Denoised hard negatives** · **CE 数据增强** |
| 课表隐喻 | 多级火箭：双塔性能在 STEP 1→3→4 连续抬升 |
| 宣称（Table 2） | MS MARCO Dev MRR@10 **37.0**；NQ R@20/100 **82.7 / 88.5**（超 ANCE / DPR） |

谱系：

```text
DPR（BM25 hard + in-batch）
  → ANCE（全局 ANN 异步刷新）
  → RocketQA（CE 去噪 hard + 伪标增强）   ← 本文
  → RocketQAv2（DE↔CE 动态 listwise 联合训）
  → NV-Retriever（正例分锚定滤假负）
  → Conan DHNM（训中按难度换池内负例）
```

---

## 问题：为什么双塔难训

论文点出三点：

1. **训推鸿沟**：推理要在百万～千万 passage 里找正例；训练受单卡显存限制，负例池极小。  
2. **未标注正例（假负例）**：MS MARCO 平均每问仅 ~1.1 个标注正例、库 8.8M；人工核查 top 未标注段，约 **70% 实际相关**。从 top-k 直接当 hard neg 会把真相关当负例。  
3. **标注规模有限**：MS MARCO / NQ 仍难覆盖全话题；需要伪标扩数据。

CE（交叉编码）交互更强、更适合当「裁判」，但扫全库太贵 → 用 CE **去噪与伪标**，DE 负责检索。

---

## 训练数据与评测基准

### 有标注

| 数据集 | 用途 | 规模（论文 Table 1） |
|--------|------|----------------------|
| **MS MARCO Passage Ranking** | 主检索榜；Bing 日志问 | train ~503k 问；库 ~8.8M passage |
| **Natural Questions（DPR 版）** | OpenQA 检索 | train ~59k 问；库 ~21M Wikipedia 段 |

### 无标注增强（STEP 4）

| 来源 | 用法 |
|------|------|
| Yahoo! Answers / ORCAS / MRQA 等 | 收集无标问题；用 CE 对 DE 召回 top-k 打高置信正/负伪标 |

阈值：CE 分 **&lt;0.1 作负、&gt;0.9 作正**；人工抽查准确率 &gt;90%。

### 评测

| 基准 | 指标 | RocketQA（论文） | 对照 |
|------|------|------------------|------|
| MS MARCO Dev | MRR@10 / R@50 / R@1000 | **37.0** / 85.5 / 97.9 | ANCE 33.0 MRR@10；ME-BERT 33.8 |
| NQ Test | R@5 / R@20 / R@100 | 74.0 / **82.7** / **88.5** | ANCE — / 81.9 / 87.5；DPR — / 78.4 / 85.4 |

另报：换 RocketQA 检索后，端到端 QA（reader）EM 也升。

---

## 三大策略

### Cross-batch negatives

单卡 in-batch：batch 内 $B$ 个问，每问约 $B-1$ 个「他人正例」作负。  
多卡时 **all-gather passage 向量**：$A$ 卡可得约 $A\cdot B-1$ 个负例（≈ 单卡 in-batch 的 $A$ 倍），缩小训推负例规模差，几乎不增算力（复用已算 embedding）。

贯穿所有训双塔步骤。

### Denoised hard negatives

流程：

1. 用当前双塔 $M_D^{(0)}$ 召回 top 候选（去掉已标注正例）；  
2. **CE 高置信判负**才保留为 hard neg（滤掉「其实相关」的假负）；  
3. 用去噪后的 hard 再训双塔 $M_D^{(1)}$。

相对 ANCE「全局 ANN 即难」、DPR「BM25 top 当 hard」：**多了一道 CE 去噪**。这是后续 NV-Retriever「正例锚定」的直接前史——RocketQA 用 CE 分数阈值，NV 用正例相似度阈值。

### Data augmentation（CE 伪标）

对无标问题集 $Q_U$：双塔召回 → CE 打高置信正/负 → 与人工 $D_L$ 合并训 $M_D^{(2)}$。可视为 **CE→DE 蒸馏**（硬伪标）。

---

## 四级火箭课表（Figure 3）

| 步 | 动作 | 产出 |
|----|------|------|
| STEP 1 | 有标数据 + cross-batch 训双塔 | $M_D^{(0)}$ |
| STEP 2 | 用 $M_D^{(0)}$ top 作负、有标正，训 **CE** | $M_C$（适配 DE 召回分布） |
| STEP 3 | CE 去噪 hard + cross-batch 再训双塔 | $M_D^{(1)}$ |
| STEP 4 | CE 伪标无监督问 + 有标，再训双塔 | $M_D^{(2)}$（最终检索器） |

实现要点（论文）：

- DE：ERNIE-base；CE：ERNIE-large；点积 + Faiss `IndexFlatIP`  
- MS MARCO：DE batch $512\times 8$；CE 阈值 0.1 / 0.9；DE 三步 epoch 约 40 / 10 / 10  
- 问/段最大长 32 / 128  

---

## RocketQAv2（简述）

v1 是 **交替 / 分步**：先冻 CE 去噪与伪标，再抬 DE。  
**RocketQAv2**（[arXiv:2110.07367](https://arxiv.org/abs/2110.07367)）做 **DE 与 CE 联合训**：

1. **Dynamic listwise distillation**：候选列表上 DE / CE 各自 softmax 成分布，最小化 **KL**；CE 另加监督 CE loss；两边参数都更新（软标签，非 v1 硬伪标）。  
2. **Hybrid data augmentation**：混合未去噪 / 去噪正负，构造多样 listwise 实例。  
3. 常用 **v1 的 DE/CE 初始化**，再联合蒸馏。

推理仍是 retrieve-then-rerank 流水线；训练上第一次真正把检索与重排 **绑在同一 listwise 目标**里互抬。

ColBERTv2 等「去噪蒸馏」同属这一代思路；学生换成 late-interaction 即见 [ColBERTv2详解.md](ColBERTv2详解.md)。

---

## 与 ANCE / NV / Conan DHNM

| | RocketQA | ANCE | NV-Retriever | Conan DHNM |
|--|----------|------|--------------|------------|
| 难负来源 | DE top + **CE 滤假负** | 学生 ANN 全局 | 教师 top + **正例锚滤** | 预挖池 + **训中替换** |
| 何时挖 | 训前多步课表 | 异步训中刷新索引 | 多训前离线 | 训中按 $S$ 判定 |
| 假负处理 | CE 置信阈值 | 不做 | MarginPos / PercPos | 弱（宜叠 NV） |
| 额外增益 | CE 伪标扩数据 | 梯度范数论证 | MTEB 冲榜配方 | 负例「变易」后换新 |

工业闭环见 [难负例挖掘工业实践.md](难负例挖掘工业实践.md)：挖 → **滤（RocketQA/NV）** → 训 → 回归 → 刷新（ANCE）/ 池内换（DHNM）。

---

## 可迁移实践

1. **cloud_emb / 领域适配**：教师或 CE 挖 hard 后 **必须去噪**；不要直接吃 ANN top-k。  
2. **滤法选型**：有 CE 算力 → RocketQA 式阈值；只有双塔分 → NV PercPos / MarginPos。  
3. **跨卡负例**：多卡 `all_gather` 扩分母（与 Conan CBB「扩负」同族，CBB 还管多任务平衡）。  
4. **伪标**：仅保留极高置信正负；低置信丢弃，宁缺毋滥。  
5. **v2 何时上**：要同时训重排器且能吃 listwise 软标签时再上联合蒸馏；只训检索器走 v1 四级火箭即可。

---

## 同目录对照

| 文档 | 关系 |
|------|------|
| [ANCE详解.md](ANCE详解.md) | 全局 ANN；假负未滤 |
| [NV-Retriever详解.md](NV-Retriever详解.md) | 正例锚定滤假负（RocketQA 后继） |
| [Conan-embedding详解.md](Conan-embedding详解.md) | DHNM：训中动态换负 |
| [ColBERTv2详解.md](ColBERTv2详解.md) | 去噪蒸馏 → multi-vector 学生 |
| [难负例挖掘工业实践.md](难负例挖掘工业实践.md) | 挖滤训闭环 |

---

## 参考文献

1. Qu et al. (2021). RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering. NAACL. [arXiv:2010.08191](https://arxiv.org/abs/2010.08191)  
2. Ren et al. (2021). RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking. [arXiv:2110.07367](https://arxiv.org/abs/2110.07367)  
3. Xiong et al. (2021). ANCE. ICLR. [arXiv:2007.00808](https://arxiv.org/abs/2007.00808)  
4. Karpukhin et al. (2020). DPR. EMNLP.  
5. Moreira et al. (2024). NV-Retriever. [arXiv:2407.15831](https://arxiv.org/abs/2407.15831)  
