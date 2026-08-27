> 原文: [arXiv:2108.05540](https://arxiv.org/abs/2108.05540)（ACL 2022）
> local PDF: `docs/papers/embedding/coCondenser_2108.05540.pdf`
> 说明: 本文为论文全文中文技术展开，公式编号与原文一致；图片自 arXiv HTML 抽取，caption 中译；表格数值保留原始数字，仅表头/说明中译。

**预印本信息：** arXiv:2108.05540v1 [cs.IR]，2021 年 8 月 12 日；会议版本：ACL 2022。

**代码：** https://github.com/luyug/Condenser

---

# 面向稠密段落检索的无监督语料感知语言模型预训练（Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval）

**作者：** Luyu Gao、Jamie Callan

**单位：** 卡耐基梅隆大学语言技术研究所（Language Technologies Institute, Carnegie Mellon University）

**邮箱：** {luyug, callan}@cs.cmu.edu

---

## 摘要（Abstract）

近期研究表明，微调好的语言模型（LM）可以做出出色的稠密检索器。**但**它们训练很难，需要精心设计的微调 pipeline 才能发挥全部潜力。本文识别并解决稠密检索器的两个根本问题：

1. **对训练数据噪声敏感**（fragility to training data noise）；
2. **需要极大的 batch 才能稳定地学到嵌入空间**。

作者基于自己之前提出的 **Condenser** 预训练架构，进一步提出 **coCondenser**：在 Condenser 之上引入一个**无监督的语料级对比损失（corpus-level contrastive loss）**——在 pretrain 阶段就把段落嵌入空间"预热"到位。

在 MS-MARCO、Natural Questions 与 Trivia QA 上，coCondenser 展示了三个显著优势：

- 不再需要大量数据工程（数据增广、伪标注、噪声过滤）；
- 不再需要大 batch 训练；
- 仅用 **batch size 64** 的简单微调，就与 **RocketQA**（batch 4096 + hard neg denoise + 数据增广的复杂 pipeline）**相当或更好**。

代码开源，可运行在 4 张 RTX 2080Ti 上。

---

## 1 引言（Introduction）

**背景**。稠密检索（dense retrieval）已成为文本检索的一种有效范式（Lee et al., 2019; Chang et al., 2020; Karpukhin et al., 2020; Qu et al., 2021）。但要让 bi-encoder 稠密检索器发挥能力，通常要一整套精心设计的微调技巧：迭代负例挖掘（Xiong et al., 2021）、多向量表示（Luan et al., 2020）、cross-encoder 蒸馏（Lin et al., 2020）等。

**RocketQA 的 pipeline**（Qu et al., 2021）代表了当时的巅峰：i) **denoising hard negative**——用 cross-encoder 检查挖出来的 hard neg，剔除被误标为负的正例；ii) **超大 batch**（4096–8192）；iii) **数据增广** ——用 cross-encoder 给外部数据打伪标。这条 pipeline 效果强，但**计算与工程成本极高**——远超学术研究常见的 4× 商用 GPU 预算。

**本文问题**：能不能把 RocketQA 的洞察**前移到 LM 预训练**里，使得下游微调时不再需要这套 pipeline？

作者的分析：

- RocketQA 里"去噪 hard neg" 的意义是——典型 LM 对训练数据的错标很敏感，会往错误方向大幅更新权重。若模型天然对噪声鲁棒，去噪就不必要。
- RocketQA 大 batch 的意义是——CLS 向量在 BERT 类模型里根本没被显式训练过（Devlin et al., 2019 只有 NSP，且只训 CLS 一件事；Liu et al., 2019 干脆不训 CLS）。这些 CLS 向量距离"能构成段落嵌入空间"还很远（Lee et al., 2019）。大 batch 帮助模型**稳定地**学出整个嵌入空间。

**思路**：借用 Condenser 架构（Gao & Callan, 2021）——它通过预训练让 CLS 主动做信息聚合，因此对局部 mask 噪声天然鲁棒。在此基础上再叠加**语料级对比损失**：给定目标语料（比如 Wikipedia 或 MS-MARCO 网页集合），每步采样 batch 内一批文档，各文档抽两个 span，让**同文档 span 的 CLS 靠近、跨文档 span 远离**。这个对比信号是**无监督**的、不依赖 query，就能预热整个语料的段落嵌入空间。

**贡献**：

1. 提出 coCondenser：Condenser + 无监督语料级对比预训练；
2. 用 **gradient cache** 技术让小显存下的大 batch 对比学习成为可能；
3. 在 MS-MARCO / NQ / TQA 上取得与 RocketQA 相当或更好的结果，微调仅需 batch 64、无 CE 去噪、无数据增广。

---

## 2 相关工作（Related Work）

**稠密检索**。Transformer LM 显著提升了 NLP 各任务，包括稠密检索。Lee et al. (2019) 用 BERT 训出第一个可用的稠密检索器；Chang et al. (2020) 系统研究检索预训练任务；Guu et al. (2020, REALM）端到端训 retriever + reader；Karpukhin et al. (2020, DPR）证明 careful 微调足以从 BERT 训出强 retriever；Xiong et al. (2021, ANCE）与 Qu et al. (2021, RocketQA）进一步改进微调。**并发工作 DPR-PAQ**（O˘guz et al., 2021）使用 6500 万条合成 QA 对做**半监督**的域内预训练，与本文的**完全无监督**做法形成对比。

作者本人先前的 Condenser（Gao & Callan, 2021）是本工作的直接基础——设计一个新的预训练**架构**（而非任务）。

**对比学习**。视觉里对比学习（Chen et al., 2020; He et al., 2020）成绩斐然；NLP 里 SimCSE（Gao et al., 2021）与 DeCLUTR（Giorgi et al., 2020）等把它用于句向量。本文的不同点：不是学**单点的表示**，而是**整个嵌入空间的几何结构**，用来 warm-start 检索器。

**大 batch 与内存**。对比学习吃 batch，稠密检索预训练同样如此（Guu et al., 2020; Chang et al., 2020）。Gao et al. (2021b) 提出 **gradient cache**，把内存峰值压到与 batch 大小几乎无关。§3.3 会说明 coCondenser 如何用它训 batch 2000 的对比预训练。

---

## 3 方法（Method）

### 3.1 Condenser 预备（Condenser Preliminary）

coCondenser 基于 Condenser（Gao & Callan, 2021）。Condenser 把 Transformer 分成三段（图 1 展示 2+2 层的简化版；作者实验用 6 early + 6 late + 2 head）：

![图 1：Condenser 架构。分为 early backbone、late backbone、Condenser Head 三段；Head 在预训练时存在，微调前丢弃](figs/fig01.png)

**图 1（原文对应图 1）：** Condenser 三段式架构。虚线内是 Condenser Head——只在预训练时存在，微调时丢弃，剩下的 backbone 与 BERT 结构一致。

- 输入加 CLS 拼接后过嵌入层：

$$
[h_0^{\text{cls}};\; h_0] = \operatorname{Embed}([\text{CLS};\; x]) \tag{1}
$$

- Early backbone 输出：

$$
[h_{\text{cls}}^{\text{early}};\; h^{\text{early}}] = \operatorname{Encoder}_{\text{early}}([h_0^{\text{cls}};\; h_0]) \tag{2}
$$

- Late backbone 输出：

$$
[h_{\text{cls}}^{\text{late}};\; h^{\text{late}}] = \operatorname{Encoder}_{\text{late}}([h_{\text{cls}}^{\text{early}};\; h^{\text{early}}]) \tag{3}
$$

- Head 输入是**晚期 CLS + 早期 token** 的拼接，通过短路把 token 表征从 early 层直接送进 head：

$$
[h_{\text{cls}}^{\text{cd}};\; h^{\text{cd}}] = \operatorname{Head}([h_{\text{cls}}^{\text{late}};\; h^{\text{early}}]) \tag{4}
$$

- Head 输出做 MLM 预测：

$$
\mathcal{L}_{\text{mlm}} = \sum_{i \in \text{masked}} \operatorname{CE}\bigl(W\, h_i^{\text{cd}},\; x_i\bigr) \tag{5}
$$

关键：Head 只能通过 $h_{\text{cls}}^{\text{late}}$ 拿到 late 层的新信息——迫使 late CLS 学到"把整段话浓缩到一个向量"的能力。

**Condenser 已经解决了什么**：CLS 学会稠密聚合，模型对 mask 位置噪声/局部错误鲁棒。**没有解决**：CLS 之间的**几何关系**（内积/相似度）——CLS 的目标只是"能让 head 预测出 mask token"，两个 CLS 的距离并没有语义。这是 coCondenser 要补的一步。

### 3.2 coCondenser：加语料级对比损失

**目标**：让同文档的 CLS 相似、跨文档的 CLS 相远。这样微调时嵌入空间已经有合理的**先验几何**，query 只需 fine-tune 一个小偏移。

**做法**：给一个 batch $n$ 篇文档 $[d_1, d_2, \dots, d_n]$，从每篇独立采样两段 span：

$$
[s_{11}, s_{12},\; s_{21}, s_{22},\; \dots,\; s_{n1}, s_{n2}]
$$

各 span 过 Condenser 后取 late CLS $h_{ij}$。对每一段 span，正对是**同文档的另一段**，负对是所有其它文档的两段 span：

$$
\mathcal{L}^{\text{co}}_{ij} = -\log \frac{\exp(\langle h_{ij},\; h_{i,\,3-j}\rangle)}{\displaystyle\sum_{(k, l) \neq (i, j)} \exp(\langle h_{ij},\; h_{kl}\rangle)} \tag{6}
$$

（$(k, l) \neq (i, j)$ 表示所有其它 span，包括同文档的另一段作分母中的正对，但分子只对齐正对；论文分母含所有非自身 span。）

**总损失**：把 MLM 与语料级对比合起来：

$$
\mathcal{L} = \frac{1}{2n}\bigl[\mathcal{L}^{\text{co}} + \sum_{i, j} \mathcal{L}^{\text{mlm}}_{ij}\bigr] \tag{7}
$$

其中 $\mathcal{L}^{\text{co}} = \sum_{i, j} \mathcal{L}^{\text{co}}_{ij}$。作者取 $n$ 为几百到几千，batch 里合计 $2n$ 个 span。

### 3.3 Gradient Cache：小卡机也能做大 batch 对比

对比学习需要大 batch 才能有足够负例。coCondenser 每步用 2000 篇文档 × 2 段 span = **4000 个 span**——常规做法根本装不下。作者借用 Gao et al. (2021b) 的 **gradient cache** 技术：

**核心思想**：把"CLS 表征梯度"与"encoder 参数梯度"两个计算解耦。分两个阶段：

**阶段 A：无梯度前向**。对整个 batch 做一次前向，只计算 CLS 表征 $\{h_{ij}\}$——**不构造计算图**、显存开销小：

$$
\text{无梯度：} h_{ij} = \operatorname{Encoder}(\text{span}_{ij})
$$

**阶段 B：对 CLS 求导**。用整 batch 的 $\{h_{ij}\}$ 算 $\mathcal{L}^{\text{co}}$ 和它对每个 CLS 的梯度：

$$
v_{ij} = \frac{\partial \mathcal{L}^{\text{co}}}{\partial h_{ij}} \tag{9}
$$

把 $\{v_{ij}\}$ 存进 **gradient cache** $C = [v_{11}, v_{12}, \dots, v_{n1}, v_{n2}]$。

**阶段 C：分块反向**。把 batch 切成小 chunk，逐 chunk 重新前向、构造计算图，然后：

$$
\frac{\partial \mathcal{L}^{\text{co}}}{\partial \Theta} = \sum_{i, j} v_{ij}^\top \frac{\partial h_{ij}}{\partial \Theta} \tag{11}
$$

$$
\frac{\partial \mathcal{L}}{\partial \Theta} = \frac{1}{2n}\sum_{i, j} \left[v_{ij}^\top \frac{\partial h_{ij}}{\partial \Theta} + \frac{\partial \mathcal{L}^{\text{mlm}}_{ij}}{\partial \Theta}\right] \tag{12}
$$

每个 chunk 只计算与 chunk 内 span 有关的项，把梯度累加。**总显存 ≈ 一个 chunk 的量**，但等效梯度 = 全 batch 训练。

作者用 4× RTX 2080Ti（每卡 11GB 显存）做到 batch 2000 的 coCondenser 预训练。

### 3.4 微调（Fine-tuning）

预训练结束后，丢弃 Condenser Head，剩下的 backbone 与 BERT 结构一致。用它初始化**查询编码器** $f_q$ 与**段落编码器** $f_p$（DPR 双塔），各取最后一层 CLS 作向量：

$$
s(q, p) = \langle f_q(q),\; f_p(p)\rangle \tag{13}
$$

监督微调用对比学习：给 query $q$ 与正例 $d^+$、若干负例 $\{d_l^-\}$：

$$
\mathcal{L} = -\log \frac{\exp(s(q, d^+))}{\exp(s(q, d^+)) + \sum_l \exp(s(q, d_l^-))} \tag{14}
$$

**两轮训练**（DPR 风格）：

1. **第一轮**：用 BM25 negatives 训一个初始 retriever；
2. **用第一轮 retriever 挖 hard negatives**，扩充负例池；
3. **第二轮**：用扩充后的负例池训第二轮 retriever。

这与 RocketQA 的复杂多阶段 pipeline（Cross-batch → HN → Denoise → Data Aug）形成鲜明对比。

### 3.5 与 RocketQA 训练流水线的对比

- **RocketQA pipeline**（Qu et al., 2021）：
  - Cross-batch negatives (batch 8192)
  - + Hard negatives (batch 4096)
  - + CE denoising（去假负例）
  - + 外部数据增广（CE 打伪标）
- **coCondenser pipeline**：
  - 用 coCondenser 初始化
  - Round 1：BM25 neg，batch 64
  - Round 2：Round 1 挖的 hard neg，batch 64

两条 pipeline 的直观比较（原文 Figure 2）：RocketQA 是"复杂微调 + 大 batch"，coCondenser 是"复杂预训练 + 简单微调"。coCondenser 把工作前移到了 **无监督预训练**——从而**无需外部 QA 数据、无需 CE 去噪、无需大 batch**。

---

## 4 实验（Experiments）

### 4.1 预训练细节（Pre-training）

**两阶段**：

**阶段 1：通用 Condenser 预训练**。从 BERT-base 初始化 backbone（6 early + 6 late），随机初始化 head（2 层）。数据用 BERT 原始的 English Wikipedia + BookCorpus。这一步的产物是通用 Condenser。

**阶段 2：语料感知的 coCondenser 预训练**。用阶段 1 的 Condenser（backbone + head 全部）warm-start，在**目标语料**（Wikipedia 或 MS-MARCO 网页集合）上继续训，加上语料级对比损失 $\mathcal{L}^{\text{co}}$。

**优化**：AdamW，lr $10^{-4}$，wd 0.01，线性学习率衰减。每步 2000 篇文档。4× RTX 2080Ti + gradient cache。

**为什么两阶段而非一步到位**？作者的解释：语料级对比 loss 在早期训练时梯度很大且**容易崩坏**（尚未收敛的 CLS 之间的相似度无意义，随机的对比 loss 会推 encoder 到错方向）；先做通用 Condenser 让 CLS 学会聚合，再上对比 loss，训练更稳。

预训练结束后，丢弃 head，得到与 BERT-base 完全同架构的 encoder。

### 4.2 稠密段落检索（Dense Passage Retrieval）

**数据**：

- **MS-MARCO Passage**：Bing 查询 + 网页段落，约 50 万训练查询；主指标 MRR@10、Recall@1000。
- **Natural Questions（NQ）**：Google 查询 + Wikipedia 段落；DPR 预处理版本；约 6 万训练查询；R@5/20/100。
- **Trivia QA（TQA）**：琐事问题；DPR 版本；R@5/20/100。

**训练细节**：

- MS-MARCO：AdamW，lr 5e-6，batch 64，3 epoch。**只在本任务数据上训**——RocketQA 训练时用了多集拼接，coCondenser 没有。
- NQ / TQA：用 DPR 官方超参数与工具箱；加 gradient cache 处理显存约束。
- 所有模型在**单张 RTX 2080Ti** 上训练。
- 验证：由于稠密检索验证需要重编码整个语料，成本高，作者遵循 DPR 建议**只用最后一个 checkpoint** 评估。

**基线**：
- 稀疏系：BM25、DeepCT、DocT5Query、GAR（用 BART 做 query 扩展的深度稀疏系）。
- 稠密系：DPR、ANCE、ME-BERT、RocketQA。
- 并发工作 **DPR-PAQ**（O˘guz et al., 2021）：用 6500 万合成 QA 对做**半监督**域预训练。作者比较 4 个 DPR-PAQ 变体（BERT/RoBERTa × base/large）。
- 作者还比较**只用阶段 1** 的 Condenser。

**表 1：三个数据集上的主要结果**

| 方法 | MS-MARCO MRR@10 | R@1000 | NQ R@5 | R@20 | R@100 | TQA R@5 | R@20 | R@100 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 18.7 | 85.7 | — | 59.1 | 73.7 | — | 66.9 | 76.7 |
| DeepCT | 24.3 | 90.9 | — | — | — | — | — | — |
| docT5query | 27.7 | 94.7 | — | — | — | — | — | — |
| GAR | — | — | 60.9 | 74.4 | 85.3 | 73.1 | 80.4 | 85.7 |
| DPR | — | — | — | 74.4 | 85.3 | — | 79.3 | 84.9 |
| ANCE | 33.0 | 95.9 | — | 81.9 | 87.5 | — | 80.3 | 85.3 |
| ME-BERT | 33.8 | — | — | — | — | — | — | — |
| RocketQA | 37.0 | 97.9 | 74.0 | 82.7 | 88.5 | — | — | — |
| Condenser | 36.6 | 97.4 | — | 83.2 | 88.4 | — | 81.9 | 86.2 |
| DPR-PAQ (BERT-base) | 31.4 | — | 74.5 | 83.7 | 88.6 | — | — | — |
| DPR-PAQ (BERT-large) | 31.1 | — | 75.3 | 84.4 | 88.9 | — | — | — |
| DPR-PAQ (RoBERTa-base) | 32.3 | — | 74.2 | 84.0 | 89.2 | — | — | — |
| DPR-PAQ (RoBERTa-large) | 34.0 | — | 76.9 | 84.7 | 89.2 | — | — | — |
| **coCondenser** | **38.2** | **98.4** | **75.8** | **84.3** | **89.0** | **76.8** | **83.2** | **87.3** |

（表 1：三个数据集上的检索性能对比。全 base 骨干下 coCondenser 在所有指标上最佳。）

**观察**：

1. 稠密系整体优于稀疏系；有 careful 预训练 / 微调的（RocketQA / DPR-PAQ / Condenser / coCondenser）显著强于早期稠密系（DPR / ANCE / ME-BERT），验证**低维稠密向量本身容量充足，只是难以简单微调发挥**。
2. **coCondenser 全面优于 RocketQA**：MS-MARCO +1.2 MRR@10（38.2 vs 37.0），NQ R@5 +1.8，同时**训练成本大幅下降**（batch 64 vs 4096；见表 2）。
3. **coCondenser 优于 Condenser**：MS-MARCO +1.6 MRR，NQ / TQA 各 +1 左右——证明**语料级对比预训练**这一步的必要性。它让 CLS 之间的空间几何被显式训练，微调更稳。
4. **DPR-PAQ 与 coCondenser 的对比**：在 NQ 上 DPR-PAQ RoBERTa-large 略优（76.9 R@5），因为它用了 6500 万合成 QA 对（domain-matched 半监督）；base 骨干下 DPR-PAQ 与 coCondenser 打平。**但在 MS-MARCO 上 DPR-PAQ 明显低于 coCondenser**（因为 PAQ 数据来自 NQ/TQA 训的 reader，与 MS-MARCO 域距离远）。结论：**当没有大规模合成数据且资源有限时，coCondenser 是更实用的选择**。

### 4.3 与 RocketQA 各阶段的详细对比

作者做了一份细致的 pipeline 对比（表 2）：

| 方法 / 阶段 | Batch | MRR@10 | R@1000 |
| :--- | ---: | ---: | ---: |
| **RocketQA** | | | |
| Cross-batch negatives | 8192 | 33.3 | — |
| + Hard negatives | 4096 | **26.0** | — |
| + Denoising | 4096 | 36.4 | — |
| + Data augmentation | 4096 | 37.0 | 97.9 |
| **coCondenser** | | | |
| Condenser w/o Hard neg | 64 | 33.8 | 96.1 |
| Condenser + Hard neg | 64 | 36.6 | 97.4 |
| coCondenser w/o Hard neg | 64 | 35.7 | 97.8 |
| **coCondenser + Hard neg** | 64 | **38.2** | **98.4** |

（表 2：各微调阶段的对比。coCondenser 全程 batch 64。）

**关键读点**：

1. **RocketQA 直接用 hard neg 反而掉分**（33.3 → 26.0）——mined hard neg 里的假负例伤到了它。必须加 CE denoising 才能救回（→ 36.4）。
2. **coCondenser 直接用 hard neg 就能涨**（35.7 → 38.2）——不用 CE 去噪。作者的解释：coCondenser 已经把 CLS 训练成一个"好的信息压缩器"，对局部 mislabel 天然鲁棒。
3. **coCondenser 完整版**（batch 64）> RocketQA 完整版（batch 4096 + denoise + data aug）：38.2 vs 37.0。
4. **Condenser vs coCondenser**：两者都是 batch 64，coCondenser 都比 Condenser 高 1.5–2 分——**语料级对比是关键的一步**。

**训练成本对比（作者的估算）**：

- RocketQA 完整 pipeline：需要极大 batch（8192）× 多阶段 × 训 CE 去噪器 × 外部数据打伪标——通常需要几十张 V100/A100 卡训好几天。
- coCondenser：4× RTX 2080Ti × pretrain 1 周（gradient cache）+ 单卡 3 epoch 微调。整体成本约 **RocketQA 的 1/5**。

---

## 5 结论（Conclusion）

本文识别了稠密检索器的两个根本问题——对噪声敏感、需要大 batch 学空间——并把它们**统一在预训练阶段**解决。coCondenser 通过：

1. Condenser 的三段式架构 + Head + CLS 强聚合 → 解决噪声鲁棒；
2. 语料级对比预训练 → 预热嵌入空间几何；

让下游微调变得**极简**：batch 64 + 单轮 hard neg mining + 标准 InfoNCE，即可与 RocketQA 相当。作者的立场：**"该做的事应该在预训练做，而不是在微调堆技巧"**。

作者也承认局限性：coCondenser 是完全无监督预训练；当有大规模合成 QA 数据可用时，语义相似的 domain-matched 半监督（如 DPR-PAQ RoBERTa-large）仍能获得进一步优势。

---

## 附录索引（Appendix Highlights）

- **A** 预训练超参数：AdamW lr 1e-4；每步 2000 篇文档（4× RTX 2080Ti + gradient cache）；训 8 epoch。
- **B** 微调超参数：MS-MARCO 用 batch 64、lr 5e-6、3 epoch；NQ/TQA 沿用 DPR 官方超参。
- **C** gradient cache 实现代码链接：https://github.com/luyug/GC-DPR

---

*翻译约定：稠密检索（dense retrieval）、bi-encoder（bi 编码器）、语料级对比（corpus-level contrastive）、语言模型预训练（LM pre-training）、梯度缓存（gradient cache）、难负例（hard negative）、CE 去噪（CE denoising）、数据增广（data augmentation）。DPR / ANCE / RocketQA / MS-MARCO / NQ / TQA / BM25 / DPR-PAQ / DeepCT / docT5query / GAR / ME-BERT 按惯例不译。*
