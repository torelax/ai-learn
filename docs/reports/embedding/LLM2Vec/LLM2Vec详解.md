# LLM2Vec 技术详解

> **paper**: [arXiv:2404.05961](https://arxiv.org/abs/2404.05961) · [PDF](https://arxiv.org/pdf/2404.05961)
> **code**: [McGill-NLP/llm2vec](https://github.com/McGill-NLP/llm2vec)
> **refs**: [SimCSE](https://aclanthology.org/2021.emnlp-main.552/) · [Echo / Repetition](https://arxiv.org/abs/2402.15449) · [E5-Mistral](https://arxiv.org/abs/2401.00368)
> **backbone**: Sheared-LLaMA-1.3B / LLaMA-2-7B / Mistral-7B / Llama-3-8B（Decoder-only）
> **date**: 2024-04（arXiv）；作者 BehnamGhader et al., McGill / Mila / ServiceNow
> **modality**: 文本
> **languages**: 实验主英文；配方语言无关，换语需重跑 MNTP/SimCSE
> 本文写全 **双向注意力、MNTP、无监督 SimCSE、监督对比微调**、Echo 对照、词级探针对齐（附录 D.1.2 / D.1.3）与表示分析。

---

## 1. 一句话定位

**LLM2Vec** 是一套 **无监督、参数高效** 的配方：把任意因果 Decoder LLM 改造成强文本编码器，而无需 GPT-4 合成数据或昂贵全量适配：


| 项            | 内容                                                                      |
| --------------- | --------------------------------------------------------------------------- |
| 三步          | ① 启用双向注意力 → ②**MNTP** → ③ **无监督 SimCSE**                   |
| 适配          | **LoRA**；每步约 1000 steps                                               |
| 验证底座      | Sheared-LLaMA-1.3B、LLaMA-2-7B、Mistral-7B、Llama-3-8B                    |
| 无监督 MTEB   | Mistral-7B+LLM2Vec 达**56.80**（当时无监督 SOTA）                         |
| +公开数据监督 | Llama-3-8B+LLM2Vec（仅 Bi+MNTP）达**65.01**（公开数据训练 SOTA，2024-05） |

核心主张：**Decoder「偷偷」具备通用编码能力，只需极少适配即可释放**；相对 Echo Embeddings（复制输入翻倍长度），LLM2Vec **不增加推理序列长度**。

---

## 2. 问题背景

### 2.1 为何 Decoder 不直接适合做句向量

因果自注意力：

$$
\mathbf{O}
=
\mathrm{softmax}\!\left(
\frac{\mathcal{M}_{\{j\le i\}}\,\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}
\right)\mathbf{V},

$$

位置 $i$ 的表示 **看不到** $i+1,\ldots,N$。生成需要这约束；嵌入则希望每个 token（及句向量）融合全句。仅取 last-token / EOS 在因果模型里是「只能汇总前文」的折中，对词级任务与细粒度语义常次优。

### 2.2 为何仍想用 Decoder

相对 Encoder-only：

1. **样本效率**：LM 对所有 token 计损失，而非仅 15% mask；
2. **生态**：工具链、继续预训练、指令对齐成熟；
3. **指令遵循**：利于 universal embedding（query 侧自然语言任务描述）。

LLM2Vec 要解决的是：**在不毁掉这些优势的前提下，补上双向上下文与句级对比几何**。

---

## 3. 三步配方详解

### 3.1 Step 1 — Enabling bidirectional attention

把因果掩码换成全 1 矩阵（all-ones），使任意 token 可 attend 全文。这一步 **零训练**，但多数模型上会 **立刻掉点**——权重从未为「看见未来」优化过。例外：论文发现 **Mistral-7B** 对裸双向异常鲁棒（见 §6）。

### 3.2 Step 2 — Masked Next Token Prediction (MNTP)

目标：让模型学会 **利用双向上下文**，同时尽量对齐 Decoder 预训练的「下一词预测」接口。

给定序列 $\mathbf{x}=(x_1,\ldots,x_N)$，随机 mask 一部分位置。预测被 mask 的 $x_i$ 时，**用位置 $i-1$ 的表示** 出 logits，而不是用被 mask 位置自身：

$$
\mathcal{L}_{\mathrm{MNTP}}
=
-\sum_{i\in \mathcal{M}}
\log P_\theta\!\big(x_i \mid \mathbf{h}_{i-1}^{\mathrm{bi}}\big).

$$

其中 $\mathbf{h}_{i-1}^{\mathrm{bi}}$ 来自 **双向** Transformer。直觉：

- 仍是「用前文位置预测下一词」，贴近 NTP；
- 但 $\mathbf{h}_{i-1}$ 已混入未来信息，迫使注意力学会读右侧。

Mask token 用下划线 `_`（词表无专用 [MASK]）。掩码率与 BERT/RoBERTa 策略做网格搜索；按 SICK-R 选型（附录 D.1.1）：


| 模型                        | 策略       | Mask 概率 |
| ----------------------------- | ------------ | ----------- |
| S-LLaMA / LLaMA-2 / Llama-3 | BERT 式    | 20%       |
| Mistral-7B                  | RoBERTa 式 | **80%**   |

训练：Wikipedia Wikitext-103，**LoRA $r=16,\alpha=32$**，1000 steps，batch 32；7B/8B 约 **100 分钟 / 单卡 80GB A100**。

### 3.3 Step 3 — Unsupervised contrastive (SimCSE)

MNTP 改善 **词级上下文**；句向量还需显式句级目标。采用 **SimCSE**（Gao et al., 2021）：同一句过模型两次，独立 dropout，得 $\mathbf{z}_i,\mathbf{z}_i^+$；batch 内其它句为负例。

$$
\mathcal{L}_{\mathrm{SimCSE}}
=
-\log
\frac{
e^{\mathrm{sim}(\mathbf{z}_i,\mathbf{z}_i^+)/\tau}
}{
\sum_{j=1}^{B}
e^{\mathrm{sim}(\mathbf{z}_i,\mathbf{z}_j^+)/\tau}
}.

$$

数据：Gao et al. 发布的 Wikipedia 句子子集（无配对正负句）。句向量对 token 隐状态做 **mean pooling**。

#### 附录 D.1.2 训练细节（实现必读）

1. **先 merge MNTP LoRA**，再 **新初始化** SimCSE LoRA（$r=16,\alpha=32$），避免遗忘、目标互抢同一套适配器。
2. **Dropout = 0.3**（原版 SimCSE 常用 0.1）。作者发现大 Decoder 上 0.1 扰动太弱，两个视图几乎一样，对比学不动；提到 0.3 才有效。
3. **1000 steps**；LLaMA-2-7B / Mistral-7B **batch 128**；S-LLaMA-1.3B **batch 32**（对比学习靠 in-batch 负例，7B 用更大 batch）。
4. 7B：bf16 + FlashAttention-2 + gradient checkpointing；单卡 80GB A100 约 **2.5–3 小时**。

### 3.4 Pooling

消融三种：

1. **EOS / last-token**；
2. **Mean pooling**；
3. **Weighted mean**（SGPT 式，越靠后权重越大）。

结论：**因果基线**偏好 weighted mean；**LLM2Vec（双向）** 偏好 **mean pooling**。评测时指令 token **不参与** pooling。

---

## 4. 监督对比学习（可选第四步）

在公开 E5 数据复刻（约 1.5M，Springer et al. 整理）上继续 InfoNCE：

$$
\mathcal{L}
=
\frac{
e^{\lambda s(q,d^+)}
}{
e^{\lambda s(q,d^+)}
+
\sum_{d^-\in N}
e^{\lambda s(q,d^-)}
},

$$

含 hard negatives + in-batch negatives。LoRA 同样 $r=16$；从 MNTP（或 MNTP+SimCSE）权重初始化；batch 512，1000 steps，8×A100。

重要现象：**有强监督时，无监督 SimCSE 不是必须**——「仅 Bi+MNTP」在 Llama-3/Mistral 上甚至略优；但 SimCSE 显著提升 **样本效率**（更少 step 达到更高分），在数据/算力受限时仍关键。

---

## 5. 实验结果

### 5.1 词级任务（CoNLL-2003 线性探针）

**CoNLL-2003** 上三个经典词级标注任务（线性探针，见附录 D.1.3）：


| 任务         | 含义                     |
| -------------- | -------------------------- |
| **POS**      | 词性标注（名/动/形…）   |
| **Chunking** | 浅层短语块（如 NP）      |
| **NER**      | 命名实体（人/地/组织…） |

协议要点：冻住最后一层隐状态；上头线性分类器（dropout 0.1）；约 14k/3.25k/3.45k train/val/test；1500 steps，lr $5\times10^{-4}$，batch 8。测的是 **token/词表示**是否好用，不是句向量检索。

趋势（对应论文 **Figure 2**）：

1. 因果 Uni 已强于 DeBERTa-v3-large（模型更大、数据更多）；
2. **裸 Bi** 对 LLaMA 系大伤，对 Mistral 伤害小甚至 NER 微升；
3. **Bi+MNTP** 全面最好；再加 SimCSE 对词级 **略降**。

**为何 Figure 2 上 Bi+MNTP+SimCSE < Bi+MNTP？**
论文原文：SimCSE 把表示往 **序列级** 对比几何拉，干扰 token 级探针——**预期现象，不是图画错**。对照：无监督 MTEB（句级）上全三步远强于「仅 Bi+MNTP」（见 §5.2）。取舍：要词级停 Bi+MNTP；要 embedding 加 SimCSE。

### 5.2 无监督 MTEB（56 任务）


| 设定               | Mistral-7B Avg | Llama-2-7B | S-LLaMA-1.3B | Llama-3-8B  |
| -------------------- | ---------------- | ------------ | -------------- | ------------- |
| Uni + w.mean       | 42.46          | 44.54      | 35.05        | 43.98       |
| Bi + Mean（无训）  | **46.86**      | —         | —           | 30.56（崩） |
| LLM2Vec w/o SimCSE | 49.43          | 45.70      | 41.43        | 48.84       |
| **LLM2Vec 全三步** | **56.80**      | **55.36**  | **49.42**    | **56.23**   |
| Echo（同底座）     | 50.26          | 45.36      | 39.10        | 45.32       |

相对最佳因果基线，全三步相对提升约 **23%–50%**（子集消融口径）。Mistral 无监督 **56.80** 为当时 unsupervised SOTA。

### 5.3 监督 MTEB（仅公开数据）


| 模型                             | Avg       |
| ---------------------------------- | ----------- |
| Instructor-xl                    | 61.79     |
| BGE-large-en-v1.5                | 64.23     |
| E5-Mistral + public              | 64.56     |
| Echo-Mistral                     | 64.68     |
| GritLM-Mistral + public          | 64.70     |
| Mistral LLM2Vec (Bi+MNTP)        | 64.80     |
| **Llama-3-8B LLM2Vec (Bi+MNTP)** | **65.01** |

训练曲线：LLM2Vec 初始化后，**更少 step 达到更高子集分数**——适配把优化带到更好盆地。

### 5.4 相对 Echo Embeddings（机制展开）

**Echo**（Springer et al., 2024）：不改因果掩码、通常零训练；把输入 **复制一份接在后面**，在 **第二遍（副本）** 上取句向量。

设原文 $x_1,\ldots,x_n$，输入变为长度 $\approx 2n$：

$$
[\,x_1,\ldots,x_n \;\Vert\; x_1,\ldots,x_n\,].

$$

因果注意力下，副本位置已能看见整段第一遍，从而「用长度换反向依赖」。

**SET（Sentence Embedding Token）**：约定用来读出句向量的位置（常是 Prompt 末尾或序列最后 token），不是词表里的特殊符号。Echo 必须取 **副本上的 SET**（或只对第二段做 mean），若误取第一遍 SET，Echo 机制几乎白做。

对比：


|            | Echo      | LLM2Vec        |
| ------------ | ----------- | ---------------- |
| 改注意力？ | 否        | **是（双向）** |
| 训练？     | 通常否    | MNTP + SimCSE  |
| 长度       | **≈2×** | **1×**        |

效果：可比设定下 LLM2Vec 更高或持平；推理 Echo 更慢（MTEB 全量、Mistral：Echo ≈64h vs LLM2Vec ≈44h，8×A100）。长文场景 2× 易顶满上下文。

---

## 6. 表示分析：LLM2Vec 改了什么

### 6.1 未来信息是否进入前缀表示

构造共享前缀 $A$、正续写 $C$、负续写 $D$ 的句三元组；只对前缀 $A$ pooling，看 $\mathrm{cos}(q,s^+)$ vs $\mathrm{cos}(q,s^-)$。
因果模型难分；**Bi+MNTP** 后正负分离清晰——证明未来 token 信息写入了前缀向量。

### 6.2 为何 Mistral「裸双向」也能用

对同一 Wikipedia 段，比较因果隐状态 $\mathbf{H}^\ell_{c}$ 与双向 $\mathbf{H}^\ell_{\mathrm{bi}}$ 的逐层余弦。LLaMA 系几乎全层低相似（表示被打乱）；**Mistral 全层高相似**。作者推测：Mistral 预训练可能含 **prefix LM / 某种双向成分**（未公开证实）。多版本 Mistral（base/instruct v0.1/v0.2）重复该现象。

实践含义：

- 换底座时 **不要假设裸 Bi 安全**；
- MNTP 仍是默认必要步；Mistral 只是「伤得少」。

---

## 7. 词级表示的特殊 pooling（附录 D.1.3）

LLM 为子词词表：词向量 = 其子词隐状态平均。例：`w1 w2 w3` 分词为 `BOS t11 t12 | t21 t22 t23 | t31`。

**普通设定**（因果 / 裸 Bi 等）：

$$
e_1=\tfrac{e_{11}+e_{12}}{2},\quad
e_2=\tfrac{e_{21}+e_{22}+e_{23}}{3},\quad
e_3=e_{31}.

$$

**经过 MNTP 的模型**：损失用 **位置 $i-1$** 预测 $x_i$，特征与标签左对齐。评测须改用 **上一词相关子词**（附录公式）：

$$
e_1=\tfrac{e_{\mathrm{BOS}}+e_{11}}{2},\quad
e_2=\tfrac{e_{12}+e_{21}+e_{22}}{3},\quad
e_3=e_{23}.

$$

漏掉这一步会导致探针与训练几何错位，复现词级分数对不上论文。这也解释了 §5.1：词级协议本身测的是对齐后的 token 表示；SimCSE 再拉句级空间会伤这些探针。

---

## 8. 指令评测协议

与 Wang et al. (E5-Mistral) 同一套 MTEB 任务指令；指令加在 query；对称任务两侧同指令。这使 LLM2Vec 结果可与 Echo / E5-Mistral 公平对比。

---

## 9. 局限（论文诚实清单）

1. **模型大**：7B 索引维常 4096，建库贵——可用 LoRA 学生或更小底座（1.3B 已验证配方）。
2. **污染**：底座预训数据不透明，MTEB 可能部分重叠。
3. **语言**：实验主英文；配方语言无关，但需目标语无标注语料重跑 MNTP/SimCSE。
4. **全量微调**：论文走 LoRA；与「从零训嵌入 LLM」（Conan-v2）是不同成本曲线。

---

## 10. 实现对照清单

```text
1. 加载 Decoder LLM；attention mask → bidirectional
2. LoRA + MNTP on Wiki（mask 策略按底座选，D.1.1）
3. merge MNTP LoRA；新 LoRA + SimCSE（dropout 0.3, mean pool, D.1.2）
4. （可选）公开检索/STS 数据监督 InfoNCE
5. 评测：任务指令 + mean pool（排除指令 token）
6. 词级探针：MNTP 模型用「上一词子词」对齐（D.1.3）
```

可验证目标：

1. 词级：Bi+MNTP > Uni > Bi；且 Bi+MNTP ≥ Bi+MNTP+SimCSE；
2. 无监督 MTEB：三步 > 两步 > Uni；
3. 前缀未来信息探测：MNTP 后正负余弦可分；
4. 监督：LLM2Vec 初始化收敛更快。

---

## 11. 与相关路线对照


| 方法                   | 双向方式                                | 无监督句向量    | 推理代价       |
| ------------------------ | ----------------------------------------- | ----------------- | ---------------- |
| 裸 last-token          | 无                                      | 弱              | 基线           |
| Echo                   | 复制输入；**副本 SET / 第二段 pooling** | 可选            | **2× 长度**   |
| Token Prepending       | 否；层间 SET→句首                      | 否（零训练）    | +1 token       |
| LLM2Vec                | 改 mask + MNTP                          | **SimCSE**      | **1×**        |
| E5-Mistral / gte-Qwen2 | 训练期双向或等价                        | 多靠合成/弱监督 | 1× 但需大数据 |
| GritLM                 | 部分层/任务混合 Gen+Emb                 | 混合目标        | 1×            |
| Conan-v2               | **soft-mask 渐变**                      | 弱监督+监督     | 从零训 1.4B    |

LLM2Vec 的独特价值是：**最低标注与算力门槛下的 Decoder→Encoder 万能转接头**。

---

## 12. 小结

LLM2Vec 用三步把「秘密编码器」变为可部署现实：

1. **Bi** —— 打开全句信息流；
2. **MNTP** —— 用「上一位置预测被 mask 词」对齐因果预训练并教会读未来；
3. **SimCSE** —— 无配对句数据即可塑造句向量空间（大 Decoder 用 dropout 0.3）。

再加可选公开数据监督，即可在不碰专有合成数据的约束下冲击 MTEB 前列。对工程：优先在现有 Instruct LLM 上跑通 Bi+MNTP+mean-pool，再决定是否加 SimCSE/监督；词级应用勿漏 D.1.3 对齐。对研究：Mistral 的「天生双向耐受」仍是开放谜题。

同目录对照：《GTE系列详解.md》（大数据多阶段 LLM 嵌入）、《Conan-embedding-v2详解.md》（soft-mask 与从零训练）、《E5详解.md》（对比学习公共底座）、《Token-Prepending详解.md》（免训练因果补丁）。

---

## 参考

1. BehnamGhader et al. (2024). LLM2Vec. [arXiv:2404.05961](https://arxiv.org/abs/2404.05961)
2. Gao et al. (2021). SimCSE. EMNLP.
3. Springer et al. (2024). Echo / Repetition Improves LM Embeddings. [arXiv:2402.15449](https://arxiv.org/abs/2402.15449)
4. Wang et al. (2024). Improving Text Embeddings with LLMs. [arXiv:2401.00368](https://arxiv.org/abs/2401.00368)
5. Tjong Kim Sang & De Meulder (2003). CoNLL-2003 NER shared task.
