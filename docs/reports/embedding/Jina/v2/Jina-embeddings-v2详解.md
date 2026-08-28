# Jina Embeddings v2 技术详解

> 基于论文 [Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents](https://arxiv.org/abs/2310.19923)（Günther et al., Jina AI；arXiv:2310.19923）。
> v1 谱系：[Jina Embeddings](https://arxiv.org/abs/2307.11224)（arXiv:2307.11224）——T5 encoder + 两阶段对比微调，奠定数据清洗与双向 InfoNCE 配方。
> 本文把 **双向 ALiBi BERT、从零 MLM、两阶段嵌入微调、InfoNCE / 难负例损失、MTEB / LoCo / NarrativeQA 长文评测** 写全。

---

## 1. 一句话定位

**Jina Embeddings v2** 是开源侧较早把 **8192-token 长上下文** 做成通用文本嵌入的系列：改 BERT 为 **双向 ALiBi**，在 C4 上从零预训练，再经 **文本对 → 难负例** 两阶段对比微调，MTEB 上接近当时闭源 `text-embedding-ada-002`。


| 项 | 内容 |
| --- | --- |
| 规模 | Small **33M** / Base **137M** / Large **455M**（Jina BERT） |
| 上下文 | **8192**（预训练与微调主程仍以 ≤512 为主，推理外推） |
| 位置编码 | **双向 ALiBi**（无绝对位置嵌入） |
| 池化 | **Mean pooling** |
| 损失 | 双向 InfoNCE → 含 15 难负例的 $\mathcal{L}_{\mathrm{NCE}+}$ |
| 宣称 | MTEB 上 base≈**60.37**；与 ada-002（≈60.99）同档；LoCo / NarrativeQA 长文有增益 |

相对 v1（T5 encoder、短上下文）：v2 的核心卖点是 **「训短测长」**——用 ALiBi 把开源 BERT 系嵌入从 512 推到 8K，避免「切块 → 向量爆炸」的工程税。

---

## 2. 问题背景与 v1 谱系

### 2.1 为何要做长文档嵌入

实践中 BERT 系开源嵌入大多卡在 **512 tokens**。常见 workaround 是切段落分别编码：

- 语义被拆碎，全局主题难进单一向量；
- 库内向量数爆炸 → 内存、索引、检索延迟同步上升；
- DB 侧索引结构往往按向量条数计费，切块成本被放大。

Press et al.（ALiBi）证明：去掉绝对位置嵌入、在 attention logits 上加 **线性距离偏置**，语言模型可「训短测长」。当时 ALiBi 多用于 **因果 LM**；v2 把 **对称编码器版 ALiBi** 嵌进 BERT，专门服务双向编码与句向量。

### 2.2 与 v1 的继承关系

v1（arXiv:2307.11224）已固定 Jina 嵌入训练范式的骨架：

1. **大规模高质量文本对**（去重、语言过滤、一致性过滤）；
2. **双向 InfoNCE** 对训练；
3. **难负例三元组** 再精调；
4. 否定敏感数据等专项补丁。

v2 **换底座、加长上下文**，嵌入阶段损失与数据哲学与 v1 / E5 / GTE 同属 **多阶段对比学习** 一族，但 backbone 从 T5 encoder 换成 **自研 Jina BERT + ALiBi**。

### 2.3 三阶段总览

```text
Stage I    从零预训练 Modified BERT（MLM，C4，序列≤512）
Stage II   文本对微调（双向 InfoNCE，~40 数据源）
Stage III  难负例微调（每样本 1 正 + 15 负，$\mathcal{L}_{NCE+}$）
```

Stage II/III 都是嵌入任务；论文强调 **III 对检索与分类** 尤为关键。

---

## 3. 模型架构：Jina BERT + 双向 ALiBi

### 3.1 规格


| 模型 | Layers | Hidden | Params | Head dim |
| --- | --- | --- | --- | --- |
| Jina BERT Small | 4 | 512 | 33M | 64 |
| Jina BERT Base | 12 | 768 | 137M | 64 |
| Jina BERT Large | 24 | 1024 | 455M | 64 |

注意力头数按 **每头 64 维** 选取。嵌入模型在预训练 BERT 上加 **mean pooling**（无额外可训池化参数），输出单向量。

### 3.2 双向 ALiBi

标准绝对位置嵌入在外推到训练长度之外时失效。ALiBi **不训练位置嵌入**，而是在 softmax 前对 attention score 加与距离成正比的负偏置。因果版只偏置「看左侧」；编码器需要 **双向互看**，故采用 **镜像对称** 的 encoder ALiBi：每个 head 有独立斜率 $m_i$。

对 $n$ 个注意力头，斜率按论文 Eq.(1)：

$$
a = 2^{\lfloor \log_2 n \rfloor},\qquad
b = 2^{-8 / 2^{\lceil \log_2 n \rceil}},
$$

$$
m_i =
\begin{cases}
b^{2i} & i < a \\
b^{1 + 2(i-a)} & i \ge a
\end{cases}.
\tag{1}
$$

直觉：近邻 token 的 attention 偏置更小（更易互相注意），远距离被线性压制；不同 head 用不同 $m_i$ 覆盖多尺度距离偏好。**因无位置嵌入**，预训练可固定在 512，推理直接外推到 8192——这是 v2「8K 嵌入」的几何基础。

### 3.3 GLU FFN 与 LayerNorm

- FFN 侧采用 **Gated Linear Units**（Dauphin et al.；Shazeer GLU 变体）：
  - Small / Base：**GEGLU**（GELU 门控）；
  - Large：**ReGLU**（ReLU）——作者观察到 Large + GEGLU 虽早期 MLM 好看，但训练不稳。
- LayerNorm：采用 Transformer 经典 **post-LN**；试过 pre-LN（Megatron / Nguyen–Salazar）未见稳定或效果收益，故未采用。

### 3.4 与 RoBERTa / BERT 的差异小结


| 设计点 | 经典 BERT/RoBERTa | Jina BERT v2 |
| --- | --- | --- |
| 位置 | 绝对位置嵌入 | **双向 ALiBi** |
| FFN | 标准 FFN | **GLU** |
| NSP | BERT 有 / RoBERTa 无 | **无 NSP**（跟 RoBERTa） |
| 序列打包 | 常多文档打包 | **单文档前 512，不跨文档打包** |

---

## 4. Stage I：预训练

### 4.1 数据

英文 **C4**（Colossal Cleaned Common Crawl）：约 **3.65 亿** 文档、约 **1700 亿** tokens；仅保留英文，过滤不当内容。留 **1%** 作 validation（MLM loss / accuracy）。→ 官方开源 v2-en 为 **单语英文**（后续有双语变体系列，不在本篇论文主文）。

### 4.2 MLM 目标

掩码 **30%** tokens，**whole-word masking**：

- 80% → `[MASK]`；
- 10% → 随机词；
- 10% → 保持原词。

解码器 $f:\mathbb{R}^d\to\mathbb{R}^{|V|}$ 对掩码位置输出词表分布。设 $I(k)$ 为第 $k$ 个掩码位置的真词下标，论文 Eq.(2)（交叉熵形式，文中写作对预测概率取对数再求和）：

$$
\mathcal{L}_{\mathrm{MLM}}(t) := \sum_{k=1}^{n} \ln f(e_i)_{I(k)},
\tag{2}
$$

实践中等价于标准负对数似然 $-\sum\log p(w_k\mid\text{context})$。**不做 NSP**。

### 4.3 序列与批策略

- 全局 batch **4096**；序列上限 **512**；
- 每文档只取 **前 512 tokens**，不采样多段、不跨文档拼装；
- 因序列长度变化，每 batch 掩码 token 数不固定。

无位置嵌入 → 「先短训、后长测」合法。作者在 C4 val 上扫序列长度做 MLM accuracy：Jina BERT **外推到 8192 时 MLM 不崩**；标准 BERT/RoBERTa 因绝对位置表无法算 >512。

### 4.4 优化

- AdamW：$\beta_1=0.9,\ \beta_2=0.98,\ \epsilon=10^{-6}$，weight decay $0.01$，dropout / attn dropout $0.1$；
- 线性 warmup 10k steps 到峰值 $\eta$，再线性降到 0（至 100k steps）；
- $\eta$：Small $10^{-3}$ / Base $6\times10^{-4}$ / Large $4\times10^{-4}$；
- **FP16** 动态混合精度 + DeepSpeed；试 BF16 在 MLM 与下游 GLUE 上不佳。

### 4.5 Backbone GLUE 速览

微调后提交 GLUE test：Jina BERT Base 平均约 **80.7**，Large **81.6**——低于 RoBERTa Large 峰值，但对「嵌入底座」够用；重点证据是 **长序列 MLM 外推曲线平坦**。

---

## 5. Stage II–III：嵌入微调

### 5.1 Mean pooling

对最后一层 token 隐状态做均值，得到固定维句向量。无额外池化参数，与 Sentence-BERT 系一致。

### 5.2 Stage II：文本对 + 双向 InfoNCE

**数据**：约 **40** 个异构源（延续 v1 配比与采样）；**标题–摘要** 对显著利于聚类。用一致性过滤（Dai et al.；Wang et al. / E5）抬高质量。每个新 batch **先随机选一个数据源**，再抽满 batch；按质量/体量设不同采样率。

对 batch $B\subset D_{\mathrm{pairs}}$，余弦相似度 $s(\cdot,\cdot)$，温度 $\tau$，单向 InfoNCE（论文 Eq.3）：

$$
\mathcal{L}_{\mathrm{NCE}}^{\mathrm{pairs}}(B)
:=
\mathbb{E}_{(q,p)\sim B}
\left[
-\ln
\frac{e^{s(q,p)/\tau}}
{\sum_{i=1}^{k} e^{s(q,p_i)/\tau}}
\right].
\tag{3}
$$

对称性要求 **双向**（Eq.4）：

$$
\mathcal{L}^{\mathrm{pairs}}(B)
:=
\mathcal{L}_{\mathrm{NCE}}^{\mathrm{pairs}}(B)
+
\mathcal{L}_{\overline{\mathrm{NCE}}}^{\mathrm{pairs}}(B),
$$

其中 $\mathcal{L}_{\overline{\mathrm{NCE}}}$ 把 $(q,p)$ 对调，用 $p$ 去对齐所有 $q_i$。经验上 $\tau=0.05$。

这与 v1 的 $\mathcal{L}^{\mathrm{pairs}}$ 同构：in-batch 负例越多，对比信号越强 → **大 batch 几乎是硬需求**。

### 5.3 Stage III：难负例 + $\mathcal{L}_{\mathrm{NCE}+}$

**数据**：MS MARCO、Natural Questions 等检索集 + NLI 等非检索集。每个样本形如 $(q,p,n_1,\ldots,n_{15})$：**1 正 + 15 负**。

- 检索集：用检索模型挖 **hard negatives**，再用 **cross-encoder** 校验「负例相关性确实低于正例」；
- 非检索集：**随机负例**——硬负例在「相似/不相似」非二元场景下常伤质量，但仍需混入以保住 STS 等非检索能力。

扩展 InfoNCE（论文 Eq.5，双向）：

$$
\mathcal{L}_{\mathrm{NCE}+}(B)
:=
\mathbb{E}_{r\sim B}
\Bigg[
-\ln
\frac{e^{s(q,p)/\tau}}
{\sum_{i=1}^{k}\Big(
e^{s(q,p_i)/\tau}
+\sum_{j=1}^{15} e^{s(q,n_{j,i})/\tau}
\Big)}
\Bigg]
+
\mathbb{E}_{r\sim B}
\Bigg[
-\ln
\frac{e^{s(p,q)/\tau}}
{\sum_{i=1}^{k} e^{s(p,q_i)/\tau}}
\Bigg],
\tag{5}
$$

其中 $r=(q,p,n_1,\ldots,n_{15})$。相对 Stage II：分母显式加入 **同 batch 其它正例 + 显式 hard negatives**，逼模型区分「相关」与「仅语义相近」。

### 5.4 显存工程

InfoNCE 的有效负例数随 batch 线性增。作者用：

- 混合精度；
- DeepSpeed；
- **Activation checkpointing**（每个 BERT layer 后插入）。

以换更大 global batch，避免「小 batch 对比面太窄」。

---

## 6. 评测

### 6.1 MTEB（短文为主）

MTEB：8 类任务、58 数据集。论文 Table 3（节选）：


| Model | Params | CF | CL | PC | RR | RT | STS | SM | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text-embedding-ada-002 | — | 70.93 | 45.90 | 84.89 | 56.32 | 49.25 | 80.97 | 30.80 | **60.99** |
| e5-base-v2 | 110M | 73.84 | 43.80 | 85.73 | 55.91 | 50.29 | 81.05 | 30.28 | **61.50** |
| jina-small-v2 | 33M | 68.82 | 40.08 | 84.44 | 55.09 | 45.64 | 80.00 | 30.56 | **58.12** |
| jina-base-v2 | 137M | **73.45** | 41.74 | 85.38 | 56.99 | **48.45** | 80.70 | 31.60 | **60.37** |

解读：

- **base** 分类很强，检索接近 ada-002；
- **small** 在检索上仍有竞争力，体现「Stage III 难负例」对检索的杠杆；
- 相对未做二阶段精调的 `all-MiniLM` / `all-mpnet`，检索优势符合预期。

### 6.2 长文任务：为何要自造集

MTEB 文本普遍偏短，测不出 8K。作者补充：

1. **PatentClustering**（BigPatent）：专利按类别聚类，均长约 **6376** tokens；
2. **WikiCitiesClustering**：城市条目按国家聚类，均长约 **2031**；
3. **NarrativeQA** 检索：文学/剧本，均长约 **7.5 万** tokens（极长）；
4. 另评 **LoCo**（Stanford M2-BERT 文配套长文检索套件）。

扫 `max_length` 截断曲线（Figure 3）：

- **NarrativeQA**：加长上下文收益极大——答案常不在文首，512 截断必败；
- **BigPatent** 聚类：长上下文通常更好；
- **WikiCities**：更长有时略伤——国家信息多在首段，后文噪声干扰 mean pooling。

结论：**「更长」不是单调增益**，取决于任务信息落点；但 NarrativeQA 类「全文理解」任务是 8K 的硬价值证明。

### 6.3 LoCo（论文 Table 11）

`jina-base-v2`（137M, 8192）avg nDCG@10 **85.4**；`jina-small-v2` **83.4**。对比：M2-BERT-32768 微调后可更高，但 jina-base-v2 在 **未专门为 LoCo 微调** 的设定下已接近一众 512 模型的微调版，并远好于当时部分闭源长上下文 API 在该榜的表现（表内 ada / voyage 等分数需注意评测设定差异）。

### 6.4 Backbone 长序列 MLM

Figure 2：训练长度 512，推理到 8192，Jina BERT MLM accuracy 基本不掉——支撑「嵌入微调虽多在 ≤512 文本上，推理仍可喂长文」的工程叙事。

---

## 7. 局限与使用注意

1. **预训练与微调长度仍短**：8K 能力主要靠 ALiBi 外推，而非大规模 8K 对比训练；极端长依赖任务仍可能弱于「真·长上下文对比训」模型。
2. **英文主模型**：主文 C4 英；多语/双语是后续产品线，勿把本篇论文当成多语 SOTA 声明。
3. **Mean pooling + 超长文**：后半噪声可能稀释信号（WikiCities 现象）；生产上仍可能要「智能切块 + 多向量」折中。
4. **Large 训练不稳**：GEGLU→ReGLU 的经验选择，暗示 GLU 变体与模型规模耦合。
5. **与 E5-base**：MTEB 平均略逊 e5-base-v2；卖点是 **上下文长度与开源可复现**，不是全面碾压同尺寸短上下文模型。
6. **评测缺口**：标准 MTEB 不能充分反映 8K；应用方应用 NarrativeQA / LoCo / 自有长 PDF 做回归。

---

## 8. 谱系位置与后续

```text
v1 (2307.11224)  T5 encoder · 数据清洗 · 双向 InfoNCE · 难负例 / 否定
        │
        ▼
v2 (2310.19923)  Jina BERT + 双向 ALiBi · 8192 · 同对比配方升级
        │
        ├─ 双语变体（zh/es/de 等，另文 Mohr et al.）
        ▼
v3 (2409.10173)  XLM-R + RoPE · Task LoRA · MRL · 多语 570M
        ▼
v4 (2506.18902)  Qwen2.5-VL-3B · 单/多向量 · 视觉文档 · 32K
```

v2 在谱系中的历史角色：**把「开源通用嵌入」的上下文上限从 512 拉到 8K**，并证明 ALiBi 编码器路线可行；对比损失与两阶段数据流被 v3/v4 继续继承并扩展为 Task LoRA / 多模态。

---

## 9. 公式速查

| 符号 | 含义 |
| --- | --- |
| $m_i$ | ALiBi 第 $i$ 头斜率（Eq.1） |
| $\mathcal{L}_{\mathrm{MLM}}$ | 掩码语言模型损失（Eq.2） |
| $\mathcal{L}^{\mathrm{pairs}}$ | 双向 InfoNCE（Eq.3–4） |
| $\mathcal{L}_{\mathrm{NCE}+}$ | 含 15 hard negatives 的双向 InfoNCE（Eq.5） |
| $s(\cdot,\cdot)$ | 余弦相似度 |
| $\tau$ | 温度，常用 $0.05$ |

---

## 10. 小结

Jina Embeddings v2 = **双向 ALiBi BERT（训短测长）** + **v1 系两阶段对比微调**。它不是参数量最大的嵌入模型，但在 2023 开源生态里清晰回答了：「长文档能否进 **一个** 向量、且下游不崩？」——答案是可以，且 NarrativeQA / LoCo 显示 **上下文预算本身就是特征**。后续 v3 用 RoPE + Task LoRA 把多语与任务特化做深；v2 则是长上下文开源嵌入的关键里程碑。
