# Conan-embedding-v2 技术详解

> 基于论文 [Conan-Embedding-v2: Training an LLM from Scratch for Text Embeddings](https://arxiv.org/abs/2509.12892)（Li et al., Tencent PCG；亦见 EMNLP 2025）。
> v1 动态难负例谱系：[Conan-embedding](https://arxiv.org/abs/2408.15710)（arXiv:2408.15710）。
> 本文把 **从零训练 1.4B 嵌入向 LLM、soft-masking、跨语检索数据集 CLR、动态难负例 DHNM、InfoNCE/CoSENT 公式与 MTEB SOTA 宣称（2025-05）** 写全。

---

## 1. 一句话定位

**Conan-embedding-v2** 不是「在 Mistral/Qwen 上 LoRA 一下」，而是 **为嵌入任务从零预训练并微调的 1.4B LLM**：

| 项 | 内容 |
| --- | --- |
| 规模 | **≈1.4B**；**8 层 × hidden 3584**（与 gte-Qwen2-7B 同宽、浅得多） |
| 上下文 | **32,768** |
| 词表 | 自研多语 tokenizer，**150K** |
| 嵌入维 | **3584**；支持 **MRL**（256…3584） |
| 关键创新 | **Soft-mask**（因果→双向渐变）、**CLR 跨语检索数据**、**DHNM 动态难负例** |
| 宣称 | 约 **2025-05-19** 在 **MTEB 英文与 C-MTEB** 达 SOTA（论文表：EN **73.52** / ZH **74.24**） |

相对「大底座 + LoRA」路线，Conan-v2 直接攻击两个鸿沟：**数据鸿沟**（LLM 语料 ≠ 嵌入语料）与 **训练鸿沟**（token-level 因果 LM ≠ sentence-level 双向对比）。

---

## 2. 问题背景：为何 LoRA 不够

### 2.1 数据鸿沟

E5-Mistral / gte-Qwen2 等依赖通用 LLM 预训练语料。嵌入更需要：新闻标题–正文、问答、检索三元组、跨语平行检索等 **配对结构**。仅在嵌入阶段 LoRA，难以改写底座里「非配对」的先验。

### 2.2 训练鸿沟

| | LLM 预训练 | 嵌入训练 |
| --- | --- | --- |
| 掩码 | **因果**（上三角屏蔽） | **双向** |
| 损失 | token CE | **句向量 InfoNCE / CoSENT** |
| 输出 | 下一词分布 | 单向量 |

论文引用：全量微调在鸿沟上易 **表征崩塌**，LoRA 因更新子空间小反而更稳（Biderman et al.；Zhang et al.）。但 LoRA 表达力上限也受限。Conan-v2 的回答：

1. 预训练就混入嵌入向数据；
2. 用 **soft-mask** 平滑因果→双向；
3. 全量微调嵌入阶段，使 soft-mask 下 **更高 rank / 全参** 真正受益。

---

## 3. 模型架构

在固定参数预算下，Kaplan 等缩放律显示层数超过约 7 后测试损失趋于平坦。作者刻意选 **8 层**，把参数砸向 **宽度**：

| 配置 | 值 |
| --- | --- |
| Layers | 8 |
| Hidden size | **3584**（对齐 gte-Qwen2-7B 宽度） |
| Attention heads | 32 |
| KV heads (GQA) | 8 |
| FFN intermediate | 8192 |
| Max length | 32768 |
| Vocab | 150000 |
| 总参数 | ≈1.4B / 报告也称 1503M |

收益：**同维教师/学生易对齐**；推理相对 7B 深模型更快（论文测 Amazon Reviews 英文 query：Conan-v2 **5.14 min** vs gte-Qwen2-7B **31.78 min**，同卡设定）。

---

## 4. 四阶段训练管线

```text
Stage 1  LLM Pre-training          ~3T tokens（加重新闻/QA/网页）
Stage 2  LLM SFT                   ~600M instruction 对
Stage 3  Embedding weakly-supervised   同对改格式 + soft-mask + InfoNCE
Stage 4  Embedding supervised          CLR + DHNM + 多任务损失
```

### 4.1 Stage 1–2：LLM 向嵌入对齐

- 预训练过滤参考 InternLM2 等标准清洗；强调 **news / QA / web**；
- SFT：把 (query, positive) 写成 instruction–input–output；约 6 亿条；
- Tokenizer 在约 40 万多语语料上训练。

### 4.2 Stage 3：弱监督嵌入

同一批 pair，改成双塔对比格式：instruction+input → query，output → positive。用 **gte-Qwen2-7B-instruct** 打分，**低于 0.4 丢弃**。损失为 in-batch InfoNCE（论文写法）：

$$
\mathcal{L}_{\mathrm{neg}}
=
-\sum_{i=1}^{N}
\log
\frac{
\exp\big(\mathrm{cos}(x_i,\, y_i^+)\big)
}{
\sum_{j=1}^{M}
\exp\big(\mathrm{cos}(x_i,\, y_j)\big)
}.

$$

其中 $x_i$ 为 query 向量，$y_i^+$ 为正例，$y_j$ 含 in-batch 其它文档。此阶段启用 **soft-mask**（下一节）。

弱监督规模约 **17.66 亿** 对（新闻 6.2 亿、社交 6.9 亿、知识库、网页、论文、社区 QA、指令数据等）。

### 4.3 Stage 4：监督多任务

任务四分：

1. **Retrieval**；
2. **Cross-lingual retrieval (CLR)**；
3. **Classification**；
4. **STS**。

前三类用 InfoNCE（query / pos / negs）；STS 用 **CoSENT**（优于 CE 的排序式相似损失）：

$$
\mathcal{L}_{\mathrm{CoSENT}}
=
\log\!\Bigg(
1+
\sum_{(i,j,k,l)\in \mathrm{Order}}
\exp\!\Big(
\frac{
\langle x_k,x_l\rangle - \langle x_i,x_j\rangle
}{\tau}
\Big)
\Bigg),

$$

其中 $\mathrm{Order}=\{(i,j,k,l):\mathrm{sim}(i,j)>\mathrm{sim}(k,l)\}$，$\mathrm{sim}$ 为标注相似度，$\langle\cdot,\cdot\rangle$ 为余弦。

监督集约 **1060 万** 对量级；检索每 query **7** 个负例；MRL 维集合 $\{256,512,1024,1536,2048,3072,3584\}$。

---

## 5. Soft-mask：因果 → 双向的连续桥

### 5.1 直接切换的两个问题

1. **上三角注意力权重**在因果训练中从未生效，突然双向会经历「从零学习」；
2. 因果 mask **满秩**，双向 mask **秩为 1**，突变易陷入低秩局部最优。

### 5.2 调度与矩阵定义

令训练步 $t$，总步 $\tau$，线性调度：

$$
\alpha(t)=\frac{t}{\tau}.

$$

软掩码（$l$ 为序列长度）：

$$
M_{ij}(t)
=
\begin{cases}
1 & \text{if } i \ge j \\
\min\!\big(\alpha(t)\cdot \tfrac{l}{i},\, 1\big) & \text{if } i < j
\end{cases}

$$

解读：

- 下三角（含对角）保持 1，保留「看过去」；
- 上三角从 0 **逐渐放大到 1**，且 **越靠前的列越早饱和到 1**——秩从接近满秩 **逐步降到双向的低秩结构**，同时符合「从前向后阅读、远期权重渐开」的直觉。

消融三种 $\alpha(t)$（仅 soft-mask、无其它组件）：**线性最好**；加速/减速二次曲线均更差。

### 5.3 与 LoRA / 全参的交互（关键表）

| Method | w/o SoftMask | w/ SoftMask |
| --- | --- | --- |
| LoRA $r=16$ | 72.18 | 72.12 |
| LoRA $r=32$ | 72.08 | 72.23 |
| LoRA $r=64$ | 71.83 | 72.40 |
| **Full FT** | **71.50** | **73.52** |

无 soft-mask 时：LoRA 优于全参，且更大 $r$ 反而变差。  
有 soft-mask 时：**$r$ 越大越好，全参最佳**——证明 soft-mask 真正弥合了优化地貌，使嵌入全量微调可行。

相对 LLM2Vec「一步改 Bi + MNTP」、gte-Qwen2「训练期直接 Bi」，Conan-v2 强调 **连续谱上的秩与权重课程**。

---

## 6. 跨语检索数据集（CLR）

### 6.1 构造

现有检索集往往单语。CLR：用 **Qwen2.5-7B** 只翻译 **query**，document 保持原语，从而得到「中→英」「西→英」等跨语检索对；覆盖约 **26 语**，约 **1000 万** 对。语言配比参考 MTEB：英 25%、中 12%、西 8%…（详见论文 Table 7）。

### 6.2 效果（嵌入分布）

在 **未参与训练** 的 Multilingual Amazon Reviews（英/日/德/法/中/西）上：无 CLR 时六语嵌入 **分团**；加 CLR 后进入 **统一流形**（论文 Figure 3）。MKQA（多语问→英答检索）上：

| Model | R@20 | R@100 | nDCG@10 |
| --- | --- | --- | --- |
| BM25 | 28.1 | 39.9 | 25.4 |
| M3-Embedding | 68.8 | 75.5 | 53.2 |
| **Conan-v2** | **72.5** | **80.2** | **59.1** |

相对最强基线 M3：R@20 **+3.6**，nDCG@10 **+5.7**。

### 6.3 局限（附录诚实分析）

- 高资源语占比高；中文在「多语→英」MKQA 上因映射关系特殊可能吃亏；
- 语族上靠近英语的 Germanic/Romance/Slavic 更强，阿拉伯语/韩语等偏低——**语言相似度 > 单纯数据量**；
- 数值 inconsistency（「3 个童话」vs「5 个童话」）仍是嵌入模型通病。

---

## 7. 动态难负例挖掘（DHNM）

### 7.1 动机（承继 Conan-v1）

静态 hard negatives（预处理用它模挖好）的问题：

1. 教师模型的「难」≠ 学生当前的「难」；
2. 训练中负例变易后仍占坑，梯度信号变弱。

v1（[arXiv:2408.15710](https://arxiv.org/abs/2408.15710)）提出训中按难度替换；v2 **加重判定并改为逐步检查**。

### 7.2 分数与替换规则

$$
S=\mathrm{cos}\big(f(q),\, f(p)\big).

$$

对第 $i$ 个难负例，记初始分 $S_0$、当前分 $S_i$：

$$
N_i
=
\begin{cases}
N_{i+1} & \text{if } S_0 < 0.4 \\
N_{i+1} & \text{if } 1.2\cdot S_i < S_0 \ \wedge\ S_i < 0.7 \\
N_i & \text{otherwise}
\end{cases}

$$

含义：

- 初始就太易（$S_0<0.4$）→ 直接换；
- 当前分相对初始「松了」且绝对值 $<0.7$ → 视为不再难，从池取 $N_{i+1}$；
- 否则保留。

v2 利用 **损失里已算好的 query–neg 相似度**，每步轻量缓存判定，**几乎零额外开销**（相对 v1 每 1k step 检查更细）。

谱系位置：DPR → ANCE（异步刷新语料索引）→ NV-Retriever（正例感知过滤假负）→ **Conan 动态池内替换**——见同目录《难负例挖掘工业实践.md》。

---

## 8. 消融：三组件协同

| SM | CLR | DHNM | Multi | Eng | Zh |
| --- | --- | --- | --- | --- | --- |
| ✔ | ✗ | ✗ | 61.73 | 70.41 | 70.99 |
| ✗ | ✔ | ✗ | 62.69 | 70.94 | 71.41 |
| ✗ | ✗ | ✔ | 61.81 | **71.50** | **72.09** |
| ✔ | ✔ | ✗ | 64.45 | 72.14 | 71.79 |
| ✔ | ✗ | ✔ | 63.03 | 72.78 | 72.44 |
| **✔** | **✔** | **✔** | **65.17** | **73.52** | **74.24** |

解读：

- 单挂 DHNM：单语最强之一；
- SM+CLR：多语跳升（+3.56 Multi vs 仅 SM）；
- 三者齐备：多语与中英 **同时** SOTA，消除两两组合的偏科。

---

## 9. MTEB 结果要点

### 9.1 主结果（论文 Table 1，约 2025-05）

| 模型 | EN Avg | ZH Avg |
| --- | --- | --- |
| e5-mistral-7b-instruct | 67.97 | 59.92 |
| gte-Qwen2-7B-instruct | 70.72 | 71.62 |
| jasper-en-v1 | 71.41 | — |
| gemini-embedding-exp-03-07 | 73.30 | — |
| **Conan-embedding-v2** | **73.52** | **74.24** |
| Conan-embedding-v1 | — | 72.50 |

英文强项：分类 **90.98**、检索 **66.24**、PairClass **92.35**；STS/Summ 相对略弱（作者归因为 STS 数据占比偏低）。中文全面领先开源对照（含 xiaobu、retrieve-zh、gte-Qwen2-7B）。

### 9.2 零样本设定

仅用 MSMARCO、NQ、XQuADRetrieval、FEVER、HotpotQA、MIRACL、MrTidy 等小部分集合时，Conan-v2 仍达 EN Avg **71.43**，高于 Linq-Embed-Mistral（69.80）等 7B 级——支撑「从零训 + soft-mask」的泛化叙事。

### 9.3 实用因素

| 模型 | Size (M) | Dim | Infer (min) | MRL | EN Avg |
| --- | --- | --- | --- | --- | --- |
| stella-en-1.5B-v5 | 1543 | 1536 | 5.54 | ✔ | 69.43 |
| gte-Qwen2-7B | 7613 | 3584 | 31.78 | ✗ | 70.72 |
| **Conan-v2** | **1503** | **3584** | **5.14** | **✔** | **73.52** |

同维 3584 下，浅而宽的 1.4B 在延迟上接近 1.5B 蒸馏系，分数更高，并支持 MRL 截维部署。

---

## 10. 训练资源（实现向）

| 阶段 | GPU | 时间量级 | 其它 |
| --- | --- | --- | --- |
| LLM PT | 64×Ascend 910B | ~219 h | AdamW $1\mathrm{e}{-4}$，bs 256 |
| LLM SFT | 16×910B | ~38 h | LR $2\mathrm{e}{-5}$，bs 64 |
| Emb weak | 16×910B | ~97 h | soft-mask + InfoNCE |
| Emb sup | 16×910B | ~13 h | MRL 多维；retr bs 4 / STS bs 32 |

混合精度 + DeepSpeed ZeRO-1。最大长度 32768。

---

## 11. 与相关工作对照

| 维度 | LLM2Vec | gte-Qwen2 | Conan-v2 |
| --- | --- | --- | --- |
| 底座来源 | 现成 Decoder | Qwen2 | **从零训** |
| 双向获得 | 一步 Bi + MNTP | 训练期 Bi | **Soft-mask 课程** |
| 负例 | 静态 hard（监督） | 多阶段挖掘 | **DHNM 训中替换** |
| 多语 | 英文主 | 强 | **CLR 跨语检索** |
| 参数 | 1.3B–8B | 1.5B/7B | **1.4B 浅宽** |
| 适配 | LoRA 友好 | 全量对比 | soft-mask 下 **全参更优** |

v1→v2：从 BERT 系「更多更好负例」扩展到 **嵌入专用 LLM + 掩码课程 + 跨语**。

---

## 12. 实现对照清单

```text
1. 设计 8L×3584d LLM；多语 150K vocab；PT 混新闻/QA
2. SFT：pair → instruction 格式
3. 弱监督：同 pair + gte-Qwen2 滤分 + soft-mask InfoNCE
4. 监督：Retr/CLR/Cls InfoNCE + STS CoSENT；MRL 多维
5. DHNM：按 S0/Si 规则逐步换负例
6. 评 MTEB EN / C-MTEB / MKQA
```

可验证目标：

1. Table 6：无 soft-mask 时全参 < LoRA；有则全参最佳；
2. Table 4：三组件齐备才多语+单语双赢；
3. Figure 3：CLR 前后多语分布由分团→融合；
4. 延迟：同维下相对 7B 深模型有数量级优势。

---

## 13. 小结

Conan-embedding-v2 把嵌入模型重新定义为 **原生嵌入 LLM**，而非生成 LLM 的后贴适配：

1. **数据**：预训练即配对化，再加 CLR 跨语检索；
2. **优化**：soft-mask 填平因果 LM 与双向对比的秩/权重鸿沟，解锁全参微调；
3. **负例**：DHNM 让难负例随学生进化；
4. **产品形态**：1.4B、3584-d、32K、MRL——在 2025-05 口径下冲击中英 MTEB SOTA，同时保持可部署延迟。

同目录对照：《LLM2Vec详解.md》（低成本改现成 Decoder）、《gte-Qwen2详解.md》（宽而深的 Qwen2 教师）、《难负例挖掘工业实践.md》（DHNM 工业位置）、《E5详解.md》（对比学习公共祖先）。

---

## 参考

1. Li et al. (2025). Conan-Embedding-v2. [arXiv:2509.12892](https://arxiv.org/abs/2509.12892) / EMNLP 2025
2. Li et al. (2024). Conan-embedding: General Text Embedding with More and Better Negative Samples. [arXiv:2408.15710](https://arxiv.org/abs/2408.15710)
3. Li et al. (2023). GTE multi-stage contrastive learning. [arXiv:2308.03281](https://arxiv.org/abs/2308.03281)
4. BehnamGhader et al. (2024). LLM2Vec. [arXiv:2404.05961](https://arxiv.org/abs/2404.05961)
5. Su (2022). CoSENT; Gao et al. SimCSE; Muennighoff et al. MTEB
