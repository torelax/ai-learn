# Jina Embeddings v4 技术详解

> 基于论文 [jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval](https://arxiv.org/abs/2506.18902)（Günther, Sturua, Akram, Mohr et al., Jina AI；arXiv:2506.18902）。
> 前序：[v3 Task LoRA / MRL](https://arxiv.org/abs/2409.10173)；底座 [Qwen2.5-VL-3B-Instruct](https://arxiv.org/abs/2502.13923)。
> 本文把 **统一 VLM 通路、单向量 + late interaction 联合训练、KL 对齐、Task LoRA、Jina-VDR、模态间隙分析** 写全。

---

## 1. 一句话定位

**jina-embeddings-v4** 是基于 **Qwen2.5-VL-3B** 的 **3.8B 多模态多语嵌入**：同一套权重同时输出 **可截断单向量（2048→128）** 与 **ColBERT 式多向量（每 token 128-d）**，用 **Task LoRA** 覆盖检索 / STS / 代码，并在 **视觉富文档**（表格、图表、扫描件、混合版式）上引入基准 **Jina-VDR**。


| 项 | 内容 |
| --- | --- |
| 骨干 | **Qwen2.5-VL-3B-Instruct**（3.8B）；训练中 **冻结**，只训 LoRA + 投影 |
| 上下文 | 文本至 **32768**；图像统一缩放到 **20MP** |
| 单向量 | **2048-d**，MRL 可截到 **128**；mean pooling |
| 多向量 | **128-d / token**（含「图像 token」）；late interaction |
| LoRA | 三套各 **~60M**（retrieval / text-matching / code），<2% 显存税 |
| 亮点 | 缩小 CLIP 式 **modality gap**；J-VDR / ViDoRe SOTA（dense + late） |

相对 v3：从「多语文本 Task LoRA」升级为「**图文统一空间 + 双输出范式**」；相对 ColPali / DSE：同一模型覆盖文本、代码、截图检索，而不只做视觉文档 QA。

---

## 2. 问题背景

### 2.1 专业化的碎片化

生态里同时存在：文本嵌入、CLIP 图文、代码嵌入、视觉文档（ColPali）模型。生产多模型并存 → 多套索引、多套运维、跨模态分数不可比。v4 目标：**一个底座投影到统一语义空间**。

### 2.2 Dual encoder 的 modality gap

CLIP 式双塔：图、文分塔对比学习后，**跨模态正对齐常弱于同模态弱对齐**（Liang et al.）。VLM（图→视觉 token 序列 → 同一 LLM）共享编码器，归纳偏置更利于 **共享区域**（Eslami & de Melo）。v4 明确走后者。

### 2.3 单向量 vs Late interaction

| | 单向量 | Late interaction（ColBERT / ColPali） |
| -- | --- | --- |
| 存储 | 低 | 高（每 token 一向量） |
| 精度 | 通常较低 | 通常较高 |
| 打分 | 余弦 / 点积 | MaxSim 求和 |

v4 **联合训练两种输出**，部署按延迟/精度切换。

---

## 3. 模型架构

### 3.1 统一多模态通路

```text
文本 ──tokenize──► token 嵌入 ──┐
                            ├─► Qwen2.5-VL LM（上下文注意力）──► 双输出头
图像 ──► Vision Encoder ──►「图像 token」序列 ──┘
         (+ Task LoRA 注入注意力 / 线性层)
```

与 CLIP 差异：视觉塔 **不是** 输出最终嵌入，而是把图像变成 LLM 可消费的 token 序列；可与文本 **同场** 进解码器。真正「单输入场多模态」。

### 3.2 双输出

1. **Dense**：最后一层 mean pool → 2048-d；MRL 训练后可截断。
2. **Multi-vector**：额外投影层 → 每输入 token（含图像 token）一个 **128-d** 向量，对齐 ColBERT / ColPali 用法。

### 3.3 Late interaction 打分

对 query 多向量 $q=\{\mathbf{q}_i\}_{i=1}^{n}$、文档 $p=\{\mathbf{p}_j\}_{j=1}^{m}$：

$$
s_{\mathrm{late}}(q,p)
=
\sum_{i=1}^{n}
\max_{j\in\{1,\ldots,m\}}
\mathbf{q}_i\cdot\mathbf{p}_j^{\mathsf{T}}.
\tag{1}
$$

每个 query token 找文档中最匹配 token（MaxSim），再求和。训练时需归一化分数（见 §5）；在线检索 query 固定，可不除。

### 3.4 Task LoRA（三套）

继承 v3 思想，收敛为三类：

| Task | 语义 |
| --- | --- |
| `retrieval` | 非对称 query–document（**前缀**区分角色，而非 v3 的双 adapter） |
| `text-matching` | STS / 对称检索 |
| `code` | NL↔code、code↔code、技术 QA |

每套 ~60M；可热切换；均支持图与文。v3 消融显示「双 adapter + 前缀」收益有限，故 v4 **检索侧只保留前缀法**。

---

## 4. 训练方法

### 4.0 总原则

- 初始化：`Qwen/Qwen2.5-VL-3B-Instruct`；
- **骨干冻结**；随机初始化 multi-vector 投影与 LoRA；
- 两阶段：**(1) 单一 LoRA 配对联合训练** → **(2) 复制为三套任务 LoRA 精调**；
- 全程对单向量施加 **MRL**。

### 4.1 Stage 1：配对联合训练（Dense + Late）

每 step 两个 batch：

- $\mathcal{B}_{\mathrm{text}}$：文本对；
- $\mathcal{B}_{\mathrm{multi}}$：文本–图像对。

分别算 dense 相似度矩阵 $\mathbf{S}_{\mathrm{dense}}$（余弦）与 late 矩阵 $\mathbf{S}_{\mathrm{late}}$。训练用归一化 late 分数：

$$
s'_{\mathrm{late}}(q_i,p_j)
=
\frac{s_{\mathrm{late}}(q_i,p_j)}{t},
\tag{2}
$$

$t=|q_i|$（query token 数），使尺度可进 InfoNCE。

Softmax / InfoNCE（Eq.3–4）：

$$
\mathrm{softmax}(\mathbf{S},\tau,i,j)
:=
\ln
\frac{e^{s_{i,j}/\tau}}
{\sum_{k=0}^{n} e^{s_{i,k}/\tau}},
\tag{3}
$$

$$
\mathcal{L}_{\mathrm{NCE}}(\mathbf{S}(\mathcal{B}),\tau)
:=
-\sum_{i=0}^{n}
\mathrm{softmax}(\mathbf{S}(\mathcal{B}),\tau,i,i).
\tag{4}
$$

Late 通常比 dense **更易拟合、误差更小**。为联合训练，加入 **KL 蒸馏式对齐**（Hinton et al. 知识蒸馏思想）：

$$
\mathcal{L}_{D}(\mathcal{B},\tau)
:=
D_{\mathrm{KL}}\!\left(
\mathbf{S}'_{\mathrm{dense}}(\mathcal{B})\,\middle\|\,
\mathbf{S}'_{\mathrm{late}}(\mathcal{B})
\right),
\quad
\mathbf{S}'_{i,j}=\mathrm{softmax}(\mathbf{S},\tau,i,j).
\tag{5}
$$

联合损失（Eq.6）：

$$
\begin{aligned}
\mathcal{L}_{\mathrm{joint}}(\mathcal{B}_{\mathrm{txt}},\mathcal{B}_{\mathrm{multi}},\tau)
&=
w_1\mathcal{L}_{\mathrm{NCE}}(\mathbf{S}_{\mathrm{dense}}(\mathcal{B}_{\mathrm{txt}}),\tau)
+
w_2\mathcal{L}_{\mathrm{NCE}}(\mathbf{S}_{\mathrm{late}}(\mathcal{B}_{\mathrm{txt}}),\tau)
+
w_3\mathcal{L}_{D}(\mathcal{B}_{\mathrm{txt}})
\\
&\quad+
w_4\mathcal{L}_{\mathrm{NCE}}(\mathbf{S}_{\mathrm{dense}}(\mathcal{B}_{\mathrm{multi}}),\tau)
+
w_5\mathcal{L}_{\mathrm{NCE}}(\mathbf{S}_{\mathrm{late}}(\mathcal{B}_{\mathrm{multi}}),\tau)
+
w_6\mathcal{L}_{D}(\mathcal{B}_{\mathrm{multi}}).
\end{aligned}
\tag{6}
$$

$w_{1\ldots6}$ 与 $\tau$ 为超参。直觉：

- 文本 / 多模态各有 dense + late 对比项；
- KL 迫使 **单向量相似度结构追随更强的 late 结构**，缓解「双头互抢」。

#### 配对数据

>300 源。文本对：沿用 v3 过滤。图文对刻意 **不止 caption**：网站截图、渲染 Markdown、图表、表格、「野生」版式；query 含问题、关键词、长描述、事实陈述——直接服务视觉文档检索，而非仅自然图像 captioning。

### 4.2 Stage 2：任务 LoRA

复制 Stage-1 LoRA → 三套独立精调。

#### 4.2.1 Retrieval → $\mathcal{L}_{\mathrm{NCE}+}$

Hard negatives 三元组；batch 内其它正例亦作负例。扩展 InfoNCE：

$$
\mathcal{L}_{\mathrm{NCE}+}(\mathbf{S}(\mathcal{B}),\tau)
:=
\sum_{r\in\mathcal{B}}
\Bigg[
-\ln
\frac{e^{s(q,p)/\tau}}
{\sum_{i=1}^{k}\Big[
e^{s(q,p_i)/\tau}
+\sum_{j=1}^{m}e^{s(q,n_{j,i})/\tau}
\Big]}
\Bigg],
$$

$r=(q,p,n_1,\ldots,n_m)$。嵌入联合损失框架，替换对应 NCE 项。文本难负类似 v3；多模态难负：Wiki-SS、VDR-multilingual、自挖掘。

角色区分：**input prefix**（如 query / passage），不用第二套 adapter。

#### 4.2.2 Text-matching → CoSENT + InfoNCE

有连续相似度标注时用 CoSENT：

$$
\mathcal{L}_{\mathrm{co}}(\mathbf{S}(\mathcal{B}),\tau)
:=
\ln\Bigg[
1+\sum_{\substack{(q_1,p_1),(q_2,p_2)\\ \zeta(q_1,p_1)>\zeta(q_2,p_2)}}
\frac{e^{s(q_2,p_2)}-e^{s(q_1,p_1)}}{\tau}
\Bigg].
$$

无标注对回退 Stage-1 InfoNCE。数据：STS12、SICK 等。

#### 4.2.3 Code → 三元组 NCE+

面向 NL→code、code→code、技术 QA。**不更新视觉部分**（代码路径纯文本）。底座已见过 StackExchangeQA / CodeSearchNet；LoRA 数据含 CodeSearchNet、CodeFeedback、APPS、CornStack。前缀与 $\tau=0.02$ 与检索三元组设定对齐。

---

## 5. Jina-VDR 基准

动机：ViDoRe 偏 **英/法 QA + PDF 图表页**；真实视觉检索还有地图、广告、扫描历史档案、多语 README 截图、非问句 query 等。

Jina-VDR = ViDoRe + **约 30** 项扩展：

1. **改造**：DonutVQA、TableVQA、MPMQA、CharXiv、PlotQA 等 → 检索对；  
2. **人工**：Stanford 课件、TQA 教材图、Jina Yearbook、日文拉面手册、上海总体规划等；  
3. **合成**：Europeana 多语扫描、Hindi gov、俄语饮料目录、TweetStock 图、Airbnb 表（多语模板）等；  
4. **非 QA query**：GitHub README 渲染 ↔ 多语描述；Wikimedia 地图 ↔ 文本描述。

设计要点：LLM 过滤保证 query 像真实信息需求；多语可达 ~20 语种；与 ViDoRe 协议兼容。集合发布于 Hugging Face `jinaai/jinavdr-*`。

---

## 6. 评测（Table 3 级摘要）


| Model | J-VDR | ViDoRe | CLIPB | MMTEB-RT | MTEB-en-RT | COIR | LEMB | STS-m | STS-en |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **v4 dense** | **73.98** | **84.11** | **84.11** | **66.49** | 55.97 | 71.59 | 67.11 | 72.70 | **85.89** |
| **v4 late** | **80.55** | **90.17** | — | — | — | — | — | — | — |
| v3 | 47.82 | 26.02 | — | 58.58 | 54.33 | 55.07 | 55.66 | 75.77 | 85.82 |
| text-embedding-3-large | — | — | — | 59.27 | **57.98** | 62.36 | 52.42 | 70.17 | 81.44 |
| colpali-v1.2 (late) | 63.80 | 83.90 | — | — | — | — | — | — | — |
| dse-qwen2-2b-mrl (dense) | 67.25 | 85.80 | — | — | — | — | — | — | — |
| voyage-code | — | — | — | — | — | **77.33** | — | — | — |

指标：J-VDR / ViDoRe / CLIPB 用 nDCG@5；文本检索类 nDCG@10；STS 为 Spearman。文本模型在 J-VDR 上经 EasyOCR 转文字后评——凸显 **像素级文档理解** 相对 OCR 管线的优势。

解读：

- **视觉文档**：dense 已超 ColPali / DSE 平均；late 再拉一截；
- **纯文本检索**：与 SOTA 同档，英检略低于 text-embedding-3-large / gemini-embedding，多语 MMTEB 很强；
- **长文 LEMB**：67.11，显著高于 v3，仅次于部分 voyage-3；
- **代码 COIR**：71.59，逊于专用 voyage-code（77.33），但远超 v2-code / 通用 v3；
- **STS**：英文最优档之一。

评测时多数检索用 retrieval adapter；ArguAna 等对称任务用 text-matching，并可加任务前缀（如反驳检索指令）。

---

## 7. 嵌入空间分析

### 7.1 Modality gap

Flickr8K 上：匹配 **图–文** 余弦分布 vs 匹配 **文–文**。CLIP / jina-clip-v2 跨模态峰明显左偏（间隙大）；**v4 两峰接近**——共享 LLM 通路压缩间隙。

### 7.2 Cross-modal alignment

定义：匹配图文对的平均余弦。1K 采样：


| Model | Flickr30K | MSCOCO | CIFAR-100 |
| --- | --- | --- | --- |
| OpenAI CLIP | 0.15 | 0.14 | 0.20 |
| jina-clip-v2 | 0.38 | 0.37 | 0.32 |
| **jina-embeddings-v4** | **0.71** | **0.72** | **0.56** |

CIFAR 标签过短，对齐弱于描述性 caption——符合预期。

### 7.3 Cone effect

对比学习易把跨模态匹配拉成「锥」：正负对相似度差偏小。v4 正/负图文相似度分布峰分离远大于 CLIP / jina-clip-v2 → **更充分利用球面、图文重叠更好**。

---

## 8. 局限

1. **3.8B + 多向量**：索引与 GPU 成本远高于 v3 570M；late 存储按 token 线性涨。
2. **骨干冻结**：表达力天花板受 Qwen2.5-VL-3B 预训练分布约束；低资源语在 Crossmodal3600 上可能输给覆盖更广的 NLLB-SigLIP。
3. **代码非 SOTA**：专用 code 模型仍领先；code LoRA 不改视觉，多模态代码场景未深挖。
4. **MRL 下限 128**：相对 v3 可到 32，极致压缩场景需另策。
5. **Jina-VDR 含合成 query**：泛化到完全分布外的版式/语言需持续监测。
6. **英检 MTEB 非全面第一**：通用「万能」换来局部榜单让步——选模型仍要看主任务。

---

## 9. 谱系笔记

```text
v1  T5 · 数据清洗 · 双向 InfoNCE
v2  ALiBi BERT · 8K 英文
v3  XLM-R+RoPE · Task LoRA×5 · MRL · 多语 8K
v4  Qwen2.5-VL-3B · 冻结骨干+LoRA×3 · Dense∥Late 联合+KL · 32K · Jina-VDR
```

继承线：

- **对比学习骨架**：InfoNCE → NCE+ → CoSENT（v1–v4 一脉）；
- **任务特化**：v3 五 LoRA → v4 三 LoRA + 前缀；
- **长上下文**：ALiBi → RoPE → VLM 原生 32K；
- **新维度**：单/多向量共训、视觉富文档、模态间隙显式分析。

与 ColPali / ColQwen 族比较：v4 不是「只做截图 MaxSim」，而是 **产品向统一嵌入**；late 是可选项而非唯一形态。

---

## 10. 公式速查

| 公式 | 含义 |
| --- | --- |
| Eq.1 $s_{\mathrm{late}}$ | MaxSim 晚交互打分 |
| Eq.2 $s'_{\mathrm{late}}$ | 训练用按 query 长度归一化 |
| Eq.3–4 InfoNCE | Dense / Late 对比 |
| Eq.5 $\mathcal{L}_D$ | Dense←Late 的 KL 对齐 |
| Eq.6 $\mathcal{L}_{\mathrm{joint}}$ | 文本+多模态、双头加权总损失 |
| $\mathcal{L}_{\mathrm{NCE}+}$ | 检索硬负例 |
| $\mathcal{L}_{\mathrm{co}}$ | STS CoSENT |
| MRL | 单向量前缀维可截断 |

---

## 11. 小结

jina-embeddings-v4 用 **冻结的 Qwen2.5-VL-3B + 轻量 LoRA**，把文本、图像、视觉文档与代码拉进 **同一套可切换任务子空间**，并首次在 Jina 系列里把 **单向量与 late interaction 用 KL 联合对齐**。benchmark 上视觉富文档与多语检索是最大收益；代价是 3.8B 级推理与多向量索引成本。若主场景是 PDF/截图/图表检索且需要与文本库统一，v4（尤其 late）是相对 ColPali 单点方案更「平台化」的选择；若只要廉价多语文本，v3 仍更划算。
