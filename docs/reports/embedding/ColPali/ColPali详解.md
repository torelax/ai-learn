# ColPali：论文精炼研读

> 直接将文档页作为图像输入 VLM，保留 patch 多向量，与文本查询做 MaxSim 匹配，减少 OCR 与版面解析链路。
>
> ICLR 2025 · [原论文](https://arxiv.org/abs/2407.01449v6) · [补充解读](ColPali解读.md)

## 摘要

视觉丰富的 PDF 不只包含可抽取文字，表格结构、图表和版面也携带检索证据。ColPali 用 PaliGemma 将页面直接编码为低维多向量，让查询 token 在页面 patch 中寻找匹配，并建立 ViDoRe 基准检验页级检索。

约 11.9 万英语 query–page 对的 LoRA 训练，使模型在 ViDoRe 上达到约 81.3 的平均 nDCG@5，显著高于论文的 OCR、caption 和单向量基线。本文同时测量索引吞吐、查询延迟、token pooling 与分辨率消融，说明简化解析流程有价值，但每页大量向量、细字分辨率与跨页信息仍是边界。

## 引言

![原论文图 1：ColPali 将传统 PDF 解析链替换为页面视觉编码及 late interaction。](figs/S1-F1-01.webp)


### 工业文档检索的真实瓶颈

RAG / 站内搜索里，embedding 模型本身往往不是第一瓶颈；**PDF 入库流水线**才是：

1. PDF 解析或 OCR 抽字；
2. Layout 检测（标题/段落/表/图）；
3. Chunk 策略；
4. 可选：对图/表跑 VLM caption，再当文本嵌入。

论文实验（Table 2）：**优化入库** 往往比换一个更强文本 embedding 更能涨点；但 OCR+Captioning **极慢、脆、难端到端训练**。

### 现有评测缺口

- 文本检索榜（BEIR、MTEB）：passage 文本，不测版面/图/表；
- 图文对比检索：自然图为主，不是扫描件/财报/信息图。

→ 需要 **页级、视觉丰富、多域** 的文档检索基准：**ViDoRe**。

### 设计目标（R1–R3）

| 编号 | 要求 | 含义 |
| ------ | ------ | ------ |
| R1 | 强检索质量 | nDCG / Recall / MRR |
| R2 | 低在线延迟 | query 编码 + 匹配要快 |
| R3 | 高索引吞吐 | 页/秒；少依赖多模型串联 |

ColPali 用「整页图像 → 一次 VLM 前向 → 多向量索引」同时服务 R1–R3。

## 问题定义与相关工作

### 文本检索

BM25 和稠密文本编码器从已抽取的文本检索；对 PDF 还需解析、切块与页级汇总。这类方案容易丢掉表格布局、图表与视觉强调。

### 整合视觉特征

OCR 能补充图片里的文字，VLM caption 能把视觉元素转成文本，但多阶段处理增加入库成本。CLIP 类模型直接对齐图片和文本，却通常围绕自然图片及全局单向量训练；ColPali 研究以文档页为对象的细粒度视觉检索。

## ViDoRe 基准

### 基准设计

#### 任务定义

**页级检索**：给定 query，系统是否把正确 **文档页** 排到前面。原子检索单元 = 单页（论文中 document 即 page）。

指标主报 **nDCG@5**；另有 Recall@K、MRR；并报告 query 延迟与索引吞吐。

#### 数据集构成

| 类型 | 数据集 | 语言 | #Q / #Docs | 焦点 |
| ------ | -------- | ------ | ------------ | ------ |
| Academic | DocVQA | EN | 500 / 500 | 扫描件 |
| Academic | InfoVQA | EN | 500 / 500 | 信息图 |
| Academic | TAT-DQA | EN | 1600 / 1600 | 财报 |
| Academic | arXivQA | EN | 500 / 500 | 科学图 |
| Academic | TabFQuAD | FR | 210 / 210 | 工业表（本文发布） |
| Practical | Energy / Gov / Health / AI | EN | 100 / 1000 | 主题 PDF + Claude 生成问句 |
| Practical | Shift Project | FR | 100 / 1000 | 环境报告 |

Academic：由 VQA 三元组改写——问题当 query，所在页当正例。
Practical：公开 PDF + Claude-3 Sonnet 生成问句，人工过滤；主题收窄以制造 **硬负例（近邻文档多）**。

训练参考集同步开源（`vidore/colpali_train_set`），约 **118,695** 对：学术 train **63%** + 网页 PDF 合成问句 **37%**；**全英**，用于测零样本法语等。

### 现有系统评估

1. **Unstructured 文本流水线**：layout + OCR + by-title chunk；可选丢掉视觉元素；
2. **+ OCR 视觉元素** / **+ Captioning（Claude-3）**；
3. 文本嵌入：BM25、**BGE-M3**；页分 = chunk 分 max-pool；
4. 对比 VLM：Jina-CLIP、Nomic Vision、SigLIP。

观察：Captioning/OCR 抬点但索引极慢；纯对比 VLM 在文档页上明显偏弱——**自然图对比预训练 ≠ 文档检索**。

## 基于后期交互的视觉检索

### 架构

#### 骨干选择：PaliGemma-3B

- 视觉：SigLIP-So400m/14 patch 嵌入；
- 语言：Gemma-2B；patch 投影进 LLM 输入空间；
- 前缀（指令+图像）上 **full-block attention**，利于文档理解；
- 体积相对小，有多分辨率 checkpoint。

ColPali 在 LLM 输出上加 **线性投影** → $D=128$（对齐 ColBERT 惯例），得到轻量 bag-of-embeddings。

#### 文档侧（页图像）

页渲染为图像 → ViT patches（论文 448×448 设置产生 **1024** patches）→ 经 LLM 上下文 → 每 token 一向量 → 投影到 128-d。
另含少量文本前缀（如 “Describe the image.” 一类），与 patch 一并进入序列。

#### Query 侧

文本 query 走同一语言模型路径，得 $N_q$ 个向量；**Query augmentation**：追加 **5** 个特殊 token（同 ColBERT 软扩展思想）。

#### Late Interaction（论文 Eq.1）

$$
\mathrm{LI}(q,d)=\sum_{i\in[1,N_q]}\max_{j\in[1,N_d]}\big\langle E_q^{(i)}\,\big|\,E_d^{(j)}\big\rangle.

\tag{1}
$$

与 ColBERT Eq.3 **同构**：

$$
S_{q,d}^{\mathrm{ColBERT}}=\sum_i\max_j E_{q_i}\cdot E_{d_j}^{\mathsf{T}}.

$$

差别只在 $E_d$ 的来源：

| | ColBERT | ColPali |
| -- | --------- | --------- |
| $E_d$ | 文本 WordPiece 上下文向量 | **图像 patch（经 VLM）** 向量 |
| $E_q$ | 文本 | 文本（可共享 VLM 的语言塔） |
| 索引对象 | passage 文本 | **页截图** |
| OCR | 不需要（文本已给） | **刻意不依赖** |

对齐空间来自 VLM 多模态训练：图像 patch 与文本 token 已映射到同一 LLM 隐空间，再通过检索微调建立适合 MaxSim 的 token 级对齐；仅共享隐空间并不足以保证有效检索。

#### 对比损失（论文 Eq.2）

batch $\{(q_k,d_k)\}_{k=1}^{b}$，正分 $s_k^{+}=\mathrm{LI}(q_k,d_k)$， hardest in-batch 负分

$$
s_k^{-}=\max_{l\neq k}\mathrm{LI}(q_k,d_l).

$$

$$
\mathcal{L}=\frac{1}{b}\sum_{k=1}^{b}\log\big(1+\exp(s_k^{-}-s_k^{+})\big)
=\frac{1}{b}\sum_k\mathrm{softplus}(s_k^{-}-s_k^{+}).

\tag{2}
$$

即 pairwise CE 的 softplus 稳定写法（与原 ColBERT pairwise softmax CE 同族）。全 late interaction **可微**，可端到端反传。

### 模型训练

| 项 | 设定 |
| ---- | ------ |
| 数据 | 118,695 英 query–page；2% 作验证 |
| Epoch | 1 |
| 精度 | bfloat16 |
| 适配 | **LoRA** $\alpha=32,\ r=32$ 于 LM Transformer + 随机初始化投影层 |
| 优化器 | paged_adamw_8bit |
| LR | $5\times 10^{-5}$，线性 decay，2.5% warmup |
| Batch | 32（8 GPU 数据并行） |
| Query aug | +5 tokens |

**不**默认解冻视觉编码器：解冻 ViT 时平均 nDCG@5 **略降 0.7**（数据规模相对 SigLIP 预训练太小）。

合成数据注意：ViDoRe 与 train 禁止同一多页 PDF 泄漏。

## 结果

### 检索性能

| 方法 | Avg |
| ------ | ----- |
| Unstructured+OCR + BGE-M3 | 66.1 |
| Unstructured+Captioning + BGE-M3 | 67.0 |
| SigLIP Vanilla | 51.4 |
| BiSigLIP（文档数据微调） | 58.6 |
| BiPali（LLM 单向量池化） | 58.8 |
| **ColPali（+ Late Interaction）** | **81.3** |

分任务上 ColPali 全面领先；在 **InfoVQA / arXivQA / TabFQuAD** 等视觉密集任务上与文本流水线差距尤其大。文本偏重页（如部分 Practical）上 ColPali 仍更高——版面与局部强调也有信号。

#### 从单向量到多向量

1. **任务数据**：BiSigLIP 相对 Vanilla SigLIP 全面提升（图/表尤甚）；
2. **+LLM**：BiPali 英语略纠结，但 **法语零样本** 得益于 Gemma 多语；
3. **+Late interaction**：相对 BiPali 平均 **+22.5** nDCG@5 —— **多向量是跃迁主因**。

负向结果（重要）：

- **ColSigLIP**（在 SigLIP 上直接 late interaction）表现很差：SigLIP 对比损失只对齐 **池化** 向量，不优化 patch/token 级几何；
- 错配「SigLIP 图像向量 + PaliGemma 文本向量」同样差：多模态对齐必须来自 **同一 VLM 隐空间**。

### 延迟与存储占用

![原论文图 2：离线入库步骤与耗时比较；收益包含省去 OCR 和 caption 的成本，不只是神经编码器吞吐。](figs/S3-F2-01.png)


| 阶段 | 观察 |
| ------ | ------ |
| 在线 query | ColPali LM 编码约 **30 ms** / 15 token 级 query（L4，bs=1）；BGE 约 22 ms |
| MaxSim | 小库开销约 **1 ms / 1k pages**；大规模需 PLAID / fast-plaid 等引擎 |
| 离线索引 | **跳过 OCR/layout/caption**，整页一次前向；吞吐显著高于 Unstructured+Captioning（论文 Figure 2） |
| 存储 | $D=128$ 时约 **257.5 KB/页**（含 6 个文本前缀向量）；可用 ColBERTv2 式压缩或 token pooling |

**Token pooling**（层次均值）：pool factor=3 时向量数 **-66.7%**，保留约 **97.8%** 性能；**Shift（密文本）** 更敏感——冗余白边少，乱池易伤。

### 可解释性

![原论文图 3 右：查询 token 对不同页面区域的匹配热力图展示局部证据。](figs/S5-F3-02.webp)


将每 query token 的 MaxSim 热力叠回原图：可高亮对「hour/hours」等词响应的轴标签与数字区域——展示无需外部 OCR 的文字区域匹配能力，无需显式识别流水线。

## 消融研究

| 实验 | 结论 |
| ------ | ------ |
| 低/高分辨率设置（原文称 512 vs 1024 patches） | 512 掉约 **24.8** nDCG@5，省内存 |
| ColIdefics2-8B（64 resampled tokens） | 强于 ColPali-512，仍略低于 ColPali-1024；更慢 |
| 去掉 query aug | 英语几乎不变；**法语任务上升**（Shift +9.8，TabF +6.3） |
| 全 in-batch softmax vs hardest pairwise | hardest softplus 更好约 **+1.6** |
| 加 1552 法语表样本 | TabFQuAD +2.6 nDCG@5 / +5 R@1，其它不掉 |
| ColQwen2-VL（同数据策略，≤768 patches） | 比 ColPali **+5.3** ，支持该对照中骨干能力改善能迁移到检索 |
| 仅 DocMatix 子集训练 | ViDoRe 仅 **-2.2**，仍大幅超文本流水线 ，支持该设置下的域外泛化 |

## 结论

ColPali 用整页视觉编码与 token—patch 后期交互，将文档页面中的文字、表格和图表统一到可训练的检索流程。ViDoRe 显示这一组合优于论文中的文本解析流水线及全局图文向量，并减少离线 OCR、布局和描述生成步骤。查询仍需 VLM 编码，多向量存储及高分辨率开销并未消失。未来工作包括更好的数据和骨干、更强评测，以及与视觉问答结合的端到端 RAG。

## 可复现性与附录

作者公开数据、适配器、代码和评测。附录给出各任务构成、合成数据、训练超参、存储与计时口径，以及完整逐数据集结果；解释主表时需同时保留英语训练、页级检索及多页 PDF 去重的条件。

## 参考资料

1. Faysse, Sibille, Wu, et al. *ColPali: Efficient Document Retrieval with Vision Language Models*. [arXiv:2407.01449](https://arxiv.org/abs/2407.01449), 2024.
2. Khattab & Zaharia. *ColBERT*. [arXiv:2004.12832](https://arxiv.org/abs/2004.12832), 2020.
3. Santhanam et al. *ColBERTv2*. [arXiv:2112.01488](https://arxiv.org/abs/2112.01488), 2021.
4. Beyer et al. *PaliGemma*. 2024；Alabdulmohsin et al. *SigLIP*. 2023.
5. HF: https://huggingface.co/vidore ；代码：https://github.com/illuin-tech/colpali
6. Clavié et al. token pooling；Santhanam et al. PLAID —— 多向量检索工程相关。
