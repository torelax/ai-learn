# 图文 Embedding 模型技术综述

> 图文（image-text）多模态嵌入模型的分类、发展脉络与关键技术 · 截至 2026 年 1 月

> 图文 Embedding 的目标是把图像、文本（乃至视频、视觉文档）映射到**同一个语义向量空间**，
> 使得语义相近的内容向量距离更近，从而支撑跨模态检索、分类、视觉问答、RAG 等下游任务。
> 按技术路线可以分为四大类，它们在时间上大致**依次演进又相互交叠**：
> 从双塔对比学习，到融合编码器，再到基于大模型（MLLM）的通用嵌入，以及面向视觉文档的多向量后交互路线。
> 作为补充，本综述在图文（跨模态）之外，另设两章分别调研**纯图片 Embedding**（第九章，图搜图 / 实例检索，以 DINO 系为代表）
> 与**纯文本 Embedding**（第十章，text-text 语义检索 / RAG，以 BGE、Qwen3-Embedding 系为代表），
> 它们不做跨模态对齐，各自在单一模态内构建向量空间。

## 一、主要分类

#### ① 双塔对比学习类

*Dual-Encoder / CLIP-style · 2021 起*

图像编码器与文本编码器完全独立，用对比学习把配对的图文拉近、非配对推远。 推理时两侧可分别离线编码，检索效率极高，是产业界最经典的范式。

代表：`CLIP` `ALIGN` `OpenCLIP` `SigLIP / SigLIP2` `EVA-CLIP` `Chinese-CLIP` `jina-clip-v2`

- 优点：速度快、可扩展、零样本能力强
- 短板：细粒度对齐弱，难处理复杂组合语义、长文本、指令

#### ② 融合编码器 / 跨模态类

*Fusion Encoder · 2021–2022*

在双塔基础上增加跨模态注意力层，让图文特征深度交互，配合多任务 （对比 ITC、匹配 ITM、掩码语言建模 MLM）联合训练，提升细粒度理解。

代表：`ALBEF` `BLIP / BLIP-2` `CoCa` `BEiT-3` `FLAVA`

- 优点：对齐更精细，理解与生成兼顾
- 短板：跨模态交互层使检索成本上升，难做超大规模索引

#### ③ 基于 MLLM 的通用嵌入类

*LLM/VLM-based Universal Embedding · 2024 起（当前主流）*

直接拿视觉语言大模型（Qwen-VL、LLaVA 等）当骨干，用对比学习微调成嵌入模型。 天然支持**指令感知**、任意图文交错输入，是目前 MMEB 榜单霸榜的路线。

代表：`E5-V` `VLM2Vec / V2` `GME` `BGE-VL` `Ops-MM` `RzenEmbed` `Seed1.6-Embedding` `Qwen3-VL-Embedding`

- 优点：语义与指令理解最强，统一处理文本/图/视频/文档
- 短板：模型大、推理成本高（需 MRL/量化优化落地）

#### ④ 视觉文档多向量后交互类

*ColPali-style Late Interaction · 2024 起*

面向 PDF、截图、图表等"视觉文档"，不压缩成单一向量，而是保留 patch 级多向量， 检索时用 MaxSim 做后交互（late interaction），在文档检索上效果拔尖。

代表：`ColPali` `ColQwen2.5 / 3` `ColNomic` `Jina-embeddings-v4` `llama-nemoretriever-colembed`

- 优点：视觉文档 / OCR 密集场景 SOTA
- 短板：多向量存储与检索开销大，属专用而非通用嵌入

> **说明**：这四类不是严格互斥的。近两年趋势是**融合**：以 MLLM 为骨干（第③类）， 在文档场景引入后交互（第④类），并全面继承双塔对比学习（第①类）的训练范式。

> **说明**：**易混淆示例 · Jina 系两条路线：** `jina-clip-v2` 是标准双塔 CLIP-style（属第①类）——文本侧 Jina-XLM-RoBERTa、图像侧 EVA02， 两塔独立编码后对比对齐，仅叠加了多语言、长文本、Matryoshka 变维度等工程增强； 而 `jina-embeddings-v4` 以 Qwen2.5-VL 为骨干并支持后交互多向量，属第③/④类。选型时注意区分。

### Jina 系列模型对比

| 模型 | 支持模态 | 视觉底座 | 文本底座 | 参数量 | 向量维度 | 核心适用场景 |
| --- | --- | --- | --- | --- | --- | --- |
| `jina-clip-v2` | 仅图文 | EVA02-L14 | Jina-XLM-RoBERTa | 0.865B（561M+304M） | 1024 | 低成本、CPU 图文检索、商品图搜 |
| `jina-embeddings-v4` | 仅图文 | Qwen2.5-VL-3B ViT | Qwen2.5-VL LLM（Qwen2.5） | 3.8B | 2048 | 高精度、图表 / 扫描文档、超长文本图文检索 |
| `v5-omni-small` | 图文 + 音视频 | Qwen3VL vision tower（+ Qwen2.5-Omni 音频塔） | Qwen3（v5-text-small，28 层 / 1024） | 1.56B | 1024 | 存量 v5-text 索引升级多模态、企业混合素材库 |
| `v5-omni-nano` | 图文 + 音视频 | Qwen3VL vision tower（+ Qwen2.5-Omni 音频塔） | EuroBERT（双向，12 层 / 768） | 1.04B | 768 | 边缘设备、低配 GPU、小规模混合多媒体检索 |

> **说明**：**架构核实说明（据 Jina 官方模型卡）：** jina-clip-v2 视觉塔为 EVA02-L14、文本塔为 Jina-XLM-RoBERTa；jina-embeddings-v4 基于 Qwen2.5-VL。 v5-omni 系列采用**"冻结塔组合"（frozen-tower composition）**：在 v5-text 文本塔上外挂视觉/音频编码器， 只训练轻量连接器。两款 v5-omni 的视觉塔均为 `Qwen3VL vision tower`、音频塔为 `Qwen2.5-Omni audio tower`； small 文本塔为 Qwen3（28 层 / 1024），nano 为 EuroBERT（双向，12 层 / 768）。 **各字段逐个核对（据官方模型卡）：** jina-clip-v2 全维 **1024**（可 Matryoshka 截断至 64，原稿 768 已订正），0.865B = 文本 561M + 视觉 304M，文本上下文 8192 token； jina-embeddings-v4 单向量 2048 / 多向量 128（Matryoshka 128–2048），32768 token； v5-omni-small 1.56B / 1024 维（Matryoshka 32–1024，32768 token）； v5-omni-nano 1.04B / 768 维（Matryoshka 32–512，8192 token）。 （原稿"SigLIP2"视觉底座、jina-clip-v2 的 768 维经核实有误，均已订正。）

## 二、发展顺序（时间线）

- **2021 · 双塔对比范式开端**  
  **CLIP**（OpenAI）与 **ALIGN**（Google）用 4 亿级图文对做对比学习， 首次证明大规模弱监督对齐能带来强零样本检索/分类能力，奠定"共享嵌入空间"的基础。

- **2021–2022 · 融合编码器兴起**  
  **ALBEF** 提出"先对齐再融合"，**BLIP** 引入 caption 自举与去噪， **CoCa** 把对比与生成合一，**BEiT-3** 用统一 Transformer 处理多模态，细粒度理解显著提升。

- **2023 · 对比训练规模化与改良**  
  **SigLIP** 用 Sigmoid loss 替代 Softmax 提升大 batch 稳定性； **EVA-CLIP**、**OpenCLIP** 持续 scaling；**Chinese-CLIP** 等推动中文与多语能力。

- **2024 · MLLM 嵌入 + 基准诞生**  
  **E5-V** 首次用多模态大模型直接产出嵌入；**VLM2Vec** 提出对比训练框架 并发布 **MMEB** 基准（36 数据集，4 类任务）统一评测；**ColPali** 开创视觉文档后交互路线。

- **2025 · MLLM 通用嵌入成为主流并刷榜**  
  **GME**、**BGE-VL**、**MMRet** 推进通用多模态检索； **VLM2Vec-V2 / MMEB-V2** 扩展到视频与视觉文档（57–78 数据集）； **Ops-MM**、**RzenEmbed**、字节 **Seed1.6-Embedding** 相继登顶。

- **2026 初 · 统一表征 + 工程化**  
  **Qwen3-VL-Embedding-8B** 以 MMEB-V2 全域 77.8 登顶（图文域 80.1）， 统一文本/图/视频/文档，并把 **MRL**、**量化感知训练**、多阶段蒸馏做进训练管线， 标志着从"刷分"走向"高效可部署"。

## 三、四类模型能力对比

| 维度 | ① 双塔对比 | ② 融合编码器 | ③ MLLM 通用嵌入 | ④ 多向量后交互 |
| --- | --- | --- | --- | --- |
| 代表模型 | CLIP / SigLIP | BLIP / CoCa | Qwen3-VL-Emb / RzenEmbed | ColPali / ColQwen |
| 代表模型参数量 | CLIP ViT-L/14 ≈4.28 亿；<br>SigLIP-So400m 视觉塔 ≈4 亿 | BLIP(ViT-L) ≈4.7 亿；<br>BLIP-2 约 3.4B–12B（含冻结 LLM）；<br>CoCa 原论文 ≈2.1B | Qwen3-VL-Embedding 2B / 8B；<br>RzenEmbed 2B / 8B | ColPali ≈3B（基于 PaliGemma）；<br>ColQwen2 ≈2B / ColQwen2.5 ≈3B |
| 检索效率 | ★★★★★ | ★★★ | ★★★（需 MRL/量化） | ★★（多向量开销大） |
| 细粒度/语义理解 | ★★ | ★★★★ | ★★★★★ | ★★★★ |
| 指令感知 | 否 | 否 | 是 | 部分 |
| 视觉文档场景 | 弱 | 中 | 强 | 最强 |
| 典型部署成本 | 低 | 中 | 高 | 高（存储） |

## 四、关键技术

### 基础训练技术

**对比学习（Contrastive Learning / InfoNCE）**

核心目标函数：拉近正样本图文对、推远负样本，通过温度系数 `τ` 调节分布锐度。 是所有图文嵌入模型的基石；SigLIP 用 Sigmoid loss 缓解大 batch 下的归一化压力。

**大规模弱监督图文对**

从互联网抓取的数亿级 (image, alt-text) 对是能力来源。数据的规模、清洗质量与多样性， 往往比模型结构更决定上限。近期普遍用 VLM 做数据合成与打标以补足稀缺场景。

**硬负例挖掘（Hard Negative Mining）**

用当前模型检索出"似是而非"的高相似度负例参与训练，显著提升判别力； 同时需要"假负例过滤"（相似度超过正例即屏蔽），避免把真正相关项误当负例。

### MLLM 嵌入路线的关键技术

**Pooling 策略（表征提取）**

如何从骨干网络取出句/图向量：CLIP 用 CLS，文本模型常用 mean pooling， 而 MLLM 类多在输入末尾加 `<EOS>`/`PAD` token，取其最后隐藏态作为整体表征（last-token pooling）。

**指令感知（Instruction-aware Embedding）**

把任务指令（如"检索能回答该问题的图片"）作为输入的一部分，让同一模型按任务产出不同表征， 是通用嵌入模型泛化到多任务的关键，也是 MLLM 骨干相比 CLIP 的天然优势。

**多阶段训练（Multi-stage Pipeline）**

典型流程：大规模合成数据**对比预训练** → 高质量数据**多任务微调** → 从 reranker **蒸馏** → **模型合并**。逐级"以模型提数据质量、以数据促模型能力"，形成正循环。

**知识蒸馏（Reranker → Embedding）**

用更强的 cross-encoder 重排模型给出细粒度相关性分数，作为软标签监督双塔嵌入模型， 在检索任务上大幅提点；再与微调分支合并，兼顾分类/QA 不掉分。

### 高效落地技术

**Matryoshka 表征学习（MRL）**

训练时对同一向量的多个前缀维度同时施加损失，使得推理时可按需截断维度 （如 4096→512）而性能损失很小，直接换来存储减半、检索提速一倍。

**量化感知训练（QAT）**

训练中同时对全精度与低精度（int8/binary）嵌入计算损失，让模型学出"抗量化"的表征。 int8 几乎无损、可大幅压缩索引存储与算力。

**后交互 / 多向量（Late Interaction / MaxSim）**

ColPali 系不把文档压成单向量，而保留 patch 级多向量，查询时对每个查询 token 取最大相似度求和。 换取视觉文档检索的高精度，代价是索引存储与计算开销上升。

**统一表征空间与模型合并（Model Merging）**

让文本、图像、视频、文档共享一个向量空间以支持任意跨模态检索； 并通过参数合并调和不同训练阶段/任务间的能力冲突，得到均衡的最终模型。

## 五、双塔（Dual-Encoder）架构详解

双塔架构是 CLIP 一系的核心。精髓是**两个模态各走各的编码器，只在最末端用向量相似度对齐**—— 即"双塔 + 晚融合（late fusion）"，中间没有任何跨模态注意力，这正是它区别于第②类融合编码器的根本点。

图像输入 x_img 图像编码器 ViT / CNN → [CLS] 投影层 + L2 归一化 → 单位超球面 v_img (d 维) 文本输入 x_txt 文本编码器 Transformer → [EOS]/mean 投影层 + L2 归一化 → 单位超球面 v_txt (d 维) 两塔独立 参数不共享 余弦相似度（点积） sim = v_img · v_txt 对比学习 InfoNCE（对角线为正样本）

### 数据流逐层拆解

**图像塔（Image Encoder）**

把图像编码成固定长度向量。**ViT**：图像切 patch（如 14×14）+ 位置编码，过 Transformer，取 `[CLS]` 输出（CLIP-L/14、SigLIP、EVA02 均属此类）；早期也用 **CNN**（CLIP 用过 ResNet，ALIGN 用 EfficientNet-L2）。

**文本塔（Text Encoder）**

文本 tokenize + 位置编码后过 Transformer。CLIP 取序列结尾 `[EOS]` token 的隐藏态作句向量；BERT 类常用 `[CLS]` 或 mean pooling（jina-clip-v2 文本塔即 mean pooling）。

**投影层 + L2 归一化（对齐的关键）**

两塔输出维度往往不同，各接一个**线性投影层**映射到**同一维度 d**（共享嵌入空间），再做 **L2 归一化**压到单位超球面。这样点积就直接等于余弦相似度： `sim = (v_img·v_txt) / (‖v_img‖‖v_txt‖) = v̂_img · v̂_txt`。

### 训练目标：对比学习 InfoNCE

一个 batch 有 N 对配对图文，算出 N×N 相似度矩阵，**对角线是正样本、其余 N²−N 个是负样本**， 目标是拉高对角线、压低非对角线。采用对称 InfoNCE（图→文、文→图各算一次交叉熵）：

`L = −(1/N) Σᵢ log[ exp(sim(vᵢ_img, vᵢ_txt)/τ) / Σⱼ exp(sim(vᵢ_img, vⱼ_txt)/τ) ]`

**温度系数 τ**：缩放 logits 控制分布锐度，CLIP 把它设为可学习参数（`logit_scale`）。

**大 batch 至关重要**：负样本来自同 batch，batch 越大负例越多、对齐越强（CLIP 用 32k）。SigLIP 的关键改进是把 Softmax 换成 **Sigmoid loss**，让每对图文独立做二分类，摆脱对超大 batch 全局归一化的依赖。

### 推理与检索流程（双塔最大的工程优势）

因为两塔独立，图侧可**离线预编码并复用**：

1. 把图库所有图片提前过图像塔，向量存入向量库（FAISS / Milvus 等）；
1. 来一个文本 query，只需过一次文本塔得到 query 向量；
1. 在向量库做近似最近邻（ANN）检索，点积取 Top-K。

在线只算 query 一侧，这正是双塔能撑起**亿级图库实时检索**的原因，也是它相比融合编码器（每个 query-doc 对都要重跑一遍跨模态网络）的决定性效率优势。

> **说明**：**能力与局限：** 强项是零样本分类/检索、可扩展、速度快；短板是**细粒度/组合语义弱**——整图整句各压成一个向量，丢失 token 级对齐（如"红衣服的人在蓝车左边"这类关系易错），且不支持指令、长文本较弱。 后续改进（SigLIP 改 loss、EVA-CLIP 提 scaling、jina-clip-v2 加多语言/长文本/Matryoshka）多在补这些短板，但骨架仍是这套双塔。

### 案例：SigLIP 2 —— 双塔编码器的集大成改进

SigLIP 2 是 Google 于 2025 年 2 月发布的**多语言视觉-语言编码器家族**，仍是双塔路线、SigLIP 的第二代。 核心思路：**在 SigLIP 的 Sigmoid loss 之上，把此前多个独立发展的训练技术统一进一套配方**， 同时提升语义理解、定位能力与稠密特征。在所有规模上，其零样本分类、图文检索及作为 VLM 视觉底座的迁移能力均超过同规模 SigLIP。

#### 四项训练改进

**① Sigmoid loss（继承自 SigLIP）**

基础图文对比对齐目标：每对图文独立做二分类，摆脱对超大 batch 全局归一化的依赖。

**② 加解码器：Decoder-based 预训练**

在双塔外加一个**文本解码器**，承担三个任务：预测整图 caption、给定区域描述预测 bounding box 坐标、给定坐标预测区域 caption。这个额外信号让视觉编码器**位置感知**，显著改善 OCR 与定位。

**③ 自蒸馏：Global-Local Loss + Masked Prediction**

用同一模型做师生（teacher 是 student 参数的滑动平均 EMA）：**Global-Local Loss** 让 student 只看局部视图去匹配 teacher 的整图表征；**Masked Prediction Loss** 遮住 student 50% 的 patch，在被遮位置匹配 teacher 特征。二者提升细粒度局部语义与稠密预测。为省算力，这些损失只在训练完成 **80%** 后才加入。

**④ 在线数据筛选 + 多语言数据**

训练中做 online data curation，并引入多语言语料，带来多语言能力。

#### NaFlex：动态分辨率变体

图像模型对分辨率/长宽比敏感。SigLIP 2 提供两种适配：**固定分辨率变体**（取 95% 训练 checkpoint，resize 位置/patch 编码后继续训到目标分辨率）； **动态分辨率变体 NaFlex**（融合 FlexiViT 变序列长度 + NaViT 原生长宽比，**一个模型兼顾 OCR 与文档理解**，带 `-naflex` 后缀）。

#### 模型规格

| 规格 | 参数量 | patch | 分辨率 |
| --- | --- | --- | --- |
| Base | 86M | 32 / 16 | 224–512 |
| Large | 303M | 16 | 256 / 384 / 512 |
| So400m（形状优化） | ≈400M | 14 / 16 | 224 / 384 / 512 |
| Giant | 1B | 16 | 256 / 384 |

相比 SigLIP 1 新增了 **Giant (1B)** 系列与 **NaFlex** 动态分辨率变体；各档嵌入维度分别为 Base 768 / Large 1024 / So400m 1152 / Giant 1536（so400m-patch14-384 即 1152）。

#### 文本塔（Text Encoder）

SigLIP 2 的文本塔是一个标准 **Transformer 编码器**（非解码器 LLM），与视觉塔并列组成双塔，宽度/深度按变体匹配视觉塔，输出维度与该档嵌入维度一致（如 ViT-B/16 → 768）。

**Tokenizer：改用 Gemma 多语言词表（相比 SigLIP 1 的最大变化）**

SigLIP 1 用约 32k 的多语言 SentencePiece 词表；**SigLIP 2 换成 Gemma 多语言 tokenizer（约 256k 词表）**，这是其多语言能力（含中文可直接编码）的直接来源。代价：256k 词表使 **文本塔的嵌入表非常大**（约 256000×d，Base 768 宽度下仅嵌入表就约 2 亿参数，往往比同档视觉塔还重）——估算内存/算力时不能只按 Transformer 层数计。

**输入与池化**

文本**定长 64 token**、右侧 padding 到满，推理阶段**不使用 attention mask**（直接过完整 64 长度）。池化取**最后一个 token** 的隐藏态 → 线性 head → 句向量，再 **L2 归一化**。注意与视觉侧不同：视觉塔用 **MAP 多头注意力池化**，两塔池化方式不一致。

> **说明**：**推理与部署要点（双塔特性）：** ① 图、文两塔**独立编码**，图库可离线预编码建索引，在线只编 query 一侧； ② 两塔输出均 L2 归一化后，**点积即余弦相似度**； ③ 零样本分类依赖 **prompt 模板**（英文常用 `This is a photo of a {}.`，中文可用中文类名/模板），且换类别标签时必须用 **SigLIP 2 自带的 Gemma tokenizer** 重新生成 token（定长 64），不能混用其它分词器； ④ 板端部署常见做法：离线生成 `input_ids [N,64]` 存盘，设备端只跑视觉/文本两个编码器，最后做余弦匹配。

> **说明**：**主要用途：**零样本分类 / 图文检索全规模领先；因解码器 + 自蒸馏，定位与稠密预测（分割、深度）明显增强； 最重要的应用是**作为 VLM 的视觉底座**——PaliGemma / PaliGemma 2 即用 SigLIP 家族当视觉塔，第④类的 ColPali 骨干 PaliGemma-3B 里的视觉塔正是 SigLIP-So400m。 来源：[SigLIP 2 论文 arXiv:2502.14786](https://arxiv.org/abs/2502.14786) 与 [HuggingFace 官方博客](https://huggingface.co/blog/siglip2)，内容经改写。

## 六、融合编码器（Fusion Encoder）架构详解

融合编码器（也叫跨模态编码器）的核心是**让图像和文本在网络内部就通过跨模态注意力深度交互**， 而非双塔那样各编各的、只在最后点积一次。它换来更强的细粒度对齐，代价是检索效率。

图像输入 图像编码器 (ViT) 图 token 序列 文本输入 文本编码器 文 token 序列 ITC 先对齐（进融合前） 跨模态融合层 Cross-Attention（token 级深度交互） ITM 匹配 MLM 掩码 VQA / caption 多任务预训练目标

### 两种融合结构

**① 单流（Single-stream）**

把图像 token 与文本 token **拼接成一个序列**喂进同一个 Transformer，靠 self-attention 完成跨模态交互。代表：VisualBERT、UNITER、BEiT-3（用模态专家 FFN 的变体）。

**② 双流 / 协同注意力（Dual-stream / Co-attention）**

两个模态各有一支，通过 **cross-attention** 层互相交换信息。代表：ViLBERT、LXMERT、ALBEF（文本后半段做 fusion）。

### "先对齐，再融合"（ALBEF 的关键思想）

早期模型直接把未对齐的图文特征丢进融合层，跨模态注意力很难学。ALBEF 提出：**先用对比损失 ITC 把图、文特征在进入融合层前拉到同一空间**， 让融合层拿到已粗对齐的特征再做深度交互。这几乎成了后续所有融合模型的标配。

### 多任务预训练目标（这一类的灵魂）

**ITC · Image-Text Contrastive**

图文对比做粗粒度对齐（同 CLIP 对比损失），也用于检索初筛。

**ITM · Image-Text Matching**

把 [CLS] 联合表征过二分类头判断"图文是否匹配"，通常配 **hard negative mining**（用 ITC 相似度挑最难负例），是细粒度判别的关键。

**MLM · Masked Language Modeling（及可选 MIM / captioning）**

遮住部分文本 token，让模型结合图像预测，逼它建立词—区域关联。部分模型还加掩码图像建模、图像描述生成（CoCa、BLIP）、PrefixLM 等。

### 四个代表模型的结构差异

| 模型 | 结构要点 |
| --- | --- |
| ALBEF | ViT-B/16 + BERT-base 拆成"6 层文本编码 + 6 层跨模态融合"；用**动量蒸馏**对抗网络噪声标签 |
| BLIP | 提出 **MED**（多模态编码-解码混合），一套参数可切"单模态编码/图文匹配/图文生成"三模式；用 **CapFilt** 生成+过滤清洗数据 |
| CoCa | 图像编码器 + **文本解码器**，同训"对比损失（前半）+ captioning 生成（后半）"，对比与生成合流 |
| BEiT-3 | **Multiway Transformer**：共享自注意力 + 按模态切换的专家 FFN（视觉/语言/视觉-语言专家），把图、文、图文统一建模 |

### 推理与检索：为什么慢

融合模型判断一对图文是否匹配，必须**把这对图文一起过一遍融合层**，无法像双塔那样离线建索引（N 张图理论上要跑 N 次融合前向）。 工程上用**两阶段（retrieve-then-rerank）**缓解：先用 ITC 分支（等价双塔）快速召回 Top-K，再用 ITM/融合分支精排，兼顾速度与精度。

> **说明**：**能力与局限：**强项是细粒度对齐、token—区域级理解，天然支持 VQA / 图文匹配 / captioning，精排精度高于纯双塔； 短板是检索/索引成本高、模型更重、多目标训练更复杂，不适合亿级实时向量检索。 演进主线：**双塔 ITC 对齐 + 跨模态融合层（ITM/MLM）+ 可选生成目标**，用计算换对齐精度——后来的第③类 MLLM 通用嵌入某种程度上是"用大 VLM 内化融合能力，再退回双塔式输出向量"，取两者之长。

## 七、四大类经典模型详细对比

下列数据均取自各模型**原始论文或官方模型卡**，并注明来源。各类采用其领域内公认的经典基准： ① 用 ImageNet 零样本 + COCO/Flickr 检索；② 用微调后 COCO/Flickr 检索 + VQA；③ 用 MMEB；④ 用 ViDoRe。 由于不同论文口径不同（零样本 vs 微调、基准版本差异），跨类分数**不可直接横向比较**。

### ① 双塔对比学习类 · 基准：ImageNet 零样本 top-1 / 检索 R@1

| 模型 | 总参数 | 图像底座 | 文本底座 | ImageNet 零样本 | 检索 R@1（零样本） |
| --- | --- | --- | --- | --- | --- |
| CLIP (ViT-L/14) | ≈427.6M | ViT-L/14 ≈304M | Transformer ≈124M | 75.5%（336px 76.2%） | COCO 图→文58.4/文→图37.8；Flickr 88.0/68.7 |
| ALIGN | ≈800M | EfficientNet-L2 ≈480M | BERT-Large ≈340M | 76.4% | COCO 58.6/45.6；Flickr 88.6/75.7 |
| SigLIP (SoViT-400m/14@384) | ≈878M | So400m ≈400M | ≈478M | 83.2% | COCO 70.2/52.0；Flickr 93.5/80.5 |
| **SigLIP 2 (So400m/14@384)** | ≈1.14B | So400m ≈400M | ≈736M（推算，含 256k 词表嵌入） | **84.1%** | COCO 71.7/55.8；Flickr 94.9/85.7 |
| SigLIP 2 (Giant g-opt/16@384) | ≈1.87B | g ≈1B | 未单列 | 85.0% | COCO 72.8/56.1；Flickr 95.4/86.0 |
| OpenCLIP (ViT-bigG/14, LAION-2B) | ≈2.5B | ViT-bigG ≈1.8B | ≈0.69B | 80.1% | 论文/卡未列 R@1 |
| jina-clip-v2 | ≈0.865B | EVA02-L14 ≈304M | Jina-XLM-RoBERTa ≈561M | —（主打多语跨模态检索） | 较 v1 检索 +3%（见技术报告） |

来源：CLIP [2103.00020](https://arxiv.org/abs/2103.00020) / HF openai-clip-vit-large-patch14；ALIGN [2102.05918](https://arxiv.org/abs/2102.05918)；SigLIP [2303.15343](https://arxiv.org/abs/2303.15343)；SigLIP 2 [2502.14786](https://arxiv.org/abs/2502.14786)（Table 1，ImageNet 0-shot 与检索 R@1；总参取自 HF safetensors）；OpenCLIP [LAION](https://laion.ai/blog/large-openclip/)。检索 R@1 部分引自 CoCa / SigLIP2 论文对比表（均为零样本口径）。CLIP/ALIGN/OpenCLIP 及 SigLIP 文本底座拆分含推算/社区统计（SigLIP 2 文本塔参数论文未单列，为总参减图像塔的推算值）。

### ② 融合编码器 / 跨模态类 · 基准：微调检索 R@1 / VQA test-dev

| 模型 | 总参数 | 图像底座 | 文本底座 | COCO / Flickr 检索 R@1（微调） | VQA test-dev |
| --- | --- | --- | --- | --- | --- |
| ALBEF | ≈210M | ViT-B/16 ≈85.8M | BERT-base ≈123.7M | COCO 77.6/60.7；Flickr 95.9/85.6 | 75.84 |
| BLIP | ≈252M（ViT-B） | ViT-B/16 ≈86M（可选 ViT-L） | BERT-base（MED） | COCO 80.6/63.1（ViT-L 达 82.4/65.1）；Flickr 96.6/87.2 | 78.25 |
| CoCa | ≈2.1B | ViT-giant ≈1B | 文本解码器 ≈1.1B（非 BERT） | COCO 66.3/51.2；Flickr 92.5/80.4（零样本） | 82.3 |
| BEiT-3 | ≈1.9B | Multiway Transformer（视觉专家 692M + 语言专家 692M + VL 专家 52M + 共享注意力 317M） | COCO 84.8/67.2；Flickr 98.0/90.3 | 84.19 |  |

来源：ALBEF [2107.07651](https://arxiv.org/abs/2107.07651)；BLIP [2201.12086](https://arxiv.org/abs/2201.12086)；CoCa [2205.01917](https://arxiv.org/abs/2205.01917)；BEiT-3 [2208.10442](https://arxiv.org/abs/2208.10442)。注：CoCa 检索为零样本、其余为微调；CoCa 用文本解码器、BEiT-3 用模态专家 FFN，均非传统 BERT 文本塔。检索格式为 图→文 / 文→图。

### ③ MLLM 通用嵌入类 · 基准：MMEB（多模态嵌入基准）

| 模型 | 骨干 VLM | 总参数 | 图像底座 | 文本底座（LLM） | MMEB 分数 |
| --- | --- | --- | --- | --- | --- |
| E5-V | LLaVA-NeXT-8B | ≈8.4B | CLIP ViT-L/14-336 ≈0.3B | Llama-3-8B | 早于 MMEB，原文报 COCO/Flickr/CIRR 检索 |
| VLM2Vec | Phi-3.5-V / LLaVA-1.6 | 4.15B / 7.57B | ViT-L/14-336 ≈0.3B | Phi-3.5-mini 3.8B / Mistral-7B | MMEB-V1 Overall 60.1（Phi）/ 62.9（LLaVA 最佳） |
| GME | Qwen2-VL-2B / 7B | 2.2B / 8.2B | Qwen2-VL ViT ≈675M | Qwen2-1.5B / 7B | MMEB-V1 Overall 51.9 / 56.0（第三方复测） |
| RzenEmbed | Qwen2-VL-2B / 7B | 2.21B / 8.29B | Qwen2-VL ViT ≈675M | Qwen2-1.5B / 7B | MMEB-V1 Overall 72.3 / 75.9；MMEB-V2 All 67.2 / 71.6 |
| Qwen3-VL-Embedding | Qwen3-VL-2B / 8B | 2B / 8B | Qwen3-VL 视觉编码器 | Qwen3 LM | MMEB-V2 All 73.2 / **77.8**（8B 居榜首） |

来源：E5-V [2407.12580](https://arxiv.org/abs/2407.12580)；VLM2Vec [2410.05160](https://arxiv.org/abs/2410.05160)；GME [2412.16855](https://arxiv.org/abs/2412.16855)；RzenEmbed [2510.27350](https://arxiv.org/abs/2510.27350)；Qwen3-VL-Embedding [2601.04720](https://arxiv.org/abs/2601.04720)。MMEB-V1=36 数据集 Precision@1；MMEB-V2=78 数据集。各嵌入论文普遍只给总参数、不单列视觉/文本底座（此处为骨干推算）。

### ④ 视觉文档后交互类 · 基准：ViDoRe（nDCG@5）

| 模型 | 骨干 VLM | 总参数 | 图像底座 | 文本底座 | ViDoRe nDCG@5 |
| --- | --- | --- | --- | --- | --- |
| ColPali | PaliGemma-3B | ≈2.9B | SigLIP-So400m ≈400M | Gemma-2B ≈2B | v1 = **81.3**（对比最佳基线 67.0） |
| ColQwen2-v1.0 | Qwen2-VL-2B | ≈2.2B | Qwen2-VL ViT ≈675M | Qwen2-1.5B | v1 较 ColPali 约 +5~8（确值以 ViDoRe 榜单为准） |
| ColNomic-embed-multimodal-3b | Qwen2.5-VL-3B | 3B | ≈675M（未单列） | ≈3B（未单列） | v1 = **91.0**；v2 = 63.5（论文）/ 61.2（卡） |
| jina-embeddings-v4 | Qwen2.5-VL-3B | 3.8B | ≈675M（未单列） | ≈3B（未单列） | 支持单/多向量；确值见技术报告 |

来源：ColPali [2407.01449](https://arxiv.org/abs/2407.01449)；ColQwen2 同论文 / HF vidore-colqwen2-v1.0；ColNomic [2507.05513](https://arxiv.org/abs/2507.05513)；jina-v4 [2506.18902](https://arxiv.org/abs/2506.18902)。ColQwen2 与 jina-v4 的确切 ViDoRe 平均分未能从一手来源核实到单一数值，故按论文/榜单描述给出。

## 八、小结与选型指引

技术演进主线：**双塔对比（快）→ 融合编码器（准）→ MLLM 通用嵌入（强且通用）→ 工程化高效部署（MRL/量化/后交互）**。

· 大规模快速检索、算力紧张 → CLIP / SigLIP 系； · 追求综合效果与指令泛化、可私有化 → Qwen3-VL-Embedding、RzenEmbed 等 MLLM 通用嵌入； · 走 API 且要顶尖效果 → Seed1.6-Embedding； · 视觉文档 / PDF / 图表 RAG → ColPali / ColQwen / Jina-v4 等后交互模型。

## 九、纯图片 Embedding 模型调研

前八章的图文模型把图像对齐到**文本语义空间**；本章讨论的**纯图片 Embedding**则完全不涉及文本， 目标是把一张图映射成一个向量，使**视觉内容相近**的图向量距离更近，直接支撑 **图搜图（instance / near-duplicate retrieval）、图像去重、聚类、以图打标（kNN 分类）、版权与溯源**等任务。 它与 CLIP 的图像塔有本质区别：CLIP 图像塔被"拉向文本"，偏高层类别语义；纯视觉模型不受文本约束， 对**颜色、纹理、构图、实例外观**等低层视觉特征更敏感，因而在同模态图搜图上通常更强。

#### ① 监督分类预训练特征

*Supervised Backbone Features · 2015 起*

直接取 ImageNet 等监督分类模型（ResNet / ViT）的倒数层特征当嵌入，或在地标数据上专门微调检索模型 （GeM 池化 + 度量学习）。是最早、最工程化的图像检索路线。

代表：`ResNet` `ViT (sup.)` `AP-GeM` `DELG` `CVNet`

- 优点：成熟、轻量、易部署
- 短板：跨域泛化弱，通用图搜图不如自监督

#### ② 自监督对比 / 自蒸馏（当前主流）

*Self-Supervised (Contrastive / Self-Distillation) · 2020 起*

无需标签，靠"同图不同增广视图应相近"来学表征。从对比式（SimCLR/MoCo/BYOL）到自蒸馏式 （DINO/iBOT），再到规模化的 **DINOv2 / DINOv3**，冻结特征即可做强图搜图。

代表：`SimCLR` `MoCo` `BYOL` `DINO / iBOT` `DINOv2` `DINOv3`

- 优点：实例级检索 SOTA、跨域鲁棒、免微调
- 短板：大模型算力高（可用蒸馏小档）

#### ③ 掩码图像建模（MIM）

*Masked Image Modeling · 2021 起*

遮住大部分 patch 让模型重建，学出强**可微调**骨干。但其"开箱"特征（不微调直接取向量） 在检索/线性探测上通常弱于自蒸馏路线，更适合作下游微调初始化。

代表：`MAE` `BEiT` `SimMIM`

- 优点：微调上限高、预训练高效
- 短板：冻结特征做图搜图偏弱

#### ④ CLIP 图像塔（弱监督语义）

*Weakly-Supervised Image Tower · 2021 起*

把 CLIP/SigLIP 的图像塔单独拿来产图向量。语义类别检索强，但因被文本对齐"拉高层"， 对实例级、低层视觉相似的图搜图不如 DINO 系。

代表：`CLIP-ViT` `OpenCLIP` `SigLIP2 vision`

- 优点：语义/类别检索强、生态成熟
- 短板：实例级图搜图弱于 DINO 系

> **说明**：**核心结论：**纯图搜图首选**自监督自蒸馏路线（DINOv2 / DINOv3）**。DINOv2 论文实测：在 ROxford/RParis 实例检索上， 参数远大的 OpenCLIP ViT-G/14（≈1.8B）显著落后于 DINOv2 ViT-B（86M）——说明**"图搜图看视觉特征质量，不看是否对齐文本、也不唯参数量论"**。

### 发展脉络（时间线）

- **2015–2019 · 监督特征 + 度量学习检索**  
  取 **ResNet** 分类特征做检索；地标检索用 **GeM 池化**、**AP-GeM**、**DELG** 等做度量学习与局部特征重排。

- **2020 · 对比式自监督崛起**  
  **SimCLR / MoCo / BYOL** 证明无标签也能学出可媲美监督的表征，为免标注的通用视觉嵌入铺路。

- **2021 · 自蒸馏与掩码建模**  
  **DINO** 提出无标签自蒸馏，注意力天然分割前景，检索特征优秀；同期 **MAE / BEiT** 走掩码重建路线。

- **2023 · DINOv2 规模化**  
  **DINOv2**（Meta）用 1.42 亿精选图 + 大模型，产出开箱即用、跨域鲁棒的视觉特征，成为图搜图/稠密任务的通用底座。

- **2025 · DINOv3 再扩 6 倍**  
  **DINOv3** 模型规模 ×6（达 **7B**）、数据 ×12，提出 **Gram anchoring** 抑制长训练下稠密特征退化， 并放出从 ViT-7B 蒸馏出的 ViT-S/B/L 及 ConvNeXt 全家桶，SSL 首次在多项探测任务上超越弱监督模型。

### 关键技术

**特征池化（CLS / GeM / 多尺度）**

ViT 常取 `[CLS]` 作全局向量；检索专用模型多用 **GeM（广义平均池化）**聚合 patch 特征，比平均/最大池化更利于实例检索；也可拼接多尺度/多层特征增强鲁棒性。

**实例检索评测口径（mAP / kNN）**

图搜图的核心指标是 **ROxford / RParis 的 mAP**（冻结特征 + 余弦最近邻），衡量"找回同一地标/实例"的能力；**ImageNet kNN top-1** 常作检索质量的代理指标（不训练分类头，直接近邻投票）。

**寄存器 token（Register Tokens）**

DINOv2/v3 引入额外 **register token** 吸收注意力中的异常"伪影"，让 patch 特征更干净，提升稠密任务与检索稳定性（DINOv3 为 1 CLS + 4 register + patch tokens）。

**Gram Anchoring（DINOv3 稠密特征保鲜）**

长时间训练会使 patch 级稠密特征逐渐退化。DINOv3 用 **Gram anchoring** 约束特征的 Gram 矩阵，抑制退化，保住高分辨率稠密特征质量——这对密集匹配、分割、局部检索尤为关键。

**蒸馏出小档以便部署**

DINOv2/v3 均用大教师（DINOv3 为 ViT-7B）蒸馏出 ViT-S/B/L 甚至 ConvNeXt 变体，让边缘/设备端也能用上接近大模型的视觉特征，是落地图搜图的主力档位。

### 代表模型对比 · 基准：实例检索 mAP 与 ImageNet kNN

| 模型 | 类别 | 参数量 | 维度 | ROxford-H (mAP) | RParis-H (mAP) | ImageNet kNN | 许可 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DINOv2 ViT-S/14 | 自蒸馏 SSL | 21M | 384 | 43.2 | 68.5 | 79.0 | Apache-2.0 |
| **DINOv2 ViT-B/14** | 自蒸馏 SSL | 86M | 768 | 49.5 | 78.6 | 82.1 | Apache-2.0 |
| DINOv2 ViT-L/14 | 自蒸馏 SSL | 300M | 1024 | 54.0 | 83.5 | 83.5 | Apache-2.0 |
| DINOv2 ViT-g/14 | 自蒸馏 SSL | 1.1B | 1536 | 52.3 | 82.6 | 83.5 | Apache-2.0 |
| ↓ 以下 DINOv3 的 **Ox.-H †** 取自**官方模型卡**（Oxford-Hard 实例检索 mAP，patch-16、DINOv3 自有评测协议）；**与上方 DINOv2 的 ROxford-H 口径不同，不可直接横向比较**，仅供 DINOv3 内部各档对比。模型卡未给 RParis-H 与 ImageNet kNN（官方改报 IN-ReaL 等 probe 指标）。 |  |  |  |  |  |  |  |
| DINOv3 ViT-S/16 | 自蒸馏 SSL（蒸馏） | 21M | 384 | 49.5 † | 未报告 | 未报告 | DINOv3 License |
| DINOv3 ViT-S+/16 | 自蒸馏 SSL（蒸馏） | 29M | 384 | 50.0 † | 未报告 | 未报告 | DINOv3 License |
| DINOv3 ViT-B/16 | 自蒸馏 SSL（蒸馏） | 86M | 768 | 58.5 † | 未报告 | 未报告 | DINOv3 License |
| DINOv3 ViT-L/16 | 自蒸馏 SSL（蒸馏） | 300M | 1024 | 63.1 † | 未报告 | 未报告 | DINOv3 License |
| DINOv3 ViT-H+/16 | 自蒸馏 SSL（蒸馏） | 840M | 1280 | 64.5 † | 未报告 | 未报告 | DINOv3 License |
| DINOv3 ViT-7B/16 | 自蒸馏 SSL（教师） | 6716M | 4096 | **72.8 †** | 未报告 | 未报告 | DINOv3 License |
| OpenCLIP ViT-G/14（CLIP 对照） | 弱监督 CLIP 图像塔 | ≈1.8B | — | 19.7 | 60.2 | 81.7 | — |
| MAE ViT-L（对照） | 掩码建模 MIM | 300M | 1024 | 冻结特征检索/kNN 偏弱，长于微调 | CC-BY-NC |  |  |
| SigLIP2-Base 视觉塔（对照） | 弱监督图像塔 | 86M | 768 | 语义检索强、实例图搜图弱于 DINO | Apache-2.0 |  |  |

DINOv2 数值取自论文 [2304.07193](https://arxiv.org/abs/2304.07193) Table 9（ROxford/RParis-Hard 实例检索 mAP，冻结特征）与 Table 4（ImageNet kNN top-1）；参数/维度见论文 Table 17 与 Meta 官方模型卡。 DINOv3 见 [2508.10104](https://arxiv.org/abs/2508.10104) 与 [官方模型卡](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md)（ViT-S 21M / S+ 29M / B 86M / L 300M / H+ 840M / 7B 6716M，嵌入维度 384/384/768/1024/1280/4096，patch 16、含 4 个寄存器 token；除 7B 教师外均从 ViT-7B 蒸馏，另有 ConvNeXt 变体）。 **† 标记的 Ox.-H** 为官方模型卡给出的 **Oxford-Hard 实例检索 mAP**（DINOv3 自有协议，patch-16），**与上方 DINOv2 论文 Table 9 的 ROxford-H 口径不同，切勿直接横比**；模型卡未提供 RParis-H 与 ImageNet kNN（官方改报 IN-ReaL / IN-R / ObjectNet 等 probe 指标）。若需与 DINOv2 严格同口径对比，仍建议在自有数据实测。 OpenCLIP 对照与"CLIP 偏语义、DINO 偏视觉"结论见 DINOv2 论文及 [arXiv:2510.11835](https://arxiv.org/html/2510.11835v1)。

**案例：DINOv3 —— 纯视觉自监督的集大成**

DINOv3（Meta，2025-08）在 DINOv2 的自蒸馏范式上把**模型规模扩到 7B、训练数据扩 12 倍**， 核心创新是 **Gram anchoring** 解决长训练下稠密特征退化，并用 post-hoc 策略适配不同分辨率与模型尺寸。 它首次让**纯 SSL 模型在广泛探测任务上超越弱监督（CLIP 式）模型**，且放出 ViT-7B 教师 + 蒸馏小档（ViT-S/S+/B/L/H+）与 ConvNeXt 变体， 兼顾极致质量与边缘部署。官方模型卡也给出了 **Oxford-Hard 实例检索 mAP**（Ox.-H：ViT-B 58.5、ViT-L 63.1、ViT-7B 72.8），在其自有协议下随规模稳步提升——但该口径与 DINOv2 论文不同，横比需谨慎。 注意其许可为自定义 **DINOv3 License**（非 Apache），商用前需核对条款。

> **说明**：**选型指引（纯图搜图）：** ① 通用图搜图 / 免微调 → **DINOv2 或 DINOv3 的 ViT-B/L**，质量与成本均衡； ② 边缘/设备端 → DINOv2 ViT-S（21M）或 DINOv3 ViT-S（≈21.6M）蒸馏档 + int8 量化； ③ 追求极致 / 稠密任务 → DINOv3 ViT-7B 或 ViT-L； ④ 若检索目标是"同类别/同语义"而非"同实例"（如商品分类聚合）→ 可考虑 CLIP/SigLIP2 图像塔； ⑤ 需自训微调 → MAE/BEiT 作骨干初始化。商用许可上 DINOv2 系（Apache-2.0）最省心。

## 十、纯文本 Embedding 模型调研

纯文本 Embedding 把一段文本（词、句、段落、文档）映射为稠密向量，使**语义相近**的文本向量更近， 是 **语义检索、RAG、去重、聚类、分类、相似问答（STS）、代码检索**的基础设施。 与图文模型的文本塔不同，它**不需要对齐图像**，可以把全部容量投入到纯文本语义， 因而在 text-text 检索上通常显著强于 CLIP 类的文本塔。评测事实标准是 **MTEB / C-MTEB**。

#### ① BERT 时代句向量

*Sentence Encoders · 2019 起*

在 BERT 上做孪生网络或对比学习得到句向量，奠定"池化 + 余弦相似度"范式。轻量、易部署，是很多早期检索系统的基石。

代表：`Sentence-BERT` `SimCSE`

- 优点：小、快、成熟
- 短板：泛化与多任务能力有限

#### ② 对比学习通用嵌入（弱监督）

*Contrastive Universal Embedding · 2022 起*

用海量弱监督文本对做大规模对比预训练 + 高质量数据微调，产出跨任务通用嵌入。 中文/多语首选 **BGE、GTE、E5** 系；**BGE-M3** 还统一稠密/稀疏/多向量三功能。

代表：`E5` `GTE` `BGE / BGE-M3`

- 优点：中文/多语强、体积适中、开源可商用
- 短板：复杂指令与推理不如 LLM 骨干

#### ③ 指令微调嵌入

*Instruction-tuned · 2022 起*

把任务指令写进输入，让同一模型按任务产出不同表征。**Instructor** 首创，后被 LLM 骨干模型广泛继承，是通用性关键。

代表：`Instructor` `TART`

- 优点：一模型多任务、可定制检索意图
- 短板：需构造合适指令

#### ④ LLM 骨干嵌入（当前 SOTA）

*LLM-based Embedding · 2024 起*

直接用解码器大模型（Mistral / Qwen 等）当骨干做对比微调，last-token 池化 + 指令感知， 长文本与推理能力最强，霸榜 MTEB。代表 **Qwen3-Embedding、NV-Embed、gte-Qwen2、E5-mistral**。

代表：`E5-mistral-7b` `NV-Embed-v2` `gte-Qwen2` `SFR-Embedding` `Qwen3-Embedding`

- 优点：语义/指令/长文最强，多语覆盖广
- 短板：大、慢、显存高（需 MRL/量化）

> **说明**：**路线演进主线：**BERT 句向量 → 大规模对比学习通用嵌入（BGE/GTE/E5）→ 指令微调 → **LLM 骨干嵌入**。 与图文第③类高度同构：都是"以大模型为骨干 + 对比学习 + last-token 池化 + 指令感知 + MRL/量化落地"。

### 关键技术

**池化策略（CLS / mean / last-token / latent-attention）**

BERT 类多用 **[CLS]** 或 **mean pooling**；LLM 骨干因单向注意力常用 **last-token pooling**（取 `<EOS>` 隐藏态）。NV-Embed 提出 **latent attention 层**做可学习池化，进一步提点。

**双向注意力改造（LLM2Vec）**

解码器 LLM 天生单向，直接做嵌入会丢失后文信息。**LLM2Vec** 等把因果掩码改为双向 + 掩码 token 预测再对比微调；NV-Embed 也移除因果掩码以增强表征。

**对比学习 + 硬负例 + 大 batch**

沿用 InfoNCE，配 **in-batch 负例 + 挖掘的硬负例**；LLM 骨干需大 batch 与梯度缓存。数据配方（query-doc 对的质量与多样性）往往比模型更决定上限。

**指令感知 + 多任务**

输入前缀写任务指令（如"为这条查询检索相关文档"），使同一模型服务检索/分类/聚类/STS 等多任务，是 MTEB 高分的关键。

**MRL + 长文本 + 多语言**

**Matryoshka** 支持按需截断维度省存储；LLM 骨干天然支持 **长上下文**（Qwen3-Embedding 达 32k）；多语言语料带来跨语检索能力（BGE-M3、Qwen3-Embedding 覆盖 100+ 语言）。

### 代表模型对比 · 基准：MTEB / C-MTEB

| 模型 | 类别 | 骨干 / 参数量 | 维度 | 上下文 | 基准分数 | 许可 |
| --- | --- | --- | --- | --- | --- | --- |
| bge-large-zh-v1.5 | 对比（BERT 类） | 326M | 1024 | 512 | C-MTEB 平均 64.53 | MIT |
| bge-m3 | 对比（多语/多功能） | 568M | 1024 | 8192 | MIRACL 多语检索 SOTA（稠密+稀疏+多向量） | MIT |
| gte-Qwen2-7B-instruct | LLM 骨干 | Qwen2-7B | 3584 | 32k | MTEB(en) 约 70（发布时居前） | Apache-2.0 |
| E5-mistral-7b-instruct | LLM 骨干 | Mistral-7B | 4096 | 32k | MTEB(en) 约 66（首个 LLM 骨干爆款） | MIT |
| NV-Embed-v2 | LLM 骨干 + latent attn | Mistral-7B | 4096 | 32k | MTEB(en v1) 约 72.31（曾登顶） | CC-BY-NC（研究） |
| Qwen3-Embedding-0.6B | LLM 骨干 | 0.6B | 1024 | 32k | MTEB 多语 64.33 | Apache-2.0 |
| Qwen3-Embedding-4B | LLM 骨干 | 4B | 2560 | 32k | MTEB 多语 69.45 | Apache-2.0 |
| **Qwen3-Embedding-8B** | LLM 骨干 | 8B | 4096 | 32k | **MTEB 多语 70.58（发布时 No.1）** | Apache-2.0 |

来源：bge 系列 C-MTEB 见 [BAAI 模型卡](https://huggingface.co/BAAI/bge-large-zh-v1.5)；BGE-M3 [2402.03216](https://arxiv.org/abs/2402.03216)； Qwen3-Embedding 系列（0.6B/4B/8B、维度 1024/2560/4096、32k 上下文、MRL、8B MTEB 多语 70.58 且截至 2025-06-05 居榜首）见论文 [2506.05176](https://arxiv.org/abs/2506.05176) 与 [Qwen 官方博客](https://qwenlm.github.io/blog/qwen3-embedding/)； gte-Qwen2 / E5-mistral / NV-Embed-v2 分数为各自发布时 MTEB 口径（英文 v1），榜单实时更新、版本口径不同**不可直接横比**。0.6B 参数以外的维度/上下文以官方模型卡为准。

**案例：Qwen3-Embedding —— LLM 骨干文本嵌入的当前标杆**

基于 Qwen3 基础模型，提供 **0.6B / 4B / 8B** 三档嵌入 + 配套 reranker，继承多语言（100+ 语言）、长文本（32k）与推理能力。 训练用**多阶段范式**：大规模弱监督对比预训练 → 高质量数据 + 合成数据微调 → 模型合并，并支持 **指令感知**与 **MRL 变维**。 8B 档以 MTEB 多语 70.58 于 2025-06 登顶多语榜单；**0.6B 档**在设备端尤其实用——1024 维、Apache-2.0 可商用，是本综述第十一章设备端选型的核心候选之一。

> **说明**：**选型指引（纯 text-text）：** ① 中文为主、体积敏感 → **bge-large-zh-v1.5**（326M/MIT）或 bge-base； ② 多语言 / 长文档 / 需稀疏+稠密混合 → **bge-m3**； ③ 追求最高精度且算力充足 → **Qwen3-Embedding-8B / 4B** 或 gte-Qwen2-7B； ④ 设备端 / 低资源 → **Qwen3-Embedding-0.6B** 或 bge-small，配 MRL 截维 + int8 量化； ⑤ 走 API → OpenAI text-embedding-3、Cohere Embed v3、Voyage-3、Gemini Embedding 等（免运维、多语强，但数据出域需评估合规）。

## 十一、设备端 0.6B 双模型选型（图搜图 + text-text）

**场景约束**：设备端 embedding 能力总预算 **≈0.6B 参数，且为"并发常驻总量"**——图像模型与文本模型必须**同时装载**，两者参数之和 ≤0.6B。需求为两条独立检索线：

· **图搜图**（同模态、以图找相似图，非跨模态文搜图）； · **text-text**（纯文本语义检索 / RAG，非跨模态）。

> **说明**：**关键判断：为什么是两个独立模型、两个向量空间。** 纯 text-text 检索需要专门的文本 embedding 模型（CLIP 文本塔对齐到图像空间、做 text-text 偏弱）； 而图搜图只需纯视觉编码器，**不需要 CLIP 的文本塔**。二者互不共享向量空间，各自域内检索。 0.6B 并发预算下无法再塞下"强独立文本 + 强独立图像 + 跨模态"三合一，故按两条独立线拆分预算。

### 推荐组合（并发常驻总量 ≤0.6B）

| 档位 | 图搜图（视觉） | text-text（文本） | 合计参数 | 适用 |
| --- | --- | --- | --- | --- |
| **均衡（推荐）** | DINOv2 ViT-B/14 · 86M | bge-large-zh-v1.5 · 326M | **≈412M** | 图文检索质量均衡，留出量化余量 |
| 图像优先 | DINOv2 ViT-L/14 · 300M | bge-base-zh-v1.5 · ~102M | ≈402M | 图像相似检索质量要求高 |
| 文本优先/极省 | DINOv2 ViT-S/14 · 21M | bge-large-zh-v1.5 · 326M | ≈347M | 文本为主，图像召回够用 |
| ⚠️ 不建议 | DINOv2 ViT-L/14 · 300M | bge-large-zh-v1.5 · 326M | ≈626M | 略超 0.6B 并发预算 |

### 图搜图候选：视觉编码器

纯"图搜图"看重视觉外观/实例相似性。研究表明 **CLIP 偏高层语义（类别、文字），DINO 系对颜色、样式等低层视觉特征更敏感**， 更契合内容级图像相似检索；DINOv2 提供开箱即用、跨域鲁棒的视觉特征，无需微调。

| 模型 | 参数量 | 维度 | ROxford-H | RParis-H | AmsterTime | ImageNet k-NN | 许可 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DINOv2 ViT-S/14 | 21M | 384 | 43.2 | 68.5 | 43.5 | 79.0 | Apache-2.0 |
| **DINOv2 ViT-B/14（推荐）** | 86M | 768 | 49.5 | 78.6 | 45.6 | 82.1 | Apache-2.0 |
| DINOv2 ViT-L/14 | 300M | 1024 | **54.0** | **83.5** | **50.0** | 83.5 | Apache-2.0 |
| DINOv2 ViT-g/14（超预算） | 1.1B | 1536 | 52.3 | 82.6 | 46.7 | 83.5 | Apache-2.0 |
| OpenCLIP ViT-G/14（CLIP 对照） | ≈1.8B | — | 19.7 | 60.2 | 24.6 | 81.7 | — |
| DINOv3 ViT-B/16 · L/16 | 86M · 300M | 768 · 1024 | 更新更强，官方未给同口径 mAP（需自测） | DINOv3 License |  |  |  |
| SigLIP2-Base 视觉塔（备选） | 86M | 768 | 语义特征为主，图搜图弱于 DINO；工具链更顺 | Apache-2.0 |  |  |  |

指标口径：**ROxford-Hard / RParis-Hard / AmsterTime 为 mAP（%），是"图搜图"实例级检索的核心指标**（冻结特征 + 余弦相似度最近邻）；ImageNet k-NN 为最近邻分类 top-1（%），可作检索质量代理。 数值取自 **DINOv2 论文 [2304.07193](https://arxiv.org/abs/2304.07193) Table 9（实例检索）与 Table 4（k-NN）**；参数/维度见论文 Table 17 及 Meta 官方模型卡。 **关键对照**：参数远大的 OpenCLIP ViT-G/14 在实例检索上显著落后（ROxford-H 19.7 vs DINOv2-B 49.5、RParis-H 60.2 vs 78.6），印证**纯图搜图应选 DINO 系而非 CLIP**（特征差异见 [arXiv:2510.11835](https://arxiv.org/html/2510.11835v1)）。 注：个别对象实例检索设定下 DINOv2 也可能不及初代 DINO（[arXiv:2401.00463](https://arxiv.org/abs/2401.00463)），上线前建议在自有数据实测。

### text-text 候选：中文文本 embedding（C-MTEB 实测）

| 模型 | 参数量 | 维度 | C-MTEB 平均 | Retrieval | 许可 |
| --- | --- | --- | --- | --- | --- |
| **bge-large-zh-v1.5** | 326M | 1024 | **64.53** | 70.46 | MIT（可商用） |
| bge-base-zh-v1.5 | ~102M | 768 | 63.13 | 69.49 | MIT（可商用） |
| bge-small-zh-v1.5 | ~24M | 512 | 57.82 | 61.77 | MIT（可商用） |
| Qwen3-Embedding-0.6B* | 0.6B | 1024 | —（MTEB 多语 64.33） | — | Apache-2.0 |

来源：bge 系列 C-MTEB 分数取自 [BAAI 官方模型卡](https://huggingface.co/BAAI/bge-large-zh-v1.5)（C-MTEB：31 数据集 / 6 任务，平均、Retrieval 等分任务分）； bge-base/small 参数为社区通行值。*Qwen3-Embedding-0.6B 单模型即占满 0.6B 预算，**并发场景下无余量再放视觉模型**，仅在"单模型上限/可时分复用"时才适用；其 64.33 为 MTEB 多语口径，**与 C-MTEB 不可直接横比**（来源 siliconflow / Vercel 模型页）。

> **说明**：**落地要点：** ① 两个模型分别 int8/int4 量化后，索引存储与算力可再降，配合本机量化工具链即可； ② DINOv2 走 Apache-2.0，商用最省心；若选 DINOv3 需确认其自定义 License 条款； ③ 若担心 DINO 在编译工具链上的适配成本，可退而用 SigLIP2-Base 视觉塔（图搜图质量略降但集成更顺）； ④ 本方案**不含跨模态"文搜图"**——若后续需要，需另配一个双塔 CLIP（如 Chinese-CLIP ViT-B/16，188M），届时要重新核算并发预算。

---

*数据与结论主要参考 MMEB / MMEB-V2 基准及 Qwen3-VL-Embedding、RzenEmbed、VLM2Vec 等技术报告， 部分内容经改写以符合引用规范。榜单为实时更新，具体分数请以 [MMEB Leaderboard](https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard) 最新版本为准。 生成时间：2026-07-02*
