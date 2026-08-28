> 原文: [arXiv:2412.13663](https://arxiv.org/abs/2412.13663)（2024 年 12 月 19 日 v2）
> 说明: 本文为论文全文中文技术展开，公式/表格编号与原文一致；原文正文以**表格**为主，未附独立方法示意图（架构信息见 Table 3、Table 4，效率/评测见 Table 1、Table 2、Table 5-11），本译稿相应以"表格 + 公式 + 文字"覆盖，正文提到"图见原文"处对应原文表格与文字描述。

**预印本信息：** arXiv:2412.13663v2 [cs.CL]，2024 年 12 月 19 日。

**开源：** https://github.com/AnswerDotAI/ModernBERT （Apache 2.0 许可）；FlexBERT 框架（基于修改后的 MosaicBERT 代码库）与全部中间训练检查点同步开源。

---

# Smarter, Better, Faster, Longer：一个面向快速、显存友好、长上下文微调与推理的现代双向编码器（Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference）

**作者：** Benjamin Warner†、Antoine Chaffin†、Benjamin Clavié†、Orion Weller、Oskar Hallström、Said Taghadouini、Alexis Gallagher、Raja Biswas、Faisal Ladhak\*、Tom Aarsen、Nathan Cooper、Griffin Adams、Jeremy Howard、Iacopo Poli（†核心作者；\*工作完成于 Answer.AI 期间）

**单位：** Answer.AI、LightOn、Johns Hopkins University、NVIDIA、HuggingFace

**通讯邮箱：** `{bw,bc}@answer.ai`，`antoine.chaffin@lighton.ai`

---

## 摘要（Abstract）

以 BERT 为代表的 **encoder-only** 模型在检索与分类任务上具有优异的"性能/参数量"折中，长期是众多生产管线的主力。但自 2019 年 BERT 发布以来，编码器方向鲜有系统的 Pareto 级改进。作者提出 **ModernBERT**：把近年 decoder-only LLM 上验证过的架构改进系统性移植到编码器上，用 **2 万亿 token** 训练、原生 **8192** 序列长度，覆盖多样分类任务、单向量与多向量检索（含代码），在同参数量级各类下游任务上刷新 SOTA。同时它是当前推理与显存效率最好的编码器，专门面向常见 GPU 优化推理。

---

## 1 引言（Introduction）

**背景与动机。** 尽管 GPT/Llama/Qwen 等 decoder-only LLM 火热，encoder-only 模型仍在检索与判别式任务上被广泛使用：
- **推理便宜**：可批量处理大规模文档；
- **参数量/质量折中好**：在同等下游性能下比 encoder-decoder 或 decoder-only LLM 更小；
- **RAG 一级召回**、语义搜索、分类、NER、agent 路由、有害内容检测等场景都离不开小编码器。

然而，这些管线普遍还在用 2019 年的原版 BERT 或 RoBERTa 作为骨干，面临诸多痛点：
- 序列长度被卡在 **512**；
- 模型设计与词表尺寸都停留在过去的最佳实践（如没有考虑硬件对齐）；
- 训练数据体量小、领域窄，**普遍缺少代码数据**；
- 训练语料时间截止早，无法覆盖新事件。

**已有现代化尝试的局限。** MosaicBERT、CrammingBERT、AcademicBERT 主要优化训练效率，仍以对齐原 BERT 精度为目标；NomicBERT、GTE-en-MLM 引入了长上下文，但更偏向检索、忽视了分类与推理效率，且训练数据仍偏旧，尤其欠缺代码。

**贡献。** 本文提出 ModernBERT：
1. 系统整合 RoPE、GeGLU、Pre-LN、alternating global/local attention、unpadding、Flash Attention 2/3、`torch.compile` 等；
2. 面向硬件设计 "Deep & Narrow" 骨架，最大化在 T4/A10/L4/RTX 3090/RTX 4090/A100/H100 上的利用率；
3. **2T token 训练**，数据混合了 web、代码、STEM/科技文献；
4. 发布 base（149M）与 large（395M）两个尺寸，全面刷新编码器 SOTA；
5. 处理 8192 长序列的吞吐比现有编码器快近 2 倍；
6. 同步开源模块化框架 **FlexBERT**（基于修改后的 MosaicBERT 代码库），并像 Pythia 一样开源全部中间训练检查点。

---

## 2 方法（Methods）

### 2.1 架构改进（Architectural Improvements）

作者把标准 Transformer 分成三层改造：现代 Transformer 通用改动（§2.1.1）、效率导向的架构与实现改动（§2.1.2）、GPU 感知的模型形状选择（§2.1.3）。每一项都做了消融（详见附录 D）。

#### 2.1.1 现代 Transformer 通用改动

**去 bias。** 参照 Dayma et al.（2021），除最终解码线性层外的所有线性层都去掉 bias；Layer Norm 也全部去掉 bias。作者的假设是把参数预算更多放到主要线性权重上；保留解码 bias 是为了缓解 embedding 与解码权重 tying 带来的负面效应。

**位置编码：RoPE。** 用旋转位置编码（RoPE，Su et al., 2024）替代绝对位置编码。选它的理由是：短/长上下文都被大量论文验证过、主流框架实现成熟、易做上下文外推。

**归一化：Pre-LayerNorm。** 采用 Pre-Norm + 标准 LayerNorm，训练更稳。类似 CrammingBERT，在 embedding 层之后再加一层 LayerNorm；为了避免第一层里 LN 重复，第一层注意力块的第一个 LN 被移除。

**激活：GeGLU。** 用 GeGLU（Shazeer, 2020）替代原 BERT 的 GeLU。GeGLU 是 GLU 家族在 GeLU 之上的变体，多项工作证明它带来稳定的下游收益。

#### 2.1.2 效率导向改进

**交替全局/局部注意力（Alternating Attention）。** 借鉴 Gemma 2 的做法，ModernBERT 的注意力层在**全局**与**局部滑动窗口**间交替：
- **每 3 层 1 次全局注意力**：任意 token 可以看到序列中的任意 token，RoPE θ = 160,000；
- **其余层用 128 token 的局部滑窗**（Longformer 风格）：RoPE θ = 10,000。

这样在保证长依赖建模能力的同时，大幅降低长上下文推理开销。作者的消融表明：即使在 100B token 规模下，"每 3 层 1 次全局 + 其余 128 局部"和"全局每层"下游几乎完全打平，但速度大幅提升。

**Unpadding（去填充）。** 编码器传统做法是把一个 batch 里的所有序列 pad 到相同长度，这会在语义上无意义的 padding token 上浪费算力。ModernBERT 沿用 MosaicBERT 与 GTE 的 unpadding：
- **训练与推理都开**；
- 把 batch 里所有序列**串接成一条大序列**，作为 batch size = 1 处理；
- 之前的 unpadding 实现会在不同层里反复 unpad/repad，浪费显存带宽；
- ModernBERT 直接借助 Flash Attention 的 **variable-length attention** 与 **RoPE** 内核，允许 jagged mask 与 RoPE 直接作用在未 pad 的连续序列上；
- 输入进入 token embedding 前就完成 unpad，输出层可选 repad；
- 相比其他 unpadding 方案再提升 **10-20% 吞吐**。

**Flash Attention 2/3 混合。** 在启动本工作时，Flash Attention 3（面向 H100）不支持滑动窗口。因此 ModernBERT **全局层用 FA3、局部层用 FA2**，都能吃到内存与算力最优内核。

**`torch.compile`。** 用 PyTorch 2 的图编译对所有兼容模块整体编译，吞吐再涨 ~10%，编译开销可忽略。

#### 2.1.3 硬件感知的模型形状（Deep & Narrow）

同参数量下，"深而窄"的模型在下游任务上强于"浅而宽"（Tay et al., 2022；Liu et al., 2024），但推理会稍慢。Anthony et al.（2024）进一步指出：**硬件感知**的模型设计能显著提高运行时利用率。作者以此为指导，在满足以下约束的前提下把 ModernBERT 做得尽量"深"：

- **Tensor Core 对齐**：权重矩阵维度可被 64 整除；
- **Tile Quantization**：权重矩阵可被 128 × 256 分块；
- **Wave Quantization**：块数可被目标 GPU 的 SM 数整除（跨多 GPU 时无法严格满足，作者按 SM 利用率启发式取折中）；
- 目标 GPU 篮子：服务器端 T4/A10/L4/A100/H100 + 消费端 RTX 3090/4090，其中**更偏向推理 GPU**（不含 A100/H100）。

最终形状（详见 Table 4）：

- **base**：22 层，hidden 768，GLU 中间维 2304，12 头，共 149M；
- **large**：28 层，hidden 1024，GLU 中间维 5248，16 头，共 395M。

### 2.2 训练（Training）

#### 2.2.1 数据

**混合数据。** 两个尺寸都在**约 2 万亿 token**主要英文数据上训练，来源包括 web 文档、代码、科学文献；最终混合比例由一系列消融决定。

**Tokenizer。** 抛弃老 BERT 的词表，改用**改造过的 OLMo BPE tokenizer**：token 效率更高、对代码更友好，同时保留 `[CLS]/[SEP]` 与模板以做向后兼容。词表大小 **50,368**（= 64 的倍数），并预留 83 个未使用 token 供下游任务扩展。避免了 T5 词表不含花括号那类历史坑。

**序列打包（Sequence Packing）。** unpadding 后各 minibatch 有效大小方差较大。作者用贪心序列打包（Raffel et al., 2020；Krell et al., 2022），把小样本贴成长样本，**packing 效率 >99%**，训练 batch 均匀。

#### 2.2.2 训练设置

**MLM 目标。** 沿用 MosaicBERT 的 MLM 设置：
- **去掉 NSP**（对性能无益、只是开销）；
- **掩码率 30%**（原始 15% 已被 Wettig et al., 2023 证明次优）。

**优化器：StableAdamW。** 在 AdamW 之上加了 **Adafactor 风格的按参数 update clipping**，作为每参数学习率的自适应调节。相比标准 gradient clipping，在下游任务上更强、训练更稳。

**学习率调度：Warmup-Stable-Decay（WSD，梯形调度）。** 短暂 warmup → 长期恒定 → 短暂衰减。
- 匹配 cosine 调度性能，但**允许在任意 checkpoint 上继续训练而不冷启动**；
- 衰减阶段用 **1 − √t** 而非线性/余弦，作者的消融显示 1-sqrt 更优。

**base 训练日程：** 3B token warmup → **1.7T token 恒定 LR = 8e-4**。

**large 训练日程：** 2B token warmup → **900B token 恒定 LR = 5e-4**。在 5e-4 训练到几百亿 token 时 loss 平台化，于是**回滚重启**，用更低 LR = **5e-5** 继续训练剩下的 800B token（同时 weight decay 从 1e-5 降到 1e-6）。作者以此警示"常数 LR"的风险：LR 过高或 batch 过小时，其他调度可以靠衰减自救，常数 LR 不能。

**Batch Size Schedule。** batch 从小到大 warmup，可用低效的初始权重上少浪费算力：
- **base**：50B token 内从 768 → 4608；
- **large**：10B token 内从 448 → 4928；
- 非均匀 token 分配，保证每档 batch size 更新步数一致。

**权重初始化与 tiling。** base 用 Megatron 风格随机初始化；**large 直接从 base 权重"tile"过来**：借鉴 Phi 家族（Li et al., 2023；Javaheripi et al., 2023），采用"中心 tile + wraparound"—— base 权重放在 large 权重矩阵的中心（对齐每个 token embedding 与 attention head），四周用 wraparound 填充，并叠加 Gopher layer scaling。消融表明该初始化极大加速 large 的初期 loss 下降。

**上下文扩展（1024 → 8192）。** base/large 前期都在 1024 序列长度、RoPE θ = 10,000 上训练 1.7T token；随后：
1. **Phase One**（250B token）：把全局注意力 RoPE θ 提到 **160,000**、序列长度提到 **8192**，在**原预训练数据分布**上按 Fu et al.（2024）采样的长文本进行常数 LR 训练；base 用 3e-4，large 用 5e-5；
2. **Phase Two**（50B token）：按 Gao et al.（2024）**上采样更高质量长文本**，配 **1 − √t** LR 衰减。

作者的消融显示单独用其中一种策略都会伤到检索或分类某一头，两阶段组合最平衡。

**权重衰减。** 不对 bias 与 norm 层做 weight decay；使用 Loshchilov & Hutter（2019）的 **fully decoupled** 权重衰减，而非 PyTorch 的默认解耦。

**最终 checkpoint 选择。** 借鉴近期 checkpoint averaging 工作：
- **base**：三个最好的 annealing checkpoint + 最终 checkpoint 做平均；
- **large**：averaging 未能进一步涨点，直接取最好 annealing checkpoint。
- 尝试过 Llama 3 那种 EMA，未见提升。

### 训练配置汇总（Table 3 中译）

| 项 | 主预训练（base） | 主预训练（large） | 上下文扩展 Phase 1（base） | Phase 1（large） | Phase 2（base） | Phase 2（large） |
|---|---|---|---|---|---|---|
| 训练 token 数 | 1.719 万亿 | 1.719 万亿 | 2500 亿 | 2500 亿 | 500 亿 | 500 亿 |
| 最大序列长度 | 1,024 | 1,024 | 8,192 | 8,192 | 8,192 | 8,192 |
| Batch Size | 4,608 | 4,928 | 72 | 77 | 72 | 78 |
| LR warmup（token） | 500 亿 | 100 亿 | - | - | - | - |
| Microbatch | 96 | 56 | 12 | 7 | 12 | 6 |
| 学习率 | 8e-4 | 5e-4 → 5e-5 | 3e-4 | 5e-5 | 3e-4 | 5e-5 |
| LR 调度 | 梯形（WSD） | 梯形 | 常数 | 常数 | 1-√t 衰减 | 1-√t 衰减 |
| Warmup（LR，token） | 30 亿 | 20 亿 | - | - | - | - |
| Decay（token） | - | - | - | - | 500 亿 | 500 亿 |
| Weight Decay | 1e-5 | 1e-5 → 1e-6 | 1e-5 | 1e-6 | 1e-5 | 1e-6 |
| 总耗时（小时） | 194.2 | 425.3 | 39.9 | 80.7 | 11.5 | 21.7 |
| 训练耗时（小时） | 191.1 | 420.4 | 36.3 | 75.1 | 7.5 | 15.3 |
| 初始化 | Megatron | 从 base tile | - | - | - | - |
| 注意力 Dropout | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| 其他层 Dropout | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Optimizer | StableAdamW | StableAdamW | 同 | 同 | 同 | 同 |
| Betas | (0.90, 0.98) | 同 | 同 | 同 | 同 | 同 |
| Epsilon | 1e-06 | 同 | 同 | 同 | 同 | 同 |
| 硬件 | 8× H100 | 8× H100 | 同 | 同 | 同 | 同 |
| 并行策略 | DDP | DDP | 同 | 同 | 同 | 同 |
| 框架 | PyTorch 2.4.0, CUDA 12.4.0, Composer 0.24.1, FA 2.6.3, FA3 commit 32792d3 |

### 模型设计汇总（Table 4 中译）

| 项 | base | large |
|---|---|---|
| 词表大小 | 50,368 | 50,368 |
| 保留未用 token | 83 | 83 |
| 层数 | 22 | 28 |
| Hidden Size | 768 | 1024 |
| Transformer 块 | Pre-Norm | Pre-Norm |
| 激活函数 | GeGLU（表内简写为 GeLU） | GeGLU |
| 线性层 bias | 无 | 无 |
| 注意力 | Multi-head | Multi-head |
| Head 数 | 12 | 16 |
| 全局注意力 | 每 3 层 1 次 | 每 3 层 1 次 |
| 局部窗口 | 128 | 128 |
| 中间维（Intermediate） | 1,152 | 2,624 |
| GLU 扩展 | 2,304 | 5,248 |
| 归一化 | LayerNorm | LayerNorm |
| Norm ε | 1e-5 | 1e-5 |
| Norm bias | 无 | 无 |
| 全局注意力 RoPE θ | 160,000 | 160,000 |
| 局部注意力 RoPE θ | 10,000 | 10,000 |

> 注：Table 4 原文 Activation 行写作 `GeLU`（GeGLU 由 GeLU 门控组合而成，作者惯常口径），与 §2.1.1 描述 "GeGLU" 一致。

---

## 3 下游评测（Downstream Evaluation）

作者对 ModernBERT-base（149M）与 ModernBERT-large（395M）在**分类、单向量检索、多向量检索、长上下文检索、代码检索**五类任务上做了系统评测，对比对象：

- **base 组**（<150M）：BERT-base、DeBERTa-v3-base（183M，稍大）、RoBERTa-base、NomicBERT（8192）、GTE-en-MLM-base（8192）；
- **large 组**（300M-500M）：BERT-large-uncased、DeBERTa-v3-large、RoBERTa-large、GTE-en-MLM-large。

### 3.1 评测设置

**3.1.1 GLUE（NLU）。** 用 GLUE dev 集在每个子任务上分别做超参搜索（LR、weight decay、epoch），是公认的编码器 NLU 基准。搜索空间与最终参数见附录 E.1、Table 6。

**3.1.2 短上下文检索（BEIR）。**
- **DPR**（单向量）：用 MS-MARCO + 挖掘硬负例（Xuan et al., 2020）在 1.25M 样本、batch 16、5% warmup 下用 `sentence-transformers` 微调；
- **ColBERT**（多向量）：走 JaColBERTv2.5 训练路线（ColBERTv2 更新版），以 BGE-M3 为 teacher，KL 散度做 distillation，810k MS-MARCO 样本、batch 16、5% warmup，代码库 PyLate；
- 报告 BEIR 15 个子集的 nDCG@10 平均值。

**3.1.3 长上下文检索。** 用 MLDR 英文子集（>200k 长文档）分三种设置：
- **单向量 – Out-Of-Domain**：只在短上下文 MS-MARCO 上训练，直接在 MLDR 上评；
- **单向量 – In Domain**：MS-MARCO 上先训练，再在 MLDR 训练集上继续微调后评；
- **多向量 – Out-Of-Domain**：ColBERT 天然可泛化到长上下文，直接用 §3.1.2 的最好 checkpoint 评。

**3.1.4 代码检索。** 在 CoIR 框架下评：
- **CodeSearchNet**（CSN）：给一段代码，找对应 docstring/注释；
- **StackOverflow-QA**（SQA）：文本+代码混合的长文档，query/document 平均 1400/1200 词，token 数常超 2000。

### 3.2 主结果（Table 1 中译）

Table 1 汇总五类任务的核心指标（分类为 GLUE，检索为 nDCG@10，代码为 CoIR 平均）。MLDR<sub>ID</sub> 表示在 MLDR 训练集上微调后的 in-domain 评测；MLDR<sub>OOD</sub> 表示未在 MLDR 上微调直接评。

**Base（<150M）：**

| 模型 | BEIR (DPR) | MLDR<sub>OOD</sub> (DPR) | MLDR<sub>ID</sub> (DPR) | BEIR (ColBERT) | MLDR<sub>OOD</sub> (ColBERT) | GLUE | CodeSearchNet | StackQA |
|---|---|---|---|---|---|---|---|---|
| BERT | 38.9 | 23.9 | 32.2 | 49.0 | 28.1 | 84.7 | 41.2 | 59.5 |
| RoBERTa | 37.7 | 22.9 | 32.8 | 48.7 | 28.2 | 86.4 | 44.3 | 59.6 |
| DeBERTaV3 | 20.2 | 5.4 | 13.4 | 47.1 | 21.9 | **88.1** | 17.5 | 18.6 |
| NomicBERT | 41.0 | 26.7 | 30.3 | 49.9 | 61.3 | 84.0 | 41.6 | 61.4 |
| GTE-en-MLM | 41.4 | **34.3** | **44.4** | 48.2 | 69.3 | 85.6 | 44.9 | 71.4 |
| **ModernBERT** | **41.6** | 27.4 | 44.0 | **51.3** | **80.2** | 88.4 | **56.4** | **73.6** |

**Large（300M–500M）：**

| 模型 | BEIR (DPR) | MLDR<sub>OOD</sub> (DPR) | MLDR<sub>ID</sub> (DPR) | BEIR (ColBERT) | MLDR<sub>OOD</sub> (ColBERT) | GLUE | CodeSearchNet | StackQA |
|---|---|---|---|---|---|---|---|---|
| BERT | 38.9 | 23.3 | 31.7 | 49.5 | 28.5 | 85.2 | 41.6 | 60.8 |
| RoBERTa | 41.4 | 22.6 | 36.1 | 49.8 | 28.8 | 88.9 | 47.3 | 68.1 |
| DeBERTaV3 | 25.6 | 7.1 | 19.2 | 46.7 | 23.0 | **91.4** | 21.2 | 19.7 |
| GTE-en-MLM | 42.5 | **36.4** | **48.9** | 50.7 | 71.3 | 87.6 | 40.5 | 66.9 |
| **ModernBERT** | **44.0** | 34.3 | 48.6 | **52.4** | **80.4** | 90.4 | **59.5** | **83.9** |

### 3.3 结果讨论

**短上下文检索。** 两个尺寸的 ModernBERT 在 DPR 和 ColBERT 下的 BEIR 平均分都超过所有编码器基线，包括专为检索设计的近期模型 NomicBERT 与 GTE-en-MLM。base 只在 DPR 上略微超过 GTE-en-MLM-base，large 优势更明显（44.0 vs 42.5），且参数更少（395M vs 435M）。

**长上下文单向量。** 在 MLDR 上未做长文微调时（OOD），ModernBERT 显著优于短上下文模型和 NomicBERT，但落后 GTE-en-MLM 一截；一旦允许 in-domain 微调，两者几乎持平。作者认为这可能与局部注意力占比更高、以及 GTE-en-MLM 花在长序列上的预训练算力更多有关，具体成因留待后续研究。

**长上下文多向量（ColBERT）。** 所有长上下文编码器（GTE-en-MLM、NomicBERT、ModernBERT）都相比短上下文模型高 40 分以上，验证了 Bergum（2024）的观察：ColBERT 天然适合长文本。而在长上下文模型内部，ModernBERT 在 base 与 large 上都比次佳多向量结果领先至少 **9 分 nDCG@10**。作者推测这归因于：
1. 长时长的预训练让每个 token 都被充分训练；
2. 局部注意力与 ColBERT 的 MaxSim（token 级）粒度天然契合。

**NLU（GLUE）。** ModernBERT-base 是**第一款用纯 MLM 就超过 DeBERTa-v3-base（RTD 目标）的模型**（88.4 vs 88.1），出乎意料——过去普遍认为 RTD 更适合 NLU。ModernBERT-large 90.4，是 large 组第二名，仅略落后 DeBERTa-v3-large 的 91.4，但参数少约 10%、推理快约 2 倍。

**代码任务。** 在 CodeSearchNet 与 StackQA 上 ModernBERT 大幅领先——**56.4/73.6（base）、59.5/83.9（large）**——因为它是唯一在预训练时纳入代码数据、且使用代码友好 tokenizer 的编码器。同时它并未牺牲自然文本处理能力。

---

## 4 效率（Efficiency）

### 4.1 评测协议

在**单卡 NVIDIA RTX 4090**（ModernBERT 设计瞄准的推理 GPU 之一）上，用 4 组合成数据集，每组 8192 篇文档、取 10 次平均：

- **fixed short**：全部 512 token；
- **fixed long**：全部 8192 token；
- **variable short**：长度按均值 256 的正态分布采样；
- **variable long**：长度按均值 4096 的正态分布采样。

GTE-en-MLM 分别在**原生**与**接入 xformers**（可用 unpadding）两种配置下评。数据统计见 Table 10。

### 4.2 结果（Table 2 中译）

单位：显存里的最大 batch（BS），以及推理吞吐（千 token/秒）。

**Base：**

| 模型 | 参数量 | Short BS | Fixed Short | Var. Short | Long BS | Fixed Long | Var. Long |
|---|---|---|---|---|---|---|---|
| BERT | 110M | 1096 | 180.4 | 90.2 | — | — | — |
| RoBERTa | 125M | 664 | 179.9 | 89.9 | — | — | — |
| DeBERTaV3 | 183M | 236 | 70.2 | 35.1 | — | — | — |
| NomicBERT | 137M | 588 | 117.1 | 58.5 | 36 | 46.1 | 23.1 |
| GTE-en-MLM | 137M | 640 | 123.7 | 61.8 | 38 | 46.8 | 23.4 |
| GTE-en-MLM<sub>xformers</sub> | 137M | 640 | 122.5 | 128.6 | 38 | 47.5 | 67.3 |
| **ModernBERT** | 149M | **1604** | **148.1** | **147.3** | **98** | **123.7** | **133.8** |

**Large：**

| 模型 | 参数量 | Short BS | Fixed Short | Var. Short | Long BS | Fixed Long | Var. Long |
|---|---|---|---|---|---|---|---|
| BERT | 330M | 792 | 54.4 | 27.2 | — | — | — |
| RoBERTa | 355M | 460 | 42.0 | 21.0 | — | — | — |
| DeBERTaV3 | 434M | 134 | 24.6 | 12.3 | — | — | — |
| GTE-en-MLM | 435M | 472 | 38.7 | 19.3 | 28 | 16.2 | 8.1 |
| GTE-en-MLM<sub>xformers</sub> | 435M | 472 | 38.5 | 40.4 | 28 | 16.5 | 22.8 |
| **ModernBERT** | 395M | **770** | **52.3** | **52.9** | **48** | **46.8** | **49.8** |

**要点：**
- **短上下文**：ModernBERT 比其他近代编码器都快，只比原版 BERT/RoBERTa 略慢（原版参数更少）；
- **长上下文（8192）**：ModernBERT 在 base/large 上分别是次快模型的 **2.65× / 3×**；ModernBERT-large 处理 8192 的吞吐（46.8k tok/s）已接近 GTE-en-MLM-**base**（47.5k tok/s），远高于 GTE-en-MLM-large（16.5k tok/s）；
- **变长输入**：unpadding 让 ModernBERT 与 GTE-en-MLM 都显著领先，但 ModernBERT 短上下文再多 14.5–30.9%、长上下文再多 98.8–118.8%，主要功劳来自局部注意力；
- **显存**：ModernBERT-base 的最大 batch 是其他 base 编码器的**两倍以上**；ModernBERT-large 短上下文最大 batch 只稍逊 BERT-large，但在长上下文下比其他 large 至少大 60%。

Table 11（附录 F）给出了每种配置下带方差的绝对运行时长（毫秒）。DeBERTa-v3 显存占用是 ModernBERT 的 5–7 倍、速度慢约 2 倍，即使在所有序列都撑满 max length（unpadding 收益归零）的情形下也是如此。

---

## 5 结论（Conclusion）

ModernBERT 把 decoder-only LLM 上验证过的架构与训练升级系统性带回 encoder-only 家族，在原生 8192 序列长度下同时刷新分类、短上下文/长上下文单向量与多向量检索、代码检索的 SOTA：

- **GLUE**：ModernBERT-base 成为 2021 年以来首个 MLM 训练击败 DeBERTa-v3-base 的模型；
- **代码 + ColBERT 长上下文**：分别领先次佳 **6.85 / 9.1 分**；
- **短上下文检索**：单/多向量双设定都 SOTA；
- **效率**：短上下文比 DeBERTa-v3 快 2 倍，长上下文比次佳模型快 2 倍，显存占用同类最优；
- **首个开源、全模型链路 unpadding 的编码器**，也是首个真正做硬件感知设计的编码器。

---

## 6 局限（Limitations）

1. **语言**：只在英文上训练，对其他语言、尤其是低资源语言，不能直接迁移；
2. **偏见**：训练语料以 web 为主，代表性偏见与语料一致；
3. **有害生成风险**：MLM 让模型有能力替换 `[MASK]`，但 ModernBERT 不擅长自回归续写，出现长段有害内容的可能性远低于生成式 LLM；
4. **仅 MLM 目标**：DeBERTa-v3 在分类上强、检索上弱，可能 MLM + RTD 混合训练更平衡，值得后续研究；
5. **未探索的 scaling 维度**：本文重点是数据 + 架构 scaling，未系统扫参数量 scaling law。

---

## 附录 A 训练细节补充

**A.1 Batch Size Warmup。** batch size warmup 是训练中大 batch 时的经典手段：初期用较小 batch，让权重从"随机糟糕分布"上更新更快；等分布合理后再切到目标 batch。它相当于给一段隐式的高初始 LR + 迷你 LR decay。本文 base 与 large 的 warmup 见 §2.2.2。

**A.2 权重 Tiling（Weight Tiling）。** 借鉴 Phi 家族：`ModernBERT-large` 直接从 `ModernBERT-base` 预训练权重开始——用中心 tiling + wraparound，同时对 token embedding 和每个 attention head 都做居中，缺失部分用 wraparound。作者尝试过"中心 + 随机边缘"和"从边缘 tile"两种变体，都不如中心 tile + wraparound；配合 Gopher layer scaling，large 训练初期 loss 下降大幅加速。

**A.3 Weight Decay。** bias 与 norm 层不做 weight decay；其余层用 fully decoupled weight decay（Loshchilov & Hutter, 2019）。

**A.4 Final Checkpoints。**
- **base**：三个最佳 annealing checkpoint + 最终 checkpoint 的平均值；
- **large**：averaging 未见提升，直接用最佳 annealing checkpoint；
- Llama 3 那种 annealing 期 EMA 在本工作中未见提升。

---

## 附录 B 模型形状设计（Model Design）

Anthony et al. (2024) 指出：在 fp16/bf16 下最大化 GPU 利用率需要满足：
- **Tensor Core**：权重维度可被 64 整除；
- **Tile Quantization**：权重可被 128 × 256 分块；
- **Wave Quantization**：块数可被 SM 数整除。

跨 GPU 时"wave quantization"无法严格满足。作者遂选定 GPU 篮子（T4/A10/L4/RTX 3090/RTX 4090/A100/H100），按"块数 mod SM 数"估算 SM 利用率，把该值作为启发式指标，权衡多种形状后选定 22/28 层 + 768/1024 hidden + 2304/5248 GLU 扩展。

---

## 附录 C 训练日志（部分）

**C.1 采样器 bug。** 首轮 base 预训练意外发散，loss 有缓慢锯齿并最终发散。虽用了 PyTorch 分布式随机采样器，但监控指标表明训练不是真的随机——作者与 OLMo 作者一样，最终定位到 **PyTorch 采样器在样本数介于 5 亿–10 亿之间时会返回顺序偏置的样本**。用 NumPy 的 `PCG64DXSM` 替换后修复。

**C.2 large 回滚。** large 在 5e-4 恒定 LR 下训到几百亿 token 后训练 loss、validation 与 MNLI live eval 都平台化；随即回滚，把 LR 降到 5e-5、weight decay 从 1e-5 降到 1e-6，剩余 800B token 稳定改善。同时期 base 用 8e-4 恒定 LR 全程持续但递减地改善。作者以此反思常数 LR 的风险：cosine 之类的调度靠"衰减"能自救 LR 过高或 batch 过小的问题，常数 LR 不能。

---

## 附录 D 架构消融（Architecture Ablations）

绝大多数消融在 8–20B token 规模上进行：

- **GeGLU vs SwiGLU**：几乎无差异，最终选 GeGLU；
- **RoPE 覆盖 head 维度的百分比**（50/75/100%）：低比例略优，但差距小；训练规模小，最终保守选 100%；
- **LayerNorm vs RMSNorm**：结果基本一致；虽然 RMSNorm 理论上更快，但当时 PyTorch 没有原生实现，为了开箱效率选 LayerNorm；
- **Parallel Attention（MLP 与 Attention 并行计算）**：在本文目标尺寸与预训练序列长度下加速有限、但显著伤下游，不采用；
- **交替 global/local**（每 3 层 1 全局 + 128 局部）：在 100B token 规模下与全 global 打平，速度大幅提升，采用；
- **Tokenizer**：老 BERT/RoBERTa tokenizer 在 MNLI 上有竞争力，但代码支持弱；Llama 2 tokenizer 反而伤下游；改造后的 OLMo tokenizer 最好。

---

## 附录 E 详细评测

### E.1 GLUE 全量结果（Table 5 中译）

Base：

| 模型 | 参数 | Seq | CoLA | SST-2 | MRPC | STS-B | QQP | MNLI | QNLI | RTE |
|---|---|---|---|---|---|---|---|---|---|---|
| BERT | 110M | 512 | 59.0 | 93.1 | 89.5 | 89.4 | 91.4 | 85.4 | 91.6 | 78.2 |
| RoBERTa | 125M | 512 | 63.6 | 94.8 | 90.2 | 91.2 | 91.9 | 87.6 | 92.8 | 78.7 |
| DeBERTa-v3 | 183M | 512 | 69.2 | 95.6 | 89.5 | 91.6 | 92.4 | 90.0 | 94.0 | 83.8 |
| MosaicBERT-128 | 137M | 128 | 58.2 | 93.5 | 89.0 | 90.3 | 92.0 | 85.6 | 91.4 | 83.0 |
| NomicBERT-2048 | 137M | 2048 | 50.0 | 93.0 | 88.0 | 90.0 | 92.0 | 86.0 | 92.0 | 82.0 |
| GTE-en-MLM | 137M | 8192 | 57.0 | 93.4 | 92.1 | 90.2 | 88.8 | 86.7 | 91.9 | 84.8 |
| **ModernBERT** | 149M | 8192 | **65.1** | **96.0** | **92.2** | **91.8** | **92.1** | 89.1 | 93.9 | **87.4** |

Large：

| 模型 | 参数 | Seq | CoLA | SST-2 | MRPC | STS-B | QQP | MNLI | QNLI | RTE |
|---|---|---|---|---|---|---|---|---|---|---|
| BERT | 330M | 512 | 56.2 | 93.3 | 87.8 | 90.6 | 90.9 | 86.3 | 92.8 | 83.8 |
| RoBERTa | 355M | 512 | 68.0 | 96.4 | 90.9 | 92.4 | 92.2 | 90.2 | 94.7 | 86.6 |
| DeBERTa-v3 | 434M | 512 | **75.3** | 96.9 | 92.2 | 93.0 | 93.3 | **91.8** | **96.0** | **92.7** |
| GTE-en-MLM | 434M | 8192 | 60.4 | 95.1 | **93.5** | 91.4 | 89.2 | 89.2 | 93.9 | 88.1 |
| **ModernBERT** | 395M | 8192 | 71.4 | **97.1** | 91.7 | 92.8 | 92.7 | 90.8 | 95.2 | 92.1 |

超参搜索空间：
- LR ∈ {1e-5, 3e-5, 5e-5, 8e-5}
- Weight decay ∈ {1e-6, 5e-6, 8e-6, 1e-5}
- Epochs：SST-2/MNLI/RTE 用 {1,2,3}；QNLI/QQP/CoLA/MRPC/STS-B 用 {2,5,10}
- 所有微调都用 early stopping；RTE/MRPC/STS-B 从 MNLI checkpoint 起始。最终各任务超参见原文 Table 6。

### E.2 BEIR 详表（略取）

单向量（Table 7）与多向量（Table 8）分别给出 15 个子集的 nDCG@10；ModernBERT 在 TREC-COVID 上大幅领先，一部分原因是训练截止时间更近，但 NomicBERT / GTE 也训过更近的数据，因此 knowledge cutoff 并非唯一因素。为每个模型选择最终 checkpoint 时，作者在 NFCorpus、SciFact、TREC-COVID、FiQA 的平均分上做 LR 扫描（LR ∈ {1e-5, 2e-5, 3e-5, 5e-5, 8e-5, 1e-4}），最终选定的 LR 见 Table 9。

BEIR 15 子集平均（nDCG@10，节选自 Table 7 / Table 8，完整数值见原文）：

- **DPR 单向量**：ModernBERT-base **41.6**（>GTE 41.4, NomicBERT 41.0）；ModernBERT-large **44.0**（>GTE 42.5, RoBERTa 41.4）；
- **ColBERT 多向量**：ModernBERT-base **51.3**（>NomicBERT 49.9）；ModernBERT-large **52.4**（>GTE 50.7）。

---

## 附录 F 效率合成数据集统计（Table 10 中译）

| 项 | Short-Fixed | Short-Variable | Long-Fixed | Long-Variable |
|---|---|---|---|---|
| 总 token 数 | 4,194,304 | 2,096,510 | 67,108,864 | 33,604,913 |
| 标准差 | 0 | 64 | 0 | 1,024 |
| 平均长度 | 512 | 256 | 8,192 | 4,102 |
| 最长序列 | 512 | 476 | 8,192 | 7,624 |
| 最短序列 | 512 | 32 | 8,192 | 171 |
| 序列数 | 8,192 | 8,192 | 8,192 | 8,192 |

Table 11（部分列出的推理绝对运行时长，单位秒，10 次平均 ± 标准差）显示 DeBERTa-v3 在 base 层级已达 59.7 秒，是 ModernBERT-base 的 ~2 倍；large 层级 DeBERTa-v3 170.8 秒，是 ModernBERT-large 的 ~2.1 倍。

---

## 术语与翻译约定

| 英文 | 中文 |
|---|---|
| encoder-only / decoder-only | 编码器（only 结构）/ 解码器（only 结构） |
| bidirectional encoder | 双向编码器 |
| unpadding | 去填充 |
| sequence packing | 序列打包 |
| alternating global/local attention | 交替全局/局部注意力 |
| sliding window attention | 滑动窗口注意力 |
| Rotary Positional Embedding (RoPE) | 旋转位置编码 |
| Pre-LayerNorm / Pre-Norm | 前置层归一化 |
| Gated Linear Unit (GLU) / GeGLU | 门控线性单元 / GeGLU |
| Masked Language Modeling (MLM) | 掩码语言建模 |
| Replaced Token Detection (RTD) | 替换 token 检测 |
| Next Sentence Prediction (NSP) | 下一句预测 |
| Warmup-Stable-Decay (WSD) / trapezoidal schedule | 梯形学习率调度 |
| StableAdamW | StableAdamW 优化器 |
| Deep & Narrow | 深而窄结构 |
| Tensor Core Requirement | Tensor Core 对齐 |
| Tile Quantization / Wave Quantization | 分块/波次量化对齐 |
| Streaming Multiprocessor (SM) | 流式多处理器 |
| Dense Passage Retrieval (DPR) | 稠密段落检索（单向量） |
| ColBERT / MaxSim | 多向量检索 / 最大相似度算子 |
| Retrieval-Augmented Generation (RAG) | 检索增强生成 |
| BEIR / MLDR / MS-MARCO / CodeSearchNet / StackOverflow-QA | 直接沿用英文名 |
| checkpoint averaging | 检查点平均 |
| weight tiling / center tiling / wraparound | 权重 tile / 中心 tile / 环绕填充 |
| flash attention / variable-length attention | Flash Attention / 变长注意力 |
| xformers | 直接沿用英文名 |
| knowledge cutoff | 知识截止时间 |
