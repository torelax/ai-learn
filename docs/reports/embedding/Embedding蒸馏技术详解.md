# Embedding 蒸馏技术详解

> **读者前提**：已读 [主文 §3–§5](Embedding调研报告.md)（表示/交互、对比学习课表、难负例）。  
> **读完收获**：能选蒸馏信号、组损失、排训练阶段，并搭一条「大 Teacher → ≤0.6B Student」可验证实验。  
> **与总论关系**：[`../distillation/知识蒸馏技术深度调研报告.md`](../distillation/知识蒸馏技术深度调研报告.md) §7 为 **Embedding 蒸馏索引**；本文是 **机制与课表专文**，不重复 LLM token-KD（MiniLLM/OPD/R1-Distill 等见总论 §5）。

---

## 目录

1. [优化什么：与 LLM 蒸馏的边界](#1-优化什么与-llm-蒸馏的边界)
2. [蒸馏信号分类](#2-蒸馏信号分类)
3. [Cross-Encoder → Bi-Encoder：检索蒸馏经典路径](#3-cross-encoder--bi-encoder检索蒸馏经典路径)
4. [向量对齐与跨架构投影](#4-向量对齐与跨架构投影)
5. [多教师蒸馏（Jasper / Stella 路线）](#5-多教师蒸馏jasper--stella-路线)
6. [自蒸馏与集成教师（BGE-M3）](#6-自蒸馏与集成教师bge-m3)
7. [任务靶向蒸馏（jina-v5 等）](#7-任务靶向蒸馏jina-v5-等)
8. [多模态 Embedding 蒸馏](#8-多模态-embedding-蒸馏)
9. [Late Interaction / ColBERT 蒸馏](#9-late-interaction--colbert-蒸馏)
10. [训练课表：阶段、损失组合、与难负例的配合](#10-训练课表阶段损失组合与难负例的配合)
11. [实验协议与常见翻车](#11-实验协议与常见翻车)
12. [两条可照着做的配方](#12-两条可照着做的配方)
13. [参考文献](#13-参考文献)

---

## 1. 优化什么：与 LLM 蒸馏的边界

### 1.1 Embedding 蒸馏的目标函数

Embedding 模型在线上的「行为」是：

$$\text{rank}(d \mid q) = \text{sort}_{d \in \mathcal{C}} \; s_\theta(q, d), \quad s_\theta = \text{sim}(f_\theta(q), f_\theta(d)) \;\text{或 MaxSim 等}$$

蒸馏要迁移的是 **排序结构** 和/或 **几何结构**（向量夹角、相对距离），而不是下一个 token 的概率。

| 维度 | LLM 生成式 KD | Embedding KD |
|------|---------------|--------------|
| 教师输出 | 词表 logits / 生成序列 | 向量、相似度、候选排序 |
| 典型损失 | KL on vocab、SeqKD、OPD | KL/MSE on scores、cosine on vectors、InfoNCE+软标签 |
| 学生任务 | 续写、对话 | 检索、STS、聚类（以检索为主本文） |
| 能否黑盒 API | 可以（只要文本） | 可以（只要分数/embedding API） |
| 负样本 | 生成多样性 | **检索质量核心**（§5 主文、难负例专题） |

**错误用法**：把 MiniLLM 的 reverse-KL 直接套在 Bi-Encoder 上，却不构造 $(q,d)$ 候选集——优化目标与 ANN 检索无关。

### 1.2 为什么要蒸馏 Embedding（而不是只量化 Teacher）

| 手段 | 压缩什么 | 保留什么 | 局限 |
|------|----------|----------|------|
| **量化 INT8/INT4** | 权重/激活 bit | 同一模型的决策边界 | 8B 量化仍比 0.6B 慢；显存仍大 |
| **MRL 截维** | 向量维度 | 同一模型子空间 | 需要 MRL 训练或接受截断损失 |
| **Embedding KD** | 参数量 + 架构 | Teacher 的排序/几何 | 训练成本高；需对齐模板与负例 |

**≤0.6B 自训场景**（主文 §10.2）：常见路径是 **小 Student + 大 Teacher 蒸馏 + 领域难负例**，而不是把 Qwen3-Emb-8B 量化到端侧。

### 1.3 学生约束清单（设计实验前先填）

1. **参数量**（如 0.6B / 1.5B / 335M）  
2. **交互**：Bi Dense（默认）还是 Late Multi-Vector  
3. **模态**：纯文 / 图文双塔 / MLLM 单向量  
4. **维度与 MRL**：目标索引维度（256/768/1024）  
5. **延迟**：单条 query embed ms 级上限  
6. **接口**：是否与 Teacher 相同 instruct 模板  

---

## 2. 蒸馏信号分类

所有 Embedding 蒸馏都可归为：**教师提供什么监督 → 学生用什么损失接**。

### 2.1 总表

| 信号类型 | 教师提供 | 典型损失 | 最适用 | 主要风险 |
|----------|----------|----------|--------|----------|
| **分数 / 相似度** | $s_T(q,d)$ 或 $\cos(v_T(q),v_T(d))$ | MSE、Huber、KL on softmax | CE→Bi、API 黑盒 Teacher | 候选集不含 hard neg → 蒸馏太易 |
| **排序 / 列表** | Top-K 序或 Plackett-Luce 分布 | ListNet、ListMLE、KL on rank | 检索主任务 | 只蒸 Top-1 忽略尾部相关 doc |
| **向量** | $v_T(x)$ | $\|v_S - W v_T\|$、$1-\cos$ | 同任务、可投影对齐 | 维数/池化不一致硬对齐 |
| **关系 / 结构** | 样本间距离矩阵 | triplet、 pairwise RKD | 数据少 | 计算 $O(B^2)$ |
| **集成软标签** | 多路 $s$ 加权（Self-KD） | KL 到集成分布 | 多头统一（BGE-M3） | 权重未调 → 某路噪声主导 |

### 2.2 分数蒸馏：把 Teacher 排序写进损失

对固定 query $q$，候选集 $\mathcal{D}=\{d_1,\ldots,d_K\}$（含 $d^+$ 与 hard negatives），Teacher 打分 $s_i^T = s_T(q,d_i)$，Student $s_i^S$。

**MSE / Huber**（回归绝对分）：

$$\mathcal{L}_{\text{score}} = \frac{1}{K}\sum_i \ell(s_i^S, s_i^T)$$

**KL 蒸馏**（更常用，与 CE→Bi 一致）：对候选做温度 softmax

$$p_i^T = \frac{\exp(s_i^T/\tau_T)}{\sum_j \exp(s_j^T/\tau_T)}, \quad
p_i^S = \frac{\exp(s_i^S/\tau_S)}{\sum_j \exp(s_j^S/\tau_S)}$$

$$\mathcal{L}_{\text{KL}} = \mathrm{KL}(p^T \| p^S)$$

直觉：Student 不必复现 Teacher 的绝对余弦值，只需 **在候选集上的相对排序** 一致。$\tau_T$ 越大分布越软，暗知识越多；$\tau_S$ 常取 1 或与训练 $\tau$ 对齐。

**与 InfoNCE 的关系**：若候选集 = $\{d^+\}\cup\{d^-_k\}$ 且 $\tau$ 与对比学习一致，KL 蒸馏 ≈ 用 Teacher 软标签替代 one-hot 正类。

### 2.3 向量蒸馏

$$\mathcal{L}_{\text{vec}} = 1 - \cos(W h_S(x), h_T(x)) \quad \text{或} \quad \|W h_S - h_T\|_2^2$$

- $h$ 为 L2 归一化前的 pooled 向量；$W$ 为线性投影（Teacher 4096 → Student 768 时**必须训** $W$，不能随机投影就评测）。  
- **SimTDE**（SIGIR 2023）：同时蒸馏 token 级与句级，适合 Encoder 学生。  
- **TAVA**：Student 预训练 + MLP adapter 残差映射 Teacher 空间，追求极致压缩。

**翻车**：Teacher 用 last-token pooling、Student 用 mean pooling——不对齐池化就蒸向量，STS 可能涨、检索不涨。

### 2.4 关系蒸馏

对 batch 内样本，匹配 Teacher 的 pairwise 相似度矩阵：

$$\mathcal{L}_{\text{rel}} = \sum_{i<j} \left( \cos_S(i,j) - \cos_T(i,j) \right)^2$$

保 **流形形状**，标注少时有用；检索仍建议叠加 MNRL 或分数蒸馏。

### 2.5 与对比损失的混合（通用配方）

几乎从不「只蒸馏」：

$$\mathcal{L} = \lambda_{\text{distill}} \mathcal{L}_{\text{distill}} + \lambda_{\text{task}} \mathcal{L}_{\text{InfoNCE/MNRL}} + \lambda_{\text{aux}} \mathcal{L}_{\text{MRL}}$$

- **只蒸馏**：Student 易过拟合 Teacher 候选集，域外 Recall 掉。  
- **只对比**：小模型达不到 Teacher 排序细粒度。  
- **jina-v5** 等结论：**蒸馏 + 任务对比 > 纯蒸馏 > 纯对比**（见 §7）。

---

## 3. Cross-Encoder → Bi-Encoder：检索蒸馏经典路径

这是工业上最清晰的一条：**Teacher 精、Student 快**。

### 3.1 机制

```
离线:
  对每个训练 query q:
    用 BM25 / 上轮 Student 召回候选 {d_i}
    Teacher CE(q, d_i) → s_i^T
    写入 (q, d_i, s_i^T) 或 top-K 序

在线训练 Student Bi:
  s_i^S = cos(f_S(q), f_S(d_i))
  L = KL(softmax(s^T) || softmax(s^S))  +  λ · MNRL(q, d+)
```

**NV-Retriever**（Zhang et al., 2024, arXiv:2407.15831）与 **ColBERTv2** 训练都依赖 **强 Teacher 提供 hard negatives 或软标签**；Embedding 蒸馏与 **难负例挖掘** 是同一条流水线（见《[难负例挖掘工业实践](难负例挖掘工业实践.md)》）。

### 3.2 候选集怎么构造

| 来源 | 作用 |
|------|------|
| BM25 top-50 | 词法 hard neg |
| 上轮 Student top-50 | on-policy 难例 |
| 标注 $d^+$ | 锚点 |
| Teacher CE 过滤 | positive-aware 去假负例 |

**断环**：候选集只有 random neg → Teacher 分数全接近 0 → KL 梯度弱，Student 学不到排序。

### 3.3 Cross-Encoder 当 Teacher 的注意点

- CE 与 Bi **模板必须一致**（同一 max length、同一 q/d 拼接方式只在 CE 侧）。  
- CE 分数**未校准**时，MSE 不如 KL。  
- 域外 CE（通用 MS MARCO CE 训）蒸领域 Student → 需领域 CE 或 Bi Teacher 微调。

---

## 4. 向量对齐与跨架构投影

### 4.1 维数与 tokenizer 不一致

| 问题 | 表现 | 对策 |
|------|------|------|
| $d_T \neq d_S$ | 无法直接 MSE | 可训线性 $W: \mathbb{R}^{d_S}\to\mathbb{R}^{d_T}$ 或反过来 |
| tokenizer 不同 | 无法对齐 token hidden | **EMO**（MinED + CKA + OT）；或 **只蒸句子级向量/分数** |
| 池化不同 | 向量语义不对齐 | 统一池化策略，或只蒸 score |

**EMO**（EMNLP 2025）：Cross-Tokenizer KD，用编辑距离对齐 token，CKA 对齐注意力，OT 对齐 hidden——适合 **BERT Student ← LLM Teacher** 的分类/STS；检索仍推荐 **分数蒸馏 + MNRL** 更稳。

### 4.2 投影层训练

- $W$ 与 Student **联合训练**，不要冻结随机 $W$。  
- 先 Stage1 只训 $W$ + Student 顶层，再 Stage2 全量微调，避免 Student 随机初始化破坏 Teacher 几何。

---

## 5. 多教师蒸馏（Jasper / Stella 路线）

**Jasper / Stella**（Zhang et al., 2024, arXiv:2412.19048）代表 **无标注、多 Teacher、多阶段** 压缩 SOTA Embedding。

### 5.1 思路

Teacher 池：NV-Embed-v2、bge-en-icl、LLM2Vec 等（不同强项）。Student（如 2B）无法同时 MSE 三个向量空间，做法：

1. **拼接或加权 Teacher 向量** 到统一目标（或分别算 loss 再平均）。  
2. **可训投影** $W_k$ 把 Student 映射到各 Teacher 维。  
3. **Cosine + Triplet** 约束相对序。  
4. **MRL**：同一 Student 多截断维同时蒸馏。

示意（与总论 §7.3.2 一致，此处强调机制）：

$$\mathcal{L} = \lambda_1 \cos(W s, \text{concat}(t_1,t_2,\ldots)) + \lambda_2 \mathcal{L}_{\text{triplet}} + \lambda_3 \mathcal{L}_{\text{MRL}}$$

### 5.2 多教师何时有用、何时有害

| 情况 | 建议 |
|------|------|
| Teachers 在同一检索域、排序相关 | 加权融合软标签 |
| Teachers 一个偏 STS、一个偏 Retrieval | **分开任务损失**，不要平均向量 |
| Teachers 指令模板不一致 | 先统一模板再蒸馏 |
| 学生极小（<400M） | 单 Teacher 往往更稳；多 Teacher 易糊 |

### 5.3 与 Qwen3-Embedding 的关系

Qwen3-Emb **8B/4B/0.6B** 主要是 **合成数据 + 对比课表 + SLERP 合并**，不是「把 NV-Embed 蒸进 0.6B」的 Jasper 路线——但 **0.6B 档** 仍可再叠 **更大 Qwen3-Emb 或 API Teacher 的分数蒸馏** 做二次压缩（官方报告强调合成数据；工程上 Teacher 蒸馏是互补手段）。

---

## 6. 自蒸馏与集成教师（BGE-M3）

外部 Teacher 成本过高时，用 **模型自身多路输出** 作教师（详见《[BGE-M3三功能统一详解报告](BGE-M3三功能统一详解报告.md)》§8）。

### 6.1 机制

三路分数 $s_{dense}, s_{lex}, s_{mul}$ 异构 → 集成

$$s_{\text{inter}} = w_1 s_{dense} + w_2 s_{lex} + w_3 s_{mul}$$

对每一路用 $s_{\text{inter}}$ 的 softmax 分布做 KL，避免 Dense/Sparse/Late 三头互相打架。

### 6.2 与外部蒸馏的区别

| | Self-KD（M3） | 外部 Teacher |
|--|---------------|--------------|
| 教师来源 | 自身三头集成 | NV-Embed / CE / CLIP |
| 目标 | 多头一致、Hybrid 可用 | 小模型逼近大模型 |
| 风险 | 集成权重 $w_2$ 过大早期 Sparse 噪声 | Teacher 域不匹配 |

**可组合**：M3 式 Self-KD 训三头 + 另用 CE 蒸馏 Dense 头（工程复杂，需分阶段）。

---

## 7. 任务靶向蒸馏（jina-v5 等）

**jina-embeddings-v5-text**（2026）强调 **Task-Targeted Embedding Distillation**：

1. 大 Teacher 提供向量/分数（通用能力）。  
2. 叠加 **任务对比损失**（检索、STS、分类等按采样比例）。  
3. 结论顺序：**蒸馏+任务 > 纯蒸馏 > 纯对比**。

对你自己的 ≤0.6B 实验含义：

- 不要只蒸 MTEB 平均 Teacher；在 **目标检索域** 上保留 MNRL。  
- $\lambda_{\text{distill}} : \lambda_{\text{task}}$ 在验证集上网格搜索；常见起点 0.3–0.7 : 0.7–0.3。

---

## 8. 多模态 Embedding 蒸馏

主文 §10：图搜图 / 文搜图 / 视觉文档。蒸馏额外约束：**哪一塔对齐 Teacher**。

### 8.1 CLIP / SigLIP Teacher → 小双塔 Student

**目标**：小模型在 **共享图文空间** 里复现 Teacher 的 $\cos(v_I, v_T)$ 排序。

| 组件 | 蒸馏方式 |
|------|----------|
| 图像塔 | $\mathcal{L}_{\cos}$ 对齐 Teacher 图像向量，或 KL on 图文 batch 内相似度 |
| 文本塔 | 同上 |
| 对齐空间 | 只蒸 **同 batch 内** InfoNCE 的 Teacher 软标签（SigLIP 用 sigmoid 版） |

**图搜图**：可只蒸视觉塔（DINO Teacher → 小 ViT），但 **文搜图** 必须蒸文本塔或联合对比。

**≤0.6B 配方要点**（与 `0.6B图搜图文搜图自训学习行动路线.md` 一致）：

1. Teacher：OpenCLIP/SigLIP 2 或 GME-7B（MLLM 单向量）。  
2. Student：0.3B–0.6B ViT + Text Encoder 或小型 MLLM。  
3. 损失：$\lambda_1 \mathcal{L}_{\text{img-img}}$ + $\lambda_2 \mathcal{L}_{\text{text-img}}$ + $\lambda_3 \mathcal{L}_{\text{vec-KD}}$。  
4. 数据：标题–商品图弱对 + hard neg（同品类错 SKU）。  
5. **分开评测** Image→Image 与 Text→Image。

### 8.2 只蒸文本塔 vs 两塔同时蒸

| 策略 | 适用 |
|------|------|
| 只蒸文本塔 | 图库固定、视觉 Teacher 已够好（如冻结 DINO 库、只优化文搜图） |
| 两塔同蒸 | 图搜图 + 文搜图都要；Student 容量够 |
| MLLM 单向量 | GME 式：Teacher 是 MLLM，Student 模仿 **last-token 向量**（不是 ColPali 多向量） |

### 8.3 ColPali 类 Teacher → 小 Late Student

Teacher 输出 **patch 多向量**，Student 若也是 Late Interaction：

- 可蒸馏 **MaxSim 分数**（对同一 $(q, \text{page})$ 候选集 KL），比逐 patch MSE 更稳。  
- 存储仍随 patch 数线性涨；Student 通道数 $d'$ 可先压缩（ColBERTv2 式 residual compression）。  
- **可行性**：0.6B 级 ColQwen/ColSmol 路线 = 小 VLM + 蒸馏 ColQwen2 Teacher 的 **排序 + 中间层**；全 patch 向量 MSE 成本高，优先 **listwise 分数蒸馏**。

### 8.4 SigLIP 2 自蒸馏（视觉-语言 Encoder）

SigLIP 2 在 **同一族内** 用更大/更强 checkpoint 或 EMA 作 Teacher，蒸馏到更小 ViT——属于 **表示学习预训练** 环节，不是检索管线里的 CE→Bi，但可作为 **你的 Student 视觉塔初始化**。

---

## 9. Late Interaction / ColBERT 蒸馏

**ColBERTv2**（Khattab et al., 2022）：用 **Cross-Encoder 软标签** 训练 Late Student，是「CE → 可预计算结构」的典范。

要点：

- Teacher：CE 或 ColBERT 自身集成。  
- Student：token 向量维度压缩（残差量化）+ KL on MaxSim 分数。  
- **PLAID** 解决的是检索工程，不是蒸馏；但 Student 压缩后仍需 **Recall 回归**。

对 BGE-M3 **ColBERT 头**：外部 CE 蒸馏 Dense 头 + Self-KD 协调三头，是不同组合；M3 论文强调 **Self-KD 训三头**，CE 蒸馏是可选增强。

---

## 10. 训练课表：阶段、损失组合、与难负例的配合

### 10.1 推荐阶段（文本检索 Student）

```
Stage 0: Student 初始化（开源 BGE-small / MiniLM / Qwen3-Emb-0.6B 基座）
Stage 1: 领域 MNRL（弱监督对）— 撑开空间
Stage 2: Teacher 离线打分 (q, candidates) → KL 蒸馏 + MNRL  — 对齐排序
Stage 3: 刷新 hard negatives（Student 或 Teacher 检索）→ 重复 Stage 2
Stage 4 (可选): MRL 截断维 / INT8 量化 — 部署压缩（非 KD）
```

**先蒸再挖 vs 交替**：  
- **先蒸再挖**：Teacher 固定，Student 先收敛，再换 Student top-k 挖 neg（实现简单，易 stale）。  
- **交替**（推荐）：每 $N$ steps 用当前 Student 重挖 neg，Teacher 可周期性更新（NV-Retriever 思路）。  

### 10.2 损失权重起点（必须在验证集上改）

| 项 | 起点 | 调参信号 |
|----|------|----------|
| $\lambda_{\text{KL}}$ | 0.3–0.5 | Recall 不涨 → 略升；过拟合 Teacher → 略降 |
| $\lambda_{\text{MNRL}}$ | 0.5–0.7 | STS 掉 → 略升 MNRL |
| $\tau_T$ | 2–5（CE 分）或 0.02–0.05（cos 分） | 与 Teacher 分尺度有关 |
| hard neg / q | 2–4 | 假负例多 → 减 neg + positive-aware |

### 10.3 与 MRL / 量化的顺序

- **MRL**：可与 Stage 2 同时（$\mathcal{L}_{\text{MRL}}$ 加在 Student 向量上）。  
- **量化**：**训练后** PTQ/QAT；在蒸馏收敛后再做，避免量化噪声干扰 KD。

---

## 11. 实验协议与常见翻车

### 11.1 最小协议（与主文 §11.4 一致）

1. **固定验证集**：Recall@K / nDCG；与 Teacher **同 instruct、同归一化**。  
2. **对照组**：  
   - Student 无蒸馏，只 MNRL  
   - Student 只蒸 STS（预期：STS↑ Recall 平或↓）  
   - Teacher 上限（同库同模板）  
3. **报告**：参数量、维度、MRL/量化、索引类型。  
4. **失败签名**：STS↑ Recall↓ → 蒸馏目标或模板错；loss↓ Recall↓ → 假负例。

### 11.2 翻车对照

| 现象 | 机制原因 |
|------|----------|
| Student ≈ 随机 Teacher | 投影层未训；候选无 hard neg |
| 验证集虚高 | 测试 doc 进 Teacher 打分语料 |
| 线上比验证差 | 训练有 instruct、线上裸 query |
| 多 Teacher 更差 | 平均了矛盾排序 |
| 图文文搜图失败 | 只蒸了图像塔 |
| ColPali Student 巨大 | 逐 patch MSE 未做压缩 |

### 11.3 离线 Teacher 缓存

大 Teacher 推理贵，应 **流式写盘**：

```text
{"qid": "...", "query": "...", "doc_id": "...", "score_teacher": 0.82, "rank": 3}
```

支持 `--resume` 断点续跑；与 modelforge 流式规范一致。Student 训练只读缓存，可重复试验 $\lambda$。

---

## 12. 两条可照着做的配方

### 12.1 配方 A：8B Bi Teacher → 0.6B 文本 RAG Student

| 步 | 动作 |
|----|------|
| Teacher | Qwen3-Emb-8B 或 NV-Embed-v2；冻结 |
| Student | Qwen3-Emb-0.6B 或 BGE-base 级 Encoder |
| 数据 | 领域 $(q,d^+)$ + BM25/Student hard neg |
| Stage 1 | MNRL，2 epoch |
| Stage 2 | Teacher 对每 q 的 top-20 候选 KL + MNRL，$\lambda=0.4:0.6$ |
| Stage 3 | 每 2 epoch 用 Student 重挖 neg，重复 Stage 2 |
| 验证 | 领域 Recall@10；对照无 KD Student |
| 部署 | MRL 512d；Hybrid + BGE-Reranker（可选） |

### 12.2 配方 B：SigLIP 2 Teacher → 0.6B 图文 Student（文搜图 + 图搜图）

| 步 | 动作 |
|----|------|
| Teacher | SigLIP 2 或 OpenCLIP ViT-L |
| Student | 小 ViT + Text Transformer（总参 ≤0.6B） |
| 损失 | $\mathcal{L}_{\text{siglip}}$（Student 自身对比） + $0.5\cdot\mathcal{L}_{\text{KD-cos}}$ 对齐 Teacher 图文相似度 |
| 图搜图 | 同 batch 图像–图像 负样本 |
| 数据 | 商品 title–image；hard neg 同品类 |
| 验证 | **分开** T→I Recall@K 与 I→I Recall@K |
| 勿做 | 把图片路径字符串送文本 Embedding |

---

## 13. 参考文献

### 核心论文

1. Hinton et al. (2015). Distilling the Knowledge in a Neural Network. *arXiv:1503.02531*.  
2. Reimers & Gurevych (2019). Sentence-BERT. *EMNLP*. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)  
3. Karpukhin et al. (2020). DPR. *EMNLP*. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)  
4. Khattab & Zaharia (2020). ColBERT. *SIGIR*. [arXiv:2004.12832](https://arxiv.org/abs/2004.12832)  
5. Santhanam et al. (2022). ColBERTv2. *NAACL*. [arXiv:2112.01488](https://arxiv.org/abs/2112.01488)  
6. Xiao et al. (2024). BGE-M3. [arXiv:2402.03216](https://arxiv.org/abs/2402.03216) — Self-KD 见本仓库 [BGE-M3 专题](BGE-M3三功能统一详解报告.md) §8  
7. Lee et al. (2024). NV-Embed. [arXiv:2405.17400](https://arxiv.org/abs/2405.17400)  
8. Zhang et al. (2024). NV-Retriever: Hard-negative Mining. [arXiv:2407.15831](https://arxiv.org/abs/2407.15831)  
9. Zhang et al. (2024). Jasper & Stella: Distillation of SOTA Embedding Models. [arXiv:2412.19048](https://arxiv.org/abs/2412.19048)  
10. Kusupati et al. (2022). Matryoshka Representation Learning. *NeurIPS*.  
11. Wang et al. (2023). SimTDE. *SIGIR*.  
12. EMO / Cross-Tokenizer KD (2025). *EMNLP*.  
13. Qwen Team (2025). Qwen3 Embedding Technical Report.  
14. Radford et al. (2021). CLIP. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)  
15. Tschannen et al. (2025). SigLIP 2. [arXiv:2502.14786](https://arxiv.org/abs/2502.14786)  
16. Faysse et al. (2025). ColPali. [arXiv:2407.01449](https://arxiv.org/abs/2407.01449)  

### 本仓库关联文档

| 文档 | 关系 |
|------|------|
| [主文 §11](Embedding调研报告.md) | 蒸馏版图导读 |
| [知识蒸馏总报告 §7](../distillation/知识蒸馏技术深度调研报告.md) | 索引与案例表 |
| [BGE-M3 专题 §8](BGE-M3三功能统一详解报告.md) | Self-KD 公式 |
| [难负例专题](难负例挖掘工业实践.md) | CE→Bi 候选与刷新 |
| [资料清单_论文与博客.md](资料清单_论文与博客.md) | 延伸阅读 |

---

*版本：v1.0 · 2026-07-20 · Embedding 蒸馏专题正文；HTML 由 `convert_embedding_report.py` 生成。*
