# Conan-embedding（v1）技术详解

> paper: [Conan-embedding: General Text Embedding with More and Better Negative Samples](https://arxiv.org/abs/2408.15710)（arXiv:2408.15710）  
> authors: Shiyu Li（北大 / 腾讯 PCG 实习）, Yang Tang, Shi-Zhe Chen, Xi Chen（腾讯 PCG）  
> HF: [Conan-embedding-v1](https://huggingface.co/)（文中声明已上传）  
> backbone: BERT-large + 线性扩维 1024→1792（约 **326M**）  
> date: 2024-08  
> modality: 文本嵌入（中文主攻）  
> languages: 中文（CMTEB）；预训练含多源中文对  

> 本文把 **两阶段对比课表、动态难负例（DHNM）、Cross-GPU Batch Balance（CBB）、LLM prompt–response 作嵌入数据、CMTEB 冲榜与消融** 写全。  
> v2（从零训 1.4B LLM + soft-mask + CLR）见 [Conan-embedding-v2详解.md](Conan-embedding-v2详解.md)。工业负例位置见 [难负例挖掘工业实践.md](难负例挖掘工业实践.md)。

---

## 一句话定位

**Conan-embedding（v1）** 是腾讯 PCG 的通用文本嵌入：在 BERT-large 上做弱监督→监督两阶段对比学习，核心不是换更大骨干，而是让训练过程始终吃到 **更多、更贴当前权重的难负例**——

1. **Dynamic Hard Negative Mining（DHNM）**：训中周期性判定「负例是否仍难」，变易则换池中下一批；  
2. **Cross-GPU Batch Balance Loss（CBB）**：同 iteration 内跨卡扩负例数，并与 STS 任务联合平衡更新；  
3. 额外发现：**LLM 的 prompt–response** 过滤后可当嵌入预训练对。

| 项 | 内容 |
| --- | --- |
| 参数量 | ≈ **326M**（BERT-large + 投影到 1792-d） |
| 嵌入维 | **1792**；**MRL** 支持 256…1792 |
| 最大长度 | **512** |
| 宣称 | 当时 **CMTEB 中文榜第 1**，Avg **72.62** |

谱系：

```text
E5 / GTE 两阶段对比
  → BGE / gte-Qwen2（更大骨干或 LLM）
  → Conan-v1：BERT-large + DHNM + CBB     ← 本文
  → Conan-v2：从零 1.4B LLM + soft-mask + CLR + 加重 DHNM
```

---

## 问题动机

RAG 把检索质量推到前台；嵌入训练几乎都是对比学习，**负例质量**是上限。既有 hard negative 策略（含 ANCE 异步、NV-Retriever 正例锚定等）多在 **预处理** 挖一次：

- 对固定教师/学生权重，「难」是静态的；  
- 训几轮后同一批负例变「易」，梯度变弱，分数曲线平台震荡。

v1 主张：难负例应 **随学生进化**；同时单卡 batch 装不下足够负例，要用 **跨 GPU 共享 query/正例、分摊负例**。

---

## 训练课表

### 弱监督预训练

多阶段对比（对齐 GTE 系 Li et al. 2023）。收集约 **0.75B** 文本对，规则过滤 + **bge-large-zh-v1.5** 打分，丢弃相似度 **&lt; 0.4**，保留约 **0.4B**。

数据构成（Table 1）：

| 类别 | 格式 | 占比 | 量级 |
|------|------|------|------|
| 社交媒体 | title–content | 39.9% | 341M |
| 新闻 | title–content | 27.3% | 233M |
| Instruction | prompt–response | 11.7% | 100M |
| 知识库 QA | Q–A | 7.7% | 66M |
| 学术 | title–content | 6.0% | 51M |
| 网页 | input–output | 4.6% | 39M |
| Community QA | Q–A | 1.6% | 14M |
| LLM 生成 | Q–A | 1.2% | 10M |

损失：In-batch InfoNCE（无显式 hard）：

$$
\mathcal{L}_{\mathrm{neg}}=-\sum_{i=1}^{N}\log\frac{\exp(\mathrm{sim}(x_i,y_i^+))}{\sum_{j=1}^{M}\exp(\mathrm{sim}(x_i,y_j))}
$$

其中 $y_j$ 为同 batch 其它 passage。优化：AdamW，lr $10^{-5}$，warmup 0.05，wd 0.001，batch 8；**64× Ascend 910B，约 138h**。

**要点**：明确把 **指令微调对（prompt–response）** 与 **LLM 合成 QA** 纳入弱监督——与后续 LLM-DA / QZhou 合成同属「大模型当数据机」。

### 监督微调

任务分两类：

| 任务 | 格式 | 损失 | 规模 |
|------|------|------|------|
| Retrieval | (q, pos, negs) | InfoNCE | 1.8M（+0.5M 生成） |
| STS | (text, pair, score) | **CoSENT** | 1.3M（+0.6M 生成） |

分类数据并入检索：同类当正、异类当负。CoSENT：

$$
\mathcal{L}_{\mathrm{cos}}=\log\!\left(1+\sum_{\mathrm{sim}(i,j)>\mathrm{sim}(k,l)}\exp\frac{\cos(x_k,x_l)-\cos(x_i,x_j)}{\tau}\right)
$$

微调：batch retrieval=4、STS=32；MRL 维 {256,512,768,1024,1280,1536,1792}；**16× Ascend 910B，约 13h**。

---

## 动态难负例挖掘（DHNM）

### 动机

预处理 hard 相对「挖负时的模型」固定；学生更新后，原难负例分数不再下降、开始震荡 → 这批负例已学完。

### 判定与替换（v1 原文）

对每条样本记录难负例相对 query 的 **初始平均分**。**每 100 step** 检查：

若

$$
1.15\cdot S_{\mathrm{cur}} < S_{\mathrm{init}} \quad\wedge\quad |S_{\mathrm{cur}}| < 0.8
$$

则视为 **不再难**，触发新一轮挖掘。

第 $i$ 次替换时，取候选池区间：

$$
\big[(i-1)\cdot n+10,\ i\cdot n+10\big]
$$

其中 $n$ 为每次使用的 hard 个数。文称整次动态挖掘开销约等于 **一步 iteration**。

与标准 hard 对比（Figure 2）：标准法负例分后期平台震荡；DHNM 在判定「变易」后换新难负，负例分可继续被压低。

### 与 v2 / ANCE / NV 对照

| | Conan-v1 DHNM | Conan-v2 DHNM | ANCE | NV-Retriever |
|--|---------------|---------------|------|--------------|
| 何时 | 训中每 **100** step | 逐步 / 更细判定 | 异步全库 ANN | 多训前离线 |
| 判定 | $1.15 S_i < S_0$ 且 $\|S\|<0.8$ | $1.2 S_i < S_0$ 且 $S_i<0.7$ 等 | top ANN 即难 | 正例锚定滤假负 |
| 池 | 预挖候选滑动窗口 | 同族加重 | 全库刷新 | 教师 top-k |

v1 贡献的是 **「训中按难度换池内负例」** 范式；假负过滤仍宜叠加 NV 式正例锚。

---

## Cross-GPU Batch Balance Loss（CBB）

### 动机

多任务若按 iteration **随机抽一种任务**（一会儿 STS、一会儿 Retrieval），单步搜索方向与全局目标不一致，loss 震荡、两任务不同步下降。

### 做法

同一 Forward–Loss–Backward 周期内 **同时** 出现 Retrieval 与 STS：

- **Retrieval**：多卡共享同一批 query/正例，**各卡持有不同难负例**，扩大有效负例数；各卡算 loss 后聚合。  
- **STS**：另卡（或另组）跑更大 batch，吃更多成对比较。  
- 总损失：

$$
\mathcal{L}_{\mathrm{CBB}}
=
-\frac{1}{n}\sum_i\log\frac{\exp(s(x_i,y_i^+)/\tau)}{\exp(s(x_i,y_i^+)/\tau)+\sum_{k=1}^{N}\sum_{j=1}^{n}\exp(s(x_i,y_{k,j}^-)/\tau)}
+\beta\,\mathcal{L}_{\mathrm{cos}}
$$

$N$：共享同一 $(x_i,y_i^+)$ 的 GPU 数；$\beta=0.8$（经验）。

Figure 4：分开训时 retri+STS loss 波动大、不同步；CBB 的 cross loss 平滑下降，终值（约 0.08）远小于两任务分训之和（约 0.38）。

---

## 实验

### CMTEB 主结果（Table 3）

| 模型 | Avg | CLS | Cluster | Rerank | Retri | STS | PairCLS |
|------|-----|-----|---------|--------|-------|-----|---------|
| piccolo-large-zh-v2 | 70.95 | … | | | | | |
| gte-Qwen2-7B-instruct | 72.05 | | | | | | |
| xiaobu-embedding-v2 | 72.43 | | | | | | |
| **Conan-embedding** | **72.62** | 75.03 | 66.33 | 72.76 | 76.67 | 64.18 | 91.66 |

约 **326M** 在中文榜上超过当时更强/更大的若干开源对照（含 7B 级 gte-Qwen2）。

### 消融（Table 4）

| 设定 | Avg |
|------|-----|
| Baseline（仅预训练） | 62.9 |
| Vanilla FT | 68.8 |
| 仅 CBB | 70.4 |
| 仅 DHNM | 71.2 |
| **Conan（DHNM+CBB）** | **72.62** |

Retrieval / Rerank 提升最明显——与「更多更难负例」叙事一致。

---

## 对本仓库的可迁移实践

1. **cloud_emb Stage2**：难负例不要训死一套；按 step 监控 $S_{\mathrm{cur}}/S_{\mathrm{init}}$，变易则换池（对齐 `mine_*` + resume）。  
2. **阈值起点**：v1 用 $1.15$ 与 $0.8$；可按分数尺度扫 $\{1.1,1.15,1.2\}$ × $\{0.7,0.8\}$。  
3. **多卡**：同 query/正例、分卡负例再 all-gather 算 InfoNCE（CBB 检索半边）。  
4. **数据**：指令对 / LLM 合成 QA 可进弱监督，但需分数过滤（文中 &lt;0.4 丢弃思路）。  
5. 上 v2 前：先弄清 v1 的 DHNM+CBB；v2 换的是骨干与 soft-mask/CLR，负例哲学一脉相承。

---

## 局限

- 主评测在 **CMTEB**；英文 MTEB 非本文焦点（v2 才双榜）。  
- 骨干仍是 **BERT-large 512**，长文 / 多语能力不如后续 LLM 嵌入。  
- DHNM 依赖预挖候选池与启发式阈值，未显式做假负例正例锚定。  
- CBB 实现绑定多卡通信与任务配比，复现成本高于「训前挖一次」。  
- 论文较短，部分实现细节（候选池构造教师、精确 $n$）需对照 HF/代码。

---

## 同目录对照

| 文档 | 关系 |
|------|------|
| [Conan-embedding-v2详解.md](Conan-embedding-v2详解.md) | 从零 LLM + soft-mask；DHNM 加重版 |
| [难负例挖掘工业实践.md](难负例挖掘工业实践.md) | Dynamic 替换在工业菜单中的位置 |
| [NV-Retriever详解.md](NV-Retriever详解.md) | 假负过滤；可与 DHNM 叠加 |
| [ANCE详解.md](ANCE详解.md) | 异步全库难负例另一极 |
| [E5详解.md](E5详解.md) / [GTE系列详解.md](GTE系列详解.md) | 两阶段对比公共祖先 |
| [LLM-DA文本行人检索数据增强详解.md](LLM-DA文本行人检索数据增强详解.md) | 另一路「LLM 数据」 |

---

## 参考文献

1. Li et al. (2024). Conan-embedding: General Text Embedding with More and Better Negative Samples. [arXiv:2408.15710](https://arxiv.org/abs/2408.15710)  
2. Li et al. (2025). Conan-Embedding-v2. [arXiv:2509.12892](https://arxiv.org/abs/2509.12892)  
3. Li et al. (2023). Towards General Text Embeddings (GTE). [arXiv:2308.03281](https://arxiv.org/abs/2308.03281)  
4. Moreira et al. (2024). NV-Retriever. [arXiv:2407.15831](https://arxiv.org/abs/2407.15831)  
5. Xiao et al. (2023). C-Pack / BGE.  
6. Kusupati et al. (2022). Matryoshka Representation Learning.  
