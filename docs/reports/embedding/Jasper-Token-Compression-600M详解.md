# Jasper-Token-Compression-600M 技术详解

> 基于技术报告 [arXiv:2511.14405v2](https://arxiv.org/abs/2511.14405)（[HTML](https://arxiv.org/html/2511.14405v2)）与 Hugging Face 模型卡 [infgrad/Jasper-Token-Compression-600M](https://huggingface.co/infgrad/Jasper-Token-Compression-600M)。  
> 训练代码：https://github.com/DunZhang/Jasper-Token-Compression-Training  
> 本文把架构、四阶段训练、全部公式与实验结论写全，便于对照实现与复现。

---

## 1. 一句话定位

**Jasper-Token-Compression-600M** 是 Prior Shape / infgrad 团队在 2025-11 开源的 **中英双语文本 Embedding** 模型：

| 项 | 内容 |
|----|------|
| 规模 | ≈ **600M**（初始化自 **Qwen3-Embedding-0.6B**） |
| 输出维 | **2048**（相对基座 1024 扩维） |
| 教师 | **Qwen3-Embedding-8B** + **QZhou-Embedding（7B）** 双教师融合 |
| 训练主线 | 无监督向量蒸馏 → Token 压缩蒸馏 → 动态压缩 + 相似度结构蒸馏 → 检索对比学习 |
| 核心创新 | Embedding 层后接 **SwiGLU MLP + AdaptiveAvgPool1d** 的 **弹性 Token 压缩**，推理时可调压缩比 |
| 宣称效果 | 英文 MTEB Mean(Task) **74.75**、中文 **73.51**；效率高于普通 0.6B，质量接近 8B 量级教师 |

谱系上延续英文 **Stella / Jasper 蒸馏配方**（[arXiv:2412.19048](https://arxiv.org/abs/2412.19048)），本报告将其扩展到 **中英双语**，并加入 **对比学习** 与 **可调序列压缩**。

---

## 2. 问题背景与设计动机

高分 MTEB Embedding 往往 **参数大、维度高**，部署贵。知识蒸馏（KD）能把大教师能力压到小 student，已在英文 Stella/Jasper 上验证；但 **多语场景下的蒸馏**、以及 **注意力 $O(L^2)$ 带来的长序列推理成本**，仍是缺口。

本文的两条主线：

1. **双教师互补蒸馏**：Qwen3-8B 检索强（报告写 Retrieval 69.44），QZhou STS 强（91.65）；融合后给学生更完整的语义空间。
2. **弹性 Token 压缩**：受 DeepSeek-OCR「上下文光学压缩」启发，在进入 Transformer 注意力前压缩 token 序列长度；训练时随机采样压缩比，推理时可按延迟预算选 $\rho$。

---

## 3. 模型架构

### 3.1 与 Qwen3-Embedding-0.6B 的差异

报告 Figure 1 对比：

| 模块 | Qwen3-Embedding-0.6B | Jasper-Token-Compression-600M |
|------|----------------------|-------------------------------|
| Backbone | Qwen3 0.6B | 同初始化 |
| Token 路径 | `word_emb → Transformer` | `word_emb → Qwen3MLP(SwiGLU) → AdaptiveAvgPool1d → Transformer` |
| Pooling | last-token | **改为 mean pooling** |
| 投影 | 无 / 原生 1024-d | **随机初始化 Linear：1024 → 2048**，再 L2 Norm |
| 推理 | 固定全长 | **可配置** `length_threshold` + `compression_ratio` |

### 3.2 Token 压缩前向（核心）

设 tokenizer 得到 token 序列后，embedding 层输出为

$$
\mathbf{X} \in \mathbb{R}^{L_{\mathrm{in}} \times d},
$$

其中 $L_{\mathrm{in}}$ 为输入 token 数，$d$ 为隐层维度。

1. **特征变换（可训练）**：过一层与 Qwen3 同构的 **Qwen3MLP（SwiGLU）**

$$
\mathbf{H} = \mathrm{Qwen3MLP}(\mathbf{X}) \in \mathbb{R}^{L_{\mathrm{in}} \times d}.
$$

2. **目标长度 $L_{\mathrm{tgt}}$**：由阈值 1 计算；若为 `NULL` 则不压缩，直接 $\mathbf{H}'=\mathbf{H}$。

3. **1D 自适应平均池化（无参）**：

$$
\mathbf{H}' = \mathrm{AdaptiveAvgPool1d}(\mathbf{H};\; L_{\mathrm{tgt}}) \in \mathbb{R}^{L_{\mathrm{tgt}} \times d}.
$$

PyTorch 的 `AdaptiveAvgPool1d` 会按目标长度自动选 kernel/stride，把变长序列压到固定 $L_{\mathrm{tgt}}$。

4. $\mathbf{H}'$ 再进入后续 **注意力 / FFN 堆叠**；最后 **mean pooling + Linear(1024→2048) + L2 Norm** 得到句向量 $\mathbf{E}_s$。

要点：

- **只有 MLP 有参数**；`AdaptiveAvgPool1d` **training-free**。
- 压缩发生在 **注意力之前**，直接砍掉 $O(L^2)$ 中的 $L$，这是延迟下降的主因。
- HF 卡说明：推荐推理压缩比约 **0.3–0.8**；评测默认常用 $L_{\mathrm{th}}=80,\ \rho=0.5$。

### 3.3 目标序列长度（Algorithm 1）

输入：$L_{\mathrm{in}}$，阈值 $L_{\mathrm{th}}$，压缩比 $\rho$。

$$
L_{\mathrm{tgt}} =
\begin{cases}
\texttt{NULL} & \text{if } L_{\mathrm{in}} \le L_{\mathrm{th}} \quad (\text{短文不压}) \\
L_{\mathrm{th}} + (L_{\mathrm{in}} - L_{\mathrm{th}})\times \rho & \text{otherwise}
\end{cases}
$$

伪代码等价：

```text
if real_length <= length_threshold:
    # 不压缩
else:
    target_length = length_threshold + (real_length - length_threshold) * compression_ratio
```

**几何含义**：超过阈值的「超长部分」按比例 $\rho$ 保留；阈值本身始终保留。例如 $L_{\mathrm{th}}=80,\ \rho=0.33,\ L_{\mathrm{in}}=1000$：

$$
L_{\mathrm{tgt}} = 80 + (1000-80)\times 0.33 = 80 + 303.6 \approx 384.
$$

相对原长约 $0.384\times$，而不是简单的 $0.33\times$ 全长——短句保护 + 长句平滑压缩。

---

## 4. 双教师向量构造

两教师维度不同，且能力互补。报告用 **统一到 2048-d 的融合教师向量** $\mathbf{E}_t$。

### 4.1 Qwen3-Embedding-8B 分支（MRL 截断）

Qwen3-8B 原生 **4096-d**，训练用了 **Matryoshka Representation Learning (MRL)**，前缀维度仍有效。取前 **1024** 维：

$$
\mathbf{E}_{\mathrm{qwen}} = \mathrm{Prefix}_{1024}\!\left(\mathbf{E}_{\text{Qwen3-8B}}\right) \in \mathbb{R}^{1024}.
$$

### 4.2 QZhou-Embedding 分支（分段求和）

QZhou 原生 **3584-d**，**不支持 MRL**。取前 **3072** 维，切成 **3 段连续 1024-d**，再 **逐元素相加**：

$$
\mathbf{v}_1,\mathbf{v}_2,\mathbf{v}_3 \in \mathbb{R}^{1024},\quad
\mathbf{E}_{\mathrm{QZhou}}^{[1:3072]} = [\mathbf{v}_1;\mathbf{v}_2;\mathbf{v}_3],
$$

$$
\mathbf{E}_{\mathrm{qzhou}} = \mathbf{v}_1 + \mathbf{v}_2 + \mathbf{v}_3 \in \mathbb{R}^{1024}.
$$

（作者也试过 Qwen3-8B + Qwen3-4B 组合，提升有限，故最终选 QZhou。）

### 4.3 融合与归一化

$$
\mathbf{E}_t
=
\mathrm{Norm}\!\Big(
  \mathrm{Norm}(\mathbf{E}_{\mathrm{qwen}})
  \;\big\|\;
  \mathrm{Norm}(\mathbf{E}_{\mathrm{qzhou}})
\Big)
\in \mathbb{R}^{2048},
$$

其中 $\Vert$ 表示特征维拼接，$\mathrm{Norm}$ 为 **L2 归一化**。

学生侧：mean-pool 得 1024-d，经 Linear 映射到 2048-d 后再 L2 Norm，记为 $\mathbf{E}_s$。

---

## 5. 四阶段训练（完整公式）

整体流水线：

```text
Stage 1  标准 KD（无压缩）
   ↓
Stage 2  固定压缩比 KD（引入压缩模块）
   ↓
Stage 3  动态压缩比 + Cosine + Similarity MSE
   ↓
Stage 4  InfoNCE + Soft KL + Cosine（检索增强）
```

---

### 5.1 Stage 1：标准知识蒸馏

**目标**：让 student 对齐融合教师 $\mathbf{E}_t$ 的绝对方向。

#### 损失（式 1）——Cosine Loss

$$
\mathcal{L}_{\mathrm{cosine}} = 1 - \mathbf{E}_s \cdot \mathbf{E}_t.
$$

因两者均已 L2 Norm，点积即余弦相似度，故 $\mathcal{L}_{\mathrm{cosine}} \in [0,2]$，最小化即拉近角度。

训练时实际优化：

$$
\mathcal{L}_{\mathrm{s1}} = 10 \cdot \mathcal{L}_{\mathrm{cosine}}.
$$

#### Training Setup（Stage 1）

| 项 | 设置 |
|----|------|
| 数据 | **1200 万** 中英无监督段落，中英约 **1:1** |
| 最大长度 | **1030** tokens |
| 优化器 | Adam，lr $1\times 10^{-4}$，warmup ratio **0.005**，cosine schedule |
| Epoch | **2** |
| 硬件 | 4× RTX 4090；per-GPU batch **4**，grad accum **16** → global batch **256** |
| 其它 | FlashAttention-2 |

---

### 5.2 Stage 2：带 Token 压缩的蒸馏

在 Stage 1 权重上 **插入压缩模块**，继续用同一数据与 $\mathcal{L}_{\mathrm{cosine}}$（同样 ×10 量级设定延续 Stage 1 配方）。

- **全模型参数可训**；压缩模块里 **仅 Qwen3MLP 可训**，`AdaptiveAvgPool1d` 无参。
- 固定：$L_{\mathrm{th}}=80,\ \rho=0.33$。
- Adam lr $7\times 10^{-5}$，仍 2 epoch；其余与 Stage 1 同。

作用：先在 **单一压缩强度** 下让 backbone + MLP 学会「压缩后仍能对齐教师」。

---

### 5.3 Stage 3：动态压缩 + 相似度结构蒸馏

#### 5.3.1 压缩比采样（Algorithm 2）

每 batch 采一个 $\rho$：

$$
r \sim \mathrm{Uniform}(0,1),
$$

$$
\rho =
\begin{cases}
\mathrm{Uniform}(0.1,\ 0.33) & r < 0.1 \\
0.33333 & 0.1 \le r < 0.5 \\
\mathrm{Uniform}(0.33,\ 0.66) & 0.5 \le r < 0.8 \\
\mathrm{Uniform}(0.66,\ 1.0) & r \ge 0.8
\end{cases}
$$

概率解读：

| 区间 | 概率 | 含义 |
|------|------|------|
| 固定 $\rho\approx 0.33$ | **40%** | 与 Stage 2 对齐，稳定主工况 |
| 其它区间合计 | **60%** | 探索更强/更弱压缩，换推理弹性 |

#### 5.3.2 Similarity Loss（式 2）

batch 内学生 / 教师 embedding 矩阵 $\mathbf{BE}_s,\mathbf{BE}_t \in \mathbb{R}^{B\times 2048}$（$B$ 为 batch size）。构造 **成对相似度矩阵**（Gram / 余弦矩阵，因向量已归一化）：

$$
\mathbf{S}_s = \mathbf{BE}_s\,\mathbf{BE}_s^{\mathsf{T}},\qquad
\mathbf{S}_t = \mathbf{BE}_t\,\mathbf{BE}_t^{\mathsf{T}} \in \mathbb{R}^{B\times B}.
$$

$$
\mathcal{L}_{\mathrm{similarity}}
=
\mathrm{MSE}\!\left(\mathbf{S}_s,\ \mathbf{S}_t\right)
=
\frac{1}{B^2}\sum_{i,j}\left((\mathbf{S}_s)_{ij}-(\mathbf{S}_t)_{ij}\right)^2.
$$

**直觉**：不只对齐单点 $\mathbf{E}_s\approx\mathbf{E}_t$，还对齐 batch 内 **相对相似度结构**（谁和谁更像），对检索/聚类更关键。

#### 5.3.3 Stage 3 总损失（式 3）

$$
\mathcal{L}_{\mathrm{s3}}
=
10\cdot\mathcal{L}_{\mathrm{cosine}}
+
100\cdot\mathcal{L}_{\mathrm{similarity}}.
$$

权重上 **结构项远大于绝对余弦项**（100 vs 10）。

#### Training Setup（Stage 3）

| 项 | 设置 |
|----|------|
| 步数 | **800** steps（非完整 epoch） |
| lr | $7\times 10^{-5}$ |
| $L_{\mathrm{th}}$ | 80；$\rho$ 动态采样 |
| grad accum | **32** → global batch **512** |
| 其余 | 同 Stage 1 数据与硬件设定 |

---

### 5.4 Stage 4：检索向对比学习

蒸馏三阶段后，多数 MTEB 任务已接近教师，但 **非对称 Retrieval** 仍落后（约 65.5 vs 教师 69.4）。故引入 **InfoNCE + Soft KL**。

数据：采用 **QZhou Embedding 同款检索训练数据**。

#### 5.4.1 Contrastive Loss / InfoNCE（式 4–5）

batch 有 $N$ 个训练实例。对第 $i$ 个 query $q_i$，正文档 $d_i^+$，$K$ 个 hard negative $d_{i,k}^-$，以及 batch 内其它文档作 easy negative。

相似度 $s(\cdot,\cdot)$ 为 **余弦相似度**，温度 $\tau$：

$$
\mathcal{L}_{\mathrm{cl}}
=
-\frac{1}{N}\sum_{i=1}^{N}
\log
\frac{\exp\!\big(s(q_i,d_i^{+})/\tau\big)}{Z_i},
$$

$$
\begin{aligned}
Z_i
&=
\exp\!\big(s(q_i,d_i^{+})/\tau\big)
\\
&\quad+
\sum_{k=1}^{K}\exp\!\big(s(q_i,d_{i,k}^{-})/\tau\big)
\\
&\quad+
\sum_{j\neq i}\exp\!\big(s(q_i,d_j)/\tau\big).
\end{aligned}
$$

第三项里 $d_j$ 覆盖 **其它 query 的正例及对应 hard negatives**，报告写明共产生 $(N-1)(1+K)$ 个 easy negatives。

#### 5.4.2 Soft Distillation Loss（式 6）

令学生 / 教师在同一组候选上的相似度分数向量为

$$
\mathbf{S}^{(s)},\ \mathbf{S}^{(t)} \in \mathbb{R}^{N(1+K)},
$$

（HF 卡表述：query 对 **正例 + hard negatives + in-batch docs** 的打分向量。）

温度 $\alpha$，$\mathrm{sm}$ 为 softmax，$\mathrm{D_{KL}}$ 为 KL 散度：

$$
\mathcal{L}_{\mathrm{soft}}
=
\mathrm{D_{KL}}\!\Big(
  \mathrm{sm}\!\big(\mathbf{S}^{(s)}/\alpha\big)
  \;\Big\|\;
  \mathrm{sm}\!\big(\mathbf{S}^{(t)}/\alpha\big)
\Big).
$$

**直觉**：不只学「谁是正例」（hard InfoNCE），还蒸馏教师的 **整张分数分布**（soft ranking）。

#### 5.4.3 Stage 4 总损失（式 7）

再加余弦正则，对齐绝对向量：

$$
\mathcal{L}_{\mathrm{s4}}
=
\mathcal{L}_{\mathrm{cl}}
+
16\cdot\mathcal{L}_{\mathrm{soft}}
+
10\cdot\mathcal{L}_{\mathrm{cosine}}.
$$

#### Training Setup（Stage 4）

| 项 | 设置 |
|----|------|
| 步数 | **5000** |
| lr | $2\times 10^{-5}$（Adam） |
| $N,\ K$ | **16, 3** |
| $\tau,\ \alpha$ | **0.3, 0.1** |
| 压缩 | $L_{\mathrm{th}}=80$ + **动态 $\rho$** |
| 并行 | 4 GPU，grad accum **1**，启用 gradient checkpointing |

---

## 6. 公式速查表

| 编号 | 名称 | 公式 | 主要出现阶段 |
|------|------|------|--------------|
| (1) | Cosine | $\mathcal{L}_{\mathrm{cosine}}=1-\mathbf{E}_s\cdot\mathbf{E}_t$ | 1–4 |
| (2) | Similarity MSE | $\mathcal{L}_{\mathrm{similarity}}=\mathrm{MSE}(\mathbf{BE}_s\mathbf{BE}_s^{\mathsf{T}},\mathbf{BE}_t\mathbf{BE}_t^{\mathsf{T}})$ | 3 |
| (3) | Stage3 总损 | $\mathcal{L}_{\mathrm{s3}}=10\mathcal{L}_{\mathrm{cosine}}+100\mathcal{L}_{\mathrm{similarity}}$ | 3 |
| (4)(5) | InfoNCE | 见 §5.4.1 | 4 |
| (6) | Soft KL | $\mathcal{L}_{\mathrm{soft}}=\mathrm{D_{KL}}(\mathrm{sm}(\mathbf{S}^{(s)}/\alpha)\|\mathrm{sm}(\mathbf{S}^{(t)}/\alpha))$ | 4 |
| (7) | Stage4 总损 | $\mathcal{L}_{\mathrm{s4}}=\mathcal{L}_{\mathrm{cl}}+16\mathcal{L}_{\mathrm{soft}}+10\mathcal{L}_{\mathrm{cosine}}$ | 4 |
| Alg.1 | 目标长度 | $L_{\mathrm{tgt}}=L_{\mathrm{th}}+(L_{\mathrm{in}}-L_{\mathrm{th}})\rho$（短于阈值则不压） | 2–4 / 推理 |
| Alg.2 | $\rho$ 采样 | 40% 钉在 0.33，60% 探索其它区间 | 3–4 |

教师构造：

$$
\mathbf{E}_t=\mathrm{Norm}\big(\mathrm{Norm}(\mathbf{E}_{\mathrm{qwen}})\ \|\ \mathrm{Norm}(\mathbf{E}_{\mathrm{qzhou}})\big).
$$

---

## 7. 实验结果

评测默认：**$L_{\mathrm{th}}=80,\ \rho=0.5$**。

### 7.1 英文 MTEB（Table 1 摘要）

| 模型 | Size | Dim | Mean(Task) | Mean(TaskType) |
|------|------|-----|------------|----------------|
| Qwen3-Embedding-0.6B（学生初始化） | 595M | 1024 | 70.70 | 64.88 |
| **Jasper-Token-Compression-600M** | 600M | 2048 | **74.75** | **68.46** |
| Qwen3-Embedding-4B | 4B | 2560 | 74.60 | 68.10 |
| QZhou-Embedding（教师） | 7B | 3584 | 75.97 | 69.52 |
| Qwen3-Embedding-8B（教师） | 8B | 4096 | 75.22 | 68.71 |

相对基座：Mean(Task) **+4.05**，Mean(TaskType) **+3.58**；整体接近 4B–8B 开源模型，与 Seed1.5-Embedding（74.76）同档。

### 7.2 中文 MTEB（Table 2 摘要）

| 模型 | Mean(Task) | Mean(TaskType) |
|------|------------|----------------|
| Qwen3-Embedding-0.6B | 66.33 | 67.45 |
| **Jasper-TC-600M** | **73.51** | **75.00** |
| Qwen3-Embedding-8B | 73.84 | 75.00 |
| QZhou-Embedding | 76.99 | 78.58 |

相对基座：**+7.18 / +7.55**；中文提升幅度大于英文，说明双语蒸馏有效。仍低于部分更大/专用中文模型（如 Youtu、QZhou-Zh）。

### 7.3 消融：对比学习（Table 3，英文 Task Type）

| Task Type | Stage 3 | Stage 4 | Qwen3-8B |
|-----------|---------|---------|----------|
| Classification | 90.49 | 90.35 | 90.43 |
| Clustering | 59.71 | 59.44 | 58.57 |
| PairClassification | 90.08 | 90.15 | 87.52 |
| Reranking | 50.84 | 50.60 | 51.56 |
| **Retrieval** | **65.53** | **66.19** | **69.44** |
| STS | 88.73 | 88.79 | 88.58 |
| Summarization | 33.28 | 33.66 | 34.83 |
| Mean(Task) | 74.65 | 74.75 | 75.22 |

结论：

- Stage 3 后 **分类/聚类/STS 已齐教师**；
- Stage 4 把 Retrieval **65.53 → 66.19（+0.66）**，有用但 **仍差教师约 3.25**；
- 其它任务几乎不动——符合「检索专项微调」预期。

### 7.4 消融：压缩比 vs 质量 / 延迟（Table 4）

在 1600 条定长文本、batch=32 上测 **单样本平均编码 ms**：

| 设置 | Mean(Task) | L=128 | 256 | 512 | 1024 | 2048 |
|------|------------|-------|-----|-----|------|------|
| Qwen3-Emb-0.6B（无压缩） | 70.70 | 3.22 | 6.47 | 12.20 | 24.24 | 49.99 |
| Jasper $\rho=0.50$ | **74.75** | 2.62 | 3.96 | 7.39 | 13.11 | 25.07 |
| Jasper $\rho=0.33$ | 74.70 | 2.52 | 3.52 | 5.41 | 9.38 | 17.52 |
| Jasper $\rho=0.20$ | 74.58 | 2.38 | 2.91 | 4.00 | 6.56 | 11.48 |
| Jasper $\rho=0.10$ | 74.21 | 2.09 | 2.56 | 3.18 | 4.48 | 6.95 |

解读：

- $\rho: 0.5\to 0.1$，质量仅 **-0.54** Mean(Task)，但 **2048 长输入延迟约 25→7 ms（约 3.6×）**；
- 相对同规模无压缩基座，**又快又强**（质量高、长序列更明显）；
- 动态压缩训练使 **强压缩工况仍可用**，这是相对「静态裁剪/固定 pooling」的关键卖点。

---

## 8. 局限性（作者自陈 + 工程解读）

1. **检索天花板**：Stage 4 后仍 **66.19 vs 教师 69.44**；纯蒸馏 + 有限对比学习，难完全复现大模型非对称检索。
2. **压缩器过简**：MLP + 无参 AvgPool，**不可内容自适应**（不按 token 重要性选留）；作者期望未来可训、随长度/batch 自适应的压缩。
3. **长度外推**：蒸馏最大 **1030** tokens；更长文本可能掉点。Table 4 虽测到 2048 长度延迟，但 **质量评测主设定仍是压缩后的有效上下文**，长文语义保持需另测。
4. **输出维 2048**：相对 0.6B 常见 1024，**索引存储约翻倍**；若要 Matryoshka 截断，报告未声称学生侧原生 MRL（教师侧用了 Qwen MRL 截断）。

---

## 9. 实践要点（落地用法）

### 9.1 推理（Sentence-Transformers）

模型卡示例要点：

- `trust_remote_code=True`，`padding_side="left"`；
- query 用 `prompt_name="query"`（prompt 策略对齐 QZhou 评测）；
- `normalize_embeddings=True`；
- **`compression_ratio`** 越小越快、质量略降；建议 **0.3–0.8**。

### 9.2 对自研 ≤0.6B Embedding 的可迁移清单

若你方路线是「大教师蒸馏小部署模型」（参见同目录《0.6B 行动路线》《Embedding蒸馏技术详解》），可直接借鉴：

| 可抄 | 注意 |
|------|------|
| **双教师互补 + 维对齐拼接** | 需先验证教师相关/互补；乱拼可能负迁移 |
| **Cosine 绝对对齐 + batch 相似度 MSE** | Stage3 的结构项对保持教师几何很关键 |
| **检索阶段 InfoNCE + Soft KL** | Soft KL 把教师分数分布当 soft label，比单 hard CE 稳 |
| **注意力前压缩 + 动态 $\rho$ 训练** | 部署侧用「质量–延迟旋钮」；阈值保护短 query |
| mean-pool + 扩维投影 | 维数↑会抬存储；是否值得看召回收益 |

不太建议原样照搬的部分：

- 1200 万双语段落 + 双 7B/8B 教师成本高；
- 检索仍可能卡在「像教师但不如教师」——难负例挖掘、领域对比数据往往比再堆一轮通用 KD 更有效。

---

## 10. 与系列工作关系

| 工作 | 关系 |
|------|------|
| Stella / Jasper 蒸馏（2412.19048） | 英文 SOTA 蒸馏配方前身 |
| Qwen3-Embedding | 学生初始化 + 一教师 |
| QZhou-Embedding | 另一教师 + Stage4 数据/评测 prompt 对齐 |
| DeepSeek-OCR | Token/上下文压缩思想来源（跨模态启发） |
| OpenReview「Jasper-Flash / ETC+CAPD」 | 同思路的会议匿名稿叙事（Elastic Token Compression + Progressive Distillation），与本技术报告同一模型线 |

---

## 11. 结论

Jasper-Token-Compression-600M 把三条线拧在一起：

1. **多教师向量蒸馏**（余弦 + batch 相似度结构）；
2. **注意力前弹性 Token 压缩**（短文保护 + 可调 $\rho$ + 训练期随机工况）；
3. **检索向 InfoNCE + Soft KL** 补短板。

在 **≈0.6B / 2048-d** 约束下，中英 MTEB 逼近数倍大的教师，并用压缩把长序列编码延迟压到显著低于同规模稠密基座。公式上最值得记住的是：

$$
\underbrace{\mathcal{L}_{\mathrm{cosine}}}_{\text{绝对对齐}}
\;+\;
\underbrace{\mathcal{L}_{\mathrm{similarity}}}_{\text{相对结构}}
\;+\;
\underbrace{\mathcal{L}_{\mathrm{cl}}+\mathcal{L}_{\mathrm{soft}}}_{\text{检索排序}}
\;+\;
\underbrace{L_{\mathrm{tgt}}(L_{\mathrm{th}},\rho)}_{\text{推理可调长度}}.
$$

其主要未闭合问题仍是：**非对称检索与教师的差距**，以及 **更可学习的压缩器**——这也是自研蒸馏小模型时最该额外投入的方向。

---

## 参考文献

1. Zhang et al. *Jasper-Token-Compression-600M Technical Report*. arXiv:2511.14405v2, 2025.  
2. Zhang et al. *Jasper and Stella: distillation of SOTA embedding models*. arXiv:2412.19048, 2024.  
3. HF: https://huggingface.co/infgrad/Jasper-Token-Compression-600M  
4. Training code: https://github.com/DunZhang/Jasper-Token-Compression-Training  
