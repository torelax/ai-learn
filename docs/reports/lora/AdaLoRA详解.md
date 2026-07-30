# AdaLoRA 详解

> 基于 Zhang et al. *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*（[arXiv:2303.10512](https://arxiv.org/abs/2303.10512)，**ICLR 2023**）。
> 代码：https://github.com/QingruZhang/AdaLoRA
> 本文按论文章节逐节拆解 Figure 1 动机、SVD 参数化、重要性评分、全局 budget 调度与 GLUE/SQuAD/NLG 实验，便于低预算微调选型。

---

## 1. 一句话定位

**AdaLoRA** 解决标准 LoRA **对所有权重矩阵均匀分配 rank $r$** 的问题：用 **SVD 风格参数化** $\Delta = P\Lambda Q$ 把增量拆成 **奇异值三元组（triplet）**，按 **重要性分数** 动态 **剪枝奇异值**（非均匀降 rank），配合 **全局 budget 调度器**，在 **极低可训练参数**（$<\!0.1\%$）下显著优于 LoRA——例如 SQuAD2.0 **F1 +1.2%**。

| 项 | 内容 |
|----|------|
| 核心 | $\Delta = P\Lambda Q$，剪 **$\lambda_i$** 不剪整矩阵 |
| 动机 | Figure 1：**FFN > Attn**；**top layers > bottom** |
| Scheduler | 初 budget 高 → 逐步降至 target |
| 低 budget | 相对 LoRA **优势最大** |
| 宣称 | SQuAD2.0 **$<\!0.1\%$ params，F1 +1.2%** |
| PEFT | `AdaLoraConfig` + `tinit/tfinal/total_step` |

---

## 2. 论文目录与阅读路线

```text
§1  Introduction           均匀 rank 缺陷；Figure 1；贡献摘要
§2  Background             Transformer 记号；LoRA；diff pruning 对照
§3  Method                 SVD 参数化；重要性评分；global budget scheduler
§4  Experiments            DeBERTaV3 GLUE/SQuAD；BART NLG
§5  Conclusion
```

| 若你的目标是… | 优先章节 |
| --- | --- |
| 理解为何要 AdaLoRA | §1 Figure 1 |
| 实现 / 调 schedule | §3 + 本文 §8 |
| 低 budget QA | §4.3 SQuAD2.0 |
| 与 LoRA/DoRA 选型 | 本文 §7 |

---

## 3. §1 Introduction（引言）

### 3.1 动机：全量 FT 不可扩展

PLM（BERT→T5→GPT-3）规模激增；多下游任务若 **每任务一份全量 FT 副本**，内存 **prohibitive**。PEFT 两条主线：

| 主线 | 代表 | 特点 |
| --- | --- | --- |
| **加模块** | Adapter、Prefix、Prompt | 改架构或输入 |
| **重参数化增量** | diff pruning、**LoRA** | 不改架构，$\Delta W$ 结构化 |

LoRA：$\Delta = BA$，$r \ll d$，训练开销可降 **~70%**，性能 **≥ FT**。但 **每个 $\Delta$ 固定同一 rank $r$**，忽略 **矩阵/层间重要性差异**。

### 3.2 Figure 1：不均匀重要性（核心动机）

固定总可训练参数 **0.28M**，在 DeBERTaV3-base + MNLI-m 上：

**Figure 1a — 按矩阵类型**（只 LoRA 某一类权重，每层都加）：

| 只训矩阵 | MNLI-m Acc（约） |
| --- | --- |
| $W_{f1}, W_{f2}$（FFN） | **~89.9**（最高） |
| $W_q, W_v$ | ~89.3–89.4 |
| $W_k$ | ~88.6 |
| $W_o$ | ~88.5 |

**Figure 1b — 按层**（每层所有矩阵都 LoRA，但只训部分层）：

| 层段 | Acc（约） |
| --- | --- |
| 层 10–12（top） | **~88.6–89.0** |
| 层 7–9 | ~88.1 |
| 层 1–3（bottom） | **~77.9** |

**结论**：

1. **FFN > Attention**（同预算下优先 FFN）
2. **Top layers > Bottom layers**
3. 均匀分配 rank → **关键矩阵欠拟合、次要矩阵浪费/过拟合**

### 3.3 AdaLoRA 回答的问题

> How can we allocate the parameter budget adaptively according to importance of modules?

**做法概要**：

- $\Delta = P\Lambda Q$ **模仿 SVD**，按 **三元组 triplet** 剪 **奇异值**（非整矩阵删 rank）
- **重要性度量** + **全局 budget scheduler**（初 budget 略高 → 逐步降至目标）

### 3.4 主要实验宣称

| 任务 | 底座 | 亮点 |
| --- | --- | --- |
| GLUE / SQuAD | DeBERTaV3-base | 低 budget **显著优于 LoRA** |
| SQuAD2.0 | 同上 | **$<\!0.1\%$ 可训练参数，F1 +1.2%** |
| XSum / CNN-DM | BART-large | NLG 一致提升 |

---

## 4. §2 Background（背景）

### 4.1 Transformer 记号

$L$ 层，每层 MHA + FFN：

$$
\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)\, W_o
$$

$$
\mathrm{head}_i = \mathrm{Softmax}\!\left(\frac{X W_{q_i}(X W_{k_i})^{\top}}{\sqrt{d_h}}\right) X W_{v_i}
$$

$$
\mathrm{FFN}(X) = \mathrm{ReLU}(X W_{f1} + b_1)\, W_{f2} + b_2
$$

LoRA 通常作用于 $W_q, W_v$（Hu et al. 2022）；He et al. 2022 扩展至 FFN——AdaLoRA 对 **所有选定权重矩阵** 均可分配不同 **有效 rank**。

### 4.2 标准 LoRA（Eq. 2）

$$
h = W^{(0)} x + \Delta x = W^{(0)} x + B A x
$$

- $W^{(0)}, \Delta \in \mathbb{R}^{d_1 \times d_2}$，$A \in \mathbb{R}^{r \times d_2}$，$B \in \mathbb{R}^{d_1 \times r}$
- $A$ 随机高斯，$B=0$ → $\Delta=0$ 起步
- LoRA 的 **doublet**：$G_i = \{A_{i*},\, B_{*i}\}$（第 $i$ 个 rank 分量）

**局限**：所有 $G_i$ **同等保留**；所有矩阵 **同一 $r$**。

### 4.3 与 diff pruning 对比

| | diff pruning | LoRA | AdaLoRA |
| --- | --- | --- | --- |
| $\Delta$ 形式 | 稀疏全尺寸 | 低秩 $BA$ | 低秩 SVD 型 $P\Lambda Q$ |
| 预算分配 | 逐元素剪 | 固定 $r$ | **逐奇异值剪** |
| 实现 | 非结构化稀疏慢 | 友好 | 友好（避免 exact SVD） |

---

## 5. §3 Method（方法）

AdaLoRA 两组件：**(i) SVD-based adaptation**；(ii) **Importance-aware rank allocation**。

### 5.1 SVD 参数化（Eq. 3–4）

不用 exact SVD，直接参数化：

$$
W = W^{(0)} + \Delta = W^{(0)} + P \Lambda Q
$$

| 符号 | 形状 | 含义 |
| --- | --- | --- |
| $P$ | $\mathbb{R}^{d_1 \times r}$ | 左奇异向量（可学习） |
| $\Lambda$ | $\mathbb{R}^{r \times r}$（对角） | **奇异值** $\lambda_i$ |
| $Q$ | $\mathbb{R}^{r \times d_2}$ | 右奇异向量（可学习） |

**正交正则**（加在 loss 上）：

$$
\mathcal{L}_{\mathrm{ortho}} = \big\|P^{\top}P - I\big\|_F^2 + \big\|Q Q^{\top} - I\big\|_F^2
$$

避免训练中对 $P,Q$ 做 **昂贵 exact SVD**；同时 **只 zero 奇异值、保留奇异向量**，便于 **恢复** 与稳定。

**与 LoRA 关系**：LoRA 的 $BA$ 是 rank-$r$ 因子分解；AdaLoRA 显式分离 **$\lambda_i$** 便于 **按重要性删通道**。

### 5.2 三元组（Triplet）与 LoRA doublet 对照

AdaLoRA 定义 **triplet**：

$$
\mathcal{G}_i = \{\lambda_i,\; P_{*i},\; Q_{i*}\}
$$

- $\lambda_i$：第 $i$ 个奇异值
- $P_{*i}$：$P$ 第 $i$ 列；$Q_{i*}$：$Q$ 第 $i$ 行

剪枝单位是 **整个 triplet**（通过 $\lambda_i \to 0$），而非 LoRA 里固定保留全部 $r$ 个 doublet。

**关键区别**：LoRA 剪 rank 只能 **整矩阵降 $r$**；AdaLoRA 可在 **矩阵 A 保留 8 个 triplet、矩阵 B 保留 3 个**——实现 **非均匀 budget**。

### 5.3 重要性评分（Importance Scoring）

参考 SNIP、movement pruning 等 **敏感度** 思想，对 $\lambda_i$ 定义重要性（论文 Eq. 5–8 系列）。

**直觉流程**：

1. **Sensitivity**：$s_t = |\nabla_{\Lambda} \mathcal{L} \odot \Lambda|$（当前步）
2. **EMA 平滑**：$\hat{s}_t = \beta_1 \hat{s}_{t-1} + (1-\beta_1) s_t$
3. 综合 **奇异值幅度** 与 **敏感度**，得 triplet 分数 $I(\mathcal{G}_i)$
4. **全局排序**：分数低的 triplet → $\lambda_i$ **mask 为 0**

**PEFT RankAllocator 实现** 常用：

$$
I_t = \frac{m_t}{\sqrt{v_t + \epsilon}}
$$

其中 $m_t, v_t$ 为 sensitivity 的 EMA 一阶/二阶矩（类似 Adam 统计）。

**设计要点**：重要性是 **全局跨层、跨矩阵** 排序——budget 从 **不重要 triplet** 流向 **重要 triplet**。

### 5.4 全局 Budget Scheduler

**问题**：若一开始就用目标 rank，重要 triplet 尚未充分学习就被剪。

**策略**：

| 阶段 | Budget |
| --- | --- |
| 初始 | $b_{\mathrm{init}}$ **略高于** 目标 budget |
| 训练推进 | 按 schedule **单调降至** $b_{\mathrm{target}}$ |
| 剪枝触发 | 每 $T$ 步根据 $I(\mathcal{G}_i)$ **全局** 剪掉最低分 triplet |

超参（`AdaLoraConfig`）：

- `init_r` / `target_r`：初始与目标 **平均 rank**
- `tinit`：**warmup**，此期间不剪枝
- `tfinal`：开始 **严格 budget** 的阶段
- `total_step`：总步数（决定 schedule 形状）
- `deltaT`：剪枝间隔步数
- `lora_alpha`：缩放（与 LoRA 类似）

Schedule 使模型先 **过参数探索**，再 **收敛到高效稀疏结构**——**低 budget 设置尤其关键**。

### 5.5 算法总览（伪代码）

```text
初始化 P, Λ, Q（Λ 对角），r = init_r
for step t in 1..T:
    forward + loss + L_ortho
    backward
    更新 P, Λ, Q
    if t > tinit and t mod deltaT == 0:
        计算所有 triplet 的重要性 I(G_i)
        全局排名，mask 最低分奇异值直至 rank budget = schedule(t)
    if t > tfinal:
        budget = target_budget
```

### 5.6 与 LoRA 的关键差异表

| | LoRA | AdaLoRA |
| --- | --- | --- |
| 参数化 | $BA$ | $P \Lambda Q$ |
| Rank | 固定 $r$ | **动态** $r_i$ per matrix |
| 剪枝粒度 | 无 | **奇异值 triplet** |
| 额外 loss | 无 | 正交正则 |
| 超参 | $r, \alpha$ | + $tinit, tfinal, init_r, target_r$ |

---

## 6. §4 Experiments（实验）

### 6.1 设置概览

| 任务类型 | 数据集 | 模型 |
| --- | --- | --- |
| NLU | GLUE | DeBERTaV3-base |
| QA | SQuAD v1.1 / v2.0 | DeBERTaV3-base |
| NLG | XSum、CNN/DailyMail | BART-large |

对比基线：**Full FT、LoRA、HAdapter/PAdapter、BitFit** 等 PEFT；**控制总可训练参数** 公平对比。

### 6.2 GLUE（DeBERTaV3-base）

- AdaLoRA 在 **相同或更低 budget** 下 **稳定高于 LoRA**
- **低 budget 区域** 差距最大——与「均匀 rank 浪费在 bottom 层 / $W_k$」一致
- 分配结果 **可视化**：FFN 与 **高层** triplet 保留更多非零 $\lambda_i$
- 与 Figure 1 先验 **高度一致**，但是 **数据驱动** 而非手工规则

### 6.3 SQuAD v1.1 / v2.0

**论文摘要级宣称**：

> With less than **0.1%** trainable parameters of full fine-tuning, AdaLoRA achieves **1.2% F1 improvement** on SQuAD2.0 compared with state-of-the-art approaches.

**解读**：

- SQuAD2.0 含 **不可答** 子集，更难；AdaLoRA 把 budget 投向 **更关键的 Q/V 与 FFN 子空间**
- 极低 budget 下 LoRA 均匀 rank **欠表达**；AdaLoRA **等效把 rank 从无效矩阵「借」给 FFN**
- v1.1 上亦有提升，但 v2.0 差距更显著（抽取 + 拒答联合需要更强 FFN 表达）

### 6.4 NLG（BART-large，XSum / CNN-DM）

- 生成任务同样受益于 **非均匀 rank**
- ROUGE 系列指标 **一致优于 LoRA**（尤其低 budget）
- 说明方法 **不局限于分类 encoder**；decoder 侧 FFN / cross-attn 亦可自适应分配

### 6.5 消融与分配可视化

| 消融 | 影响 |
| --- | --- |
| **无 scheduler** | 过早剪枝，重要 triplet 未充分学习 |
| **无 ortho loss** | $P,Q$ 退化，rank 解释性下降 |
| **均匀剪枝**（per-matrix 而非 global） | 无法跨矩阵「借」budget |

学到的 **rank 分布热力图**：FFN 块、top 3 层 **颜色最深**——与 Figure 1 手工实验吻合。

---

## 7. §5 Conclusion（结论）

1. **模块重要性不均匀** 是 LoRA 固定 rank 的结构性缺陷
2. **SVD 参数化 + triplet 重要性 + global budget schedule** 实现 **自适应 budget 分配**
3. **避免 exact SVD** 的同时支持 **细粒度剪枝**
4. NLU / QA / NLG **广泛有效**，**低 budget 最强**

---

## 8. 与 LoRA / DoRA 选型

| 场景 | 推荐 |
| --- | --- |
| 默认中 rank（8–64）、工程简单 | **LoRA** |
| **总参数极紧**（0.01%–0.1%） | **AdaLoRA** |
| 要 **逼近 FT 更新几何** | **DoRA** |
| 已知 FFN+top 层最重要，想手工规则 | LoRA 只挂 `up_proj, down_proj` + 后几层（粗近似 AdaLoRA） |
| 长训 + 多矩阵 | AdaLoRA schedule 需 **调 $tinit/tfinal$** |

AdaLoRA 与 DoRA **正交**：前者管 **budget 放哪**，后者管 **怎么更新**；理论上可组合，工程上需自行验证 PEFT 版本支持。

---

## 9. HuggingFace PEFT 用法

```python
from peft import AdaLoraConfig, get_peft_model

config = AdaLoraConfig(
    init_r=12,
    target_r=8,
    lora_alpha=32,
    target_modules=["query_proj", "value_proj", "intermediate.dense", "output.dense"],
    lora_dropout=0.0,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    total_step=3000,
)

model = get_peft_model(model, config)
```

**注意**：

- `target_modules` 名随模型架构（BERT `query` vs LLaMA `q_proj`）
- `total_step` 应与 **实际训练步数** 一致，否则 schedule 错位
- 推理前可 **merge**（与 LoRA 类似），有效 rank 已固化在 $P\Lambda Q$

---

## 10. 公式速查

**SVD 型增量**

$$
W = W^{(0)} + P \Lambda Q, \quad \Lambda = \mathrm{diag}(\lambda_1,\ldots,\lambda_r)
$$

**Triplet**

$$
\mathcal{G}_i = \{\lambda_i,\, P_{*i},\, Q_{i*}\}, \quad \Delta = \sum_i \lambda_i\, P_{*i} Q_{i*}
$$

**正交正则**

$$
\mathcal{L}_{\mathrm{ortho}} = \|P^{\top}P - I\|_F^2 + \|Q Q^{\top} - I\|_F^2
$$

**Forward（同 LoRA 接口）**

$$
h = W^{(0)} x + P \Lambda Q x
$$

---

## 11. 实践清单

| 检查项 | 建议 |
| --- | --- |
| Budget 是否真低 | trainable% $<0.5\%$ 时 AdaLoRA 优势大 |
| Schedule | `tinit` 太短 → 过早剪枝；`tfinal` 太长 → 浪费算力 |
| Target modules | 至少覆盖 **Attn + FFN**，让 allocator 自己选 |
| 对比基线 | 同 **总 trainable params** 比 LoRA，而非同 $r$ |
| SQuAD / 抽取式 QA | 论文 **+1.2 F1** 场景值得优先试 |

---

## 12. 结论（读者 takeaway）

AdaLoRA（ICLR 2023）从 **Figure 1 的 FFN/top-layer 现象** 出发，把 LoRA 的「均匀 rank」升级为 **按 triplet 重要性动态分配 budget**：$\Delta = P\Lambda Q$ mimics SVD，**剪 $\lambda_i$ 不剪整矩阵**，配合 **全局 budget scheduler** 稳定收敛。DeBERTaV3 上 GLUE/SQuAD 与 BART 上 NLG 验证：**低 budget 设定下相对 LoRA 优势最大**；SQuAD2.0 在 **$<\!0.1\%$ 可训练参数** 宣称 **+1.2% F1**。工程上通过 HuggingFace `AdaLoraConfig` 接入，需注意 **schedule 步数与 target_modules** 对齐。

---

## 参考文献

1. Zhang et al. *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*. ICLR 2023. [arXiv:2303.10512](https://arxiv.org/abs/2303.10512)
2. Hu et al. *LoRA*. ICLR 2022.
3. Houlsby et al. *Adapter*. ICML 2019.
4. He et al. *Unified View of Parameter-Efficient Transfer Learning*. ICLR 2022.
5. Guo et al. *Diff Pruning*. NeurIPS 2020.
6. Code: https://github.com/QingruZhang/AdaLoRA
7. HuggingFace PEFT AdaLoRA: https://huggingface.co/docs/peft/conceptual_guides/adalora
