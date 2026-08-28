# DoRA 详解

> 基于 Liu et al. *DoRA: Weight-Decomposed Low-Rank Adaptation*（[arXiv:2402.09353](https://arxiv.org/abs/2402.09353)，**ICML 2024 Oral**，NVIDIA）。
> 代码：https://github.com/NVlabs/DoRA
> 本文按论文章节逐节拆解权重分解分析、DoRA 公式与梯度、实验及 QDoRA 扩展，便于对照 LoRA 选型与实现。

---

## 1. 一句话定位

**DoRA（Weight-Decomposed Low-Rank Adaptation）** 在 LoRA 基础上将预训练权重分解为 **幅度（magnitude）** 与 **方向（direction）** 两部分：**方向用 LoRA 更新、幅度单独可训**，使学习模式更接近 **全量微调（FT）** 而非 LoRA 的「幅度–方向同比例缩放」。DoRA **可 merge 回基权重、零推理开销**，在 LLaMA 常识推理、LLaVA 视觉指令微调、VL-BART 图文理解上 **稳定优于 LoRA**。

| 项 | 内容 |
|----|------|
| 分解 | $W = m \cdot V / \|V\|_c$（**列范数**） |
| DoRA | $W' = m \cdot (W_0 + BA) / \|W_0 + BA\|_c$ |
| Pattern | FT **负斜率** $\Delta M$–$\Delta D$；LoRA **正斜率**；DoRA 更接近 FT |
| 额外参数 | 幅度 $m$ 仅 **+0.01%** 量级 |
| 部署 | **mergeable**，推理零增量 |
| 亮点 | LLaMA 常识 **+3.7/+1.0/+2.9/+4.4**；LLaVA **+0.7% vs LoRA** |

---

## 2. 论文目录与阅读路线

```text
§1  Introduction              LoRA–FT 差距；Weight Norm 启发；Figure 1
§2  Related Work              PEFT 三分类；LoRA 变体 landscape
§3  Pattern Analysis          分解 W=mV/||V||c；ΔM、ΔD 度量；Figure 2
§4  Method                    DoRA 公式；梯度 Eq.6–7；detach 降开销 Eq.11
§5  Experiments               LLaMA 常识；VL-BART；LLaVA；DVoRA；SDXL
§6  Broader Impacts           QDoRA（4bit + DoRA）
§7  Conclusion
```

| 若你的目标是… | 优先章节 |
| --- | --- |
| 理解为何 beats LoRA | §3 Pattern Analysis |
| 实现 / merge | §4.2–4.4 + 本文 §10 |
| 选型（rank 低、常识推理） | §5.1 + §5.5 |
| 单卡大模型 | §6 QDoRA |

---

## 3. §1 Introduction（引言）

### 3.1 背景：PEFT 与 LoRA 的容量鸿沟

预训练大模型（LLM / LVLM）全量 FT 效果强，但 **参数量 × 任务数** 成本不可接受。PEFT 仅训少量参数；其中 **LoRA** 因 **不改架构、可 merge** 成为主流。

然而 LoRA 与 FT 之间仍有 **精度差距**，既往工作多归因于 **可训练参数少**，缺少对 **更新模式（learning pattern）** 的细粒度分析。

### 3.2 核心洞察：Weight Normalization 启发的分解

受 **Weight Normalization**（Salimans & Kingma, 2016）启发——通过重参数化改善梯度条件、加速收敛——作者提出：

1. 先将权重分解为 **幅度 + 方向**
2. 对比 **FT、LoRA、DoRA** 在训练中 $\Delta M$（幅度变化）与 $\Delta D$（方向变化）的 **联合分布**
3. 发现 **FT 呈负相关斜率、LoRA 呈正相关**；DoRA 更接近 FT

### 3.3 DoRA 做法（Figure 1）

- 初始化：$m = \|W_0\|_c$，$V = W_0$（**列向量范数**，vector-wise norm across columns）
- **冻结 $V$** 的语义基底；**$m$ 可训**；方向增量 $\Delta V = BA$（LoRA）
- 合并形式：

$$
W' = m \frac{V + \Delta V}{\|V + \Delta V\|_c} = m \frac{W_0 + BA}{\|W_0 + BA\|_c}
$$

### 3.4 贡献摘要

| 贡献 | 内容 |
| --- | --- |
| **DoRA 方法** | 幅度–方向分解 + LoRA 方向更新；**mergeable、无额外延迟** |
| **Pattern Analysis** | 首次系统对比 FT vs LoRA 的幅度/方向更新几何 |
| **广泛验证** | LLaMA 7B/13B/2/3、LLaVA-1.5-7B、VL-BART；常识推理 **+3.7/+1.0/+2.9/+4.4** 等 |

---

## 4. §2 Related Work（相关工作）

### 4.1 PEFT 三分类（与 Houlsby / Hu 谱系对齐）

| 类别 | 代表 | 推理延迟 |
| --- | --- | --- |
| **Adapter-based** | Houlsby 2019 串行；He 2021 并行 | **增加** |
| **Prompt-based** | Prefix Tuning、Prompt Tuning | 序列变长 |
| **Reparametrization** | LoRA、AdaLoRA、VeRA、**DoRA（本文）** | **可 merge，无增量** |

### 4.2 LoRA 变体 landscape

论文点名对比/兼容的 LoRA 系工作：

- **AdaLoRA**（Zhang 2023）：SVD + 动态 rank
- **VeRA**（Kopiczko 2024）：共享随机 $B,A$ + 可训缩放
- **LoRA+**、正交因子化、Hadamard 积等

**DoRA 定位**：仍属第三类；$\Delta V$ 可替换为 VeRA 等变体 → **DVoRA**。

---

## 5. §3 Pattern Analysis of LoRA and FT（模式分析）

### 5.1 LoRA 回顾（Eq. 1）

$$
W' = W_0 + \Delta W = W_0 + BA, \quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d,k)
$$

- $W_0$ **冻结**；$A$ Kaiming 初始化，$B=0$ → 初始 $\Delta W=0$
- 部署前 merge：$W' = W_0 + BA$，**与基座同 shape**

### 5.2 权重分解（Eq. 2）

对 $W \in \mathbb{R}^{d \times k}$（**按列**分解）：

$$
W = m \frac{V}{\|V\|_c}, \quad m \in \mathbb{R}^{1 \times k},\; V \in \mathbb{R}^{d \times k}
$$

- $\|\cdot\|_c$：**列向量范数**（每列一个标量范数）
- $V/\|V\|_c$ 每列为 **单位方向**；$m$ 的标量给出该列 **幅度**

等价写法：$m = \|W\|_c$，$V = W$（分解不引入新信息，只为分析更新几何）。

### 5.3 变化量度量（Eq. 3–4）

对 FT 权重 $W_{\mathrm{FT}}^t$、预训练 $W_0$，在训练步 $t$：

$$
\Delta M_{\mathrm{FT}}^t = \frac{1}{k}\sum_{n=1}^{k} \left| m_{n,\mathrm{FT}}^t - m_{n,0} \right|
$$

$$
\Delta D_{\mathrm{FT}}^t = \frac{1}{k}\sum_{n=1}^{k} \left(1 - \cos\big(V_{n,\mathrm{FT}}^t,\, W_{n,0}\big)\right)
$$

LoRA merge 后的 $W_{\mathrm{LoRA}}$ 同理计算 $\Delta M$、$\Delta D$（对 merge 权重再分解）。

### 5.4 案例：VL-BART 四图文任务

- 仅对 self-attention 的 **Q/V** 做 LoRA（同 Sung et al. 2022）
- 取 **4 个中间 checkpoint + 最终 checkpoint**
- 每层 Q 权重画 $(\Delta D,\, \Delta M)$ 散点 + 回归线

### 5.5 分析结果（Figure 2）

| 方法 | $(\Delta D, \Delta M)$ 回归斜率 | $\mathrm{corr}(\Delta D, \Delta M)$ |
| --- | --- | --- |
| **FT** | **负斜率**（多样、细粒度） | **-0.62** |
| **LoRA** | **正斜率**（幅度方向同增同减） | **+0.83** |
| **DoRA** | **负斜率**（更接近 FT） | **-0.31** |

**解读**：

- LoRA 的 $W'=W_0+BA$ 使 **幅度与方向耦合**：$\Delta W$ 增大时，列范数与列方向往往 **同向变化** → **正斜率**
- FT 常需 **解耦**：「方向小改、幅度大改」或反之——**负斜率**
- 预训练权重已含丰富先验，下游适配未必需要 **大幅同时改幅度和方向**
- DoRA 通过 **显式分解**，让 LoRA **只管方向**，$m$ **专管幅度**，恢复 FT 式灵活性

---

## 6. §4 Method — Weight-Decomposed Low-Rank Adaptation

### 6.1 初始化与训练对象

| 分量 | 初始化 | 是否可训 |
| --- | --- | --- |
| $m$ | $\|W_0\|_c$ | ✅ |
| $V$ | $W_0$ | ❌ 冻结 |
| $\Delta V = BA$ | LoRA 同策略（$B=0$） | ✅ |

训练前 $W' = W_0$（与 LoRA 一致）。

### 6.2 前向公式（Eq. 5）

$$
W' = m \frac{W_0 + BA}{\|W_0 + BA\|_c}
$$

**下划线**为可训练参数：$m$、$B$、$A$。

**与 Weight Norm 的区别**：Weight Norm **从零训** $m,V$，对初始化敏感；DoRA **从 $W_0$ 分解**，无此问题。

**与 LoRA 的区别**：LoRA 直接加性更新；DoRA 在 **归一化方向** 上乘以 **可学习幅度**。

### 6.3 梯度分析（§4.2，Eq. 6–7）

记 $V' = V + \Delta V = W_0 + BA$，损失 $\mathcal{L}$：

$$
\nabla_{V'} \mathcal{L} = \frac{m}{\|V'\|_c}\left(I - \frac{V' V'^{\top}}{\|V'\|_c^2}\right) \nabla_{W'} \mathcal{L}
$$

$$
\nabla_m \mathcal{L} = \nabla_{W'} \mathcal{L} \cdot \frac{V'}{\|V'\|_c}
$$

**要点**：

1. $\nabla_{W'} \mathcal{L}$ 被 $m/\|V'\|_c$ **缩放**，并 **投影** 到与当前权重正交的方向 → 梯度条件更接近各向同性（同 Weight Norm 论证）
2. $V' = V + \Delta V$ ⇒ $\nabla_{V'} \mathcal{L} = \nabla_{\Delta V} \mathcal{L}$，**优化收益传给 LoRA 分支**
3. 由 Eq. 7 可推出：**方向变化小的列，幅度梯度可更大** → 解释 **负斜率** 学习模式

### 6.4 训练开销削减（§4.3，Eq. 11）

严格反传 $\|V'\|_c$ 会 **额外占显存**（梯度图与 LoRA 不同）。

**工程 trick**：将 $\|V' + \Delta V\|_c$ **detach**，前向仍动态归一化，反传时不回传该范数：

$$
\nabla_{V'} \mathcal{L} = \frac{m}{C}\,\nabla_{W'} \mathcal{L}, \quad C = \|V'\|_c \;\text{(constant in autograd)}
$$

消融：**LLaMA-7B、VL-BART 上几乎无精度损失**，显存显著下降。HuggingFace PEFT 默认实现采用此 trick。

### 6.5 Merge 与推理

训练结束后，可将 $W'$ **合并写回** 单个权重矩阵，推理与原始 Linear **完全一致**——这是相对 Adapter 的核心优势。

**额外参数**：幅度向量 $m \in \mathbb{R}^{1 \times k}$ 仅 **+0.01%** 量级（相对 LoRA）；论文称 **DoRA rank 减半仍可超 LoRA**。

---

## 7. §5 Experiments（实验）

### 7.1 Commonsense Reasoning（LLaMA 系列）

**设置**：8 个子任务（BoolQ、PIQA、SIQA 等）；对比 Prefix、Series Adapter、Parallel Adapter、LoRA、**DoRA**；含 ChatGPT zero-shot CoT 基线。

**主要结论**（相对 LoRA 平均提升，论文摘要）：

| 底座 | DoRA 提升（常识推理 avg） |
| --- | --- |
| LLaMA-7B | **+3.7** |
| LLaMA-13B | **+1.0** |
| LLaMA2-7B | **+2.9** |
| LLaMA3-8B | **+4.4** |

**假设验证**：DoRA/FT 微调后权重相对 $W_0$ 的 **幅度/方向偏离更小**，却 **精度更高**——说明强底座只需 **细粒度** 调整，DoRA 的分解更新更高效；LoRA 往往 **偏离更大但效果更差**。

### 7.2 Image/Video-Text Understanding（VL-BART）

多任务框架，设置对齐 Sung et al. 2022 LoRA：

| 域 | DoRA vs LoRA | vs FT |
| --- | --- | --- |
| **Image-text** | **~+0.9%**，接近 FT | 可达 FT 精度 |
| **Video-text** | **~+1.9%** | 明显优于 LoRA |

Pattern Analysis（§3）即在此设定下完成。

### 7.3 Visual Instruction Tuning（LLaVA-1.5-7B）

- 底座：Vicuna-1.5-7B + CLIP ViT-L/336px
- 7 个 VLM benchmark：VQA v2、GQA、VisWiz、SQA、VQA<sup>T</sup>、POPE、MMBench

| 方法 | 相对关系 |
| --- | --- |
| LoRA | 已略超 FT（或 FT 过拟合） |
| **DoRA** | **平均 +0.7% vs LoRA，+1.1% vs FT** |

说明：当 FT 不佳时 DoRA 提升幅度会缩小（设计目标是 **逼近 FT 能力**，而非盲目超越 LoRA）。

### 7.4 与 VeRA 兼容（DVoRA / MT-Bench）

- **DVoRA** = DoRA + VeRA 方向参数化
- LLaMA-7B / LLaMA2-7B instruction tuning：**DVoRA > VeRA**，参数量远小于 LoRA 却 **持平或超过 LoRA 分数**
- **小样本**（1k–10k）：DoRA / DVoRA **全样本规模领先**

### 7.5 Rank 鲁棒性（Figure 5）

$r \in \{4,8,16,32,64\}$ 扫 rank：

| rank | LoRA | DoRA |
| --- | --- | --- |
| 4 | **39.49%** | **61.89%** |
| 8 | 40.74% | **77.96%** |

**极低 rank 下 DoRA 优势最大**——方向专责 LoRA、幅度独立，缓解 rank 不足。

### 7.6 Tuning Granularity（Table 6）

发现：DoRA **不必**像 LoRA 那样 QKV+MLP 全更新：

- **QKV**：幅度 + 方向都更新
- **MLP**：**仅更新幅度** $m$

在 **<50% LoRA 可训练参数** 下，LLaMA-7B **+2.8%**、13B **+0.8%** vs LoRA。

### 7.7 Text-to-Image（SDXL DreamBooth）

DoRA 个性化 **优于** LoRA（训练超参相同）；更能复现训练集细节（如 3D icon 圆角框、Lego logo）。

---

## 8. §6 Broader Impacts — QDoRA

### 8.1 QDoRA = Quantized DoRA

在 **QLoRA 式 4bit NF4 量化底座** 上应用 DoRA：

- 数据：Orca-Math **100k**
- 模型：LLaMA2-7B、LLaMA3-8B
- 指标：exact match

**结果（Figure 6）**：

- QDoRA **显著优于 QLoRA**（+0.19 / +0.23 EM）
- QDoRA **略优于 FT**，且 **显存远低于 FT**
- 结论：**参数效率（QLoRA）+ 细粒度优化（DoRA）** 可兼得

### 8.2 社区意义

降低开源社区 **大模型微调 GPU 门槛**；与 bitsandbytes、PEFT、NeMo 等栈整合（NVIDIA 技术博客宣称将进 Metropolis / NeMo / NIM 等）。

---

## 9. §7 Conclusion（结论）

1. **Pattern Analysis** 揭示 LoRA 与 FT **学习几何根本不同**（$\Delta M$–$\Delta D$ 斜率符号相反）
2. **DoRA** 用 $W' = m \cdot (W_0+BA)/\|W_0+BA\|_c$ **解耦幅度与方向**，更接近 FT，**mergeable**
3. **全面超 LoRA**：LLaMA 常识、LLaVA、VL-BART、SDXL；**兼容 VeRA → DVoRA**
4. 未来：音频等更多模态

---

## 10. 与 LoRA / AdaLoRA 选型对照

| 维度 | LoRA | AdaLoRA | DoRA |
| --- | --- | --- | --- |
| 增量形式 | $W_0 + BA$ | $W_0 + P\Lambda Q$（动态 rank） | $m \cdot (W_0+BA)/\|\cdot\|_c$ |
| 核心优化 | 低秩近似 $\Delta W$ | **预算分配** 到重要矩阵 | **学习模式** 逼近 FT |
| 极低 rank | 易崩 | 依赖 schedule | **相对稳健** |
| PEFT 类 | `LoraConfig` | `AdaLoraConfig` | `DoraConfig` |
| 与 QLoRA | 标准组合 | 可组合 | **QDoRA** |

**实践建议**：

- 默认 LoRA 已够用 → 换 DoRA **成本极低**（merge 路径相同，+0.01% 参数）
- rank ≤ 8 或常识/推理掉点 → **优先试 DoRA**
- 参数预算要 **跨层分配** → AdaLoRA；要 **更新几何** → DoRA；可 **叠加**（不同层面）

---

## 11. 公式速查

**分解**

$$
W = m \frac{V}{\|V\|_c}
$$

**DoRA**

$$
W' = m \frac{W_0 + BA}{\|W_0 + BA\|_c}
$$

**Pattern 度量**

$$
\Delta M = \frac{1}{k}\sum_n |m_n^t - m_n^0|, \quad \Delta D = \frac{1}{k}\sum_n \big(1 - \cos(V_n^t, W_n^0)\big)
$$

**Detach 梯度（实现）**

$$
\nabla_{V'} \mathcal{L} \approx \frac{m}{\|V'\|_c}\,\nabla_{W'} \mathcal{L} \;\;(\|V'\|_c \text{ stop-gradient})
$$

---

## 12. HuggingFace PEFT 用法

```python
from peft import DoraConfig, get_peft_model

config = DoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(base_model, config)
# 训练后
merged = model.merge_and_unload()
```

与 `LoraConfig` 字段 **高度兼容**；较新 PEFT 版本亦可通过 `LoraConfig(..., use_dora=True)` 启用（以官方文档为准）。

---

## 参考文献

1. Liu et al. *DoRA: Weight-Decomposed Low-Rank Adaptation*. ICML 2024 Oral. [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
2. Hu et al. *LoRA*. ICLR 2022.
3. Salimans & Kingma. *Weight Normalization*. NeurIPS 2016.
4. Kopiczko et al. *VeRA*. ICLR 2024.
5. Dettmers et al. *QLoRA*. NeurIPS 2023.
6. NVIDIA Technical Blog: https://developer.nvidia.com/blog/introducing-dora-a-high-performing-alternative-to-lora-for-fine-tuning/
7. Code: https://github.com/NVlabs/DoRA
