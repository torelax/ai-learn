# LoRA+ 技术详解

> 基于论文 [LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/abs/2402.12354)（Hayou et al., ICML 2024）。
> 本文把 **A/B 异学习率（ηB ≫ ηA）、最优比 λ=ηB/ηA≈16、无限宽特征学习理论、GLUE 与 LLaMA 实验增益** 写全。

---

## 1. 一句话定位

**LoRA+** 不改变 LoRA 的参数量与结构，仅修正一个被忽视的训练细节：**低秩矩阵 $A$ 与 $B$ 应使用不同学习率**。理论（无限宽极限）表明 $B$ 负责 **特征学习**、$A$ 负责 **缩放**，二者最优步长相差 $\Theta(\eta)$ 量级；实践上令 **$\eta_B \approx 16 \,\eta_A$**（$\lambda \approx 16$）即可在 **相同 step 数、零额外参数** 下，相对标准 LoRA 取得 **显著 GLUE /  commonsense / 生成** 提升，峰值增益可达 **+20% 相对 loss 改善**（论文 Figure 1 量级，依任务而异）。

| 项 | 内容 |
| --- | --- |
| 发表 | arXiv:2402.12354；ICML 2024 |
| 改动 | **仅优化器**：$\eta_A \neq \eta_B$；结构同 LoRA |
| 推荐比 | **λ = ηB / ηA ≈ 16**（宽网络下较稳） |
| 理论 | 无限宽下 $A$ 近似常数、$B$ 做特征学习；Finetuning 与 LoRA 谱差异 |
| 工程成本 | **零** — 任何 LoRA 实现加两组 param group 即可 |

---

## 2. 问题背景：标准 LoRA 训练「半错」

### 2.1 LoRA 回顾

对线性层：

$$
W = W_0 + \alpha_r B A, \quad A \in \mathbb{R}^{r \times d_\text{in}},\; B \in \mathbb{R}^{d_\text{out} \times r}
$$

默认初始化：$A$ 随机高斯，$B=0$，故初始 $\Delta W=0$。

### 2.2 常见实践中的隐含假设

HuggingFace / PEFT、Axolotl、ms-swift 等默认 **$A,B$ 共用同一 learning rate**（如 1e-4），且 weight decay 策略相同。Hayou 等指出：这与 LoRA 在梯度流中的 **角色不对称** 相悖 — **$B$ 的更新幅度应系统性地大于 $A$**。

### 2.3 现象：同参不同效

论文 Figure 1（RoBERTa-base GLUE）显示：在固定总 budget 下，**仅调 λ** 即可让 LoRA 逼近甚至超过「更大 rank 的标准 LoRA」，而 **不增加任何可训练参数**。这说明许多 LoRA 实验的 suboptimal 来自 **优化**，而非 rank 不够。

---

## 3. 理论：无限宽下的特征学习

### 3.1 设定

考虑一层线性网络微调，LoRA 插入预训练权重 $W_0$。取 **无限宽极限**（NTK / μP 相关技术路线）：隐藏维 $d \to \infty$，适当缩放初始化，分析梯度下降动力学。

### 3.2 核心命题（直觉版）

| 矩阵 | 渐近角色 | 最优 LR 量级 |
| --- | --- | --- |
| **$A$** | 输入侧 **固定随机投影 + 小幅缩放**；训练中接近 **慢变量** | $\eta_A = \Theta(1)$（相对单位） |
| **$B$** | 输出侧 **特征学习**；承载任务信号的主更新 | $\eta_B = \Theta(\eta)$，$\eta$ 为基学习率 |

因而 **$\eta_B / \eta_A = \Theta(\eta)$** 或等价地随宽度/初始化标度 **差一阶**。在常用 LLM 设定（$\eta \sim 10^{-4}$，宽 $d \sim 10^3$）下，离散仿真与实验拟合给出 **$\lambda \approx 16$** 为鲁棒甜点。

### 3.3 LoRA vs Full Finetuning 的谱

论文分析预训练权重 $W_0$ 的奇异值谱与 LoRA 增量 $\Delta W = BA$ 的对齐：

- **Full FT** 倾向修改 $W_0$ 的 **顶部奇异方向**（与任务最相关）；
- **标准 LoRA**（同 LR）对 **顶部与尾部** 方向更新不足或失衡，**欠拟合特征方向**；
- **LoRA+** 通过放大 $B$ 的步长，使 $\Delta W$ 的奇异值分布 **更接近 Full FT**。

这解释了「同 rank 为何 LoRA+ 更接近全参」——不是 rank 变了，是 **有效更新谱** 变了。

### 3.4 与 μP / lr scaling 的关系

LoRA+ 的 λ 与 **宽度、初始化尺度** 相关；论文在 RoBERTa（$d=768$）与 LLaMA-7B（$d=4096$）上均验证 **λ=16** 附近宽平台，说明该比值在实用 LLM 尺度上 **无需逐模型精细搜索** 即可起步。

---

## 4. 方法形式化

### 4.1 参数更新

$$
\theta_A \leftarrow \theta_A - \eta_A \nabla_{A} \mathcal{L}, \quad
\theta_B \leftarrow \theta_B - \eta_B \nabla_{B} \mathcal{L}
$$

$$
\eta_B = \lambda \, \eta_A, \quad \lambda > 1
$$

**其余与 LoRA 完全相同**：冻结 $W_0$、rank $r$、缩放 $\alpha$、target modules、merge 逻辑等。

### 4.2 实现：Param Groups

PyTorch 示例：

```python
lora_a_params = []
lora_b_params = []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if "lora_A" in name or name.endswith(".A"):
        lora_a_params.append(p)
    elif "lora_B" in name or name.endswith(".B"):
        lora_b_params.append(p)

optimizer = AdamW([
    {"params": lora_a_params, "lr": eta_a, "weight_decay": wd_a},
    {"params": lora_b_params, "lr": eta_b, "weight_decay": wd_b},
    # 其余 head / bias 等
])
```

**ms-swift / PEFT**：部分版本通过 `lorap_lr_ratio` 或等价 flag 暴露 λ；若无，自定义 callback 或 fork optimizer 即可。

### 4.3 Weight Decay

论文主实验 **对 $A,B$ 使用相同 weight decay**（或仅 decay $A$）。因 $B$ LR 更大，有效正则化已不对称；实践中 **不必** 对 $B$ 额外加大 decay，除非过拟合明显。

---

## 5. 超参数：λ 与 ηA 的选择

### 5.1 λ 扫描（RoBERTa-base GLUE）

| λ = ηB/ηA | 相对表现 |
| --- | --- |
| 1（标准 LoRA） | 基线 |
| 4 | 明显提升 |
| **16** | **最佳平台中心** |
| 32 | 略降或持平（任务依赖） |
| 64+ | 常不稳定 |

**推荐**：默认 **λ=16**；算力允许时对 **{4, 16, 32}** 做小网格。

### 5.2 ηA 与全局 LR

LoRA+ **不推翻** 原有 LR 搜索，而是在给定 $\eta_A$ 下设 $\eta_B=\lambda\eta_A$。

| 场景 | ηA 起点 |
| --- | --- |
| RoBERTa / DeBERTa GLUE | 1e-4 ~ 3e-4（与 LoRA 论文一致） |
| LLaMA-7B SFT | 1e-4 ~ 2e-4 |
| 7B + QLoRA | 2e-4 ~ 3e-4（与 QLoRA 原文接近） |

若使用 **lr scheduler**（cosine），$A,B$ 两组 **同步缩放** 即可。

### 5.3 Rank 与 λ 的交互

- **低 rank（r=4,8）**：LoRA+ 增益 **更大** — 标准 LoRA 更欠拟合，优化修正价值高；
- **高 rank（r=64+）**：增益收窄但仍常为 **正**；
- LoRA+ **不能替代** rank；rank 与 λ 正交调参。

---

## 6. 实验：GLUE（Encoder）

### 6.1 设置

- 模型：RoBERTa-base、RoBERTa-large
- LoRA：$r \in \{4,8,16\}$，注入 Q/V 或全部 linear
- 对比：**LoRA（λ=1）** vs **LoRA+（λ=16）**
- 其余：batch、epoch、seed 对齐

### 6.2 结果摘要

- LoRA+ 在 **8/9 GLUE 任务** 上优于标准 LoRA（论文 Table 2 量级）。
- 平均 GLUE 提升 **~1–2 绝对分**（base）/ 更大（large），相对 **训练 loss 最高 ~20% 相对下降**。
- **相同 epoch 早停** 下，LoRA+ 收敛 **更快** — 同等 wall-clock 更有优势。

### 6.3 与 Adapter / Full FT

LoRA+ 仍 **不增加参数**；在 RoBERTa-large 上可 **缩小与 Full FT 的差距** 而不解冻 backbone。Adapter 参数量通常大于 LoRA；LoRA+ 是 **零成本** 改 optimizer。

---

## 7. 实验：LLaMA（Decoder）

### 7.1 Commonsense Reasoning

8 数据集（BoolQ、PIQA、SIQA 等），LLaMA-7B/13B，LoRA $r=8$，仅训 adapter：

| 设置 | 平均 acc |
| --- | --- |
| LoRA λ=1 | 基线 |
| **LoRA+ λ=16** | **+1~3 pt**（依模型规模） |

### 7.2 指令微调 / 生成

在 Alpaca 类 SFT 与 E2E 生成上，LoRA+ 的 **验证 loss** 一致低于标准 LoRA；生成指标（BLEU 等）同步改善。

### 7.3 与 QLoRA 叠加

QLoRA 仅量化 $W_0$；LoRA+ 作用于 **LoRA 分支 optimizer**。二者 **完全兼容** — 推荐 QLoRA 默认配方上 **直接设 λ=16**。

---

## 8. 机制深入：梯度视角

### 8.1 链式法则下的不对称

前向（忽略 $\alpha/r$）：

$$
y = W_0 x + B (A x)
$$

对损失 $\mathcal{L}$：

$$
\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial y} (A x)^\top, \quad
\frac{\partial \mathcal{L}}{\partial A} = B^\top \frac{\partial \mathcal{L}}{\partial y} x^\top
$$

早期 $B \approx 0$ 时 **$\nabla_A$ 极小**、**$\nabla_B$** 由 $(A x)$ 支撑 — 若 $\eta_A=\eta_B$，$A$ 更新 **相对过慢/过快失衡** 取决于训练阶段。LoRA+ 用大 $\eta_B$ 加速 **特征建立**，小 $\eta_A$ 稳定 **投影缩放**。

### 8.2 与 LoRA 初始化的一致性

$B=0$ 初始化下，第一步只有 $B$ 获得有效信号；**大 $\eta_B$** 与这一 inductive bias **一致**。标准同 LR 浪费前几步的有效更新窗口。

---

## 9. 工程集成

### 9.1 HuggingFace PEFT

较新版本支持 `lorap_lr_ratio`（命名以版本为准）：

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    # 部分 fork 支持：
    # lorap_lr_ratio=16.0,
)
```

若 Config 未暴露，在 `Trainer` 的 `create_optimizer` 中拆分 param groups（§4.2）。

### 9.2 ms-swift（vlm_train 栈）

ms-swift 训练脚本常见 `--learning_rate` 全局；LoRA+ 需确认是否支持 **lora_lr_ratio** 或手动 patch。在 **无原生支持** 时：

1. 训练前 hook `model.named_parameters()` 分组；
2. 或通过 Swift 的 `custom_optimizer` 扩展点注入。

### 9.3 Axolotl / LLaMA-Factory

社区配置项常见：

```yaml
lora_r: 8
lora_alpha: 16
learning_rate: 0.0002
# loraplus_lr_ratio: 16  # 视 fork 而定
```

升级前查阅对应 README 的 **LoRA+** 章节。

---

## 10. 与其他「LoRA 改进」的关系

| 方法 | 改什么 | 与 LoRA+ 关系 |
| --- | --- | --- |
| **LoRA+** | LR 比 λ | 基线优化 |
| **rsLoRA** | 缩放 $\alpha/\sqrt{r}$ | **正交** — 可同时用 |
| **DoRA** | 幅度-方向分解 | 结构不同；可 DoRA + LoRA+ LR |
| **AdaLoRA** | 动态 rank | 可 AdaLoRA + 异 LR |
| **PiSSA / LoRA-GA** | 初始化 | 初始化与 LR 互补 |
| **VeRA** | 共享随机基 | 参数压缩；VeRA 向量是否异 LR 未在 VeRA 原文强调 |

**推荐组合（低成本）**：**LoRA + LoRA+ + rsLoRA** 已成为 2024–2025 社区 **默认三板斧**。

---

## 11. 局限与注意事项

| 点 | 说明 |
| --- | --- |
| **λ 非普适常数** | 16 是经验平台；极小模型或极大 LR 需重扫 |
| **仅适用于 LoRA 结构** | Adapter、IA³ 无 $A/B$ 分解，不适用 |
| **Embedding / Norm 层** | 若 LoRA 注入 embedding，参数命名需正确分组 |
| **分布式 ZeRO** | param group 在 shard 后仍有效；注意 **step 同步** |
| **不修复错误 target** | 只训 Q/V 而任务需要 FFN — LoRA+ 无效 |

---

## 12. 实践建议

### 12.1 默认配方（2026）

```text
LoRA r=8~64（按任务）
lora_alpha = r 或 2r（若用 rsLoRA 则 alpha 配合 sqrt 缩放）
η_A = 1e-4（7B QLoRA 可 2e-4）
η_B = 16 × η_A
λ = 16
optimizer = AdamW, betas=(0.9, 0.999)
scheduler = cosine + 3% warmup
```

### 12.2 验证 A/B

1. **同 seed、同 step**，仅改 λ：验证 dev 曲线 LoRA+ 应 **更低且更快**；
2. 扫 λ ∈ {1,4,16,32}，画 **GLUE avg vs λ** 平台；
3. 与 **merge 后推理** 对齐 — LoRA+ 只影响训练，合并权重无格式变化。

### 12.3 对本仓库（vlm_train / modelforge）

- **vlm_train** 当前脚本（如 `train_cloud_*_lora`）多为 **统一 `--learning_rate`**；接入 LoRA+ **零架构改动**，建议在 swift 层加 **双 LR** 或固定 λ=16 实验一轮。
- **modelforge cloud_emb** LoRA stage2：Embedding 对比任务对 **方向学习** 敏感，LoRA+ 值得在 **r=32/64** 上与 Instruct 对齐实验一并尝试（见 [LoRA技术深度调研报告.md](../LoRA技术深度调研报告.md) §13）。

---

## 13. 小结

LoRA+ 的核心洞察是：**LoRA 的两个低秩因子在优化意义上不是对称的** — $B$ 学特征、$A$ 学缩放，应用 **ηB ≫ ηA**。这一改动 **不增加参数、不增加推理成本**，却在 GLUE 与 LLaMA 上带来 **稳定、可复现** 的增益。**λ≈16** 是跨模型可用的默认起点；与 rsLoRA、QLoRA、DoRA 可叠加。任何正在使用 LoRA 的训练栈，都应把 LoRA+ 视为 **默认启用** 的优化选项，而非可选花哨技巧。

---

## 14. 参考文献

1. Hayou et al. **LoRA+: Efficient Low Rank Adaptation of Large Models**. ICML 2024. [arXiv:2402.12354](https://arxiv.org/abs/2402.12354)
2. Hu et al. **LoRA: Low-Rank Adaptation of Large Language Models**. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
3. Kalajdzievski. **A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA**. ICLR 2024. [arXiv:2312.03732](https://arxiv.org/abs/2312.03732)（rsLoRA）
4. Liu et al. **DoRA: Weight-Decomposed Low-Rank Adaptation**. ICML 2024. [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)

---

> **版本**: v1.0  
> **日期**: 2026-07-29  
> **关联**: [LoRA技术深度调研报告.md](../LoRA技术深度调研报告.md) · [VeRA详解.md](../VeRA/VeRA详解.md)
