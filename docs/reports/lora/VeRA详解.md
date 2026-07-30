# VeRA 技术详解

> 基于论文 [VeRA: Vector-based Random Matrix Adaptation](https://arxiv.org/abs/2310.11454)（Kopiczko et al., ICLR 2024）。
> 本文把 **共享冻结随机矩阵 A/B、仅训练逐层缩放向量、相对 LoRA 约 10× 参数压缩、以 seed 存储而非完整矩阵、GLUE/E2E 竞争力、多租户部署优势** 写全。

---

## 1. 一句话定位

**VeRA（Vector-based Random Matrix Adaptation）** 是 LoRA 的「极轻量变体」：不再为每一层独立学习低秩矩阵 $A,B$，而是在**全模型共享一对冻结的随机投影矩阵**，每层只训练两个长度分别为 $r$ 与 $d$ 的**缩放向量** $\mathbf{b}_\ell, \mathbf{d}_\ell$。在 rank $r=256$ 设定下，可训练参数量约为 LoRA 的 **1/10**，GLUE 与 E2E NLG 上仍能与 LoRA 打平或接近。

| 项 | 内容 |
| --- | --- |
| 发表 | arXiv:2310.11454；ICLR 2024 |
| 核心思想 | 共享随机 $A,B$ + 逐层向量缩放 |
| 相对 LoRA | 参数量约 **10× 更少**；checkpoint 可仅存 **seed + 向量** |
| 典型 rank | $r=256$（论文主实验） |
| 适用场景 | **多租户**、大量 adapter 并存、存储/传输受限 |
| 局限 | 表达力依赖共享基；极低 rank 时可能弱于独立 LoRA |

---

## 2. 问题背景：LoRA 仍然「太重」

### 2.1 LoRA 的成功与代价

LoRA（Hu et al., 2021）通过低秩分解 $\Delta W = BA$ 将全量微调压缩为少量可训练参数，已成为 LLM PEFT 事实标准。但对 **multi-tenant serving** 而言，每个租户一份 LoRA adapter 仍不便宜：

| 场景 | LoRA 痛点 |
| --- | --- |
| SaaS 多租户 | 1000 租户 × 每份 10–50MB adapter → 存储与加载成本高 |
| 边缘/移动端 | 频繁切换 adapter 时 I/O 与内存压力大 |
| 联邦/个性化 | 每人微调一份 LoRA，同步与版本管理复杂 |

LoRA 每层、每模块（Q/K/V/O、FFN）都要存独立的 $A \in \mathbb{R}^{r \times d_\text{in}}$、$B \in \mathbb{R}^{d_\text{out} \times r}$。对 7B 模型、rank 8、全 attention + MLP，adapter 体积仍可达 **数十 MB**。

### 2.2 能否再压缩？

两条思路：

1. **更低 rank** — 参数再减，但 GLUE/生成任务上性能常明显掉档。
2. **共享结构** — 若不同层、不同模块的 $\Delta W$ 落在相近子空间，可共享基底，只学「混合系数」。

VeRA 走第二条路：**固定一组随机投影，只学每层如何缩放**。

---

## 3. 方法：共享随机矩阵 + 向量缩放

### 3.1 标准 LoRA 回顾

对线性层 $W_0 \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$，LoRA 写：

$$
W = W_0 + \frac{\alpha}{r} B A, \quad B \in \mathbb{R}^{d_\text{out} \times r},\; A \in \mathbb{R}^{r \times d_\text{in}}
$$

训练 $A,B$；$W_0$ 冻结。

### 3.2 VeRA 分解

VeRA 将 $B,A$ **拆成共享冻结部分 + 可训练对角缩放**：

$$
W = W_0 + \Lambda_b \, B_0 \, \Lambda_d \, A_0
$$

符号约定（论文对每层 $\ell$）：

| 符号 | 形状 | 是否训练 | 含义 |
| --- | --- | --- | --- |
| $A_0$ | $r \times d_\text{in}$ | **冻结**（全层共享） | 随机高斯初始化后固定 |
| $B_0$ | $d_\text{out} \times r$ | **冻结**（全层共享） | 随机高斯初始化后固定 |
| $\Lambda_d = \mathrm{diag}(\mathbf{d}_\ell)$ | $d_\text{in} \times d_\text{in}$ | **训练** $\mathbf{d}_\ell$ | 输入侧缩放向量 |
| $\Lambda_b = \mathrm{diag}(\mathbf{b}_\ell)$ | $d_\text{out} \times d_\text{out}$ | **训练** $\mathbf{b}_\ell$ | 输出侧缩放向量 |

**实现上**不构造完整对角矩阵，而是逐元素乘：

$$
h = W_0 x + B_0 \big( \mathbf{b}_\ell \odot (A_0 (\mathbf{d}_\ell \odot x)) \big)
$$

其中 $\odot$ 为 Hadamard 积。

### 3.3 参数量对比

对**单个**线性层（忽略 bias）：

| 方法 | 可训练参数 |
| --- | --- |
| LoRA | $r \cdot d_\text{in} + r \cdot d_\text{out}$ |
| VeRA | $d_\text{in} + d_\text{out}$ |

例：$d_\text{in}=d_\text{out}=4096$，$r=256$：

- LoRA：$256 \times 4096 \times 2 \approx 2.1$M / 层
- VeRA：$4096 \times 2 \approx 8$K / 层

**约 260× 单层参数比**；全模型因 $A_0,B_0$ 共享不再重复计数，相对「每层独立 LoRA」总压缩约 **10×**（论文 Table 1 在 RoBERTa/RoBERTa-large 上实测）。

### 3.4 随机矩阵与 seed 存储

$A_0, B_0$ 由**固定随机种子**生成，**不写入 checkpoint**：

```text
checkpoint = { d_ℓ, b_ℓ for all layers } + metadata(seed, rank, target_modules)
```

部署时按 seed 现场重建 $A_0,B_0$。这对 **multi-tenant** 极友好：$N$ 个租户共享同一份「基矩阵逻辑」，磁盘上只存 $N$ 组小向量。

---

## 4. 理论直觉：为什么随机共享仍有效？

### 4.1 随机特征与 Johnson–Lindenstrauss

冻结随机 $A_0$ 可看作把输入投影到 $r$ 维随机子空间；$B_0$ 再映回输出空间。经典结论：足够维的随机投影以高概率近似保持内积结构（JL 引理）。VeRA 不学习投影方向，而学习 **「哪些输入/输出坐标在该子空间里更重要」** —— 这正是 $\mathbf{d},\mathbf{b}$ 的作用。

### 4.2 与 LoRA 的表达力关系

LoRA 同时学「子空间方向 + 子空间内的秩 $r$ 变换」；VeRA 固定方向，只学 **坐标级门控**。因此：

- rank 较高（如 256）时，随机基足够覆盖任务相关方向，**向量缩放足以补偿**；
- rank 极低时，LoRA 可旋转 $A,B$ 对齐梯度，VeRA 更吃亏。

### 4.3 与 Adapter / BitFit 的对照

| 方法 | 改什么 | 参数量级 |
| --- | --- | --- |
| BitFit | bias | 极小，但只动偏置 |
| Adapter | 插入瓶颈 MLP | 中等，改激活路径 |
| LoRA | 低秩增量矩阵 | 中–大（随 rank） |
| VeRA | 共享基 + 向量门控 | **小** |

VeRA 仍改 **权重增量路径**（与 LoRA 同族），而非仅 bias。

---

## 5. 训练细节与实现

### 5.1 初始化

- $A_0, B_0$：$\mathcal{N}(0, \sigma^2)$，Kaiming 风格；**全模型同一对**（attention 与 FFN 是否共享：论文默认 **跨层、跨模块类型共享一套** $A_0,B_0$，维度按最大 $d$ 对齐或分组建模 — 实现见官方 repo）。
- $\mathbf{d}_\ell$：初始化为 **全 1**（恒等缩放）。
- $\mathbf{b}_\ell$：**全 0**（初始 $\Delta W=0$，与 LoRA 中 $B=0$ 同理）。

### 5.2 优化超参（论文默认）

| 项 | RoBERTa / GLUE | LLaMA / E2E |
| --- | --- | --- |
| Optimizer | AdamW | AdamW |
| LR | $3\times10^{-3}$（向量） | 与 LoRA 对齐搜索 |
| Rank $r$ | 256 | 256 |
| Epoch | 任务默认 | 5–10 |
| 目标模块 | Q,V（RoBERTa） | Q,K,V,O + FFN |

向量参数少，**可用更大学习率** 相对 LoRA 的 $A,B$；但仍需 warmup 防早期震荡。

### 5.3 与 HuggingFace PEFT

PEFT 库自 v0.7+ 支持 `VeRAConfig`：

```python
from peft import VeraConfig, get_peft_model

config = VeraConfig(
    r=256,
    target_modules=["q_proj", "v_proj"],
    projection_prng_key=0,  # 共享 A0,B0 的 seed
)
model = get_peft_model(base_model, config)
```

合并权重：与 LoRA 类似，可将 VeRA 增量 **merge 进 $W_0$** 做推理部署（失去多租户切换灵活性）。

---

## 6. 实验：GLUE

### 6.1 设置

- 骨干：**RoBERTa-base / large**
- 对比：Full FT、BitFit、Adapter、LoRA（$r=8,256$）、VeRA（$r=256$）
- 指标：GLUE 8 任务平均

### 6.2 主要结果（定性）

| 方法 | 参数量（相对） | GLUE 表现 |
| --- | --- | --- |
| Full FT | 100% | 上界 |
| LoRA r=8 | 中 | 强基线 |
| LoRA r=256 | 大 | 接近 Full FT |
| **VeRA r=256** | **≈ LoRA r=256 的 1/10** | **与 LoRA r=256 相当**，优于 LoRA r=8 |
| BitFit | 极小 | 多数任务弱于 VeRA |

**结论**：在 **相同 rank** 下，VeRA 用十分之一可训练参数达到与 LoRA 相近的 GLUE 分数；说明共享随机基并未严重限制 NLU 微调。

### 6.3 消融：rank 与共享策略

- **降 rank**（如 64）：VeRA 掉点通常 **快于** LoRA — 固定基对 rank 更敏感。
- **不共享 $A_0,B_0$**（退化为「每层 LoRA 但 A,B 冻结」）：介于 LoRA 与 VeRA 之间，参数量上升。
- **只训 $\mathbf{b}$ 或只训 $\mathbf{d}$**：明显弱于两者同训。

---

## 7. 实验：E2E NLG（GPT-2 / BART）

### 7.1 任务

E2E 餐厅描述生成（BLEU、NIST、METEOR、ROUGE-L、CIDEr）。

### 7.2 结果要点

- VeRA 在 GPT-2 Medium/Large、BART 上 **匹配或略低于** 同 rank LoRA，但 **显著优于** LoRA r=8。
- 生成任务对 $\Delta W$ 结构更敏感，VeRA 与 LoRA 差距略大于 GLUE，仍在实用范围内。

---

## 8. 大模型实验（LLaMA 家族）

论文在 **LLaMA-7B / 13B** 上补充 commonsense reasoning（8 数据集平均）：

| 方法 | 可训练参数 | 平均准确率 |
| --- | --- | --- |
| LoRA r=8 | ~4.7M | 强 |
| VeRA r=256 | **~0.5M 量级** | **与 LoRA r=8 接近或略优** |

说明在大模型上，**极少参数 + 高 rank 随机基** 仍可驱动有效适配 — 对「每人一个小 adapter」场景极具吸引力。

---

## 9. Multi-tenant 与系统优势

### 9.1 存储模型

设模型有 $L$ 个注入层，隐藏维 $d$，租户数 $N$：

| 方案 | 每租户存储 | 总存储 |
| --- | --- | --- |
| LoRA r=8 | $O(L \cdot r \cdot d)$ | $O(N \cdot L \cdot r \cdot d)$ |
| VeRA r=256 | $O(L \cdot d)$ | $O(N \cdot L \cdot d)$ + **一份** $A_0,B_0$ 逻辑 |

当 $r=256$ 时 LoRA 单层参数量仍大于 VeRA 的 $2d$；**rank 越大，VeRA 优势越明显**。

### 9.2 Serving 流程

```text
加载 base LLM（共享）
  ↓
按 tenant_id 加载 b_ℓ, d_ℓ 向量（KB 级）
  ↓
用全局 seed 生成 A0, B0（可缓存）
  ↓
前向：W0 x + B0 ( b ⊙ (A0 ( d ⊙ x )) )
```

vLLM / TGI 的 multi-LoRA 已支持 batch 内多 adapter；VeRA 向量更小，**切换与缓存**成本更低（需推理引擎显式支持 VeRA op）。

### 9.3 与 QLoRA 组合

VeRA 只动 **adapter 参数**，可与 4-bit 量化基座（QLoRA）正交组合：NF4 存 $W_0$，VeRA 向量 fp16/bf16 训练。论文未主打此组合，但工程上自然。

---

## 10. 局限与失败模式

| 局限 | 说明 |
| --- | --- |
| **表达力上限** | 固定 $A_0,B_0$ 无法学习任务专属子空间旋转；极难任务可能需 LoRA/Full FT |
| **Rank 敏感** | 低 rank 下 VeRA 弱于 LoRA；实践常需 $r \gg 8$（如 256） |
| **共享基假设** | 若各层最优 $\Delta W$ 子空间差异大，共享 $A_0,B_0$ 成为瓶颈 |
| **实现生态** | 相对 LoRA，框架/内核优化较少；自定义层需手写缩放逻辑 |
| **合并部署** | merge 后失去「极小 adapter」优势；与 LoRA 相同权衡 |

---

## 11. 与相关方法对照

| 方法 | 可训练部分 | 相对 LoRA 参数量 | 备注 |
| --- | --- | --- | --- |
| **LoRA** | $A, B$ 每层独立 | 1× | 标准基线 |
| **VeRA** | $\mathbf{b}, \mathbf{d}$；共享 $A_0,B_0$ | **~0.1×** | 本文 |
| **AdaLoRA** | 自适应 rank 的 $A,B$ | 动态 | 预算分配，非压缩共享 |
| **DoRA** | LoRA + 幅度向量 | > LoRA | 表达力增强方向 |
| **LoRA+** | 同 LoRA；**异 LR** | 1× | 训练效率，非参数压缩 |
| **PiSSA** | 初始化 $A,B$ 为 SVD 主成分 | 1× | 收敛更快 |

**选型**：要 **最少参数 / 多租户** → VeRA；要 **最强单任务质量** → LoRA/DoRA + 调 rank；要 **训练更快** → LoRA+ / PiSSA。

---

## 12. 实践建议

### 12.1 何时优先考虑 VeRA

- SaaS **千级租户**，每人一个小任务 adapter；
- **带宽受限**（边缘设备频繁拉 adapter）；
- 已有 LoRA 管线，愿用 **更大 rank + 更少存储** 换几乎同等精度。

### 12.2 超参起点

| 超参 | 建议 |
| --- | --- |
| $r$ | **256**（论文主设定）；资源极紧可试 128 并验证 |
| target_modules | 与 LoRA 相同：LLM 用 `q_proj,v_proj` 或 `all-linear` |
| LR | 向量参数可用 **1e-3 ~ 3e-3**（高于 LoRA 的 1e-4 量级） |
| seed | **固定** `projection_prng_key`，多机训练保持一致 |
| warmup | 3–6% steps |

### 12.3 验证清单

1. 同数据 **LoRA r=8 vs VeRA r=256** 对比 — 预期 VeRA 参数更少且不低于 LoRA r=8；
2. 检查 **seed reproducibility** — 两台机器重建 $A_0,B_0$ 比特一致；
3. multi-tenant 压测：**冷启动加载时间** vs LoRA。

---

## 13. 小结

VeRA 的核心贡献是：**把 LoRA 的可学习自由度从「整个低秩矩阵」压缩到「逐层向量门控」**，并通过 **跨层共享冻结随机投影** 把存储降到可忽略级别。它在 GLUE、E2E 与 LLaMA 推理任务上证明：在足够 rank 下，**参数效率与性能可以同时要好**。对 multi-tenant LLM 服务，VeRA 是比「为每个客户训一份 LoRA」更经济的 PEFT 选项；对单任务极致性能，仍建议 LoRA/DoRA + 系统化超参（含 LoRA+ 学习率）。

---

## 14. 参考文献

1. Kopiczko et al. **VeRA: Vector-based Random Matrix Adaptation**. ICLR 2024. [arXiv:2310.11454](https://arxiv.org/abs/2310.11454)
2. Hu et al. **LoRA: Low-Rank Adaptation of Large Language Models**. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
3. Dettmers et al. **QLoRA: Efficient Finetuning of Quantized LLMs**. NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
4. HuggingFace **PEFT VeRA 文档**. https://huggingface.co/docs/peft/main/en/conceptual_guides/adapter#vera

---

> **版本**: v1.0  
> **日期**: 2026-07-29  
> **关联**: [LoRA技术深度调研报告.md](LoRA技术深度调研报告.md) · [LoRA+详解.md](LoRA+详解.md)
