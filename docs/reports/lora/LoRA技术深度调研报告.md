# LoRA 技术深度调研报告
## PEFT / LoRA 族：原理、谱系、工程实践与选型

> **版本**: v1.0  
> **日期**: 2026-07-29  
> **范围**: 参数高效微调（PEFT），以 LoRA 为主线，覆盖 Adapter→Prefix/Prompt→LoRA→AdaLoRA→QLoRA→VeRA→rsLoRA→LoRA+→DoRA 等；量化、推理 serving、与本仓库训练实践  
> **关联专题**: [VeRA详解.md](VeRA详解.md) · [LoRA+详解.md](LoRA+详解.md) · [资料清单_论文与博客.md](资料清单_论文与博客.md)

---

## 目录

1. [执行摘要（关键结论）](#1-执行摘要关键结论)
2. [问题背景：全量微调的成本与部署困境](#2-问题背景全量微调的成本与部署困境)
3. [PEFT 范式全景与三大家族](#3-peft-范式全景与三大家族)
4. [LoRA 核心原理与数学形式](#4-lora-核心原理与数学形式)
5. [经典论文谱系时间线](#5-经典论文谱系时间线)
6. [逐方法对比表](#6-逐方法对比表)
7. [量化与内存：QLoRA / NF4 / bitsandbytes](#7-量化与内存qlora--nf4--bitsandbytes)
8. [缩放与优化：rsLoRA、LoRA+、AdaLoRA 预算](#8-缩放与优化rslora-lora-adalora-预算)
9. [表达力增强：DoRA、全层适配](#9-表达力增强dora全层适配)
10. [工程生态：HF PEFT、ms-swift、Axolotl、vLLM / TGI](#10-工程生态hf-peftms-swiftaxolotlvllm--tgi)
11. [实践选型指南与超参决策树](#11-实践选型指南与超参决策树)
12. [开放问题与前沿](#12-开放问题与前沿)
13. [对本仓库训练实践的启示](#13-对本仓库训练实践的启示)
14. [参考文献](#14-参考文献)

**附录**

- [附录 A：术语表](#附录-a术语表)
- [附录 B：参数量估算公式](#附录-b参数量估算公式)

---

## 1. 执行摘要（关键结论）

**LoRA（Low-Rank Adaptation）** 自 2021 年提出以来，已成为大语言模型（LLM）与多模态模型 **参数高效微调（PEFT）** 的事实标准：冻结预训练权重 $W_0$，只训练低秩增量 $\Delta W = BA$，在 **<1% 可训练参数** 下逼近全量微调（Full Fine-Tuning, FFT）效果。2023–2026 年，围绕 LoRA 形成了 **量化（QLoRA）→ 缩放修正（rsLoRA）→ 优化修正（LoRA+）→ 表达力（DoRA）→ 极致压缩（VeRA）** 的方法谱系，并与 HuggingFace PEFT、ms-swift、vLLM multi-LoRA serving 等工程栈深度耦合。

### 1.1 五条关键结论

| # | 结论 | 实践含义 |
| --- | --- | --- |
| 1 | **LoRA 默认训练配方偏 suboptimal** | 启用 **LoRA+（ηB≈16ηA）** 与 **rsLoRA（α/√r）** 几乎零成本，应作为默认 |
| 2 | **QLoRA 把 65B 级微调拉到单卡 48GB** | NF4 量化基座 + bf16 LoRA；质量损失通常 <1% vs bf16 FFT |
| 3 | **rank 与 target_modules 比「多训几层」更敏感** | LLM 常 **r=8–64**；Embedding/检索可 **r=32–128** + 全 linear |
| 4 | **DoRA / 全层 LoRA 缩小与 FFT 差距** | 任务难、数据少时考虑 DoRA 或 `all-linear` |
| 5 | **Multi-tenant 看 VeRA / multi-LoRA serving** | 千级 adapter 用 VeRA 降存储；推理用 vLLM/TGI batch 多 adapter |

### 1.2 2026 年推荐默认栈

```text
基座: bf16 或 QLoRA 4-bit (NF4, double quant)
PEFT: LoRA r=16, alpha=16, target=all-linear 或 q/k/v/o + gate/up/down
优化: LoRA+ λ=16, rsLoRA 缩放, AdamW, cosine, warmup 3%
推理: merge_lora 单租户 / vLLM multi-LoRA 多租户
```

### 1.3 报告结构说明

本报告按 **背景 → 原理 → 谱系 → 对比 → 子专题（量化/优化/表达力）→ 工程 → 选型 → 前沿 → 本仓实践** 组织，与 [知识蒸馏技术深度调研报告](../distillation/知识蒸馏技术深度调研报告.md)、[Embedding调研报告](../embedding/Embedding调研报告.md) 体例对齐。

---

## 2. 问题背景：全量微调的成本与部署困境

### 2.1 全量微调（FFT）的成本

对参数量 $N$ 的 Transformer，FFT 需要：

| 资源 | 量级（7B 模型示意） |
| --- | --- |
| 可训练参数 | **7B（100%）** |
| 优化器状态（Adam） | ~**2×** 参数量（fp32 m/v） |
| 激活 + 梯度 | 与 batch、seq、并行策略相关；常 **> 权重** |
|  checkpoint | 每步存 **完整 7B×2**（bf16）≈ 14GB+ |

多任务、多租户场景下，**每个任务一份完整 FFT 权重** 在存储、版本管理、GPU 显存上均不可扩展。

### 2.2 部署困境

1. **单模型多技能**：同一基座需切换「客服 / 代码 / 检索增强」等行为 → 需要 **轻量 adapter** 而非整模复制。
2. **边缘与端侧**：无法存放多份 7B+ 全量权重。
3. **推理并发**：SaaS 需 **batch 内不同用户不同 adapter**（multi-LoRA）。
4. **合规与迭代**：仅上传 **MB 级 delta** 而非全权重，便于审计与回滚。

### 2.3 PEFT 的目标函数

在冻结 $W_0$ 的前提下，学习 $\Delta\theta$ 使：

$$
\min_{\Delta\theta} \mathcal{L}\big(W_0 \oplus \Delta\theta;\, \mathcal{D}_\text{task}\big), \quad |\Delta\theta| \ll |W_0|
$$

其中 $\oplus$ 表示插入/低秩叠加等结构。理想 PEFT 应满足：

- **参数效率**：$|\Delta\theta|/|W_0| < 1\%$
- **训练效率**：显存 ≤ QLoRA 单卡可训
- **推理效率**：可 **merge** 或 **低开销切换**
- **效果**：下游任务 ≥ 95% FFT 性能（依任务）

---

## 3. PEFT 范式全景与三大家族

### 3.1 分类轴

| 轴 | 选项 |
| --- | --- |
| 修改对象 | 权重 / 激活 / 输入 prompt |
| 插入位置 | Attention / FFN / Embedding |
| 是否改结构 | 外挂模块 vs 低秩增量 |
| 推理开销 | 零开销（merge）vs 额外 latency |

### 3.2 三大家族

#### 3.2.1 **Additive（加性低秩 / 侧路）**

- **LoRA**、VeRA、DoRA、AdaLoRA
- 形式：$W = W_0 + \Delta W$
- 优点：可 merge；实现简单
- 代表场景：LLM SFT、Embedding 微调

#### 3.2.2 **Selective（选择性微调）**

- **BitFit**（只训 bias）、**IA³**（逐通道缩放）、**VeRA 的 d/b 向量**
- 参数量极小
- 表达力通常弱于 LoRA

#### 3.2.3 **Reparameterization / 外挂（Adapter / Prefix / Prompt）**

- **Adapter**（Houlsby / Pfeiffer）：层间瓶颈 MLP
- **Prefix-Tuning / P-Tuning v2**：可学习 prefix KV
- **Prompt Tuning**：只训 soft prompt embedding
- 优点：不改权重；缺点：推理 **序列变长** 或 **额外 forward**，延迟增加

### 3.3 家族选型速览

```text
要 merge 部署、最低推理开销     → LoRA 族
要最少参数、千租户              → VeRA / IA³
要动 attention 行为不改权重      → Prefix / P-Tuning
Encoder NLU 经典强基线          → Adapter（现多被 LoRA 取代）
```

---

## 4. LoRA 核心原理与数学形式

### 4.1 基本公式

对线性层 $y = W_0 x$，LoRA 改为：

$$
y = W_0 x + \frac{\alpha}{r} B A x
$$

- $W_0 \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$：**冻结**
- $A \in \mathbb{R}^{r \times d_\text{in}}$，$B \in \mathbb{R}^{d_\text{out} \times r}$：**可训练**
- $r \ll \min(d_\text{in}, d_\text{out})$：秩
- $\alpha$：缩放超参（常见 $\alpha = r$ 或 $2r$）

### 4.2 初始化与 inductive bias

- $A$：随机高斯（Kaiming / N(0, σ²)）
- $B$：**零初始化** → 训练初 $\Delta W = 0$，不破坏预训练

### 4.3 为何低秩有效？

1. **内在维度（Intrinsic Dimension）**：Li et al. 表明微调有效自由度远低于参数量。
2. **过参数化网络的低秩更新**：任务增量常落在少数奇异方向。
3. **正则化效应**：限制 rank 抑制 catastrophic forgetting。

### 4.4 注入位置（LLM）

| 模块 | 张量 | 常见默认 |
| --- | --- | --- |
| Self-Attn | q_proj, k_proj, v_proj, o_proj | q,v 或 q,k,v,o |
| MLP | gate_proj, up_proj, down_proj | 难任务加 MLP |
| Embedding | embed_tokens | 少训 / 不训 |

**经验**：只训 Q/V 为 **省参**；**all-linear** 更接近 FFT。

### 4.5 Merge 与 Unmerge

训练后：

$$
W_\text{merged} = W_0 + \frac{\alpha}{r} B A
$$

推理单租户时可 merge，**零 adapter 开销**。多租户则 **不 merge**，运行时加载 $\Delta W$ 或 cache BA。

### 4.6 与 Conv / 多模态

LoRA 可扩展到 Conv2d（Stable Diffusion 常用）与 VLM **投影层**；视觉 backbone 常 **更大 rank** 或 **全层**。

### 4.7 Embedding 模型中的 LoRA

文本 Embedding 微调与对话 SFT 有显著差异，LoRA 配置需单独考虑：

| 维度 | 对话 SFT | Embedding / 检索 |
| --- | --- | --- |
| 目标 | 条件生成 | 句向量几何（InfoNCE / Cosent） |
| 池化 | 不适用 | last-token / mean / CLS |
| 注意力 | 因果即可 | 常需 **双向** 或 soft-mask |
| rank | r=8–32 常见 | **r=32–128** 更稳 |
| target | q,v 可起步 | **all-linear**（含 MLP） |
| 模板 | chat template | **Instruct 对齐** 训练/评测一致 |

**常见坑**（见 modelforge `cloud_emb` 评测）：裸训 LoRA 用 Instruct 模板评测 → 分布错配，MRR 远低于 Base+Instruct。修复路径是 **训练数据加 instruction**，而非单纯增大 rank。

LoRA 在 Embedding 上的表达力上限仍低于 **全参双向微调**（QZhou、gte-Qwen2 等）；工程上应先把 **LoRA+ + rsLoRA + all-linear + r↑** 调满，再决定是否 FFT。

### 4.8 分布式训练中的 LoRA

| 策略 | LoRA 行为 | 注意点 |
| --- | --- | --- |
| **DDP** | 仅 adapter 梯度同步 | 基座无梯度，通信量小 |
| **DeepSpeed ZeRO-2/3** | LoRA 参数可 shard | quantized 基座 + ZeRO-3 需版本匹配 |
| **FSDP** | wrap LoRA 层 | `use_orig_params=True` 时分 param group 更稳 |
| **多卡 ms-swift** | `--deepspeed zero2` 常见 | checkpoint 默认 **仅 adapter** |

保存策略：PEFT 训练结束通常只有 `adapter_model.safetensors`（数 MB–数百 MB），便于 artifact 管理与多实验对比。

---

## 5. 经典论文谱系时间线

### 5.1 总览时间轴

```text
2019 ─ Adapter (Houlsby et al.) ★ NLP PEFT 先驱
2019 ─ BitFit — 只训 bias
2021 ─ Prefix-Tuning (Li & Liang)
2021 ─ LoRA (Hu et al.) ★★ 低秩增量成为主流
2022 ─ P-Tuning v2 — 深层 prompt
2022 ─ AdaLoRA — 自适应 rank 预算
2023 ─ QLoRA (Dettmers et al.) ★★ 4-bit + LoRA 单卡训 65B
2023 ─ VeRA — 共享随机基 + 向量缩放
2023 ─ IA³ / 各类 selective 方法并行发展
2024 ─ rsLoRA — α/√r  rank 稳定缩放
2024 ─ LoRA+ (Hayou et al.) — A/B 异学习率
2024 ─ DoRA — 幅度-方向分解
2024 ─ PiSSA — SVD 初始化 LoRA
2024 ─ LoRA-GA — 梯度对齐初始化
2025 ─ GaLore — 低秩梯度投影（非 LoRA 但相关）
2025+ ─ Multi-adapter routing、RLHF/DPO + LoRA 标配
```

### 5.2 各节点一句话

| 年份 | 工作 | 贡献 |
| --- | --- | --- |
| 2019 | **Adapter** | 层间 down-up MLP；参数量 ~3–5% |
| 2021 | **Prefix-Tuning** | 可学习 prefix 影响 KV cache |
| 2021 | **LoRA** | 低秩 $\Delta W$；可 merge |
| 2022 | **AdaLoRA** | SVD 式参数化 + 重要性剪枝 rank |
| 2023 | **QLoRA** | NF4 基座 + LoRA；Paged Optimizer |
| 2023 | **VeRA** | 共享 $A_0B_0$，训 $\mathbf{b,d}$ |
| 2024 | **rsLoRA** | 修正 $\alpha/r$ → $\alpha/\sqrt{r}$ |
| 2024 | **LoRA+** | $\eta_B \gg \eta_A$，λ≈16 |
| 2024 | **DoRA** | $W = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_\text{row}}$ 方向 + 幅度 |
| 2024 | **PiSSA** | 用 $W_0$ 主奇异向量初始化 $A,B$ |

### 5.3 范式演进逻辑

```text
外挂模块 (Adapter/Prefix)
    ↓ 推理开销大
低秩增量 (LoRA) — 可 merge
    ↓ 显存仍紧
量化 + LoRA (QLoRA)
    ↓ 训练 suboptimal
rsLoRA + LoRA+ — 优化层
    ↓ 表达力瓶颈
DoRA / 全层 / 高 rank
    ↓ 多租户存储
VeRA — 参数再压缩
```

---

## 6. 逐方法对比表

### 6.1 主表

| 方法 | 可训练参数（7B 量级） | 训练显存 | 推理延迟（不 merge） | 可 merge | 典型场景 |
| --- | --- | --- | --- | --- | --- |
| **FFT** | 7B (100%) | 极高 | 基线 | — | 数据大、要极致 |
| **Adapter** | ~0.5–3% | 中高 | +5–15% | 否 | 老 NLU 基线 |
| **Prefix/Prompt** | ~0.01–0.1% | 中 | +（长 KV） | 否 | 少样本 prompt |
| **LoRA r=8** | ~4–8M | 低 | +1–3% | **是** | **默认首选** |
| **LoRA r=64** | ~30–60M | 中 | +2–5% | **是** | 难任务 |
| **QLoRA** | 同 LoRA | **最低** | 同左或 4bit 基座 | **是** | 单卡大模型 |
| **AdaLoRA** | 预算 B | 中 | 同 LoRA | 是 | 固定参数预算 |
| **VeRA r=256** | ~LoRA/10 | 低 | 同 LoRA | 是 | **multi-tenant** |
| **DoRA** | LoRA + 幅度 | 中 | 略高于 LoRA | 是 | 贴近 FFT |
| **BitFit** | <<0.1% | 低 | 基线 | 是 | 极省参、弱表达 |
| **IA³** | ~0.01% | 低 | 低 | 部分 | 与 VeRA 类似思路 |

*参数量随 target_modules、层数、rank 变化；7B LLaMA 全 linear r=8 约 4.7M 为社区常引数字。*

### 6.2 效果–效率象限

```text
          效果高
            │
     DoRA   │   FFT
     全层LoRA│
            │
  ──────────┼────────── 参数/显存高
            │
   LoRA+    │   QLoRA
   rsLoRA   │
     VeRA   │   BitFit
            │
          效果低
     参数/显存低
```

### 6.3 方法组合兼容性

| 组合 | 是否推荐 |
| --- | --- |
| QLoRA + LoRA+ + rsLoRA | **强烈推荐** |
| DoRA + LoRA+ | 推荐 |
| VeRA + QLoRA | 可行 |
| AdaLoRA + DoRA | 少见，需自行验证 |
| LoRA + Prefix 同时 | 一般二选一 |

---

## 7. 量化与内存：QLoRA / NF4 / bitsandbytes

### 7.1 QLoRA 架构

**QLoRA** = **4-bit 量化冻结基座** + **bf16 LoRA adapter** + **分页优化器**

```text
Forward:  W0 (NF4) ──dequant──► fp16/bf16  ×  x
          +  BA (bf16 trainable)
Backward: 仅 LoRA + 必要 buffer；基座权重不存 fp32 master
```

### 7.2 NF4（Normal Float 4）

- 针对 **正态分布权重** 优化的 4-bit 非均匀量化码本
- **Double Quantization**：对量化常数再量化，进一步省显存
- 典型配置：`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=bfloat16`

### 7.3 显存估算（经验）

| 模型 | FFT bf16 | LoRA bf16 | QLoRA 4bit |
| --- | --- | --- | --- |
| 7B | ~60GB+ | ~24–32GB | **~12–16GB** |
| 13B | OOM 单卡 | ~40GB | **~20–24GB** |
| 65B | 多卡 | 多卡 | **1×48GB（论文）** |

### 7.4 bitsandbytes 要点

- `Linear4bit` 包装冻结层
- `paged_adamw_8bit` / 32bit optimizer on LoRA only
- 与 DeepSpeed ZeRO、FSDP **可组合**（需注意 quantized 参数 shard 策略）

### 7.5 质量与坑

| 现象 | 对策 |
| --- | --- |
| 4bit 基座略降 perplexity | 增大 LoRA rank；训 longer |
| 量化 outlier 层敏感 | QLoRA 对 outlier 做 **fp16 保留**（部分实现） |
| merge 后部署 | `merge_and_unload()` → 全 bf16 权重；或 **GPTQ/AWQ** 再压缩 |

### 7.6 与训练栈

- **HF TRL / PEFT**：`BitsAndBytesConfig` + `LoraConfig`
- **ms-swift**：`--quantization_bit 4` 等 flags（以版本文档为准）
- **Axolotl**：`load_in_4bit: true` in yaml

### 7.7 训练后量化与 merge 流水线

QLoRA 解决 **训练显存**；上线常走 **merge → 再量化**：

```text
QLoRA 训练 checkpoint (adapter only)
    ↓ merge_and_unload()
bf16 全量权重 (单任务)
    ↓ AutoRound / GPTQ / AWQ
INT4/INT8 部署权重
```

本仓 `vlm_train/scripts/board_qwen35_2b_autoround_3die.sh` 即此路径：先 merge LoRA 到 `MERGED`，再 AutoRound 量化上板。**merge 前后必须同一 eval 集对比**，避免 merge 数值误差或 template 不一致导致线上回退。

| 阶段 | 格式 | 适用 |
| --- | --- | --- |
| 训练 | NF4 基座 + bf16 LoRA | 单卡 24–48GB |
| 实验 | bf16 merge 或未 merge adapter | dev 评测 |
| 生产 | INT4/INT8 全量 | 延迟 / 显存 |

---

## 8. 缩放与优化：rsLoRA、LoRA+、AdaLoRA 预算

### 8.1 原始缩放的问题

LoRA 使用 $\frac{\alpha}{r}$ 缩放 $BA$。当 **rank 增大** 时，$\Delta W$ 幅度 **随 r 线性放大**（若 $A,B$ 范数稳定），导致：

- 高 rank 训练 **不稳定**
- 不同 rank 之间 **超参不可迁移**

### 8.2 rsLoRA（Rank-Stabilized LoRA）

Kalajdzievski 提出：

$$
W = W_0 + \frac{\alpha}{\sqrt{r}} B A
$$

**效果**：跨 rank 更稳定；**LoRA+ 与 rsLoRA 正交**，建议同时启用。详见 [资料清单](资料清单_论文与博客.md) arXiv:2312.03732。

### 8.3 LoRA+

- $\eta_B = \lambda \eta_A$，**λ≈16**
- 零额外参数；详见 [LoRA+详解.md](LoRA+详解.md)

### 8.4 AdaLoRA：参数预算分配

**AdaLoRA** 将 LoRA 参数化为 SVD 形式并按 **重要性分数** 动态增减 rank：

1. 训练初期：各层较高 rank
2. 周期性：评估奇异值重要性
3. 剪枝：全局预算 $B$ 下保留重要方向

| 对比 | LoRA | AdaLoRA |
| --- | --- | --- |
| rank | 固定 | **动态** |
| 适用 | 简单 | **固定 GPU 预算、要榨性能** |
| 工程 | PEFT 原生 | 配置略复杂 |

### 8.5 三件套默认

```yaml
lora_r: 16
lora_alpha: 16          # 配合 rsLoRA 时 alpha 不需随 r 线性增
use_rslora: true
lorap_lr_ratio: 16      # LoRA+
learning_rate: 1.0e-4   # 作用于 A；B 自动 ×16
```

---

## 9. 表达力增强：DoRA、全层适配

### 9.1 DoRA（Weight-Decomposed LoRA）

将权重分解为 **幅度（magnitude）** 与 **方向（direction）**：

$$
W = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_\text{row}}
$$

- $m$：可训练 **幅度向量**（逐输出通道）
- 方向：LoRA 扰动 + 归一化

**动机**：LoRA 对 **幅度与方向耦合更新**，与 FFT 谱不匹配；DoRA 解耦后 **更接近 FFT**。

| 项 | LoRA | DoRA |
| --- | --- | --- |
| 额外参数 | — | + $d_\text{out}$ per layer |
| GLUE/LLaMA | 强 | **+0.5–2 pt 常见** |
| 推理 | merge 可行 | merge 可行 |

PEFT：`use_dora=True` in `LoraConfig`。

### 9.2 全层适配（All-Linear LoRA）

对 **所有 Linear**（Attn + MLP，有时含 lm_head）注入 LoRA：

- 参数量 ↑（仍 << FFT）
- **难任务、小数据** 收益大
- modelforge Embedding stage2 等 **表示学习** 任务常需 MLP 参与

### 9.3 更高 rank

- **r=64, 128**：接近 FFT 的子空间；配合 QLoRA 仍可控
- 与 **VeRA r=256** 不同：VeRA 是 **共享基**，LoRA 是 **独立 A,B**

### 9.4 何时上 DoRA / 全层

```text
LoRA 默认配方 + LoRA+ + rsLoRA 仍明显欠拟合？
  ├─ 是 → 先 all-linear + r↑
  │       仍不足 → DoRA
  └─ 否 → 保持省参配置
```

---

## 10. 工程生态：HF PEFT、ms-swift、Axolotl、vLLM / TGI

### 10.1 HuggingFace PEFT

| 能力 | API / 说明 |
| --- | --- |
| LoRA / AdaLoRA / VeRA / DoRA | `LoraConfig`, `AdaLoraConfig`, `VeraConfig` |
| QLoRA | `prepare_model_for_kbit_training` + bnb |
| Merge | `merge_and_unload()` |
| 多 adapter | `load_adapter`, `set_adapter`, `add_weighted_adapter` |

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

config = LoraConfig(
    r=16, lora_alpha=16, target_modules="all-linear",
    use_rslora=True, use_dora=False,
)
model = get_peft_model(model, config)
```

### 10.2 ms-swift（vlm_train 主栈）

- 统一 CLI：`swift sft` / `swift rlhf`
- 常见 flags：`--tuner_type lora`, `--lora_rank`, `--lora_alpha`, `--target_modules`
- 多模态：VLM 项目（Qwen-VL 等）同样走 LoRA
- **建议**：确认版本是否支持 **LoRA+ / rsLoRA / DoRA**；若无则 param group patch

**本仓脚本示例**（`vlm_train/scripts/platform/train_cloud_reid_0.8b_2gpu.sh`）：

```bash
--tuner_type lora \
--lora_rank 8 \
--lora_alpha 32 \
```

可在此基础上加 **λ=16** 与 **rsLoRA** 做 A/B。

### 10.3 Axolotl / LLaMA-Factory

- YAML 驱动：`adapter: lora`, `lora_r`, `load_in_4bit`
- 社区 recipe 丰富（Alpaca、ShareGPT）
- 适合 **快速复现论文** 与 **多数据集编排**

### 10.4 vLLM Multi-LoRA Serving

- **PagedAttention** + **动态 LoRA 加载**
- 同一 batch **不同 request 不同 adapter**
- API：`enable_lora=True`, `max_loras`, `max_lora_rank`
- 权重：**base 一份 + N 个 LoRA delta**

```text
Client A ──adapter_1──┐
Client B ──adapter_2──┼──► vLLM batch ──► GPU
Client C ──adapter_1──┘
```

### 10.5 TGI / SGLang

- **Text Generation Inference**：支持 adapter 热加载
- **SGLang**：RadixAttention + LoRA batching，高并发场景

### 10.6 部署路径选择

| 场景 | 路径 |
| --- | --- |
| 单租户、最低延迟 | **merge_lora** → 标准推理 |
| 多租户 SaaS | **vLLM multi-LoRA** |
| 端侧 | merge + GPTQ/AWQ |
| 训练平台 | ms-swift 产出 checkpoint → 注册 adapter 元数据 |

### 10.7 checkpoint 管理

```text
checkpoint/
  adapter_config.json   # r, alpha, target, peft_type
  adapter_model.safetensors
  README.md
```

VeRA 额外依赖 **seed** 重建共享矩阵；merge 后仅留 `model.safetensors`。

### 10.8 VLM / 多模态 LoRA 要点

VLM（Qwen-VL、LLaVA 等）LoRA 注入点除 LLM 侧 linear 外，还包括：

| 模块 | 是否常训 | 说明 |
| --- | --- | --- |
| LLM q/k/v/o + MLP | **是** | 与纯文本相同 |
| Vision projector | **视任务** | 图文检索常训 |
| ViT backbone | 少训 | 数据少时易过拟合；可 freeze ViT 只训 projector+LLM |
| Embedding 层 | 少训 | 词表扩展场景除外 |

ms-swift 对 VLM 统一 `--tuner_type lora`；rank 可 **LLM 侧 r=16、projector r=32** 分组（若框架支持多 config）。多模态 SFT 数据格式（`<image>` token 位置）错误比 LoRA rank 更常导致训练失败。

### 10.9 ms-swift LoRA 参数速查（vlm_train）

| CLI 参数 | 典型值 | 说明 |
| --- | --- | --- |
| `--tuner_type` | `lora` | PEFT 类型 |
| `--lora_rank` | 8–64 | 云侧 reid 脚本现用 8 |
| `--lora_alpha` | 16–32 | 与 r 同阶或 2× |
| `--target_modules` | `ALL` / 模块名 | 省参 vs 效果权衡 |
| `--learning_rate` | 1e-4 ~ 2e-4 | LoRA+ 时需拆 A/B 组 |

当前 `train_cloud_reid_0.8b_*.sh` 使用 `r=8, alpha=32`，未启用 LoRA+/rsLoRA；作为 **低成本 A/B** 值得在相同数据上试 `λ=16` 与 `r=16`。

---

## 11. 实践选型指南与超参决策树

### 11.1 决策树（文本）

```text
START: 需要微调 LLM/VLM
│
├─ 显存够 + 数据大 + 要 SOTA？
│   └─ YES → 考虑 FFT 或 DoRA + all-linear + 高 rank
│
├─ 单卡 24–48GB，模型 ≥7B？
│   └─ YES → QLoRA + LoRA (r=16~64)
│
├─ 多租户 / adapter 数 >100？
│   └─ YES → VeRA 或 LoRA + vLLM multi-LoRA
│
├─ 任务类型？
│   ├─ NLU / 分类 → LoRA q,v 或 all-linear, r=8~32
│   ├─ SFT / 对话 → all-linear, r=16~64, LoRA+
│   ├─ Embedding / 检索 → all-linear, r=32~128, 注意 instruct 对齐
│   └─ 扩散 / 视觉 → UNet attention + conv, r=4~32
│
└─ 默认配方 → §1.2
```

### 11.2 超参表（起点）

| 超参 | 7B SFT | 7B QLoRA | Embedding 4B |
| --- | --- | --- | --- |
| r | 16 | 16–64 | 32–64 |
| alpha | 16 | 16 | 32–64 |
| rsLoRA | on | on | on |
| LoRA+ λ | 16 | 16 | 16 |
| lr (ηA) | 1e-4 | 2e-4 | 5e-5 ~ 1e-4 |
| epochs | 1–3 | 1–3 | 1–5 |
| batch | 最大能塞 | 同上 | 对比学习需大 batch |
| target | all-linear | all-linear | all-linear |
| warmup | 3% | 3% | 3% |

### 11.3 rank 选择

| r | 参数量 | 适用 |
| --- | --- | --- |
| 4–8 | 最小 | 简单 NLU、资源极紧 |
| 16–32 | **甜点** | 多数 SFT |
| 64–128 | 较大 | Embedding、难推理 |
| 256+ | VeRA 常用 | 共享基场景 |

### 11.4 LoRA vs QLoRA vs DoRA vs 全参

| 你的情况 | 选择 |
| --- | --- |
| 第一次微调、省显存 | **QLoRA + LoRA+ + rsLoRA** |
| 单卡 80GB、要最好 | **bf16 LoRA all-linear r=64** 或 **DoRA** |
| 评测接近 FFT 仍差 2pt+ | **DoRA** 或 **FFT last layers** |
| 1000 个客户各一 adapter | **VeRA** + multi-LoRA serving |
| Embedding 裸训 vs Instruct 错配 | **先对齐数据格式**，再调 rank（见 §13） |

### 11.5 常见失败模式

| 症状 | 可能原因 | 修复 |
| --- | --- | --- |
| loss 不降 | lr 过小 / 只训了 embedding | LoRA+、扩大 target |
| 很快过拟合 | r 太大、数据太少 | dropout、减 r、早停 |
| 训评不一致 | instruct 模板不一致 | 统一 query 格式 |
| 高 rank 爆炸 | 未用 rsLoRA | 开 rsLoRA |
| merge 后变差 | merge 精度 / 顺序错 | bf16 merge、测 merge 前后 |

### 11.6 阶段化微调策略（SFT → DPO → 部署）

许多生产流水线分阶段使用 LoRA，而非一次 FFT：

```text
Stage 0: 基座 (frozen, bf16 或 4bit)
Stage 1: SFT LoRA (r=16, all-linear, LoRA++rsLoRA)
Stage 2: 可选 DPO LoRA (同 adapter 继续训或新 adapter)
Stage 3: merge + 量化  OR  vLLM multi-LoRA 热加载
```

| 阶段 | 建议 |
| --- | --- |
| SFT | 数据质量优先；epoch 1–3 防过拟合 |
| DPO | 保持 reference 与 policy **同基座**；LoRA rank 不足时先 r↑ 再 DPO |
| RLHF (PPO) | 几乎总是 QLoRA；reward model 可单独 LoRA |
| 部署 | 单租户 merge；多租户保留 adapter + vLLM |

Embedding 任务（modelforge cloud_emb）类似：**stage1 检索 warm-up → stage2 全任务 LoRA**，每阶段独立 checkpoint 与 `runs_registry.json` 登记，便于 COMPARISON_REPORT 横向对比。

---

## 12. 开放问题与前沿

### 12.1 初始化：PiSSA、LoRA-GA

- **PiSSA**：用 $W_0$ SVD 主成分初始化 $A,B$，**更快收敛**
- **LoRA-GA**：按第一步梯度对齐初始化
- 与 LoRA+ **互补**；PEFT 部分版本已支持 `init_lora_weights="pissa"`

### 12.2 梯度低秩：GaLore

- **GaLore**：对 **梯度** 做低秩投影，**全参训练** 但 optimizer 状态低秩
- 非 LoRA，但解决同一「省显存微调」问题；与 QLoRA 可对比选型

### 12.3 Multi-Adapter Routing

- **MoE 式 adapter**：按 token / 样本路由不同 LoRA
- **LoRAHub / PEFT composition**：加权组合多个 LoRA 无再训练
- 开放：**自动路由训练**、adapter 间干扰

### 12.4 RLHF / DPO + LoRA

- 对齐阶段 **几乎默认 LoRA/QLoRA**（省显存、防遗忘）
- **问题**：reward hacking 时 LoRA 容量不足 → 增大 r 或 DoRA
- **DPO**：偏好对数据 + LoRA 稳定；注意 **reference model** 与 policy 同基座

### 12.5 理论

- LoRA 与 FFT 的 **泛化界**
- 最优 rank 与 **内在维度** 的数据依赖
- **μP** 尺度下 LoRA+ 的 $\lambda$ 理论扩展

### 12.6 安全与合规

- LoRA **可能被提取 / 合并** 泄露微调知识
- 多租户 **adapter 隔离**（vLLM 内存隔离仍待审计）

---

## 13. 对本仓库训练实践的启示

### 13.1 vlm_train

| 现状 | 建议 |
| --- | --- |
| Qwen3.5 云侧脚本 `--tuner_type lora`, r=8, alpha=32 | 试 **LoRA+ λ=16**、**rsLoRA**；难任务 **r=16~32** |
| merge LoRA 后量化部署（`board_qwen35_2b_autoround_3die.sh`） | 保持 **merge 前 eval**；merge 与 QLoRA 训练图一致 |
| 多卡 DeepSpeed | LoRA param group 与 ZeRO 兼容；注意 checkpoint 只存 adapter |

### 13.2 modelforge cloud_emb

`tasks/cloud_emb/eval/COMPARISON_REPORT.md` 显示：

- **Base + Instruct** 常强于 **裸训 LoRA + Instruct 评**（分布错配）
- **LoRA Instruct 对齐**、**true_neg** 数据后可 **接近或略超** Base（MRR）
- **启示**：
  1. Embedding LoRA 优先 **all-linear + r≥32**
  2. 训练/评测 **Instruct 模板必须一致**
  3. 难负例质量 > 盲目加 rank
  4. 可试验 **LoRA+ / DoRA** 在 stage2 checkpoint 上是否进一步缩 gap

### 13.3 与 Embedding 调研的交叉

- [Embedding调研报告](../embedding/Embedding调研报告.md) 指出 LLM 骨干 + LoRA 是主流路线之一
- Conan-v2 等 **从零预训练** 路线攻击的是 LoRA **表达力上限**；工程上仍应先把 **LoRA 配方调满** 再考虑 FFT/自训底座

### 13.4 与蒸馏的交叉

- LoRA 训 **student** + 全参 **teacher** 蒸馏见 [知识蒸馏技术深度调研报告](../distillation/知识蒸馏技术深度调研报告.md)
- Jasper 等压缩路线与 LoRA **正交**（训完再压 vs 训时 PEFT）

---

## 14. 参考文献

1. Hu et al. **LoRA: Low-Rank Adaptation of Large Language Models**. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
2. Houlsby et al. **Parameter-Efficient Transfer Learning for NLP**. ICML 2019. [arXiv:1902.00751](https://arxiv.org/abs/1902.00751)
3. Li & Liang. **Prefix-Tuning: Optimizing Continuous Prompts for Generation**. ACL 2021. [arXiv:2101.00190](https://arxiv.org/abs/2101.00190)
4. Zhang et al. **Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning (AdaLoRA)**. ICLR 2023. [arXiv:2303.10512](https://arxiv.org/abs/2303.10512)
5. Dettmers et al. **QLoRA: Efficient Finetuning of Quantized LLMs**. NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
6. Kopiczko et al. **VeRA: Vector-based Random Matrix Adaptation**. ICLR 2024. [arXiv:2310.11454](https://arxiv.org/abs/2310.11454)
7. Kalajdzievski. **A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (rsLoRA)**. ICLR 2024. [arXiv:2312.03732](https://arxiv.org/abs/2312.03732)
8. Hayou et al. **LoRA+: Efficient Low Rank Adaptation of Large Models**. ICML 2024. [arXiv:2402.12354](https://arxiv.org/abs/2402.12354)
9. Liu et al. **DoRA: Weight-Decomposed Low-Rank Adaptation**. ICML 2024. [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
10. Meng et al. **PiSSA: Principal Singular Values and Singular Vectors Adaptation**. NeurIPS 2024. [arXiv:2404.02948](https://arxiv.org/abs/2404.02948)
11. Wang et al. **LoRA-GA: Low-Rank Adaptation with Gradient Approximation**. NeurIPS 2024. [arXiv:2407.05000](https://arxiv.org/abs/2407.05000)
12. Zhao et al. **GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection**. ICML 2024. [arXiv:2403.03573](https://arxiv.org/abs/2403.03573)
13. Zaken et al. **BitFit: Simple Parameter-efficient Fine-tuning**. ACL 2022. [arXiv:2106.10199](https://arxiv.org/abs/2106.10199)
14. Liu et al. **Few-Shot Parameter-Efficient Fine-Tuning (IA³)**. NeurIPS 2022. [arXiv:2205.05638](https://arxiv.org/abs/2205.05638)
15. HuggingFace **PEFT Documentation**. https://huggingface.co/docs/peft
16. vLLM **LoRA Serving**. https://docs.vllm.ai/en/latest/models/lora.html

---

## 附录 A：术语表

| 术语 | 含义 |
| --- | --- |
| **PEFT** | Parameter-Efficient Fine-Tuning |
| **FFT** | Full Fine-Tuning，全量微调 |
| **LoRA** | 低秩适配 $W=W_0+BA$ |
| **QLoRA** | 4-bit 量化基座 + LoRA |
| **NF4** | Normal Float 4-bit 量化 |
| **rsLoRA** | $\alpha/\sqrt{r}$ 缩放 |
| **LoRA+** | A/B 异学习率 |
| **DoRA** | 幅度-方向分解 LoRA |
| **VeRA** | 共享随机基 + 向量缩放 |
| **AdaLoRA** | 自适应 rank 分配 |
| **merge** | 将 BA 并入 $W_0$ 推理 |

---

## 附录 B：参数量估算公式

对 hidden size $d$，层数 $L$，rank $r$，注入 **4 个 attention linear + 3 个 MLP linear**（LLaMA 风格，每层 7 矩阵）：

$$
|\Delta\theta|_\text{LoRA} \approx L \times 7 \times r \times 2d = 14 L r d
$$

例：$L=32, d=4096, r=8$ → $14 \times 32 \times 8 \times 4096 \approx 14.7\text{M}$。

VeRA 单层约 $2d$，共享 $A_0,B_0$ 不计入每租户存储。

---

> **专题深读**: [VeRA详解.md](VeRA详解.md) · [LoRA+详解.md](LoRA+详解.md)  
> **资料索引**: [资料清单_论文与博客.md](资料清单_论文与博客.md)
