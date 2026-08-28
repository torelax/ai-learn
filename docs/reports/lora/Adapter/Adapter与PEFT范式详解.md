# Adapter 与 PEFT 范式详解

> 基于 Houlsby et al. *Parameter-Efficient Transfer Learning for NLP*（[arXiv:1902.00751](https://arxiv.org/abs/1902.00751)，ICML 2019）与 HuggingFace [PEFT](https://github.com/huggingface/peft) 生态。
> 本文按 Houlsby 论文章节逐节拆解 Adapter 原论文，并上升到 **PEFT 三大方法族** 与 **HuggingFace PEFT 工程实践**，便于对照实现与选型。

---

## 1. 一句话定位

**Adapter** 是 PEFT 的奠基工作之一：在 **冻结预训练权重** 的前提下，向 Transformer 每层 **串行注入 bottleneck 模块**（$m \ll d$），**每层 2 个 Adapter**，**近恒等初始化**，并 **同时训练 LayerNorm**；每任务仅训 **≈3.6%** 参数即可在 GLUE 上逼近全量微调（**80.0 vs 80.4，差距 0.4 分**）。它开创了「**共享底座 + 任务增量模块**」工业范式，与 Prompt Tuning、LoRA 重参数化并列为 PEFT 三大族。

| 项 | 内容 |
|----|------|
| 论文 | Houlsby et al., ICML 2019, [1902.00751](https://arxiv.org/abs/1902.00751) |
| 核心结构 | Bottleneck $f(x)=W_{\mathrm{up}}\sigma(W_{\mathrm{down}}x)$，$m \ll d$ |
| 插入 | **2 adapters / Transformer layer**（Attn 后 + FFN 后） |
| 初始化 | $W_{\mathrm{up}}, W_{\mathrm{down}} \approx 0$ → **near-identity** |
| GLUE | **3.6% params/task**，均分 **80.0**（FT **80.4**） |
| 额外可训 | **LayerNorm** scale/shift（per-task） |
| 推理 | **增加延迟**（多过 bottleneck 层） |

---

## 2. 论文目录与阅读路线

```text
§1  Introduction           多任务参数爆炸；Adapter 命题；Figure 1 trade-off
§2  Adapter tuning for NLP bottleneck 设计；Figure 2 Transformer 插入
§3  Experiments            GLUE / 17 任务 / SQuAD；参数–性能曲线
§4  Related Work           视觉 Adapter 前身；NLP 迁移学习谱系
Appendix                   更多架构变体消融（复杂设计不如简单 bottleneck）
```

| 若你的目标是… | 优先章节 |
| --- | --- |
| 理解 PEFT 历史起点 | §1 + §2 |
| 复现 Houlsby 基线 | §2.2–2.3 + §3.1 超参 |
| 对比 LoRA / Adapter 选型 | §3.4 + 本文 §7 |
| 接 HuggingFace PEFT | 本文 §8 |

---

## 3. §1 Introduction（引言）

### 3.1 问题：多任务场景下的参数爆炸

NLP 迁移学习主流有两条路：

| 路线 | 做法 | 痛点 |
| --- | --- | --- |
| **Feature-based transfer** | 固定预训练 embedding，下游另训分类头 | 每任务仍需独立下游模型 |
| **Fine-tuning (FT)** | 复制预训练权重并全参更新 | 每任务 ≈ **100%** 参数量；N 任务 ≈ **N×** 底座大小 |

云服务、多租户场景下，任务 **顺序到达**、需 **增量扩展**、且希望 **旧任务不被覆盖**——全量 FT 既不 compact 也不 extensible。

### 3.2 Adapter 的核心命题

定义原网络 $\phi_w(x)$，Adapter 引入新函数 $\psi_{w,v}(x)$：

- $w$：从预训练 **复制且冻结**
- $v$：任务专属、可训练
- 初始化 $v_0$ 使 $\psi_{w,v_0}(x) \approx \phi_w(x)$（**近恒等**）

若 $|v| \ll |w|$，则 N 任务总参数 ≈ $|w| + N|v|$，而非 $N|w|$。

### 3.3 与 Multi-task / Continual Learning 的关系

| 对比项 | Multi-task | Continual | Adapter |
| --- | --- | --- | --- |
| 任务可见性 | 需同时访问全部任务 | 顺序学习，易遗忘 | 顺序即可，**无遗忘**（$w$ 冻结） |
| 任务间干扰 | 共享梯度，可能负迁移 | 典型灾难性遗忘 | 任务参数隔离 |
| 存储 | 一份共享 + 多 head | 一份权重被覆盖 | **1 份 $w$ + N 份 $v$** |

Figure 1 给出关键 trade-off：**Adapter 用比 FT 少两个数量级的可训练参数，达到与全量 FT 相近的精度曲线**（GLUE 九任务 20/50/80 分位）。

### 3.4 本文贡献（作者自陈）

1. 为 NLP 设计 **有效 bottleneck Adapter** 及其与 Transformer 的集成方式
2. GLUE 上 **3% 量级** 任务参数即可逼近 BERT 全量 FT
3. 额外 **17 个分类数据集 + SQuAD QA** 验证泛化

---

## 4. §2 Adapter tuning for NLP

### 4.1 Adapter 模块的两条设计原则

| 原则 | 含义 | 工程后果 |
| --- | --- | --- |
| **Bottleneck（$m \ll d$）** | 中间维 $m$ 远小于隐层维 $d$ | 每任务参数增量可控 |
| **Near-identity init** | 初始化使模块 ≈ 恒等映射 | 训练稳定；偏离过远会 **无法收敛** |

Adapter 是 **通用架构修改**（插入新层），而非仅换顶层分类头；原网络权重 **完全冻结**，多任务共享。

### 4.2 Bottleneck 结构与前向

设输入特征 $x \in \mathbb{R}^d$，Adapter 前向：

$$
x' = x + f(x), \quad f(x) = W_{\mathrm{up}}\,\sigma(W_{\mathrm{down}}\,x)
$$

- $W_{\mathrm{down}} \in \mathbb{R}^{m \times d}$，$W_{\mathrm{up}} \in \mathbb{R}^{d \times m}$，**$m \ll d$**
- **内部 skip**：$x' = x + f(x)$，与外层 Transformer residual 不同
- 参数量（含 bias）：**$2md + d + m$** 每层 Adapter

**Near-identity 实现**：将 $W_{\mathrm{up}}$、$W_{\mathrm{down}}$ 初始化接近 **零**，使 $f(x) \approx 0$，训练初期 $\psi \approx \phi$。

**非线性 $\sigma$**：原文用 ReLU；后续 Parallel Adapter（He et al. 2021）等工作亦沿用 bottleneck + 激活。

### 4.3 Transformer 中的插入位置（Figure 2）

**每个 Transformer layer 插入 2 个 Adapter**（**串行**、非并行）：

```text
Multi-Head Attention → proj → [Adapter 1] → LayerNorm
         ↓ skip
Feed-Forward (2层)    →       [Adapter 2] → LayerNorm
         ↓ skip
```

要点：

- 插在 **子层输出、投影回 $d$ 维之后**，**LayerNorm 之前**
- **在加子层 skip-connection 之前** 作用于子层输出
- 共 **2 adapters / layer × L layers**；BERT_BASE 12 层 → **24 个 Adapter / 任务**

**为何两处都插？** 消融表明 Attn 与 FFN 路径对下游任务贡献不同；双点插入在 GLUE 上优于单点。

### 4.4 LayerNorm 也训练

除 Adapter 外，**每层 LayerNorm 的 scale/shift 参数 per-task 可训练**（类似 conditional BN / FiLM）。

| 只训什么 | 效果 |
| --- | --- |
| 仅 LayerNorm（~40k 参数 / BERT_BASE） | **不够**：CoLA 掉 ~3.5%，MNLI 掉 ~4% |
| Adapter + LayerNorm | 达到 FT 级性能 |

LayerNorm 提供 **轻量级任务条件仿射变换**，与 bottleneck 互补；复现时 **勿漏训 LayerNorm**。

### 4.5 与 Rebuffi et al. 2017 视觉 Adapter 的继承

Houlsby 将 Rebuffi 等在 ResNet 上提出的 Adapter 思想 **首次系统化迁移到 NLP Transformer**，并针对文本任务做了 bottleneck 与插入点实验。

§3.6 / Appendix 还对比了 **更复杂设计**（卷积 Adapter、并行结构等），结论：**简单 bottleneck 最优**——与 Karpathy「简单优先」一致。

---

## 5. §3 Experiments（实验）

### 5.1 实验设置

| 项 | 配置 |
| --- | --- |
| 底座 | 公开 **BERT**（GLUE 用 BERT_LARGE 24L/330M；额外任务用 BERT_BASE 12L） |
| 分类头 | [CLS] token embedding + 线性层（同 Devlin et al. 2018） |
| 优化 | Adam；前 10% steps 线性 warmup，之后线性衰减至 0 |
| 硬件 | 4× Google Cloud TPU，batch=32 |
| 稳定性 | **5 个随机种子**，取验证集最优 |

Adapter 专属超参：**bottleneck 大小 $m \in \{8,64,256\}$**（GLUE 按任务选最优）；学习率 $\in \{3\times10^{-5}, 3\times10^{-4}, 3\times10^{-3}\}$；epochs $\in \{3, 20\}$。

### 5.2 GLUE 主结果（Table 1）

| 方法 | 总参数量（相对 BERT_LARGE） | 每任务可训练参数 | GLUE 均分 |
| --- | --- | --- | --- |
| BERT_LARGE FT | **9.0×** | 100% | **80.4** |
| Adapters ($m$=8–256) | **1.3×** | **3.6%** | **80.0** |
| Adapters (固定 $m$=64) | 1.2× | 2.1% | 79.6 |

- 与 FT 差距：**0.4 分**（within **0.5%** 相对精度）
- 小数据集 RTE 偏好 $m$=8；MNLI 偏好 $m$=256——**任务规模决定最优 bottleneck**
- 解决 Table 1 全部 9 个 GLUE 任务：FT 需 **9×** 底座参数，Adapter 仅需 **1.3×**

**单任务解读**：Adapter 在 **大数据集**（MNLI、QQP）与 FT 几乎无差；**小数据集**（RTE、MRPC）略逊，可通过减小 $m$ 缓解过拟合。

### 5.3 额外 17 个分类任务（Table 2）

| 方法 | 平均准确率 | 总参数 / BERT_BASE | 每任务可训练 |
| --- | --- | --- | --- |
| AutoML 强基线 | 72.7 | — | — |
| BERT FT | 73.7 | 17× | 100% |
| Variable FT（仅训顶层 $n$ 层） | 74.0 | 9.9× | 52.9% |
| **Adapters** | 73.3 | **1.19×** | **1.14%** |

Adapter 比 FT 低 **0.4%**，但参数量 **两个数量级** 更少；Variable FT 虽只训约一半层，仍远不及 Adapter compact。

### 5.4 参数–性能 trade-off（§3.4，Figure 3–4）

对比三种策略：

1. **Fine-tune top-$k$ layers**
2. **仅 LayerNorm**
3. **Adapter（不同 $m$）**

结论：

- GLUE 上少层 FT **性能断崖**；Adapter 在 **0.5%–5%** 原模型参数量区间内稳定
- MNLI 例：FT 仅顶层 ≈ **9M** 可训练参数 → **77.8%**；Adapter $m$=64 ≈ **2M** 参数 → **83.7%**（FT **84.4%**）
- Adapter size 跨 **数个数量级**（8→256）性能仍稳定（86.2%→85.8%→86.7% 量级）

**设计启示**：PEFT 的价值不仅在「少参数」，更在 **参数–性能曲线的平坦区间**——Adapter 在该区间比 top-layer FT 宽得多。

### 5.5 SQuAD 与层级分析（§3.5–3.6）

- **SQuAD v1.1**：Adapter 逼近 FT F1/EM
- **层级敏感度**：Adapter 自动把更大更新集中在 **高层**；低层 Adapter 影响可忽略——与「高层更 task-specific」直觉一致
- **复杂 Adapter 变体**（卷积、并行等）未超过简单 bottleneck

---

## 6. §4 Related Work（相关工作）

### 6.1 视觉与结构化 Adapter 前身

- **Rebuffi et al. 2017**：ResNet 上 per-layer Adapter，多任务视觉识别；Houlsby 直接继承 bottleneck 思想
- **Pfeiffer et al. 2020+**：AdapterHub、多语言 Adapter 生态；模块化热插拔

### 6.2 NLP 中的其他参数高效路线（2019 时点）

| 方法 | 思路 | 与 Adapter 差异 |
| --- | --- | --- |
| **FT top layers** | 只更新顶层 | 参数仍多、低层 FT 掉点严重 |
| **Feature extraction** | 固定 encoder | 表达力受限于浅层头 |
| **Multi-task learning** | 共享全部参数联合训 | 需同时见全任务 |

### 6.3 后续演进（读论文时的「后视」）

Houlsby 2019 之后 PEFT 爆发：

| 年份 | 工作 | 相对 Houlsby |
| --- | --- | --- |
| 2021 | Parallel Adapter（He et al.） | 并行插入，降低延迟 |
| 2021 | Prefix / Prompt Tuning | 改输入不改权重 |
| 2022 | LoRA（Hu et al.） | 低秩增量，**可 merge、零推理开销** |
| 2022 | IA³ | 逐通道缩放 |
| 2023 | AdaLoRA、QLoRA | 动态 rank / 量化底座 |
| 2024 | DoRA、VeRA | 幅度–方向分解 / 共享随机基 |

---

## 7. PEFT 作为范式：三大方法族

现代文献（含 DoRA 论文 §2 分类）常将 PEFT 分为：

```text
PEFT
├── 族 A：Adapter-based     — 插入新模块（Houlsby 串行、He 并行、Compacter…）
├── 族 B：Prompt-based       — 软 prompt / prefix（Lester、Li & Liang、P-Tuning）
└── 族 C：Reparametrization  — 重参数化增量（LoRA、AdaLoRA、DoRA、rsLoRA、VeRA…）
```

### 7.1 三族对比

| 维度 | Adapter | Prompt | Reparametrization (LoRA 系) |
| --- | --- | --- | --- |
| **改架构？** | 是（加层） | 否（加 token） | 否（$W'=W_0+\Delta W$） |
| **推理延迟** | **增加**（多过一层） | 增加（序列变长） | **可 merge，零增量** |
| **多任务切换** | 换 Adapter 模块 | 换 prompt 向量 | 换 LoRA 权重或 merge |
| **典型参数量** | ~0.5%–8% / task | ~0.01%–1% | ~0.01%–1% |
| **代表库支持** | `Bottleneck` 系 | `PrefixTuning`, `PromptTuning` | `LoraConfig`, `AdaLoraConfig`, `DoraConfig` |

### 7.2 选型直觉

- **Serving 延迟敏感、要 merge 部署** → LoRA / DoRA / QLoRA
- **多任务热切换、模块可插拔** → Adapter 或独立 LoRA adapter 权重
- **极少参数、可接受 prompt 长度** → Prompt / Prefix Tuning
- **极低预算 + 要分配 rank** → AdaLoRA
- **复现 Houlsby 基线** → BERT + 2 adapters/layer + **训 LayerNorm** + bottleneck sweep

---

## 8. HuggingFace PEFT 库生态

[PEFT](https://github.com/huggingface/peft) 是 HuggingFace 对 PEFT 方法的 **统一封装**，与 `transformers` Trainer、Accelerate、bitsandbytes 深度集成。

### 8.1 核心 API 流程

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base, config)
model.print_trainable_parameters()
# trainable params: ~4M || all params: ~7B || trainable%: ~0.06%
```

| API / 类 | 作用 |
| --- | --- |
| **`PeftConfig` 子类** | `LoraConfig`, `AdaLoraConfig`, `DoraConfig`, `PromptTuningConfig`, `IA3Config`… |
| **`get_peft_model(base, config)`** | 包装底座，注入 PEFT 层，**冻结非 PEFT 参数** |
| **`PeftModel`** | 统一 forward；支持 `save_pretrained` / `load_adapter` |
| **`merge_and_unload()`** | 将 LoRA/DoRA 增量 **合并进基权重**，恢复普通模型 |
| **`PeftModel.from_pretrained`** | 加载已训 adapter 权重 |

### 8.2 常见 `PeftType` 与方法

| PeftType | 论文 | 要点 |
| --- | --- | --- |
| `LORA` | Hu et al. 2022 | $W'=W_0+BA$，$r \ll d$ |
| `ADALORA` | Zhang et al. 2023 | SVD 参数化 + 动态 rank 剪枝 |
| `DORA` | Liu et al. 2024 | 幅度–方向分解，方向用 LoRA |
| `IA3` | Liu et al. 2022 | 学习逐通道缩放向量 |
| `PREFIX_TUNING` / `P_TUNING` / `PROMPT_TUNING` | Li & Liang; Lester | 软 prefix / prompt |
| `OFT` / `BOFT` | 正交微调变体 | 结构化旋转 |

### 8.3 LoRA 系变体在 PEFT 中的关系

```text
                    ┌─ LoRA (标准低秩 BA)
                    │
PEFT Reparam ───────┼─ AdaLoRA (动态 rank，SVD 三元组剪枝)
                    │
                    ├─ DoRA (m + LoRA方向，W'=m·(W₀+BA)/||·||)
                    │     └─ QDoRA = DoRA + 4bit 量化底座
                    │
                    ├─ rsLoRA (rank-stabilized scaling: α/√r)
                    │
                    └─ QLoRA = LoRA + NF4 量化 + Double Quant
                          （bitsandbytes；非单独 PeftType，而是加载 4bit 底座 + LoraConfig）
```

| 变体 | 与 Adapter 关系 | 与 QLoRA 关系 | 与 DoRA / rsLoRA 关系 |
| --- | --- | --- | --- |
| **Adapter (Houlsby)** | 本体 | 正交：Adapter 改架构，QLoRA 量化+LoRA | DoRA 可视为 LoRA 的方向分支增强 |
| **QLoRA** | 不同族 | — | 可与 DoRA 组合为 **QDoRA** |
| **DoRA** | 同属 PEFT，无架构插入 | 量化底座 + DoRA 增量 | rsLoRA 改 scaling；DoRA 改分解方式，**可叠加** |
| **rsLoRA** | — | 常与 QLoRA 联用稳定大 rank | `LoraConfig(use_rslora=True)` |

### 8.4 训练与推理要点

**训练**

- `Trainer` + `peft` 模型：仅 PEFT 参数 `requires_grad=True`
- **QLoRA**：`BitsAndBytesConfig(load_in_4bit=True)` + `prepare_model_for_kbit_training`
- **AdaLoRA**：需 `tinit` / `tfinal` / `total_step` 控制 budget schedule
- **DoRA**：`DoraConfig`；`merge_and_unload()` 后推理同基座

**多 Adapter 热切换**

```python
model.load_adapter("path/to/task_a", adapter_name="task_a")
model.set_adapter("task_a")
```

**与 ms-swift / modelforge 工程**

- 训练 CLI 通常透传 `--lora_rank`, `--lora_alpha`, `--target_modules`
- 部署前 `merge_lora` 消除推理分支（DoRA 同理）

---

## 9. 公式速查

**Houlsby Bottleneck Adapter（单层）**

$$
x' = x + W_{\mathrm{up}}\,\sigma(W_{\mathrm{down}}\,x), \quad W_{\mathrm{down}} \in \mathbb{R}^{m \times d},\; W_{\mathrm{up}} \in \mathbb{R}^{d \times m},\; m \ll d
$$

**LoRA（对照，族 C）**

$$
W' = W_0 + BA, \quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d,k)
$$

**PEFT 共同目标**

$$
\min_{\theta_{\mathrm{peft}}} \mathcal{L}\big(f(x;\, w_{\mathrm{frozen}}, \theta_{\mathrm{peft}})\big), \quad |\theta_{\mathrm{peft}}| \ll |w|
$$

---

## 10. 实践清单

| 场景 | 建议 |
| --- | --- |
| 多任务 SaaS、频繁增任务 | Adapter 或 **独立 LoRA adapter 文件** 热切换 |
| 单任务生产 merge 部署 | **LoRA / DoRA** + `merge_and_unload` |
| 70B+ 单卡微调 | **QLoRA**（NF4）或 **QDoRA** |
| 参数预算极紧 | **AdaLoRA** 或 LoRA 只训 `q_proj,v_proj` |
| 复现 Houlsby 基线 | BERT + **2 adapters/layer** + **训 LayerNorm** + $m \in \{8,64,256\}$ sweep |
| 延迟敏感在线 serving | **避免串行 Adapter**；优先 LoRA merge |

---

## 11. 结论

Houlsby et al. 2019 首次在 NLP Transformer 上证明：**冻结预训练权重 + bottleneck Adapter（$m \ll d$，2 个/层，近恒等初始化）+ LayerNorm 微调**，可在 GLUE 上以 **3.6% 参数/任务** 达到全量 FT **99.5%** 的性能（80.0 vs 80.4），并把多任务总存储从 **9×** 压到 **1.3×** 底座。

这一工作定义了 PEFT **族 A（Adapter-based）**，与 Prompt（族 B）、LoRA 重参数化（族 C）共同构成现代高效微调选型空间。HuggingFace **PEFT** 库通过 `LoraConfig` / `get_peft_model` 等统一 API，将 LoRA、AdaLoRA、DoRA、IA³、Prompt 等与 **QLoRA（量化）**、**rsLoRA（缩放）** 组合，成为 LLM/VLM 训练工程的事实标准入口。

---

## 参考文献

1. Houlsby et al. *Parameter-Efficient Transfer Learning for NLP*. ICML 2019. [arXiv:1902.00751](https://arxiv.org/abs/1902.00751)
2. Rebuffi et al. *Learning multiple visual domains with residual adapters*. NeurIPS 2017.
3. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
4. He et al. *Towards a Unified View of Parameter-Efficient Transfer Learning*. ICLR 2022（Parallel Adapter）.
5. HuggingFace PEFT: https://github.com/huggingface/peft
6. Dettmers et al. *QLoRA*. NeurIPS 2023.
