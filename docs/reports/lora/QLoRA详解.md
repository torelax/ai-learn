# QLoRA 技术详解

> 基于论文 [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)（Dettmers, Pagnoni, Holtzman, Zettlemoyer；University of Washington；arXiv:2305.14314，2023-05）。
> 本文把 **NF4 量化数据类型、Double Quantization、Paged Optimizers、存储/计算双数据类型机制、「LoRA 必须铺满所有线性层」的关键发现、Guanaco 与 Elo 评测、数据质量 > 数据量** 以及 **附录 A–G 工程细节** 写全，便于在单卡上复现 65B 级微调。

---

## 1. 一句话定位

**QLoRA** 回答的问题是：*能否在一张消费级/单张专业卡上，把 65B 模型微调到与 16-bit 全量微调无差别的水平？* 答案是肯定的——**冻结 4-bit 量化的基座权重，反向传播穿过量化权重去更新 16-bit 的 LoRA 适配器**：

| 项 | 内容 |
| --- | --- |
| 核心机制 | 基座权重存 **NF4（4-bit NormalFloat）**，计算时反量化为 **BF16** 做 matmul；只训练 **BF16 的 LoRA (A,B)** |
| 显存账 | LLaMA-65B 微调：**>780GB → <48GB**（单卡 48GB 可训）；7B 推理部署仅需 **~5GB** |
| 三大组件 | **NF4**、**Double Quantization（DQ）**、**Paged Optimizers** |
| 关键训练发现 | 默认 LoRA 超参（只挂 Q/V）**追不平** 16-bit 全量微调；**LoRA 铺满所有线性层** 才能追平 |
| 代表性产物 | **Guanaco** 家族（7/13/33/65B）：Vicuna 基准上 65B 达 **ChatGPT 的 99.3%**，单卡 24h |
| 数据结论 | **数据质量 ≫ 数据量**：OASST1 9k 样本在 chatbot 评测胜过 FLAN v2 45 万样本 |
| 评测结论 | **MMLU 强 ≠ Vicuna 强**；GPT-4 评判与人评大体一致但有分歧案例；现有 chatbot 基准不可全信 |
| 工程落地 | 开源 CUDA 4-bit 内核，进 **bitsandbytes** 与 Hugging Face transformers 栈 |

QLoRA 的价值不止「省显存」：它把「最大公开模型的微调」从集群工程变为**单机工程**，并顺带做了一轮此前因成本无法开展的大规模指令微调实证研究（>1000 个模型）。

---

## 2. 论文目录与结构导读

```text
§1  Introduction                     问题：65B 全量微调 >780GB；提出 QLoRA 三组件 + Guanaco
§2  Background                       分块 k-bit 量化、LoRA 回顾、PEFT 显存构成的澄清
§3  QLoRA Finetuning                 NF4 / Double Quantization / Paged Optimizers / 形式化定义
§4  QLoRA vs. Standard Finetuning    NF4 vs FP4；4-bit 是否追平 16-bit 全量/LoRA
§5  Pushing the Chatbot SOTA         Guanaco：8 个指令数据集、Elo 锦标赛、GPT-4/人评
§6  Qualitative Analysis             生成样例的成功/失败剖析（lemon-picked）
§7  Related Work                     量化、PEFT、指令微调相关文献
§8  Limitations and Discussion       局限：评测噪声、长序列、3-bit 以下未探明
§9  Broader Impacts                  普惠化 vs 滥用风险
附录 A  §4 实验设置细节（超参、TK-Instruct 设置）
附录 B  Guanaco 实验细节（数据集、超参、消融、数据量 vs 质量）
附录 C  人评协议
附录 D  GPT-4 成对评测细节
附录 E  NormalFloat 4-bit 具体取值表
附录 F  预训练权重正态性的验证
附录 G  显存细账（Memory Footprint 分解）
```

读法建议：**§2 的显存澄清**（LoRA 参数本身不是显存大头）和 **§4 的「铺满所有层」**是实践中最常被忽视的两点；**附录 G** 对排产/选卡最有用。

---

## 3. 逐章节讲解

### 3.1 §1 Introduction：把 780GB 压进 48GB

论文开门见山：16-bit 全量微调 LLaMA-65B 需要 **>780GB** GPU 显存（权重 130GB + 梯度 130GB + Adam 两阶态 520GB + 激活）。已有的量化方法（GPTQ、LLM.int8() 等）只能压缩**推理**显存，训练时会崩。

QLoRA 的解法一句话：**量化基座到 4-bit 并冻结，梯度穿过量化权重流向 LoRA 适配器**。这带来三个此前不敢想的规模：

- 65B 单卡 48GB 可微调（Guanaco-65B，24 小时，Vicuna 达 ChatGPT 的 **99.3%**）；
- 33B 单卡 24GB 消费级显卡 12 小时内达 **97.8%**；
- 顺带微调了 **>1000 个模型**，做了 8 个指令数据集 × 多架构（LLaMA、T5）× 80M–65B 规模的系统研究。

引言还给出一个表 1 的 Elo 排名（GPT-4 评判）：Guanaco-65B（41GB）Elo 1022，仅次于 GPT-4（1348），高于 ChatGPT（966）与 Bard（902）；Guanaco-13B（10GB）已接近 ChatGPT。

> **笔者的边注**：论文同时埋下一条对社区的纠偏——「QLoRA 之后，微调门槛是数据质量而不是显存」。这在 §5/附录 B.4 有系统证据。

### 3.2 §2 Background：三个必须讲清的前置概念

#### 3.2.1 分块 k-bit 量化（Block-wise k-bit Quantization）

朴素 absmax 量化把 FP32 张量按 $c = 127/\text{absmax}(X)$ 缩放后取整到 Int8。问题：**一个 outlier 会撑爆整个 absmax**，导致大量量化 bin 空置、误差集中在小值区。

工程解法：把张量拉平后切成 $n = (b \times h)/B$ 个连续块，每块独立求 $c_i$ 独立量化。块越小越精确，但每块都要存一个量化常数——这就是后面 Double Quantization 要对付的开销。

#### 3.2.2 LoRA 回顾

给定投影 $Y = XW$，$X \in \mathbb{R}^{b \times h}, W \in \mathbb{R}^{h \times o}$，LoRA 计算

$$Y = XW + s \cdot X L_1 L_2, \quad L_1 \in \mathbb{R}^{h \times r},\ L_2 \in \mathbb{R}^{r \times o}$$

基座 $W$ 冻结，梯度经 $W$ 传向 $L_1, L_2$。QLoRA 完全继承这一形式，只是把 $W$ 换成 4-bit 存储。

#### 3.2.3 一个被广泛误解的显存账（本节最重要）

> **PEFT 的显存大头不是适配器参数，而是激活值/输入梯度。**

论文给出的实测（LLaMA-7B，FLAN v2，batch=1，LoRA 参数为常规 0.2% 量级）：

| 构成 | 显存 |
| --- | --- |
| LoRA 参数本身 | **26 MB** |
| LoRA 输入梯度（无 checkpointing） | **567 MB** |
| LoRA 输入梯度（开 gradient checkpointing） | ~18 MB/序列 |
| **4-bit 基座** | **5,048 MB** |

两个推论直接决定了 QLoRA 的设计取向：

1. **再往死里压 LoRA 参数量（更小 r、更少层）收益甚微**——26MB 对 5GB 是噪声；真正的大头是基座存储与激活。
2. **可以放心地把 LoRA 铺满所有层**——参数量翻倍几乎不涨总显存，而对追平 16-bit 性能**至关重要**（§4 图 2）。

这与 VeRA 那种「把适配器参数再砍 10×」的路线形成有趣对照：VeRA 省的是**存储/分发**成本（多租户场景），QLoRA 省的是**训练显存**（单卡场景），两者解决的不是同一个瓶颈。

### 3.3 §3 QLoRA Finetuning：三组件 + 形式化

#### 3.3.0 总原则：双数据类型

QLoRA 有且只有两种精度：

- **存储数据类型**：4-bit（默认 NF4）——只负责「放着省地方」；
- **计算数据类型**：BF16——所有 matmul 都在 16-bit 下进行。

每次用到权重张量，先 **dequantize 到 BF16** 再做矩阵乘。前向、反向都走这条路径，但**只对 LoRA 参数算梯度**，4-bit 权重本身不接收梯度。

#### 3.3.1 4-bit NormalFloat（NF4）

**直觉**：4-bit 一共只有 16 个可用取值，怎么摆这 16 个点损失最小？

- **FP4（浮点）**：按指数/尾数均匀铺——对钟形分布浪费 bin；
- **Int4（均匀整型）**：均匀铺——同样浪费；
- **NF4**：让 16 个点恰好是 **$N(0,1)$ 的等概率分位点**——信息论意义上对**零均值正态数据**最优（每个 bin 期望落入相同数量的值）。

具体构造（附录 E 有完整取值表）：

1. 估计 $2^k + 1$ 个 $N(0,1)$ 分位数，得到理论最优 k-bit 分位量化类型；
2. 归一化到 $[-1, 1]$；
3. 输入权重按 absmax 归一化到同一区间后按块量化。

麻烦点：对称 k-bit 量化**没有精确的 0**，而 0 对 padding、稀疏结构必须无损表示。解法：**非对称构造**——负半轴取 $2^{k-1}$ 个分位点、正半轴取 $2^k-1+1$ 个，合并后去掉重复的一个 0。这样 16 个码字里有一个精确的 0。

为何可以直接假设权重正态？附录 F 给了实证：预训练神经网络权重普遍近似零均值正态（标准差 $\sigma$ 可被 absmax 缩放吸收）。

#### 3.3.2 Double Quantization（DQ）

块量化省误差，但**量化常数本身要钱**：块大小 64、FP32 常数 → 每参数多 $32/64 = 0.5$ bit。对 65B 模型约 32.5GB——不是小数。

DQ 的做法：**对量化常数再做一次量化**：

- 一级量化：权重 → NF4，常数 $c_2^{FP32}$（块 64）；
- 二级量化：$c_2$ 先减去均值（使分布居中，可用对称量化），再用 **FP8、块 256** 量化，得到 $c_2^{FP8}$ 与二级常数 $c_1^{FP32}$。

平均开销从 0.5 bit/参数降到：

$$8/64 + 32/(64 \times 256) = 0.127 \text{ bit/参数}$$

即**每参数省 0.373 bit**，65B 模型约省 **3GB**。实验证明 FP8 二级量化无性能损失。

#### 3.3.3 Paged Optimizers

微调时最致命的不是稳态显存，而是**长序列 mini-batch 触发的激活尖峰**——梯度 checkpointing 重计算瞬间 OOM。

Paged Optimizers 借 NVIDIA Unified Memory：把优化器状态分配到**分页内存**，GPU 不够时自动换出到 CPU RAM，需要时再换回。等价于给 optimizer state 加了一层「虚拟内存」。论文实测：65B 在 48GB 卡上 batch=16 时，分页优化器与常规优化器**训练速度相同**（分页只在尖峰时刻触发）。

#### 3.3.4 形式化定义

单一线性层 + 单个 LoRA 适配器的 QLoRA：

$$Y^{BF16} = X^{BF16}\,\text{doubleDequant}(c_1^{FP32}, c_2^{k\text{-bit}}, W^{NF4}) + X^{BF16} L_1^{BF16} L_2^{BF16}$$

其中

$$\text{doubleDequant}(c_1, c_2, W) = \text{dequant}(\text{dequant}(c_1, c_2), W^{4bit}) = W^{BF16}$$

超参：$W$ 用 NF4 + 块 64（精度优先），$c_2$ 用 FP8 + 块 256（省存优先）。反向传播需要 $\partial E / \partial L_i$，其链路上要经过 $\partial X / \partial W$，故前向/反向都需把 $W^{NF4}$ 反量化为 $W^{BF16}$——但**梯度只落在 $L_1, L_2$ 上**。

### 3.4 §4 QLoRA vs. Standard Finetuning：三个实验结论

#### 3.4.1 NF4 显著优于 FP4/Int4

在 OPT/BLOOM/Pythia/LLaMA（125M–13B）上做量化后 zero-shot 精度与 Pile Common Crawl 困惑度：

| 数据类型 | Mean PPL（越低越好） |
| --- | --- |
| Int4 | 34.34 |
| Float4 (E2M1) | 31.07 |
| Float4 (E3M0) | 29.48 |
| **NFloat4 + DQ** | **27.41** |

Winogrande/HellaSwag/PiQA/Arc 五项 zero-shot 均值同趋势：NF4 明显领先，DQ 只微调显存不损精度。

#### 3.4.2 默认 LoRA 超参追不平 16-bit 全量微调——铺满所有层才行

这是 §4 对实践者**最重要的一张图**（图 2，LLaMA-7B 在 Alpaca 上）：

- 只把 LoRA 挂在 Q/V（LoRA 论文默认做法）→ **无法复现** 16-bit 全量微调性能；
- **LoRA 铺满所有线性层（attention + FFN 全部）→ 追平**；
- rank $r$ 等其它超参**对结果影响不大**（附录 A）。

同时论文指出：很多「QLoRA/LoRA 不如全量微调」的既往结论，是因为**全量微调基线本身超参欠调**。作者对 lr ∈ [1e-6, 5e-5]、batch ∈ [8, 128] 做了搜索后才建立强基线。

> **实践翻译**：「LoRA 不如全量」八成是适配位置不够或基线没调好，不是 LoRA 表达力不够。这也与 QLoRA §2.3 的显存账自洽——铺满层的参数成本可以忽略。

#### 3.4.3 4-bit QLoRA 追平 16-bit 全量 / 16-bit LoRA

两组实验：

1. **RoBERTa-large / T5-80M…3B**，GLUE 与 Super-NaturalInstructions：BF16 全量、BF16-LoRA、QLoRA-Int8/FP4/NF4+DQ **全部打平**（误差范围内）。量化造成的精度损失可由量化后的适配器微调完全找回。
2. **LLaMA 7B–65B**，Alpaca / FLAN v2 指令微调后 5-shot MMLU：**NF4+DQ 完全恢复 16-bit LoRA 水平**；FP4 版本落后约 1 个点，再次印证 NF4 的精度优势。

§4 小结还抛出一个有趣的资源分配结论：在给定微调+推理预算下，**把参数做大、把精度做低**（更多参数 × 更低 bit）优于反过来——这正是 QLoRA 路线的理论注脚。性能-精度曲线的确切拐点（3-bit 行不行？）留给未来工作。

### 3.5 §5 Pushing the Chatbot SOTA：Guanaco

#### 3.5.1 设置

- 模型：LLaMA 7/13/33/65B；
- 数据：8 个指令数据集——OASST1、HH-RLHF、Alpaca、Self-Instruct、LongForm、FLAN v2、Unnatural Instructions、Chip2（混 HongKongese/Japanese 等）；
- 评测：**Vicuna 基准**（80 条提示）+ MMLU；
- 评判：**锦标赛制**——两两模型对同一提示作答，GPT-4 或人类标注胜负，聚合为 **Elo 分**（表 1：万局随机序平均）。

#### 3.5.2 关键结果

- **Guanaco-65B（OASST1）**：Vicuna Elo 1022，**ChatGPT 的 99.3%**；单张 A100 48GB 训 24 小时；
- **Guanaco-33B**：97.8%，单张 24GB 消费卡 <12 小时；
- **Guanaco-7B**：部署仅需 ~5GB，却在 Vicuna 上**反超 26GB 的 Alpaca 20 多个点**；
- GPT-4 评判与人评在排序上**大体一致**，但存在显著分歧个例 → 模型评判便宜可用，但**不是免审金牌**。

#### 3.5.3 数据质量 ≫ 数据量（附录 B.4 系统证据）

| 数据集 | 规模 | 聊天机器人表现 |
| --- | --- | --- |
| **OASST1** | **9k** | **最强** |
| FLAN v2（子采样） | 450k | 明显更弱 |

以及：**MMLU 表现好 ≠ Vicuna 表现好**，反之亦然——数据与任务的适配性比规模更决定成败。这直接挑战了「指令数据越多越好」的直觉。

#### 3.5.4 对基准的警告

论文明确指出**当时的 chatbot 基准不可信**（Vicuna 80 题、GPT-4 评判方差、数据污染），并用 lemon-picked 失败样例（§6）展示 Guanaco 相对 ChatGPT 的具体短板：保密性、事实准确性、拒答边界等。

### 3.6 §6 Qualitative Analysis：lemon-picked 失败学

§6 不是走过场的样例展示，而是刻意挑**最差案例**（lemon-picked）做归因：

- **事实性错误**：Guanaco 会编造具体数字/日期；
- **指令层级混乱**：对「装作…」类角色指令的边界处理不如 ChatGPT 稳定；
- **安全与拒答**：偶尔该拒不拒、不该拒乱拒；
- **长上下文退化**：多轮对话中早期指令被遗忘。

§6.2 的 Considerations 部分给出冷静评估：4-bit 微调让「谁都能训大模型」，但**评测科学没跟上**——锦标赛 Elo 依赖评判模型的偏见，人评规模小，这既是论文的诚实声明，也是给后续工作的任务清单。

### 3.7 §7 Related Work

三条线：

1. **LLM 量化**：GPTQ、AWQ、LLM.int8()——都面向推理；QLoRA 首次证明 4-bit 下**可训练且不损性能**；
2. **PEFT**：Adapter、Prefix/Prompt Tuning、LoRA、(IA)³——QLoRA 与这些方法**正交**，可叠加；
3. **指令微调**：T0/FLAN 系（多任务规模路线）vs Alpaca/Vicuna 系（蒸馏路线）vs OASST（众包质量路线）——QLoRA 的实证站在了质量路线一边。

### 3.8 §8 Limitations and Discussion

- **未探明区域**：3-bit 及以下是否也能无损微调；分页开销在长序列高频触发时的真实代价；NF4 对非正态分布张量（如某些激活）的适用性；
- **评测噪声**：Elo 对评判模型敏感；Vicuna 覆盖面窄；
- **只验证了 ≤65B**：更大规模（175B+）是否仍无损未验证；
- **结论边界**：QLoRA 恢复的是「任务性能」，不保证恢复所有涌现行为（作者明确提示）。

### 3.9 §9 Broader Impacts

双面：一方面**民主化**——学术实验室、小公司、个人都能在单卡上微调最强开源模型；另一方面同一技术降低了**恶意微调**（去安全对齐、定向造谣言）的成本。作者的立场是开源利大于弊，并强调可复现性（开源全部模型、代码、CUDA 内核、标注数据）。

---

## 4. 附录要点

### 附录 A：§4 实验细节

- QLoRA 默认超参：LoRA r=64、α=16、dropout 0.1、**所有线性层**；lr 2e-4（paged AdamW 32bit）；常数 schedule；batch 16；bf16 计算；
- 再次强调：r 从 8 到 256 变化对 MMLU 影响甚微，**铺满层**才是关键变量。

### 附录 B：Guanaco 细节

- B.1 数据集构成与清洗；B.2 超参（65B 用 r=64，lr 1e-4；33B/13B/7B 类似缩放）；B.3 消融（DQ 开关、Paged 开关、层覆盖）；
- B.4 数据量 vs 质量：固定预算下，9k OASST1 胜过 450k FLAN v2（chatbot 评测），但 MMLU 上 FLAN v2 更好——**评测目标决定数据选型**。

### 附录 C/D：评测协议

- 人评：成对比较、盲评、标注一致性统计；GPT-4 评判：固定模板、随机顺序、温度 0；两者 Elo 相关性高但尾部案例分歧明显。

### 附录 E：NF4 取值表

16 个非对称分位点的精确值（负半轴 7 个 + 0 + 正半轴 8 个），bitsandbytes 的 `NF4` 量化即按此表硬编码。

### 附录 F：权重正态性验证

对 OPT/LLaMA/BLOOM 各层权重做正态性检验（Q-Q 图 / KS 统计），绝大多数线性层权重近似零均值正态，支撑 NF4 的「信息论最优」前提。

### 附录 G：显存细账（排产最有用）

以 LLaMA-65B、QLoRA、梯度 checkpointing 为例的稳态构成（量级）：

| 构成 | 近似显存 |
| --- | --- |
| 4-bit 基座权重 | ~35 GB（含 DQ 常数） |
| LoRA 参数 + 优化器态（32-bit AdamW，r=64 全层） | ~3–5 GB |
| 激活 + 梯度（checkpointed） | 数 GB，随 batch/序列长波动 |
| **合计** | **<48 GB**（Paged Optimizer 兜底尖峰） |

---

## 5. 与 LoRA / rsLoRA / DoRA 的关系

| 维度 | LoRA | QLoRA | rsLoRA | DoRA |
| --- | --- | --- | --- | --- |
| 解决什么 | 参数量/存储 | **训练显存** | 高 rank 梯度塌缩 | 表达力逼近 FT |
| 基座精度 | 16-bit | **4-bit NF4** | 16-bit（可叠加 4-bit） | 16-bit（QDoRA 可叠加） |
| 改什么 | ΔW=BA | 存储/计算分离 + 三组件 | γ=α/√r | W=m·(W0+BA)/‖·‖c |
| 正交性 | — | 可与 rsLoRA/DoRA/LoRA+ 叠加 | 可与 QLoRA 叠加 | QDoRA 论文已验证叠加增益 |

实践组合：**QLoRA（省显存）+ 铺满所有层（追平 FT）+ rsLoRA（高 rank 稳定）+ LoRA+（A/B 异学习率）** 是当前开源社区常见的「四件套」起点。

---

## 6. 实践清单（复现/迁移到本仓库训练）

1. **量化配置**：`bitsandbytes` 4-bit NF4 + `double_quant=True`，compute dtype `bf16`；块 64（权重）/256（二级常数）。
2. **LoRA 配置**：**target 所有线性层**（q/k/v/o + gate/up/down），不要只挂 q/v；r 起步 64；α 经典取 16（或 α=2r，配合 rsLoRA 时按 √r 缩放）。
3. **优化器**：`paged_adamw_32bit`；梯度 checkpointing 必开（尖峰由分页兜底）。
4. **基线纪律**：比较「QLoRA vs 全量」时，先把全量基线的 lr/batch 调强，否则结论不可信（§4 教训）。
5. **数据策略**：指令/偏好数据**质量优先**；先确定评测目标（知识型 MMLU 还是对话型），再选数据配比。
6. **评测纪律**：Elo/GPT-4 评判可用但需人评抽检；报告训练显存时区分稳态与尖峰。

---

## 7. 历史地位与影响

- **工程拐点**：QLoRA 是「单卡微调大模型」时代的开端，bitsandbytes + PEFT + transformers 的组合成为 2023 年后开源微调的事实栈；
- **方法论拐点**：把「量化」从推理技术升格为**训练技术**，启发了 LoftQ（量化感知初始化）、LQ-LoRA、QA-LoRA 等一整支后续工作；
- **实证遗产**：>1000 个模型的系统微调研究，「数据质量 > 数据量」「MMLU ≠ 对话能力」成为后续指令微调论文的标配对照；
- **争议与后续**：Guanaco 的 Vicuna/Elo 结论被后续工作指出基准过窄、评判有偏；3-bit 无损微调至今没有完全解决。

---

## 8. 参考文献

- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. **QLoRA: Efficient Finetuning of Quantized LLMs.** arXiv:2305.14314, 2023. 代码：[artidoro/qlora](https://github.com/artidoro/qlora)、[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- Hu, E. et al. **LoRA: Low-Rank Adaptation of Large Language Models.** arXiv:2106.09685, 2021.
- Dettmers, T. et al. **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.** arXiv:2208.07339.
- Dettmers, T. & Zettlemoyer, L. **The Case for 4-bit Precision: k-bit Inference Scaling Laws.** arXiv:2212.09720.
- Frantar, E. et al. **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.** arXiv:2210.17323.
- Chiang, W.-L. et al. **Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality.** 2023.
- Kalajdzievski, D. **A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (rsLoRA).** arXiv:2312.03732, 2023.
- Liu, S.-Y. et al. **DoRA: Weight-Decomposed Low-Rank Adaptation.** arXiv:2402.09353, ICML 2024.
- Li, Y. et al. **LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models.** arXiv:2310.08659.
