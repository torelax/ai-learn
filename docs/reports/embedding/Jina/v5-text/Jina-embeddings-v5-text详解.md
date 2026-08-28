# Jina-embeddings-v5-text 详解：任务定向蒸馏与 Nano / Small

> 基于论文 [jina-embeddings-v5-text: Task-Targeted Embedding Distillation](https://arxiv.org/abs/2602.15547)（Akram, Sturua, Havriushenko, Herreros, Günther, Werk, Xiao；arXiv:2602.15547）。  
> 本文把 **Qwen3-Embedding 教师蒸馏、学生投影对齐、RoPE $\theta$ 训推分离、四类任务 LoRA、InfoNCE / CoSENT / GOR、nano vs small、消融与量化鲁棒** 写全。

---

## 1. 一句话定位

**jina-embeddings-v5-text** 是一对 **紧凑多语文本 embedding**：先用 **Qwen3-Embedding-4B** 做 **嵌入空间蒸馏**，再 **冻结骨干**、按任务训 **LoRA adapters**（检索 / STS / 聚类 / 分类）。

| 项 | nano | small |
| --- | --- | --- |
| 骨干 | **EuroBERT-210M** | **Qwen3-0.6B-Base** |
| 基座参数 | ~212M | ~596M |
| LoRA | $4\times$ ~6.7M | $4\times$ ~20.2M |
| 嵌入维 | **768** | **1024** |
| 推理最大长 | **8K** | **32K** |
| 推理 RoPE $\theta$ | 1M（表 1） | **3.5M** |
| 教师 | 同为 **Qwen3-Embedding-4B** | 同 |
| 池化 | **last-token（EOS）** | 同 |
| 前缀 | `Query:` / `Document:`（检索非对称） | 同 |

宣称：在同量级多语模型中 **MMTEB / 英文 MTEB 平均分领先或持平**；支持 **MRL 截断** 与 **二值量化**（GOR 增强鲁棒）。

---

## 2. 动机：小模型为何要「蒸馏 + 任务损失」

三条常见路线：

| 路线 | 代表 | 问题 |
| --- | --- | --- |
| 纯对比 | E5 / 多数 bi-encoder | 小骨干难吃满大规模难负例信号 |
| 纯蒸馏 | Jasper 等 | 可对齐教师，但对 **检索 vs STS vs 聚类** 冲突优化不够 |
| 指令微调大模型 | Qwen3-Embedding | 效果强，但 **提示工程成本** 与 **体积** 不适合边缘 |

论文主张并验证：

1. **嵌入蒸馏 ≫ 朴素 InfoNCE**（同数据同预算下终局检索更好）  
2. **蒸馏 + 任务专用目标 ≫ 纯蒸馏**  
3. **任务冲突** 用 **LoRA 分身** 解决，而不是一个 adapter 通吃  

相对教师 Qwen3-Embedding-4B：学生用 **固定前缀**，避免把教师的细粒度 instruction 迷宫搬进小模型。

---

## 3. 架构

```text
输入文本
  → 可选前缀 Query: / Document:
  → Transformer（+ 当前任务 LoRA）
  → last-token pooling
  → L2 归一化（部署常见）
  → 可选 MRL 截断前缀维
```

### 3.1 与教师的维数差

教师嵌入维 $m$ **高于** 学生维 $n$。蒸馏时用可学习仿射：

$$
\psi(\mathbf{z}) = W\mathbf{z} + \mathbf{b},\qquad
\psi:\mathbb{R}^{n}\to\mathbb{R}^{m}
$$

把学生投到教师空间再比余弦——消融表明：**学生→教师投影优于教师→学生**；随机初始化且 **不冻结** 投影头终局最好，冻结则收敛更快。

### 3.2 任务 LoRA

四类 adapter（rank=32, $\alpha$=32，附录）：

| Adapter | 前缀策略 | 核心损失 |
| --- | --- | --- |
| Asymmetric retrieval | Query / Document | InfoNCE + 蒸馏 + GOR |
| Text matching (STS) | 仅 Document | CoSENT 或 NCE+蒸馏 |
| Clustering | Document | 聚类指令下的蒸馏 |
| Classification | Document | 双向 NCE + 关系蒸馏正则 |

推理时 **显式选择任务**，与 jina-embeddings-v3 的 LoRA 任务路由同思路。

---

## 4. 第一阶段：Embedding Distillation

### 4.1 数据与教师指令

- 数据：$(q,d)$ 对，来自 **300+ 数据集、30+ 语言**（标题–摘要、问答等）  
- 学生：仅 `Query:` / `Document:`  
- 教师：统一用 Sentence-Transformers 默认指令  
  *“Given a web search query, retrieve relevant passages that answer the query”*  
  ——刻意 **最小化** instruction 歧义，便于蒸馏迁移  

### 4.2 RoPE $\theta$：训短推长

训练语料偏短，但要服务长文。做法：**训练 $\theta$ 远小于推理 $\theta$**，利于外推（Zhang / Liu 等 RoPE 外推工作）。

| 阶段 | small $\theta$ | nano $\theta$ | max tokens |
| --- | --- | --- | --- |
| 通用蒸馏 | 1M | 250K | 512 |
| 长文续训（仅 small） | **500K** | — | **4096** |
| 推理（表 1） | **3.5M** | 1M | 32K / 8K |

small 在通用蒸馏后长文仍弱，故加 **Long Context Training**：噪声长文中的针检索、章节–LLM 查询对、多语 1k–4096 token 对。

### 4.3 蒸馏损失

batch 上学生 / 教师嵌入：

$$
\mathcal{B}_{S}=\{(\mathbf{x}^{S}_{i},\mathbf{y}^{S}_{i})\}_{i=1}^{B},\quad
\mathbf{x}^{S},\mathbf{y}^{S}\in\mathbb{R}^{n}
$$

$$
\mathcal{B}_{T}=\{(\mathbf{x}^{T}_{i},\mathbf{y}^{T}_{i})\}_{i=1}^{B},\quad
\mathbf{x}^{T},\mathbf{y}^{T}\in\mathbb{R}^{m}
$$

$$
\mathcal{L}_{\mathrm{distill}}
=
\sum_{i=1}^{B}
\sum_{\mathbf{z}\in\{\mathbf{x},\mathbf{y}\}}
\bigl[
1-\phi\bigl(\psi(\mathbf{z}^{S}_{i}),\,\mathbf{z}^{T}_{i}\bigr)
\bigr]
$$

其中 $\phi$ 为余弦相似度。即：**逐样本、query 与 document 两侧**，学生投影后与教师嵌入对齐。

### 4.4 超参（附录 A1 摘录）

| | small | nano |
| --- | --- | --- |
| Steps | 50,000 | 50,000 |
| Devices × batch | $8\times512$ | $8\times1024$ |
| LR | $1\times10^{-4}$ | 同 |
| 长文续训 | 6,500 step，$2\times64$，4096 tok | 无（或未强调） |

---

## 5. 第二阶段：任务 Adapter（骨干冻结）

投影层权重从第一阶段 **复用**；骨干冻结，只训 LoRA。

### 5.1 检索 Adapter

数据：难负例三元组 + 长文数据（动态调 seq / batch）。三损失：

**（1）带难负的 InfoNCE**（query→doc）：

$$
S(\mathbf{x},\mathbf{y})=\exp\bigl(\phi(\mathbf{x},\mathbf{y})/\tau\bigr)
$$

$$
\mathcal{L}_{\mathrm{NCE}}^{q\rightarrow d}
=
-\frac{1}{B}\sum_{i=1}^{B}
\ln
\frac{S(\mathbf{x}_{i},\mathbf{y}_{i})}
{S(\mathbf{x}_{i},\mathbf{y}_{i})+\sum_{\mathbf{n}\in\mathcal{N}_{x_{i}}}S(\mathbf{x}_{i},\mathbf{n})}
$$

$\mathcal{N}_{x_{i}}$ = in-batch 非匹配文档 ∪ 挖掘难负；$\tau$ **可学习**。

**（2）蒸馏** $\mathcal{L}_{\mathrm{distill}}$：防止任务微调冲掉教师几何。

**（3）GOR（Global Orthogonal Regularizer）**：

$$
\mathcal{L}_{\mathrm{GOR}}
=
\frac{1}{B(B-1)}
\sum_{i\neq j}(\mathbf{x}_{i}^{\top}\mathbf{x}_{j})^{2}
+
\frac{1}{B(B-1)}
\sum_{i\neq j}(\mathbf{y}_{i}^{+\top}\mathbf{y}_{j}^{+})^{2}
$$

鼓励嵌入在球面上更均匀，**利 ANN 与二值量化**。

联合：

$$
\mathcal{L}_{\mathrm{retrieval}}
=
\lambda_{\mathrm{NCE}}\mathcal{L}_{\mathrm{NCE}}^{q\rightarrow d}
+
\lambda_{D}\mathcal{L}_{\mathrm{distill}}
+
\lambda_{S}\mathcal{L}_{\mathrm{GOR}}
$$

附录默认：$\lambda_{D}=2$，其余 $=1$，$\tau=0.02$。最后 **checkpoint 平均**（末点与较早点）。

### 5.2 STS / Text-matching Adapter

对称任务 → **只用 `Document:`**。有等级分时用 **CoSENT**：

对 batch $\{(\mathbf{x}_{i},\mathbf{y}_{i},s_{i})\}$，$s_{i}$ 为真值相似度：

$$
\mathcal{L}_{\mathrm{co}}
=
\ln\!
\Bigg[
1+
\sum_{s_{i}>s_{j}}
\frac{
e^{\phi(\mathbf{x}_{j},\mathbf{y}_{j})}
-
e^{\phi(\mathbf{x}_{i},\mathbf{y}_{i})}
}{\tau'}
\Bigg]
$$

无分数时退回 $\lambda_{\mathrm{NCE}}\mathcal{L}_{\mathrm{NCE}}+\lambda_{D}\mathcal{L}_{\mathrm{distill}}$（比例 **1:2**）。数据含 STS12、SICK、多语机翻 STS、平行句与释义对。

### 5.3 Clustering Adapter

通用检索指令蒸馏对聚类 **不够**（附录 A15）。改为教师指令：

> Identify the topic or theme of the given document:

数据：新闻标题–描述等聚类友好对；学生侧统一 `Document:`；损失仍为嵌入蒸馏式。

### 5.4 Classification Adapter

标签数据改成 **1 anchor + 1 同标正 + 7 异标负**。双向 NCE：

$$
\mathcal{L}=\mathcal{L}_{\mathrm{NCE}}^{q\rightarrow d}+\mathcal{L}_{\mathrm{NCE}}^{d\rightarrow q}
$$

另加 **关系知识蒸馏** $\mathcal{L}_{r}$（相对「无 adapter 的基座」），防特征坍塌、保零样本：

$$
\mathcal{L}_{r}
=
\sum_{i,j}
\frac{1}{M^{2}}
\left(
\frac{1-\phi(\mathbf{s}_{i},\mathbf{s}_{j})}{\mu_{S}}
-
\frac{1-\phi(\mathbf{t}_{i},\mathbf{t}_{j})}{\mu_{T}}
\right)^{2}
$$

权重示例：$\lambda_{\mathrm{NCE}}=1$，$\lambda_{R}=20$。

---

## 6. 主结果（论文表）

### 6.1 MMTEB（Multilingual v2）

| Model | Params | Avg Tasks | Ret | STS | Cls |
| --- | --- | --- | --- | --- | --- |
| Qwen3-4B（教师） | 4B | **69.5** | 69.6 | 80.9 | 72.3 |
| Qwen3-0.6B (instr.) | 596M | 64.3 | 64.7 | 76.2 | 66.8 |
| j-v5-text-**small** | 677M | **67.0** | 64.9 | 78.9 | **71.3** |
| Gemma-300M | 308M | 61.1 | 62.5 | 74.7 | 60.9 |
| j-v5-text-**nano** | **239M** | **65.5** | 63.3 | 78.2 | 69.2 |

同量级中 small / nano **平均分最高档**；分类与多标签分类（MLC）尤其突出。教师仍大幅领先——蒸馏不是魔法，是 **参数效率**。

### 6.2 英文 MTEB v2

| Model | Avg Tasks | Ret | STS | Cls |
| --- | --- | --- | --- | --- |
| Qwen3-4B | 74.6 | 68.5 | 88.7 | 89.8 |
| j-v5-text-small | **71.7** | 60.1 | 88.1 | **90.4** |
| KaLM-mini-v2.5 | 71.3 | 58.5 | 84.8 | 90.5 |
| j-v5-text-nano | 71.0 | 58.8 | **88.3** | 89.7 |
| Qwen3-0.6B (instr.) | 70.5 | **61.8** | 86.6 | 84.6 |

small 在小多语模型里 **英文平均最高**；检索上指令版 Qwen3-0.6B 仍略强——符合「教师同骨干 + 细指令」优势。

### 6.3 检索专项聚合（Table 4）

跨 MTEB-M / MTEB-E / RTEB(public) / BEIR / LongEmbed：

- **small**：任务级均分 **63.28**，同级最高；RTEB **66.84** 强  
- **nano**：BEIR / 英文 MTEB 在 **&lt;500M** 中很强；Voyage-4-nano 在 RTEB / Long 更高但维数 2048、体量更大  
- 相对 jina-v3 / snowflake-l-v2 / mE5-large-instruct：**全面代差提升**

---

## 7. 消融：方法为什么成立

### 7.1 三种训练目标（检索）

在 S2ORC 与全量混合上比：

| 目标 | 行为 |
| --- | --- |
| Score distillation（相似矩阵 MSE） | 起速快，早平台 |
| InfoNCE | 起速快，中后乏力 |
| **Embedding distillation** | 前期慢，**终局最高** |

结论：对齐 **向量本身** 比对齐 **softmax 分数** 更能持续改进。

Score 损失形式：

$$
\mathcal{L}_{\mathrm{score}}
=
\sum_{\mathbf{z}\in\{\mathbf{x},\mathbf{y}\}}
\frac{1}{B}
\sum_{i,j}
\bigl(p^{S}_{i,j}(\mathbf{z})-p^{T}_{i,j}(\mathbf{z})\bigr)^{2}
$$

$$
p^{\alpha}_{i,j}(\mathbf{z})
=
\frac{\exp(\phi(\mathbf{z}^{\alpha}_{i},\mathbf{z}^{\alpha}_{j})/\tau)}
{\sum_{k}\exp(\phi(\mathbf{z}^{\alpha}_{i},\mathbf{z}^{\alpha}_{k})/\tau)}
,\quad \tau=0.02
$$

### 7.2 检索三损失组合

| 配置 | MTEB Retr | RTEB |
| --- | --- | --- |
| NCE + distill + GOR | **64.50** | **66.45** |
| NCE + distill | 64.21 | 66.16 |
| NCE + GOR | 64.11 | 66.11 |
| distill alone | 63.16 | 64.37 |

**仅蒸馏不够** → 必须有 Stage 2 任务损失；三者齐全最好。

### 7.3 GOR 与二值量化

| | MTEB BF16 → Binary | RTEB BF16 → Binary |
| --- | --- | --- |
| with GOR | 64.50 → 62.60（**-1.90**） | 66.45 → 63.94（**-2.51**） |
| w/o GOR | 64.21 → 61.13（**-3.08**） | 66.16 → 62.24（**-3.92**） |

全精度增益不大；**量化掉点明显减半**——GOR 的主价值在部署压缩。

### 7.4 MRL 截断

MMTEB 检索随维数下降：到 **256** 仍较稳；**&lt;256** 开始可观跌落，与 Johnson–Lindenstrauss 式直觉一致。生产可优先 **512 / 256** 试水。

---

## 8. Nano vs Small：怎么选

| 维度 | 选 nano | 选 small |
| --- | --- | --- |
| 延迟 / 显存 | 边缘、CPU、高 QPS | GPU 服务 |
| 上下文 | ≤8K 够用 | 需要 **32K**、LongEmbed 更稳 |
| 语言覆盖 | EuroBERT 偏欧亚主要语 | Qwen3 词表 / 语种面更广 |
| 分数 | 已超多数 &lt;300–500M | 同级 MMTEB / 英文平均顶尖 |
| 与 v5-omni | → **omni-nano** 文本塔 | → **omni-small** 文本塔 |

二者 **训练菜谱同构**；差别主要在骨干容量、维数、长文续训与推理 $\theta$。

---

## 9. 与兄弟产品的边界

| | v5-text | jina-clip-v2 | v5-omni |
| --- | --- | --- | --- |
| 输入 | 文本 | 文本 + 图 | 文本 + 图/视频/音频 |
| 训练 | 蒸馏 + 任务 LoRA | 双塔联合对比 | **冻结** v5-text + 训投影 |
| 文本向量稳定性 | 自身即权威 | 联合训，≠ v5-text | **bit-identical 继承** |
| 任务 LoRA | 有 | clip 路线不同 | **继承** + 任务投影套件 |

**不要**用 v5-text 的「检索 adapter」去解释 clip 的图文分数；也 **不要**假设 omni 改过文本塔——论文强调未改。

---

## 10. 实现核对清单

- [ ] 推理任务与 **LoRA 选择**一致（retrieval ≠ STS）  
- [ ] 非对称检索：query / document **前缀**与训练一致  
- [ ] 池化为 **EOS last-token**，非 mean（与 clip-v2 文本 mean 不同）  
- [ ] 长文：确认 RoPE $\theta$ 为 **推理配置**，非训练小 $\theta$  
- [ ] 蒸馏复现：投影方向为 **学生→教师空间**  
- [ ] 量化部署：优先用 **带 GOR** 的检索 adapter 权重  
- [ ] MRL 截断维落在训练前缀集合内  

---

## 11. 小结

**v5-text** 的配方可以压成一句话：**大教师嵌入蒸馏打底 → 冻结骨干 → 四任务 LoRA 各用各的损失**。消融表明嵌入蒸馏终局优于分数蒸馏与纯 NCE；检索侧 **NCE+蒸馏+GOR** 齐全最优，GOR 特化为压缩服务。**nano（EuroBERT）** 与 **small（Qwen3-0.6B）** 覆盖从 0.24B 到 0.68B 的部署光谱，并为后续 **v5-omni** 提供可锁定的文本几何。
