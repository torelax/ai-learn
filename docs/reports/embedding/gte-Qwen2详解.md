# gte-Qwen2 技术详解

> 方法底本：[Towards General Text Embeddings with Multi-stage Contrastive Learning](https://arxiv.org/abs/2308.03281)（Li et al., 阿里 GTE 家族，2023）。
> 模型卡：[Alibaba-NLP/gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct) / [gte-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct)（2024-06 前后发布）。
> 本文把 **GTE 多阶段对比配方 → Qwen2 LLM 骨干改造（双向注意力、query-side instruction、last-token 池化）** 与相对 E5-Mistral 的对照写全。

---

## 1. 一句话定位

**gte-Qwen2-*-instruct** 是阿里 GTE（General Text Embeddings）系列从 **BERT/Encoder 多阶段对比** 跃迁到 **Qwen2 Decoder LLM 骨干** 的开源嵌入模型：


| 项 | gte-Qwen2-7B-instruct | gte-Qwen2-1.5B-instruct |
| --- | --- | --- |
| 骨干 | Qwen2-7B | Qwen2-1.5B |
| 输出维 | **3584** | **1536** |
| 上下文 | **32,000** | 32,000 |
| 注意力 | **双向（去掉因果掩码）** | 同左 |
| 指令 | **仅 query 侧** instruction | 同左 |
| 池化 | **last-token** | last-token |
| 许可 | Apache-2.0 | Apache-2.0 |
| MTEB / C-MTEB（卡称，约 2024-06） | **~70.24 / ~72.05** | ~67.16 / ~67.65 |

相对 GTE-base/large（Encoder、512 或 v1.5 的 8K），gte-Qwen2 用 LLM 世界知识与长上下文换取榜单领先；相对 **e5-mistral-7b-instruct**，同为 7B 级 Decoder 嵌入，但底座、维数、中文表现与注意力改造细节不同。

---

## 2. 问题背景：GTE 家族在解决什么

### 2.1 通用嵌入的「多任务冲突」

单一嵌入模型要同时服务：

- **非对称检索**（短 query → 长 passage）；
- **对称相似度**（STS、复述）；
- **分类 / 聚类**（类别语义几何）。

若只用 MS-MARCO 一类检索数据，STS 易伤；只用 NLI，检索又弱。GTE 的回答是 **多阶段对比学习**：先用海量弱监督对齐「广义相关」，再用小而干净的监督数据精修任务几何，必要时再加指令对齐。

### 2.2 从 Encoder-GTE 到 LLM-GTE

早期 GTE（base/large）走 BERT/RoBERTa + 多阶段 InfoNCE，在英文 MTEB 上与 E5/BGE 同档竞争。瓶颈包括：

1. **上下文长度**（经典 512；v1.5 用外推到 8K 仍受 Encoder 预训练分布限制）；
2. **世界知识与推理**弱于同参数 Decoder LLM；
3. **中文 / 多语** 需另训或依赖多语 Encoder。

gte-Qwen* 把同一对比哲学接到 Qwen 系列 Decoder 上：先有 gte-Qwen1.5-7B-instruct（MTEB ≈67.34 / C-MTEB ≈69.52），再升级到 **Qwen2** 底座得到本报告主体。

---

## 3. GTE 多阶段对比学习（方法底本）

设 query 嵌入 $\mathbf{q}$、正例文档 $\mathbf{d}^+$、负例集合 $\{\mathbf{d}^-_j\}$，相似度多为温度余弦：

$$
s(\mathbf{a},\mathbf{b})
=
\frac{\mathbf{a}^\top\mathbf{b}}{\tau\|\mathbf{a}\|\,\|\mathbf{b}\|}.

$$

**InfoNCE（单方向 query→doc）**：

$$
\mathcal{L}_{\mathrm{InfoNCE}}
=
-\log
\frac{
\exp\!\big(s(\mathbf{q},\mathbf{d}^+)\big)
}{
\exp\!\big(s(\mathbf{q},\mathbf{d}^+)\big)
+
\sum_{j}\exp\!\big(s(\mathbf{q},\mathbf{d}^-_j)\big)
}.

$$

实践中 $\mathbf{d}^-_j$ 来自 **in-batch negatives**，监督阶段再叠加 **hard negatives**。多阶段通常划分为：

### 3.1 Stage A — 弱监督 / 粗粒度对比预训练

- 数据：论坛标题–正文、网页标题–段落、QA、学术 title–abstract 等大规模弱对；
- 目标：学会「相关 vs 随机」的粗语义几何；
- 技巧：大 batch、任务前缀或简单模板，避免对称/非对称信号冲突。

### 3.2 Stage B — 监督对比精修

- 数据：MS-MARCO、NQ、NLI、FEVER、HotpotQA 等；
- 目标：提升检索 nDCG 与 STS Spearman；
- 负例：挖掘 hard negatives（可用既有强模型召回 top-$k$ 再排除正例）。

### 3.3 Stage C — 指令嵌入对齐（instruct 变体）

对 LLM 骨干尤为关键：**在 query 前拼接自然语言任务指令**，例如：

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query: {user_query}
```

文档侧通常 **不加指令**（或仅加极简角色标记），以降低语料侧索引成本——这与 E5-Mistral / Instructor 的「query-side instruction」一致。

三段可视为同一 InfoNCE 在 **数据分布与模板** 上的课程学习，而非更换损失函数家族。

---

## 4. gte-Qwen2 架构改造

### 4.1 底座：Qwen2-7B

公开结构量级（与 Qwen2-7B 一致）：

- **28** Transformer 层；
- hidden size **$d=3584$**（故嵌入维默认 3584）；
- GQA、SwiGLU、RoPE、RMSNorm；
- 原生因果 LM 预训练。

嵌入任务不需要词表生成头参与打分，但保留 LLM 表示能力。

### 4.2 双向注意力（Bidirectional attention on decoder）

标准 Decoder 自注意力带因果掩码 $\mathcal{M}_{j\le i}$：

$$
\mathrm{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V})
=
\mathrm{softmax}\!\left(
\frac{\mathcal{M}_{j\le i}\odot(\mathbf{Q}\mathbf{K}^\top)}{\sqrt{d_h}}
\right)\mathbf{V}.

$$

嵌入需要 **整句上下文**，因果掩码使位置 $i$ 无法看见 $i+1,\ldots,n$。gte-Qwen2 在嵌入训练/推理时 **去掉因果约束**，改用全 1（或等价 bidirectional mask）：

$$
\mathcal{M}_{ij}^{\mathrm{bi}} = 1\quad \forall\, i,j.

$$

这与 LLM2Vec / NV-Embed / Conan-v2 soft-mask 终点态同向：**Decoder 权重 + Encoder 式信息流**。

注意：底座是因果预训练的，直接改 mask 会有分布偏移；GTE 团队用多阶段对比（及指令数据）把注意力权重重新校准到双向语义匹配。

### 4.3 Query-side instruction

设任务指令文本为 $I$，用户查询为 $x_q$，文档为 $x_d$：

$$
\mathbf{q} = f_\theta\!\big([I;\, x_q]\big),\qquad
\mathbf{d} = f_\theta\!\big(x_d\big),

$$

其中 $f_\theta$ 为双向 Qwen2 + 池化。对称任务（STS）可将同一指令套在两侧，或使用「Retrieve semantically similar text」类通用指令——以模型卡评测脚本为准。

**只在 query 侧加指令** 的工程收益：

1. 文档库 **一次编码、多任务复用**；
2. 换任务只需改 query 模板，无需重建索引；
3. 与 RAG「同一语料、多种提问方式」部署模式契合。

### 4.4 Last-token pooling

对长度为 $L$ 的 token 隐状态 $\mathbf{H}=[\mathbf{h}_1,\ldots,\mathbf{h}_L]$：

$$
\mathbf{e} = \mathrm{Norm}(\mathbf{h}_L) \in \mathbb{R}^{3584}.

$$

在 **双向** 设定下，末 token 已能 attend 全文，故 last-token 近似「句向量汇聚位」，不同于纯因果 LM 必须依赖 EOS 才能汇总前文。也可与 mean pooling 对照；模型卡默认 last-token。

### 4.5 与 GTE Encoder 代际对比


| 维度 | GTE-base / large | gte-Qwen2-7B-instruct |
| --- | --- | --- |
| 骨干 | Encoder-only | Qwen2 Decoder→双向 |
| 维数 | 768 / 1024 | **3584** |
| 上下文 | 512（v1.5→8K） | **32K** |
| 指令 | 弱 / 前缀式 | 自然语言 instruct |
| 典型 MTEB | ~62–63 | **~70.24**（发布时英文榜前列） |
| 中文 C-MTEB | 另有 gte-zh 线 | **~72.05** |

---

## 5. 训练目标再展开

### 5.1 批内负例与 hard negative

对 batch $\mathcal{B}=\{(q_i,d_i)\}_{i=1}^{B}$，令

$$
\mathcal{L}
=
-\frac{1}{B}\sum_{i=1}^{B}
\log
\frac{
e^{s(q_i,d_i)/\tau}
}{
e^{s(q_i,d_i)/\tau}
+
\sum_{j\neq i}e^{s(q_i,d_j)/\tau}
+
\sum_{m=1}^{H}e^{s(q_i,d_{i,m}^{\mathrm{hn}})/\tau}
}.

$$

弱监督阶段常取 $H=0$、极大 $B$；监督阶段引入 $H$ 个难负例。温度 $\tau$ 控制分布尖锐度（实践常见 $0.01$–$0.05$ 量级，以实现为准）。

### 5.2 多语与弱监督语料

模型卡强调：在 **大规模多语弱监督 + 监督** 语料上综合训练，覆盖检索、分类、STS 等多场景。这解释了为何同参数下 **C-MTEB** 相对 e5-mistral（偏英文指令合成）更强。

### 5.3 与「生成式 LM 损失」的关系

嵌入阶段 **不以 next-token CE 为主损失**；骨干的语言能力来自 Qwen2 预训练，嵌入能力来自对比阶段。若混用生成损失（GritLM 路线），需权衡 Gen/Emb；gte-Qwen2 公开叙述以 **对比多阶段** 为主。

---

## 6. 评测结果（模型卡口径）

### 6.1 英文 MTEB / 中文 C-MTEB（约 2024-06-16）


| 模型 | MTEB (EN) | C-MTEB | MTEB-Code 等（卡列） |
| --- | --- | --- | --- |
| multilingual-e5-large | 61.50 | 58.81 | — |
| e5-mistral-7b-instruct | 66.63 | 60.81 | — |
| gte-Qwen1.5-7B-instruct | 67.34 | 69.52 | — |
| NV-Embed-v1 | 69.32 | — | — |
| **gte-Qwen2-7B-instruct** | **70.24** | **72.05** | 68.25 / 67.86（卡内其它列） |
| gte-Qwen2-1.5B-instruct | 67.16 | 67.65 | 66.60 / 64.04 |

解读：

1. **Qwen1.5→Qwen2** 同为 7B instruct 嵌入，英文 +≈2.9，中文 +≈2.5；
2. 相对 **E5-Mistral**，英文高约 **3.6**，中文高约 **11+**——中文/多语是 gte-Qwen2 相对优势；
3. 1.5B 变体用约 **1/5** 激活参数换取仍强于多数 BERT 系 large 的分数，适合成本敏感部署。

### 6.2 任务画像（定性）

LLM 骨干嵌入通常：

- **检索 / 分类 / 指令敏感任务** 提升明显；
- **纯短句 STS** 不一定相对专用小模型有同等涨幅；
- **长文档** 受益于 32K 上下文，但评测集若本身短，优势会被截断评测抹平——需在真实 RAG 语料验证。

---

## 7. 与 E5-Mistral 对照


| 维度 | e5-mistral-7b-instruct | gte-Qwen2-7B-instruct |
| --- | --- | --- |
| 底座 | Mistral-7B | **Qwen2-7B** |
| 训练叙事 | 合成指令数据 + 对比（Wang et al. 2024） | **GTE 多阶段对比** + 多语语料 |
| 注意力 | 实现上常保留/部分改；社区多用 last-token | **明确双向注意力** |
| 指令 | query-side | query-side（同源思想） |
| 维数 | 4096 | **3584** |
| 中文 | 中等 | **明显更强（C-MTEB）** |
| 开源许可 | MIT 系（以卡为准） | Apache-2.0 |
| 同期 MTEB EN | ~66.6 | **~70.2** |

共同点：**Decoder LLM + 对比学习 + query 指令** 已成为 2024 开源榜单主流。差异在于底座语言配比、是否强制 bidirectional、以及多阶段数据配比——选型时：

- 偏英文、已有 Mistral 生态 → E5-Mistral / 其继任；
- 需 **中英双强 + 32K + Apache** → gte-Qwen2；
- 要更小维/更快 → 1.5B 变体或后续 mGTE Encoder 线。

---

## 8. 推理与部署要点

### 8.1 模板

务必使用官方/评测脚本中的 instruction 列表（与 E5-Mistral 任务指令表高度同源，但需以 Alibaba-NLP 脚本为准）。错误模板会导致「同模型不同分」。

### 8.2 归一化与相似度

检索与 STS 使用 **L2 归一化 + 点积（≡余弦）**。分类任务有的实现会关闭归一化（E5/Nomic 亦有类似经验）——按任务消融。

### 8.3 成本

| 项 | 影响 |
| --- | --- |
| 7B 前向 | 建库贵；可用 1.5B 或蒸馏学生 |
| 3584-d 索引 | 内存/ANN 比 768-d 高约 $4$–$5\times$ |
| 32K 上下文 | 注意力 $O(L^2)$，长文建议分块或配压缩方案 |
| 双向相对因果 | 实现需改 attention mask；确认框架支持 |

### 8.4 复现评测

模型卡指向 `scripts/eval_mteb.py` 复现 MTEB/C-MTEB。对比他人分数时核对：**英文 v1 任务集、截断长度、指令表、是否归一化**。

---

## 9. 实现对照清单

```text
1. 加载 Qwen2-*-instruct 权重
2. 替换因果 mask → bidirectional
3. 定义 last-token pooling + L2 norm
4. 弱监督 InfoNCE（大 batch, in-batch neg）
5. 监督 InfoNCE（+ hard neg）
6. Query 侧拼接任务指令；文档侧不加（检索）
7. 多语数据配比；按官方脚本评 MTEB / C-MTEB
```

可验证目标：

1. 仅改双向 + 对比，相对「因果 + EOS」基线在检索集上应有稳定增益；
2. 去掉 query instruction，指令敏感任务应掉分；
3. 7B vs 1.5B 同数据曲线，验证缩放是否符合预期。

---

## 10. 小结与谱系位置

gte-Qwen2 把 **GTE 多阶段 InfoNCE** 接到 **Qwen2** 上，并用三项改造对齐嵌入归纳偏置：

1. **双向注意力** —— 补齐 Decoder 的上下文；
2. **Query-side instruction** —— 一库多任务；
3. **Last-token + 3584-d + 32K** —— LLM 级表示与长文接口。

在 2024-06 口径下，它以开源身份同时站上英文与中文嵌入榜前列，并成为后续 Qwen3-Embedding、蒸馏学生（如 Jasper）的重要教师候选。同目录可对照：《E5详解.md》（前缀时代起点）、《LLM2Vec详解.md》（无监督改双向配方）、《Conan-embedding-v2详解.md》（从零训嵌入向 LLM）。

---

## 参考

1. Li et al. (2023). Towards General Text Embeddings with Multi-stage Contrastive Learning. [arXiv:2308.03281](https://arxiv.org/abs/2308.03281)
2. Hugging Face: [gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct)
3. Wang et al. (2024). Improving Text Embeddings with Large Language Models. [arXiv:2401.00368](https://arxiv.org/abs/2401.00368)
4. Muennighoff et al. MTEB.
