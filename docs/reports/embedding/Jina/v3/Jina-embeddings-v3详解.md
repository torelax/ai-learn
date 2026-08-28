# Jina Embeddings v3 技术详解

> 基于论文 [jina-embeddings-v3: Multilingual Embeddings With Task LoRA](https://arxiv.org/abs/2409.10173)（Sturua, Mohr, Akram, Günther et al., Jina AI；arXiv:2409.10173）。
> 前序：[v2](https://arxiv.org/abs/2310.19923)（8192 / ALiBi）、双语变体 Mohr et al. 2024。
> 本文把 **XLM-R + RoPE、Task LoRA、InfoNCE / CoSENT / 分离损失、MRL、合成难例修补、MTEB / LongEmbed 评测** 写全。

---

## 1. 一句话定位

**jina-embeddings-v3** 是 **570M 级多语长上下文嵌入**：在 XLM-RoBERTa 上换 **RoPE**、接 **任务专用 LoRA**，并纳入 **MRL 可截断维度**；英文 MTEB 超过当时 OpenAI / Cohere 多语产品线口径，多语任务全面压过 `multilingual-e5-large-instruct`，同时远小于 7B LLM 嵌入。


| 项 | 内容 |
| --- | --- |
| 规模 | Base **559M** + adapters → **572M**（adapter <3%） |
| 上下文 | **8192**（RoPE；训时 rotary base 1e4，推理可调至 2e4） |
| 输出维 | **1024**；MRL 可截到 32…1024 |
| 任务头 | 5 个 LoRA：**retrieval.query / retrieval.passage / separation / classification / text-matching** |
| 词表 | XLM-R **250K**；Mean pooling；FlashAttention 2 |
| 宣称 | EN MTEB avg ≈ **65.5+**；相对 e5-mistral-7B 仅低 ~1% 分但小一个数量级 |

相对 v2：从「单语 8K BERT」升级为「**多语 + 任务路由 + 可截维**」；相对 Instruct-Embedding：用 **显式 LoRA 选择** 替代难写的自然语言 instruction。

---

## 2. 问题背景

### 2.1 「通用嵌入」并不通用

同一向量空间要同时服务：非对称检索、STS、聚类、分类。E5 系用 `query:` / `passage:` 前缀缓解不对称；Instructor 系用自然语言指令。痛点：

- 指令难写、难复现，「prompt vibe」不稳定；
- 单一底座难在所有任务上同时最优；
- 7B LLM 嵌入涨分有限（论文称相对本模型 EN 仅约 +1%）但部署成本差一个数量级。

### 2.2 设计选择

1. **Task LoRA**：冻结骨干，按任务加载低秩适配器（rank 4）；
2. **合成数据修补**检索四类系统失败；
3. 集成 **MRL、instruction 思想、长上下文（RoPE）**——但主路径是 adapter 而非纯指令。

---

## 3. 模型架构

### 3.1 骨干改造

初始化 **XLM-RoBERTa** 权重，再改：

| 组件 | 改动 |
| --- | --- |
| 位置 | 绝对位置 → **RoPE** |
| 注意力 | **FlashAttention 2** |
| 池化 | **Mean** |
| 任务 | Attention 内 embedding / 线性层挂 **LoRA（rank 4）** |
| Tokenizer | 保留原 XLM-R 词表 |

试过 BGE-M3 式「延长位置表」：长文任务差，作者归因于 **短文为主训练 + mean pooling**（对比 BGE 的 multi-CLS）。

**Rotary base**：训练用 $10^4$，推理提到 $2\times10^4$，长文更好且短文不掉——与「增大 base 利于长文」（Xiong et al.；Zhang et al. mGTE）同族技巧。

### 3.2 Task LoRA 接口

每个输入带 **task id**（整数），动态选择 adapter：

| Adapter | 用途 |
| --- | --- |
| `retrieval.query` | 非对称检索的 query |
| `retrieval.passage` | 非对称检索的 passage |
| `text-matching` | STS、对称检索（如重复检测） |
| `classification` | 下游分类器用的特征 |
| `separation` | 聚类、重排 |

LoRA：对权重 $W$ 学习 $\Delta W=BA$（低秩），推理 $W'=W+BA$。五套 adapter 总开销 <3% 参数，可并存于内存。

评测约定（论文 §5.2）：分类 / pair-cls → classification；STS + ArguAna / CQADupstack / Quora → text-matching；其余检索 → retrieval；聚类 / rerank → separation。

---

## 4. 三阶段训练

```text
I    MLM 继续预训练（CulturaX，短→长）
II   配对嵌入微调（双向 InfoNCE，>10 亿对 / 300+ 子集）
III  冻结骨干，分别训练五个 Task LoRA
```

### 4.1 Stage I：多语 MLM

- 数据：**CulturaX**，89 语，英文约 20%；**单语 batch**，batch 间轮换语言；
- Whole-word masking；**不训 LoRA / 池化**；
- 先 **100k** steps @512，再 **60k** steps 长序列（附录：8192、更小 batch）；
- 尽管位置编码从绝对改为 RoPE，**从 XLM-R 初始化仍比随机快收敛**；
- 长文能力一度落后 v2，**加长长文 MLM** 后 NarrativeQA 等回升。

附录超参（摘要）：短文 8 卡、bs $128\times8$、lr $10^{-4}$；长文 bs $8\times8$、lr $5\times10^{-5}$。

### 4.2 Stage II：配对 InfoNCE

Mean pooling 后，双向 InfoNCE（论文 Eq.1–2）：

$$
\mathcal{L}_{\mathrm{pairs}}(B)
:=
\mathcal{L}_{\mathrm{NCE}}(B)+\mathcal{L}_{\mathrm{NCE}}(B^{\dagger}),
\tag{1}
$$

$$
\mathcal{L}_{\mathrm{NCE}}(B)
:=
-\sum_{(x_i,y_i)\in B}
\ln
\frac{e^{s(x_i,y_i)/\tau}}
{\sum_{i'=1}^{k} e^{s(x_i,y_{i'})/\tau}},
\tag{2}
$$

其中 $B=((p_1,q_1),\ldots,(p_k,q_k))$，$B^{\dagger}$ 为对调。数据：>**10 亿** 对，>**300** 子数据集；每 batch **只来自一个子集**。额外过滤：若短文本中 ≥80% 词（且至少 4 词）是长文本子串则丢弃——降低纯字面重叠捷径。同样 **先短后长**（短：192 seq、大 batch；长：1024、较小 batch）。$\tau$：短 0.05 / 长 0.02。

### 4.3 Stage III：Task Adapters

骨干冻结；各任务独立数据与损失。

#### 4.3.1 Classification → $\mathcal{L}_{\mathrm{triplet}}$

同构 Gecko：同 batch 内同类别 $(q,p)$ + 7 个异类负例。扩展 InfoNCE（Eq.3）：

$$
\mathcal{L}_{\mathrm{triplet}}(B)
:=
\mathbb{E}_{r\sim B}
\Bigg[
-\ln
\frac{e^{s(q,p)/\tau}}
{\sum_{i=1}^{k}\Big[
e^{s(q,p_i)/\tau}+\sum_{j=1}^{m}e^{s(q,n_{j,i})/\tau}
\Big]}
\Bigg]
+
\mathbb{E}_{r\sim B}
\Bigg[
-\ln
\frac{e^{s(p,q)/\tau}}
{\sum_{i=1}^{k}e^{s(p,q_i)/\tau}}
\Bigg],
\tag{3}
$$

$r=(q,p,n_1,\ldots,n_m)$。假负问题：其它 tuple 的同类文本可能进分母。对策（Gecko）：给每个 tuple 的文本 **追加 tuple 专属 ID**，让模型优先学「tuple 内分离」。

#### 4.3.2 Text Matching → CoSENT

适用于 STS 与对称检索。CoSENT（Eq.4；Li & Li / bojone）：

$$
\mathcal{L}_{\mathrm{co}}(B)
:=
\ln\Bigg[
1+\sum_{\substack{(q_1,p_1),(q_2,p_2)\in B \\ \zeta(q_1,p_1)>\zeta(q_2,p_2)}}
\frac{e^{s(q_2,p_2)}-e^{s(q_1,p_1)}}{\tau}
\Bigg].
\tag{4}
$$

$\zeta$ 为标注相似度。数据：STS12、SICK 等；机器翻译扩语种（WMT19、MADLAD-3B）；并混 NLI。每 step 抽 **一个** 数据集，套对应损失。

#### 4.3.3 Asymmetric Retrieval → 双 Adapter + $\mathcal{L}_{\mathrm{triplet}}$

Query / passage **两套 LoRA 联合训练**（消融显示双 adapter 优于单 adapter）。难负例：MS MARCO、NQ；无标注负例则用 BGE-large / BM25 挖掘。损失同 Eq.3。

#### 4.3.4 检索失败分析与合成修补

相对 v2 同族数据，归纳四类失败：

| ID | 失败模式 |
| --- | --- |
| F1 | **句法相似误导**：字面重叠压过真正相关 |
| F2 | **命名实体误读**：专名部分匹配 / 多义词 |
| F3 | **极性问句**：yes/no 问题取到「话题相关但不回答」 |
| F4 | **偏好低质文档**：短、重复、信息贫乏但含 query 词 |

F1–F3：提示词生成 **query + 优选回答 + 7 个建模该失败的负例**。  
F4：OpenAssistant oasst1/2 偏好分；最高质答案为正，质量低 ≥0.3 为负，不足则随机补。

#### 4.3.5 Separation → CoSENT 变体

聚类 / rerank：batch 内 $(x,l)$，同标签配对后套 $\mathcal{L}_{\mathrm{co}}$（Eq.5）：

$$
\mathcal{L}_{\mathrm{sep}}(B')
:=
\mathcal{L}_{\mathrm{co}}(B),\quad
B=\{(x_i,x_j)\mid \exists l:\ (x_i,l),(x_j,l)\in B'\}.
\tag{5}
$$

该类数据少，混入 Stage II 配对数据与对应损失，按 text-matching 同款「每 step 单数据集」调度。

### 4.4 Matryoshka Representation Learning（MRL）

训练时对多个前缀维度 $\{32,64,\ldots,1024\}$ 同时施加嵌入损失（Kusupati et al.），使向量 **按语义重要性大致有序**，部署可截断。概念上：

$$
\mathcal{L}_{\mathrm{MRL}}
=
\sum_{d\in\mathcal{D}}
w_d\,\mathcal{L}_{\mathrm{emb}}\big(\mathrm{Truncate}(e,d)\big),
$$

其中 $\mathcal{L}_{\mathrm{emb}}$ 为当前任务的 InfoNCE / CoSENT 等，$e$ 为全维嵌入。论文 Table 6：检索 nDCG 从 32→1024 由 52.54→63.35；STS 在 128 维后基本饱和——**低维对检索更敏感，对 STS 更稳健**。

---

## 5. 评测

### 5.1 骨干快检（短配对训练）

仅 1000 steps / ~200 万对、无 adapter 时，Jina-XLM-R 已优于 XLM-R / mBERT（EN avg 76.05 vs 73.38 / 69.18）。

### 5.2 MTEB（Table 3 摘要）


| Model | Avg (EN) | CF | CL | RT | STS |
| --- | --- | --- | --- | --- | --- |
| jina-embeddings-v2-base-en | 60.38 | 73.45 | 41.73 | 47.87 | 80.70 |
| **jina-embeddings-v3** | **65.52** | **82.58** | 45.27 | 53.87 | **85.8** |
| text-embedding-3-large | 64.60# | 75.45 | 49.01 | 55.44 | — |
| multilingual-e5-large-instruct | 64.41 | 77.56 | 47.10 | 52.47 | 84.78 |
| Cohere-embed-multilingual-v3.0 | 64.01 | 76.01 | 46.60 | 53.84 | 83.15 |

多语加权平均：v3 **64.44**，高于 mE5-large（59.58），接近 mE5-large-instruct（64.25）。相对 e5-mistral-7b-instruct（~66.63）：约 **+1% 分 / 12× 参数 / 4× 维度**——v3 的性价比叙事核心。

### 5.3 LongEmbed（Table 4）

text-matching adapter：avg **70.39**（NarrativeQA / Needle / Passkey / QMSum / SummScreenFD / WikiQA）。高于 v2-base-en（58.12）、text-embedding-3-large（51.30）、bge-m3（56.56）。作者解读：**RoPE + 长文继续预训练** 优于 v2 的 ALiBi 与 BGE-M3 的固定位置扩展，在该套长文检索上更稳。

### 5.4 失败用例（Table 5）

手选失败集 mAP：retrieval adapter（\*\*）相对仅 pair training（\*）在 F1/F3/F4 上明显抬升；F2（实体）改善有限。更大规模合成失败集上，\*\* 全面高于 v2 与 mE5。局限：合成评测可能与训练分布过近。

### 5.5 消融

**双 adapter vs 单 adapter × 有无 instruction**（Table 7）：平均以 **双 adapter + instruction** 最高（45.98）；**双 adapter 无 instruction**（45.62）仍明显好于单 adapter 无 instruction（43.92）。结论：**容量（两套 LoRA）比指令更关键**，二者可叠加。

---

## 6. 局限

1. **低资源语言**：结论明确列为后续重点；CulturaX 覆盖 ≠ 均衡质量。
2. **用户必须选对 adapter**：选错任务头会系统性掉分（评测协议本身说明任务敏感）。
3. **F2 实体失败**：合成修补未完全解决；专名/歧义仍需领域数据或 NER 侧车。
4. **假负与 tuple-ID 技巧**：classification 训练的工程偏置，迁移到真实分布需验证。
5. **骨干仍是 encoder ~0.57B**：极端推理/知识密集型检索可能仍输给更大 LLM 嵌入。
6. **长文主要靠 RoPE 外推 + 有限长文阶段**：不是全量 8K 对比预训练。

---

## 7. 谱系笔记

```text
v2:  双向 ALiBi BERT · 英 · 8K · 两阶段 InfoNCE
  → 双语变体（zh/es/de）
v3:  XLM-R+RoPE · 多语 · Task LoRA×5 · MRL · 失败合成数据
v4:  Qwen2.5-VL-3B · 图文统一 · 单/多向量联合训练 · LoRA×3 · 32K
```

v3 的关键遗产：

1. **用 LoRA 做任务路由** 比纯 instruction 更可运维；
2. **MRL** 成为后续产品默认；
3. **失败驱动合成数据** 进入检索 adapter 标准动作；
4. 为 v4 的「冻结 VLM + 多 LoRA」铺路。

---

## 8. 公式速查

| 公式 | 角色 |
| --- | --- |
| Eq.1–2 $\mathcal{L}_{\mathrm{pairs}}$ / $\mathcal{L}_{\mathrm{NCE}}$ | Stage II 双向对比 |
| Eq.3 $\mathcal{L}_{\mathrm{triplet}}$ | 分类 / 检索硬负例 |
| Eq.4 $\mathcal{L}_{\mathrm{co}}$ | STS / text-matching |
| Eq.5 $\mathcal{L}_{\mathrm{sep}}$ | 聚类 / rerank |
| LoRA $\Delta W=BA$ | 任务适配（r=4） |
| MRL 多维截断损失 | 可部署低维向量 |

常用 $\tau\in\{0.02,0.05\}$（见附录 A1）。

---

## 9. 小结

jina-embeddings-v3 把嵌入模型从「一个通用向量」推进到「**一个多语骨干 + 一组可切换任务子空间**」。数学上仍是 InfoNCE / CoSENT 家族，工程上用 **Task LoRA + MRL + 失败合成** 换来了接近 7B LLM 嵌入的英文分数、更好的多语与长文，以及可控的 570M 部署形态。选对 adapter、按需截维，是落地时的两个一等公民旋钮。
