# Token Prepending 技术详解

> 论文：[Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](https://aclanthology.org/2025.acl-long.159/)（ACL 2025 Long；[PDF v2](https://aclanthology.org/2025.acl-long.159v2.pdf)）。  
> 作者：Yuchen Fu\*, Zifeng Cheng\*, Zhiwei Jiang† 等（南京大学）。  
> 代码：https://github.com/fuyuchenIfyw/token_prepending.git  
> 本文把 **因果注意力下的反向依赖问题、层间 Token Prepending（TP）、early-exit、与 PromptEOL/MetaEOL 等的组合实验** 写清；并说明它与 Echo Embedding / LLM2Vec / 全参双向微调（如 QZhou）的路线差异。

---

## 1. 一句话定位

**Token Prepending（TP）** 是一种 **免训练、即插即用** 的推理期技巧：在保持 Decoder LLM **因果注意力** 与 **原权重不动** 的前提下，把「上一层解码出的句向量」接到下一层输入句首，让句中 **靠前 token 也能看见整句语义**，从而抬高 Prompt 式句向量质量。

| 项 | 内容 |
|----|------|
| 目标 | 从 LLM **零微调** 抽出更好句向量（STS / 分类） |
| 核心操作 | 层间：用上一层的 Sentence Embedding Token（SET）替换句首占位符 `<PST>` |
| 成本 | 序列只多 **1** 个 token；相对 PromptEOL 约 **1.04×** 时延 |
| 可组合 | PromptEOL、MetaEOL、Pretended CoT、Knowledge Enhancement 等 |
| 验证底座 | LLaMA2-7B/13B、LLaMA3-8B、Qwen2-7B、Gemma2-9B |

与 **Echo Embedding**（整句重复）比：TP 几乎不涨长度；与 **LLM2Vec / QZhou**（改双向注意力 + 训练）比：TP **不改架构、不训练**，但上限通常低于全参对比微调。

---

## 2. 问题：因果注意力下的「假全句」

Decoder-only LLM 的注意力是因果的：位置 $i$ 只能看 $\le i$。Prompt 系方法（PromptEOL 等）用模板把语义压到 **最后一个 token**（SET），因为 SET 理论上能看到整句：

```text
This sentence: "[Text]" means in one word: "
```

形式化：令模板化输入为 $T=[t_1,\ldots,t_n]$，嵌入层得到 $\mathbf{h}^0$，经 $L$ 层 Transformer：

$$
\mathbf{h}^{L} = \mathrm{LLM}^{1:L}(\mathbf{h}^{0}),
$$

取末层末位 $h_n^{L}$ 为句向量。

**漏洞**：SET 虽能看全序列，但句中 **靠前 token 的隐状态从未见过后面 token**。这些「半句上下文」的表示会级联污染后续层，最终仍扭曲 SET。Echo Embedding 用「输入两遍」补反向依赖，但长度约翻倍、结构被破坏。

TP 的目标：在 **不改因果掩码、不加参** 的条件下，让早期 token 也能「看见」当前层已形成的整句摘要。

---

## 3. 方法

### 3.1 总览

1. **输入层**：在 `[Text]` 前插入自定义占位符 `<PST>`（不在词表内，随机初始化嵌入）。  
2. **前若干层（prepending-enhanced）**：每层算完后，用当前层在 SET 位置的隐状态 **覆盖** `<PST>` 位置，作为下一层输入。  
3. **后续层**：停止覆盖，按标准前向。  
4. **Early-exit**：不取最后一层（偏生成头），而在验证集上选 **中间层** $M$ 的 SET 作为句向量。

论文默认配置（LLaMA2-7B + PromptEOL 等）：**前 8 层做 TP**，exit 多用 **第 27 层**（Knowledge Enhancement 用倒数第二层）。

### 3.2 Initial Token Prepending

第一层输入嵌入变为：

$$
\mathbf{h}^{0}
=
\big[
h_1^{0},\ldots,h_{i^{*}-1}^{0},\;
h_{i^{*}}^{0},\;
h_{i}^{0},\ldots,h_n^{0}
\big],
$$

其中 $i^{*}$ 为 `<PST>` 位置，$h_{i^{*}}^{0}$ 为随机初始化占位嵌入。

### 3.3 Intermediate Token Prepending

对层 $l\in[2,k]$（$k$ 为 TP 截止层）：

$$
\mathbf{h}^{l}
=
\mathrm{LLM}^{l}\!\big(f(\mathbf{h}^{l-1})\big),
$$

$f$ 把上一层在 SET（句末提示位）的向量 **写回** `<PST>` 位置：

$$
f(\mathbf{h}^{l-1})
=
\big[
h_1^{l-1},\ldots,
\underbrace{h_{\mathrm{SET}}^{l-1}}_{\text{替换 }h_{i^{*}}^{l-1}},
\ldots,h_n^{l-1}
\big].
$$

在因果掩码下，句中任意位置 $j\ge i^{*}$ 都能 attend 到这个「上一层整句摘要」，从而补上 **backward dependency**（见图 1(b)）。

截止后：

$$
\mathbf{h}^{l+1}=\mathrm{LLM}^{l}(\mathbf{h}^{l}),\quad l\in[k,M],
$$

$M$ 为 early-exit 层。

### 3.4 Early-Exit

末层更偏 next-token 预测，语义往往不如中间层（与 Liu et al. / Jin et al. 观察一致）。用 STS-B dev 扫 $M$，推理时只算到 $M$，可略加速。

---

## 4. 与相关路线对照

| 路线 | 改注意力？ | 训练？ | 长度 | 典型用途 |
|------|-----------|--------|------|----------|
| PromptEOL / MetaEOL / CoT prompt | 否 | 否 | ≈原长 | 零样本句向量 |
| **Echo Embedding** | 否 | 否 | ≈2× | 重复输入补反向依赖 |
| **Token Prepending** | 否 | 否 | +1 token | 层间注入 SET；可叠在任意 Prompt 上 |
| LLM2Vec | **是（双向）** | MNTP+SimCSE | 原长 | 训成编码器 |
| QZhou / gte-Qwen2 / E5-Mistral | **是（双向）** | 全参/LoRA 对比学习 | 原长 | 工业检索 SOTA |

**读 QZhou 时**：引言把 Echo、LLM2Vec、Conan soft-mask 等列为「缓解因果限制」的代表；Token-Prepending 同属该问题族的 **免训练补丁**，而 QZhou 最终选择 **双向注意力 + 全参多任务对比**——两条路互补而非互相替代。

---

## 5. 实验要点

### 5.1 主结果（LLaMA2-7B，STS Avg）

| 方法 | Avg | Δ | 相对 PromptEOL 时延 |
|------|-----|---|---------------------|
| PromptEOL | 70.03 | — | 1.00× |
| **PromptEOL + TP** | **77.19** | **+7.16** | 1.04× |
| MetaEOL | 75.96 | — | 8.17× |
| MetaEOL + TP | 77.91 | +1.95 | 8.29× |
| Pretended CoT | 76.86 | — | 1.18× |
| Pretended CoT + TP | 77.54 | +0.68 | 1.20× |
| Knowledge | 77.14 | — | 1.17× |
| Knowledge + TP | 77.54 | +0.40 | 1.20× |

解读：

- **对弱 Prompt（PromptEOL）增益最大**——更依赖「建模反向依赖」；强 Prompt 已注入先验，TP 边际变小。  
- TP 把不同 Prompt 的分差收窄，**提高对 Prompt 的鲁棒性**。  
- 时延几乎不变；MetaEOL 本身贵在多 Prompt，不是 TP。

### 5.2 跨底座（Pretended CoT ± TP）

| Backbone | 基线 Avg | +TP | Δ |
|----------|----------|-----|---|
| LLaMA2-7B | 76.86 | 77.54 | +0.68 |
| LLaMA2-13B | 73.34 | 74.62 | +1.28 |
| LLaMA3-8B | 76.09 | 76.78 | +0.69 |
| Qwen2-7B | 72.94 | 75.11 | **+2.17** |
| Gemma2-9B | 77.02 | 77.52 | +0.50 |

Qwen2 上增益偏大，说明 TP 对部分底座/Prompt 组合更「救命」。

### 5.3 消融与用法（论文结论摘要）

- **不必全层 TP**：早期层做 TP 更好；论文设定约到第 8 层停止。  
- **Early-exit 必要**：中间层优于盲目取 $L$。  
- 下游分类任务同样受益（论文有表；方向与 STS 一致）。  
- 与 Echo 比：质量更稳、成本更低。

---

## 6. 实践清单

**适合用 TP 时**

- 不能/不愿微调 7B+；只要 STS/聚类/轻量检索原型；  
- 已有 PromptEOL 一类脚本，想白嫖 +5～7 STS 分；  
- 与 KV cache 兼容（多 1 个位置，实现上改层间 hidden 注入）。

**不适合单独指望 TP 时**

- 要对齐 MTEB 检索 SOTA（需双向 + 难负例 + 大规模对比，走 QZhou/gte/E5-Mistral）；  
- 产线要稳定 Matryoshka/指令前缀/多任务 LoRA——TP 不提供这些 induct。

**实现要点**

1. 模板里在 `[Text]` 前插 `<PST>`；  
2. 前 $k$ 层：每层后 `hidden[:, i_pst] = hidden[:, i_set]`；  
3. 验证集扫 $(k, M)$；  
4. 推理固定该配置，勿每条样本重搜。

---

## 7. 公式速查

| 符号 | 含义 |
|------|------|
| SET | 模板末尾用于读出句向量的 token |
| `<PST>` | 句首占位，接收上一层 SET 向量 |
| $k$ | TP 截止层（含） |
| $M$ | early-exit 层 |
| $f(\cdot)$ | 将 SET 隐状态写入 `<PST>` 位置 |

核心递推（$2\le l\le k$）：

$$
\mathbf{h}^{l}
=
\mathrm{LLM}^{l}\!\big(f(\mathbf{h}^{l-1})\big),
\quad
\text{句向量}
=
h_{\mathrm{SET}}^{M}.
$$

---

## 8. 结论

Token Prepending 用 **「层间句向量回灌句首」** 在因果掩码下近似实现反向依赖，是 Prompt 式 LLM 句向量的高质量、低成本补丁。它解释并修补了「只靠 last-token 聚合」的结构缺陷；工业级检索模型（如 QZhou）则选择 **训练期改双向 + 对比学习**。两者回答同一问题的不同阶段：**TP 用于零训练榨干底座；QZhou 一类用于把底座训成专用 embedding。**

## 参考文献

1. Fu et al. Token Prepending. ACL 2025. https://aclanthology.org/2025.acl-long.159/  
2. Jiang et al. PromptEOL. 2023.  
3. Springer et al. Echo Embeddings / Repetition improves LM embeddings. 2024.  
4. BehnamGhader et al. LLM2Vec. 2024.  
5. Lei et al. MetaEOL；Zhang et al. Pretended CoT / Knowledge Enhancement.  
