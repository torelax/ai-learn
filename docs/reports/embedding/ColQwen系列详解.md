# ColQwen 系列详解：从 ColBERT / ColPali 到 ColQwen2 / 2.5 / 3

> 主文献：[ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449)（Faysse et al., arXiv:2407.01449）  
> 官方实现与模型表：[illuin-tech/colpali](https://github.com/illuin-tech/colpali)（`colpali-engine`）  
> 工程解读：[Weaviate — An Overview of Late Interaction Retrieval Models](https://weaviate.io/blog/late-interaction-overview)  
> 社区 ColQwen3 代表卡：[TomoroAI/tomoro-colqwen3-embed-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b)  
> 本文把谱系、MaxSim 数学、骨干差异、训练配方、ViDoRe 分数与工程取舍写全，便于对照实现与选型。

---

## 1. 一句话定位

**ColQwen 族**不是一套独立论文体系，而是 **ColPali（ColVision）训练范式在 Qwen-VL 骨干上的换壳与迭代**：

| 项 | 内容 |
| ---- | ------ |
| 范式 | **页图 → patch 多向量** + **文本 query → token 多向量** + **MaxSim late interaction** |
| 奠基论文 | **仅有 ColPali**（PaliGemma-3B）；**不存在单独的「ColQwen-v1 论文」** |
| 官方 Vidore | **ColQwen2**（Qwen2-VL-2B）、**ColQwen2.5**（Qwen2.5-VL-3B） |
| ColQwen3 | **社区 / 后期 checkpoint**（如 Tomoro / goodman2001），骨干多为 **Qwen3-VL**；仍应引用 ColPali，尚无独立奠基 tech report |
| 相对 OCR 管线 | 整页截图端到端编码，免 layout / OCR / caption 多段流水线 |

核心卖点：**视觉文档检索（PDF、表格、图表、版式）上，用细粒度 patch–token 对齐换精度；索引阶段更简单，在线查询仍可亚秒级。**

---

## 2. 与用户说的 ColQwen1/2/3 的对应关系

口语里常说「ColQwen(1,2,3)」。**按公开论文与 Vidore 官方仓库，应对齐如下——不要按字面去找「ColQwen1 论文」。**

| 用户口头说法 | 实际对应 | 骨干 | 权威出处 | 备注 |
| -------------- | ---------- | ------ | ---------- | ------ |
| **「ColQwen1」** | **不存在独立 ColQwen-v1** | — | — | 族谱上的「第一代视觉 ColBERT」是 **ColPali（PaliGemma）**，[arXiv:2407.01449](https://arxiv.org/abs/2407.01449) |
| **「ColQwen2」** | **ColQwen2** | `Qwen/Qwen2-VL-2B-Instruct` | Vidore：`vidore/colqwen2-v0.1`、`v1.0` | 论文 §6 已报告 **ColQwen2-VL（768 patches）** 相对 ColPali **+5.3 nDCG@5**；与 ColPali **同一训练配方** |
| **「ColQwen2.5」**（常被算进「2」的小版本） | **ColQwen2.5** | `Qwen/Qwen2.5-VL-3B-Instruct` | Vidore：`vidore/colqwen2.5-v0.1`、`v0.2` | 仍是官方 ColVision；动态分辨率；768 patches / page（训练设定） |
| **「ColQwen3」** | **社区 / 后期权重**，非 Illuin 奠基论文 | 多为 **Qwen3-VL**（如 4B） | 例：`TomoroAI/tomoro-colqwen3-embed-4b`、`goodman2001/colqwen3`；README 亦挂 `athrael-soju/colqwen3.5-4.5B-v3` | **仍应 cite ColPali**；方法仍是 MaxSim late interaction；**尚无单独 foundational tech report** |

记忆口诀：

```text
用户说的 ColQwen1  ≈  学术/产品意义上的 ColPali（PaliGemma）
用户说的 ColQwen2  ≈  官方 ColQwen2（± ColQwen2.5）
用户说的 ColQwen3  ≈  Qwen3-VL 上的社区 Col* 权重
```

后文若写「ColQwen 族」，默认包含：**ColPali 奠基范式 + 官方 ColQwen2/2.5 + 社区 ColQwen3**。

---

## 3. 谱系：ColBERT → ColPali → ColQwen2 → 2.5 → ColQwen3

### 3.1 交互范式三条线（为何需要 Late Interaction）

Dense 检索按「query–document 何时交互」可分三类（Weaviate 综述）：

| 类型 | 代表 | 离线文档编码 | 在线交互 | 精度 / 速度 |
| ------ | ------ | -------------- | ---------- | ------------- |
| **No-interaction（Bi-encoder）** | SBERT、BGE、OpenAI emb | 压成 **1 个向量** | 余弦 / 点积 | 快、可扩；细粒度弱 |
| **Full interaction（Cross-encoder）** | BERT rerank、Cohere Rerank | 几乎不预存 | query∥doc 联合注意力 | 准但贵；难做百万库一阶段 |
| **Late interaction** | **ColBERT / ColPali / ColQwen** | 存 **token/patch 多向量** | **MaxSim** | 介于二者：细粒度 + 可离线 |

Late interaction 的直觉：**文档侧像 bi-encoder 一样可预计算；打分侧像 cross-encoder 一样保留「每个 query 词对上文档哪一块」的细粒度，但交互推迟到检索时、且只做 MaxSim 而非全交叉注意力。**

### 3.2 ColBERT（2020）：文本 Late Interaction 奠基

**ColBERT**（Contextualized Late Interaction over BERT，[arXiv:2004.12832](https://arxiv.org/abs/2004.12832)）：

1. Query / Doc 各自过 BERT（加 `[Q]` / `[D]` 标记）。
2. 隐状态经线性投影到 **$D=128$**，**不池化**，保留每个 token 一个向量。
3. 用 **MaxSim** 打分（见 §4）。

后续 **ColBERTv2** 用蒸馏 + hard negative 提质，并用 residual / 低比特量化把索引体积压到接近单向量量级（仍高于纯 dense）。**PLAID** 等引擎解决大规模 late interaction 检索延迟。

局限：**纯文本**。PDF 要先 OCR / 版式解析；图、表、多栏、字体信息在文本通道里大量丢失。

### 3.3 ColPali（2024）：把 ColBERT 搬到「页图像」

**ColPali** = Contextualized Late Interaction over **PaliGemma**：

- 文档：**整页渲染成图** → ViT patches → 经 VLM（PaliGemma）上下文化 → 投影到 $D=128$ 的 **patch 多向量**。
- Query：文本 token → 同一语言模型空间 → token 多向量。
- 打分：与 ColBERT **同一套 MaxSim**。

同时发布 **ViDoRe**（Visual Document Retrieval Benchmark）：页级检索，覆盖 DocVQA / InfoVQA / 表格 / 领域 PDF（能源、政务、医疗、AI、法语 Shift 等）。论文结论：在视觉丰富文档上，**优化 ingestion（OCR/caption）往往比换文本 embedding 更有效**；而 ColPali **端到端视觉索引**在 ViDoRe 上大幅超过 Unstructured+OCR/Caption + BGE-M3 等强基线。

消融主线（同一训练集上逐步加能力）：

$$
\text{SigLIP} \xrightarrow{\text{对比微调}} \text{BiSigLIP}
\xrightarrow{+\text{LLM 上下文}} \text{BiPali}
\xrightarrow{+\text{Late Interaction}} \text{ColPali}
$$

论文 Table 2（nDCG@5 平均）：BiSigLIP ≈ 58.6，BiPali ≈ 58.8，**ColPali ≈ 81.3**——**多向量 late interaction 是跃迁主因**，不是单纯「换了个大 VLM」。

### 3.4 ColQwen2：同一配方，换 Qwen2-VL-2B

论文 §6 明确：用 **相同数据与训练策略** 训 **Qwen2-VL-2B**，得到 **ColQwen2-VL**；为与 ColPali 存储大致可比，**限制约 768 image patches**（ColPali 默认约 **1024**）。相对 ColPali **+5.3 nDCG@5**（附录 Table 7：ColQwen2(768) Avg **86.6** vs ColPali **81.3**）。

官方 Vidore 发布物（`colpali` README）：

| Checkpoint | ViDoRe 分数（README 表） | 要点 |
| ------------ | -------------------------- | ------ |
| `vidore/colqwen2-v0.1` | **87.3** | Qwen2-VL-2B；动态分辨率；**768 patches/page**；effective batch 32 |
| `vidore/colqwen2-v1.0` | **89.3** | 同架构；更大有效 batch（**256**）与更强算力 |

许可：**Apache 2.0**（相对 PaliGemma/Gemma 系更利于商用）。

### 3.5 ColQwen2.5：换 Qwen2.5-VL-3B

| Checkpoint | ViDoRe 分数 | 要点 |
| ------------ | ------------- | ------ |
| `vidore/colqwen2.5-v0.1` | **88.8** | Qwen2.5-VL-3B；动态分辨率；768 patches；batch 32 |
| `vidore/colqwen2.5-v0.2` | **89.4** | 超参微调版 |

与 ColQwen2 同属 **官方 ColVision 线**：配方继承 ColPali，差异主要在 **骨干生成能力 / 分辨率策略 / 训练超参**。

### 3.6 ColQwen3：社区延伸，非第二篇奠基论文

`colpali` README 已收录社区权重示例：

| 模型 | ViDoRe（README） | 说明 |
| ------ | ------------------ | ------ |
| `TomoroAI/tomoro-colqwen3-embed-4b` | **90.6** | Qwen3-VL 骨干；**320-d** ColBERT 式向量；动态分辨率 |
| `athrael-soju/colqwen3.5-4.5B-v3` | **90.9** | Qwen3.5-4B 混合注意力；320-d；LoRA |

Tomoro 卡补充：合并 **Qwen3-VL-4B-Instruct** 与 **Qwen3-Embedding-4B** 初始化，再在 VDR / ColPali train / VisRAG 等混合数据上微调；输出 $\text{SeqLen}\times 320$、L2 归一；页级视觉 token 预算可达 **1280**（视频帧序列更高）。**方法学仍写 ColPali-style MaxSim，cite arXiv:2407.01449。**

谱系简图：

```text
ColBERT (文本 token × 128, MaxSim)
    │
    ▼
ColPali / PaliGemma-3B   ←── 「视觉文档 ColBERT」奠基论文
    │         同一损失 / 数据 / LoRA 配方
    ├────────► ColQwen2   (Qwen2-VL-2B, 官方 Vidore)
    ├────────► ColQwen2.5 (Qwen2.5-VL-3B, 官方 Vidore)
    └────────► ColQwen3*  (Qwen3-VL 等, 社区权重；无独立奠基 TR)
```

---

## 4. 共享数学：MaxSim Late Interaction

整族（ColBERT → ColPali → ColQwen\*）共享同一打分核。

### 4.1 多向量表示

对 query $q$ 与文档页 $d$，编码器分别输出：

$$
\mathbf{E}_{q} \in \mathbb{R}^{N_{q} \times D},\qquad
\mathbf{E}_{d} \in \mathbb{R}^{N_{d} \times D}.
$$

- ColBERT：$N_{q}, N_{d}$ 为 **文本 token** 数。  
- ColPali / ColQwen：$N_{d}$ 主要为 **图像 patch（经 VLM 后的视觉 token）** 数，另可含少量提示文本 token；$N_{q}$ 仍为 query token（可含 query augmentation）。  
- $D$：投影维；经典 ColBERT/ColPali/官方 ColQwen2 **$D=128$**；部分 ColQwen3 社区模型用 **$D=320$**。

### 4.2 Late Interaction / MaxSim

论文公式 (1)：

$$
\mathrm{LI}(q,d)
=
\sum_{i=1}^{N_{q}}
\max_{j=1,\ldots,N_{d}}
\big\langle \mathbf{E}_{q}^{(i)},\, \mathbf{E}_{d}^{(j)} \big\rangle.
$$

含义：

1. 对每个 query 向量 $i$，在文档所有向量上取 **最大点积**（MaxSim）。  
2. 再对所有 query 向量 **求和**，得页级相关分。

等价写法（余弦时先 L2 归一化再点积）：

$$
s(q,d)
=
\sum_{i}
\max_{j}
\cos\!\big(\mathbf{e}_{q}^{(i)}, \mathbf{e}_{d}^{(j)}\big).
$$

**为何不是全连接再求和？** 全连接既贵，又易被长文档「分数膨胀」淹没；MaxSim 近似「每个查询词在页上找最亮的一块」，保留可解释的对齐，同时复杂度对文档侧是 $O(N_{q} N_{d})$ 的矩阵 max，可用 Triton 融合核避免物化 $[B,B,L_q,L_d]$ 巨大中间张量（见 `colpali-engine[lik]`）。

### 4.3 对比损失（训练）

batch $\{(q_k, d_k)\}_{k=1}^{b}$，正样本分 $s_k^{+}=\mathrm{LI}(q_k,d_k)$，in-batch 最难负例

$$
s_k^{-}
=
\max_{l\neq k}\mathrm{LI}(q_k,d_l).
$$

论文公式 (2)（softplus 稳定形式）：

$$
\mathcal{L}
=
\frac{1}{b}\sum_{k=1}^{b}
\log\!\big(1+\exp(s_k^{-}-s_k^{+})\big)
=
\frac{1}{b}\sum_{k=1}^{b}
\mathrm{softplus}(s_k^{-}-s_k^{+}).
$$

即 pairwise CE：只盯 **最强 in-batch 负例**，而非对所有负例做完整 softmax（消融显示完整 in-batch CE 略差约 1.6 nDCG@5）。

### 4.4 Query augmentation

沿用 ColBERT：在 query 后追加 **5 个可学习/特殊 token**，作软查询扩展与重加权。论文消融：英任务影响小；法语任务上去掉后反而有时更好——实现时注意语言与 checkpoint 设定。

### 4.5 可解释性

MaxSim 的 $\arg\max_j$ 可映回图像 patch 网格，叠加热力图：可见模型是否对齐到 OCR 文字区、坐标轴、图例等（ColPali Fig. 3）。这是单向量 dense 难以直接给出的定位信号。

---

## 5. 架构差异：骨干、动态分辨率、patch 数、嵌入维

### 5.1 总览对照

| 维度 | ColPali | ColQwen2 | ColQwen2.5 | ColQwen3（Tomoro 4B 例） |
| ------ | --------- | ---------- | ------------ | -------------------------- |
| 骨干 VLM | PaliGemma-3B（SigLIP + Gemma-2B） | Qwen2-VL-2B | Qwen2.5-VL-3B | Qwen3-VL-4B（常与 Embedding 权重融合初始化） |
| 参数量级 | ~3B | ~2B | ~3B | ~4B（卡标 ~4.4B） |
| 分辨率 | 固定网格（论文主设 448 → **1024 patches**） | **动态分辨率** | **动态分辨率** | **动态分辨率**；页视觉 token 预算可达 **1280** |
| 训练/论文常用 patch 上限 | 1024（消融有 512） | **768**（与 ColPali 存储对齐） | **768** | 更高上限；实现相关 |
| 投影维 $D$ | **128** | **128**（官方线） | **128** | 常见 **320** |
| License | Gemma | Apache 2.0 | Apache 2.0 | 多为 Apache 2.0 |
| 官方维护 | Vidore | Vidore | Vidore | **社区** |

### 5.2 ColPali / PaliGemma 细节

- 视觉：SigLIP-So400m/14 patches → 投影进 Gemma。  
- PaliGemma 特点：prefix 上对 **图像 + 指令** 做 **full-block attention**（Prefix-LM），利于「带着任务看图」。  
- 检索头：对 **每个输出 token（图或文）** 线性投到 $D=128$。  
- 页表示还常拼短提示（如 “Describe the image”）对应的若干文本向量；论文估 **float16 约 257.5 KB/页**（1024+ 量级向量 × 128 维）。

固定 patch 数 ⇒ batching 规整、VRAM 可预期；代价是分辨率–信息密度折中（512 patches 相对 1024 掉约 24.8 nDCG@5）。

### 5.3 Qwen2-VL / Qwen2.5-VL（官方 ColQwen）

Qwen2-VL 系支持 **任意分辨率**：按图动态切 patch，并用空间合并等机制控制序列长。ColQwen 训练时 **显式 cap（如 768）**，在效果与索引体积间折中。

Weaviate 概述中的流水线对全家通用：

1. PDF 页 → 图像  
2. 切 patch / 视觉 token  
3. Vision encoder → VLM 上下文化  
4. 投影到与 query 共享的检索空间  
5. 离线存多向量；在线 MaxSim  

Query 侧通常只用 **语言模型部分**（纯文本）；文档侧走完整视觉栈。

### 5.4 ColQwen3（社区）额外差异

以 Tomoro 为例（非唯一实现）：

- **Encoder-only 用法** + **320-d** 投影头，强调相对「全维多向量」的存储压缩（其文宣相对某些 3k-d 方案可达约 **13×** 体积优势）。  
- 初始化可融合 **文本 Embedding 模型** 的检索先验，再视觉微调。  
- 数据混合往往 **超出** 原始 118k ColPali train（VDR、VisRAG 等）。  
- 部分权重扩展到 **短视频帧**；仍属 ColPali-style 打分。

选型时：**官方 ColQwen2/2.5 = 可复现、与论文/ Vidore 对齐；ColQwen3 = 追榜与新骨干，需逐卡核对训练数据、维数与评测协议（ViDoRe v1 vs v2/v3）。**

### 5.5 同仓库旁支（帮助理解「换壳」）

`colpali` 还维护 ColSmol、Gemma-3 系 NetraEmbed 等——同一 MaxSim 配方、不同骨干。说明 **Col\* 品牌的本质是「VLM + 投影 + late interaction」配方**，而不是单一网络结构。

---

## 6. 训练配方：ColPali 论文如何落到 ColQwen

官方声明：ColQwen2 / 2.5 **沿用 ColPali 训练策略**，换 backbone 与分辨率超参。可复现清单如下。

### 6.1 数据

| 来源 | 规模（论文） | 说明 |
| ------ | -------------- | ------ |
| 学术 VQA 改编为页级 (q, page) | ~63% | DocVQA、InfoVQA、TAT-DQA、arXivQA 等 |
| 爬取 PDF + Claude-3 Sonnet 伪问题 | ~37% | 多领域页图 |
| **总计** | **118,695** | **全英文**；验证集约 2% |
| 发布 | `vidore/colpali_train_set` | 与 ViDoRe 测页无跨 PDF 泄漏 |

零样本法语等表现依赖骨干多语预训练，而非训练混合语。

### 6.2 优化与适配器

| 项 | 设定 |
| ---- | ------ |
| 时长 | **1 epoch**（后续官方 ColPali-v1.3 等会改 epoch / batch） |
| 精度 | bfloat16 |
| 适配 | **LoRA** $r=\alpha=32$，加在 **LM transformer** + **随机初始化投影层** |
| 优化器 | paged_adamw_8bit |
| LR | $5\times 10^{-5}$，线性衰减，warmup **2.5%** |
| Batch | 论文：**全局 32**（多卡 data parallel；单卡常 4 因序列长） |
| Query aug | +5 tokens |

不更新视觉塔时略优（解冻视觉约 −0.7 nDCG@5）；数据规模上去后结论可能变。

### 6.3 落到 ColQwen 时实践差异

| 项 | ColPali 论文主实验 | 官方 ColQwen2/2.5 README |
| ---- | -------------------- | -------------------------- |
| 骨干 | PaliGemma-3B | Qwen2-VL-2B / Qwen2.5-VL-3B |
| Patches | 1024 | **768**（训练设定） |
| Effective batch | 32 | v0.1：**32**；ColQwen2-v1.0：**256** |
| 动态分辨率 | 否（固定） | **是** |
| 代码入口 | `scripts/configs/pali/...` | `scripts/configs/qwen2/train_colqwen25_model.py` 等 |

论文已验证：**更强的生成式 VLM → 同配方下更好的视觉检索器**（Qwen2-VL → ColQwen2）。这是「换壳值得」的理论依据。

### 6.4 工程加速（训练 / 打分）

- Flash Attention 等 LLM 生态优化可直接受益。  
- `pip install "colpali-engine[lik]"`：融合 MaxSim，避免二次型显存爆炸；H100 上 ColQwen2+LoRA 可训 batch 从 64→128 量级（以官方 PR 为准）。  
- 环境变量 `COLPALI_SCORES_BACKEND`：`auto` / `torch` / `lik`。

### 6.5 负结果（避免错误换壳）

- **ColSigLIP**：仅在 SigLIP 上做 late interaction → 崩溃级分数。原因：SigLIP 对比预训练只优化 **池化后单向量**，未对齐 patch 级几何。  
- **跨塔乱拼**（SigLIP 图向量 + Gemma 文向量）微调后仍劣于原生 SigLIP。  
启示：**必须在「已经对齐到同一 LM 空间」的 VLM 输出上做多向量投影**——这正是 ColPali/ColQwen 路径成立的前提。

---

## 7. ViDoRe 分数（论文 + README 表）

### 7.1 论文主表：ColPali 相对强基线（nDCG@5）

选取 Table 2 关键行（完整 10 子集见论文）：

| 系统 | Avg nDCG@5 |
| ------ | ------------ |
| Unstructured + OCR + BGE-M3 | 66.1 |
| Unstructured + Captioning + BGE-M3 | 67.0 |
| SigLIP Vanilla | 51.4 |
| BiSigLIP | 58.6 |
| BiPali | 58.8 |
| **ColPali** | **81.3** |

视觉硬任务（InfoVQA、ArxivQA、TabFQuAD）上 late interaction 增益尤其大。

### 7.2 论文附录：ColQwen2(768) 分项

Table 7 摘录（nDCG@5）：

| | ArxivQ | DocQ | InfoQ | TabF | TATQ | Shift | AI | Energy | Gov. | Health. | **Avg** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ColPali (448/ref) | 79.1 | 54.4 | 81.8 | 83.9 | 65.8 | 73.2 | 96.2 | 91.0 | 92.7 | 94.4 | **81.3** |
| **ColQwen2 (768)** | 86.4 | 56.2 | 89.8 | 88.7 | 75.2 | 85.7 | 98.8 | 94.8 | 93.6 | 97.3 | **86.6** |

### 7.3 官方 `colpali` README 模型表（ViDoRe 🏆）

以下分数直接摘自仓库 README「List of ColVision models」（与 leaderboard 同步更新；以页面为准）：

| 模型 | Score | License | 注释 |
| ------ | ------: | --------- | ------ |
| `vidore/colpali` | 81.3 | Gemma | 论文 checkpoint；现 ❌ 默认支持 |
| `vidore/colpali-v1.1` | 81.5 | Gemma | 修 query right padding |
| `vidore/colpali-v1.2` | 83.9 | Gemma | |
| `vidore/colpali-v1.3` | 84.8 | Gemma | 有效 batch 256，3 epochs |
| `vidore/colqwen2-v0.1` | **87.3** | Apache 2.0 | 768 patches；batch 32 |
| `vidore/colqwen2-v1.0` | **89.3** | Apache 2.0 | 更大有效 batch 256 |
| `vidore/colqwen2.5-v0.1` | **88.8** | Apache 2.0 | Qwen2.5-VL-3B |
| `vidore/colqwen2.5-v0.2` | **89.4** | Apache 2.0 | 超参变体 |
| `TomoroAI/tomoro-colqwen3-embed-4b` | **90.6** | Apache 2.0 | Qwen3-VL；320-d |
| `athrael-soju/colqwen3.5-4.5B-v3` | **90.9** | Apache 2.0 | Qwen3.5-4B；320-d |
| `vidore/colSmol-256M` / `500M` | 80.1 / 82.3 | Apache 2.0 | 小模型旁支 |

**注意：** README 单列「Score」大致对应 ViDoRe 聚合；Tomoro 另报 **ViDoRe V2/V3** 多语分项，与 v1 聚合分 **不可混比**。业务选型应固定基准版本与语种子集。

### 7.4 延迟与索引吞吐（论文量级）

同一 L4 设定下（附录）：

| | Unstructured(+caption) 管线 | ColPali 页编码 |
| --- | ----------------------------- | ---------------- |
| 量级延迟 | ~**7.2 s/页**（layout+OCR+caption+encode） | ~**0.39 s/页** |
| 在线 query 编码 | 文本 bi-encoder ~22 ms | ColPali LM ~30 ms |
| MaxSim | — | 约 **1 ms / 1k 页** 量级开销（未优化引擎） |

结论：**索引路径去掉 PDF 解析是最大工程收益**；查询路径与强文本向量模型同量级，大规模库需 PLAID / fast-plaid / 向量库多向量支持。

---

## 8. 工程：多向量存储、压缩、何时用 / 不用

### 8.1 存储形态

每页存 $\mathbf{E}_{d}\in\mathbb{R}^{N_{d}\times D}$，而非单个 $\mathbb{R}^{D}$。

粗算（float16，未量化）：

$$
\mathrm{Bytes/page} \approx N_{d}\times D\times 2.
$$

例：

- ColPali 量级：$N_{d}\approx 1030$, $D=128$ → 论文 **~257.5 KB/页**。  
- 100 万页 → 原始约 **240+ GB** 量级（尚未计索引结构）。  

对比：单向量 1024-d float16 仅 **2 KB/页** 量级。多向量贵在 **条数 × 维数**。

### 8.2 压缩与系统栈

| 手段 | 思想 | 出处 / 工具 |
| ------ | ------ | ------------- |
| **降维投影** | 训练时投到 128 / 320 | ColBERT / ColPali / Tomoro |
| **Token pooling** | 层次均值合并冗余 patch（白边等） | Clavié et al.；`HierarchicalTokenPooler`；因子 3 → 向量数 **−66.7%**，性能约 **97.8%** |
| **量化 / residual** | ColBERTv2 式低比特 | Vespa 等博客可扩到大规模 ColPali |
| **专用引擎** | PLAID、fast-plaid、Vespa / Qdrant / Weaviate / Milvus 多向量 | README Community Projects |
| **融合 MaxSim 核** | 训练与打分省显存 | `late-interaction-kernels` |

Token pooling 注意：文字极密页（如 Shift）冗余少，压缩更伤效果。

### 8.3 推理 API 形态（官方）

```python
# 概念流程（ColQwen2）
image_embeddings = model(**processor.process_images(pages))   # [B, N_d, D]
query_embeddings = model(**processor.process_queries(queries)) # [B, N_q, D]
scores = processor.score_multi_vector(query_embeddings, image_embeddings)
```

大规模：batch 编码页 → CPU 列表 → `create_plaid_index` / 外部向量库 → `get_topk_plaid`。

### 8.4 何时用 ColQwen 族 vs 稠密单向量

**优先 ColPali / ColQwen 当：**

1. 语料是 **扫描件 / 复杂 PDF / 幻灯 / 报表**：表头、合并单元格、图表、多栏版式关键。  
2. OCR+chunk+BGE 管线 **贵、脆、慢**，且 caption 成本高。  
3. 需要 **页级** 召回，并可接受多向量索引成本。  
4. 需要 **对齐可解释性**（热力图验伪召回）。  
5. 许可要求 **Apache** → 倾向 ColQwen2/2.5/社区 ColQwen3，而非 Gemma 系 ColPali。

**优先稠密单向量（BGE / E5 / Qwen3-Embedding / 图文双塔等）当：**

1. 已是干净纯文本或 HTML，版式不重要。  
2. 语料极大，**存储 / QPS** 硬约束，且业务对「表/图细粒度」不敏感。  
3. 现有栈只有标准 ANN（单向量），短期无法上多向量引擎。  
4. 一阶段只要粗召回，精排再用 cross-encoder / 小 MaxSim 重排。

**常见混合：**

```text
PDF 页 ──ColQwen──► Top-K 页
              └──（可选）页内 OCR/VLM 答题
纯文本库 ──Dense──► Top-K chunk
```

或：Dense 做海选，Col\* 只对 PDF 子库建多向量索引。

### 8.5 与「MLLM 单向量嵌入」的边界

| | ColQwen 族 | Qwen3-VL-Embedding / GME 等 |
| --- | ------------ | ----------------------------- |
| 输出 | **多向量** | 常 **单向量**（last-token / 池化） |
| 交互 | MaxSim late | Bi-encoder 余弦 |
| 强项 | 视觉文档页检索 | 通用图文 / 指令嵌入、跨任务 |
| 索引 | 专用多向量 | 标准向量库 |

二者都可「用 VLM」，但 **交互范式不同**——不可只比参数量。

---

## 9. 实现检查清单（落地）

1. **命名**：对外文档写清「无 ColQwen1 论文；奠基为 ColPali；2/2.5 官方；3 社区」。  
2. **基准**：固定 ViDoRe 版本；业务集另建（发票、合同、中文扫描）。  
3. **维数与范数**：确认 checkpoint 是 128 还是 320、是否 L2；MaxSim 实现与训练一致。  
4. **分辨率 cap**：动态分辨率必须设 max patches，否则索引体积与 latency 失控。  
5. **向量库**：确认支持 multi-vector / late interaction；否则只能暴力 MaxSim。  
6. **压缩**：先 token pooling，再量化；在自有集上扫 pool_factor。  
7. **许可**：Gemma vs Apache 决定产品路径。  
8. **引用**：方法与训练 cite **Faysse et al., 2024/25, arXiv:2407.01449**；具体权重 cite 对应模型卡。

---

## 10. 小结

- **ColQwen 族 = ColBERT 的视觉文档化（ColPali）+ Qwen-VL 换骨**；共享 **MaxSim** 与 **in-batch pairwise softplus 损失**。  
- **没有 ColQwen-v1 论文**；用户口中的「1」应映射到 **ColPali**；「2」≈ **ColQwen2（+2.5）**；「3」≈ **Qwen3-VL 社区权重**。  
- 官方线在 README ViDoRe 表上从 ColPali **81.3** 升到 ColQwen2-v1.0 **89.3** / ColQwen2.5-v0.2 **89.4**；社区 ColQwen3 进一步到 **90+**，但需单独核验基准版本。  
- 工程矛盾集中在 **多向量存储**；用降维、pooling、量化与专用引擎缓解。  
- **选型**：版式敏感 PDF RAG → ColQwen；纯文本海量检索 → Dense；二者可按库拆分。

---

## 11. 参考文献与链接

1. Faysse et al. *ColPali: Efficient Document Retrieval with Vision Language Models*. [arXiv:2407.01449](https://arxiv.org/abs/2407.01449).  
2. Khattab & Zaharia. *ColBERT*. [arXiv:2004.12832](https://arxiv.org/abs/2004.12832).  
3. Wang et al. *Qwen2-VL*. [arXiv:2409.12191](https://arxiv.org/abs/2409.12191).  
4. Macé et al. *ViDoRe Benchmark V2*. [arXiv:2505.17166](https://arxiv.org/abs/2505.17166).  
5. Clavié et al. *Token Pooling*. [arXiv:2409.14683](https://arxiv.org/abs/2409.14683).  
6. [illuin-tech/colpali](https://github.com/illuin-tech/colpali) / [vidore HF](https://huggingface.co/vidore).  
7. Weaviate. *Late Interaction Retrieval Models: ColBERT, ColPali, and ColQwen*.  
8. [TomoroAI/tomoro-colqwen3-embed-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b).  
9. 本目录相关：[Embedding调研报告.md](Embedding调研报告.md)、[图文Embedding模型技术综述.md](图文Embedding模型技术综述.md)、[资料清单_论文与博客.md](资料清单_论文与博客.md)。
