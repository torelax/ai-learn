# jina-clip 系列详解：v1 → v2 双塔 CLIP 式统一检索

> v1：[Jina CLIP: Your CLIP Model Is Also Your Text Retriever](https://arxiv.org/abs/2405.20204)（Koukounas et al., arXiv:2405.20204）  
> v2：[jina-clip-v2: Multilingual Multimodal Embeddings for Text and Images](https://arxiv.org/abs/2412.08802)（Koukounas et al., arXiv:2412.08802）  
> 本文把 **双塔架构、三阶段多任务 InfoNCE、长 caption / 难负例、多语 + MRL + 渐进分辨率、ViDoRe 与「勿与 v4 / v5-omni 混淆」** 写全。

---

## 1. 一句话定位

**jina-clip** 是 Jina 的 **CLIP 式双塔**（独立文本塔 + 独立视觉塔 → 同一维稠密向量），用 **文–文 + 图–文联合对比** 让「一个模型」同时扛 **跨模态检索** 与 **纯文本检索**。

| 项 | jina-clip-v1 | jina-clip-v2 |
| --- | --- | --- |
| 论文 | arXiv:2405.20204 | arXiv:2412.08802 |
| 文本塔 | JinaBERT（AliBi） | **Jina-XLM-RoBERTa**（与 jina-embeddings-v3 同族） |
| 视觉塔 | EVA02 **B/16** | EVA02 **L/14** |
| 总参 | ~223M 量级（相对小） | **~865M**（文本 561M + 视觉 304M） |
| 语言 | **英文为主** | **多语**（约 89 语；训练语料 ~30 语） |
| 图分辨率 | 固定 **224×224** | **224 → 384 → 512** 渐进 |
| 嵌入维 | 固定 | **1024**，支持 **MRL 截断**（至 256 几乎无损） |
| 文档图 | 弱 | **ViDoRe** 大幅提升（含 PDF/图表/信息图数据） |

核心卖点：**系统不必再为「纯文本」和「图文」各备一套索引与模型**——在双塔 CLIP 路线上，把文本塔训到接近专用文本 embedding。

---

## 2. 先分清：clip / v4 / v5-omni 不是同一条路

产品名都带「jina」与「多模态」，**架构与训练目标差得很远**。选型前务必对齐：

| 家族 | 架构范式 | 模态 | 文本侧是否「被联合训烂」 | 典型用途 |
| --- | --- | --- | --- | --- |
| **jina-clip-v1/v2** | **双塔 CLIP**（两塔各自编码，对比对齐） | 文本 + 图像 | 用 **文–文 InfoNCE / 难负例** 显式保住文本检索 | 图搜文 / 文搜图 + 可用的文本检索；v2 还做多语与文档图 |
| **jina-embeddings-v4** | 更偏 **统一多模态 embedding**（VLM 风格演进线，见官方 tech report） | 多模态检索 | 与 clip **不是**同一套双塔配方 | 更大、更「万能」的多模态检索（参数量级远高于 clip-v2） |
| **jina-embeddings-v5-text** | **纯文本** LLM/编码器 + 蒸馏 + **任务 LoRA** | 仅文本 | — | 文本检索 / STS / 聚类 / 分类 |
| **jina-embeddings-v5-omni（GELATO）** | **冻结文本塔** + 冻结 ViT/音频塔 + **可训投影器**（VLM 式拼序列） | 文本 / 图 / 视频 / 音频 | **文本权重 bit-identical 于 v5-text** | 在不改坏文本索引的前提下挂上非文本模态 |

记忆口诀：

```text
jina-clip     ≈  经典双塔 CLIP，两塔一起训，输出各 1 个向量
v5-omni       ≈  冻结一切大塔，只训投影；文本路径与 v5-text 位级相同
v4            ≠  clip；不要把「jina 多模态」默认成同一 checkpoint
```

**modality gap**（Liang et al.）：双塔对比学容易让不同模态占据嵌入空间中分离的锥体。clip 系列用 **多任务双损失**（文–文与图–文分支）缓解「只会图文、不会文本」；v2 分析里也指出 **统一 batch 单温度** 并不优于分任务双损失（见 §8）。v5-omni 则换路线：让非文本特征进入 **同一语言模型序列空间**，几何上更易交织。

---

## 3. 问题背景：为何「普通 CLIP」当不了文本检索器

经典 CLIP（Radford et al., 2021）用短 caption 做图–文对比：

1. **有效上下文极短**：后续工作测过 OpenAI CLIP 有效文本长度常 **&lt; 20 token**；长文检索几乎不可用。
2. **训练目标只有图文**：没有 query–passage、难负例、STS 等级标注等文本检索配方。
3. **实证**：OpenAI CLIP ViT-B/16 在 MTEB 检索上远弱于 jina-embeddings-v2；即便 EVA-CLIP / LongCLIP 改善了图文或长文，**纯文本检索仍远落后专用文本模型**。

工程后果：RAG / 多模态检索系统常 **双模型双索引**（文本一套、图文一套），成本与一致性都差。jina-clip 的回答是：**同一双塔、同一嵌入空间，联合优化两种配对**。

---

## 4. 架构

### 4.1 双塔与池化

两塔输出同维向量，相似度用余弦。推理时：

- 文本 → 文本塔 → 向量 $\mathbf{t}$
- 图像 → 视觉塔 → 向量 $\mathbf{i}$
- 检索：$\mathrm{cos}(\mathbf{q}, \mathbf{d})$

### 4.2 v1：JinaBERT + EVA02-B

| 组件 | 选择 | 理由（论文） |
| --- | --- | --- |
| 文本 | **JinaBERT** + AliBi；先做 **MLM** 预训练再对比 | 比「已训好的文本 embedding 再继续」终局更好 |
| 视觉 | **EVA02 ViT-B/16**，EVA 预训练初始化 | 同规模下优于 DinoV2、OpenCLIP ViT-B/16 |
| 输出维 | 与双塔对齐的固定维（训练框架沿 OpenCLIP） | — |
| 输入图 | **224×224**，patch 16×16 | 大 batch |

### 4.3 v2：Jina-XLM-RoBERTa + EVA02-L14

| Feature | Text Encoder | Image Encoder |
| --- | --- | --- |
| Base | Jina-XLM-RoBERTa（源自 XLM-R，FlashAttn2 + RoPE + LoRA 能力） | EVA02 **L/14** |
| Params | **561M** | **304M** |
| 输入 | 最长 **8192** token（训练阶段见下） | 终局 **512×512** |
| 输出 | **1024** | **1024** |
| Pooling | **Mean** | **CLS** |
| 语言 | **89 languages**（声明） | patch **14×14** |

v2 文本塔与 **jina-embeddings-v3** 同族初始化，这是多语文本能力的底座；但 **对比联合训练后，纯文本检索仍略弱于专用 jina-embeddings-v3**（论文明确写出 trade-off）。

---

## 5. 三阶段多任务训练（v1 / v2 共用骨架）

```text
Stage 1  短文 + 短 caption 图文对     对齐模态，保文本下限
Stage 2  长文 + 长 caption           拉长有效上下文
Stage 3  文本难负例三元组 + 长 caption  拉齐文本检索，稳住图文
```

每一阶段都是 **两个 InfoNCE 之和**（文–文一支 + 图–文一支），而不是把图和文混进同一个对比矩阵（v2 消融证明 unified batch 更差，见 §8）。

### 5.1 数据符号

| 符号 | 含义 | v1 | v2 |
| --- | --- | --- | --- |
| $\mathbb{D}^{\mathrm{txt;p}}$ / $\mathbb{C}^{\mathrm{text}}_{\mathrm{pairs}}$ | 文本对 | ~40 个文本对数据集 | jina-embeddings-v3 多语文本对（~30 语） |
| $\mathbb{D}^{\mathrm{txt;t}}$ / $\mathbb{C}^{\mathrm{text}}_{\mathrm{triplets}}$ | 1 正 + **7 难负** | MSMarco / NQ / HotpotQA / NLI | v3 同源高质量难负例（侧重 Retrieval / STS） |
| $\mathbb{D}^{\mathrm{mm;s}}$ | 短 caption 图文 | **LAION-400M** | **DFN ~400M 英** + CommonPool 过滤 **~400M 非英** + 文档类数据 |
| $\mathbb{D}^{\mathrm{mm;l}}$ | 长 caption 图文 | **ShareGPT4V** ~1.2M | 长 caption 子集 + **GPT-4V 多语长描述 ~1.2M** + DocVQA / InfographicsVQA / SciGraphQA / ArXivQA / WIT / ViDoRe synthetic 等 |

v2 对 CommonPool：按语言与长宽比过滤、caption 至少 5 词，再用 **multilingual SigLIP** 按相似度保留约 top 50%。QA 类数据集把 **query∥answer** 拼成「伪 caption」；WIT 短侧用 reference caption，长侧拼标题与章节描述。

### 5.2 损失：双向 InfoNCE 与难负例扩展

对 batch $\mathbf{B}$ 中的配对 $(\mathbf{q},\mathbf{p})$，温度 $\tau$：

$$
\mathcal{L}_{\mathrm{nce}}(\mathbf{B})
:=
\mathcal{L}_{\mathrm{nce}}^{\rightarrow}(\mathbf{B})
+
\mathcal{L}_{\mathrm{nce}}^{\leftarrow}(\mathbf{B})
$$

$$
\mathcal{L}_{\mathrm{nce}}^{\rightarrow}(\mathbf{B})
=
\mathbb{E}_{(\mathbf{q},\mathbf{p})\sim\mathbf{B}}
\left[
-\ln
\frac{e^{\mathrm{cos}(\mathbf{q},\mathbf{p})/\tau}}
{\sum_{i=1}^{k} e^{\mathrm{cos}(\mathbf{q},\mathbf{p}_{i})/\tau}}
\right]
$$

对称方向把 query / target 对调。v1：**文本对 $\tau=0.05$ 固定**；**图文对 $\tau$ 可学习**（OpenCLIP 默认）。

Stage 3 文本支用扩展损失（每 query 附 7 个难负 $\mathbf{n}_{1..7}$）：

$$
\mathcal{L}_{\mathrm{nce}^{+}}(\mathbf{B})
=
\mathbb{E}_{r\sim\mathbf{B}}
\Bigg[
-\ln
\frac{e^{\mathrm{cos}(\mathbf{q},\mathbf{p})/\tau}}
{\sum_{i}\Big(
e^{\mathrm{cos}(\mathbf{q},\mathbf{p}_{i})/\tau}
+
\sum_{j=1}^{7}e^{\mathrm{cos}(\mathbf{q},\mathbf{n}_{j,i})/\tau}
\Big)}
\Bigg]
+
\text{（passage→query 对称项，仅 in-batch）}
$$

三阶段联合目标（v1 / v2 同构）：

$$
\begin{aligned}
\mathcal{L}_{1}
&=
\mathcal{L}_{\mathrm{nce}}(\mathbf{B}_{\mathrm{txt;p}})
+
\mathcal{L}_{\mathrm{nce}}(\mathbf{B}_{\mathrm{mm;s}})
\\
\mathcal{L}_{2}
&=
\mathcal{L}_{\mathrm{nce}}(\mathbf{B}_{\mathrm{txt;p}})
+
\mathcal{L}_{\mathrm{nce}}(\mathbf{B}_{\mathrm{mm;l}})
\\
\mathcal{L}_{3}
&=
\mathcal{L}_{\mathrm{nce}^{+}}(\mathbf{B}_{\mathrm{txt;t}})
+
\mathcal{L}_{\mathrm{nce}}(\mathbf{B}_{\mathrm{mm;l}})
\end{aligned}
$$

### 5.3 v1 训练超参（附录 Table 2）

| Parameter | Stage 1 | Stage 2 | Stage 3 |
| --- | --- | --- | --- |
| Peak LR | $1\times10^{-4}$ | $5\times10^{-6}$ | $1\times10^{-6}$ |
| Batch（图文 / 文本） | **32768** | **8192** | **1024** |
| Max seq | **77** | **512** | **512** |
| Steps | 60000 | 1500 | 7000 |
| 样本规模（量级） | 各 ~2B | 各 ~12M | 各 ~7M |
| GPU | 8×H100 | 同 | 同 |
| 墙钟 | ~180h | ~3h | ~4.5h |

为何单独 Stage 2：长 caption 数据量远小于 LAION；若塞进 Stage 1 会被淹没；若只放 Stage 3 又会削弱难负例阶段对文本的精细打磨。

### 5.4 v2 额外：渐进分辨率 + MRL

**分辨率日程**：

1. Stage 1：先 **224×224** 冲大 batch；末段插值到 **384×384** 热身  
2. Stage 2：保持 **384**，文本上下文 **77→512**  
3. Stage 3：再插值到 **512×512**，并吃难负例  

**Matryoshka Representation Learning**：每个损失在全维 **1024** 以及截断维 **\{64,128,256,512,768\}** 上各算一遍再求和。推理时可截断向量降存储；**1024→256（减 75%）** 时跨模态与文本任务退化通常 **&lt;1%**。

---

## 6. 评估要点

### 6.1 v1：CLIP Benchmark + MTEB

论文 Table 1（节选）：

| Model | txt→img R@5 | img→txt R@5 | MTEB Retr nDCG@10 | MTEB Avg |
| --- | --- | --- | --- | --- |
| OpenAI CLIP B/16 | 75.62 | 88.12 | 17.63 | 43.95 |
| EVA-CLIP B/16 | 82.15 | 90.59 | 26.03 | 47.64 |
| LongCLIP B/16 | 81.72 | 90.79 | 28.76 | 47.71 |
| jina-embeddings-v2 | — | — | 47.85 | 60.38 |
| jina-clip-v1 **final** | **80.31** | **89.91** | **48.33** | **60.12** |

解读：跨模态与 EVA-CLIP 同档；**文本侧已接近 jina-embeddings-v2**。Stage 1→2 抬图文；Stage 3 显著抬文本检索（Retr nDCG 约 40→48），图文略回落——典型的 **模态–任务权衡**。

### 6.2 v2：多语跨模态 + 文本 + ViDoRe

**英文 CLIP Benchmark**（Flickr30K + COCO 汇总，论文 Table 2）：

| Model | T→I R@5 | I→T R@5 |
| --- | --- | --- |
| nllb-siglip-large | **81.54** | 88.15 |
| jina-clip-v1 | 77.75 | 87.65 |
| jina-clip-v2 | 79.09 | **89.73** |

**Crossmodal-3600（多语）**：v2 T→I / I→T ≈ **81.4 / 83.2**，逼近 nllb-siglip-large；相对英文-only 的 v1，非英语几乎从「不可用」拉到可用（v1 在多数非英语上接近 0）。

**MTEB Retrieval / STS**（Table 3）：

| Model | EN Retr | Multi Retr | EN STS | Multi STS |
| --- | --- | --- | --- | --- |
| nllb-siglip-large | 24.91 | — | 74.89 | — |
| jina-embeddings-v3 | **53.87** | **72.58** | **85.80** | **69.81** |
| jina-clip-v1 | 48.33 | — | 80.92 | — |
| jina-clip-v2 | 49.32 | 69.85 | 81.29 | 67.77 |

结论：**远强于「只会 CLIP」的多语模型**；相对 **专用文本 v3** 仍有缺口——多模态联合训练会拖累纯文本上限。

**ViDoRe（视觉文档）** nDCG@5 平均：

| Model | Avg |
| --- | --- |
| jina-clip-v1 | 17.72 |
| nllb-siglip-large | 46.55 |
| jina-clip-v2 stage1 / 2 / final | 37.37 / 47.76 / **52.65** |

相对 v1 **约 +35 个百分点**；文档数据 + 分辨率日程是主因。

**MRL**：256 维时 CLIP / Crossmodal / 文本 Retr / STS 几乎持平 1024；128 / 64 才明显掉。

---

## 7. 分辨率消融（v2 §5.1）

在 Stage 1 末（只见过 224）的 checkpoint 上，用同一套视觉富文档数据再训 3500 step，分辨率分别为 224 / 384 / 512 / 768：

| 分辨率 | ViDoRe 均 nDCG@5（量级） |
| --- | --- |
| 224 | ~0.256 |
| 384 | ~0.454（跃迁最大） |
| 512 | 再明显抬一截（论文选为甜点） |
| 768 | 相对 512 仅 **+0.019**，但 patch 数约 **×2.25**（patch=14） |

**工程建议**：文档检索优先保证 **≥384**，生产默认 **512**；盲目上 768 性价比差。

---

## 8. Unified Batch vs Multi-Task（v2 分析）

动机：把图–文与文–文混进 **同一 InfoNCE**，共享 $\tau$，并加 SimCLR 式图像自监督，指望跨模态 in-batch 负例缩小 modality gap。

结果：早期略好，后期跨模态落后；可学习 $\tau$ 卡在 ~0.02，而分任务训练里 $\tau$ 可持续降到 ~0.015。

假设：**图像信息密度 ≫ caption**，两种配对的「难负例硬度」不同，**强制同一温度次优**。因此 v1/v2 坚持 **双损失多任务**，而不是「一个大统一对比」。

---

## 9. v1 → v2 演进清单

| 维度 | 变化 |
| --- | --- |
| 语言 | 英 → **多语文本塔 + 多语图文 / 文本对** |
| 视觉容量 | B/16 → **L/14**；224 → **渐进 512** |
| 文档 | 几乎无 → **ViDoRe 向数据 + 分辨率** |
| 向量成本 | 固定维 → **MRL 可截到 256** |
| 文本上限 | 接近 jina-emb-v2 → 多语接近 v3，但 **仍低于专用文本塔** |
| 局限残留 | 异构模态混合候选池上的 **modality gap / bias** 未彻底解决（相关工作指向 MM-Embed 等 LLM 检索器） |

---

## 10. 选型与落地建议

1. **只要英文图文 + 还要像样的文本检索**：v1 已够用；要多语 / PDF 页图 → **v2**。  
2. **纯文本 SOTA、长上下文、任务适配**：用 **v5-text**（或历史 v3），不要用 clip 文本塔硬顶。  
3. **文本索引已上线、还要挂图/音视频且不能漂移文本向量**：**v5-omni（GELATO）**，不是 clip。  
4. **存储紧**：v2 优先试 **256–512 维 MRL 截断**。  
5. **文档页检索**：clip-v2 是双塔单向量基线；若要 late interaction 极致细粒度，另看 ColPali / ColQwen（本目录专题）。

---

## 11. 公式与实现核对清单

- [ ] 训练是否 **同时** 有文–文与图–文 batch（而非只拉 caption）  
- [ ] Stage 3 是否真的喂了 **7 难负** 的 $\mathcal{L}_{\mathrm{nce}^{+}}$  
- [ ] v2 推理分辨率是否与训练终局一致（**512**）  
- [ ] MRL 截断维是否在 **训练见过的集合** 内（64…1024）  
- [ ] 评估时文本池化：**v2 mean**；图像：**CLS**——与训练一致  
- [ ] 勿把 clip checkpoint 与 **v5-text / v5-omni** 混装进同一「版本」叙事

---

## 12. 小结

**jina-clip** 证明：在 **双塔 CLIP** 上做 **三阶段、双任务 InfoNCE**，可以把文本检索拉到接近专用模型，同时保持强图文检索。**v2** 在此骨架上补齐 **多语、MRL、渐进高分辨率与视觉文档数据**，ViDoRe 相对 v1 质变；但 **纯文本仍略逊 jina-embeddings-v3**，且 **与 v4 / v5-omni 不是同一架构族**——选型时按「双塔对比 vs 冻结塔投影 vs 专用文本蒸馏」三条线分离即可。
