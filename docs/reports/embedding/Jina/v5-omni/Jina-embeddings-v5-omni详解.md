# Jina-embeddings-v5-omni 详解：GELATO 锁定对齐塔

> 基于论文 [jina-embeddings-v5-omni: Geometry-preserving Embeddings via Locked Aligned Towers](https://arxiv.org/abs/2605.08384)（Hönicke, Günther, Koukounas, Akram, Martens, Sturua, Xiao；arXiv:2605.08384）。  
> 方法名 **GELATO** = **Ge**ometry-preserving **E**mbeddings via **L**ocked **A**ligned **TO**wers。  
> 本文把 **冻结文本塔（与 v5-text bit-identical）、冻结视觉/音频塔、只训投影与分隔符、序列构造、Matryoshka InfoNCE、评测与消融** 写全。

---

## 1. 一句话定位

**jina-embeddings-v5-omni** 把 **已训好的 v5-text** 扩成 **文本 / 图像 / 视频 / 音频** 统一嵌入：**所有大塔冻结**，只训练 **模态投影器 + 少量分隔符 embedding**（约 **总权重的 0.35%**），且 **纯文本路径与 v5-text 位级相同**。

| 项 | omni-nano | omni-small |
| --- | --- | --- |
| 文本骨干 | v5-text-**nano**（~0.24B 文本侧） | v5-text-**small**（~0.67B 文本侧） |
| 全模态加载参数 | **~0.95B** | **~1.57B** |
| 视觉塔 | Qwen3.5-**0.8B** 视觉（SigLIP2 Base 适配） | Qwen3.5-**2B** 视觉（SigLIP2 So400m 适配） |
| 音频塔 | Qwen2.5-Omni 音频（Whisper-large-v3 适配） | 同 |
| 文本隐维 $d_{\mathrm{text}}$ | **768** | **1024** |
| 可训比例 | 极低（投影 + delimiter） | 同 |
| 任务变体 | retrieval / classification / clustering / text-matching × 规模 | 同（共 8 个任务导出 + base） |

相对 **jina-clip**：不是双塔一起对比训；相对 **全参 VLM embedding（GME、Qwen3-VL-Embedding、LCO-Omni）**：不改语言模型权重，**文本索引可原地复用**。

---

## 2. 为何需要 Locked Aligned Towers

多模态检索的三条历史路线：

| 路线 | 代表 | 代价 |
| --- | --- | --- |
| 双塔 CLIP 对比 | CLIP、SigLIP、jina-clip | 易有 **modality gap**；文本侧常弱 |
| 全参 / 半参 VLM 改写 LM | E5-V、GME、LCO-Omni | 强，但 **文本几何漂移**，旧向量库失效 |
| 冻结文本 + 可训视觉插件 | LiT、VISTA、MARVEL | 多停在图文；少见 **图+视频+音频且编码器也冻** |

GELATO 主张：

1. 选用 **已经过语言对齐** 的 Qwen 视觉 / 音频编码器（而非裸 SigLIP / Whisper）  
2. **冻住** 文本 embedding 塔与模态编码器  
3. 只学 **进 LM 隐空间的桥**（线性 / 浅 MLP 投影 + 特殊 token）  

这样：**训练便宜、文本 bit-identical、模块可插拔（modality 属性可只加载 text / vision / audio / omni）**。

---

## 3. 架构总览

```text
[冻结] 文本 tokenizer → 文本 Transformer + 任务 LoRA → last-token pool
[冻结] ViT → LayerNorm + 2×2 merge + fc_vision_1 → [可训] fc_vision_2 → 填入 image/video pad
[冻结] Audio encoder → [可训] fc_audio → 填入 audio pad
         ↓
   拼成单一 token 序列进冻结文本塔
         ↓
   任务 LoRA + last-token + L2
```

Figure 2 要点：冻结塔 → 可训投影 → 冻结文本骨干；**任务导出**时同时切换 **LoRA + 对应投影/分隔符套件**。

### 3.1 仿射层记号

$$
\ell_{W,\mathbf{b}}(\mathbf{x})=W\mathbf{x}+\mathbf{b}
$$

- `fc_vision_1` / `fc_vision_2` / `fc_audio` 均为该类映射。

### 3.2 视觉投影

Qwen3.5 视觉塔输出 patch 后：

1. **LayerNorm**（冻）  
2. **$2\times2$ spatial merge**（pixel-unshuffle：四邻 patch 拼成 $4d_{\mathrm{vit}}$，空间 token **÷4**）  
3. `fc_vision_1` + **GELU**（冻）  
4. **`fc_vision_2`（可训，随机初始化）** → $d_{\mathrm{text}}$

对四邻组 $\mathbf{V}_{i}=[\mathbf{v}_{i,1},\ldots,\mathbf{v}_{i,4}]$：

$$
\mathbf{m}^{(i)}_{\mathrm{vis}}
=
\bigl[\mathrm{LN}(\mathbf{v}_{i,1});\ldots;\mathrm{LN}(\mathbf{v}_{i,4})\bigr]
\in\mathbb{R}^{4d_{\mathrm{vit}}}
$$

$$
\mathbf{z}^{(i)}_{\mathrm{vis}}
=
\mathrm{GELU}\bigl(\ell_{W_{\mathrm{v1}},\mathbf{b}_{\mathrm{v1}}}(\mathbf{m}^{(i)}_{\mathrm{vis}})\bigr)
$$

$$
\mathbf{h}^{(i)}_{\mathrm{vis}}
=
\ell_{W_{\mathrm{v2}},\mathbf{b}_{\mathrm{v2}}}(\mathbf{z}^{(i)}_{\mathrm{vis}})
$$

源 checkpoint 里 `fc_vision_2` 映射到 Qwen 文本隐维（2B：$4096\to2048$；0.8B：$3072\to1024$），与 Jina 的 1024 / 768 **不一致**，故 **替换并重训** `fc_vision_2`：

| 规模 | 新 `fc_vision_2` |
| --- | --- |
| Small | $4096\to1024$ |
| Nano | $3072\to768$ |

### 3.3 音频投影

音频编码器状态 $\mathbf{A}=[\mathbf{a}_{1},\ldots,\mathbf{a}_{K}]\in\mathbb{R}^{K\times1280}$（冻）：

$$
\mathbf{h}^{(i)}_{\mathrm{aud}}
=
\ell_{W_{\mathrm{aud}},\mathbf{b}_{\mathrm{aud}}}(\mathbf{a}_{i})
,\qquad
W_{\mathrm{aud}}\in\mathbb{R}^{d_{\mathrm{text}}\times1280}
$$

`fc_audio` **全程随机初始化并训练**。

### 3.4 序列构造（进文本塔）

图像：

$$
\texttt{<|vision\_start|>}\;
\underbrace{\texttt{<|image\_pad|>}\times N}_{\text{视觉槽}}\;
\texttt{<|vision\_end|>}
$$

音频：

$$
\texttt{<|audio\_start|>}\;
\underbrace{\texttt{<|audio\_pad|>}\times K}_{\text{音频槽}}\;
\texttt{<|audio\_end|>}
$$

视频：按采样帧拼接视觉段（用 `video_pad`）；若有音轨则 **音频段在前**：

$$
\mathbf{s}_{\mathrm{aud}}\,\|\,\mathbf{s}_{\mathrm{vid}}
$$

混合文档：文本跨度与模态段按文档顺序拼接。pad 位置 **被投影特征覆盖**，不作为独立可训词向量；**start/end delimiter** 可学（small：视觉+音频 delim；nano：论文称主要学音频 delim）。

### 3.5 可训参数集合

| 可训 | 冻结 |
| --- | --- |
| `fc_vision_2` | 文本 Transformer、任务 LoRA（继承）、ViT、音频塔 |
| `fc_audio` | LayerNorm、`fc_vision_1` |
| 模态 start/end embedding | pad 槽本身 |

**按任务分别训投影**（retrieval / classification / clustering / text-matching），与继承的任务 LoRA 对齐；再按模态拆成 vision run / audio run → 合计 **$2\times4\times2=16$** 次投影训练（两规模）。

### 3.6 动态加载

- **task**：决定 LoRA + 哪套 `fc_vision_2` / `fc_audio` / delimiter  
- **modality**：`text-only` / `vision-only` / `audio-only` / `omni`，省略不需要的塔以省显存  

纯文本请求：路径与 **v5-text 完全一致**（权重 bit-identical）。

---

## 4. 训练目标

双向 in-batch InfoNCE + **Matryoshka** 多前缀维。温度 $\tau=0.02$。对配对 $\{(\ell_{i},r_{i})\}$，前缀维 $k$：

$$
s_{ij}^{(k)}=\frac{\mathrm{cos}(\mathbf{u}_{i,1:k},\mathbf{v}_{j,1:k})}{\tau}
$$

$$
p_{\ell\to r}^{(k)}(j|i)
=
\frac{\exp(s_{ij}^{(k)})}{\sum_{m}\exp(s_{im}^{(k)})}
$$

$$
\mathcal{L}_{\mathrm{NCE}}^{(k)}
=
-\frac{1}{2B}
\sum_{i}
\Bigl[
\log p_{\ell\to r}^{(k)}(i|i)
+
\log p_{r\to\ell}^{(k)}(i|i)
\Bigr]
$$

$$
\mathcal{L}
=
\sum_{k\in\mathcal{K}}
\mathcal{L}_{\mathrm{NCE}}^{(k)}
$$

论文给出的前缀集合（OCR 稿易吞逗号；按 MRL 惯例理解为由小到大的截断维，small 含至满维 **1024**，nano 至 **768**）：

$$
\mathcal{K}_{\mathrm{Small}}\supset\{32,64,128,256,512,768,1024\}
,\quad
\mathcal{K}_{\mathrm{Nano}}\supset\{32,64,128,256,512,768\}
$$

（实现时以官方配置为准；要点是 **多前缀求和**，使截断仍可用。）

优化：AdamW（$\beta_{1}=0.9,\beta_{2}=0.999$，wd $0.01$），grad clip $\|\nabla\|_{2}\le1$，LR $2\times10^{-4}$，500 step warmup，bf16，**4×H100**，全局 batch **256**，每 run **15,000** step。视觉与音频 **分开训**；每 step 按混合权重抽单一源数据集。

数据混合强调 **企业文档图**：扫描件、图表、OCR 页、医疗影像、商品图、UI；音频侧重音乐、环境音、英 / 多语语音等（论文 Figure 3 token share）。

---

## 5. 评测

### 5.1 四模态总分（Table 1）

| Model | Params (B) | Text | Image | Video | Audio | Avg |
| --- | --- | --- | --- | --- | --- | --- |
| omni-**nano** | 0.95 | 65.52 | 47.87 | 26.87 | 49.69 | 47.49 |
| LanguageBind | 1.14 | 27.34 | 47.80 | 48.06 | 20.08 | 35.82 |
| omni-**small** | 1.57 | **67.00** | 58.00 | 41.20 | 49.96 | **54.04** |
| Omni-Embed-Nemotron-3B | 4.70 | 47.64 | 44.47 | 24.46 | 48.27 | 41.21 |
| LCO-Omni-3B | 4.70 | 57.55 | 58.42 | 46.84 | 52.51 | 53.83 |
| LCO-Omni-7B | 8.93 | 59.31 | 58.64 | **47.41** | **52.37** | **54.43** |

解读：

- **Text**：直接继承 v5-text 公开 MMTEB，**碾压**同场 omni 基线  
- **Avg**：small 在 **&lt;2B** 开源 omni 中极强，逼近 7B LCO  
- **Video**：明显短板（帧聚合进单向量信息挤兑）  
- **Audio**：接近但未超 LCO  

### 5.2 视觉文档（ViDoRe-in-MIEB）

| Model | 活跃参（文+图路径） | Doc retrieval |
| --- | --- | --- |
| omni-nano | **0.31B** | **79.25** |
| omni-small | 0.92B | **79.25** |
| LCO-3B | 4.07B | 78.24 |
| LCO-7B | 8.93B | 80.32 |
| Nemotron-3B | 4.70B | **85.64** |

nano 以约 **1/10** LCO-3B 活跃参数打平 / 略超 LCO-3B——**文档页检索是 GELATO 的强项**（训练混合偏 OCR / 图表）。

### 5.3 分项强弱（Table 4 摘要）

**相对强**：图像分类 / 聚类、Visual STS、多语图像检索、文档检索、音频分类与 text-matching。  

**相对弱**：通用图像检索（相对 LCO）、**MMEB-Video 多数子任务**、**音频聚类**（6.1 vs CLAP 专家 22.7）。

### 5.4 模态几何（§5.2）

在 MS-COCO Karpathy 与 Clotho 上量：

| 指标 | 含义 |
| --- | --- |
| Centroid $L_{2}$ | 模态均值距离（gap） |
| Lift | 配对余弦 − 随机对余弦 |
| R@1 ↔ | 跨模态检索 |

结论要点：

- **图–文**：small R@1 68.0 / 57.0，接近 LCO-3B（71.6 / 58.0），参数仅其约 **42%**  
- **音–文**：落后 LCO 约 **11–15** 个点——**音频桥更弱**  
- UMAP：LanguageBind 呈经典分簇 gap；**GELATO 与 LCO 呈模态交织**——统一解码器几何，而非四锥分离  

冻结塔 **不能** 像联合对比那样把 gap 压到极限，但交织几何说明投影进 LM 空间是有效的。

---

## 6. 消融

### 6.1 视觉：只训 `fc_vision_2` 够不够？

五配置（CIRR / NIGHTS 快评）：

| ID | 可训范围 | 结果 |
| --- | --- | --- |
| I | **仅 fc_vision_2** | **足够（基线）** |
| II | fc_vision_1+2 | 略差 |
| III | + 解冻 ViT（从零） | **崩溃**（随机投影反传毁掉编码器） |
| IV/V | I 后再扩训 | 几乎无增益（+0.001 级） |

发布配方锁定 **I**：简单、稳、产物少。

### 6.2 音频：仅 `fc_audio` vs 解冻编码器

| ID | 结果 |
| --- | --- |
| I | 仅投影，达标 |
| II | 同时解冻编码器，更差 |
| III | 先 I 再解冻编码器 | **+0.022**（有潜力，未进发布以控复杂度） |

与 §5.2 音–文落差、MAEB 聚类偏弱一致：**线性音频桥丢了不少模态内方差**；后续可考虑预算内的音频塔续训。

### 6.3 Matryoshka 跨模态

截断曲线：

- **文本 ≈ 图像**：几乎重合 → `fc_vision_2` **继承**了文本塔的前缀结构  
- **音频**：256 维仍大部分保持  
- **视频**：小维掉得更狠——多帧挤进单一向量，前缀容量先耗尽  

### 6.4 训练效率（Table 5）

相对「全参更新」同一 15k step 预算：

| | Projector vs Full |
| --- | --- |
| Vision | ~**1.8×** 更快，显存更低 |
| Audio | ~**3.2–3.9×** 更快 |

small 视觉可训约 **4.2M** vs 全量 ~920M；音频 ~1.3M vs ~1.2B。

---

## 7. 与 jina-clip / v5-text / v4 对照

| | jina-clip-v2 | v5-text | **v5-omni** |
| --- | --- | --- | --- |
| 范式 | 双塔对比 | 文本蒸馏+LoRA | **锁定塔 + 投影** |
| 文本权重 | 联合训练产出 | 权威文本模型 | **≡ v5-text（bit）** |
| 视频 / 音频 | 无 | 无 | **有** |
| 改文本索引 | 需重嵌 | — | **不需要** |
| 可训成本 | 高（双塔全开） | 中（蒸馏+LoRA） | **极低（投影）** |
| 模态 gap | 双塔经典问题 | — | 交织改善但仍存 |

**v4** 是另一条更大的多模态 embedding 线，**不是**「omni 的教师」也不是 clip 的别名；omni 的教师式依赖是 **v5-text + Qwen 感知塔**。

---

## 8. 选型建议

1. **已有 v5-text 索引，要加图/音/视频**：直接上对应规模 **omni**，文本无需重算。  
2. **只要图文、要双塔经典部署、可接受重嵌文本**：clip-v2 仍可；但无音频、无 bit-identical 承诺。  
3. **文档页 RAG**：优先看 omni 的 **document retrieval**；更细粒度再比 ColPali/ColQwen。  
4. **视频为主**：当前 omni **偏弱**，需降期望或等后续。  
5. **音频聚类 / 细粒度音乐**：考虑 CLAP 等专家，或等待「解冻音频塔」配方。  
6. **显存**：用 `modality=` 只加载需要的塔；任务切换加载对应投影包。

---

## 9. 实现核对清单

- [ ] 文本路径权重是否与 **同规模 v5-text** 字节一致（应用层勿再微凋文本塔）  
- [ ] 任务切换是否 **同时** 换 LoRA 与投影/delimiter  
- [ ] 视觉：确认只替换 **`fc_vision_2`**，未误训 ViT  
- [ ] 视频：帧采样与是否前置音轨与训练一致  
- [ ] MRL 截断：视频任务勿激进砍到 32 维  
- [ ] 评估协议：几何实验用 **retrieval** 变体，勿混 classification LoRA  

---

## 10. 小结

**GELATO** 把多模态 embedding 做成「**插件**」：锁住已对齐的感知塔与 **位级不变的文本塔**，只学进 LM 空间的桥。v5-omni-small 在 **&lt;2B** 开源全模态模型中综合分接近数倍参数的 LCO-7B，**文本与视觉文档**突出，**视频与音频细粒度**仍是短板。消融支持发布配方——**冻塔 + 单层投影**——作为默认；音频编码器二阶段续训是明确的未来增益点。对工程而言，最大价值或许是：**多模态能力不再以摧毁文本向量库为代价**。
