# LLM-DA：基于大模型的文本行人检索数据增强

> paper: [Data Augmentation for Text-based Person Retrieval Using Large Language Models](https://arxiv.org/abs/2405.11971)（arXiv:2405.11971）  
> HTML: https://arxiv.org/html/2405.11971v1  
> backbone: Vicuna（改写）+ Sentence-Transformers（忠实度）+ CLIP（检索学生）  
> date: 2024-05  
> modality: 图文（行人图像 ↔ 文本描述）  
> languages: 英文 TPR 基准  

> 本文把 **LLM 改写增强、Text Faithfulness Filter、Balanced Sampling、与 EDA/回译对照** 写全。对通用 embedding 的迁移点：正例文本侧扩写 + 语义过滤 + 原/增采样比——与难负例挖掘互补（一正侧、一负侧）。

---

## 一句话定位

**LLM-DA** 面向 **Text-based Person Retrieval（TPR）**：用 LLM **改写**行人描述以扩充训练文本；用 **TFF** 滤掉幻觉改写；用 **BSS** 控制原句与增广句进 batch 的比例。**不改模型结构与损失形式**，可插拔到 CLIP 等双塔。

| 项 | 内容 |
| --- | --- |
| 痛点 | 行人图隐私限制 + 标注贵 → 文本短、多样性差 |
| 传统文本增强 | 随机删/换、回译：易毁句法或增益弱 |
| LLM-DA | 改写保语义 + 扩词句结构 |
| 默认超参（文中） | 忠实度阈值 $\alpha\approx 0.6$；增广采样比 $\beta\approx 0.2$ |

---

## 任务与基线框架

给定图集 $\mathcal{V}=\{V_i\}$、描述 $\mathcal{T}=\{T_i\}$，TPR 用文本查行人图。双塔：

$$
s(V_i,T_i)=\mathrm{sim}\big(f_{\mathrm{img}}(V_i),\,f_{\mathrm{text}}(T_i)\big)
$$

论文以 **CLIP**（ViT-B/32、B/16）为干净基线，刻意不加 TPR 专用对齐模块，以隔离「数据增强」增益。

三大基准：CUHK-PEDES、ICFG-PEDES、RSTPReid；指标 Rank-1/5/10、mAP。

---

## 方法三件套

### LLM 改写

$$
T_i^{\mathrm{aug}}=\mathrm{LLM}\big(\mathrm{Concat}(T_i^{\mathrm{ori}},\,\text{``Rewrite this image caption.''})\big)
$$

实现用 **Vicuna**（LLaMA 上 ShareGPT 微调）。目标：保留关键属性（衣着、体貌、动作），换措辞与句式。

### Text Faithfulness Filter（TFF）

LLM 会幻觉、出乱码或其它语言。用 Sentence-Transformers 编码原/增文本，余弦相似度：

$$
s(T_i^{\mathrm{ori}},T_i^{\mathrm{aug}})=\frac{f_{\mathrm{st}}(T_i^{\mathrm{ori}})^\top f_{\mathrm{st}}(T_i^{\mathrm{aug}})}{\|f_{\mathrm{st}}(T_i^{\mathrm{ori}})\|\,\|f_{\mathrm{st}}(T_i^{\mathrm{aug}})\|}
$$

规则：

- $s<\alpha$：丢弃并**重新改写**；  
- $s\ge\alpha$：进入增广池。

CUHK-PEDES 上：$>90\%$ 增广与原文相似度 $>0.6$；尾部噪声靠 TFF 切掉。

$\alpha$ 过大（如 $>0.8$）→ 增广≈复读，多样性塌；过小 → 噪声进训。文中 ICFG 上约 **$\alpha=0.6$** 最优。

### Balanced Sampling Strategy（BSS）

不全量把增广并进数据集（分布偏移 + 残余噪声），而是按样本随机：

$$
T_i^{*}=\begin{cases}
T_i^{\mathrm{ori}}, & r_i>\beta \\
T_i^{\mathrm{aug}}, & r_i\le\beta
\end{cases}
\quad r_i\sim U[0,1]
$$

batch 内相似度矩阵混用原/增文本，再算 CLIP 式双向对比损失：

$$
\mathcal{L}^{v\to t}=-\sum_{i=1}^{N}\log\frac{\exp(s(V_i,T_i^{*})/\tau)}{\sum_{j}\exp(s(V_i,T_j^{*})/\tau)}
$$

（$t\to v$ 对称。）$\beta$ 过大（$>0.3$）易掉点；文中约 **$\beta=0.2$** 最优——**少量高质量增广**优于「翻倍脏数据」。

---

## 实验结论

### 对 CLIP 的增益（ViT-B/16 更明显）

| 数据集 | 指标示例 | 趋势 |
|--------|----------|------|
| CUHK-PEDES | Rank-1 / mAP | +LLM-DA 全面升（B/16 上 Rank-1 约 +1.9） |
| RSTPReid | Rank-1 / mAP | 同向；强骨干增益更大 |
| ICFG-PEDES | Rank-1 / mAP | 同向；mAP 相对升幅可观 |

### 对比传统文本增强（RSTPReid）

| 方法 | 相对基线 |
|------|----------|
| Random Deletion / Swap | 部分指标持平或伤 mAP |
| Back Translation | 小幅升，多样性有限 |
| **LLM-DA** | **全面最佳** |

定性：EDA 类易删关键词、打乱语法；LLM 改写更完整、句式更丰富。

### 消融（CUHK，CLIP B/16）

| DA | TFF | BSS | 结论 |
|----|-----|-----|------|
| ✓ | | | 微升 |
| ✓ | ✓ | | 明显升（去噪关键） |
| ✓ | | ✓ | 小升 |
| ✓ | ✓ | ✓ | **最优** |

---

## 与 embedding 数据工程的关系

LLM-DA **不是**难负例论文，但与检索微调数据管线高度同构：

| 维度 | LLM-DA | 难负例线（ANCE / NV-Retriever） |
|------|--------|--------------------------------|
| 作用对象 | **正例文本**多样性 | **负例**硬度与纯度 |
| 噪声控制 | TFF（原–增语义相似） | 正例锚定 / CE 去噪 |
| 用量控制 | BSS（$\beta$） | 每 query 2–4 负、刷新节奏 |
| 可插拔 | 不改损失形式 | 通常只改训练 jsonl |

对本仓库（cloud_emb / 图文检索）：

1. **Query / caption 扩写**：Vicuna / Qwen 等改写 + embedding 余弦过滤（TFF）。  
2. **不要无脑全量替换**：用 $\beta$ 或「每 epoch 以概率 p 用增广句」。  
3. 与 `expand_online_queries.py` 一类管线对齐：扩写 → 过滤 → 再挖负。  
4. 行人域结论可推广到「短描述、标注贵」的图文对；隐私敏感场景优先**文本侧**增强。

相关：LaCLIP / CLIPS 等「LLM rewrite caption」思路（文中引用 Fan et al.）；QZhou / Jina 的 LLM 合成数据是同一家族的规模化版本。

---

## 局限

- 任务专指 TPR；未系统评 MTEB / 通用检索。  
- 改写 LLM 与忠实度模型都有成本；需离线预生成。  
- $\alpha/\beta$ 依赖领域与相似度尺度，需验证集重扫。  
- 只增强文本，不解决「行人图本身不够」；图侧仍靠传统增广或合成图。

---

## 同目录对照

| 文档 | 关系 |
|------|------|
| [NV-Retriever详解.md](../NV-Retriever/NV-Retriever详解.md) | 负例过滤；与 TFF 同属「质量门」 |
| [难负例挖掘工业实践.md](../难负例挖掘工业实践.md) | 工业闭环 |
| [DeVE-QA稠密视频事件问答详解.md](../DeVE-QA/DeVE-QA稠密视频事件问答详解.md) | LLM 合成 QA 数据另一极 |
| [QZhou-Embedding详解.md](../QZhou-Embedding/QZhou-Embedding详解.md) | 大规模 LLM 结构/语义/难负合成 |
| [0.6B图搜图文搜图自训学习行动路线.md](../0.6B图搜图文搜图自训学习行动路线.md) | 「大模型当数据机」行动线 |

---

## 参考文献

1. Li et al. (2024). Data Augmentation for Text-based Person Retrieval Using Large Language Models. [arXiv:2405.11971](https://arxiv.org/abs/2405.11971)  
2. Cao et al. (2024). An Empirical Study of CLIP for Text-based Person Search. AAAI.  
3. Fan et al. (2023). Improving CLIP Training with Language Rewrites. NeurIPS.  
4. Wei & Zou (2019). EDA.  
