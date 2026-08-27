# DeVE-QA：稠密视频事件问答与数据构建

> paper: [Question-Answering Dense Video Events](https://arxiv.org/abs/2409.04388)（SIGIR 2025；arXiv:2409.04388v4）  
> HTML: https://arxiv.org/html/2409.04388v4  
> code/data: 录用后开源（文中声明）  
> backbone: LLaMA-VID（层级 caption）+ GPT-4o / Gemini（QA）+ CLIP（一致性 / 干扰项）  
> date: 2024–2025  
> modality: 长视频 · 多事件 · 时间定位  
> languages: 英文  

> 本文把 **DeVE-QA 数据集构建（GPT-4 + 人工）、DeVi 免训练管线（层级稠密描述 / 时序记忆 / 自洽校验）** 写全。  
> **与 embedding 调研的接口**：它不是双塔嵌入模型，但是 **视频侧「稠密事件 → QA / 描述」数据工程** 的标杆；构建干扰答案与自洽检查大量使用 **CLIP 向量相似度**，产出的事件 caption / query 可反哺视频检索与时序定位语料（参见 modelforge `datasets/embedding/query_corpus/`）。

---

## 一句话定位

提出任务 **Question-Answering on Dense Video Events**：在**长视频、多事件**上答题，并**时间定位**到支撑片段。配套基准 **DeVE-QA**（约 10.6k 视频、78k 题、26k 事件）与免训练方法 **DeVi**。

| 项 | 内容 |
| --- | --- |
| 相对短视频 QA | 事件密、视频长、必须 grounding |
| 数据来源 | ActivityNet-Captions → GPT-4 出题 → 人工校测试集 |
| DeVi 三件套 | 层级稠密 caption · Temporal Event Memory · Self-consistency |
| 宣称增益 | DeVE-QA 上 Acc@GQA 相对强基线约 **+4.8%**；NExT-GQA 约 **+2.1%** |

---

## 为何要新任务 / 新数据

既有 VideoQA（MSRVTT、MSVD、TGIF、ActivityNet-QA、NExT-QA…）多聚焦：

- 短视频、**单一全局事件**；或  
- 长视频但不强调**多事件交织**；或  
- 有 grounding（NExT-GQA、TVQA）但片段更短、事件密度不足。

稠密事件理解传统走 **dense captioning**，但：句子生成易幻觉、评测主观（BLEU/CIDEr）。QA + 多选 + 强制定位，评测更硬、更可复现。

形式化：视频 $v$、事件集 $E$、问题 $q$、候选 $C$（5 选 1），预测答案与区间：

$$
\{\hat{c},\hat{t}\}=\psi(c,t\mid E,q,C)\,\phi(E\mid v),\quad \hat{t}=\{t_s,t_e\}
$$

---

## DeVE-QA 数据集

### 构建流水线

1. **过滤 ActivityNet-Captions**：偏长 caption、去掉几乎覆盖整段视频的事件（否则 grounding 平凡）；随机子采样 → **10,643** 视频、**26,111** 事件描述。  
2. **GPT-4o 自动 QA**：每事件最多 3 题；控制题/答长度；鼓励 why/how/before/after；CLIP 相似度去近重题。  
3. **干扰项（对 embedding 很重要）**：  
   - 同视频相似题借答案候选；  
   - 用**题外时段**视频帧的 CLIP 近邻答案作「相关但时段错」干扰；  
   - 用题内时段近邻但非正解作细粒度干扰；  
   - 再加随机答案保多样性。  
4. **人工**：35 人约 267 小时校测试集；约 **74%** QA 被改。

### 规模与对比

| | Train | Test |
|--|-------|------|
| 视频 | 7,179 | 3,464 |
| 问题 | 53,361 | 24,963 |
| 均长 | ~127s | ~125s |
| 段长 | ~39s | ~41s |

相对 NExT-GQA：视频更长、片段更长、**显式 dense-event**；证书长度（答所需片段）约 **5.5×**。

---

## DeVi：免训练求解器

### 层级稠密事件描述

多尺度切段（DeVE-QA 例：**15s / 35s / 65s**），各层均匀抽帧送 LLaMA-VID 出 caption + 时间戳，得 $E=\{E_s,E_m,E_l\}$。消融：3 层最优；换成「逐帧 caption」（LLoVi 式）掉点，且对**高事件密度**子集掉得更狠。

### Temporal Event Memory

孤立段落 caption 缺长程依赖。用 LLM 在「全层 caption + 问题」上**上下文改写**，并生成全局 synopsis $e_y$。同时 1fps CLIP ViT-L/14 帧特征 $f_v$ 写入记忆：

$$
M=\{E,E',f_v\}
$$

### 事件锚定 QA + 自洽

LLM 读 $E'$ 选题并输出 $[t_s,t_e]$。常出现「答对但定位错 / 相反」。用 CLIP 查答案与定位片段一致性：

$$
R_{va}=\cos(f_v,f_a)
$$

若 $R_{va}<\sigma$（文中 $\sigma\approx 0.4$），带上分数与上一轮结果做 **dynamic verification** 再问，最多 $\delta=2$ 轮。

---

## 实验结果（摘要）

- **纯 QA**：DeVi-Gemini ≈ **71.8%** Acc@QA，显著高于 LongVA / LLoVi / VideoAgent 等。  
- **Grounded QA（DeVE-QA）**：Acc@GQA 约 **27.7%**，超 LLoVi 约 4.8%；IoP/IoU 同步升——不是只刷答题。  
- **NExT-GQA**：Acc@GQA 仍领先约 2.1%。  
- 消融：去层级 caption / 去上下文 / 去自洽，QA 与 GQA 均明显下降；长视频、密事件上相对优势更大。  
- 实现：captioner 用 LLaMA-VID 优于 VideoBLIP；QA 骨干越大越好（Gemini > GPT-4o ≫ 小开源）。

人类 Acc@GQA 仍远高于机器（差距可达三十多个点）——任务仍很难。

---

## 与 embedding / 本仓库数据的接口

| 可迁移点 | 做法 |
|----------|------|
| **视频 query 语料** | 事件 caption、合成 why/how 问句 → 检索 query 池（对齐 `query_corpus`） |
| **难负 / 干扰构造** | 「时段错但语义近」的 CLIP 挖法，可类比文本假负治理的反面：显式造 hard distractor |
| **一致性门** | $R_{va}$ 阈值 ≈ LLM-DA 的 TFF / NV 的正例锚：用向量相似度做质量门 |
| **多尺度描述** | 短/中/长片段 caption 可作多粒度正例对（clip–text / video–text 对比） |
| **非目标** | 不要把 DeVi 当成 embedding 训练算法；冲 MTEB 仍看 E5/NV/ANCE 线 |

对 cloud_emb / 监控场景：长视频多事件是常态；「先层级描述再检索 / QA」比单向量一口吃完全片更稳。

---

## 局限

- 强依赖 GPT-4 / Gemini API 与较强 caption MLLM；开销与可复现性受制于闭源。  
- DeVE-QA 源自 ActivityNet，领域偏互联网活动视频。  
- Acc@GQA 绝对值仍低；自洽迭代增加延迟。  
- 训练集约自动生成，虽测试集人工重，训练噪声仍在。

---

## 同目录对照

| 文档 | 关系 |
|------|------|
| [LLM-DA文本行人检索数据增强详解.md](LLM-DA文本行人检索数据增强详解.md) | 另一条 LLM 数据增强（图文行人） |
| [ColPali详解.md](ColPali详解.md) / [ColQwen系列详解.md](ColQwen系列详解.md) | 视觉文档检索；同属「视觉当页/段」 |
| [难负例挖掘工业实践.md](难负例挖掘工业实践.md) | 向量相似度做质量门 |
| [0.6B图搜图文搜图自训学习行动路线.md](0.6B图搜图文搜图自训学习行动路线.md) | 大模型当数据机 |

---

## 参考文献

1. Qin, Xiao, Yao (2025). Question-Answering Dense Video Events. SIGIR 2025. [arXiv:2409.04388](https://arxiv.org/abs/2409.04388)  
2. Krishna et al. (2017). Dense-Captioning Events in Videos. ICCV.  
3. Xiao et al. (2024). NExT-GQA. CVPR.  
4. Zhang et al. (2024). LLoVi. EMNLP.  
