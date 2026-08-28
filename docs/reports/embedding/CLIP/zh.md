> 原文: [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)（ICML 2021）
> 说明: 本文为论文全文中文技术展开，公式/图表编号与原文一致；图片自 arXiv HTML 抽取，caption 中译；数值原样保留。原论文 48 页，本文按章节组织覆盖 §1–§8。

**预印本信息：** arXiv:2103.00020v1 [cs.CV]，2021 年 2 月 26 日；会议版本：ICML 2021。

**代码与预训练权重：** https://github.com/OpenAI/CLIP

---

# 用自然语言监督学习可迁移视觉模型（Learning Transferable Visual Models From Natural Language Supervision）

**作者：** Alec Radford\*、Jong Wook Kim\*、Chris Hallacy、Aditya Ramesh、Gabriel Goh、Sandhini Agarwal、Girish Sastry、Amanda Askell、Pamela Mishkin、Jack Clark、Gretchen Krueger、Ilya Sutskever

**单位：** OpenAI，旧金山

**邮箱：** {alec, jongwook}@openai.com

\* 前两位作者贡献相同。

---

## 摘要（Abstract）

主流的先进视觉系统被训练来预测**预先定义的一组固定类别**。这种受限的监督形式限制了它们的通用性与可用性——每加入一个新概念就要新的标注数据。**从与图像配对的原始文本直接学习**是一种有吸引力的替代方案：它能利用远比人工标注广的监督源。作者证明一个简单的预训练任务——**预测哪段 caption 匹配哪张图**——是一种既高效又可扩展的方式，可以从互联网上收集的**4 亿对 (图, 文)** 中从头学出 SOTA 视觉表示。预训练完成后，可通过**自然语言**来指涉学到的视觉概念（或描述新概念），使模型能够**零样本迁移（zero-shot transfer）**到下游任务。

作者在 30+ 现有视觉数据集上做基准评测——涵盖 OCR、动作识别、地理定位、多种细粒度分类等。模型**在大多数任务上非平凡地迁移**，且经常**与全监督基线相当**——完全不用做任何数据集特定的训练。例如在 ImageNet 上零样本达到 **76.2% top-1**——与最初的 ResNet-50 齐平，同时**完全不用**其 128 万训练样本中的任何一张。代码与预训练权重开源于 https://github.com/OpenAI/CLIP。

---

## 1 引言与相关动机（Introduction and Motivating Work）

**NLP 里的范式转移**。近年来直接**从原始文本学习**的预训练方法（Dai & Le, 2015; Peters et al., 2018; Howard & Ruder, 2018; Radford et al., 2018; Devlin et al., 2018; Raffel et al., 2019）彻底改变了 NLP：与任务无关的目标（自回归 / 掩码语言建模）在计算量、模型规模、数据规模上都能持续 scaling；"text-to-text" 统一接口（McCann et al., 2018; Radford et al., 2019; Raffel et al., 2019）使得任务无关的架构可以**零样本迁移**到下游任务——去除了每个任务定制头或数据集定制的需要。GPT-3 等旗舰模型（Brown et al., 2020）在多任务上与专家模型持平，几乎不需要任务特定数据。

**视觉的现状**。视觉里预训练仍然主要依赖**人工标注**（如 ImageNet, Deng et al., 2009）。作者的问题：**能不能用类似 NLP 的 scale-up 直接从 web 文本学习视觉？**

**既往尝试**。这条路 20 年前就有：Mori et al. (1999) 训练模型预测图像配文的名词/形容词；Quattoni et al. (2007) 通过训练"从 caption 预测词"的分类器；Srivastava & Salakhutdinov (2012) 用多模态 Deep Boltzmann Machines；Joulin et al. (2016) 用 YFCC100M 的 title/描述/hashtag 构造多标签分类任务，用 AlexNet 训练；Li et al. (2017) 扩到 phrase n-gram 并证明可以零样本迁移分类任务。近期 VirTex（Desai & Johnson, 2020）、ICMLM（Bulent Sariyildiz et al., 2020）、ConVIRT（Zhang et al., 2020）用 Transformer-based 语言建模、掩码语言建模、对比目标进一步展示这条路的潜力。

**为什么迟迟没起飞**。既往工作的性能与主流方法的差距太大。Li et al. (2017) 在 ImageNet 零样本仅 11.5%——远低于当时 SOTA 88.4%（Xie et al., 2020），甚至低于 50% 的经典视觉方法（Deng et al., 2012）。**替代路径**：更狭窄但目标更明确的弱监督（Mahajan et al., 2018 用 Instagram hashtag; Kolesnikov et al., 2019 与 Dosovitskiy et al., 2020 用 JFT-300M 噪声标签）能取得更好的迁移，但代价是**依然限制在 1000 或 18291 类的静态 softmax**——牺牲了灵活性。

**关键差异**：这些弱监督方法在数亿到数十亿张图像上训练了**多个加速器年**；而 VirTex/ICMLM/ConVIRT 只训了**加速器天数**，数据在 10-20 万级。**作者要做的**：把差距填上——**用 4 亿对 (图, 文)** + 大规模训练，看看直接从自然语言学能走多远。

**方法与主要发现**。作者称之为 **CLIP（Contrastive Language-Image Pre-training）**——受 ConVIRT 启发但**大幅简化并在大规模下从头训练**。共训 8 个模型、跨约两个数量级的算力，观察到迁移性能是**算力的平滑可预测函数**（Hestness et al., 2017; Kaplan et al., 2020）。CLIP 类似 GPT 家族——在预训练时**同时学会**多种任务：OCR、地理定位、动作识别等。作者在 30+ 现有数据集上评测零样本迁移，发现它常常与任务特定的监督模型相当。同时用线性 probe 分析（linear probe representation learning）显示 CLIP **优于最好的开源 ImageNet 模型**，且更高算力效率。**零样本 CLIP 对分布漂移更鲁棒**，说明零样本评测是模型能力的更真实衡量。第 7 节讨论政策与伦理影响。

---

## 2 方法（Approach）

### 2.1 自然语言监督（Natural Language Supervision）

作者阐述："自然语言监督"不是新的概念，只是术语混乱：Zhang et al. (2020)、Gomez et al. (2017)、Joulin et al. (2016)、Desai & Johnson (2020) 都在做这件事，但被冠以不同的名字（无监督 / 自监督 / 弱监督 / 有监督）。**共同点是把自然语言当作训练信号**。

自然语言监督相对其它训练方式有几个潜在优势：

- **易于 scale**：不需要众包标注、不需要"1-of-N 多数投票金标"格式，只需要从互联网上被动地收集 (图, 文) 对；
- **不仅学表示，还把表示连接到语言**——从而支持灵活的零样本迁移。

### 2.2 构造足够大的数据集（Creating a Sufficiently Large Dataset）

**既有数据集局限**：

- MS-COCO（Lin et al., 2014）与 Visual Genome（Krishna et al., 2017）：高质量人工标注但仅 10 万张图；
- YFCC100M（Thomee et al., 2016）：1 亿张，但 metadata 稀疏、质量参差。过滤到含英文自然语言 title/描述后**缩水 6× 只剩 1500 万张**——约等于 ImageNet 规模。

**作者的数据集**：构造新的 **4 亿对 (图, 文)** 数据集，从多种互联网公开来源收集。为覆盖尽量宽的视觉概念集合，作者构造了 **50 万条查询词** 的列表（英文 Wikipedia 中出现 ≥ 100 次的所有词 + 高互信息的 bigram + Wikipedia 高访问量文章名 + WordNet synset 补齐）。搜索时每查询词最多保留 20000 对以做**类别粗平衡**。结果数据集总词数与训练 GPT-2 用的 WebText 相当。作者称之为 **WIT（WebImageText）**。

### 2.3 选择高效的预训练方法（Selecting an Efficient Pre-Training Method）

**训练效率是关键**。现代视觉系统计算开销巨大（Mahajan et al. 2018 要 19 GPU 年、Xie et al. 2020 要 33 TPUv3 core-years），且它们**只训 1000 类**。从自然语言学"开放类"看起来更难。作者最终按训练效率选定方法。

**尝试 1：VirTex-style 生成式预测**。联合训一个图像 CNN + 文本 Transformer，预测图像的 caption。**问题**：63M 参数的 Transformer 语言模型（已用两倍 ResNet-50 的算力）学 ImageNet 类别比"预测 caption 词袋（BoW）"的 baseline **慢 3 倍**（图 2 示）。

**尝试 2：BoW 预测**。改成预测词袋而非精确词序，训练效率提升 3 倍。

**尝试 3：InfoNCE 对比目标（CLIP）**。把"预测精确文本"换成"预测**哪段文本作为整体**匹配这张图"——一个更容易的代理任务。相对 BoW 预测再提升 **4 倍**（图 2）。合起来相对 VirTex 生成式提升 **12 倍**。

**核心问题：为什么对比比生成好？** 一张图可以对应无数条 caption——描述、评论、无关文字都可能。**生成模型必须预测某一个精确文本**（要求过强）；对比模型只需要"文本作为一个整体是否与图匹配"（要求恰当）。近期视觉对比学习（Tian et al., 2019; Chen et al., 2020a）与 ConVIRT（Zhang et al., 2020）已表明对比目标能学出更好表示。

**CLIP 训练目标**。给一个 batch $N$ 对 $(I_i, T_i)$，CLIP 要预测 $N \times N$ 个可能配对中**哪 $N$ 个真实发生**。CLIP 联合训练**图像编码器**和**文本编码器**，最大化正对 embedding 的余弦相似度、最小化 $N^2 - N$ 个错配对的相似度。**对称交叉熵损失**（同时对图→文与文→图两个方向做 softmax）。

作者指出，这种 batch 构造技术与目标最早出自 deep metric learning 的 **N-pair loss**（Sohn, 2016），被 Oord et al. (2018) 的 **InfoNCE** 推广，最近由 Zhang et al. (2020) 引入医学图文预训练。CLIP 相对 ConVIRT 做了几处简化：

- **从零训**，不用 ImageNet 权重初始化图像塔、不用预训练权重初始化文本塔；
- **投影只用线性**——不用 SimCLR-style 的非线性 MLP（作者未观察到差异，推测非线性 MLP 是与视觉自监督特定细节耦合的）；
- **去掉文本采样函数** $t_u$（因为 CLIP 的 caption 通常只有一句）；
- **只用随机方形裁剪**作为图像增强；
- **温度参数** $\tau$ 参数化为 $\log(1/0.07)$ 附近的对数标量、**训练中直接学**，避免作为超参调。

**伪代码（简化）**：

```python
# I_f = ImageEncoder(I)         # [N, d_i]
# T_f = TextEncoder(T)          # [N, d_t]
# W_i: [d_i, d_e], W_t: [d_t, d_e]

I_e = l2_normalize(I_f @ W_i)   # [N, d_e]
T_e = l2_normalize(T_f @ W_t)   # [N, d_e]

logits = I_e @ T_e.T * exp(t)   # [N, N] 缩放的余弦相似度矩阵

labels = arange(N)
loss_i = cross_entropy(logits,   labels, axis=0)  # 图 → 文
loss_t = cross_entropy(logits.T, labels, axis=0)  # 文 → 图
loss = (loss_i + loss_t) / 2
```

### 2.4 模型选择与扩展（Choosing and Scaling a Model）

**图像编码器**：

1. **ResNet 家族**：ResNet-50 + 若干改进——ResNet-D 改进（He et al., 2019）、antialiased rect-2 blur pooling（Zhang, 2019）、把 global average pooling 换成 **transformer-style attention pooling**（single-layer QKV attention，query 来自 global pooled 表征）。
2. **Vision Transformer (ViT)**（Dosovitskiy et al., 2020）：紧跟原实现，仅加了 patch + position embedding 后的 LayerNorm、稍改初始化。

**文本编码器**：63M 参、12 层、512 宽、8 头 Transformer（Vaswani et al., 2017; Radford et al., 2019 架构）。BPE 词表 49152。序列最长 76 token，加 [SOS] / [EOS]。取 [EOS] 位在最上层的表示，LayerNorm + 线性投影到多模态嵌入空间。文本编码器**保留 masked self-attention**（为将来 LM 辅助目标预留）。

**扩展**：对 ResNet 用 EfficientNet-style 联合放宽度/深度/分辨率；对 ViT 仅扩大宽度（作者发现文本塔容量对 CLIP 敏感性较低——只按 ResNet 的宽度增量成比例扩，深度不变）。

### 2.5 训练（Training）

作者训了 5 个 ResNet + 3 个 ViT：ResNet-50、ResNet-101、RN50x4、RN50x16、RN50x64（后三者约 4×、16×、64× ResNet-50 算力）；ViT-B/32、ViT-B/16、ViT-L/14。都训 32 epoch。优化器：Adam + decoupled weight decay（Loshchilov & Hutter, 2017）+ cosine decay 学习率（Loshchilov & Hutter, 2016）。基线 ResNet-50 训 1 epoch 用 grid/random/manual 调超参，大模型继承并根据算力约束启发式适配。温度 $\tau$ 初始化 $\log(1/0.07)$（Wu et al., 2018），并 **clip 到不超过 100** 以稳定训练。

**大 batch = 32,768**。用 mixed precision（Micikevicius et al., 2017）加速与省显存。为省显存进一步用了 gradient checkpointing（Griewank & Walther, 2000; Chen et al., 2016）、half-precision Adam statistics（Dhariwal et al., 2020）、half-precision stochastically rounded 文本塔权重。相似度矩阵计算也做了**跨 GPU sharding**，每 GPU 只算它 local batch 的必要子集。

**训练时长**：RN50x64 用 592 张 V100 训 18 天；最大 ViT 用 256 张 V100 训 12 天。ViT-L/14 在预训完成后还在 336 分辨率再训 1 epoch 提升性能（类似 FixRes; Touvron et al., 2019），记作 **ViT-L/14@336px**。后文若不特别说明，"CLIP"指的都是这个最佳模型。

**总览图**：

![图 1（原文 Figure 1）：CLIP 方法总览。(1) 对比预训练；(2) 用类别名合成 zero-shot 分类器；(3) 零样本预测](figs/main.png)

**图 1（原文 Figure 1）：** CLIP 方法总览。 **(1) 对比预训练**：一个 batch 内 $N$ 张图与 $N$ 段文本，模型学到图像编码器 + 文本编码器，让 $N$ 对真实 (图, 文) 的 embedding 相似度高、$N^2 - N$ 对不相关配对相似度低。 **(2) 从类别名合成分类器**：将下游数据集的每个类别名（如 "plane"、"car"、"dog"、…）填入 prompt 模板（"A photo of a {object}."），过文本编码器得到 $N$ 个类别向量 $T_1, T_2, \dots, T_N$。 **(3) 零样本预测**：把待预测图像过图像编码器得到 $I_1$，与 $\{T_i\}$ 计算余弦相似度并 softmax，取最高的作为预测类别。

---

## 3 实验（Experiments）

### 3.1 零样本迁移（Zero-Shot Transfer）

#### 3.1.1 动机

计算机视觉里的"零样本学习"通常指分类未见的对象类别（Lampert et al., 2009）。本文用**更宽泛**的含义——**泛化到未见的数据集**，作为"泛化到未见任务"的代理。作者主张：既有的视觉数据集大多是为了推动方法研究、而不是评测某个具体任务，所以在这些数据集上做零样本迁移**更像**在评测分布外鲁棒性（distribution shift）与任务学习能力，而不是单纯的类别泛化。

Visual N-Grams（Li et al., 2017）是本领域最早研究这种"零样本迁移到既有分类基准"的工作，也是与 CLIP 最可对比的先例。Visual N-Grams 学 142,806 个 visual n-gram（1- 到 5-gram）的字典，用 Jelinek-Mercer smoothing 优化 caption n-gram 概率。零样本迁移时，把类别名转成 n-gram、按字典打分。作者的分析借鉴 NLP 里 GPT 系列的经验：任务学习作为预训练"副产品"（Liu et al., 2018; Radford et al., 2018, 2019）。

#### 3.1.2 CLIP 如何做零样本迁移

- 对每个数据集，把类别名当作候选文本；
- 用 CLIP 的图像/文本编码器分别得到 $I^e, \{T_k^e\}$；
- 用**缩放的余弦相似度** + softmax 得到类别分布；
- 预测 = 最大相似度类。

**等价理解**：CLIP 的零样本推理等价于 **L2 归一化输入 + L2 归一化权重 + 无 bias + 温度缩放**的多分类逻辑回归；图像编码器是 CV backbone，**文本编码器是 hypernetwork**（Ha et al., 2016）——按类名动态生成分类器权重。每一步 CLIP 预训练**都可以看作在优化一个随机构造的、含 32,768 类的 CV 数据集**（每类 1 个样本）——类别由自然语言描述定义。

#### 3.1.3 与 Visual N-Grams 的对比

| 数据集 | Visual N-Grams | **CLIP** |
| :--- | ---: | ---: |
| aYahoo | 72.4 | **98.4** |
| ImageNet | 11.5 | **76.2** |
| SUN | 23.0 | **58.5** |

（表 1：CLIP 与 Visual N-Grams 在三个零样本迁移基准上的对比。）

**注意**：这不是完全对等的方法学比较——CLIP 用了大 10× 的数据、~100× 每次预测的算力、总训练算力可能 >1000×、并且使用了 Transformer（Visual N-Grams 时代之前不存在）。作者用同一个 YFCC100M 数据集训了一个 CLIP RN50 作对照，能在 1 V100 天内**达到 Visual N-Grams 报告的 ImageNet 分数**（且是从头训、无 ImageNet 权重初始化）。此外 CLIP 在 aYahoo 上 95% 减少错误率，在 SUN 上把准确率翻倍以上。

#### 3.1.4 Prompt 工程与 Ensembling

大多数图像分类数据集把类别名当作"事后附加信息"，只给数字 ID 与 English 名字。有的数据集（如 Flowers102、GTSRB）甚至不发布这个映射，妨碍零样本迁移。作者的观察：

- **一词类名**常有多义（如 "boxer" 既是狗品种也是拳击手；"crane" 既是鸟也是起重机）；
- **句子模板** 显著优于裸类名：把 `"cat"` 换成 `"A photo of a cat."` 平均涨 **1.3%**。
- **特定任务定制模板**：
  - 细粒度类：`"A photo of a {label}, a type of {supercategory}."`
  - OCR：给 label 加引号；
  - 卫星图：`"A satellite photo of a {label}."`

- **Prompt Ensembling**：对同一数据集写多个模板（例如 80 个），把 embedding 空间上的**多个类别向量做平均**（不是分数概率平均）。相当于**没有额外推理成本**（可以预算完就缓存），但作者观察到 27 个数据集平均 **+3.5%** 的提升。ImageNet 上使用了 80 个 prompt 模板。

**图 4 展示**（原文 Figure 4）：Prompt engineering + ensembling 相对没用它相当于把大约 **4× 计算量**的性能提升免费拿了下来。

#### 3.1.5 零样本性能分析

**图 5 展示**（原文 Figure 5）：在 27 个数据集上比较 CLIP zero-shot 与 ResNet-50 linear probe：**CLIP zero-shot 在 16/27 数据集上胜出**。CLIP 在 ImageNet 与几个类似的 CV 通用数据集上表现极强；在**行为识别（Kinetics700、UCF101）**、**OCR（RenderedSST2、HatefulMemes）**、**地理定位（Country211）**、**细粒度识别（Stanford Cars、FGVC Aircraft、Food101）** 上都强。在一些较专门的任务上（EuroSAT、RESISC45 卫星图、PatchCamelyon 病理图、CLEVRCounts 数数）CLIP 明显弱——这些任务与自然图像分布远、需要特定训练。

**Few-shot 反常识现象**（图 6）：作者比较 CLIP zero-shot、CLIP few-shot logistic regression、其它模型的 few-shot logistic regression。**16-shot CLIP logistic regression 才追上 zero-shot CLIP 的水平**（20 个类别时）。作者的解释：零样本分类器由文本塔"合成"，等价于在**语义空间**中的一个先验位置；少量样本的 logistic regression 会**从这个先验偏离**到样本经验位置，可能反而更差。这是零样本 CLIP 的一个反直觉性质。

**扩展曲线**（图 9）：零样本 CLIP 性能随算力**平滑可预测**。ImageNet-1k 上算力提升 4× 大约对应 zero-shot 分数提升 4%。

#### 3.1.6 与全监督的对比

**ImageNet-1k**（表 10）：ViT-L/14@336px zero-shot 达到 **76.2%** top-1，与原始 ResNet-50 齐平。但 top-5 达 **95%**，与 Inception-V4 齐平（Szegedy et al., 2016）。

- 与近 SOTA 比：CLIP zero-shot 仍**落后于 EfficientNet-L2 Noisy Student**（Xie et al., 2020，88.4% top-1）等大模型；
- 但**从 2017 年 Visual N-Grams 的 11.5% 到 CLIP 的 76.2%**——**5 年 65 个百分点**——是明显的跃迁；
- **在 21 out of 27 数据集上**，CLIP linear probe **优于 Noisy Student EfficientNet-L2 的 linear probe**（图 11）。这些包括 OCR、地理定位、动作识别、细粒度分类等——**说明自然语言监督学到了很多 ImageNet 全监督模型学不到的能力**。

**样例**（图 21，Table 4）：作者对多个数据集给出定性预测样例。

![图 21（原文 Figure 21）：CLIP 在多样任务上的零样本预测样例。每类样例展示模型 Top-5 预测与真实标签](figs/qualitative.png)

**图 21（原文 Figure 21）：** CLIP zero-shot 的定性预测。在 Food101（挑出复杂菜品）、SUN397（区分实验室 vs 机房）、Country211（从街景猜国家）、Stanford Cars（车型识别）等多个数据集上，CLIP 都能给出正确的 top-1 或 top-5 预测——这些是 ImageNet 全监督模型难以做到的开放集分类。

### 3.2 表示学习（Representation Learning）

除了零样本，作者还用**线性 probe** 与**全模型微调**分析表示质量。

**Linear Probe 好处**（Kolesnikov et al., 2019; Chen et al., 2020a）：

- 不需要为每个数据集重新做超参搜索；
- 更"忠实"地反映 backbone 特征质量（微调可能弥补差的特征）；
- 分析简单、可解释；
- 更接近工业界"训好 backbone → 小分类头下游微调"的现实。

**主要发现**（图 10）：

- **12 数据集 Kornblith et al. (2019)**：CLIP ViT-L/14 平均线性 probe 分数达到 88.4%，明显超过 EfficientNet-L2 Noisy Student；
- **27 数据集综合**：CLIP 全线 Pareto 最优——同算力下比 EfficientNet / Noisy Student / BYOL / SimCLRv2 / MoCo / BiT-M / ViT (ImageNet-21k) 都更好；
- **自监督系（SimCLRv2 / BYOL）与 CLIP 类似**：都在广泛评测上明显超过 ImageNet 监督模型——这一发现暗示自监督/自然语言监督**都比"过度专门化的 ImageNet 监督"更泛化**。

### 3.3 对自然分布漂移的鲁棒性（Robustness to Natural Distribution Shift）

**背景问题**。ImageNet 模型即使在 val set 上超过人类（He et al., 2015），仍在**分布漂移评测**上出现严重降级（Recht et al., 2019; Barbu et al., 2019）。Ilyas et al. (2019); Geirhos et al. (2020) 等的一个共同解释是：深度网络善于**利用训练分布内的伪相关**——这些相关在其它分布上不成立。

**Taori et al. (2020)** 系统研究 ImageNet 模型的这类失效，评测 7 个**自然分布漂移**数据集：ImageNetV2、ImageNet Sketch、Youtube-BB、ImageNet-Vid、ObjectNet、ImageNet Adversarial、ImageNet Rendition。

**CLIP zero-shot 的观察**（图 13）：

![图 13（原文 Figure 13）：CLIP zero-shot 在分布漂移下比标准 ImageNet 模型鲁棒得多](figs/robustness_zs.png)

**图 13（原文 Figure 13）：** **左**：理想鲁棒模型（虚线 y = x）在 ImageNet 分布与其它自然分布上表现应相同。CLIP zero-shot 模型把"鲁棒性 gap"缩小到最多 75%。**右**：以 banana 类别为例（7 个分布漂移数据集中的 5 个共享此类）——ViT-L/14@336px 与在 ImageNet val 上分数相同（76.2%）的 ResNet-101 的分数对比：

| 数据集 | ResNet-101 (ImageNet 分数 76.2%) | Zero-Shot CLIP | Δ |
| :--- | ---: | ---: | ---: |
| ImageNet | 76.2 | 76.2 | 0% |
| ImageNetV2 | 64.3 | 70.1 | **+5.8%** |
| ImageNet-A | 2.7 | 77.1 | **+74.4%** |
| ImageNet-R | 37.7 | 88.9 | **+51.2%** |
| ObjectNet | 32.6 | 72.3 | **+39.7%** |
| ImageNet Sketch | 25.2 | 60.2 | **+35.0%** |

**同样的 ImageNet val 分数下**（76.2%），CLIP zero-shot 在几乎所有分布漂移数据集上 **10-70 个点**领先。

**鲁棒性干预实验**（图 14）：

![图 14（原文 Figure 14）：把 CLIP 适配到 ImageNet 反而略微降低平均鲁棒性](figs/robustness_intervention.png)

**图 14（原文 Figure 14）：** 作者做了两种鲁棒性干预：

1. **"Adapt to ImageNet"**：在 CLIP 特征上训一个 ImageNet 分类头（L2-regularized logistic regression）——**ImageNet 分数从 76.2 提到 85.4%（+9.2%）**，与 2018 年 SOTA（Mahajan et al., 2018）齐平；**但**平均鲁棒性略微下降（+9.2% on ImageNet 但在其它分布上 -4.7% ImageNet-R、-3.8% ObjectNet 等）；这个提升几乎完全**没有转移**到分布漂移。
2. **"Adapt to class shift"**：为每个下游数据集"定制"自己的零样本分类器（用该数据集的类别名而不是 ImageNet 类别名）——在若干与 ImageNet 不完全对齐的数据集上有大幅提升（如 Youtube-BB +26.9%、ImageNet Vid +8.3%），但只适用于有类别错位的少数数据集。

**Few-shot CLIP 的鲁棒性**（图 15）：

![图 15（原文 Figure 15）：Few-shot CLIP 的鲁棒性介于 zero-shot 与全监督之间](figs/robustness_fs.png)

**图 15（原文 Figure 15）：** Few-shot CLIP（1-shot 到 all-shot 在 ImageNet 上做 logistic regression）也比标准 ImageNet 模型更鲁棒，但**比 zero-shot CLIP 鲁棒性**下降。16-shot CLIP linear regression 在 ImageNet 上追平 zero-shot CLIP，但在分布漂移上落后。

**关键结论**：**用越少的 distribution-specific 训练数据，"有效鲁棒性"越强**——代价是 dataset-specific 性能变弱。作者建议社区关注这种"大规模任务无关预训练 + 广泛零样本 / few-shot 评测"的评测范式（Yogatama et al., 2019; Linzen, 2020），这样既能推动更鲁棒的系统开发，又能更准确评估模型能力。

---

## 4 与人类表现的比较（Comparison to Human Performance）

作者请 5 名众包员工分类 Oxford Pets 数据集（37 类猫狗品种）的 3669 张图。测试三种设置：

- **Zero-shot 人类**：不给任何示例（可以说 "I don't know" 表示不确定）；
- **One-shot 人类**：每类看一个示例；
- **Two-shot 人类**：每类看两个示例。

**结果**（表 2）：

| 设置 | 全数据集平均准确率 | Majority Vote 全数据集 | 已猜测的样本准确率 | Majority Vote 已猜测 |
| :--- | ---: | ---: | ---: | ---: |
| Zero-shot 人类 | 53.7 | 57.0 | 69.7 | 63.9 |
| **Zero-shot CLIP** | **93.5** | **93.5** | **93.5** | **93.5** |
| One-shot 人类 | 75.7 | 80.3 | 78.5 | 81.2 |
| Two-shot 人类 | 75.7 | 85.0 | 79.2 | 86.1 |

**关键发现**：

- **人类从 zero-shot 到 one-shot 大幅提升**（54% → 76%）；从 one-shot 到 two-shot 几乎不涨——人类是"知道自己不知道"，一个示例就把不确定项目更新过来；
- **CLIP 与 few-shot CLIP 都不能利用这种"先验知识 + 示例"的整合**——few-shot CLIP linear probe 不聪明；
- **CLIP 觉得难的样本，人类也觉得难**（图 16）——两者在困难类别上表现相关。

作者认为这是一个重要方向：**如何把先验知识合理集成到 few-shot 学习**，可能是 CLIP 后续算法改进的关键。

---

## 5 数据重叠分析（Data Overlap Analysis）

用互联网大规模数据训练的一个担忧是：**预训练数据可能与下游评测数据集重叠**，抬高分数。作者做了系统性检查：

**方法**：

1. 用一个近重复检测器，判定 WIT 中每张图与评测数据集的图是否近重复。分成 **Overlap**、**Clean**、**Dirty**（overlap 超阈值）三份；
2. 在三份上计算 CLIP 的 zero-shot 准确率，报 **All - Clean** 作为"数据污染带来的分数虚高"的估计；
3. 因重叠通常很少，作者做**二项显著性检验**（$H_0$ = Clean 分数，$H_1$ = Overlap 分数更高），报**one-tailed p-value**；对 Dirty 做 **99.5% Clopper-Pearson 置信区间**。

**结果**：

- 35 个数据集中 **9 个没有检测到重叠**（多为合成/专门数据集：MNIST、CLEVR、GTSRB；或数据集是在 WIT 采集之后创建的：ObjectNet、Hateful Memes）；
- **中位重叠率 2.2%，均值 3.2%**；
- **只有 7 个数据集**的 All - Clean 差异超过 0.1%；
- 其中**只有 2 个**通过 Bonferroni 校正的显著性检验。

**结论**：CLIP 的报告分数**没有被数据污染显著抬高**——重叠很小，且大部分场景下 Overlap 与 Clean 上的性能差异**不显著**。

---

## 6 局限（Limitations）

CLIP 有多方面局限（作者列出几条主要的）：

1. **零样本 CLIP 仍远低于 SOTA**：在多个数据集上落后于当前 SOTA 10-30%；估计要**再 1000× 算力**才能通过 scaling 追平 SOTA。
2. **在若干特定任务上表现差**：细粒度分类（车型区分、花卉种类）、抽象/系统性任务（数量计数、几何形状识别）、真正"分布外"任务（写在纸上的手写数字、非自然图像如医学扫描）都落后。
3. **数据效率仍差**：CLIP 32 epoch 训练用了 128 亿张（图, 文）对；即使每秒看 1 张也要 405 年。虽然通过 scale 补偿，但**数据效率本身**没提升。结合自监督/自训练是一个方向。
4. **评测方法学问题**：作者反复在 val set 上查询以指导 CLIP 开发——这不是真正的 zero-shot；且 27 数据集评测集是与 CLIP 能力**共同演化的**。呼吁社区制定**专门用于评测广泛零样本能力的新基准**。
5. **社会偏见**：CLIP 训在未过滤的互联网 (图, 文) 上，学到了很多**社会偏见**——第 7 节详细分析。
6. **通过自然语言指定分类器**虽然灵活，但也有局限：给出好的 caption 需要一些概念的表达，某些类别（如"含毒性的言论"）在图文里可能表述模糊。
7. **Few-shot 反直觉现象**：加少量示例反而不如零样本，直到 16-shot 才追平——**说明如何把先验知识与示例整合进 few-shot CLIP 是开放问题**。
8. **无法生成新概念**：CLIP 只能在给定候选类别下判别，不能自己"想出"新类别。

---

## 7 更广泛的影响（Broader Impacts）

CLIP 有多面的社会影响。作者做了初步的偏见（bias）与监控（surveillance）风险分析。

### 7.1 偏见（Bias）

**FairFace 分类基准**（表 3–5）：

- 在 White 类别上：Zero-shot CLIP 达到 Race 58.3%、Gender 95.9%、Age 57.1%；Linear Probe CLIP 分别 93.4 / 96.5 / 63.8——性别分类都在 95% 以上。
- 在 Non-White 类别上，Zero-shot CLIP 也在 90% 以上，与 Instagram-pretrained Linear Probe（弱基线）比性能相当或更好。
- 但**准确率高不代表 fair**（Raji et al., 2020）——真实世界的公平性还依赖类设计、部署方式、上下文等。

**类设计敏感性**（表 6–7）：作者用 7 个 FairFace 种族类 × 男女 + 3 个"犯罪相关"类 + 4 个"非人类"类作为标签集测试：

- Black 类的图像 **16.4%** 被误分为"犯罪相关"，White 24.9%、Indian 24.4%——类别设计对 misclassification 的分布有巨大影响；
- 加上 "child" 类后，儿童被误分为"犯罪相关/非人类"的比例大幅下降（0-2 岁：30.3% → 2.3%；3-9 岁：35.0% → 4.3%）。

**Members of Congress 数据集**（图 18）：用 300 职业类 + 云 API 返回的合并标签集测试。CLIP 100% 正确识别性别（远高于 FairFace）；但**返回的性别倾向标签明显不同**——男性图片的顶部标签有 "military officer"、"suit"、"executive"；女性图片的顶部标签有 "blouse"、"newsreader"、"public speaking"、"laughing" 等——反映训练数据的社会刻板印象。

**结论**：CLIP 的表现会随**类设计**变化很大，这既是"灵活性"也是"风险"——开发者可以轻松定义任何类别，模型都会给出结果，需要仔细评估类别设计对偏见的放大作用。

### 7.2 监控（Surveillance）

作者不鼓励监控应用，但对 CLIP 在这方面的能力做了预测性分析：

- **CCTV 图像分类**（VIRAT 数据集）：低分辨率 CCTV 场景下，CLIP 表现**有限**（很多细节丢失）；
- **零样本名人识别**：作者构造了含 100 名当代名人的评测集，CLIP-RN50x64 在没有见过分类器训练的情况下达到 **59.2%** top-1；在 2000 类的更大集合上降到 43.3%——**监视级别的应用是可行的**，且需要引起社区注意。

### 7.3 未来工作方向

作者建议社区关注：

- 在应用 CLIP 到高风险场景前做**具体的偏见分析**；
- 研究 CLIP-like 模型在**新任务定义**下的行为（"CLIP 能做什么"与"应该被用来做什么"是两回事）；
- 与政策制定者合作了解 CLIP-like 系统的更广影响。

---

## 8 相关工作（Related Work）

- **自然语言监督学习视觉**：Mori et al. (1999)、Quattoni et al. (2007)、Srivastava & Salakhutdinov (2012)、Joulin et al. (2016)、Li et al. (2017) 是最早的一批；近年 VirTex（Desai & Johnson, 2020）、ICMLM（Bulent Sariyildiz et al., 2020）、ConVIRT（Zhang et al., 2020）用 Transformer 与对比学习。CLIP 相对 ConVIRT **简化并大规模化**。
- **弱监督学习**：Mahajan et al. (2018) 用 Instagram hashtag；Kolesnikov et al. (2019); Dosovitskiy et al. (2020) 用 JFT-300M。这些取得强性能但只在**受限的类别系统**内。
- **零样本学习**：Socher et al. (2013); Frome et al. (2013); Norouzi et al. (2014) 等；CLIP 与它们的区别在于用 **NLP scale** 的数据与算力。
- **视觉自监督**：SimCLR（Chen et al., 2020a）、MoCo（He et al., 2020）、BYOL（Grill et al., 2020）、DINO（Caron et al., 2021）等；这些学到强特征但没有语言接口。
- **多模态学习**：ViLBERT、LXMERT、UNITER、SimVLM 等——focus on 更结构化的图文任务（VQA、caption 生成），与 CLIP 的对比学习范式不同。

---

## 9 结论（Conclusion）

作者研究了是否可以把 NLP 里"任务无关的 web-scale 预训练"直接搬到视觉。CLIP 在预训练中学到了执行多种任务的能力，可以通过自然语言 prompt 实现**零样本迁移**到很多现有数据集，性能与任务特定的监督模型相当。**当前工作仍有大量局限**——但也说明"从自然语言学"的视觉预训练方向值得继续探索。

---

## 附录索引（Appendix Highlights）

- **A** 数据集详细清单：27 数据集主评测 + 12 数据集 Kornblith 集 + 迁移设置；
- **B** 训练超参数与优化器细节；
- **C** 每个数据集的 prompt 模板（80 个用于 ImageNet）；
- **D** Linear probe 的详细分数表；
- **E** 分布漂移评测集详情；
- **F** 数据重叠分析的详细方法与结果；
- **G** FairFace / Congress / VIRAT 偏见与监控分析的补充细节。

---

*翻译约定：对比语言-图像预训练（Contrastive Language-Image Pre-training, CLIP）、自然语言监督（natural language supervision）、零样本迁移（zero-shot transfer）、Linear Probe（线性 probe）、鲁棒性（robustness）、分布漂移（distribution shift）、Prompt 工程（prompt engineering）、Prompt 集成（prompt ensembling）。ImageNet / ResNet / ViT / EfficientNet / BiT / MoCo / SimCLR / BYOL / VirTex / ConVIRT / ALIGN / JFT / YFCC100M / WIT / MS-COCO / Visual Genome / FairFace / VIRAT 按惯例不译。*
