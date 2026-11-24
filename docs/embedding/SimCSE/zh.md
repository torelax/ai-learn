> 原文: [arXiv:2104.08821](https://arxiv.org/abs/2104.08821)（EMNLP 2021）
> 说明: 本文为论文全文中文技术展开，公式编号与原文一致；图片自 arXiv HTML 抽取，caption 中译；表格数值保留原始数字，仅表头/说明中译。

**预印本信息：** arXiv:2104.08821v4 [cs.CL]，2021 年 4 月首次提交，2022 年 5 月最终修订；会议版本：EMNLP 2021。

**代码与模型：** https://github.com/princeton-nlp/SimCSE

---

# SimCSE：面向句向量的简单对比学习框架（Simple Contrastive Learning of Sentence Embeddings）

**作者：** Tianyu Gao†\*、Xingcheng Yao‡\*、Danqi Chen†

**单位：** † 普林斯顿大学计算机系；‡ 清华大学交叉信息研究院

**邮箱：** {tianyug, danqic}@cs.princeton.edu · yxc18@mails.tsinghua.edu.cn

\* 前两位作者贡献相同。工作在 Xingcheng 远程访问 Princeton NLP 小组期间完成。

---

## 摘要（Abstract）

本文提出 **SimCSE**，一个简单但显著推进了句向量（sentence embedding）技术水平的对比学习（Contrastive Learning）框架。作者首先给出**无监督（unsupervised）**版本：把一句话当作输入，让模型在对比学习目标下预测它自己，唯一的"噪声"是标准的 Dropout。这一极简做法表现出乎意料地好，与之前的**有监督**方法平起平坐。作者观察到，Dropout 相当于最"轻量"的数据增强，把它拿掉会导致表示崩塌（representation collapse）。随后作者提出**有监督（supervised）**版本：把自然语言推理（Natural Language Inference, NLI）数据中的"蕴含（entailment）"对当作正对（positive pair），把"矛盾（contradiction）"对当作难负例（hard negative），把它们塞进同一个对比学习目标里。

在 7 个语义文本相似度（Semantic Textual Similarity, STS）任务上，基于 BERT-base 的无监督 SimCSE 达到 **76.3%** 的平均 Spearman 相关系数，有监督 SimCSE 达到 **81.6%**，分别比此前最好结果高 **4.2** 与 **2.2** 个百分点。作者还从理论与实证两个角度说明：对比学习目标能够对预训练嵌入的**各向异性（anisotropic）**空间做正则化，让其分布更加均匀（uniform）；当有监督信号存在时，正对之间的**对齐（alignment）** 也进一步改善。

---

## 1 引言（Introduction）

学习通用的句向量是自然语言处理领域的一个基础问题，已有大量研究（Kiros et al., 2015; Hill et al., 2016; Conneau et al., 2017; Logeswaran and Lee, 2018; Cer et al., 2018; Reimers and Gurevych, 2019 等）。本文推进现有水平，展示了对比学习目标在与预训练语言模型（如 BERT、RoBERTa）结合时的巨大威力。SimCSE 是一个非常简单的对比学习框架：无论是无标签还是有标签数据，它都能训出高质量的句向量。

**无监督 SimCSE** 的做法只有一句话：把一句话输入编码器**两次**，两次前向使用不同的 Dropout mask，得到两份不同的向量，并把它们视为对比学习的正对（Figure 1(a)）；同一 mini-batch 里其它句子构成负例，模型的任务是在这些负例中"认出"另一个自己。这个想法看上去过于朴素，但比"预测下一句话"（Logeswaran and Lee, 2018）或各种离散数据增强（删词、替换等）都强不少，甚至能与此前的有监督方法并驾齐驱。作者做了一次细致分析发现：Dropout 相当于对隐状态的"最小数据增强"，一旦拿掉，模型会立即出现**表示崩塌**——正对被拉得越来越近，但整个空间的均匀性没有任何改善。

**有监督 SimCSE** 建立在近年利用 NLI 数据训练句向量的成功之上（Conneau et al., 2017; Reimers and Gurevych, 2019）。传统方法把 NLI 当作 3 分类任务，本文不再如此，而是**直接把"前提-蕴含假设"对当作对比学习的正对**（Figure 1(b)）。作者还进一步利用 NLI 中每条前提都自带的"矛盾假设"，把它作为**难负例**塞进对比学习目标里。这个"用 NLI 数据的更好方式"带来了显著提升。作者也比较了其它候选的有监督数据集，最终确认 NLI 尤为适合句向量学习。

为了理解 SimCSE 为何有效，作者借用了 Wang and Isola (2020) 的分析工具，即**对齐**（正对之间距离小）与**均匀性**（表示空间上均匀分布）两个指标。实证分析显示：**无监督 SimCSE** 主要在改善均匀性的同时借助 Dropout 噪声避免了对齐性退化；**有监督**下 NLI 训练信号进一步压紧了正对之间的对齐。作者还从**奇异谱（singular spectrum）**角度证明：对比学习目标可以"抹平"句向量矩阵的奇异值分布，从而降低各向异性。

**主要贡献：**

1. 提出无监督 SimCSE，仅用 Dropout 作为噪声、以自身为正对，就把 STS 平均 Spearman 从 72.05 提到 76.25（BERT-base）。
2. 提出有监督 SimCSE，把 NLI 蕴含对作正、矛盾对作难负，进一步提到 81.57（BERT-base）；RoBERTa-large 可达 **83.76**。
3. 从对齐-均匀性、奇异谱两个角度给出理论与实证解释；并指出既往 STS 评测中常见的口径混乱，呼吁统一评测协议。

---

## 2 背景：对比学习（Background: Contrastive Learning）

对比学习通过**拉近语义相近对、推开语义不相关对**来学习表示（Hadsell et al., 2006）。给一批带正对标记的样本 $\mathcal{D} = \{(x_i, x_i^+)\}_{i=1}^m$，其中 $x_i$ 与 $x_i^+$ 语义相关。作者沿用 Chen et al. (2020) 的对比框架，采用带**批内负例（in-batch negatives）**的交叉熵目标（Chen et al., 2017; Henderson et al., 2017）。设 $h_i, h_i^+$ 为 $x_i, x_i^+$ 的表示，$N$ 为 mini-batch 大小，则 $(x_i, x_i^+)$ 对应的训练损失为：

$$
\ell_i = -\log \frac{\exp\bigl(\operatorname{sim}(h_i, h_i^+)/\tau\bigr)}{\displaystyle\sum_{j=1}^{N} \exp\bigl(\operatorname{sim}(h_i, h_j^+)/\tau\bigr)} \tag{1}
$$

其中 $\tau$ 为温度超参数，$\operatorname{sim}(h_1, h_2)$ 为余弦相似度 $\tfrac{h_1^\top h_2}{\|h_1\| \cdot \|h_2\|}$。本文使用 BERT / RoBERTa 等预训练模型编码句子，即 $h = f_\theta(x)$，随后用式 (1) 微调全部参数。

**正对怎么来？** 对比学习的关键设计之一是如何构造 $(x_i, x_i^+)$。在视觉中，常见做法是对同一张图片做两次独立的随机变换（裁剪、翻转、颜色扰动、旋转等；Dosovitskiy et al., 2014）。语言中也有类似尝试（词删除、重排、替换；Wu et al., 2020; Meng et al., 2021），但受限于语言的离散性效果一般。§3 将展示：**在中间表示上直接使用标准 Dropout**，胜过所有离散变换。

在 NLP 里，也有一系列工作把式 (1) 应用到不同场景（Henderson et al., 2017; Gillick et al., 2019; Karpukhin et al., 2020）：这些做法通常从有监督数据（例如问答对）里拿到 $(x_i, x_i^+)$。由于 $x_i$ 与 $x_i^+$ 结构上差异较大，它们普遍用**双编码器（dual-encoder）**——$x_i$ 与 $x_i^+$ 分别过两个独立的 $f_{\theta_1}, f_{\theta_2}$。在句向量方向上，Logeswaran and Lee (2018) 也用过对比学习 + 双编码器，其"正对"是相邻的两句话。

**对齐（alignment）与均匀性（uniformity）**。Wang and Isola (2020) 指出对比学习的两个关键性质，并将其量化。给定正对分布 $p_{\text{pos}}$，**对齐** 计算配对表示之间的期望距离（假设已 L2 归一化）：

$$
\ell_{\text{align}} \triangleq \mathbb{E}_{(x, x^+) \sim p_{\text{pos}}} \bigl\lVert f(x) - f(x^+) \bigr\rVert^2 \tag{2}
$$

**均匀性** 描述表示在超球面上分布是否均匀：

$$
\ell_{\text{uniform}} \triangleq \log \mathbb{E}_{x, y \overset{\text{i.i.d.}}{\sim} p_{\text{data}}} e^{-2\lVert f(x) - f(y) \rVert^2} \tag{3}
$$

其中 $p_{\text{data}}$ 为数据分布。两者恰好对应对比学习的目标：正对应互相靠近，随机样本的表示应尽量散开。本文后续用这两个指标解释 SimCSE 的行为。

---

## 3 无监督 SimCSE（Unsupervised SimCSE）

无监督 SimCSE 的思想非常简单：一批句子 $\{x_i\}_{i=1}^m$，直接令 $x_i^+ = x_i$。要让这个"输入即自身"的正对在对比学习中不退化，作者利用 Transformer 里**独立采样的 Dropout mask**：全连接层与注意力权重上都有 Dropout（默认 $p = 0.1$）。同一句话过编码器两次会得到两份不同的向量。记 $h_i^z = f_\theta(x_i, z)$，$z$ 为随机 dropout mask，则无监督 SimCSE 的损失为：

$$
\ell_i = -\log \frac{\exp\bigl(\operatorname{sim}(h_i^{z_i}, h_i^{z_i'})/\tau\bigr)}{\displaystyle\sum_{j=1}^{N} \exp\bigl(\operatorname{sim}(h_i^{z_i}, h_j^{z_j'})/\tau\bigr)} \tag{4}
$$

注意 $z$ 就是 Transformer 里默认的 Dropout，作者**没有**额外插入新的 Dropout 层。

**Dropout 视作数据增强**。这可以理解为一种"最小数据增强"：正对是同一句话，两个向量的唯一差异来自不同的 Dropout mask。作者把 SimCSE 与常见离散数据增强作对比（表 1），采用 BERT-base，无 STS 训练集参与，在 STS-B 开发集上度量：

| 数据增强 | STS-B |
| :--- | ---: |
| **无（无监督 SimCSE）** | **82.5** |
| 裁剪 10% / 20% / 30% | 77.8 / 71.4 / 63.6 |
| 词删除 10% / 20% / 30% | 75.9 / 72.2 / 68.2 |
| 删除一个词 | 75.9 |
| 无 Dropout | 74.2 |
| 同义词替换 | 77.4 |
| MLM 替换 15% | 62.2 |

（表 1：数据增强的对比。SimCSE 显著优于所有离散扰动。）

作者还比较了不同的无监督目标（表 2）：预测"下一句"、"下 3 句中随机一句"，或者"删除一个词后自己预测自己"。无监督 SimCSE（82.5）大幅胜过预测下一句（67.4）；并且**使用同一编码器**（$f_\theta$）比"两塔独立编码器"$(f_{\theta_1}, f_{\theta_2})$ 明显更好。

**Dropout 概率的影响**。表 3 给出不同 dropout 概率的比较。所有变体都不如默认的 $p = 0.1$。两个极端尤其值得注意：$p = 0$（把 Dropout 关掉）与 "**Fixed 0.1**"（保留默认 dropout 但让两次前向共用同一份 mask）——这两种情况下正对得到**完全相同**的向量，STS-B 分数分别掉到 71.1 与 43.6，出现严重退化。

| $p$ | 0.0 | 0.01 | 0.05 | 0.1 | 0.15 | 0.2 | 0.5 | Fixed 0.1 |
| :-- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| STS-B | 71.1 | 72.6 | 81.1 | **82.5** | 81.4 | 80.5 | 71.0 | 43.6 |

（表 3：Dropout 概率的影响；Fixed 0.1：两次前向共用同一 dropout mask。）

**为何有效——对齐-均匀性视角**。作者每 10 步取一次检查点，在 STS-B 上追踪 $(\ell_{\text{align}}, \ell_{\text{uniform}})$。轨迹显示：从预训练检查点出发，所有变体的**均匀性都在改善**，但 "No dropout" 与 "Fixed 0.1" 的**对齐性急速恶化**（正对没有变化性、被拉得越来越远）；"Delete one word" 也能改善对齐但对均匀性提升有限；只有无监督 SimCSE 在改善均匀性的同时保住了对齐。这解释了为什么 SimCSE 能显著提升 STS——**同时优化两个指标**才是关键。

---

## 4 有监督 SimCSE（Supervised SimCSE）

第 3 节表明 Dropout 噪声足以保证正对 $(x, x^+) \sim p_{\text{pos}}$ 的对齐。本节问：**有监督信号能否让对齐更好？**

**选哪种带标签数据？** 作者比较了四类：

1. **QQP**：Quora 问答重复问题对
2. **Flickr30k**：每张图 5 条人写描述；同一图任意两条视作正对
3. **ParaNMT**：用机器翻译回译构造的大规模释义数据集
4. **NLI**（SNLI + MNLI）：前提-蕴含-中立-矛盾三元标注

对每个数据集，用同一份超参数训练，再在 STS-B 开发集比较。为公平起见，也做了**同样条数**（134k 对）的采样比较。

| 数据集 | 采样 134k | 全量 |
| :--- | ---: | ---: |
| 无监督 SimCSE（1M 句） | — | 82.5 |
| QQP（134k） | 81.8 | 81.8 |
| Flickr30k（318k） | 81.5 | 81.4 |
| ParaNMT（5M） | 79.7 | 78.7 |
| SNLI+MNLI 蕴含（314k） | **84.1** | **84.9** |
| SNLI+MNLI 中立（314k） | 82.6 | 82.9 |
| SNLI+MNLI 矛盾（314k） | 77.5 | 77.6 |
| SNLI+MNLI 全部（942k） | 81.7 | 81.9 |
| SNLI+MNLI 蕴含 + **难负例（矛盾）** | — | **86.2** |
| + ANLI（52k） | — | 85.0 |

（表 4：不同有监督数据集作对比学习正对时的效果；最后两行为最终有监督 SimCSE。）

NLI 数据显著优于其它候选。作者的解释：NLI 标注质量高、人写的假设与前提**词面重合率低**——蕴含对的词面重合率仅 39%，而 QQP 与 ParaNMT 分别是 60% 与 55%。也就是说，NLI 蕴含对在**语义相关但表面不像**这一维度上给了模型最强的信号。

**矛盾对作难负例**。NLI 每条前提天然配有一条"绝对不相关"的矛盾假设。作者把它作为难负例塞进对比学习：把 $(x_i, x_i^+)$ 扩展为 $(x_i, x_i^+, x_i^-)$，其中 $x_i^-$ 是矛盾假设。损失变为：

$$
\ell_i = -\log \frac{\exp\bigl(\operatorname{sim}(h_i, h_i^+)/\tau\bigr)}{\displaystyle\sum_{j=1}^{N}\Bigl[\exp\bigl(\operatorname{sim}(h_i, h_j^+)/\tau\bigr) + \exp\bigl(\operatorname{sim}(h_i, h_j^-)/\tau\bigr)\Bigr]} \tag{5}
$$

引入矛盾对作难负后，STS-B 从 84.9 提到 **86.2**，这是最终的有监督 SimCSE。作者也试了额外加 ANLI（对抗性 NLI）或与无监督 SimCSE 混合，都没有进一步提升。同样，双编码器结构在有监督设置下也会掉分（86.2 → 84.2）。

---

## 5 与各向异性的联系（Connection to Anisotropy）

近年多篇工作观察到语言表示存在**各向异性问题**（Ethayarajh, 2019; Li et al., 2020）——学习到的向量占据整个空间中的一个狭窄"锥形"，极大限制了表达能力。Gao et al. (2019) 证明，输入输出词嵌入绑定训练的语言模型会导致各向异性词嵌入；Ethayarajh (2019) 在预训练上下文表示中进一步观察到此现象；Wang et al. (2020) 显示词嵌入矩阵的奇异值下降非常剧烈，只有几个主奇异值主导，其它几乎为零。

缓解各向异性的常见思路有二：一是**后处理**——消除主成分（Arora et al., 2017; Mu and Viswanath, 2018）或把向量映射为各向同性分布（Li et al., 2020; Su et al., 2021）；二是训练中加**正则化**（Gao et al., 2019; Wang et al., 2020）。本文从理论与实证两侧证明：**对比学习目标本身**就能缓解各向异性——它与"均匀性"是同一件事。

沿用 Wang and Isola (2020) 的推导，当负例数量趋于无穷、$f(x)$ 已归一化时，对比学习的渐近损失可写作：

$$
-\frac{1}{\tau}\,\mathbb{E}_{(x, x^+) \sim p_{\text{pos}}}\bigl[f(x)^\top f(x^+)\bigr] + \mathbb{E}_{x \sim p_{\text{data}}} \log\, \mathbb{E}_{x^- \sim p_{\text{data}}}\bigl[e^{f(x)^\top f(x^-)/\tau}\bigr] \tag{6}
$$

前一项保持正对相似，后一项推开随机负对。若 $p_{\text{data}}$ 在有限样本 $\{x_i\}_{i=1}^m$ 上均匀，$h_i = f(x_i)$，则第二项可由 Jensen 不等式化简为：

$$
\begin{aligned}
\mathbb{E}_x \log \mathbb{E}_{x^-}\bigl[e^{f(x)^\top f(x^-)/\tau}\bigr]
&= \frac{1}{m} \sum_{i=1}^m \log \left(\frac{1}{m}\sum_{j=1}^m e^{h_i^\top h_j / \tau}\right) \\
&\geq \frac{1}{\tau m^2} \sum_{i=1}^m \sum_{j=1}^m h_i^\top h_j \tag{7}
\end{aligned}
$$

令 $W$ 为 $\{h_i\}$ 堆叠的句向量矩阵，第二项优化下界等价于**最小化 $\operatorname{Sum}(WW^\top) = \sum_{i,j} h_i^\top h_j$**。由于向量已归一，$WW^\top$ 对角元恒为 1，$\operatorname{tr}(WW^\top)$（所有特征值之和）为常数。根据 Merikoski (1984)，若 $WW^\top$ 元素都非负（附录中的分布显示大部分情况成立），$\operatorname{Sum}(WW^\top)$ 就是其**最大特征值**的上界。最小化第二项等价于压低最大特征值，进而"抹平"整条奇异谱、让表示更各向同性。

相较 Li et al. (2020) 与 Su et al. (2021) 的后处理只追求各向同性，SimCSE 的对比学习目标里的**第一项** ——正对对齐——是它优于纯后处理的关键（第 7 节实证）。

---

## 6 实验（Experiment）

### 6.1 评测设置（Evaluation Setup）

主评测在 **7 个 STS 任务** 上进行：STS 2012–2016（Agirre et al., 2012, 2013, 2014, 2015, 2016）、STS Benchmark（Cer et al., 2017）与 SICK-Relatedness（Marelli et al., 2014）。所有 STS 实验**完全无监督**——不使用任何 STS 训练集；"有监督 SimCSE"仅指训练时使用了额外带标签的 NLI 数据。

作者在附录 B 中指出，此前论文在 STS 评测协议上存在混乱：

1. 有的用了额外的回归器，有的没有；
2. 有的用 Spearman 相关，有的用 Pearson；
3. 结果聚合方式不同（"all" 聚合 vs. 分数据集加权平均）。

本文统一采用 Reimers and Gurevych (2019) 的做法：**不使用回归器、Spearman 相关、"all" 聚合**，并呼吁社区未来评测统一口径。作者还在附录中给出了本文对既有方法的复现分数与不同口径下的结果。

**训练细节**。从 BERT（uncased）或 RoBERTa（cased）预训练 checkpoint 出发，句向量取 [CLS] 表征（§6.3 会比较不同池化方式）；BERT 原实现在 [CLS] 之上有一个 MLP，本文保留并随机初始化。无监督 SimCSE 训练在 106 条随机采样的英文 Wikipedia 句子上；有监督 SimCSE 训练在 SNLI + MNLI 合并（314k 样本）上。更多细节见附录 A。

### 6.2 主要结果（Main Results）

表 5（下表）给出 7 个 STS 数据集的评测结果。无论是否使用 NLI 监督，SimCSE 都全面推进当时的最好水平：

| 模型 | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | 平均 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **无监督模型** | | | | | | | | |
| GloVe 平均 | 55.14 | 70.66 | 59.73 | 68.25 | 63.66 | 58.02 | 53.76 | 61.32 |
| BERT-base（first-last 平均） | 39.70 | 59.38 | 49.67 | 66.03 | 66.19 | 53.87 | 62.06 | 56.70 |
| BERT-base-flow | 58.40 | 67.10 | 60.85 | 75.16 | 71.22 | 68.66 | 64.47 | 66.55 |
| BERT-base-whitening | 57.83 | 66.90 | 60.90 | 75.08 | 71.31 | 68.24 | 63.73 | 66.28 |
| IS-BERT-base | 56.77 | 69.24 | 61.21 | 75.23 | 70.16 | 69.21 | 64.25 | 66.58 |
| CT-BERT-base | 61.63 | 76.80 | 68.47 | 77.50 | 76.48 | 74.31 | 69.19 | 72.05 |
| **SimCSE-BERT-base** | **68.40** | **82.41** | **74.38** | **80.91** | **78.56** | **76.85** | **72.23** | **76.25** |
| RoBERTa-base（first-last 平均） | 40.88 | 58.74 | 49.07 | 65.63 | 61.48 | 58.55 | 61.63 | 56.57 |
| RoBERTa-base-whitening | 46.99 | 63.24 | 57.23 | 71.36 | 68.99 | 61.36 | 62.91 | 61.73 |
| DeCLUTR-RoBERTa-base | 52.41 | 75.19 | 65.52 | 77.12 | 78.63 | 72.41 | 68.62 | 69.99 |
| **SimCSE-RoBERTa-base** | 70.16 | 81.77 | 73.24 | 81.36 | 80.65 | 80.22 | 68.56 | 76.57 |
| **SimCSE-RoBERTa-large** | **72.86** | **83.99** | **75.62** | **84.77** | **81.80** | **81.98** | **71.26** | **78.90** |
| **有监督模型** | | | | | | | | |
| InferSent-GloVe | 52.86 | 66.75 | 62.15 | 72.77 | 66.87 | 68.03 | 65.65 | 65.01 |
| Universal Sentence Encoder | 64.49 | 67.80 | 64.61 | 76.83 | 73.18 | 74.92 | 76.69 | 71.22 |
| SBERT-base | 70.97 | 76.53 | 73.19 | 79.09 | 74.30 | 77.03 | 72.91 | 74.89 |
| SBERT-base-flow | 69.78 | 77.27 | 74.35 | 82.01 | 77.46 | 79.12 | 76.21 | 76.60 |
| SBERT-base-whitening | 69.65 | 77.57 | 74.66 | 82.27 | 78.39 | 79.52 | 76.91 | 77.00 |
| CT-SBERT-base | 74.84 | 83.20 | 78.07 | 83.84 | 77.93 | 81.46 | 76.42 | 79.39 |
| **SimCSE-BERT-base** | **75.30** | **84.67** | **80.19** | **85.40** | **80.82** | **84.25** | **80.39** | **81.57** |
| SRoBERTa-base | 71.54 | 72.49 | 70.80 | 78.74 | 73.69 | 77.77 | 74.46 | 74.21 |
| SRoBERTa-base-whitening | 70.46 | 77.07 | 74.46 | 81.64 | 76.43 | 79.49 | 76.65 | 76.60 |
| **SimCSE-RoBERTa-base** | 76.53 | 85.21 | 80.95 | 86.03 | 82.57 | 85.83 | 80.50 | 82.52 |
| **SimCSE-RoBERTa-large** | **77.46** | **87.27** | **82.36** | **86.66** | **83.93** | **86.70** | **81.95** | **83.76** |

**表 5：** 7 个 STS 任务上的评测（Spearman 相关，"all" 聚合）。同一预训练编码器下的最好数字加粗。所有对比结果或引自 Reimers and Gurevych (2019) / Zhang et al. (2020)，或由本文作者复现/重新评测。对 BERT-flow 与 whitening，只报告"NLI"设置。

- **BERT-base 上**：无监督 SimCSE 把平均 Spearman 从 72.05（CT-BERT-base）提到 **76.25**，甚至超过部分有监督基线；加入 NLI 后，有监督 SimCSE 达到 **81.57**。
- **RoBERTa 上**：收益更明显，RoBERTa-large 的有监督 SimCSE 达到 **83.76**。
- 附录 E 展示，SimCSE 在 SentEval **迁移任务**上也与既有方法持平或更好，并且引入一个辅助的 MLM 目标可进一步提升。

### 6.3 消融实验（Ablation Studies）

**池化方式**（表 6）。Reimers and Gurevych (2019); Li et al. (2020) 指出，对预训练 BERT，取词向量平均（尤其是首末两层平均）比 [CLS] 好。作者对 [CLS] 表征给了三种设置：a) 保留 MLP；b) 去掉 MLP；c) 训练时用 MLP、测试时去掉。结果：

| Pooler | 无监督 | 有监督 |
| :--- | ---: | ---: |
| [CLS] + MLP | 81.7 | **86.2** |
| [CLS] + MLP（仅训练时） | **82.5** | 85.8 |
| [CLS]（无 MLP） | 80.9 | **86.2** |
| First-last 平均 | 81.2 | 86.1 |

（表 6：无监督/有监督 SimCSE 上不同池化方式的比较；BERT-base，STS-B 开发集。）

无监督 SimCSE 默认 **[CLS] + MLP（仅训练时）**；有监督 SimCSE 默认 **[CLS] + MLP**。

**难负例的权重**（表 7）。为区分难负与批内负例，作者在式 (5) 上加权：

$$
-\log \frac{e^{\operatorname{sim}(h_i, h_i^+)/\tau}}{\displaystyle\sum_{j=1}^{N}\left[e^{\operatorname{sim}(h_i, h_j^+)/\tau} + \alpha^{\mathbb{1}_i^j}\, e^{\operatorname{sim}(h_i, h_j^-)/\tau}\right]} \tag{8}
$$

$\mathbb{1}_i^j \in \{0, 1\}$ 为指示，$i = j$ 时为 1。扫 $\alpha \in \{0.5, 1.0, 2.0\}$，同时也试了把"中立"假设作难负。结果 $\alpha = 1$ 最好；把中立假设作难负没有额外收益。

| Hard neg | 无 | 矛盾 | | | 矛盾 + 中立 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| $\alpha$ | — | 0.5 | 1.0 | 2.0 | 1.0 |
| STS-B | 84.9 | 86.1 | **86.2** | 86.2 | 85.3 |

（表 7：难负例策略的比较。）

---

## 7 分析（Analysis）

**对齐-均匀性可视化**。将各模型放在 $(\ell_{\text{align}}, \ell_{\text{uniform}})$ 平面上，可以直观看到 SimCSE 相对基线的位置。

![图 3：各句向量模型在 alignment-uniformity 平面上的分布](figs/fig03.png)

**图 3（原文 Figure 3）：** 各基于 BERT-base 的句向量模型在 $\ell_{\text{align}}$-$\ell_{\text{uniform}}$ 平面上的分布。点的颜色与括号中的数字表示 7 STS 任务上的平均 Spearman 相关。Next3Sent 指"从后 3 句中随机采样一条作为正对"的对比目标。可以看到：

1. 预训练模型（Avg. BERT、Next3Sent）位于图右上角，对齐尚可、但均匀性极差（各向异性）。
2. BERT-flow / BERT-whitening 极大改善了均匀性，但对齐性有明显退化。
3. 无监督 SimCSE 有效改善均匀性的同时**保住**了预训练的对齐性。
4. 加入 NLI 监督后，有监督 SimCSE 进一步压紧对齐。

**大体规律**：两指标都好的模型，STS 分数也高，与 Wang and Isola (2020) 的结论吻合。附录 F 还展示 SimCSE 有效"抹平"了预训练句向量的奇异谱分布；附录 G 显示 SimCSE 让不同句对之间的余弦相似度**更可区分**（分布更宽、判别更容易设阈值）。

**定性检索**（表 8）。作者取 Flickr30k 中的 15 万条 caption，用 SBERT-base 与有监督 SimCSE-BERT-base 各建索引，任取一句作 query，比较 cosine 相似度 top-3 检索：

**Query：** "A man riding a small boat in a harbor."

- SBERT top-3：
  1. A group of men traveling over the ocean in a small boat.
  2. Two men sit on the bow of a colorful boat.
  3. A man wearing a life jacket is in a small boat on a lake.
- SimCSE top-3：
  1. A man on a moored blue and white boat.
  2. A man is riding in a boat on the water.
  3. A man in a blue boat on the water.

**Query：** "A dog runs on the green grass near a wooden fence."

- SBERT top-3：
  1. A dog runs on the green grass near a grove of trees.
  2. A brown and white dog runs through the green grass.
  3. The dogs run in the green field.
- SimCSE top-3：
  1. The dog by the fence is running on the grass.
  2. Dog running through grass in fenced area.
  3. A dog runs on the green grass near a grove of trees.

（表 8：两模型 top-3 检索定性对比。）

肉眼可见 SimCSE 的检索更好抓住了"栅栏"、"船上"等**具体细节**，而 SBERT 更容易被"狗跑草地"这样的**大类别**主导。

---

## 8 相关工作（Related Work）

早期句向量沿"分布假设"思路，通过预测上下文句子学习表示（Kiros et al., 2015; Hill et al., 2016; Logeswaran and Lee, 2018）。Pagliardini et al. (2018) 显示把 word2vec 思想扩展到 n-gram 就能得到强 baseline。近年（与本文同期或稍后）多篇工作采用对比目标，通过对同一句子做数据增强或复制不同视图作正对（Zhang et al., 2020; Giorgi et al., 2021; Wu et al., 2020; Meng et al., 2021; Carlsson et al., 2021; Kim et al., 2021; Yan et al., 2021）。相较这些工作，SimCSE 用**最简单**的思路——同一句话不同 dropout 视图——在 STS 上表现最好。

有监督句向量方面，Conneau et al. (2017) 提出在 NLI 数据上微调 Siamese 网络，被后续多篇扩展到不同编码器与预训练模型（Cer et al., 2018; Reimers and Gurevych, 2019）。Wieting and Gimpel (2018); Wieting et al. (2020) 显示双语对齐与回译语料同样对语义相似度学习有用。还有一条线专注**正则化嵌入**（Li et al., 2020; Su et al., 2021; Huang et al., 2021）以缓解表示退化。

---

## 9 结论（Conclusion）

本文提出 SimCSE，一个极简的对比学习句向量框架。**无监督**下用 Dropout 造正对，**有监督**下用 NLI 蕴含对作正、矛盾对作难负。两个版本都显著推进了 STS 上的最好水平。作者还从**对齐-均匀性**与**奇异谱**两个角度给出理论解释；并在评测协议、消融实验、可视化分析等多方面提供了详细材料，供后续研究参考。

---

## 附录索引（Appendix Highlights）

原论文附录较长，此处不逐段翻译，只列出对复现最重要的几点，方便对着 PDF 查：

- **A** 训练超参数：优化器 AdamW；学习率 3e-5（unsup）/ 1e-5（sup）；batch size 64；温度 $\tau = 0.05$；训练 1 epoch。
- **B** 评测协议差异表：Spearman vs Pearson、是否用 regressor、"all" vs macro 聚合的详细对照。
- **C** 与既有方法的详细对比与复现表。
- **D** 消融：MLM 辅助目标、归一化、温度扫描。
- **E** SentEval 迁移任务：MR / CR / SUBJ / MPQA / SST-2 / TREC / MRPC；SimCSE 与 SBERT 打平或略好，加辅助 MLM 会更好。
- **F** 奇异值分布：SimCSE 能显著抹平预训练 BERT 的奇异谱。
- **G** 余弦相似度分布：SimCSE 的分数动态范围更大，STS 判别更容易设阈值。

---

*翻译约定：句向量（sentence embedding）、对比学习（contrastive learning）、正对（positive pair）、难负例（hard negative）、对齐（alignment）、均匀性（uniformity）、各向异性（anisotropy）、批内负例（in-batch negative）、自然语言推理（NLI）、蕴含（entailment）、矛盾（contradiction）、语义文本相似度（STS）、Dropout / MLP / MLM 按业内惯例不译。*
