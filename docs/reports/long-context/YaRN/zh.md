# YaRN：大型语言模型的高效上下文窗口扩展

> 原文：[arXiv:2309.00071v3](https://arxiv.org/html/2309.00071v3)，作者及机构见下。
> 中文全译稿：以机器初译为底稿，统一技术术语，并校订摘要和关键方法段落；覆盖主文、附录、算法、表格、图注和脚注。参考文献保留原始书目，原图内部文字与代码保留英文。英文转写见 [en.md](en.md)，方法分析另见 [YaRN详解.md](YaRN详解.md)。

> 原作作者 Bowen Peng、Jeffrey Quesnelle、Honglu Fan、Enrico Shippole；原作采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本文件为中文翻译并调整版式。论文 2023 年 8 月首次发布，发表于 ICLR 2024；此处对应 2026-02-06 的 arXiv v3 修订版。

## 作者与机构

Bowen Peng Jeffrey Quesnelle Honglu Fan Enrico Shippole

Affiliations: Nous Research; EleutherAI; University of Geneva.

## 摘要

旋转位置嵌入（RoPE）已经能够在基于 Transformer 的语言模型中有效编码位置信息，但这些模型无法泛化到超出训练序列长度的输入。我们提出 YaRN（Yet another RoPE extensioN method），一种计算高效的上下文窗口扩展方法，相比既有方法所需 token 数量减少约 10 倍、训练步数减少约 2.5 倍。通过 YaRN，我们表明 LLaMA 模型能够有效利用并外推到远超原始预训练所允许的上下文长度，同时超越此前上下文窗口扩展方法的最佳水平。此外，我们还展示了 YaRN 外推到超过微调数据有限上下文长度的能力。代码见 https://github.com/jquesnelle/yarn 。

## 引言

基于 Transformer 的大型语言模型（Vaswani 等人，2017）（LLM）已成为许多自然语言处理（NLP）任务的几乎普遍的选择，其中上下文学习（ICL）等远程能力至关重要。

在执行 NLP 任务时，由训练过程决定的序列最大长度（上下文窗口）一直是预训练 LLM 的主要限制之一。能够通过少量微调（或无需微调）来动态扩展上下文窗口已经变得越来越可取。为此，Transformer的位置编码是讨论的中心。

最初的 Transformer 架构使用绝对正弦位置编码，后来改进为可学习的绝对位置编码（Gehring et al., 2017）。从那时起，相对位置编码方案（Shaw et al., 2018）进一步提高了 Transformers 的性能。目前，最流行的相对位置编码是 T5 相对偏差 (Roberts et al., 2019)、RoPE (Su et al., 2022)、XPos (Sun et al., 2022) 和 ALiBi (Press et al., 2022)。

位置编码的一个反复出现的限制是无法泛化训练期间看到的上下文窗口。虽然 ALiBi 等一些方法能够进行有限的泛化，但没有一种方法能够泛化到明显长于其预训练长度的序列（Kazemnejad 等人，2023）。

已经做了一些工作来克服这种限制。 ( Chen et al., 2023 ) 和同时 ( kaiokendev, 2023 ) 提出通过位置插值 (PI) 稍微修改 RoPE 并在少量数据上进行微调来扩展上下文长度。作为替代方案，（bloc97，2023b）通过考虑高频损失提出了“NTK-aware”插值。此后，提出了“NTK-aware”插值的两种改进，各有侧重：

- 用于预训练模型的“动态NTK”插值方法（emozilla，2023），无需微调。

- “NTK-by-parts”插值方法（bloc97, 2023a）在对少量较长上下文数据进行微调时效果最佳。

“NTK-aware”插值和“动态 NTK”插值已经出现在开源模型中，例如 Code Llama (Rozière et al., 2023)（使用“NTK-aware”插值）和 Qwen 7B (17)（使用“动态 NTK”）。

在本文中，除了对之前未发表的“NTK-aware”、“Dynamic NTK”和“NTK-by-parts”插值的工作进行了完整的说明之外，我们还提出了 YaRN（另一种 RoPE 扩展方法），这是一种改进的方法，可以有效地扩展使用旋转位置嵌入（RoPE）训练的模型的上下文窗口，包括 LLaMA（Touvron 等人，2023a）、 GPT-NeoX（Black 等人，2022）和 PaLM（Chowdhery 等人，2022）系列模型。

不同方法之间的关系以及它们如何演变为YaRN可以总结为下图：

![Figure S1.F1](figs/S1-F1-01.png)

图 1：不同插值方法之间关系的概述。

在对小于 $\sim$ 0.1% 的原始预训练数据进行微调后，YaRN 在上下文窗口扩展方面达到了最先进的性能。同时，通过与称为动态缩放的推理时间技术相结合，Dynamic-YaRN 无需任何微调即可实现超过 2 倍的上下文窗口扩展。

## 背景及相关工作

### 旋转位置嵌入

我们工作的基础是 (Su et al., 2022) 中引入的旋转位置嵌入 (RoPE)。我们研究隐藏层，其中隐藏神经元集用 $D$ 表示。给定一个向量序列 ${\bm{x}}_{1},\cdots,{\bm{x}}_{L}\in\mathbb{R}^{|D|}$ ，遵循 ( Su et al., 2022 ) 的符号，注意力层首先将向量转换为查询向量和键向量：

$$
\displaystyle{\bm{q}}_{m}=f_{q}({\bm{x}}_{m},m)\in\mathbb{R}^{|D|},~{\bm{k}}_{n}=f_{k}({\bm{x}}_{n},n)\in\mathbb{R}^{|D|}.\tag{1}
$$

接下来，注意力权重计算为

$$
\text{softmax}(\dfrac{{\bm{q}}_{m}^{T}{\bm{k}}_{n}}{\sqrt{|D|}}),\tag{2}
$$

其中 ${\bm{q}}_{m},{\bm{k}}_{n}$ 被视为列向量，因此 ${\bm{q}}_{m}^{T}{\bm{k}}_{n}$ 只是欧几里德内积。在 RoPE 中，我们首先假设 $|D|$ 是偶数，并将嵌入空间和隐藏状态识别为复向量空间：

$$
\mathbb{R}^{|D|}\cong\mathbb{C}^{|D|/2}
$$

其中内积 ${\bm{q}}^{T}{\bm{k}}$ 成为标准 Hermitian 内积 $\text{Re}({\bm{q}}^{*}{\bm{k}})$ 的实部。更具体地说，同构交错实部和复数部分

$$
\displaystyle\big(({\bm{x}}_{m})_{1},\cdots,({\bm{x}}_{m})_{|D|}\big)\mapsto\big(({\bm{x}}_{m})_{1}+i({\bm{x}}_{m})_{2},\cdots,(({\bm{x}}_{m})_{|D|-1}+i({\bm{x}}_{m})_{|D|})\big),\tag{3}
$$

$$
\displaystyle\big((\textbf{q}_{m})_{1},\cdots,(\textbf{q}_{m})_{|D|}\big)\mapsto\big((\textbf{q}_{m})_{1}+i(\textbf{q}_{m})_{2},\cdots,((\textbf{q}_{m})_{|D|-1}+i(\textbf{q}_{m})_{|D|})\big).\tag{4}
$$

为了将嵌入 ${\bm{x}}_{m},{\bm{x}}_{n}$ 转换为查询向量和键向量，我们首先给出 $\mathbb{R}$ 线性运算符

$$
{\bm{W}}_{q},{\bm{W}}_{k}:\mathbb{R}^{|D|}\rightarrow\mathbb{R}^{|D|}.
$$

让 $\bm{\theta}=\text{diag}(\theta_{1},\cdots,\theta_{|D|/2})$ 。在复坐标中，我们定义

$$
f_{{\bm{W}}}({\bm{x}}_{m},m,\bm{\theta})=e^{im\bm{\theta}}{\bm{W}}{\bm{x}}_{m},\tag{5}
$$

对于任何线性算子 ${\bm{W}}$ 。 RoPE 中的函数 $f_{q},f_{k}$ 由下式给出

$$
f_{q}=f_{{\bm{W}}_{q}},~f_{k}=f_{{\bm{W}}_{k}}.\tag{6}
$$

其中 $\theta_{d}=b^{-2d/|D|}$ 和 $b=10000$ 。这样，RoPE 将每个（复值）隐藏神经元与单独的频率 $\theta_{d}$ 相关联。这样做的好处是查询向量和键向量之间的点积仅取决于相对距离 $m-n$ 。

在后面的讨论中，上下文长度插值通常旨在修改方程 Eq. 5.要为这些讨论建立统一的约定，请注意修改 $f^{\prime}_{{\bm{W}}}$ 可以采用以下形式：

$$
\displaystyle f^{\prime}_{{\bm{W}}}({\bm{x}}_{m},m,\bm{\theta})=f_{{\bm{W}}}({\bm{x}}_{m},g(m),{\bm{h}}(\bm{\theta})),\tag{7}
$$

其中 $g(m)$ 是实数之间的映射，并且 ${\bm{h}}(\bm{\theta})$ 根据函数 $h$ 由 $\text{diag}(h(\theta_{1}),\cdots,h(\theta_{|D|/2}))$ 统一作用于对角矩阵 $\bm{\theta}$ 的条目。 $g$ 和 $h$ 是方法相关函数。

在接下来的章节中，当我们介绍一种新的插值方法时，形式为： 7、我们只指定函数$g(m)$和$h(\theta_{d})$。

### 附加符号

给定预训练的最大上下文长度 $L$ ，我们的目标是将其扩展到 $L^{\prime}>L$ ，无论是否进行微调。我们引入由 $s=\frac{L^{\prime}}{L}$ 定义的比例因子 $s$ 的概念。

为了方便讨论，我们还引入了与 RoPE 的第 $d$ 隐藏维度相关的波长 $\lambda_{d}$，如下所示：

$$
\displaystyle\lambda_{d}=\dfrac{2\pi}{\theta_{d}}=2\pi b^{\frac{2d}{|D|}}.\tag{8}
$$

波长描述了尺寸 $d$ 处的旋转位置嵌入执行完整旋转 ( $2\pi$ ) 所需的令牌长度。

### 相关工作

位置插值（PI）是 Chen 等人提出的扩展 RoPE 上下文长度的早期工作之一。 (2023) ，同时 kaiokendev (2023) 。根据方程的符号。 7、正在设置

$$
g(m)=s\cdot m,~{\bm{h}}(\bm{\theta})=\bm{\theta},\tag{9}
$$

> 译注：本式照录原文。按 $s=L^{\prime}/L$ 的定义，PI 应为 $g(m)=m/s$，与附录式 (16) 的 $mL/L^{\prime}$ 一致；正文此式的 $s\cdot m$ 与之不一致。

其中 $s$ 是比例因子 $\frac{L^{\prime}}{L}$ 。我们在附录 A.1 中包含了一些详细信息。

ReRoPE（Su，2023）还旨在扩展使用 RoPE 预训练的现有模型的上下文大小，并声称“无限”上下文长度，无需任何微调。在 Llama 2 13B 模型上，随着上下文长度增加到 16k，损失单调递减支持了这一说法。它通过修改注意力机制来实现上下文扩展，因此并不是纯粹的嵌入插值方法。由于它目前与 Flash Attention 2 (Dao, 2023) 不兼容，并且在推理过程中需要两次注意力传递，因此我们不考虑将其进行比较。

与我们的工作同时，LM-Infinite（Han et al., 2023）提出了与 YaRN 类似的想法，但侧重于非微调模型的“即时”长度泛化。由于它们还修改了模型的注意力机制，因此它不是嵌入插值方法，并且不能立即与 Flash Attention 2 兼容。

## 方法

尽管 PI 同等地拉伸所有 RoPE 维度，但我们发现 PI 描述的理论插值界限（Chen et al., 2023）不足以预测 RoPE 和 LLM 内部嵌入之间的复杂动态。在下面的小节中，我们将描述我们单独识别和解决的 PI 的主要问题，以便为读者提供我们协同使用的每种方法的背景、起源和理由，以获得完整的 YaRN 方法。

### 高频信息损失与 NTK-aware 插值

如果我们仅从信息编码的角度来看待旋转位置嵌入（RoPE），（Tancik et al., 2020）中使用神经正切核（NTK）理论表明，如果输入维度较低并且相应的嵌入缺乏高频分量，则深度神经网络难以学习高频信息。在这里我们可以看到相似之处：令牌的位置信息是一维的，而 RoPE 将其扩展为 n 维复向量嵌入。 RoPE 在许多方面与傅里叶特征（Tancik et al., 2020）非常相似，因为可以将 RoPE 定义为傅里叶特征的特殊一维情况。

在位置插值 (PI) 的情况下，当我们将所有维度平均拉伸 $s$ 因子时，它会消除 RoPE 的高频分量。随着比例因子 $s$ 的增长，这种退化现象会更加严重，并且在某个时刻，网络将无法恢复。之前使用 PI 进行的微调（kaiokendev，2023）（Chen 等人，2023）（Together.ai，2023）（Quesnelle 等人，2023）只能在 LLM 的输出开始降级之前实现大约 $s=8$ 的缩放因子，即使在微调之后也是如此。

为了缓解这个问题，（bloc97，2023b）开发了“NTK-aware”插值。我们不是将 RoPE 的每个维度均等地缩放 $s$ 因子，而是通过减少高频和增加低频来将插值压力分散到多个维度。可以通过多种方式获得这种转换，但最简单的方法是对 $\theta$ 的值执行基本更改。详细信息在附录 A.2 中描述，该方法已被一些开源采用 [^footnote1] 。

这种“NTK-aware”缩放的一个主要问题是，很难确定应该使用哪种最佳基础来进行 $s$ 次的预期上下文扩展。用于“NTK-aware”插值的最佳基础通常必须根据经验找到，这显着增加了获得成功的微调模型的难度和成本。尽管有其局限性，NTK 理论的观察结果是有效的，并且在下一节介绍的“按部分 NTK”插值中仍然保留并以不同的方式执行以下想法。

### 局部相对距离损失与 NTK-by-parts 插值

为了理解为什么“NTK-aware”插值比 PI 效果更好并消除其缺点，我们必须仔细研究 RoPE。在本节中，我们重点考虑方程 1 中定义的波长 $\lambda_{d}$。 RoPE公式中的8。为简单起见，我们省略了 $\lambda_{d}$ 中的下标 $d$，并鼓励读者将 $\lambda$ 视为任意周期函数的波长。

理论上，由于 RoPE 是一种相对位置嵌入，因此它无法泛化到未见过的较长上下文大小，这应该是相当令人惊讶的。然而，我们可以证明，在实践中，RoPE 不仅仅编码相对位置。我们可以观察到，给定上下文大小 $L$ ，有一些维度 $d$ 的波长比预训练期间看到的最大上下文长度长（ $\lambda>L$ ），这表明某些维度的旋转嵌入可能不会均匀分布在旋转域中（即不会对整个训练上下文大小执行完整旋转）。在这种情况下，我们假设具有唯一的位置对 [^footnote2] 意味着绝对位置信息在这些维度中保持完整。相反，当波长较短时，网络只能访问相对位置信息。

根据这些观察，保持仅编码相对位置信息的维度不变十分重要，因为它们对于网络区分相邻 token 的相对顺序至关重要。与此同时，仅编码绝对位置信息的维度应始终采用插值，因为更大的距离会落到分布之外。与其像 NTK-aware 插值那样任意改变基数（其作用与这里的思路大致相似），我们可以构造一种明确且有针对性的插值方法，将上述因素统一考虑。

换句话说，

- 如果波长$\lambda$ 远小于上下文大小$L$，我们不进行插值；

- 如果波长 $\lambda$ 等于或大于上下文大小 $L$ ，我们只想进行插值并避免任何外推（与之前的“NTK-aware”方法不同）；

- 中间的尺寸可以同时具有两者，类似于“NTK-aware”插值。

结果，更方便地引入原始上下文尺寸$L$与波长$\lambda$之间的比率$r=\frac{L}{\lambda}$。该比率表示给定固定预训练上下文长度 $L$ 时某个 RoPE 维度进行的旋转次数。在第 $d$ 隐藏状态中，比率 $r$ 取决于 $d$，如下所示：

$$
r(d)=\dfrac{L}{\lambda_{d}}=\dfrac{L}{2\pi b^{\frac{2d}{|D|}}}.\tag{10}
$$

为了定义上述不同插值策略的边界，我们引入了两个额外的参数 $\alpha,\beta$ 。所有隐藏维度 $d$ 其中 $r(d)<\alpha$ 是我们按比例 $s$ 线性插值的维度（与 PI 完全相同，避免任何外推），而 $d$ 其中 $r(d)>\beta$ 是我们根本不插值的维度。将斜坡函数 $\gamma$ 定义为

$$
\displaystyle\gamma(r)=\begin{cases}0,&\text{if }r<\alpha\\
1,&\text{if }r>\beta\\
\dfrac{r-\alpha}{\beta-\alpha},&\text{otherwise}.\end{cases}\tag{11}
$$

借助斜坡函数，“NTK-by-parts”方法可以描述如下。

#### 定义1

“NTK-by-parts”插值是使用等式 1 的 RoPE 的修改。 7.具有以下功能[^footnote3]。

$$
\displaystyle g(m) \displaystyle=m\tag{12}
$$

$$
\displaystyle h(\theta_{d}) \displaystyle=\Big(1-\gamma\big(r(d)\big)\Big)\frac{\theta_{d}}{s}+\gamma\big(r(d)\big)\theta_{d}.\tag{13}
$$

$\alpha$ 和 $\beta$ 的值应根据具体情况进行调整。例如，我们通过实验发现，对于 Llama 系列模型，$\alpha$ 和 $\beta$ 的良好值为 $\alpha=1$ 和 $\beta=32$ 。

使用本节中描述的技术，生成的方法的变体以“NTK-by-parts”插值的名称发布（bloc97，2023a）。这种改进的方法比之前的 PI (Chen et al., 2023) 和“NTK-aware”3.1 插值方法表现得更好，无论是使用非微调模型还是微调模型，如 (bloc97, 2023a) 和第 4.2 节所示。

### YaRN

除了前面的插值技术之外，我们还观察到，在注意 softmax 之前在 logits 上引入温度 $t$ 对困惑度具有均匀的影响，无论数据样本和扩展上下文窗口上的标记位置如何（参见附录 A.3）。更准确地说，而不是等式。 2 ，我们将注意力权重的计算修改为

$$
\text{softmax}\left(\dfrac{{\bm{q}}_{m}^{T}{\bm{k}}_{n}}{t\sqrt{|D|}}\right).\tag{14}
$$

把 RoPE 重参数化为一组二维矩阵，为实现注意力缩放带来明显便利：只需将复数旋转位置嵌入乘以常数 $\sqrt{1/t}$，即可通过长度缩放技巧使 ${\bm{q}}_{m}$ 与 ${\bm{k}}_{n}$ 同时乘以相同系数。这样，无需修改注意力代码，YaRN 就能有效改变注意力机制。此外，旋转位置嵌入可以预先生成并在所有前向传播中复用，因此这种修改在推理和训练中都没有额外开销。将其与 NTK-by-parts 插值结合，就得到 YaRN 方法。

#### 定义2

通过“YaRN方法”，我们指的是等式1中注意力缩放的组合。 14 和第 3.2 节中介绍的“NTK-by-parts”插值。

对于 LLaMA 和 Llama 2 型号，我们建议使用以下值：

$$
\displaystyle\sqrt{\frac{1}{t}}=0.1\ln({s})+1.\tag{15}
$$

上述公式来自如下拟合：在不微调的 LLaMA 7B、13B、33B 和 65B 模型上使用 NTK-by-parts 方法（第 3.2 节），针对不同扩展倍率 $s$，寻找困惑度最低时的 $\sqrt{1/t}$ 并拟合其关系。我们注意到，相同的 $t$ 值也较好地适用于 Llama 2（7B、13B 和 70B）模型。这说明，熵增加的现象和温度常数 $t$ 可能具有一定普适性，能够在某些模型和训练数据之间迁移。

YaRN 方法结合了我们所有的发现，并在微调和非微调场景中超越了之前的所有方法。由于占用空间小，YaRN 允许与修改注意力机制的库直接兼容，例如 Flash Attention 2（Dao，2023）。

### 动态缩放与 Dynamic NTK 插值

在许多用例中，使用从 $1$ 到最大上下文大小的不同序列长度执行多次前向传递。一个典型的例子是自回归生成，其中序列长度在每一步后增加 $1$。有两种方法可以应用使用比例因子 $s$ 的插值方法（包括 PI、“NTK-aware”、“NTK-by-parts”和 YaRN）：

- 1.在整个推理周期中，嵌入层是固定的，包括比例因子$s=L^{\prime}/L$，其中$L^{\prime}$是扩展上下文大小的固定数量。

- 2. 在每次前向传递中，位置嵌入都会更新比例因子 $s=\text{max}(1,l^{\prime}/L)$，其中 $l^{\prime}$ 是当前序列的序列长度。

(1) 的问题是，当序列长度小于 $L$ 时，模型可能会出现性能折扣，而当序列长度大于 $L^{\prime}$ 时，模型可能会出现突然退化。但是，通过按照 (2) 进行动态缩放，它允许模型在达到训练的上下文限制 $L^{\prime}$ 时优雅地降级，而不是立即中断。我们将这种推理时间方法称为动态缩放方法。当它与“NTK-aware”插值相结合时，我们将其称为“动态NTK”插值。它首次作为 Reddit 帖子在 (emozilla, 2023) 中公开出现。

一个值得注意的事实是，“动态 NTK”插值在 $L$ 上预训练的模型上运行得非常好，无需任何微调 ( $L^{\prime}=L$ )。附录 B.7 中的实验支持了这一点。

在多次前向传播中，通常使用 KV cache（Chen, 2022）复用先前的键值向量，以提高整体效率。需要指出的是，在某些会缓存旋转位置嵌入的实现中，必须谨慎修改代码，才能让动态缩放与 KV cache 正确配合。正确实现应缓存施加旋转位置嵌入之前的 KV 表示，因为当 $s$ 改变时，每个 token 的 RoPE 都会改变。

## 实验

### 训练

我们大致遵循 Chen 等人中概述的培训和评估程序。 （2023）。

为了训练 128k 上下文窗口大小模型，我们扩展了 Llama 2（Touvron 等人，2023b）7B 和 13B 参数模型。除了第 3.3 节中所述的 $s=16$ 和 $s=32$ 的嵌入频率计算之外，LLaMA 模型架构没有进行任何更改。

我们使用了 $2\times 10^{-5}$ 的学习率，没有权重衰减和 20 步的线性预热以及 AdamW（Loshchilov 和 Hutter，2019）$\beta_{1}=0.9$ 和 $\beta_{2}=0.95$。对于 $s=16$ 模型，我们在 PG19 数据集（Rae 等人，2023）上使用 PyTorch（Paszke 等人，2019）、完全分片数据并行（Zhao 等人，2023）和 Flash Attention 2（Dao，2023）对全局批量大小 $64$ 进行了 400 个步骤的微调。 2020）分成 64k 个段，用 BOS 和 EOS token预订。对于 $s=32$，我们遵循相同的过程，但由于计算限制，我们从完成的 $s=16$ 检查点开始，仅进行额外 200 个步骤的训练。请注意，$s=32$ 模型也使用 64k 上下文数据进行训练，但我们在第 4.2 节中表明它能够外推到 128k 的上下文大小。

对于消融研究，我们使用 LLaMA 7B 模型。它具有与较新的 Llama 2 模型相同的架构，只是预训练上下文窗口大小 [^footnote4] 更短，这减少了计算要求并允许更快的训练和评估。训练过程与 128k 模型类似，但我们将 PG19 数据集分成 32k 段，并使用 $s=16$ 进行 400 个步骤的训练。如图 2 所示，与其他插值技术相比，YaRN 在训练期间收敛速度更快，并且始终具有较低的损失。

![Figure S4.F2](figs/S4-F2-01.png)

![Figure S4.F2](figs/S4-F2-02.png)

图 2：使用不同插值技术扩展到 32k 上下文大小的 LLaMA 7B 模型的训练损失曲线。右侧的图表已放大。

### 长序列语言建模

为了评估长序列语言建模性能，我们使用 GovReport（Huang et al., 2021）和 Proof-pile（Azerbayev et al., 2022）数据集，这两个数据集都包含许多长序列样本。对于所有评估，仅使用两个数据集的测试分割。所有困惑度评估均使用 Press 等人的滑动窗口方法计算。 (2022) 和 $S=256$ ，它考虑了整个文档的困惑度贡献，即使模型的上下文窗口较短。

首先，我们从证明堆中选择 10 个随机样本，每个样本至少有 128k 个令牌，并评估每个样本在从 2k 个令牌到 128k 个令牌的序列长度以 2k 步截断时的困惑度。表 1 显示了经过微调的 Llama 2 $s=16$ 和 $s=32$ 模型的长序列性能。我们证明 YaRN 能够泛化和外推到未见过的上下文长度，并从迁移学习中受益，因为 $s=32$ 模型仅使用具有 64k 数据的 $s=16$ 检查点进一步训练 200 个步骤，并且能够外推到 128k 上下文。

| 模型 | 扩展 | 是否 | 训练 | 扩展 | 评测上下文窗口长度 | 评测上下文窗口长度 | 评测上下文窗口长度 | 评测上下文窗口长度 | 评测上下文窗口长度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 规模 | 方法 | 微调 | 步数 | 倍率 $s$ | 8192 | 16384 | 32768 | 65536 | 131072 |
| 7B | YaRN | ✓ | 400 | 4k $\times$ 16 | 3.51 | 2.99 | 2.65 | 2.42 | $>10^{1}$ |
| 7B | YaRN | ✓ | 400+200 | 4k $\times$ 32 | 3.56 | 3.04 | 2.70 | 2.45 | 2.37 |
| 13B | YaRN | ✓ | 400 | 4k $\times$ 16 | 3.25 | 2.79 | 2.50 | 2.29 | $>10^{1}$ |
| 13B | YaRN | ✓ | 400+200 | 4k $\times$ 32 | 3.29 | 2.83 | 2.53 | 2.31 | 2.24 |

表 1：通过 YaRN 扩展的 Llama 2 模型上 10 个 128k 证明堆文档的滑动窗口困惑度 ($S=256$)。我们展示了成功的上下文大小外推和从 64k 到 128k 的迁移学习，仅将 64k 上下文作为训练数据。

为了进一步确认 YaRN 的有效性，我们比较了左侧图 3 和附录 B.1 中表 5 中的所有四种插值方法，作为消融研究。当使用相同数量的训练步骤时，在非微调和微调场景中，YaRN 始终优于其他方法（困惑度更低）。我们还在附录 B.2 中证明了 YaRN 与 PI 相比具有更好的训练效率。与开放模型的更多比较可以在附录 B.3 中找到。

![Figure S4.F3](figs/S4-F3-01.png)

![Figure S4.F3](figs/S4-F3-02.png)

图 3：针对使用不同插值技术在 400 个步骤中微调至 32k 上下文的微调 LLaMA 7B 模型，十个 128k 证明堆文档的滑动窗口困惑度 ($S=256$) 和不同提示长度下的密钥检索准确性。在相同的训练预算下，YaRN 的性能优于其他插值方法。

### 口令检索

Mohtashami 和 Jaggi（2023）中定义的密钥检索任务衡量模型从大量无意义文本中检索简单密钥（即五位数字）的能力。为了评估经过微调的 32k LLaMA 7B 模型，我们对密钥检索任务进行了 50 次迭代，并将密钥放置在评估上下文窗口中均匀分布的随机位置，提示长度从 2k 到 32k 不等。在给定类似的训练预算时，与其他插值方法相比，YaRN 获得了更高的分数，如右图 3 所示。 Llama 2 模型的更多结果和比较显示在附录 B.5 中。

### 标准化基准

Hugging Face 开放法学硕士排行榜（Hugging Face，2023）通过一组标准化的四个公共基准对众多法学硕士进行了比较。具体来说，我们使用 25-shot ARC-Challenge (Clark et al., 2018)、10-shot HellaSwag (Zellers et al., 2019)、5-shot MMLU (Hendrycks et al., 2021) 和 0-shot TruthfulQA (Lin et al., 2022)。

为了测试上下文扩展下模型的短上下文性能的退化，我们使用该套件评估了 Llama 2 和 32k LLaMA 7B 模型，并将其与基线的既定分数进行比较。结果总结于表10和表3中。 Llama 2 模型的更多结果显示在附录 B.6 中。

| 扩展 | 是否 | 扩展 | ARC-c | Hellaswag | MMLU | TruthfulQA |
| --- | --- | --- | --- | --- | --- | --- |
| 方法 | 微调 | 倍率 $s$ | ARC-c | Hellaswag | MMLU | TruthfulQA |
| 无 | ✗ | - | 51.0 | 77.8 | 35.7 | 34.3 |
| PI | ✓ | 2k $\times$ 16 | 44.8 | 70.2 | 25.9 | 34.1 |
| NTK-aware | ✓ | 2k $\times$ 16 | 47.4 | 73.9 | 27.7 | 32.6 |
| NTK-by-parts | ✓ | 2k $\times$ 16 | 48.5 | 76.6 | 32.7 | 33.4 |
| YaRN | ✓ | 2k $\times$ 16 | 48.1 | 77.2 | 30.0 | 35.1 |

表 2：在 Hugging Face Open LLM 基准套件上与原始 LLaMA 7B 基准相比，上下文窗口扩展方法的性能（微调了 400 个步骤）。

| 模型 | 扩展 | 是否 | 扩展 | ARC-c | Hellaswag | MMLU | TruthfulQA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 规模 | 方法 | 微调 | 倍率 $s$ | ARC-c | Hellaswag | MMLU | TruthfulQA |
| 7B | 无 | ✗ | - | 53.1 | 77.8 | 43.8 | 39.0 |
| 7B | YaRN | ✓ | 4k $\times$ 16 | 52.3 | 78.8 | 42.5 | 38.2 |
| 7B | YaRN | ✓ | 4k $\times$ 32 | 52.1 | 78.4 | 41.7 | 37.3 |
| 13B | 无 | ✗ | - | 59.4 | 82.1 | 55.8 | 37.4 |
| 13B | YaRN | ✓ | 4k $\times$ 16 | 58.1 | 82.3 | 52.8 | 37.8 |
| 13B | YaRN | ✓ | 4k $\times$ 32 | 58.0 | 82.2 | 51.9 | 37.3 |

表 3：YaRN 在 Hugging Face Open LLM 基准套件上的性能与原始 Llama 2 基准相比。

我们观察到，YaRN 模型与其各自的 Llama 2 基线之间的性能下降很小。由于我们用于微调的 PG19 数据集（Rae 等人，2020）与用于 LLaMA 和 Llama 2 模型的原始预训练数据有很大不同，因此存在一些差异是可以预料的。我们还观察到，YaRN $s=16$ 和 $s=32$ 模型之间的分数平均下降了 0.49%，并且可以得出结论，从 64k 到 128k 的迭代扩展导致的性能损失可以忽略不计。

### 计算效率

鉴于当上下文窗口大小固定为预设长度 $L$ 时，旋转位置嵌入在训练和推理过程中被缓存，与以前的上下文扩展方法相比，修改旋转位置嵌入的插值不会产生额外的计算或内存成本，这是本工作中概述的所有四种插值方法的情况。与其他方法相比，YaRN 在训练过程中收敛速度最快，因此计算效率最高，如表 4 所示。

| 模型 | 模型 | 扩展 | 扩展 | 有效 | 训练用时 |
| --- | --- | --- | --- | --- | --- |
| 规模 | 名称 | 方法 | 倍率 $s$ | 上下文 | A100 GPU 小时 |
| 7B | LLaMA YaRN | YaRN | 2k $\times$ 16 | 32k | 128 |
| 7B | Llama 2 YaRN | YaRN | 4k $\times$ 16 | 64k | 256 |
| 7B | Llama 2 YaRN | YaRN | 4k $\times$ 32 | 128k | 256 + 128 |
| 7B | ( Chen et al., 2023 ) | PI | 2k $\times$ 8 | 16k | 640 |
| 7B | ( Together.ai, 2023 ) | PI | 4k $\times$ 8 | 32k | ? |
| 7B | ( Xiong et al., 2023 ) | NTK-aware | 4k $\times$ 44.2 | $\approx$ 50k | 64000 |
| 7B | ( Rozière et al., 2023 ) | NTK-aware | 4k $\times$ 88.6 | $\approx$ 100k | 6400 |

表 4：使用不同扩展方法的不同开放和封闭模型的 A100 小时训练时间比较。

## 结论

总之，我们已经证明 YaRN 改进了所有现有的 RoPE 插值方法，并且可以作为 PI 的直接替代品，没有任何缺点，并且实现工作量最少。经过微调的模型在多个基准上保留了其原始能力，同时能够处理非常大的上下文大小。此外，YaRN 允许通过对较短数据集进行微调来进行有效外推，并且可以利用迁移学习来实现更快的收敛，这两者在计算受限的情况下都至关重要。最后，我们展示了 YaRN 外推法的有效性，它能够“短训练，长测试”。

## 可复现性

为了提高可重复性，我们提供了表 7 中用于训练 YaRN 模型的完整代码，以及生成图 7 和表 6、7、10、8 和 9 的评估代码作为补充材料。该代码还包含本文中引用的各种扩展方法的实现。为了训练 YaRN，我们使用了公开可用的 PG19 数据集（Rae 等人，2020），该数据集被标记为 64k 标记的连续块。

## 参考文献

- Azerbayev et al. (2022) Z. Azerbayev, E. Ayers, and B. Piotrowski Proof-pile . External Links: [Link](https://github.com/zhangir-azerbayev/proof-pile) Cited by: §4.2 .

- Black et al. (2022) S. Black, S. Biderman, E. Hallahan, Q. Anthony, L. Gao, L. Golding, H. He, C. Leahy, K. McDonell, J. Phang, M. Pieler, U. S. Prashanth, S. Purohit, L. Reynolds, J. Tow, B. Wang, and S. Weinbach GPT-NeoX-20B: an open-source autoregressive language model . Note: arXiv: 2204.06745 External Links: 2204.06745 Cited by: §1 .

- bloc97 (2023a) bloc97 Add NTK-Aware interpolation "by parts" correction . External Links: [Link](https://github.com/jquesnelle/scaled-rope/pull/1) Cited by: 2nd item , §3.2 .

- bloc97 (2023b) bloc97 NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation. . External Links: [Link](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) Cited by: §A.2 , §1 , §3.1 .

- Chen (2022) Carol. Chen Transformer Inference Arithmetic . External Links: [Link](https://kipp.ly/blog/transformer-inference-arithmetic/) Cited by: §3.4 .

- Chen et al. (2023) S. Chen, S. Wong, L. Chen, and Y. Tian Extending context window of large language models via positional interpolation . Note: arXiv: 2306.15595 External Links: 2306.15595 Cited by: §A.1 , §A.2 , §B.2 , §1 , §2.3 , §3.1 , §3.2 , §3 , §4.1 , Table 4 .

- Chowdhery et al. (2022) A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, P. Schuh, K. Shi, S. Tsvyashchenko, J. Maynez, A. Rao, P. Barnes, Y. Tay, N. Shazeer, V. Prabhakaran, E. Reif, N. Du, B. Hutchinson, R. Pope, J. Bradbury, J. Austin, M. Isard, G. Gur-Ari, P. Yin, T. Duke, A. Levskaya, S. Ghemawat, S. Dev, H. Michalewski, X. Garcia, V. Misra, K. Robinson, L. Fedus, D. Zhou, D. Ippolito, D. Luan, H. Lim, B. Zoph, A. Spiridonov, R. Sepassi, D. Dohan, S. Agrawal, M. Omernick, A. M. Dai, T. S. Pillai, M. Pellat, A. Lewkowycz, E. Moreira, R. Child, O. Polozov, K. Lee, Z. Zhou, X. Wang, B. Saeta, M. Diaz, O. Firat, M. Catasta, J. Wei, K. Meier-Hellstern, D. Eck, J. Dean, S. Petrov, and N. Fiedel PaLM: scaling language modeling with pathways . Note: arXiv: 2204.02311 External Links: 2204.02311 Cited by: §1 .

- Clark et al. (2018) P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord Think you have solved question answering? try ARC, the AI2 Reasoning Challenge . Note: arXiv: 1803.05457 External Links: 1803.05457 Cited by: §4.4 .

- Computer (2023) T. Computer RedPajama: an open source recipe to reproduce llama training dataset . External Links: [Link](https://github.com/togethercomputer/RedPajama-Data) Cited by: §A.3 , footnote ∥ .

- Dao (2023) T. Dao FlashAttention-2: faster attention with better parallelism and work partitioning . Note: arXiv: 2307.08691 External Links: 2307.08691 Cited by: §2.3 , §3.3 , §4.1 .

- emozilla (2023) emozilla Dynamically Scaled RoPE further increases performance of long context LLaMA with zero fine-tuning . External Links: [Link](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/) Cited by: 1st item , §3.4 .

- Gehring et al. (2017) J. Gehring, M. Auli, D. Grangier, D. Yarats, and Y. N. Dauphin Convolutional sequence to sequence learning . Note: arXiv: 1705.03122 External Links: 1705.03122 Cited by: §1 .

- Han et al. (2023) C. Han, Q. Wang, W. Xiong, Y. Chen, H. Ji, and S. Wang LM-Infinite: simple on-the-fly length generalization for large language models . Note: arXiv: 2308.16137 External Links: 2308.16137 Cited by: §2.3 .

- Hendrycks et al. (2021) D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt Measuring massive multitask language understanding . Proceedings of the International Conference on Learning Representations (ICLR) . Cited by: §4.4 .

- Huang et al. (2021) L. Huang, S. Cao, N. Parulian, H. Ji, and L. Wang Efficient attentions for long document summarization . In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , pp. 1419–1436 . Cited by: §4.2 .

- Hugging Face (2023) Hugging Face Open LLM Leaderboard . External Links: [Link](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) Cited by: §4.4 .

- [17] Introducing Qwen-7B: Open foundation and human-aligned models (of the state-of-the-arts) . External Links: [Link](https://github.com/QwenLM/Qwen-7B/blob/main/tech_memo.md) Cited by: §1 .

- kaiokendev (2023) kaiokendev Things I’m learning while training superhot. . External Links: [Link](https://kaiokendev.github.io/til#extending-context-to-8k) Cited by: §1 , §2.3 , §3.1 .

- Kazemnejad et al. (2023) A. Kazemnejad, I. Padhi, K. N. Ramamurthy, P. Das, and S. Reddy The impact of positional encoding on length generalization in transformers . Note: arXiv: 2305.19466 External Links: 2305.19466 Cited by: §1 .

- Lin et al. (2022) S. Lin, J. Hilton, and O. Evans TruthfulQA: measuring how models mimic human falsehoods . In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pp. 3214–3252 . Cited by: §4.4 .

- Loshchilov and Hutter (2019) I. Loshchilov and F. Hutter Decoupled weight decay regularization . In International Conference on Learning Representations , Cited by: §4.1 .

- Mohtashami and Jaggi (2023) A. Mohtashami and M. Jaggi Landmark attention: random-access infinite context length for transformers . Note: arXiv: 2305.16300 External Links: 2305.16300 Cited by: §4.3 .

- Paszke et al. (2019) A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Köpf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala PyTorch: an imperative style, high-performance deep learning library. . In NeurIPS , pp. 8024–8035 . Cited by: §4.1 .

- Press et al. (2022) O. Press, N. Smith, and M. Lewis Train Short, Test Long: attention with linear biases enables input length extrapolation . In International Conference on Learning Representations , Cited by: §1 , §4.2 .

- Quesnelle et al. (2023) J. Quesnelle, E. Shippole, and "Kaiokendev" LLongMA: scaling rotary embeddings through linear positional interpolation . HuggingFace . Note: [https://huggingface.co/conceptofmind/LLongMA-2-7b/](https://huggingface.co/conceptofmind/LLongMA-2-7b/) Cited by: §3.1 , footnote ∥ .

- Rae et al. (2020) J. W. Rae, A. Potapenko, S. M. Jayakumar, C. Hillier, and T. P. Lillicrap Compressive transformers for long-range sequence modelling . In International Conference on Learning Representations , Cited by: §4.1 , §4.4 , §6 .

- Roberts et al. (2019) A. Roberts, C. Raffel, K. Lee, M. Matena, N. Shazeer, P. J. Liu, S. Narang, W. Li, and Y. Zhou Exploring the limits of transfer learning with a unified text-to-text transformer . Technical report Google . Cited by: §1 .

- Rozière et al. (2023) B. Rozière, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E. Tan, Y. Adi, J. Liu, T. Remez, J. Rapin, A. Kozhevnikov, I. Evtimov, J. Bitton, M. Bhatt, C. C. Ferrer, A. Grattafiori, W. Xiong, A. Défossez, J. Copet, F. Azhar, H. Touvron, L. Martin, N. Usunier, T. Scialom, and G. Synnaeve Code Llama: open foundation models for code . Note: arXiv: 2308.12950 External Links: 2308.12950 Cited by: §B.3 , §1 , Table 4 , footnote * .

- Shaw et al. (2018) P. Shaw, J. Uszkoreit, and A. Vaswani Self-attention with relative position representations . In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers) , New Orleans, Louisiana , pp. 464–468 . Cited by: §1 .

- Su et al. (2022) J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu RoFormer: enhanced transformer with rotary position embedding . Note: arXiv: 2104.09864 External Links: 2104.09864 Cited by: §1 , §2.1 .

- Su (2023) J. Su Rectified rotary position embeddings . Note: [https://github.com/bojone/rerope](https://github.com/bojone/rerope) Cited by: §2.3 .

- Sun et al. (2022) Y. Sun, L. Dong, B. Patra, S. Ma, S. Huang, A. Benhaim, V. Chaudhary, X. Song, and F. Wei A length-extrapolatable transformer . Note: arXiv: 2212.10554 External Links: 2212.10554 Cited by: §1 .

- Tancik et al. (2020) M. Tancik, P. P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. T. Barron, and R. Ng Fourier features let networks learn high frequency functions in low dimensional domains . In Proceedings of the 34th International Conference on Neural Information Processing Systems , NIPS’20 , Red Hook, NY, USA . External Links: ISBN 9781713829546 Cited by: §3.1 .

- Together.ai (2023) Together.ai LLaMA-2-7B-32K . External Links: [Link](https://huggingface.co/togethercomputer/LLaMA-2-7B-32K) Cited by: §B.3 , §3.1 , Table 4 .

- Touvron et al. (2023a) H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample LLaMA: open and efficient foundation language models . Note: arXiv: 2302.13971 External Links: 2302.13971 Cited by: §1 .

- Touvron et al. (2023b) H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom Llama 2: open foundation and fine-tuned chat models . External Links: 2307.09288 Cited by: §4.1 .

- Vaswani et al. (2017) A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, \. Kaiser, and I. Polosukhin Attention is all you need . In Advances in Neural Information Processing Systems , Vol. 30 , pp. . Cited by: §1 .

- Xiong et al. (2023) W. Xiong, J. Liu, I. Molybog, H. Zhang, P. Bhargava, R. Hou, L. Martin, R. Rungta, K. A. Sankararaman, B. Oguz, M. Khabsa, H. Fang, Y. Mehdad, S. Narang, K. Malik, A. Fan, S. Bhosale, S. Edunov, M. Lewis, S. Wang, and H. Ma Effective long-context scaling of foundation models . External Links: 2309.16039 Cited by: Table 4 .

- Zellers et al. (2019) R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi HellaSwag: can a machine really finish your sentence? . In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics , Cited by: §4.4 .

- Zhao et al. (2023) Y. Zhao, A. Gu, R. Varma, L. Luo, C. Huang, M. Xu, L. Wright, H. Shojanazeri, M. Ott, S. Shleifer, A. Desmaison, C. Balioglu, B. Nguyen, G. Chauhan, Y. Hao, and S. Li PyTorch FSDP: experiences on scaling fully sharded data parallel . Note: arXiv: 2304.11277 External Links: 2304.11277 Cited by: §4.1 .

## 有关插值方法的其他详细信息

### 位置插值

正如2.2节中提到的，PI是扩展RoPE上下文长度的早期作品之一。我们在这里包含一些额外的细节：

虽然直接外推法在序列 $w_{1},\cdots,w_{L}$ 和 $L$ 大于预训练限制的情况下表现不佳，但他们发现，在少量微调的帮助下，在预训练限制内插值位置索引效果很好。具体来说，给定一个带有 RoPE 的预训练语言模型，他们通过以下方式修改 RoPE：

$$
\displaystyle f^{\prime}_{\bm{W}}\left({\bm{x}}_{m},m,\bm{\theta}\right)=f_{\bm{W}}\left({\bm{x}}_{m},\dfrac{mL}{L^{\prime}},\bm{\theta}\right),\tag{16}
$$

其中 $L^{\prime}>L$ 是超出预训练限制的新上下文窗口。利用原始的预训练模型加上修改后的 RoPE 公式，他们在减少几个数量级的标记（Chen 等人[2023]中为几十亿）上进一步微调语言模型，并成功实现了上下文窗口扩展。

### “NTK-aware”插值的详细信息

在3.1节中，我们在“NTK-aware”插值方法的定义中引入了从$b$到$b^{\prime}$的基数变换。

准确地说，遵循第 2.1 节等式中规定的符号。如图7所示，我们定义“NTK-aware”插值方案如下：

#### 定义3

“NTK-aware”插值是使用等式 1 的 RoPE 的修改。 7.具有以下函数，给出$s$作为比例因子。

$$
\displaystyle g(m) \displaystyle=m\tag{17}
$$

$$
\displaystyle h(\theta_{d}) \displaystyle={b^{\prime}}^{-2d/|D|},\tag{18}
$$

哪里

$$
\displaystyle{b^{\prime}} \displaystyle=b\cdot s^{\frac{|D|}{|D|-2}}\tag{19}
$$

$$
\displaystyle{s} \displaystyle=\frac{L^{\prime}}{L}.\tag{20}
$$

鉴于 [bloc97, 2023b] 的结果，与 PI 相比，该方法在扩展非微调模型的上下文大小方面表现得更好 [Chen et al., 2023]。然而，这种方法的一个主要缺点是，鉴于它不仅仅是一种插值方案，某些维度会稍微外推到“越界”值，因此使用“NTK-aware”插值进行微调 [bloc97, 2023b] 会产生比 PI 差的结果 [Chen et al., 2023]。此外，由于“出界”值，理论比例因子$s$不能准确地描述真实的上下文扩展比例。实际上，缩放值 $s$ 必须设置为高于给定上下文长度扩展的预期缩放。

基数变化的数学推导如下：

回想一下，我们的目标是使用基数变换将插值压力分散到隐藏维度上，而不是按固定因子 $s$ 缩放频率。我们要保证的特性是：最低频率需要与线性位置缩放一样多，而最高频率需要保持恒定。

我们引入了一个新的基数 $b^{\prime}$ ，使得最后一个维度与具有比例因子 $s$ 的线性插值波长相匹配。由于原始 RoPE 方法跳过奇数维度以便将 $\cos(\frac{2\pi x}{\lambda})$ 和 $\sin(\frac{2\pi x}{\lambda})$ 分量连接到单个嵌入中，因此最后一个维度 $d\in D$ 是 $|D|-2$ 。

可以选择新底座 $b^{\prime}$，以便

$$
{b^{\prime}}^{\frac{|D|-2}{|D|}}=s\cdot b^{\frac{|D|-2}{|D|}}.\tag{21}
$$

求解 $b^{\prime}$ 产量

$$
{b^{\prime}}=b\cdot s^{\frac{|D|}{|D|-2}}.\tag{22}
$$

### YaRN pre-softmax 缩放对困惑度的影响

在3.3节中，我们提到了注意力权重的softmax计算中因子$t$的影响。在这里，我们修复了 RedPajama [Computer, 2023] [^footnote5] 中的 $896$ $16$ k-token 文档，并使用不同的缩放比例 $1/\sqrt{t}$ 计算它们的困惑度分数。结果如图4所示。为了进行比较，请记住我们在这种情况下推荐的因子（$s=8$）由以下给出。

$$
\sqrt{\frac{1}{t}}=0.1\ln({s})+1\approx 1.208.\tag{23}
$$

![Figure A1.F4](figs/A1-F4-01.png)

图 4：修复 $s=8$ ，比较 $896$ $16$ k-token 文档在不同缩放 $1/\sqrt{t}$ 上的 LLaMA 7b 困惑。阴影区域代表 $1$ 标准差 ( $68\%$ )。

为了显示因子 $1/\sqrt{t}$ 对不同令牌位置的影响，我们将每个 $16$ k 令牌文档切成 $2048$ 令牌块，并进一步以百分比绘制与 $t=1$ 相比的平均困惑度变化

$$
\dfrac{\text{ppl}(t)-\text{ppl}(t=1)}{\text{ppl}(t=1)}\tag{24}
$$

每个块的。该图如图 5 所示。

![Figure A1.F5](figs/A1-F5-01.png)

图 5：修复 $s=8$ ，比较不同缩放 $1/\sqrt{t}$ 上 $896$ $16$ k token文档上不同token位置段的困惑度变化百分比 $\dfrac{\text{ppl}(t)-\text{ppl}(t=1)}{\text{ppl}(t=1)}$ 的平均值。

为了进一步证明 $t$ 在不同令牌位置上的所有样本中的最佳值，我们在图 6 中绘制了 $16$ k 令牌范围上的每个 $8$ 位置段在给定 $1/\sqrt{t}$ 处具有最小困惑度的样本计数。

![Figure A1.F6](figs/A1-F6-01.png)

图 6：对于 $16$ k token范围内给定的token位置段，在给定 $1/\sqrt{t}$ 上具有最小困惑度的样本计数（在 $896$ 样本中）。

我们观察到：

- 对于合适的 $t$ ，样本可以在扩展上下文窗口中获得更好的困惑度分数；

- $t$ 的最佳值在不同样本和不同位置上基本一致。

我们注意到，这一发现对于 $s$ 的不同值是一致的，并且 $t$ 的最佳值紧密遵循我们推荐的公式（方程 15）。

## 附加表格和图表

### 消融研究

| 扩展 | 是否 | 训练 | 扩展 | 评测上下文窗口长度 | 评测上下文窗口长度 | 评测上下文窗口长度 | 评测上下文窗口长度 | 评测上下文窗口长度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 方法 | 微调 | 步数 | 倍率 $s$ | 2048 | 4096 | 8192 | 16384 | 32768 |
| 无 | ✗ | - | - | 4.05 | - | - | - | - |
| PI | ✗ | - | 2k $\times$ 2 | 4.36 | 3.90 | - | - | - |
| NTK-aware | ✗ | - | 2k $\times$ 2 | 4.08 | 5.97 | - | - | - |
| NTK-by-parts | ✗ | - | 2k $\times$ 2 | 4.12 | 3.71 | - | - | - |
| YaRN | ✗ | - | 2k $\times$ 2 | 4.07 | 3.67 | - | - | - |
| PI | ✗ | - | 2k $\times$ 4 | 7.09 | 6.39 | 6.18 | - | - |
| NTK-aware | ✗ | - | 2k $\times$ 4 | 4.27 | 3.84 | $>10^{1}$ | - | - |
| NTK-by-parts | ✗ | - | 2k $\times$ 4 | 4.39 | 4.03 | 4.11 | - | - |
| YaRN | ✗ | - | 2k $\times$ 4 | 4.19 | 3.77 | 3.65 | - | - |
| PI | ✗ | - | 2k $\times$ 8 | $>10^{1}$ | $>10^{1}$ | $>10^{1}$ | $>10^{1}$ | - |
| NTK-aware | ✗ | - | 2k $\times$ 8 | 4.64 | 4.27 | 4.24 | $>10^{1}$ | - |
| NTK-by-parts | ✗ | - | 2k $\times$ 8 | 4.98 | 4.91 | 5.33 | 5.79 | - |
| YaRN | ✗ | - | 2k $\times$ 8 | 4.37 | 3.95 | 3.81 | 3.33 | - |
| PI | ✗ | - | 2k $\times$ 16 | $>10^{2}$ | $>10^{2}$ | $>10^{2}$ | $>10^{2}$ | $>10^{2}$ |
| NTK-aware | ✗ | - | 2k $\times$ 16 | 5.23 | 5.02 | 5.22 | 6.85 | $>10^{1}$ |
| NTK-by-parts | ✗ | - | 2k $\times$ 16 | 6.04 | 7.54 | $>10^{1}$ | $>10^{1}$ | $>10^{1}$ |
| YaRN | ✗ | - | 2k $\times$ 16 | 4.61 | 4.24 | 4.18 | 3.66 | 3.45 |
| PI | ✗ | - | 动态 | 4.05 | 3.90 | 6.18 | $>10^{1}$ | $>10^{2}$ |
| NTK-aware | ✗ | - | 动态 | 4.05 | 5.97 | $>10^{1}$ | $>10^{1}$ | $>10^{1}$ |
| NTK-by-parts | ✗ | - | 动态 | 4.05 | 3.71 | 4.11 | 5.79 | $>10^{1}$ |
| YaRN | ✗ | - | 动态 | 4.05 | 3.67 | 3.65 | 3.33 | 3.45 |
| PI | ✓ | 400 | 2k $\times$ 16 | 5.70 | 4.95 | 4.64 | 3.97 | 3.57 |
| NTK-aware | ✓ | 400 | 2k $\times$ 16 | 4.39 | 3.92 | 3.73 | 3.21 | 8.49 |
| NTK-by-parts | ✓ | 400 | 2k $\times$ 16 | 4.14 | 3.75 | 3.62 | 3.12 | 2.81 |
| YaRN | ✓ | 400 | 2k $\times$ 16 | 4.19 | 3.77 | 3.30 | 3.09 | 2.77 |

表 5：通过不同方法扩展的 LLaMA 7B 模型上十个 128k 证明堆文档的滑动窗口困惑度 ($S=256$)。

### YaRN 的训练效率

表 6 显示了通过 PI (LLongMA-2 7B [^footnote6] ) 和 YaRN 从 $4096$ 上下文长度扩展到 $8192$ 的 Llama 2 7B 模型的并排比较。请注意，PI 模型是使用 Chen 等人的方法进行训练的。 [2023]，而 YaRN 使用相同的方法，但训练步骤和数据减少了 2.5 倍，如第 4 节所述。即使与 PI 的 1000 个步骤相比，YaRN 仅微调了 400 个步骤，我们也获得了与 PI 相似的结果。

| 模型 | 扩展 | 是否 | 训练 | 扩展 | 评测上下文窗口长度 | 评测上下文窗口长度 | 评测上下文窗口长度 | 评测上下文窗口长度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 规模 | 方法 | 微调 | 步数 | 倍率 $s$ | 2048 | 4096 | 6144 | 8192 |
| 7B | 无 | ✗ | - | - | 4.00 | 3.58 | - | - |
| 7B | PI | ✗ | - | 4k $\times$ 2 | 4.30 | 3.84 | 3.83 | 3.65 |
| 7B | YaRN | ✗ | - | 4k $\times$ 2 | 4.03 | 3.61 | 3.60 | 3.49 |
| 7B | PI | ✓ | 1000 | 4k $\times$ 2 | 3.92 | 3.51 | 3.51 | 3.34 |
| 7B | YaRN | ✓ | 400 | 4k $\times$ 2 | 3.91 | 3.50 | 3.51 | 3.35 |

表 6：通过具有不同训练步骤的 PI 和 YaRN 扩展的 Llama-2 7b 模型上 10 个 128k 证明堆文档的滑动窗口困惑度 ( $S=256$ )。 YaRN 使用更少的训练步骤获得了与 PI 相当的结果。

### 滑动窗口下不同方法的困惑度比较

我们进一步评估了使用 YaRN 在比例因子 $s=16,32$ 上微调的 Llama 2 模型，并将它们与从 Llama-2 微调的一些长上下文开源模型进行了比较，例如 Together.ai [Together.ai, 2023] 和“NTK-aware”Code Llama [Rozière et al., 2023]。结果总结在表 7 中（图 7 中有更详细的图）。

| 模型 | 模型 | 上下文 | 扩展 | 评测上下文窗口长度 | 评测上下文窗口长度 | 评测上下文窗口长度 | 评测上下文窗口长度 | 评测上下文窗口长度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 规模 | 名称 | 窗口 | 方法 | 8192 | 32768 | 65536 | 98304 | 131072 |
| 7B | Together | 32k | PI | 3.50 | 2.64 | $>10^{2}$ | $>10^{3}$ | $>10^{4}$ |
| 7B | Code Llama | 100k | NTK | 3.71 | 2.74 | 2.55 | 2.54 | 2.71 |
| 7B | YaRN ( $s=16$ ) | 64k | YaRN | 3.51 | 2.65 | 2.42 | $>10^{1}$ | $>10^{1}$ |
| 7B | YaRN ( $s=32$ ) | 128k | YaRN | 3.56 | 2.70 | 2.45 | 2.36 | 2.37 |
| 13B | Code Llama | 100k | NTK | 3.54 | 2.63 | 2.41 | 2.37 | 2.54 |
| 13B | YaRN ( $s=16$ ) | 64k | YaRN | 3.25 | 2.50 | 2.29 | $>10^{1}$ | $>10^{1}$ |
| 13B | YaRN ( $s=32$ ) | 128k | YaRN | 3.29 | 2.53 | 2.31 | 2.23 | 2.24 |

表 7：十个 128k 证明堆文档的滑动窗口困惑度（$S=256$）被截断为评估上下文窗口大小

我们观察到该模型在整个目标上下文大小上表现出强大的性能，YaRN 插值是第一个成功地将 Llama 2 的有效上下文大小扩展到 128k 的方法。

此外，在附录 B.4 中，我们在表 8 中显示了在没有动态缩放的 32k 最大上下文窗口设置下评估的 50 个未截断的 GovReport 文档（每个样本至少有 16k 个标记）的平均困惑度结果。与 Proof-pile 结果类似，GovReport 结果表明，使用 YaRN 进行微调在长序列上取得了良好的性能。

表 7 总结了结果，图 7 中提供了可视化且更详细的视图。

![Figure A2.F7](figs/A2-F7-01.png)

图 7：1.28M token证明堆文档的滑动窗口困惑度（$S=256$）被截断为微调模型的上下文窗口大小

### GovReport 评估

在4.2节中，我们提到了对GovReport文件的评估。评估结果详述于下表8中。

| 模型 | 模型 | 上下文 | 扩展 | 困惑度 |
| --- | --- | --- | --- | --- |
| 规模 | 名称 | 窗口 | 方法 | 困惑度 |
| 7B | Together | 32k | PI | 3.67 |
| 7B | Code Llama | 100k | NTK | 4.44 |
| 7B | YaRN ( $s=16$ ) | 64k | YaRN | 3.59 |
| 7B | YaRN ( $s=32$ ) | 128k | YaRN | 3.64 |
| 13B | Code Llama | 100k | NTK | 4.22 |
| 13B | YaRN ( $s=16$ ) | 64k | YaRN | 3.35 |
| 13B | YaRN ( $s=32$ ) | 128k | YaRN | 3.39 |

表 8：固定上下文窗口大小为 32k 的 50 个长 GovReport 文档的滑动窗口困惑度 ( $S=256$ )

### 口令检索

为了评估 64k 和 128k 模型，我们执行了 10 次密钥检索任务迭代，并将密钥放置在均匀分布在评估上下文窗口中的随机位置，上下文窗口大小范围从 8k 到 128k。使用 YaRN 在 128k 上下文大小上进行微调的 7b 和 13b 模型都在整个上下文窗口大小内以非常高的准确度（ $>99\%$ ）通过了密钥检索任务。

| 模型 | 模型 | 缩放 | 上下文 | 训练 | 扩展 | 口令检索 | 口令检索 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 规模 | 名称 | 倍率 $(s)$ | 窗口 | 数据上下文长度 | 方法 | 上下文 | 准确率 |
| 7B | Together | 4 | 32k | 32k | PI | 32k | 100% |
| 7B | Code Llama | 88.6 | 100k | 16k | NTK | 112k | 94.3% |
| 7B | YaRN | 16 | 64k | 64k | YaRN | 64k | 96.3% |
| 7B | YaRN | 32 | 128k | 64k | YaRN | 128k | 99.4% |
| 13B | Code Llama | 88.6 | 100k | 16k | NTK | 128k | 99.4% |
| 13B | YaRN | 16 | 64k | 64k | YaRN | 64k | 97.5% |
| 13B | YaRN | 32 | 128k | 64k | YaRN | 128k | 99.4% |

表 9：各种模型的密钥检索性能。密钥上下文表示最大测试上下文窗口大小，其中密钥检索的准确性为 $>=80\%$ ，并且密钥准确性是在小于或等于密钥上下文大小的所有测试上下文大小上密钥检索的平均准确性。

在这里我们可以观察到，仅最低困惑点并不能全面描述法学硕士可以关注的“有效背景大小”。虽然 Code Llama 13b 模型在上下文长度超过 100k 时表现出越来越大的复杂性，但它仍然能够在 128k 上下文长度下准确检索密钥。这表明，虽然 Code Llama 的输出质量在超过 100k 上下文大小时可能开始下降，但它仍然能够保持强大的检索能力。

此外，由于使用 $s=32$ 的 YaRN 比使用 $s=16$ 的 YaRN 多了 200 个步骤，同时具有更高的密钥准确性和类似的困惑度，因此我们假设困惑度可能不是 LLM 是否能够处理所有令牌的一个很好的指标，并且不能详尽地确定长上下文性能。这也表明具有 $s=16$ 的 YaRN 模型对于密钥检索任务的训练可能相对不足。

### 标准化基准

为了测试上下文扩展下模型性能的下降，我们使用该套件评估了我们的模型，并将其与 Llama 2 基线以及公开可用的 PI 和“NTK-aware”模型的既定分数进行比较。

| 模型 | 模型 | 上下文 | 扩展 | ARC-c | Hellaswag | MMLU | TruthfulQA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 规模 | 名称 | 窗口 | 方法 | ARC-c | Hellaswag | MMLU | TruthfulQA |
| 7B | Llama 2 | 4k | 无 | 53.1 | 77.8 | 43.8 | 39.0 |
| 7B | Together | 32k | PI | 47.6 | 76.1 | 43.3 | 39.2 |
| 7B | Code Llama | 100k | NTK-a | 39.9 | 60.8 | 31.1 | 37.8 |
| 7B | YaRN ( $s=16$ ) | 64k | YaRN | 52.3 | 78.8 | 42.5 | 38.2 |
| 7B | YaRN ( $s=32$ ) | 128k | YaRN | 52.1 | 78.4 | 41.7 | 37.3 |
| 13B | Llama 2 | 4k | 无 | 59.4 | 82.1 | 55.8 | 37.4 |
| 13B | Code Llama | 100k | NTK-a | 40.9 | 63.4 | 32.8 | 43.8 |
| 13B | YaRN ( $s=16$ ) | 64k | YaRN | 58.1 | 82.3 | 52.8 | 37.8 |
| 13B | YaRN ( $s=32$ ) | 128k | YaRN | 58.0 | 82.2 | 51.9 | 37.3 |

表 10：与原始 Llama 2 基线相比，Hugging Face Open LLM 基准套件上的上下文窗口扩展方法的性能

### 无需任何微调的模型动态缩放

我们首先回顾 3.4 节，动态缩放技术是一种推理时间技术，可动态更新 PI、“NTK-by-parts”和 YaRN 等插值方法中的因子 $s$。我们选择原始的 Llama 2，在 GovReport 中固定样本，并使用 RoPE、Dynamic-PI 和 Dynamic-YaRN 在 $256$ token的滑动窗口上计算其困惑度。

![Figure A2.F8](figs/A2-F8-01.png)

图 8：在长 GovReport 样本上使用 Llama 2 进行 RoPE、Dynamic-PI 和 Dynamic-YaRN 之间的比较。该模型尚未针对长上下文进行微调。

由于 Llama 2 的原始最大上下文长度为 $4096$ ，我们观察到 Dynamic Sc​​aling 有效地扩展了推理长度，并且 Dynamic-YaRN 实现了比 Dynamic-PI 更好的性能。生成的图表如图 8 所示。

我们看到

- 动态缩放有效防止困惑度分数超出预训练上下文窗口的范围；

- Dynamic-YaRN 在预训练 Llama-2 上的远程困惑度方面优于 Dynamic-PI，无需任何微调。

## 脚注

[^footnotex1]: 通讯地址：{bloc,emozilla}@nousresearch.com

[^footnote1]: 我们注意到，在本文发布前不久，Code Llama（Rozière et al., 2023）发布，并通过手动将基础 $b$ 缩放到 1M 来使用“NTK-aware”缩放，他们将这种方法称为 RoPE“调整基础频率”（ABF）。

[^footnote2]: 由于维度在预训练期间至少不会完全旋转一次，因此如果我们选择第一个标记作为锚点，则预训练期间的每个其他标记都具有唯一的距离，神经网络可以使用该距离来确定其绝对位置。

[^footnote3]: $h$ 上的线性斜坡插值可以有替代方案，例如从波长上的线性插值转换而来的 $\theta_{d}/s$ 和 $\theta_{d}$ 上的调和平均值。这里选择 $h$ 是为了实现简单，但两者都可以工作。

[^footnote4]: LLaMA 模型的预训练上下文大小为 2k 个标记，而 Llama 2 模型有 4k 个。

[^footnote5]: 我们选择 RedPajama 是因为据我们所知，它是最接近 LLaMA 训练数据集的开源数据集。

[^footnote6]: LLongMA-2 7B [Quesnelle et al., 2023] 是根据 Llama 2 7B 进行微调的，使用 RedPajama 数据集 [Computer, 2023] 通过 PI 以 8k 上下文长度进行训练。
