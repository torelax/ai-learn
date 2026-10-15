# PLAID：高效的后期交互检索引擎（中译）

> 原作：Keshav Santhanam、Omar Khattab、Christopher Potts、Matei Zaharia；前两位作者同等贡献。  
> paper：https://arxiv.org/abs/2205.09707v1 · 2022-05-19  
> 许可：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本文为中文翻译和排版改编；版权与署名归原作者。  
> 范围：摘要、正文全部原文段落、公式、图表及脚注；参考文献保留英文。段落 HTML ID 用于与英文稿对照。图表后的中文解释含译者说明，并非全部逐字图注。译注与作者原话分开；结构检查不能代替语义审校。

## 摘要

<!-- abstract1.1 -->
预训练语言模型正成为多种信息检索范式中越来越重要的组成部分。ColBERT 引入、随后由 ColBERTv2 改进的后期交互，是一种流行范式，在多个 benchmark 上达到最先进水平。为显著降低后期交互的搜索延迟，我们提出性能优化的后期交互驱动器 PLAID（Performance-optimized Late Interaction Driver）。PLAID 引入新的质心交互机制，将每个段落视为轻量的“质心袋”，在不影响质量的情况下迅速排除低分段落。结合用于稀疏化质心袋的质心剪枝机制，以及高度优化的执行引擎，PLAID 相对原始 ColBERTv2 将 GPU 搜索延迟最多降低 7 倍、CPU 搜索延迟最多降低 45 倍，同时继续保持最先进的检索质量。这使 ColBERTv2 的 PLAID 引擎即使在我们评测的最大规模——1.4 亿段落——上，也能实现 GPU 数十毫秒、CPU 数十至数百毫秒的延迟。

## 引言

<!-- S1.p1.1 -->
神经信息检索的近期进展，在检索 benchmark 和依赖检索的 NLP 任务上带来了显著提升。ColBERT 提出的后期交互范式，在段落排序、开放域问答、对话任务及其他任务上，都能达到最先进的质量。ColBERT 及其变体把查询和文档编码为 token 级向量，通过可扩展、细粒度的 token 交互完成打分（图 1），缓解单向量表示的点积瓶颈。近期的 ColBERTv2 表明，后期交互模型在训练域内和域外，通常都明显优于近期的单向量和稀疏表示模型；多项研究也观察到了这一现象。（原段落引文：[22], [42], [14], [48], [21], [24], [38], [35], [20], [54], [26], [53], [44], [29], [51], [43]。）

<!-- S1.p2.1 -->
尽管检索质量强，后期交互仍需要专用基础设施才能实现低延迟，因为它把每个查询和文档都编码为完整矩阵。大多数 IR 模型用单一向量表示文档，或者是 BM25、SPLADE 这样的稀疏向量，或者是 DPR、ANCE 这样的稠密向量。因此，WAND 等成熟稀疏检索策略或 HNSW 等稠密 kNN 方法，无法直接或最优地用于后期交互。近期工作虽然优化了 ColBERT 流水线的个别组件，但据我们所知，还没有研究端到端优化的执行引擎。（原段落引文：[22], [25], [41], [11], [18], [49], [5], [31], [28], [45], [42]。）

<!-- S1.p3.1 -->
我们考虑检索的全部步骤，研究如何在大规模场景下降低后期交互搜索延迟。我们以最先进的 ColBERTv2 为基础。ColBERTv2 不仅用去噪监督改进质量，也积极压缩后期交互的存储占用，借助残差表示将索引大小最多降低一个数量级（原文 §3.1）。在这种表示中，每个段落向量被编码为最近质心的 ID，以及一个量化残差向量。这些质心来自 k-means 聚类，数量为数万至数十万，用于近似相应 token 的语义。

<!-- S1.p4.1 -->
我们提出高效检索引擎 PLAID。相对原始 ColBERTv2，它在保持高质量的同时，将 GPU 后期交互搜索延迟降低 2.5–7 倍、CPU 延迟降低 9–45 倍。这使名为 PLAID ColBERTv2 的实现，即使面对 1.4 亿段落这样的大规模语料，也能把纯 CPU 延迟降到数十或数百毫秒，把 GPU 延迟降到数十毫秒。关键在于，PLAID ColBERTv2 仍能保持最先进的检索质量。

<!-- S1.p5.1 -->
为了大幅加速搜索，PLAID 利用 ColBERTv2 表示中的质心部分，即每个 token 的紧凑整数 ID。它不再对最近邻搜索找到的所有段落做完整打分，而是先用质心识别高分段落，无须加载较大的残差即可淘汰较弱候选。我们用多阶段流水线完成这一过程，并引入质心交互，把段落视为轻量的“质心 ID 袋”。我们发现，这种不使用向量残差、仅使用质心的多向量搜索仍有高召回率（原文 §3.3），因此可以只对极少数候选做完整打分。质心来自固定集合，相当于一个离散词表，所以查询向量与全部质心之间的距离只需在搜索时计算一次，并能在全部段落的质心袋表示间复用。质心剪枝进一步利用这些分数，在检索早期跳过远离所有查询向量的质心 ID，使质心袋表示稀疏化。

<!-- S1.p6.1 -->
PLAID 引擎实现了质心交互、质心剪枝，以及用于 ColBERTv2 残差表示的数据移动、解压和打分的优化且模块化的内核（原文 §4.5）。我们在训练域内和域外，包括 MS MARCO v1、v2、Wikipedia 和 LoTTE 上广泛评测质量与效率，覆盖 200 万至 1.4 亿段落、10/100/1000 的检索深度，以及单线程、多线程 CPU 和 GPU 设置（原文 §5.2）。我们还通过详细消融研究质心交互、质心剪枝与加速内核分别带来的收益（原文 §5.3）。（原段落引文：[36], [6], [42]。）

<!-- S1.p7.1 -->
我们的贡献如下。

<!-- S1.I1.i1.p1.1 -->
分析仅使用质心的 ColBERTv2 检索，证明经过剪枝的质心袋表示可以支持高召回候选生成（原文 §3）。

<!-- S1.I1.i2.p1.1 -->
提出 PLAID 检索引擎，引入质心交互和质心剪枝及其优化实现，显著降低后期交互搜索延迟（原文 §4）。

<!-- S1.I1.i3.p1.1 -->
广泛评测 PLAID，并在最大 1.4 亿段落上开展大规模评测；据我们所知，这是当时后期交互检索器的最大规模评测（原文 §5）。

![图 1：后期交互架构](figs/fig01.png)

**图 1。** 给定查询和段落时的后期交互架构；经许可使用 Khattab 等（2021b）的示意图。每个查询 token 与段落 token 做 MaxSim，再求和；蓝色段落表示可以离线编码。

## 相关工作

### 神经信息检索

<!-- S2.SS1.p1.1 -->
IR 社区提出了许多基于预训练 Transformer 的神经检索模型。早期模型主要是联合关注查询和段落的交叉编码器；后续许多模型为了提高效率，分别生成查询和段落的独立表示。有些模型生成稀疏词权重，有些把每个段落或查询编码为单向量，还有些采用本文研究的多向量表示。这些选择在效率和质量之间形成不同权衡：稀疏词权重与单向量模型在部分设置下非常轻量，但多向量后期交互通常能取得明显更强的质量与鲁棒性。与查询—文档交互建模方式的选择相互独立，研究者也通过更难的负例、蒸馏、去噪等方式改进神经模型的监督。本文扩展 ColBERTv2，它结合后期交互、难负例和去噪监督，在独立检索器中取得最先进质量。（原段落引文：[37], [27], [7], [32], [49], [18], [39], [22], [42], [21], [15], [12], [52], [40], [13]。）

### 稀疏与稠密检索中的剪枝

<!-- S2.SS2.p1.1 -->
对稀疏检索，传统 IR 已有大量关于 top-k 搜索中快速跳过文档的研究。这些策略通常保存词项分数上界等元信息，跳过低分候选，多数采用按文档逐个处理（DAAT）的打分方式。Tonellotto 等（2018）详细讨论了近期方法。与我们场景的关键区别是，这些策略都假定已有预计算分数，尤其是每个词项—文档对上有用的上界；后期交互中的词项—文档分数，即 MaxSim，却必须在查询时完成矩阵—向量乘法后才能得到。关于质心加速后期交互的观察，让问题更接近传统 IR，但查询—质心分数只能在查询时获知，仍是一项挑战。（原段落引文：[47], [5], [9], [8], [33], [19], [46]。）

<!-- S2.SS2.p2.1 -->
对采用单向量表示的稠密检索，近似 k 近邻搜索（ANN）是经过充分研究的问题。我们的研究将其从单向量扩展到两个矩阵之间的后期交互。（原段落引文：[16], [17], [1], [31]。）

![图 2：延迟分解](figs/fig02.png)

**图 2。** 在 TITAN V GPU 上运行 MS MARCO v1 dev 查询时，原始 ColBERTv2 与 PLAID ColBERTv2 的延迟分解。(a) 原始系统，nprobe=4、ncandidates=$2^{16}$；(b) PLAID，k=1000。原始系统主要受索引读取与解压制约，PLAID 正是针对这一问题。图中约 287/58 ms 的值属于这张分解图，不能直接替代结果表的计时值。

## ColBERTv2 检索分析

<!-- S3.p1.1 -->
我们先初步研究 ColBERTv2 的延迟与打分模式（原文 §3.2、§3.3），以说明 PLAID 的动机。为使本节自足，原文 §3.1 先回顾 ColBERTv2 的建模、存储与监督。

### 建模、存储与检索

<!-- S3.SS1.p1.1 -->
PLAID 优化采用 ColBERT 后期交互架构的模型，包括 ColBERTv2、Baleen、Hindsight、DrDecr 等。如图 1 所示，Transformer 分别把查询和段落编码为 token 级向量。为保证可扩展性，段落表示离线预计算。搜索时，查询 $q$ 与段落 $d$ 的相似度由多个 MaxSim 操作求和得到：对查询矩阵中的每个向量，取其与全部段落向量之间最大的余弦相似度。（原段落引文：[20], [38], [24]。）

<!-- equation 1 -->
$$
S_{q,d}=\sum_{i=1}^{|Q|}\max_{j=1}^{|D|}Q_{i}\cdot D_{j}^{T}
$$

式 (1)，保留原式。

<!-- S3.SS1.p1.2 -->
这里 $Q$ 与 $D$ 分别是查询和段落的矩阵表示。该打分函数把每个查询 token 对齐到“最相似”的段落 token，以这些词项级分数之和估计相关性。更完整的后期交互讨论见 Khattab 与 Zaharia（2020）。（原段落引文：[22]。）

<!-- S3.SS1.p2.1 -->
存储段落表示时，我们采用 ColBERTv2 的残差压缩。相比用 16-bit 浮点向量直接保存后期交互 embedding，它能让索引最多缩小一个数量级。具体做法是对全部 token 级向量高效聚类，把每个向量编码为最近簇质心的 ID 和一个量化残差向量；每个维度用 1 或 2 bit 表示质心与原始未压缩向量之间的差值。解压时需找到使用 4 bytes 编码的质心 ID，以及残差。默认 embedding 为 128 维时，1-bit、2-bit 残差分别占 16、32 bytes。

<!-- S3.SS1.p3.1 -->
我们沿用 ColBERTv2 的压缩，但改进检索策略。原始策略称为 vanilla ColBERTv2 retrieval。压缩与原始检索的细节见 Santhanam 等（2021）。（原段落引文：[42]。）

### 延迟分解

<!-- S3.SS2.p1.1 -->
图 2 分解了 MS MARCO Passage Ranking v1 上 GPU 查询延迟，并比较原始 ColBERTv2（图 2a）与新的 PLAID ColBERTv2（图 2b）。延迟由查询编码、候选生成、索引读取（收集候选段落的压缩向量表示）、残差解压，以及最终 MaxSim 打分构成。

<!-- S3.SS2.p2.1 -->
对原始 ColBERTv2，索引读取与残差解压是主要瓶颈。索引中的向量收集需要大量内存带宽：原文在此写每个向量包含一个“4-bit 质心 ID”和 32-byte 残差，每个段落有数十个向量，候选段落可多达 $2^{16}$。此外，索引读取要实时构造带 padding 的张量，处理段落长度不一的问题。残差解压涉及位展开和大量求和等非平凡操作；当 MS MARCO v1 这样的场景生成约 1–4 万个初始候选段落时，成本很高。虽然可以缩小候选集，但这会降低召回率（原文 §5）。

> 译注：原文此处的 **4-bit** 与前节 **4 bytes**、整数质心 ID 的存储及代码均不一致。保留原文用词以便回查；深读计算按 4 bytes 理解。

### 单靠质心即可识别强候选

<!-- S3.SS3.p1.1 -->
图 2b 的分解表明，对大量候选进行完整打分，尤其是收集并解压残差，会产生可观开销。ColBERTv2 用质心减少存储占用；本文进一步表明，质心还可作为段落 embedding 的代理，在保持质量的同时加速搜索。这样无需读取或解压低分段落的残差即可跳过它们；虽然候选生成增加少量开销，但后续阶段节省很多（图 2b）。（原段落引文：[42]。）

<!-- S3.SS3.p2.1 -->
我们假设，只用质心的检索也能找到原始 ColBERTv2 检索到的高分段落。验证方法是把原始 ColBERTv2 的 top-k 段落，与一个仅使用质心、完全不使用残差的修改版实现比较。图 3 在 $k\in\{10,100,1000\}$ 时，画出原始系统的 top-k 段落在质心系统不同检索深度内的平均召回。也就是说，我们报告原始系统 top-k 段落中，有多少出现在仅用质心系统的 top-$k'$ 内，其中 $k'\geq k$。

<!-- S3.SS3.p3.1 -->
结果支持这一假设，既包括域内 MS MARCO v1，也包括域外 LoTTE pooled dev 的 search 查询。例如，只用质心检索 $10k$ 个段落，这些段落仍包含完整 ColBERTv2 流水线 top-k 中超过 99% 的段落。（原段落引文：[42]。）

![图 3：相对原始检索器 top-k 的召回](figs/fig03.png)

**图 3。** 仅质心版本相对于完整 ColBERTv2 的 top-k 段落召回，分别对应 (a) k=10、(b) k=100、(c) k=1000。10k 或更少候选可以找回几乎全部基线 top-k，支持质心交互筛选。这里的 recall 是对基线结果集的覆盖，不能等同于对人工相关性标签的召回。

### 每次查询只需要部分质心

<!-- S3.SS4.p1.1 -->
我们进一步假设，对给定查询，只有少量段落 embedding 簇在判断相关性时明显比其他簇重要。如果如此，就可以优先计算高权重质心，丢弃那些不会显著贡献最终排序的质心。为验证这一点，我们随机选取 15 个 MS MARCO v1 查询，画出每个质心在全部查询 token 上最大相关性分数的经验累积分布函数，见图 4。确实存在一小段高权重尾部，其相关性分数远高于其他质心。我们还在 LoTTE pooled 查询上重复实验，观察到相似分布，图 4 未展示这些结果。

![图 4：质心最大分数的经验分布](figs/fig04.png)

**图 4。** 随机选取的 15 个 MS MARCO v1 dev 查询，在 ColBERTv2 下的质心分数分布。横轴为质心对全部查询 token 的最大分数，纵轴为 eCDF；高分质心只占小部分，为阈值剪枝提供经验依据。

## PLAID 方法

<!-- S4.p1.1 -->
图 5 展示由连续检索、过滤、排序阶段构成的 PLAID 打分流水线。第一阶段计算每个质心相对于查询 embedding 的相关性，生成初始候选集。中间阶段用新的质心交互和质心剪枝方法，积极但有效地过滤候选。最后，用完整重建的段落 embedding 对剩余候选排序。下面分别讨论这些模块。

![图 5：PLAID 四阶段流水线](figs/fig05.png)

**图 5。** 第一阶段用质心生成初始候选；第二阶段用质心剪枝和交互保留 ndocs；第三阶段不剪枝地做质心交互，在实验中保留 ndocs/4；最后解压残差、完整 MaxSim 打分并输出 top-k。这里“完整”指压缩表示的完整解压，不是保证恢复压缩前浮点向量。

### 候选生成

<!-- S4.SS1.p1.1 -->
给定查询 embedding 矩阵 $Q$ 与索引中的质心向量集合 $C$，PLAID 用矩阵乘法计算 token 级查询—质心相关性分数 $S_{c,q}$。

<!-- equation 2 -->
$$
S_{c,q}=C\cdot Q^{T}\\

$$

式 (2)，保留原式。

<!-- S4.SS1.p1.2 -->
随后选择与每个查询 token 的 top-t 质心“接近”的段落，组成初始候选集。一个段落接近某个质心，当且仅当它的一个或多个 token 在索引阶段的 k-means 聚类中被分配给该质心。原始 ColBERTv2 把 t 称为 nprobe；PLAID ColBERTv2 延续这一术语。

<!-- S4.SS1.p2.1 -->
PLAID 的初始候选生成与原始 ColBERTv2 有两个关键区别。首先，原始系统的倒排表从质心映射到 embedding ID；PLAID 则从质心映射到对应的唯一 passage ID。段落远少于 embedding，因此存 passage ID 所需信息更少，也能用 32-bit 整数替代可能需要的 64-bit 长整数。实际在包含约 1.4 亿段落的 MS MARCO v2 上，倒排表空间减少 2.7 倍，从 71 GB 降为 27 GB。（原段落引文：[6]。）

<!-- S4.SS1.p3.1 -->
其次，原始系统在初始候选过多、超过 ncandidates 指定规模时，会选择部分候选 embedding 做完整残差解压、打分与排序以剪枝。这些 embedding 就是原始质心→embedding ID 倒排表中的向量，解压开销如前述很高。PLAID 不对初始候选数量施加限制，因为后续可用质心交互与剪枝廉价过滤候选段落。

### 质心交互

<!-- S4.SS2.p1.1 -->
质心交互在标准 MaxSim 中用每个 token 最近的质心替代其 embedding，以很低成本近似段落相关性。把它作为额外过滤阶段后，大量候选可以跳过昂贵的 embedding 重建，使端到端检索更快。直观上，PLAID 模仿传统词袋检索：质心相关性分数扮演 BM25 等系统中词项相关性分数的角色。不过，由于查询仍有向量表示，质心相关性在查询时计算，而不像传统系统那样预计算词项分数。

<!-- S4.SS2.p2.1 -->
具体过程如下。式 (2) 的 $S_{c,q}$ 存储每个质心对查询 token 的相关性。设 $I$ 为候选集各 token 对应的质心索引列表，$S_{c,q}[i]$ 表示分数矩阵的第 i 行。PLAID 据此构造基于质心的近似分数 $\tilde D$。

<!-- equation 3 -->
$$
\tilde{D}=\begin{bmatrix}S_{c,q}[I_{1}]\\
S_{c,q}[I_{2}]\\
\cdots\\
S_{c,q}[I_{|\tilde{D}|}]\end{bmatrix}
$$

式 (3)，保留原式。

<!-- S4.SS2.p2.2 -->
用 $\tilde D$ 对候选段落排序时，PLAID 计算以下 MaxSim 分数。

<!-- equation 4 -->
$$
S_{\tilde{D}}=\sum_{i}^{|Q|}\max_{j=1}^{|\tilde{D}|}\tilde{D}_{i,j}
$$

式 (4)，保留原式。

> 译注：式 (3) 按“质心/token 行、查询列”堆叠；式 (4) 却把查询索引写在第一维。此处保留原式，维度一致的表达与代码核对见深读稿“公式审查”。

<!-- S4.SS2.p2.3 -->
按 $S_{\tilde D}$ 选出的 top-k 最相关段落，构成过滤后的候选段落集合。

<!-- S4.SS2.p3.1 -->
PLAID 提供优化内核，高效执行质心交互及更一般的 MaxSim 操作，详见原文 §4.5。

### 质心剪枝

<!-- S4.SS3.p1.1 -->
作为额外优化，PLAID 利用原文 §3.3 的观察，在构造 $\tilde D$ 前剪去低分质心。在这个过滤阶段，只对对应质心的最大分数达到阈值 $t_{cs}$ 的 token 打分。具体来说，只有当 token 对应的质心 i 满足下式时，该 token 才进入 $\tilde D$。

<!-- equation 5 -->
$$
\max_{j=1}^{|Q|}S_{c,q_{i,j}}\geq t_{cs}
$$

式 (5)，保留原式。

<!-- S4.SS3.p2.1 -->
我们用 ndocs 表示第二阶段选出的候选文档数。经验上，第三阶段选择 ndocs/4 个候选会得到较好结果；原文 §5 的全部结果都使用这一启发式。

### 最终打分

<!-- S4.SS4.p1.1 -->
和原始 ColBERTv2 一样，PLAID 通过残差解压重建最终候选集的 embedding，再用 MaxSim 排序。设 $D$ 为最终候选集解压后的重建向量，则最终 $S_{q,d}$ 按式 (1) 计算。

<!-- S4.SS4.p2.1 -->
原文 §4.5 讨论加速 MaxSim 与解压操作的内核。

### 无 padding 的 MaxSim 与解压优化

<!-- S4.SS5.p1.1 -->
图 2a 表明，索引读取是原始 ColBERTv2 的重要开销。其中一个原因是，需要把二维索引张量重新组织并 padding，增加一个代表最大段落长度的维度。得到的三维张量便于对不等长 token 向量列表批量执行 MaxSim。为避免 padding，我们编写自定义 C++ 代码，直接在打包的二维索引张量上计算 MaxSim；其中多个不同长度的二维子张量沿同一维拼接。内核遍历各段落 token，计算其对每个查询 token 的最大分数，再沿全部查询 token 求和。这种设计容易按段落并行；每个线程只需分配一个输出向量保存并原地更新各查询 token 的最大值，内存用量为 $O(|Q|)$。padding 方式则需要 $O(|D|\cdot|Q|)$ 空间。我们把这一设计用于质心交互和最终 MaxSim（图 5 第四阶段）。这些内核仅实现于 CPU；对应的 GPU 内核仍是未来工作。

<!-- S4.SS5.p2.1 -->
ColBERTv2 的残差解压先计算质心向量列表，确定从质心出发的 $2^b$ 种可能差值，然后保存每个 embedding 向量对应的差值索引。每个压缩的 8-bit 值保存 $8/b$ 个取值范围为 $[0,2^b)$ 的索引。图 2a 显示解压开销很高，部分原因是原始实现要显式展开 bit，执行昂贵的位移和求和以恢复数值。PLAID 则预计算一个 8-bit 打包值可能编码的全部 $2^8$ 组索引，把结果存入查找表，解压时直接查表而不用手动展开。我们为 CPU 和 GPU 都提供了优化实现。GPU 的自定义 CUDA 内核为压缩残差张量中的每个字节分配一个线程；对 d 维 embedding，线程块大小为 $bd/8$。CPU 则按段落并行解压。

## 实验评测

<!-- S5.p1.1 -->
评测希望回答以下研究问题。

<!-- S5.I1.i1.p1.1 -->
PLAID 如何影响不同 IR benchmark 的端到端延迟和检索质量（原文 §5.2）？

<!-- S5.I1.i2.p1.1 -->
各项优化对加速分别贡献多少（原文 §5.3）？

<!-- S5.I1.i3.p1.1 -->
随着语料规模和并行度增长，PLAID 的扩展性如何（原文 §5.4）？

### 实验设置

<!-- S5.SS1.SSS0.Px1.p1.1 -->
PLAID 包含质心交互与残差解压优化，以模块化方式扩展基于 PyTorch 的 ColBERTv2 实现，尤其是搜索组件。CPU 上，质心交互和解压全部用多线程 C++ 实现。GPU 上，质心交互使用 PyTorch，快速解压使用 CUDA 内核。整体增加约 300 行 Python 和 700 行 C++。

<!-- table 1 -->
| 数据集 | 段落数 | Token 数 | 查询数 | 原始索引 GiB | PLAID 索引 GiB |
| --- | --- | --- | --- | --- | --- |
| MS MARCO v1 | 8.8M | 597.9M | 6980 | 24.6 | 21.6 |
| Wikipedia | 21.0M | 2.6B | 8757 | 105.2 | 92.0 |
| LoTTE pooled | 2.4M | 339.4M | 2931 | 14.0 | 12.3 |
| MS MARCO v2 | 138.4M | 9.4B | 3903 | 246.0 | 202.2 |

**表 1。** 四个评测 benchmark 的统计信息及原始/PLAID 索引大小。M/B 分别表示百万/十亿，索引单位为 GiB。

<!-- S5.SS1.SSS0.Px2.p1.1 -->
评测覆盖表 1 的四个 IR benchmark。MS MARCO v1 与 Wikipedia 开放域问答属于域内评测，检索器分别针对其任务训练；基于 StackExchange 的 LoTTE、TREC 2021 Deep Learning Track 的 MS MARCO v2 属于域外评测，使用在 MS MARCO v1 上训练的 ColBERTv2。Wikipedia 使用 2018 年 12 月的 dump，查询来自 Natural Questions。LoTTE 使用 pooled dev 的 search 风格查询。MS MARCO v2 使用 augmented passage 版本，加入段落标题、忽略 headings。由于评测多个模型配置，全部评测使用 development set 查询。（原段落引文：[42], [6], [18], [23], [2]。）

<!-- table 2 -->
| k | nprobe | t_cs | ndocs |
| --- | --- | --- | --- |
| 10 | 1 | 0.5 | 256 |
| 100 | 2 | 0.45 | 1024 |
| 1000 | 4 | 0.4 | 4096 |

**表 2。** PLAID 不同 k 档位的超参数。

<!-- S5.SS1.SSS0.Px3.p1.1 -->
端到端比较包括原始 ColBERTv2、PLAID ColBERTv2、ColBERTv1、BM25、SPLADEv2 和 DPR。原始 ColBERTv2 使用其论文为各 benchmark 报告的超参数，在结果表中以 p（nprobe）、c（ncandidates）表示。PLAID 测试 k=10、100、1000 三档；k 控制最终打分文档数量及方法节描述的检索超参数，各档配置见表 2。经验上，在最终打分阶段排序 ndocs/4 个文档会得到较强结果。原始和 PLAID 两者对除 MS MARCO v2 外的全部数据集均采用每维 2 bit 压缩；MS MARCO v2 为每维 1 bit。（原段落引文：[22], [41], [10], [18]。）

<!-- S5.SS1.SSS0.Px4.p1.1 -->
全部实验使用配备 28 个 Intel Xeon Gold 6132 2.6 GHz CPU 核心的服务器，每核心两个线程、共 56 线程，并各配备 4 张 NVIDIA TITAN V GPU。服务器有两个 NUMA socket，单 socket 内存延迟约 92 ns、跨 socket 约 142 ns；带宽分别约 72 GBps 与 33 GBps。每张 TITAN V 有 12 GB 高带宽显存。

<!-- S5.SS1.SSS0.Px5.p1.1 -->
端到端延迟测量先计算全部查询的平均延迟（查询数量见表 1），重复三次，报告三次平均值中的最小值。其他实验的计时方法在相应章节说明。遵循 Mackenzie 等（2021），神经模型的查询编码延迟不计入，包括 ColBERTv1、原始 ColBERTv2、PLAID、SPLADEv2；此前研究指出，量化、蒸馏等标准技术可让 BERT 执行成本很小。计时在无其他负载的机器上进行；命令加 `numactl --membind 0`，保证 socket 内 I/O，但 MS MARCO v2 索引可能需要两侧 NUMA 节点，因此不使用该设置。GPU 结果允许使用全部 56 CPU 线程；纯 CPU 结果用 `torch.set_num_threads` 限制为 1 或 8 线程。非 ColBERT 系统的单线程延迟取自 Mackenzie 等（2021），测量硬件和实现不同，只用于说明 PLAID 有竞争力，不能作为严格的绝对比较。（原段落引文：[22], [42], [10], [30], [4]。）

### 端到端结果

<!-- table 3 -->
| 系统 | MRR@10 | R@100 | R@1k | CPU 1线程 ms | CPU 8线程 ms | GPU ms |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 (PISA; k=1000) | 18.7* | - | - | 8.3* | - | - |
| SPLADEv2 (PISA; k=1000) | 36.8* | - | 97.9* | 220.3* | - | - |
| ColBERTv1 | 36.1 | 87.3 | 95.2 | - | - | 54.3 |
| Vanilla ColBERTv2 (p=2,c=2^13) | 39.7 | 90.4 | 96.6 | 3485.1 | 921.8 | 53.4 |
| Vanilla ColBERTv2 (p=4,c=2^16) | 39.7 | 91.4 | 98.3 | - | 4568.5 | 259.6 |
| PLAID ColBERTv2 (k=10) | 39.4 | - | - | 185.5 | 31.5 | 11.5 |
| PLAID ColBERTv2 (k=100) | 39.8 | 90.6 | - | 222.3 | 52.9 | 20.2 |
| PLAID ColBERTv2 (k=1000) | 39.8 | 91.3 | 97.5 | 352.3 | 101.3 | 38.4 |

**表 3。** MS MARCO v1 域内评测。星号质量数字来自 Formal 等，星号延迟来自 Mackenzie 等。MRR/Recall 按原表百分数尺度呈现；“-”表示未报告。

<!-- S5.SS2.p1.1 -->
表 3 展示 MS MARCO v1 的域内结果。在最保守的 k=1000 配置下，PLAID 的 MRR@10 和 Recall@100 与原始 ColBERTv2 相当，同时 GPU 加速 6.8 倍、CPU 加速 45 倍。允许质量小幅下降后，可进一步达到 GPU 12.9–22.6 倍、CPU 86.4–145 倍加速。相对其他系统，PLAID 的延迟同样具有竞争力，与 SPLADEv2 的差距在 1.6 倍内，同时检索质量更高。

<!-- table 4 -->
| 系统 | Success@5 | Success@100 | CPU 8线程 ms | GPU ms |
| --- | --- | --- | --- | --- |
| DPR | 66.8 | 85.0 | - | - |
| ColBERT-QA uncompressed | 75.3 | 89.2 | - | - |
| Vanilla ColBERT-QA (p=4,c=2^15; residual compression) | 74.3 | 89.0 | 5077.9 | 204.1 |
| PLAID ColBERT-QA (k=10; residual compression) | 73.3 | - | 67.1 | 13.6 |
| PLAID ColBERT-QA (k=100; residual compression) | 74.1 | 88.0 | 120.1 | 26.9 |
| PLAID ColBERT-QA (k=1000; residual compression) | 74.4 | 88.9 | 228.4 | 55.3 |

**表 4。** Wikipedia 开放域问答的域内检索评测。使用 ColBERT-QA 的 NQ checkpoint，再应用 ColBERTv2 残差压缩，比较原始与 PLAID 检索；DPR 结果来自 Karpukhin 等。压缩检索器与未压缩 ColBERT-QA 的差异不能算成 PLAID 单独造成的差异。

<!-- S5.SS2.p2.1 -->
Wikipedia OpenQA 的域内评测也呈现相似趋势，见表 4。相对原始 ColBERTv2，PLAID 在无质量损失时 GPU 加速 3.7 倍、CPU 加速 22 倍；允许小幅质量损失后，GPU 为 7.6–15 倍、CPU 为 42.3–75.7 倍。

<!-- table 5 -->
| 系统 | Success@5 | Success@100 | CPU 8线程 ms | GPU ms |
| --- | --- | --- | --- | --- |
| BM25 | 47.8* | 77.6* | - | - |
| SPLADEv2 | 67.0* | 89.0* | - | - |
| Vanilla ColBERTv2 (p=2,c=2^13) | 69.3 | 90.3 | 1508.4 | 66.9 |
| PLAID ColBERTv2 (k=10) | 69.1 | - | 35.5 | 9.2 |
| PLAID ColBERTv2 (k=100) | 69.4 | 89.9 | 64.8 | 17.4 |
| PLAID ColBERTv2 (k=1000) | 69.6 | 90.5 | 163.1 | 27.3 |

**表 5。** LoTTE pooled dev 的域外评测。星号数字来自 Santhanam 等的 ColBERTv2 工作。

<!-- S5.SS2.p3.1 -->
LoTTE pooled 的结果证明 PLAID 也适用于域外设置。表 5 中，k=1000 时，相对原始 ColBERTv2，GPU 加速 2.5 倍、CPU 加速 9.2 倍，而且质量有所提高。允许少量质量损失后，GPU 为 3.8–7.3 倍、CPU 为 23.2–42.5 倍。LoTTE 上 PLAID 的 CPU 延迟比 MS MARCO v1 更高，是因为平均段落长度大约为后者的两倍。

<!-- table 6 -->
| 系统 | MRR@100 | R@100 | R@1k | CPU 8线程 ms | GPU ms |
| --- | --- | --- | --- | --- | --- |
| BM25 (Anserini; augmented) | 8.7 | 40.3 | 69.3 | - | - |
| Vanilla ColBERTv2 (p=4,c=2^16) | 18.0 | 68.2 | 88.1 | 5228.5 | OOM |
| PLAID ColBERTv2 (k=10) | - | - | - | 136.4 | 47.1 |
| PLAID ColBERTv2 (k=100) | 17.9 | 67.0 | - | 181.9 | 96.1 |
| PLAID ColBERTv2 (k=1000) | 18.0 | 68.4 | 85.7 | 251.3 | OOM |

**表 6。** MS MARCO v2 域外评测；BM25 结果来自 Anserini 回归结果。OOM 表示显存不足。

<!-- S5.SS2.p4.1 -->
表 6 表明 PLAID 可以有效扩展到包含约 1.38 亿段落、94 亿 token 的 MS MARCO v2，规模约为 v1 的 16 倍。与其他数据集的趋势一致，在最多 100 个段落的范围内，PLAID 的 CPU 速度快 20.8 倍且没有质量损失。不过 k=1000 时，原始与 PLAID 两者都在 GPU 上发生显存不足。我们认为，实现原文 §4.5 所述用于 GPU 的无 padding MaxSim 内核，可以缓解这一问题。

> 译注：“没有质量损失”保留作者表述，但表 6 的 k=100 档 MRR@100 为 18.0→17.9、R@100 为 68.2→67.0；k=1000 档 R@1000 为 88.1→85.7。评估结论需要逐指标核对，不能理解为每个深度下完全相同。

### 消融

<!-- S5.SS3.p1.1 -->
图 6 分别分解 GPU 和 CPU 的性能提升。测量使用随机抽取的 500 个 MS MARCO v1 查询，因此与表 3 的绝对值有少量差异。以原始 ColBERTv2 为基线，先加入不剪枝的质心交互（图 5 第三阶段），再加入带质心剪枝的交互（第二阶段），最后加入原文 §4.5 的优化内核。适用时使用表 2 中最保守的 k=1000 超参数。

<!-- S5.SS3.p2.1 -->
打分流水线的算法改进和实现优化都对 PLAID 很重要。仅质心交互阶段即可带来 GPU 5.2 倍、CPU 8.6 倍加速；加入实现优化后，分别再提高约 1.3 倍、4.9 倍。只在 CPU 上启用优化 C++ 内核而不使用质心交互时（图 6 未展示），端到端仅加速 3 倍，而完整 PLAID 加速 42.4 倍。

![图 6：逐步加入优化的消融](figs/fig06.png)

**图 6。** (a) GPU：1.0→3.7→5.2→6.6 倍；(b) 8 线程 CPU：1.0→4.2→8.6→42.4 倍。各条都是相对同一 baseline 的累计倍率，不能把这些数值逐项相乘。

### 可扩展性

<!-- S5.SS4.p1.1 -->
我们从语料大小和 CPU 并行度两个方面评估 PLAID 的可扩展性。

<!-- S5.SS4.p2.1 -->
图 7 首先绘出各 benchmark 上测得的端到端 PLAID 延迟与数据集大小的关系；大小以 embedding 数量计。不同数据集的延迟不一定可以直接比较，例如段落长度不同，但我们仍希望观察高层趋势。整体而言，PLAID 延迟似乎随数据集大小的平方根增长。直觉解释是，ColBERTv2 把质心数设置为与 embedding 数量的平方根成正比，而候选生成开销与分区数量负相关。

![图 7：延迟随 embedding 规模变化](figs/fig07.png)

**图 7。** (a) GPU、(b) 8 线程 CPU，展示三种 k 配置的端到端延迟；坐标为双对数，数据集大小来自表 1，延迟来自表 3–6。此图只是跨数据集趋势，不是固定工作负载下的复杂度证明。

<!-- S5.SS4.p3.1 -->
图 8 展示可用 CPU 线程数与 PLAID 延迟的关系，分别测试 k=10、100、1000，使用随机抽取的 500 个 MS MARCO v1 查询。PLAID 能利用额外线程；尤其 k=1000 时，16 线程相对单线程加速 4.9 倍，但并非理想线性扩展。可能原因包括本工作尚未底层优化的原始 ColBERTv2 候选生成，以及段落长度不均造成的线程负载失衡。更充分的 profiling 与解决方案留待未来工作。

![图 8：CPU 线程扩展](figs/fig08.png)

**图 8。** 横轴为线程数，对 k=10/100/1000 绘制延迟；线程增多有收益，但 16 线程不会自然得到 16 倍加速。

## 结论

<!-- S6.p1.1 -->
本文提出高效后期交互引擎 PLAID，通过积极、廉价地过滤候选段落加速检索。我们发现，相对于原始 ColBERTv2，仅用质心仍有高召回，且质心相关性分数分布偏向较小值。基于这些观察，我们提出质心交互，把它融入 PLAID ColBERTv2 的多阶段打分流水线，同时实现无 padding MaxSim 与残差解压的定制优化内核。多个 IR benchmark 的评测显示，相对原始 ColBERTv2，PLAID 在几乎没有质量损失的情况下，GPU 加速 2.5–6.8 倍、CPU 加速 9.2–45 倍，并能有效扩展到 1.4 亿段落。

## 原文脚注

- 作者标记：Santhanam 与 Khattab 同等贡献。
- 脚注 1：代码维护于官方 ColBERT 仓库。原文写于 2022 年 5 月，当时 PLAID 位于 `fast_search` 分支，即将合并至主分支。译注：本次另核对了当前 main，不能将当前代码版本冒充论文当年的实现。
- 脚注 2：32-bit passage ID 假设语料不超过 $2^{32}$、约 40 亿段落，这仍约为 MS MARCO v2 规模的 30 倍。

## 术语对照

| English | 本文译法 |
| --- | --- |
| late interaction | 后期交互（亦称迟交互） |
| centroid interaction | 质心交互 |
| centroid pruning | 质心剪枝 |
| bag of centroids | 质心袋 |
| residual decompression | 残差解压 |
| padding-free | 无 padding |
| passage / passage ID | 段落 / 段落 ID |
| recall of baseline top-k | 对基线 top-k 结果集的召回 |

## 参考文献（保留英文书目信息）

[1] Abuzaid et al . (2019) Firas Abuzaid, Geet Sethi, Peter Bailis, and Matei Zaharia. 2019. To index or not to index: Optimizing exact maximum inner product search. In 2019 IEEE 35th International Conference on Data Engineering (ICDE) . IEEE, 1250–1261.

[2] Authors (2021) Anserini GitHub Repo Authors. 2021. Passage Collection (Augmented). https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-v2.md#passage-collection-augmented

[3] Authors (2022) Anserini GitHub Repo Authors. 2022. Anserini Regressions: MS MARCO (V2) Passage Ranking. https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-v2-passage-augmented.md

[4] Bergum (2021) Jo Kristian Bergum. 2021. Pretrained Transformer Language Models for Search - part 3. https://blog.vespa.ai/pretrained-transformer-language-models-for-search-part-3/

[5] Broder et al . (2003) Andrei Z Broder, David Carmel, Michael Herscovici, Aya Soffer, and Jason Zien. 2003. Efficient query evaluation using a two-level retrieval process. In CIKM .

[6] Craswell et al . (2022) Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Jimmy Lin. 2022. Overview of the TREC 2021 deep learning track. In Text REtrieval Conference (TREC) . TREC. https://www.microsoft.com/en-us/research/publication/overview-of-the-trec-2021-deep-learning-track/

[7] Dai and Callan (2020) Zhuyun Dai and Jamie Callan. 2020. Context-Aware Term Weighting For First Stage Passage Retrieval. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020 , Jimmy Huang, Yi Chang, Xueqi Cheng, Jaap Kamps, Vanessa Murdock, Ji-Rong Wen, and Yiqun Liu (Eds.). ACM, 1533–1536. https://doi.org/10.1145/3397271.3401204

[8] Dimopoulos et al . (2013) Constantinos Dimopoulos, Sergey Nepomnyachiy, and Torsten Suel. 2013. Optimizing top-k document retrieval strategies for block-max indexes. In WSDM .

[9] Ding and Suel (2011) Shuai Ding and Torsten Suel. 2011. Faster top-k document retrieval using block-max indexes. In SIGIR .

[10] Formal et al . (2021a) Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. 2021a. SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval. arXiv preprint arXiv:2109.10086 (2021). https://arxiv.org/abs/2109.10086

[11] Formal et al . (2021b) Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021b. SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval . 2288–2292.

[12] Gao et al . (2021) Luyu Gao, Zhuyun Dai, and Jamie Callan. 2021. COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies . Association for Computational Linguistics, Online, 3030–3042. https://doi.org/10.18653/v1/2021.naacl-main.241

[13] Hofstätter et al . (2020) Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, and Allan Hanbury. 2020. Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation. arXiv preprint arXiv:2010.02666 (2020). https://arxiv.org/abs/2010.02666

[14] Hofstätter et al . (2022) Sebastian Hofstätter, Omar Khattab, Sophia Althammer, Mete Sertkan, and Allan Hanbury. 2022. Introducing Neural Bag of Whole-Words with ColBERTer: Contextualized Late Interactions using Enhanced Reduction. arXiv preprint arXiv:2203.13088 (2022).

[15] Humeau et al . (2020) Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. 2020. Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020 . OpenReview.net. https://openreview.net/forum?id=SkxgnnNFvH

[16] Jegou et al . (2010) Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence 33, 1 (2010), 117–128.

[17] Johnson et al . (2019) Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with gpus. IEEE Transactions on Big Data (2019).

[18] Karpukhin et al . (2020) Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) . Association for Computational Linguistics, Online, 6769–6781. https://doi.org/10.18653/v1/2020.emnlp-main.550

[19] Khattab et al . (2020) Omar Khattab, Mohammad Hammoud, and Tamer Elsayed. 2020. Finding the best of both worlds: Faster and more robust top-k document retrieval. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval . 1031–1040.

[20] Khattab et al . (2021a) Omar Khattab, Christopher Potts, and Matei Zaharia. 2021a. Baleen: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval. In Thirty-Fifth Conference on Neural Information Processing Systems .

[21] Khattab et al . (2021b) Omar Khattab, Christopher Potts, and Matei Zaharia. 2021b. Relevance-guided Supervision for OpenQA with ColBERT. Transactions of the Association for Computational Linguistics 9 (2021), 929–944.

[22] Khattab and Zaharia (2020) Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020 , Jimmy Huang, Yi Chang, Xueqi Cheng, Jaap Kamps, Vanessa Murdock, Ji-Rong Wen, and Yiqun Liu (Eds.). ACM, 39–48. https://doi.org/10.1145/3397271.3401075

[23] Kwiatkowski et al . (2019) Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A Benchmark for Question Answering Research. Transactions of the Association for Computational Linguistics 7 (2019), 452–466. https://doi.org/10.1162/tacl_a_00276

[24] Li et al . (2021) Yulong Li, Martin Franz, Md Arafat Sultan, Bhavani Iyer, Young-Suk Lee, and Avirup Sil. 2021. Learning Cross-Lingual IR from an English Retriever. arXiv preprint arXiv:2112.08185 (2021).

[25] Lin (2022) Jimmy Lin. 2022. A proposed conceptual framework for a representational approach to information retrieval. In ACM SIGIR Forum , Vol. 55. ACM New York, NY, USA, 1–29.

[26] Lupart and Clinchant (2022) Simon Lupart and Stéphane Clinchant. 2022. Toward A Fine-Grained Analysis of Distribution Shifts in MSMARCO. arXiv preprint arXiv:2205.02870 (2022).

[27] MacAvaney et al . (2019) Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. 2019. CEDR: Contextualized Embeddings for Document Ranking. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval . 1101–1104. https://doi.org/10.1145/3331184.3331317

[28] Macdonald and Tonellotto (2021) Craig Macdonald and Nicola Tonellotto. 2021. On approximate nearest neighbour selection for multi-stage dense retrieval. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management . 3318–3322.

[29] Macdonald et al . (2021) Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2021. On Single and Multiple Representations in Dense Passage Retrieval. arXiv preprint arXiv:2108.06279 (2021).

[30] Mackenzie et al . (2021) Joel Mackenzie, Andrew Trotman, and Jimmy Lin. 2021. Wacky Weights in Learned Sparse Representations and the Revenge of Score-at-a-Time Query Evaluation. arXiv preprint arXiv:2110.11540 (2021).

[31] Malkov and Yashunin (2018) Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE transactions on pattern analysis and machine intelligence 42, 4 (2018), 824–836.

[32] Mallia et al . (2021) Antonio Mallia, Omar Khattab, Torsten Suel, and Nicola Tonellotto. 2021. Learning passage impacts for inverted indexes. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval . 1723–1727.

[33] Mallia et al . (2017) Antonio Mallia, Giuseppe Ottaviano, Elia Porciani, Nicola Tonellotto, and Rossano Venturini. 2017. Faster blockmax wand with variable-sized blocks. In SIGIR .

[34] Mallia et al . (2019) Antonio Mallia, Michal Siedlaczek, Joel Mackenzie, and Torsten Suel. 2019. PISA: Performant indexes and search for academia. Proceedings of the Open-Source IR Replicability Challenge (2019).

[35] Minas Krasakis et al . (2022) Antonios Minas Krasakis, Andrew Yates, and Evangelos Kanoulas. 2022. Zero-shot Query Contextualization for Conversational Search. arXiv e-prints (2022), arXiv–2204.

[36] Nguyen et al . (2016) Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. MS MARCO: A Human-Generated MAchine Reading COmprehension Dataset. arXiv preprint arXiv:1611.09268 (2016). https://arxiv.org/abs/1611.09268

[37] Nogueira and Cho (2019) Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT. arXiv preprint arXiv:1901.04085 (2019). https://arxiv.org/abs/1901.04085

[38] Paranjape et al . (2022) Ashwin Paranjape, Omar Khattab, Christopher Potts, Matei Zaharia, and Christopher D Manning. 2022. Hindsight: Posterior-guided Training of Retrievers for Improved Open-ended Generation. In International Conference on Learning Representations . https://openreview.net/forum?id=Vr_BTpw3wz

[39] Qu et al . (2021) Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. 2021. RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies . Association for Computational Linguistics, Online, 5835–5847. https://doi.org/10.18653/v1/2021.naacl-main.466

[40] Ren et al . (2021) Ruiyang Ren, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Qiaoqiao She, Hua Wu, Haifeng Wang, and Ji-Rong Wen. 2021. RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking. arXiv preprint arXiv:2110.07367 (2021). https://arxiv.org/abs/2110.07367

[41] Robertson et al . (1995) Stephen E Robertson, Steve Walker, Susan Jones, Micheline M Hancock-Beaulieu, Mike Gatford, et al . 1995. Okapi at TREC-3. NIST Special Publication (1995).

[42] Santhanam et al . (2021) Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2021. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. arXiv preprint arXiv:2112.01488 (2021).

[43] Thai et al . (2022) Katherine Thai, Yapei Chang, Kalpesh Krishna, and Mohit Iyyer. 2022. RELIC: Retrieving Evidence for Literary Claims. arXiv preprint arXiv:2203.10053 (2022).

[44] Thakur et al . (2021) Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2) . https://openreview.net/forum?id=wCu6T5xFjeJ

[45] Tonellotto and Macdonald (2021) Nicola Tonellotto and Craig Macdonald. 2021. Query embedding pruning for dense retrieval. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management . 3453–3457.

[46] Tonellotto et al . (2018) Nicola Tonellotto, Craig Macdonald, Iadh Ounis, et al . 2018. Efficient Query Processing for Scalable Web Search. Foundations and Trends® in Information Retrieval (2018).

[47] Turtle and Flood (1995) Howard Turtle and James Flood. 1995. Query evaluation: strategies and optimizations. IP & M (1995).

[48] Wang et al . (2021) Xiao Wang, Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2021. Pseudo-relevance feedback for multiple representation dense retrieval. In Proceedings of the 2021 ACM SIGIR International Conference on Theory of Information Retrieval . 297–306.

[49] Xiong et al . (2020) Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. In International Conference on Learning Representations .

[50] Yang et al . (2018) Peilin Yang, Hui Fang, and Jimmy Lin. 2018. Anserini: Reproducible ranking baselines using Lucene. Journal of Data and Information Quality (JDIQ) 10, 4 (2018), 1–20.

[51] Zeng et al . (2022) Hansi Zeng, Hamed Zamani, and Vishwa Vinay. 2022. Curriculum Learning for Dense Retrieval Distillation. arXiv preprint arXiv:2204.13679 (2022).

[52] Zhan et al . (2020) Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Min Zhang, and Shaoping Ma. 2020. Learning To Retrieve: How to Train a Dense Retrieval Model Effectively and Efficiently. arXiv preprint arXiv:2010.10469 (2020). https://arxiv.org/abs/2010.10469

[53] Zhan et al . (2022) Jingtao Zhan, Xiaohui Xie, Jiaxin Mao, Yiqun Liu, Min Zhang, and Shaoping Ma. 2022. Evaluating Extrapolation Performance of Dense Retrieval. arXiv preprint arXiv:2204.11447 (2022).

[54] Zhong et al . (2022) Wei Zhong, Jheng-Hong Yang, and Jimmy Lin. 2022. Evaluating Token-Level and Passage-Level Dense Retrieval Models for Math Information Retrieval. arXiv preprint arXiv:2203.11163 (2022).

