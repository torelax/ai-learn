# PLAID：把质心压缩变成低延迟多向量检索

> paper：[PLAID: An Efficient Engine for Late Interaction Retrieval](https://arxiv.org/abs/2205.09707v1)  
> code / project：[官方 ColBERT](https://github.com/stanford-futuredata/ColBERT)  
> refs：[ColBERT](https://arxiv.org/abs/2004.12832)、[ColBERTv2](https://arxiv.org/abs/2112.01488)  
> backbone：实验基于 BERT 系 ColBERTv2；Wikipedia 使用 ColBERT-QA 的 NQ checkpoint  
> date：2022-05-19，arXiv v1；CIKM 2022  
> modality：文本查询—文本段落；languages：本文评测为英语任务，未验证多语迁移  
> 状态：研究报告；实验数字来自论文，未在本环境复现 benchmark。

读者前提：理解 ColBERT 的 token 级 MaxSim 与 ColBERTv2 的质心＋量化残差。  
读完收获：能画出四阶段流水线、解释质心剪枝的维度、识别加速与质量指标的边界。  
挂接主文：Embedding 表示范式、Pipeline 工程、多模态页检索和向量部署章节。

## 为什么这篇值得补读

已有 [ColBERT 深读](../ColBERT/ColBERT详解.md) 解释为什么多个 token 向量比单向量更细粒度，[ColBERTv2 深读](../ColBERTv2/ColBERTv2详解.md) 解释如何压缩它们。PLAID 补上第三个问题：**面对全库检索，怎样少读残差、少解压、少做昂贵的完整打分？**

PLAID 是检索执行引擎，不是新的 embedding 训练损失。它用已有压缩表示中的质心 ID 先排除较弱候选，把完整残差解压推迟到最后。算法筛选和执行内核共同产生收益；仅更换低层内核不能解释全部加速。

## 瓶颈不是只有 BERT 编码

![论文 Figure 1：后期交互](figs/fig01.png)

**Figure 1（PDF 第 2 页）。** 查询和段落独立编码，段落离线索引；查询 token 对段落 token 逐个取最大相似度，再求和。保留 token 细节提高表达力，也使每个候选需要读取多个向量。

![论文 Figure 2：延迟分解](figs/fig02.png)

**Figure 2（PDF 第 2 页）。** 五色图例分别为 query 编码、候选生成、索引读取、解压、打分。原始系统 (a) 的索引读取和解压占大头；PLAID (b) 减少这两部分。图示约 287→58 ms，是这张分解图的数值，不能代替 Table 3 的 259.6→38.4 ms，更不能把正文排除 query 编码的计时口径忘掉。

候选不是只存一个 128 维向量，而是段落的 token 矩阵。原始系统对大量候选取残差、构造 padding 张量、解压和打分，内存带宽及临时张量成本叠加。因此，“压缩后能放进内存”不等于“全库检索足够快”。

## 从残差表示到质心袋

一个 token 向量可用质心 ID 和量化残差表示。用本文的解释性记号：

$$
d_j\approx c_{z_j}+\hat r_j,\qquad
\hat d_j=\operatorname{normalize}(c_{z_j}+\hat r_j).
$$

这不是论文新增的训练公式。它概括 ColBERTv2 压缩与当前实现的重建、归一化过程；残差量化本身有误差，最终“完整解压”也不会保证恢复压缩前浮点值。

默认 128 维时，原始 FP16 token 向量为 256 bytes。4-byte 质心 ID 加 1-bit/2-bit 每维残差，主体约为 20/36 bytes；这些只是单 token 的理论主体，不包含质心表、倒排表、文档长度、偏移与其他元数据。不能据此声称整个索引固定缩小 12.8/7.1 倍。

如果早期只看 $z_j$，就能暂时不读残差。一个段落被看成其质心 ID 的集合：对 MaxSim 来说，重复的同一质心不会增加最大值。它与 BM25 词袋有结构相似之处，但不是词频模型，也不是把自然语言 token 变成精确词法匹配；质心分数仍取决于本次查询向量。

## 两个经验观察，不能误读为保证

![论文 Figure 3：质心候选覆盖](figs/fig03.png)

**Figure 3（PDF 第 3 页）。** 三个子图分别为 k=10/100/1000，横轴为质心检索的候选数；在 MS MARCO v1 与 LoTTE pooled dev 上，取约 10k 个质心候选可以覆盖基线 top-k 的 99% 以上。分母是 **完整 ColBERTv2 的 top-k 集合**，不是人工 qrels；这是候选覆盖证据，不能当成 99% 的任务相关性召回。

![论文 Figure 4：最大质心分数分布](figs/fig04.png)

**Figure 4（PDF 第 3 页）。** 对随机 15 个 MS MARCO v1 查询，每条曲线统计质心对所有查询 token 的最大分数的 eCDF。大部分质心分数低，只有小尾部较高，支持先剪掉弱质心。但图中样本很少；部署时阈值仍需在自己的查询和语料分布上验证。

## 四阶段流水线

![论文 Figure 5：PLAID 方法图](figs/fig05.png)

**Figure 5（PDF 第 5 页）。** Stage 1 用 query–centroid 分数选择每个 query token 的 top-nprobe 质心并查倒排表，合并 passage ID；Stage 2 先质心剪枝，再质心 MaxSim，保留 ndocs；Stage 3 恢复全部质心做交互，保留 ndocs/4；Stage 4 才读取、解压这些候选的残差，完整 MaxSim 后取 top-k。

| 阶段 | 读取的信息 | 主要操作 | 保留规模 |
| --- | --- | --- | --- |
| 候选生成 | 质心表、质心→passage ID 倒排表 | 每个查询 token 的 top-nprobe，取 ID 并集去重 | 没有原始 ncandidates 式硬上限 |
| 剪枝质心交互 | 质心 ID、预计算分数 | 跳过低最大分数质心，MaxSim | ndocs |
| 全质心交互 | 候选全部质心 ID | 不剪枝地再次 MaxSim | ndocs/4 |
| 残差打分 | 候选质心及量化残差 | 解压、归一化、最终 MaxSim | 输出 top-k |

“恢复全部质心”指 Stage 3 重新使用 **已保留段落**的全部质心，不会把 Stage 2 淘汰的段落找回来。最终全分辨率打分只能校正剩余候选的排序，不能挽救早期丢失的相关文档。

倒排表从 embedding ID 改存唯一 passage ID 是另一项关键工程改进。段落数少于 embedding 数，还可采用 32-bit ID；论文报告 MS MARCO v2 倒排表约 71→27 GB、2.7 倍节省。这不是整个索引的 2.7 倍缩小，整个索引从 246.0→202.2 GiB（Table 1）。

## 公式审查：先固定轴，再写 MaxSim

设查询有 m 个 token，段落有 n 个 token，质心数为 K，向量维度为 h。本文统一采用“token/质心是行”的约定。

| 对象 | 形状 | 来源 |
| --- | --- | --- |
| $Q$ | $m\times h$ | 查询矩阵 |
| $D$ | $n\times h$ | 段落矩阵 |
| $C$ | $K\times h$ | 质心矩阵 |
| $A=CQ^T$ | $K\times m$ | 论文式 (2) |
| $B=A[z,:]$ | $n\times m$ | 论文式 (3) 的逐行 gather |

论文式 (1) 的标准分数是：

$$
S(q,d)=\sum_{i=1}^{m}\max_{j=1}^{n} q_i^T d_j.
$$

每个 query token 对 **段落轴**取最大值，然后沿 **查询轴**求和。归一化后点积等于余弦；不是所有 token pair 分数求和，也不是平均 pooling 的单向量点积。该分数通常非对称。

按上面的 B 形状，维度一致的质心近似为：

$$
\tilde S(q,d)=\sum_{i=1}^{m}\max_{j=1}^{n} B_{j,i}
=\sum_{i=1}^{m}\max_{j=1}^{n} A_{z_j,i}.
$$

**原式疑点：** 论文式 (3) 按行堆叠质心分数，式 (4) 却写 $\tilde D_{i,j}$，其 i 指查询、j 指段落；在同一轴约定下与式 (3) 相反。可以改用 $B_{j,i}$，或者先把 gather 结果转置再使用原下标。本稿采用前者；英文和译稿保留原式，不默默纠正。官方代码 `filter_pids.cpp` 对每个段落循环 code，对每个 query token 更新最大值，最后沿 query 求和，与本文规范化表达一致。

论文式 (5) 的剪枝判断在本文记号下是：

$$
\operatorname{keep}(c)=
\left[\max_{i=1}^{m} A_{c,i}\geq t_{cs}\right].
$$

判断是一个质心对 **任意查询 token**的最大分数达到阈值。只要该质心通过，就保留其对所有查询 token 的分数；不能把每个低于阈值的单元单独置零，这会变成另一种算法。边界是 $\geq$，不是 $>$。

质心近似不是严格的原向量分数上界。例如 $c=(1,0)$、$d=(0.8,0.6)$，对 $q=(0,1)$，质心打分 0、原向量打分 0.6；对 $q=(1,0)$，则是 1 与 0.8。它既可能低估也可能高估。因此，这种剪枝基于经验质量/速度权衡，不能照搬 WAND 的保守上界保证。

另外，§3.2 的 “4-bit centroid ID” 与 §3.1 的 “4 bytes” 有单位冲突。本文按 4 bytes 计算；译稿保留原文并加注。

## 内核：减少临时张量和位运算

CPU 的无 padding MaxSim 直接遍历打包的二维 token 张量，每个线程维护长度 m 的最大值缓冲，按段落并行。这里的 $O(m)$ 是 **每线程打分缓冲**，不表示整个索引或算法总内存都是 $O(m)$；计算仍需要相应 token 交互。

解压优化对可能的 256 个 packed byte 值预计算索引列表，省去反复移位和展开。b-bit 每维残差的一个字节包含 8/b 个索引，随后据量化桶表恢复差值。论文实现了 CPU 与 GPU 解压优化，但无 padding MaxSim 定制内核仅实现于 CPU；GPU 相应内核在论文中仍属未来工作。不能把后来 main 的能力全部归给 2022 年论文。

## 训练数据与关键数据集

PLAID 没有引入新的对比学习损失或新的模型训练集；它改检索引擎。实验中的 ColBERTv2 checkpoint 在 **MS MARCO v1** 上训练，用于 v1 域内和 LoTTE/v2 域外评测；Wikipedia 使用在 **NQ** 上训练的 ColBERT-QA checkpoint，再加残差压缩。去噪监督、难负例和模型训练配方由 ColBERTv2 工作继承。PLAID 本文没有完整重述训练样本数量与配方，不能把评测语料的 597.9M token 写成训练样本数。

MS MARCO 是面向网页段落检索的查询—段落 benchmark。v1 语料约 880 万段落；本文在 dev 查询上评测。v2 采用 augmented passage 版本，加入 passage title、忽略 headings，语料约 1.384 亿段落，检验大规模域外检索。

Wikipedia OpenQA 使用 2018 年 12 月 Wikipedia dump 和 Natural Questions 查询，衡量是否检索到包含答案的段落；使用独立 NQ checkpoint，所以这里的域内不等于“所有实验都用 MS MARCO checkpoint”。

LoTTE 来自 StackExchange，包含多主题、长尾问题；这里用 pooled **dev、search 风格**查询。它既检验域外泛化，也提醒我们段落长度改变会影响解压与打分成本。不能把这组结果当成 LoTTE 所有拆分或所有查询类型的综合成绩。

| 数据集（论文 Table 1） | 段落 | Token/embedding | 评测查询 | 原始→PLAID 索引 GiB |
| --- | --- | --- | --- | --- |
| MS MARCO v1 | 8.8M | 597.9M | 6980 | 24.6→21.6 |
| Wikipedia | 21.0M | 2.6B | 8757 | 105.2→92.0 |
| LoTTE pooled | 2.4M | 339.4M | 2931 | 14.0→12.3 |
| MS MARCO v2 | 138.4M | 9.4B | 3903 | 246.0→202.2 |

v1、Wikipedia、LoTTE 均用每维 2 bit 残差；v2 为每维 1 bit。比较索引大小和精度时需保留这一条件。

## 比较结果与计时口径

论文的主要因果比较是 **同模型、同压缩表示的 vanilla 与 PLAID 检索**。ColBERTv1、BM25、SPLADEv2、DPR 提供背景竞争力对照，其中部分延迟来自其他硬件与实现，不能当成严格的等硬件 benchmark。

| 论文表与设置 | 指标变化 | GPU 延迟 ms | CPU 8线程延迟 ms |
| --- | --- | --- | --- |
| Table 3：MS MARCO v1，vanilla p=4,c=$2^{16}$ → PLAID k=1000 | MRR@10 39.7→39.8；R@100 91.4→91.3；R@1000 98.3→97.5 | 259.6→38.4 | 4568.5→101.3 |
| Table 4：Wikipedia，压缩 QA → PLAID k=1000 | Success@5 74.3→74.4；Success@100 89.0→88.9 | 204.1→55.3 | 5077.9→228.4 |
| Table 5：LoTTE，vanilla → PLAID k=1000 | Success@5 69.3→69.6；Success@100 90.3→90.5 | 66.9→27.3 | 1508.4→163.1 |
| Table 6：MS MARCO v2，vanilla → PLAID k=1000 | MRR@100 18.0→18.0；R@100 68.2→68.4；R@1000 88.1→85.7 | 两者 OOM | 5228.5→251.3 |

MS MARCO v1 的 6.8× GPU、45× CPU 加速分别由 259.6/38.4 和 4568.5/101.3 得到。这里“几乎不损失质量”意味着多个主要指标接近，**不是所有深度完全不掉分**：v1 的 R@1000 下降 0.8 个百分点，v2 下降 2.4 个百分点。所有质量数字按原表百分数尺度，不把百分点变化误写为相对百分比。

计时排除了神经模型 query 编码；每次试验先取所有查询平均，报告三次平均的最小值；机器空闲，部分实验绑定 NUMA 节点。GPU 结果允许使用全部 56 CPU 线程；CPU 结果为 1 或 8 线程。GPU 型号 TITAN V、12 GB 显存。论文“端到端”是这一检索计时口径，不代表生产请求的完整 wall-clock、p95 或包含网络与编码的 SLA。

## 消融与扩展性

![论文 Figure 6：累计优化收益](figs/fig06.png)

**Figure 6（PDF 第 8 页）。** 随机 500 个 MS MARCO v1 查询，k=1000；(a) GPU 累计 1.0→3.7→5.2→6.6×；(b) 8线程 CPU 累计 1.0→4.2→8.6→42.4×。最后阶段相对前一阶段的增益分别约 1.3×、4.9×。只换 CPU 内核而没有质心交互约 3×，说明不能只把收益归于 C++。该图的 42.4 与 Table 3 的 45 使用不同查询样本，不能强行视为不一致。

![论文 Figure 7：随 embedding 规模变化](figs/fig07.png)

**Figure 7（PDF 第 8 页）。** GPU 与 8线程 CPU 的双对数曲线，以 embedding 数量而非段落数为横轴。不同数据集长度与查询分布不同，作者关于平方根扩展的观察不是对全部生产负载的复杂度保证；GPU v2 的 k=1000 点因 OOM 缺失，不是漏下载。

![论文 Figure 8：CPU 线程扩展](figs/fig08.png)

**Figure 8（PDF 第 8 页）。** 额外线程降低延迟，但 k=1000 的 16 线程相对单线程约 4.9×，不是 16×。候选生成、负载不均与内存访问可能限制扩展；本文未通过自己的 profiling 确认作者提出的解释。

## 如何迁移到自己的索引

论文配置为：k=10 用 nprobe=1、t_cs=0.5、ndocs=256；k=100 用 2、0.45、1024；k=1000 用 4、0.4、4096。当前官方 main 的 `Searcher.dense_search` 使用 `ncells` 对应探测单元数量，k>100 时 ndocs 默认为 max(4k,4096)。这是一份已核对的实现观察，不等于所有版本的稳定接口承诺。

实操应先固定 checkpoint 与索引压缩参数，建立无质心剪枝的参考检索结果，再逐步扫描探测单元数、质心阈值、候选预算。每一步记录基线 top-k 覆盖、qrels 的 MRR/Recall、候选数、解压字节数、平均与尾部延迟。只测速度会掩盖候选丢失；只看最终 top-10 会掩盖深召回下降。

对 ColPali/ColQwen 的视觉页 token/patch，可以研究相同的“便宜代理筛选→昂贵多向量精排”思想，但本文只验证文本，不能宣称直接迁移有效。页 token 更多、重复视觉纹理和空间信息不同，需要检验质心分布、假负例、候选覆盖和显存，重新选择预算及阈值。≤0.6B 学生模型减小编码成本，也不会自动解决索引访存瓶颈。

## 本轮研究证据与复现边界

已核对 arXiv v1 的 10 页 PDF、HTML、全部 8 幅 Figure 和 6 张表；方法图与主结果图直接来自 HTML 原资产，15 个 SVG 转为 PNG，含全部子图和独立图例。模型语料评测未重跑，本环境没有 GPU，也未下载/建立亿级索引。

本轮仅做公式维度与小规模数值检查：MaxSim 的求和/最大值方向、质心 gather、重复质心、剪枝阈值、近似分数非上界，以及表中倍率算术。它们证明稿件里的公式与例子自洽，不能证明 PLAID 的 benchmark 性能。细节和执行结果见 [研究笔记](研究笔记.md) 与 [审核清单](审核清单.md)。

## 参考与版本

- 论文：[arXiv v1](https://arxiv.org/abs/2205.09707v1)，CC BY 4.0，原作四位作者见文首。
- 官方代码固定检查版本：[`cc4f3dc91c0b45d2d08c251d9d95178285c65f1c`](https://github.com/stanford-futuredata/ColBERT/tree/cc4f3dc91c0b45d2d08c251d9d95178285c65f1c)。当前实现与论文当年实现分开标注。
- 已将资料清单中的旧编号 `2205.09153` 修正为官方确认的 `2205.09707`，并补充专题索引。
