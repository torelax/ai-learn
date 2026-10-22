# GradCache：显存受限时的真正大批次对比学习

> 先缓存整个大 batch 的表示梯度，再逐小 batch 重算编码器并反传，把跨样本对比损失与编码器激活内存解耦。
>
> Luyu Gao、Yunyi Zhang、Jiawei Han、Jamie Callan · RepL4NLP 2021 · [论文](https://aclanthology.org/2021.repl4nlp-1.31/) · [arXiv v2](https://arxiv.org/abs/2101.06983v2) · [官方代码](https://github.com/luyug/GradCache)

## 论文简介

对比学习常用同 batch 的其他文档充当负例，因此更大的 batch 同时意味着更丰富的竞争集合。但普通反向传播需要保留所有样本的编码器激活，单张显卡容易受限；简单梯度累积虽然增加一次更新所见样本数，却不能让不同小 batch 的样本互为负例。GradCache 通过链式法则，将损失对表示的梯度先算出来，再用这些缓存梯度逐块更新编码器。

论文在单张 RTX 2080 Ti 上复现 DPR 的大批次训练效果，并以约 20% 的额外时间成本扩大可训练 batch。它解决的是编码器激活内存，而非所有内存和计算开销：表示缓存、相似度矩阵及多卡通信仍随规模增长，精确等价也依赖重算时模型状态一致。

## 背景：为什么梯度累积不够

设查询向量 $f_\theta(q_i)$、文档向量 $g_\phi(d_j)$，正例索引为 $r_i$：

$$
\mathcal L=-\frac1B\sum_i\log
\frac{\exp(f_\theta(q_i)^\top g_\phi(d_{r_i})/\tau)}
{\sum_j\exp(f_\theta(q_i)^\top g_\phi(d_j)/\tau)}.
$$

每个查询的分母依赖整个文档集合。将 batch 分成独立小块并分别计算损失，改变了 softmax 分母，也改变了梯度；累积参数梯度不能恢复没参与比较的负例。

已有 gradient checkpointing 沿网络深度重算激活，ZeRO 等方法减少优化器状态冗余。本文沿 **batch 维度**切分编码器计算，因此与这些方法处理的是不同内存来源。

## 链式法则如何切开计算图

记 $u_i=f_\theta(q_i)$、$v_j=g_\phi(d_j)$，以及表示梯度

$$
a_i=\frac{\partial\mathcal L}{\partial u_i},\qquad
b_j=\frac{\partial\mathcal L}{\partial v_j}.
$$

则

$$
\frac{\partial\mathcal L}{\partial\theta}
=\sum_i a_i^\top\frac{\partial f_\theta(q_i)}{\partial\theta},\qquad
\frac{\partial\mathcal L}{\partial\phi}
=\sum_j b_j^\top\frac{\partial g_\phi(d_j)}{\partial\phi}.
$$

$a_i,b_j$ 依赖整个 batch，但计算它们只需要向量及相似度，不必保留编码器所有层。得到它们后，每个编码器 Jacobian 只依赖该样本和当前参数，可以独立重算。数学上这只是链式法则的重新安排，没有用近似负例或旧参数表示替代当前 batch。

## 四步算法

1. **无计算图前向。** 把大 batch 切成能放进显存的小 batch，逐块编码，只保存最终表示。
2. **表示层反向。** 将所有表示作为叶子张量，构造完整 batch 的对比损失并反传，缓存每个表示的梯度。
3. **编码器重算与反向。** 逐块重新运行编码器，恢复该小块计算图，将缓存的上游梯度传入，累积参数梯度；处理后即可释放该小块激活。
4. **统一更新参数。** 全部小块完成后执行一次 optimizer step。

编码器峰值激活由小 batch 控制。缓存规模约为 $(|S|+|T|)d$ 个浮点数，远小于为整个 batch 保存每层激活，但仍随 batch 线性增长。若直接构造所有 query–document 分数，还会有二次增长的相似度矩阵；“近乎恒定显存”主要针对论文原先的编码器内存瓶颈。

## 多 GPU 与一般打分函数

多卡版本在无图前向后 all-gather 表示，让每卡可以使用全局负例；各卡只缓存本地表示需要的梯度，再独立重算本地编码器，最后按普通数据并行规约参数梯度。通信没有消失，而是从大批次的完整计算图里分离出来。

论文还将方法推广到参数化距离函数 $\Phi_\Omega(u_i,v_j)$：先缓存损失对距离的梯度，再分块计算距离网络参数与表示梯度，最后回到编码器缓存流程。这个扩展增加一层缓存，说明方法不只适用于固定内积；其计算与存储成本仍由具体交互函数决定。

## 实验设置

以 **DPR** 稠密段落检索器为载体，在 **Natural Questions**上训练和评估。NQ 查询来自真实搜索问题，文档来自 Wikipedia；Top-k hit accuracy 表示前 k 个候选中是否找到包含答案的段落，并非 MRR 或逐篇相关性精确率。

单张 RTX 2080 Ti 上，每个问题配一篇正文档与一个 BM25 负例。Cache 的有效 batch 为 128，query 小 batch 为 16，passage 小 batch 为 8；另测试有效 batch 512。Sequential 每 8 条更新一次；Accumulation 累积 16 个大小 8 的小 batch，但各小 batch 单独计算对比损失。DPR 原论文结果来自 8 GPU 配置。

## 检索质量与速度

| 方法 | Top-5 | Top-20 | Top-100 |
|---|---:|---:|---:|
| DPR 原论文参照 | — | 78.4 | 85.4 |
| Sequential，batch 8 | 59.3 | 71.9 | 80.9 |
| 普通梯度累积 | 64.3 | 77.2 | 84.9 |
| GradCache，batch 128 | 68.6 | 79.3 | 86.0 |
| GradCache，batch 512 | 68.3 | 79.9 | 86.6 |

缓存方法接近或略高于 DPR 参照；作者将小幅超过原论文归于训练随机性，而不是声称同样梯度必然更准。512 相比 128 也不是所有 k 都提升。普通累积明显弱于缓存，验证真正扩大负例集合的重要性。

![原论文图 1：一次更新的时间随样本数变化](figs/S4-F1-01.png)

横轴为每次参数更新的样本数，纵轴为更新时间。Cache 相比 accumulation 多一次表示预计算，时间约增加 20%，但两者计算的是不同对比目标。论文还报告优化数据加载后可在单张 2080 Ti 上约 31 小时训练检索器；这属于其具体实现和硬件设定。

## 结论与复现边界

GradCache 的贡献是把大批次对比训练的硬件门槛降下来，同时保持目标及梯度形式不变。它不会替代数据清洗、难负例设计或蒸馏，也不是“batch 越大一定越好”的证据。

要维持等价性，重算阶段必须使用同一参数和对应随机状态；dropout 若不同，缓存梯度与重算表示就不再完全匹配。带跨样本统计的操作、损失归一化和多卡梯度缩放也需要谨慎实现。后续库中的 CachedMultipleNegativesRankingLoss 等可承接这一思想，但具体 API 与额外优化应按库文档理解，不倒写成 2021 年论文内容。
