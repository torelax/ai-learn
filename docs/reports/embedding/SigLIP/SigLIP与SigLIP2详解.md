# SigLIP 与 SigLIP 2：从成对 sigmoid 损失到更完整的视觉语言表示

> SigLIP 改变图文对比目标的归一化方式；SigLIP 2 在此基础上进一步改进数据、多语言、局部视觉表示和分辨率适配。
>
> [SigLIP](https://arxiv.org/abs/2303.15343) · [SigLIP 2](https://arxiv.org/abs/2502.14786)

完整研读分别见 [SigLIP 精炼版](SigLIP/SigLIP详解.md) 与 [SigLIP 2 精炼版](SigLIP2/SigLIP2详解.md)。本页保留两代方法的连接，避免将两篇的训练设置混成一套。

## SigLIP 改变的是训练损失

CLIP 的双向 softmax 在一个批次内为每条图像或文字归一化候选分数。SigLIP 把每个图文对当成独立二分类项：

$$
\mathcal L=-\frac1B\sum_{i,j}\log\sigma\bigl(y_{ij}(t x_i^\top z_j+b)\bigr),\qquad y_{ij}\in\{-1,+1\}.
$$

温度缩放与偏置分别调整分数尺度及正负不平衡。分母是批量大小 B，不能机械改成 B² 后仍认为梯度尺度相同。独立成对项便于分块计算，但一个大批次仍有大量负对，不能理解为只比较同位置正例。

## SigLIP 2 扩展的是训练配方

SigLIP 2 保留 sigmoid 对齐，加入多语言数据、去偏、在线数据筛选、LocCa 式生成监督及局部自蒸馏/遮挡预测。它希望视觉编码器既适合整图匹配，也更适合作为视觉语言模型骨干及密集任务表示。

NaFlex 进一步支持保持宽高比和可变分辨率；其具体训练组合与固定分辨率版本不同，不能把所有辅助目标都写成每个 checkpoint 同时使用。详细阶段、训练数据与消融见单篇笔记。

## 阅读实验时区分三件事

零样本分类测试图文对齐；线性探测测试冻结表示中有多少可用信息；接入 VLM 后的任务成绩还受语言模型和适配训练影响。图文检索也要区分图搜文与文搜图以及 Recall@K。两代模型跨参数量、数据和分辨率的总分差异不能全部归功于 sigmoid 损失。

进一步解释见 [SigLIP 解读](SigLIP/SigLIP解读.md) 与 [SigLIP 2 解读](SigLIP2/SigLIP2解读.md)。
