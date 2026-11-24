# MLLM 通用多模态嵌入：GME、VLM2Vec 与 MegaPairs

> 三条路线分别强调模态组合覆盖、多任务指令训练，以及可扩展的关系数据合成；共同目标是把视觉语言理解转为可独立编码的检索表示。
>
> [GME](https://arxiv.org/abs/2412.16855) · [VLM2Vec](https://arxiv.org/abs/2410.05160) · [MegaPairs](https://arxiv.org/abs/2412.14475)

本页保留跨论文比较。完整方法、数据、实验和消融分别见 [GME 精炼版](GME/GME详解.md)、[VLM2Vec 精炼版](VLM2Vec/VLM2Vec详解.md)、[MegaPairs 精炼版](MegaPairs/MegaPairs详解.md)；工程分析放在各自解读中。

## 三篇论文回答不同问题

| 维度 | GME | VLM2Vec | MegaPairs / MMRet |
| --- | --- | --- | --- |
| 核心问题 | 一个模型如何覆盖单模态、跨模态和融合检索 | VLM 如何按任务指令产生可迁移表示 | 如何规模化获得有关系、非重复的图像对和检索指令 |
| 主要骨干 | Qwen2-VL-2B/7B | Phi-3.5-Vision、LLaVA-1.6 | CLIP Base/Large、LLaVA-1.6-Mistral-7B |
| 数据贡献 | 多模态混合及约 1.1M 融合样本合成 | MMEB 36 集，其中 20 个训练内、16 个训练外 | 26,235,105 图像对，每对多条合成指令 |
| 关键技术 | 指令、LoRA、难负例、隐去实体名的融合问题 | 指令化对比训练、GradCache、跨任务迁移 | CLIP 图像/DINOv2/描述三个空间挖掘，再生成关系指令 |
| 主评测 | UMRB，47 个数据集 | MMEB，36 个数据集 | CIRCO/CIRR/FashionIQ/GeneCIS，以及 MMEB |

GME 正文称训练约 8M，分项规模未与总数完全对应，不能把 1.1M 合成量再次叠加到融合数据总量中。VLM2Vec 的 62.9 属于 v3 高分辨率 LLaVA 配置；Phi LoRA 版本为 60.1，低分辨率 LLaVA 为 55.0。MegaPairs 的命名与后续发布的 BGE-VL checkpoint 应依据模型卡逐一对应，不能用一个产品名替代论文中全部模型设置。

## 统一向量并未统一所有相关性

三篇都支持把输入编码成向量再比较，但“图像主题相似”“依据指令修改参考图”“找到能回答问题的页面”是不同目标。任务指令需要训练支持；把长指令直接拼给 CLIP 不一定有益。候选模板、模态组合和训练数据覆盖共同决定这个空间能服务哪些任务。

GME 重点补齐融合模态，VLM2Vec 把分类、问答和定位也转成候选匹配，MegaPairs 增加指令所表达的图像关系。这些设计可相互借鉴，但不构成一套已经验证过的可直接叠加配方。

## 实验分数如何比较

UMRB 混合 nDCG@10、nDCG@5 和不同 K 的 Recall；MMEB 主要采用指定候选集合上的 Precision@1。两者总分没有统一量纲。MMEB 的视觉问答是答案候选排序，区域定位是预裁剪区域排序，不能当作开放式生成和检测成绩。

MegaPairs 在 MMEB 微调后总体 64.1。正文的 9.1 分提升对应低分辨率 LLaVA 的 55.0，而非对 VLM2Vec 全系列的统一领先；不同版本还存在骨干和输入预算差异。对业务选择而言，应固定候选库、查询模板、图像预算和指标，再比较召回、编码吞吐与索引存储。

## 与多向量及双塔模型的关系

单向量容易接常规 ANN，但可能压缩局部证据；ColPali/ColQwen 保留多个局部表示，代价是更大索引和 MaxSim 计算。CLIP/SigLIP 双塔通常编码较轻，适合大量自然图像匹配；联合 VLM 适合需要图文交互才能理解的输入。选择取决于目标相关性和系统预算，不能仅根据参数量或是否使用 LLM 排序。

进一步阅读：[GME 解读](GME/GME解读.md) · [VLM2Vec 解读](VLM2Vec/VLM2Vec解读.md) · [MegaPairs 解读](MegaPairs/MegaPairs解读.md) · [ColQwen](ColQwen/ColQwen系列详解.md)。
