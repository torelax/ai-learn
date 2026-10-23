# RLHF、偏好优化与可验证奖励

这一专题按后训练技术组织。**DPO／GRPO 是优化方法，RLHF／RLAIF／RLVR 区分监督与奖励来源**；数学、代码的规则奖励不应笼统称为人类反馈。

| 阅读目标 | 文档 |
|---|---|
| 原论文的推导、任务、实验和消融 | [DPO 论文精炼](DPO/DPO详解.md) · [补充解读](DPO/DPO解读.md) · [全文中英对照](DPO/zh.md) |
| 实践中的参考模型、β、四个 log-prob、长度与偏好数据 | [DPO 精讲](DPO精讲.md) |
| 同题采样、优势、old／reference policy、裁剪、KL 与后续变体 | [GRPO 精讲](GRPO精讲.md) |
| GRPO 的原始提出与数学实验 | [DeepSeekMath](../DeepSeek/DeepSeekMath/DeepSeekMath详解.md) |
| 大规模推理 RL 与冷启动、拒绝采样的衔接 | [DeepSeek-R1](../DeepSeek/DeepSeek-R1/DeepSeek-R1详解.md) |
| 生成式奖励、rubrics 与可扩展评判 | [DeepSeek-GRM](../DeepSeek/DeepSeek-GRM/DeepSeek-GRM详解.md) |
| 自验证数学推理与形式证明的不同反馈来源 | [Math-V2](../DeepSeek/DeepSeekMath-V2/DeepSeekMath-V2详解.md) · [Prover-V2](../DeepSeek/DeepSeek-Prover-V2/DeepSeek-Prover-V2详解.md) |

GRPO 没有另外虚构一篇同名原始论文；它的原始来源是 DeepSeekMath。技术讲解另外参考 Hugging Face TRL 文档、RLHF Book，以及 Dr. GRPO／DAPO 等后续独立研究，并在正文标明来源。

Notion 精讲位于 LLM 的 PostTrain → RLHF；DPO 原论文位于 Paper；保留既有个人内容，首次在后方补充研读，后续原位更新已导入区域。
