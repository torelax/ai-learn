# ai-learn 知识库

Embedding、大模型与后训练方向的论文研读和技术调研笔记。**Markdown 是唯一源文件**，站点由 `scripts/site/site.sh build` 生成到仓库根目录的 `site/`，不入库。

## 领域

| 领域 | 专题 | 内容 |
|------|------|------|
| [Embedding](embedding/README.md) | 文本、多模态与检索表示 | 总览与机制专题（损失、难负例、蒸馏、领域适配），以及 DPR、ColBERT、BGE、GTE、E5、CLIP、SigLIP、Qwen3-Embedding 等七十余篇论文的精炼版、补充解读和中英对照 |
| [LLM](llm/README.md) | [DeepSeek](llm/DeepSeek/README.md) | DeepSeek 官方模型报告与相关基础研究：LLM、MoE、OCR、多模态、数学证明、训练推理系统 |
| | [Qwen](llm/Qwen/README.md) | Qwen 系列技术演进 |
| | [长上下文](llm/long-context/README.md) | 位置编码与长度外推：YaRN |
| [后训练](post-training/README.md) | [PEFT 微调](post-training/peft/README.md) | LoRA 及其变体、量化适配（QLoRA、QA-LoRA）、提示微调（Prefix / Prompt / P-Tuning） |
| | [RLHF](post-training/RLHF/README.md) | DPO 原论文，DPO / GRPO 技术精讲 |
| | [知识蒸馏](post-training/distillation/README.md) | LLM / VLM / Embedding 蒸馏总论，OPD 技术分享 |

## 文档类型

一篇论文一个文件夹，常见文件：

| 文件 | 内容 |
|------|------|
| `<方法名>详解.md` | 论文精炼版：保留原文章节与论证顺序，压缩段落，保留核心公式、原图和实验表 |
| `<方法名>解读.md` | 补充解读：跨论文比较、推导和工程取舍 |
| `zh.md` | 全文中英逐段对照（英文段落后紧接中文） |
| `figs/` | 论文原图，优先取自 arXiv HTML / ar5iv |

论文 PDF 只保存在仓库中，不发布到站点；阅读原文请用各篇文首的 arXiv 链接。
