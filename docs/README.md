# docs

学习与调研文档。阅读笔记风格与单页同步流程见 [论文阅读笔记指南](paper-reading-guide.md)。论文研究流程（PDF 副本、Notion 导入、禁删）见仓库根目录 [AGENTS.md](../AGENTS.md)。

| 路径 | 说明 |
|------|------|
| [`reports/embedding/`](reports/embedding/) | Embedding 调研报告 + 论文 PDF / 中译 / 配图（自 vlm_train / modelforge 迁入） |
| [`reports/lora/`](reports/lora/) | LLM 微调 / PEFT：LoRA、量化适配与提示微调 |
| [`reports/long-context/`](reports/long-context/) | 长上下文与位置编码：YaRN |
| [`reports/distillation/`](reports/distillation/) | 知识蒸馏调研 |
| [`reports/Qwen/`](reports/Qwen/) | Qwen 系列技术演进 |
| [`reports/LLM/DeepSeek/`](reports/LLM/DeepSeek/README.md) | DeepSeek 模型、架构、多模态、数学及训练推理系统论文 |
| [`reports/LLM/RLHF/`](reports/LLM/RLHF/README.md) | DPO 论文与 DPO／GRPO 技术精讲 |

工程实现与训练数据在同级 **modelforge**（`scripts/embedding/`、`datasets/embedding/`、`tasks/cloud_emb/`）。
