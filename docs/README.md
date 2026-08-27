# docs

学习与调研文档。论文研究流程（PDF 副本、Notion 导入、禁删）见仓库根目录 [AGENTS.md](../AGENTS.md)。

| 路径 | 说明 |
|------|------|
| [`reports/embedding/`](reports/embedding/) | Embedding 调研报告（自 vlm_train / modelforge 迁入） |
| [`reports/lora/`](reports/lora/) | LoRA / PEFT 调研报告 |
| [`reports/distillation/`](reports/distillation/) | 知识蒸馏调研 |
| [`reports/Qwen/`](reports/Qwen/) | Qwen 系列技术演进 |
| `papers/<topic>/` | 论文 PDF 本地副本（**不入库**；PDF 用论文全名 + arXiv 编号，文件夹见 `papers/embedding/GTE/` 等） |

工程实现与训练数据在同级 **modelforge**（`scripts/embedding/`、`datasets/embedding/`、`tasks/cloud_emb/`）。
