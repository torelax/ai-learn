# LLM 微调 / PEFT 阅读笔记

学习/调研文档，与 [distillation](../distillation)、[embedding](../embedding) 报告体例对齐。**Markdown 为源文件**。

> **更新**：2026-10-19 · LoRA / QLoRA / QA-LoRA 深读与提示微调专题

---

## 总览

| 文档 | 说明 |
|------|------|
| [LoRA技术深度调研报告.md](LoRA技术深度调研报告.md) | **技术总章**：rank / alpha / 学习率、QKV/O 与 MLP 目标层、量化和部署；对应 Fine-Tune 原页 LoRA 部分 |
| [LLM提示微调综述_Prefix-Prompt-P-Tuning.md](LLM提示微调综述_Prefix-Prompt-P-Tuning.md) | Prefix、Prompt、P-Tuning v1/v2：方法、原论文结果、博客补充与选型 |
| [资料清单_论文与博客.md](资料清单_论文与博客.md) | 论文 / 官方文档 / 博客链接表（含 arXiv ID） |

---

## 逐篇论文中文译解与研读

| 文档 | 论文 / 来源 |
|------|-------------|
| [LoRA详解.md](LoRA/LoRA详解.md) | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — LoRA 原文深读 |
| [QLoRA详解.md](QLoRA/QLoRA详解.md) | [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) — NF4/DQ/Paged Optimizers；单卡 65B |
| [QA-LoRA详解.md](QA-LoRA/QA-LoRA详解.md) | [arXiv:2309.14717](https://arxiv.org/abs/2309.14717) — 分组适配与量化偏移合并，ICLR 2024 |
| [rsLoRA详解.md](rsLoRA/rsLoRA详解.md) | [arXiv:2312.03732](https://arxiv.org/abs/2312.03732) — 缩放 α/√r，解锁高 rank |
| [Adapter与PEFT范式详解.md](Adapter/Adapter与PEFT范式详解.md) | Adapter / Prefix / Prompt 等 PEFT 范式 |
| [AdaLoRA详解.md](AdaLoRA/AdaLoRA详解.md) | [arXiv:2303.10512](https://arxiv.org/abs/2303.10512) — 自适应 rank 预算 |
| [VeRA详解.md](VeRA/VeRA详解.md) | [arXiv:2310.11454](https://arxiv.org/abs/2310.11454) — 共享随机基 + 向量缩放；multi-tenant |
| [LoRA+详解.md](LoRA+/LoRA+详解.md) | [arXiv:2402.12354](https://arxiv.org/abs/2402.12354) — A/B 异学习率，λ≈16 |
| [DoRA详解.md](DoRA/DoRA详解.md) | [arXiv:2402.09353](https://arxiv.org/abs/2402.09353) — 幅度-方向分解 |

---

## 关联报告

| 文档 | 说明 |
|------|------|
| [../distillation/知识蒸馏技术深度调研报告.md](../distillation/知识蒸馏技术深度调研报告.md) | 蒸馏总论（LoRA 训 student 等） |
| [../embedding/Embedding调研报告.md](../embedding/Embedding调研报告.md) | Embedding 总览（LLM+LoRA 路线） |

---

## 工程代码（本机）

| 仓库 | 路径 | 用途 |
|------|------|------|
| **vlm_train** | `scripts/platform/train_cloud_*.sh`、`scripts/local/train_cloud_*.sh` | ms-swift LoRA 训练入口 |
| **modelforge** | `tasks/cloud_emb/`、`scripts/embedding/` | cloud embedding LoRA stage2 与评测 |
| **ms-swift** | 训练 CLI | `--tuner_type lora` 等 |

---

## 阅读顺序建议

1. [LoRA 技术总章](LoRA技术深度调研报告.md)：跨论文原理与超参、权重矩阵选择。
2. [LoRA](LoRA/LoRA详解.md)：低秩增量、初始化、位置/秩消融与合并。
3. [QLoRA](QLoRA/QLoRA详解.md)：NF4、双重量化、分页与量化基座训练。
4. [QA-LoRA](QA-LoRA/QA-LoRA详解.md)：训练表示与最终低位合并如何对齐。
5. [提示微调综述](LLM提示微调综述_Prefix-Prompt-P-Tuning.md)：输入提示、逐层前缀、v1/v2 与现代实践。
6. 按需读 rsLoRA、LoRA+、VeRA、AdaLoRA、DoRA，以及总报告中的工程生态。

提示微调四篇原论文的 PDF、原图和来源按方法放在 `Prefix-Tuning/`、`Prompt-Tuning/`、`P-Tuning/`、`P-Tuning-v2/`；跨论文正文保留在本专题根目录。此处属于 LLM 微调专题。

## Notion 入口

- [Fine-Tune](https://www.notion.so/235a174cd8c68016b3d6c922d1a1d70e)：Ref 的 Paper 视图呈现微调论文，原页 LoRA 部分是技术总章。
- [LoRA 论文](https://www.notion.so/3fea174cd8c6816fbfe2c7948d64a538)、[QLoRA 论文](https://www.notion.so/3fea174cd8c681a89ecaed958b31fe40)、[QA-LoRA 论文](https://www.notion.so/3fea174cd8c681609b63da63aef9bfd2)：归入全局 Paper，保留原页面 URL。

三篇具名详解包含摘要译文、主文译述和解释性展开，并保留实验与附录要点；它们是各自的权威研读源。若另作逐句全文翻译，使用独立 `zh.md` 并明确覆盖范围。
