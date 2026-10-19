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
| [Prefix-Tuning详解.md](Prefix-Tuning/Prefix-Tuning详解.md) | [2101.00190](https://arxiv.org/abs/2101.00190) — 逐层连续前缀与生成 |
| [Prompt-Tuning详解.md](Prompt-Tuning/Prompt-Tuning详解.md) | [2104.08691](https://arxiv.org/abs/2104.08691) — 输入层软提示与模型规模 |
| [P-Tuning详解.md](P-Tuning/P-Tuning详解.md) | [2103.10385](https://arxiv.org/abs/2103.10385) — 连续模板与提示编码器 |
| [P-Tuning-v2详解.md](P-Tuning-v2/P-Tuning-v2详解.md) | [2110.07602](https://arxiv.org/abs/2110.07602) — 深层提示、NLU 与序列标注 |
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
- [Prefix-Tuning 论文](https://www.notion.so/3fea174cd8c6800fa2b2f76fcaa88044)、[Prompt Tuning 论文](https://www.notion.so/3fea174cd8c681328ef3f99d1d7636bd)、[P-Tuning 论文](https://www.notion.so/3fea174cd8c681c4b890d2be303e33d5)、[P-Tuning v2 论文](https://www.notion.so/3fea174cd8c681ada540c2a55729d81d)：各篇均有独立阅读笔记及英文/中文全文附件。

LoRA、QLoRA、QA-LoRA、Prefix-Tuning、Prompt Tuning、P-Tuning v1/v2 均按同一结构组织：

- `en.md`：完整英文转写，保留主文、附录、图表、脚注和参考文献。
- `zh.md`：对应中文全译稿，注明机器初译与实际术语/关键方法校订范围；参考文献保留书目信息，原图内部文字保留英文。
- `<方法名>详解.md`：独立论文阅读笔记，含一句话简述、短简介、方法推导、实验和局限。
- 论文全名 PDF 与 `figs/`：七篇 PDF 均不足 2 MB，保留阅读所需原资产。

临时采集、核查和同步记录放在仓库外工作目录，不提交审核清单、选题笔记或同步日志。
