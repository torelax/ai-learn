# LoRA / PEFT 调研报告

学习/调研文档，与 [distillation](../distillation)、[embedding](../embedding) 报告体例对齐。**Markdown 为源文件**。

> **版本**: v1.1 · **日期**: 2026-07-30

---

## 总览

| 文档 | 说明 |
|------|------|
| [LoRA技术深度调研报告.md](LoRA技术深度调研报告.md) | **总报告**：PEFT 全景、LoRA 谱系、量化/优化/表达力、工程生态、选型决策树 |
| [资料清单_论文与博客.md](资料清单_论文与博客.md) | 论文 / 官方文档 / 博客链接表（含 arXiv ID） |

---

## 专题深读

| 文档 | 论文 / 来源 |
|------|-------------|
| [LoRA详解.md](LoRA/LoRA详解.md) | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — LoRA 原文深读 |
| [QLoRA详解.md](QLoRA/QLoRA详解.md) | [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) — NF4/DQ/Paged Optimizers；单卡 65B |
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

```text
1. LoRA技术深度调研报告.md §1–6     → 建立全景
2. LoRA详解.md                      → 原文逐章（α/r、Q/V、GPT-3 资源账）
3. QLoRA详解.md + rsLoRA详解.md     → 显存三件套 + 高 rank 缩放修正
4. Adapter与PEFT范式详解.md          → 三大家族背景（可选）
5. LoRA+详解.md + VeRA详解.md        → 两个高性价比变体
6. AdaLoRA详解.md / DoRA详解.md      → 预算分配 / 表达力增强
7. 总报告 §7–11                    → 量化、工程、选型
8. 资料清单_论文与博客.md           → 按需深读原文
```
