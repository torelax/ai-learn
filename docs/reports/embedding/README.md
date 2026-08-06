# Embedding 调研报告

学习/调研文档（自 vlm_train / modelforge 迁入）。**Markdown 为源文件**；仅保留《图文 Embedding 模型技术综述》的 HTML 供审阅对照。

论文 PDF 本地副本：`docs/papers/embedding/`（不入库）。Notion：Embedding&Rerank → Paper 库全文导入 + ModelCard mention。规范见根目录 [AGENTS.md](../../../AGENTS.md)。

## 总览与规划

| 文档 | 说明 |
|------|------|
| [Embedding调研报告.md](Embedding调研报告.md) | 总览 |
| [调研规划.md](调研规划.md) | 规划 |
| [资料清单_论文与博客.md](资料清单_论文与博客.md) | 论文/模型卡入口 |
| [图文Embedding模型技术综述.md](图文Embedding模型技术综述.md) | 图文四类路线（由 HTML 转写，待审阅） |
| [0.6B图搜图文搜图自训学习行动路线.md](0.6B图搜图文搜图自训学习行动路线.md) | ≤0.6B 图搜图/文搜图自训行动线 |

## 专题与系列深读

| 文档 | 说明 |
|------|------|
| [BGE-M3三功能统一详解报告.md](BGE-M3三功能统一详解报告.md) | BGE-M3 |
| [Embedding蒸馏技术详解.md](Embedding蒸馏技术详解.md) | Embedding 蒸馏专题 |
| [难负例挖掘工业实践.md](难负例挖掘工业实践.md) | 难负例工业闭环 |
| [ANCE详解.md](ANCE详解.md) | 全局 ANN 难负例 + 异步刷新 |
| [RocketQA详解.md](RocketQA详解.md) | CE 去噪 hard + 伪标；v2 动态 listwise |
| [NV-Retriever详解.md](NV-Retriever详解.md) | Positive-aware 挖负（MarginPos / PercPos） |
| [LLM-DA文本行人检索数据增强详解.md](LLM-DA文本行人检索数据增强详解.md) | LLM 改写 + TFF + BSS |
| [DeVE-QA稠密视频事件问答详解.md](DeVE-QA稠密视频事件问答详解.md) | 稠密视频事件 QA 数据与 DeVi |
| [Jasper-Token-Compression-600M详解.md](Jasper-Token-Compression-600M详解.md) | Jasper 600M：双教师蒸馏 + 弹性 Token 压缩 |

## Late Interaction 族（文本 → 视觉文档）

| 文档 | 论文 / 来源 |
|------|-------------|
| [ColBERT详解.md](ColBERT详解.md) | [arXiv:2004.12832](https://arxiv.org/abs/2004.12832) |
| [ColBERTv2详解.md](ColBERTv2详解.md) | [arXiv:2112.01488](https://arxiv.org/abs/2112.01488) |
| [ColPali详解.md](ColPali详解.md) | [arXiv:2407.01449](https://arxiv.org/abs/2407.01449) |
| [ColQwen系列详解.md](ColQwen系列详解.md) | 同 ColPali 论文 + Vidore 模型卡（ColQwen2 / 2.5；社区 ColQwen3） |

## 稠密文本 Embedding 深读

| 文档 | 论文 / 来源 |
|------|-------------|
| [E5详解.md](E5详解.md) | [arXiv:2212.03533](https://arxiv.org/abs/2212.03533) |
| [gte-Qwen2详解.md](gte-Qwen2详解.md) | 方法 [arXiv:2308.03281](https://arxiv.org/abs/2308.03281) + [HF gte-Qwen2-7B](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct) |
| [Nomic-Embed详解.md](Nomic-Embed详解.md) | [arXiv:2402.01613](https://arxiv.org/abs/2402.01613) |
| [LLM2Vec详解.md](LLM2Vec详解.md) | [arXiv:2404.05961](https://arxiv.org/abs/2404.05961) |
| [Conan-embedding详解.md](Conan-embedding详解.md) | [arXiv:2408.15710](https://arxiv.org/abs/2408.15710)（v1：DHNM + Cross-GPU CBB；CMTEB） |
| [Conan-embedding-v2详解.md](Conan-embedding-v2详解.md) | [arXiv:2509.12892](https://arxiv.org/abs/2509.12892)（EMNLP 2025；从零 1.4B LLM） |
| [QZhou-Embedding详解.md](QZhou-Embedding详解.md) | [arXiv:2508.21632](https://arxiv.org/abs/2508.21632)（双向 Qwen2.5-7B + 多任务/合成/两阶段） |
| [Token-Prepending详解.md](Token-Prepending详解.md) | [ACL 2025](https://aclanthology.org/2025.acl-long.159/)（免训练层间回灌；因果注意力补丁） |

## Jina 系列

| 文档 | 论文 / 来源 |
|------|-------------|
| [Jina系列总览.md](Jina系列总览.md) | 谱系索引（v1→v5-omni；含架构类属与选型） |
| [Jina-embeddings-v2详解.md](Jina-embeddings-v2详解.md) | [arXiv:2310.19923](https://arxiv.org/abs/2310.19923)（v1 见 [2307.11224](https://arxiv.org/abs/2307.11224)） |
| [Jina-embeddings-v3详解.md](Jina-embeddings-v3详解.md) | [arXiv:2409.10173](https://arxiv.org/abs/2409.10173) |
| [jina-clip系列详解.md](jina-clip系列详解.md) | [arXiv:2405.20204](https://arxiv.org/abs/2405.20204) / [2412.08802](https://arxiv.org/abs/2412.08802) |
| [Jina-embeddings-v4详解.md](Jina-embeddings-v4详解.md) | [arXiv:2506.18902](https://arxiv.org/abs/2506.18902) |
| [Jina-embeddings-v5-text详解.md](Jina-embeddings-v5-text详解.md) | [arXiv:2602.15547](https://arxiv.org/abs/2602.15547) |
| [Jina-embeddings-v5-omni详解.md](Jina-embeddings-v5-omni详解.md) | [arXiv:2605.08384](https://arxiv.org/abs/2605.08384) |

## 关联报告

| 文档 | 说明 |
|------|------|
| [../lora/README.md](../lora/README.md) | LoRA / PEFT（Jina v3 任务 LoRA、cloud_emb stage2 等） |
| [../distillation/知识蒸馏技术深度调研报告.md](../distillation/知识蒸馏技术深度调研报告.md) | 蒸馏总论 |

工程代码与训练数据在 **modelforge**：`scripts/embedding/`、`datasets/embedding/`、`tasks/cloud_emb/`。
