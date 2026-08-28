# Embedding 调研报告

学习/调研文档（自 vlm_train / modelforge 迁入）。**Markdown 为源文件**；仅保留《图文 Embedding 模型技术综述》的 HTML 供审阅对照。

单篇/系列精读与论文 PDF、中译、`figs/` 放在同一方法文件夹（如 `GTE/GTE系列详解.md`、`InternLM2/InternLM2数据处理与过滤详解.md`）。总览、合写、索引仍在本目录。报告嵌入图在 `figures/<短名>/`。

Notion：Embedding&Rerank → Paper 库全文导入 + ModelCard mention。规范见根目录 [AGENTS.md](../../../AGENTS.md)。

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
| [BGE-M3三功能统一详解报告.md](BGE/M3/BGE-M3三功能统一详解报告.md) | BGE-M3 |
| [BGE-CPack详解.md](BGE/C-Pack/BGE-CPack详解.md) | [arXiv:2309.07597](https://arxiv.org/abs/2309.07597)（C-Pack / BGE 全家桶起点：C-MTEB + C-MTP + BGE 模型 + 三阶段训练） |
| [BGE-EN-ICL详解.md](BGE/EN-ICL/BGE-EN-ICL详解.md) | [arXiv:2409.15700](https://arxiv.org/abs/2409.15700)（Mistral-7B + causal + [EOS] + few-shot ICL 训练；MTEB 71.24 / 71.67） |
| [BGE-multilingual-gemma2详解.md](BGE/BGE-multilingual-gemma2详解.md) | Gemma-2-9B 骨干（附录 C of BGE-EN-ICL 论文）；MIRACL 74.1 SOTA / FR-MTEB / PL-MTEB SOTA |
| [BGE-Reranker详解.md](BGE/BGE-Reranker详解.md) | v2-m3 / v2-gemma / v2-minicpm-layerwise / **v2.5-gemma2-lightweight**（附录 D of BGE-EN-ICL 论文；深度+宽度双压缩） |
| [Embedding蒸馏技术详解.md](Embedding蒸馏技术详解.md) | Embedding 蒸馏专题 |
| [难负例挖掘工业实践.md](难负例挖掘工业实践.md) | 难负例工业闭环 |
| [对比学习与InfoNCE精讲.md](对比学习与InfoNCE精讲.md) | 损失演化专题：Metric learning → InfoNCE → SimCSE/DPR/CLIP/SigLIP/蒸馏/BGE-M3/GritLM |
| [InternLM2数据处理与过滤详解.md](InternLM2/InternLM2数据处理与过滤详解.md) | [arXiv:2403.17297](https://arxiv.org/abs/2403.17297)：LLM 预训练数据清洗流水线（Conan-v1 预训练引用的数据过滤方法） |
| [ANCE详解.md](ANCE/ANCE详解.md) | 全局 ANN 难负例 + 异步刷新 |
| [RocketQA详解.md](RocketQA/RocketQA详解.md) | CE 去噪 hard + 伪标；v2 动态 listwise |
| [NV-Retriever详解.md](NV-Retriever/NV-Retriever详解.md) | Positive-aware 挖负（MarginPos / PercPos） |
| [LLM-DA文本行人检索数据增强详解.md](LLM-DA-TPR/LLM-DA文本行人检索数据增强详解.md) | LLM 改写 + TFF + BSS |
| [DeVE-QA稠密视频事件问答详解.md](DeVE-QA/DeVE-QA稠密视频事件问答详解.md) | 稠密视频事件 QA 数据与 DeVi |
| [Jasper-Token-Compression-600M详解.md](Jasper/Jasper-Token-Compression-600M详解.md) | Jasper 600M：双教师蒸馏 + 弹性 Token 压缩 |

## 基石短深读（2026 补齐）

| 文档 | 论文 / 来源 |
|------|-------------|
| [无监督对比检索三部曲_SimCSE-Contriever-Condenser.md](无监督对比检索三部曲_SimCSE-Contriever-Condenser.md) | SimCSE [2104.08821](https://arxiv.org/abs/2104.08821) + Contriever [2112.09118](https://arxiv.org/abs/2112.09118) + Condenser [2104.08253](https://arxiv.org/abs/2104.08253) + coCondenser [2108.05540](https://arxiv.org/abs/2108.05540) |
| [RetroMAE与DupMAE详解.md](RetroMAE/RetroMAE与DupMAE详解.md) | RetroMAE [2205.12035](https://arxiv.org/abs/2205.12035) + DupMAE / RetroMAE v2 [2211.08769](https://arxiv.org/abs/2211.08769) |
| [INSTRUCTOR详解.md](INSTRUCTOR/INSTRUCTOR详解.md) | [arXiv:2212.09741](https://arxiv.org/abs/2212.09741)（指令化嵌入开山；GTR + 拼接指令 + MEDI 330 任务） |
| [CLIP详解.md](CLIP/CLIP详解.md) | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)（图文双塔基石；对称 InfoNCE + WIT 4 亿对 + zero-shot 分类） |
| [SigLIP与SigLIP2详解.md](SigLIP/SigLIP与SigLIP2详解.md) | SigLIP [2303.15343](https://arxiv.org/abs/2303.15343) + SigLIP 2 [2502.14786](https://arxiv.org/abs/2502.14786)（sigmoid loss + 100 语 + LocCa + SILC/TIPS + NaFlex） |

## 合写型专题（2026 补齐）

| 文档 | 覆盖内容 |
|------|---------|
| [LLM-Embedding冲榜路线_E5Mistral-NVEmbed-GritLM-SFR-Arctic-Stella.md](LLM-Embedding冲榜路线_E5Mistral-NVEmbed-GritLM-SFR-Arctic-Stella.md) | E5-Mistral / GritLM / NV-Embed-v2 / SFR-Embedding-2R / Arctic-Embed v1&2 / Stella-Jasper 六篇合写；6 个可插拔积木 + 消融交集 |
| [MLLM通用多模态嵌入_GME-VLM2Vec-BGEVL.md](MLLM通用多模态嵌入_GME-VLM2Vec-BGEVL.md) | GME [2412.16855](https://arxiv.org/abs/2412.16855) + VLM2Vec+MMEB [2410.05160](https://arxiv.org/abs/2410.05160) + BGE-VL/MegaPairs [2412.14475](https://arxiv.org/abs/2412.14475) 三篇合写 |
| [前沿短深读_LateChunking-Vec2Vec-ModernBERT-DINOv2v3-Qwen3-Seed-ViDoRev2.md](前沿短深读_LateChunking-Vec2Vec-ModernBERT-DINOv2v3-Qwen3-Seed-ViDoRev2.md) | Late Chunking [2409.04701](https://arxiv.org/abs/2409.04701) / Vec2Vec [2505.12540](https://arxiv.org/abs/2505.12540) / ModernBERT [2412.13663](https://arxiv.org/abs/2412.13663) / DINOv2 [2304.07193](https://arxiv.org/abs/2304.07193)+DINOv3 [2508.10104](https://arxiv.org/abs/2508.10104) / Qwen3-Embedding [2506.05176](https://arxiv.org/abs/2506.05176) / Seed1.5-Embedding / ViDoRe v2 [2505.17166](https://arxiv.org/abs/2505.17166) 七个方向合写 |

## Late Interaction 族（文本 → 视觉文档）

| 文档 | 论文 / 来源 |
|------|-------------|
| [ColBERT详解.md](ColBERT/ColBERT详解.md) | [arXiv:2004.12832](https://arxiv.org/abs/2004.12832) |
| [ColBERTv2详解.md](ColBERTv2/ColBERTv2详解.md) | [arXiv:2112.01488](https://arxiv.org/abs/2112.01488) |
| [ColPali详解.md](ColPali/ColPali详解.md) | [arXiv:2407.01449](https://arxiv.org/abs/2407.01449) |
| [ColQwen系列详解.md](ColQwen/ColQwen系列详解.md) | 同 ColPali 论文 + Vidore 模型卡（ColQwen2 / 2.5；社区 ColQwen3） |

## 稠密文本 Embedding 深读

| 文档 | 论文 / 来源 |
|------|-------------|
| [E5详解.md](E5/E5详解.md) | [arXiv:2212.03533](https://arxiv.org/abs/2212.03533) |
| [GTE系列详解.md](GTE/GTE系列详解.md) | GTE [2308.03281](https://arxiv.org/abs/2308.03281) → mGTE / gte-v1.5 [2407.19669](https://arxiv.org/abs/2407.19669) → [gte-Qwen2](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct)；文末有三代演变总结 |
| [Nomic-Embed详解.md](Nomic-Embed/Nomic-Embed详解.md) | [arXiv:2402.01613](https://arxiv.org/abs/2402.01613) |
| [LLM2Vec详解.md](LLM2Vec/LLM2Vec详解.md) | [arXiv:2404.05961](https://arxiv.org/abs/2404.05961) |
| [Conan-embedding详解.md](Conan-embedding/v1/Conan-embedding详解.md) | [arXiv:2408.15710](https://arxiv.org/abs/2408.15710)（v1：DHNM + Cross-GPU CBB；CMTEB） |
| [Conan-embedding-v2详解.md](Conan-embedding/v2/Conan-embedding-v2详解.md) | [arXiv:2509.12892](https://arxiv.org/abs/2509.12892)（EMNLP 2025；从零 1.4B LLM） |
| [QZhou-Embedding详解.md](QZhou-Embedding/QZhou-Embedding详解.md) | [arXiv:2508.21632](https://arxiv.org/abs/2508.21632)（双向 Qwen2.5-7B + 多任务/合成/两阶段） |
| [Token-Prepending详解.md](Token-Prepending/Token-Prepending详解.md) | [ACL 2025](https://aclanthology.org/2025.acl-long.159/)（免训练层间回灌；因果注意力补丁） |

## Jina 系列

| 文档 | 论文 / 来源 |
|------|-------------|
| [Jina系列总览.md](Jina/Jina系列总览.md) | 谱系索引（v1→v5-omni；含架构类属与选型） |
| [Jina-embeddings-v2详解.md](Jina/v2/Jina-embeddings-v2详解.md) | [arXiv:2310.19923](https://arxiv.org/abs/2310.19923)（v1 见 [2307.11224](https://arxiv.org/abs/2307.11224)） |
| [Jina-embeddings-v3详解.md](Jina/v3/Jina-embeddings-v3详解.md) | [arXiv:2409.10173](https://arxiv.org/abs/2409.10173) |
| [jina-clip系列详解.md](Jina/clip/jina-clip系列详解.md) | [arXiv:2405.20204](https://arxiv.org/abs/2405.20204) / [2412.08802](https://arxiv.org/abs/2412.08802) |
| [Jina-embeddings-v4详解.md](Jina/v4/Jina-embeddings-v4详解.md) | [arXiv:2506.18902](https://arxiv.org/abs/2506.18902) |
| [Jina-embeddings-v5-text详解.md](Jina/v5-text/Jina-embeddings-v5-text详解.md) | [arXiv:2602.15547](https://arxiv.org/abs/2602.15547) |
| [Jina-embeddings-v5-omni详解.md](Jina/v5-omni/Jina-embeddings-v5-omni详解.md) | [arXiv:2605.08384](https://arxiv.org/abs/2605.08384) |

## 关联报告

| 文档 | 说明 |
|------|------|
| [../lora/README.md](../lora/README.md) | LoRA / PEFT（Jina v3 任务 LoRA、cloud_emb stage2 等） |
| [../distillation/知识蒸馏技术深度调研报告.md](../distillation/知识蒸馏技术深度调研报告.md) | 蒸馏总论 |

工程代码与训练数据在 **modelforge**：`scripts/embedding/`、`datasets/embedding/`、`tasks/cloud_emb/`。
