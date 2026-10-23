# DeepSeek 论文与技术演进

截至 **2026-10-22**，按 [DeepSeek 官方 GitHub](https://github.com/deepseek-ai)、[官方 Hugging Face 论文列表](https://huggingface.co/deepseek-ai/papers)及原论文作者机构交叉整理，收录 **35 篇官方报告及 DeepSeek 参与主导的研究**。同篇论文的 arXiv 修订合并；V3.2-Exp 是独立的官方短报告，与正式 V3.2 分开。范围包括早期联合研究 DreamCraft3D，也包括 Cordis 这样的软件基础工作，不只限于带 DeepSeek 名称的模型。

每篇的“精炼”顺着原文覆盖方法、实验与边界；“解读”用于补充直觉、实现条件和跨论文比较。中英对照按英文段落后紧跟中文组织，只在有全文改写许可的材料中提供，保留原公式、图表及书目。中文以机器初译为底稿，已做术语整理和结构核对，**尚未逐句人工审定**。未获全文翻译许可的条目提供精炼、解读和官方入口，不以摘要替代全文。

Notion 原始页面归于 Paper，DOMAIN 为 LLM，TAG 为 DeepSeek；[LLM → Model Series](https://www.notion.so/1082a8d3bde24834a7f69559635458e0) 的 DeepSeek tab 以四级标题按 LLM、OCR、多模态、数学/证明、架构与系统等主题分组，只放论文原页引用。Notion 展示精炼正文与置顶的补充解读子页，不同步翻译。已有个人内容保留，后续修订只更新已导入区域，避免重复追加整篇。

## 建议怎样串起来读

- **主干演进**：DeepSeek LLM → DeepSeekMoE → V2（MLA）→ V3（训练系统与配方）→ R1（推理后训练）→ V3.2-Exp／V3.2（DSA 与 Agent）→ V4／V4.1-Flash。每代报告的模型、推理预算与工具协议不同，数字不能直接拼成同口径排行榜。
- **强化学习**：先看 [DPO 精讲](../RLHF/DPO精讲.md)与 [GRPO 精讲](../RLHF/GRPO精讲.md)，再读 DeepSeekMath、R1、GRM、Math-V2 和 Prover 系列。区分人类偏好、模型评判与可验证结果奖励。
- **效率与架构**：MoE → Loss-Free Balancing／ESFT；MLA → NSA → DSA；系统方面结合 Fire-Flyer、V3 Hardware、DualPath、DSpark。mHC 与 Engram 分别从残差连接与条件记忆补充架构路线。
- **多模态**：VL／VL2 关注理解，Janus 系列关注统一理解与生成，OCR 系列关注视觉压缩与文档，DreamCraft3D 是三维生成，任务不同不能用单一指标排序。

## 通用语言模型与推理

| 论文 | 阅读内容 | 中英对照 | 原文／Notion |
|---|---|---|---|
| DeepSeek-LLM | [精炼](DeepSeek-LLM/DeepSeek-LLM详解.md) · [解读](DeepSeek-LLM/DeepSeek-LLM解读.md) | 未提供（全文改写许可未确认） | [原文](https://arxiv.org/abs/2401.02954) · [Notion](https://www.notion.so/401a174c-d8c6-812b-8b10-e223c913a05f) |
| DeepSeek-V2 | [精炼](DeepSeek-V2/DeepSeek-V2详解.md) · [解读](DeepSeek-V2/DeepSeek-V2解读.md) | 未提供（全文改写许可未确认） | [原文](https://arxiv.org/abs/2405.04434) · [Notion](https://www.notion.so/401a174c-d8c6-8189-8708-ec5b60f349a1) |
| DeepSeek-V3 | [精炼](DeepSeek-V3/DeepSeek-V3详解.md) · [解读](DeepSeek-V3/DeepSeek-V3解读.md) | 未提供（全文改写许可未确认） | [原文](https://arxiv.org/abs/2412.19437) · [Notion](https://www.notion.so/401a174c-d8c6-81a9-8c27-c0364f4d41d0) |
| DeepSeek-R1 | [精炼](DeepSeek-R1/DeepSeek-R1详解.md) · [解读](DeepSeek-R1/DeepSeek-R1解读.md) | [全文对照](DeepSeek-R1/zh.md)，官方 v1 | [原文](https://arxiv.org/abs/2501.12948) · [Notion](https://www.notion.so/401a174c-d8c6-81f7-aba0-e541a112b4fa) |
| DeepSeek-V3.2-Exp | [精炼](DeepSeek-V3.2-Exp/DeepSeek-V3.2-Exp详解.md) · [解读](DeepSeek-V3.2-Exp/DeepSeek-V3.2-Exp解读.md) | [全文对照](DeepSeek-V3.2-Exp/zh.md) | [原文](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf) · [Notion](https://www.notion.so/401a174c-d8c6-8159-81cc-ccd677905cc9) |
| DeepSeek-V3.2 | [精炼](DeepSeek-V3.2/DeepSeek-V3.2详解.md) · [解读](DeepSeek-V3.2/DeepSeek-V3.2解读.md) | 未提供（全文改写许可未确认） | [原文](https://arxiv.org/abs/2512.02556) · [Notion](https://www.notion.so/401a174c-d8c6-8105-8c43-c07d288705d4) |
| DeepSeek-V4 | [精炼](DeepSeek-V4/DeepSeek-V4详解.md) · [解读](DeepSeek-V4/DeepSeek-V4解读.md) | 未提供（全文改写许可未确认） | [原文](https://arxiv.org/abs/2606.19348) · [Notion](https://www.notion.so/401a174c-d8c6-81e8-9473-de375cd157f7) |
| DeepSeek-V4.1-Flash | [精炼](DeepSeek-V4.1-Flash/DeepSeek-V4.1-Flash详解.md) · [解读](DeepSeek-V4.1-Flash/DeepSeek-V4.1-Flash解读.md) | 未提供（全文改写许可未确认） | [原文](https://arxiv.org/abs/2609.19969) · [Notion](https://www.notion.so/401a174c-d8c6-8107-b482-cd4b8f45dfdf) |

## 代码、数学与形式化证明

| 论文 | 阅读内容 | 中英对照 | 原文／Notion |
|---|---|---|---|
| DeepSeek-Coder | [精炼](DeepSeek-Coder/DeepSeek-Coder详解.md) · [解读](DeepSeek-Coder/DeepSeek-Coder解读.md) | 未提供（全文改写许可未确认） | [原文](https://arxiv.org/abs/2401.14196) · [Notion](https://www.notion.so/401a174c-d8c6-81b6-b193-df0e424bc011) |
| DeepSeek-Coder-V2 | [精炼](DeepSeek-Coder-V2/DeepSeek-Coder-V2详解.md) · [解读](DeepSeek-Coder-V2/DeepSeek-Coder-V2解读.md) | [全文对照](DeepSeek-Coder-V2/zh.md) | [原文](https://arxiv.org/abs/2406.11931) · [Notion](https://www.notion.so/401a174c-d8c6-81a9-95dd-c4e69973e411) |
| DeepSeekMath | [精炼](DeepSeekMath/DeepSeekMath详解.md) · [解读](DeepSeekMath/DeepSeekMath解读.md) | 未提供（全文改写许可未确认） | [原文](https://arxiv.org/abs/2402.03300) · [Notion](https://www.notion.so/401a174c-d8c6-819a-9b89-e12ff22f0d02) |
| DeepSeekMath-V2 | [精炼](DeepSeekMath-V2/DeepSeekMath-V2详解.md) · [解读](DeepSeekMath-V2/DeepSeekMath-V2解读.md) | [全文对照](DeepSeekMath-V2/zh.md) | [原文](https://arxiv.org/abs/2511.22570) · [Notion](https://www.notion.so/401a174c-d8c6-81d5-862a-ea78606a986b) |
| DeepSeek-Prover | [精炼](DeepSeek-Prover/DeepSeek-Prover详解.md) · [解读](DeepSeek-Prover/DeepSeek-Prover解读.md) | [全文对照](DeepSeek-Prover/zh.md) | [原文](https://arxiv.org/abs/2405.14333) · [Notion](https://www.notion.so/401a174c-d8c6-81cf-80f9-ca0a1fc8cfe1) |
| DeepSeek-Prover-V1.5 | [精炼](DeepSeek-Prover-V1.5/DeepSeek-Prover-V1.5详解.md) · [解读](DeepSeek-Prover-V1.5/DeepSeek-Prover-V1.5解读.md) | [全文对照](DeepSeek-Prover-V1.5/zh.md) | [原文](https://arxiv.org/abs/2408.08152) · [Notion](https://www.notion.so/401a174c-d8c6-8186-8ad5-c8fbdc83157c) |
| DeepSeek-Prover-V2 | [精炼](DeepSeek-Prover-V2/DeepSeek-Prover-V2详解.md) · [解读](DeepSeek-Prover-V2/DeepSeek-Prover-V2解读.md) | 未提供（全文改写许可未确认） | [原文](https://arxiv.org/abs/2504.21801) · [Notion](https://www.notion.so/401a174c-d8c6-8124-9c16-f10b7195d634) |

## 多模态、图像生成与文档理解

| 论文 | 阅读内容 | 中英对照 | 原文／Notion |
|---|---|---|---|
| DeepSeek-VL | [精炼](DeepSeek-VL/DeepSeek-VL详解.md) · [解读](DeepSeek-VL/DeepSeek-VL解读.md) | 未提供（全文改写许可未确认） | [原文](https://arxiv.org/abs/2403.05525) · [Notion](https://www.notion.so/401a174c-d8c6-81fb-b77a-ccfbf2d69fdd) |
| DeepSeek-VL2 | [精炼](DeepSeek-VL2/DeepSeek-VL2详解.md) · [解读](DeepSeek-VL2/DeepSeek-VL2解读.md) | [全文对照](DeepSeek-VL2/zh.md) | [原文](https://arxiv.org/abs/2412.10302) · [Notion](https://www.notion.so/401a174c-d8c6-8190-8f03-c350cdc14dd3) |
| Janus | [精炼](Janus/Janus详解.md) · [解读](Janus/Janus解读.md) | [全文对照](Janus/zh.md) | [原文](https://arxiv.org/abs/2410.13848) · [Notion](https://www.notion.so/401a174c-d8c6-8168-8a03-eaec71b70ce4) |
| JanusFlow | [精炼](JanusFlow/JanusFlow详解.md) · [解读](JanusFlow/JanusFlow解读.md) | [全文对照](JanusFlow/zh.md) | [原文](https://arxiv.org/abs/2411.07975) · [Notion](https://www.notion.so/401a174c-d8c6-81da-bc5f-df1fe018f366) |
| Janus-Pro | [精炼](Janus-Pro/Janus-Pro详解.md) · [解读](Janus-Pro/Janus-Pro解读.md) | [全文对照](Janus-Pro/zh.md) | [原文](https://arxiv.org/abs/2501.17811) · [Notion](https://www.notion.so/401a174c-d8c6-8184-b97f-e69d3a61bf9a) |
| DeepSeek-OCR | [精炼](DeepSeek-OCR/DeepSeek-OCR详解.md) · [解读](DeepSeek-OCR/DeepSeek-OCR解读.md) | [全文对照](DeepSeek-OCR/zh.md) | [原文](https://arxiv.org/abs/2510.18234) · [Notion](https://www.notion.so/401a174c-d8c6-81ae-92ec-d506d6ef8598) |
| DeepSeek-OCR-2 | [精炼](DeepSeek-OCR-2/DeepSeek-OCR-2详解.md) · [解读](DeepSeek-OCR-2/DeepSeek-OCR-2解读.md) | [全文对照](DeepSeek-OCR-2/zh.md) | [原文](https://arxiv.org/abs/2601.20552) · [Notion](https://www.notion.so/401a174c-d8c6-81b4-8523-c509fb7e0c45) |
| DreamCraft3D | [精炼](DreamCraft3D/DreamCraft3D详解.md) · [解读](DreamCraft3D/DreamCraft3D解读.md) | 未提供（全文改写许可未确认） | [原文](https://arxiv.org/abs/2310.16818) · [Notion](https://www.notion.so/401a174c-d8c6-81fe-9487-e47a306d689a) |

## 架构、适配与奖励建模

| 论文 | 阅读内容 | 中英对照 | 原文／Notion |
|---|---|---|---|
| DeepSeekMoE | [精炼](DeepSeekMoE/DeepSeekMoE详解.md) · [解读](DeepSeekMoE/DeepSeekMoE解读.md) | 未提供（全文改写许可未确认） | [原文](https://arxiv.org/abs/2401.06066) · [Notion](https://www.notion.so/401a174c-d8c6-81e9-bd57-d03318ddd0c1) |
| Loss-Free-Balancing | [精炼](Loss-Free-Balancing/Loss-Free-Balancing详解.md) · [解读](Loss-Free-Balancing/Loss-Free-Balancing解读.md) | 未提供（全文改写许可未确认） | [原文](https://arxiv.org/abs/2408.15664) · [Notion](https://www.notion.so/401a174c-d8c6-81d4-a66c-d790914979ae) |
| ESFT | [精炼](ESFT/ESFT详解.md) · [解读](ESFT/ESFT解读.md) | [全文对照](ESFT/zh.md) | [原文](https://arxiv.org/abs/2407.01906) · [Notion](https://www.notion.so/401a174c-d8c6-81bd-9809-c184fd433608) |
| Native-Sparse-Attention | [精炼](Native-Sparse-Attention/Native-Sparse-Attention详解.md) · [解读](Native-Sparse-Attention/Native-Sparse-Attention解读.md) | [全文对照](Native-Sparse-Attention/zh.md) | [原文](https://arxiv.org/abs/2502.11089) · [Notion](https://www.notion.so/401a174c-d8c6-8192-9d82-eb208efb83d2) |
| mHC | [精炼](mHC/mHC详解.md) · [解读](mHC/mHC解读.md) | 未提供（全文改写许可未确认） | [原文](https://arxiv.org/abs/2512.24880) · [Notion](https://www.notion.so/401a174c-d8c6-8118-b400-dba1b4081d07) |
| Engram | [精炼](Engram/Engram详解.md) · [解读](Engram/Engram解读.md) | [全文对照](Engram/zh.md) | [原文](https://arxiv.org/abs/2601.07372) · [Notion](https://www.notion.so/401a174c-d8c6-8161-9b40-e0e622d2cfce) |
| DeepSeek-GRM | [精炼](DeepSeek-GRM/DeepSeek-GRM详解.md) · [解读](DeepSeek-GRM/DeepSeek-GRM解读.md) | [全文对照](DeepSeek-GRM/zh.md) | [原文](https://arxiv.org/abs/2504.02495) · [Notion](https://www.notion.so/401a174c-d8c6-8168-9a04-d38646b5cda2) |

## 训练、推理系统与软件基础

| 论文 | 阅读内容 | 中英对照 | 原文／Notion |
|---|---|---|---|
| Fire-Flyer-AI-HPC | [精炼](Fire-Flyer-AI-HPC/Fire-Flyer-AI-HPC详解.md) · [解读](Fire-Flyer-AI-HPC/Fire-Flyer-AI-HPC解读.md) | 未提供（全文改写许可未确认） | [原文](https://arxiv.org/abs/2408.14158) · [Notion](https://www.notion.so/401a174c-d8c6-81db-9f07-dff7c8404ebb) |
| DeepSeek-V3-Hardware | [精炼](DeepSeek-V3-Hardware/DeepSeek-V3-Hardware详解.md) · [解读](DeepSeek-V3-Hardware/DeepSeek-V3-Hardware解读.md) | 未提供（全文改写许可未确认） | [原文](https://arxiv.org/abs/2505.09343) · [Notion](https://www.notion.so/401a174c-d8c6-811c-929f-dbcd9969d846) |
| DualPath | [精炼](DualPath/DualPath详解.md) · [解读](DualPath/DualPath解读.md) | 未提供（全文改写许可未确认） | [原文](https://arxiv.org/abs/2602.21548) · [Notion](https://www.notion.so/401a174c-d8c6-8196-8e24-e953fb25dc5a) |
| DSpark | [精炼](DSpark/DSpark详解.md) · [解读](DSpark/DSpark解读.md) | [全文对照](DSpark/zh.md) | [原文](https://arxiv.org/abs/2607.05147) · [Notion](https://www.notion.so/401a174c-d8c6-819c-bfe0-f711091a06f1) |
| Cordis | [精炼](Cordis/Cordis详解.md) · [解读](Cordis/Cordis解读.md) | 未提供（全文改写许可未确认） | [原文](https://arxiv.org/abs/2608.25512) · [Notion](https://www.notion.so/401a174c-d8c6-815c-b2ec-c47182790acb) |

## 版本与使用说明

R1 精炼笔记讨论后续扩展报告；全文对照采用官方仓库的 **22 页初版，与 arXiv v1 对应**，不声称覆盖后来 86 页的 v2。其他文件文首标明实际所取版本。Prover-V2 等后续修订中的数据勘误已按对应版本整理。

有授权且体积合适的 PDF 随论文目录保存；大文件通过官方下载。原图优先取自 arXiv HTML；Cordis 与 V3.2-Exp 使用 PDF 原图裁切。用于图像生成或 OCR 展示的图片保持可读细节，部分只用于全文对照的 PNG 转为无损 WebP。

模型权重许可、代码许可与论文文本许可不是同一回事。本文不会仅凭仓库中存在代码许可证，就推定所有外部 arXiv 版本均可全文改写。R1、Math-V2、V3.2-Exp 使用对应官方仓库内的授权报告，许可证随目录保留。

## 论文之外的官方实现与技术资料

这些项目补充工程实现，不为没有独立报告的仓库虚构论文条目：

| 项目 | 适合补充的主题 |
|---|---|
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | MLA 推理内核、稀疏注意力及硬件实现 |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | MoE token dispatch／combine 与通信 |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | FP8 GEMM、MoE 分组矩阵乘 |
| [DualPipe](https://github.com/deepseek-ai/DualPipe) | 计算通信重叠与流水线并行 |
| [3FS](https://github.com/deepseek-ai/3FS) | 高吞吐分布式文件系统 |
| [smallpond](https://github.com/deepseek-ai/smallpond) | 数据处理与训练数据基础设施 |
| [EPLB](https://github.com/deepseek-ai/EPLB) | 专家并行负载均衡与部署布局 |
| [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)、[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) | 模型发布说明、推理示例与版本变化 |

V2.5、V3.1、R1-0528 等发布或权重更新，通过相应官方模型仓库追踪；没有另行核实独立论文时，不把发布名重复计入论文数量。Hyper-Connections 原论文及 DreamCraft3D++ 可作为相关研究阅读，但未计入本表的 DeepSeek 论文范围。
