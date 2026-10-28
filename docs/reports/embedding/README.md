# Embedding 学习文档

按“技术总览 → 机制专题 → 原论文研读”组织 Embedding 学习内容。**Markdown 为权威正文**；历史 HTML 仅供版式对照，阅读以 MD 为准。

单篇/系列精读与论文 PDF、中译、`figs/` 放在同一方法文件夹（如 `GTE/GTE系列详解.md`、`InternLM2/InternLM2数据处理与过滤详解.md`）。总览、合写、索引仍在本目录。报告嵌入图在 `figures/<短名>/`。

Notion：Embedding&Rerank → Paper 库，一篇论文一个完整阅读页 + ModelCard mention；过程记录不创建子页。写作参照 [论文阅读笔记指南](../../paper-reading-guide.md)，权限与落点见根目录 [AGENTS.md](../../../AGENTS.md)。

## 总览与技术专题

| 要学习的内容 | 入口与分工 |
| --- | --- |
| 表示、训练、评测与部署的完整框架 | [Embedding 总览](Embedding调研报告.md)，同时承接稀疏检索、Late Interaction、指令非对称编码、污染防控和向量压缩的跨论文解释 |
| 技术演进、系列节点与代表性成绩 | [发展趋势](Embedding调研报告.md#发展趋势与代表性结果)、[逐项时间线](Embedding调研报告.md#逐项时间线方法系列与发布节点)；同时保留阶段概览和模型规格卡，区分论文首发、模型发布与正式发表 |
| 图文架构与交互方式 | [图文技术综述](图文Embedding模型技术综述.md)：双塔、融合、MLLM 与视觉文档路线；模型规格需结合各篇报告核对 |
| 图搜图／文搜图的数据与训练 | [图搜图与文搜图表示学习](图搜图与文搜图表示学习.md)：实例／语义标签、图文损失、多教师蒸馏、≤0.6B 约束和部署 |
| Embedding 损失、监督组织与梯度 | [Embedding 损失函数精讲](对比学习与InfoNCE精讲.md)：Triplet／N-pair／MS／Circle、InfoNCE／NT-Xent／SupCon、alignment／uniformity、回归／类别代理／蒸馏；结合 [GradCache](GradCache/GradCache详解.md) 区分大候选集合与梯度累积 |
| 假负例、难例和动态刷新 | [难负例实践](难负例挖掘工业实践.md)；具体算法看 ANCE、RocketQA、NV-Retriever |
| 分数、排序与特征蒸馏 | [Embedding 蒸馏专题](Embedding蒸馏技术详解.md)，结合 [Margin-MSE](Margin-MSE/Margin-MSE研读笔记.md) 与 [多语 SBERT](Multilingual-SBERT/Multilingual-SBERT研读笔记.md) |
| 领域标签和适配课表 | [领域适配实践](领域专用Embedding适配实践.md)，按科学、生物、代码、法律与推荐回查案例 |
| 原论文、官方实现、博客和未覆盖材料 | [统一资料清单](资料清单_论文与博客.md)；现有研读入口在下面各表，不把仅收录链接算作完成研读 |

按问题选读即可：先掌握“骨干／表示／交互／读出”，再读任务所需的专题。经典工作可沿 Sentence-BERT → DPR 理解句向量与问答检索的差异；RAG、FiD、BEIR 等衔接材料的覆盖情况见[经典论文清单](资料清单_论文与博客.md#奠基性论文与经典阅读顺序)。

## 论文精炼版与补充解读

精炼版按论文顺序覆盖方法、数据、实验和消融；补充解读讨论机制与工程取舍。`zh.md` 为英文段落后紧接中文的全文对照初译，已检查结构与术语，尚未逐句人工审定；Notion 仅展示精炼版与独立解读子页；首次补充保留用户内容并接在其后，后续原位更新已导入部分。

| 论文 / 模型报告 | 精炼版 | 补充解读 | 仓库全文对照 |
| --- | --- | --- | --- |
| DPR | [阅读](DPR/DPR详解.md) | [解读](DPR/DPR解读.md) | [中英对照](DPR/zh.md) |
| ColBERT | [阅读](ColBERT/ColBERT详解.md) | [解读](ColBERT/ColBERT解读.md) | 待原文副本 / 译文许可 |
| ColBERTv2 | [阅读](ColBERTv2/ColBERTv2详解.md) | [解读](ColBERTv2/ColBERTv2解读.md) | [中英对照](ColBERTv2/zh.md) |
| PLAID | [阅读](PLAID/PLAID详解.md) | [解读](PLAID/PLAID解读.md) | [中英对照](PLAID/zh.md) |
| ColPali | [阅读](ColPali/ColPali详解.md) | [解读](ColPali/ColPali解读.md) | [中英对照](ColPali/zh.md) |
| CLIP | [阅读](CLIP/CLIP详解.md) | [解读](CLIP/CLIP解读.md) | [中英对照](CLIP/zh.md) |
| SigLIP | [阅读](SigLIP/SigLIP/SigLIP详解.md) | [解读](SigLIP/SigLIP/SigLIP解读.md) | [中英对照](SigLIP/SigLIP/zh.md) |
| SigLIP2 | [阅读](SigLIP/SigLIP2/SigLIP2详解.md) | [解读](SigLIP/SigLIP2/SigLIP2解读.md) | [中英对照](SigLIP/SigLIP2/zh.md) |
| GME | [阅读](GME/GME详解.md) | [解读](GME/GME解读.md) | [中英对照](GME/zh.md) |
| VLM2Vec | [阅读](VLM2Vec/VLM2Vec详解.md) | [解读](VLM2Vec/VLM2Vec解读.md) | [中英对照](VLM2Vec/zh.md) |
| MegaPairs | [阅读](MegaPairs/MegaPairs详解.md) | [解读](MegaPairs/MegaPairs解读.md) | [中英对照](MegaPairs/zh.md) |
| Jina-v1 | [阅读](Jina/v1/Jina-embeddings-v1详解.md) | [解读](Jina/v1/Jina-embeddings-v1解读.md) | 待原文副本 / 译文许可 |
| Jina-v2 | [阅读](Jina/v2/Jina-embeddings-v2详解.md) | [解读](Jina/v2/Jina-embeddings-v2解读.md) | [中英对照](Jina/v2/zh.md) |
| Jina-v3 | [阅读](Jina/v3/Jina-embeddings-v3详解.md) | [解读](Jina/v3/Jina-embeddings-v3解读.md) | [中英对照](Jina/v3/zh.md) |
| Jina-CLIP | [阅读](Jina/clip/v1/Jina-CLIP详解.md) | [解读](Jina/clip/v1/Jina-CLIP解读.md) | [中英对照](Jina/clip/v1/zh.md) |
| Jina-CLIP-v2 | [阅读](Jina/clip/v2/Jina-CLIP-v2详解.md) | [解读](Jina/clip/v2/Jina-CLIP-v2解读.md) | [中英对照](Jina/clip/v2/zh.md) |
| Jina-ColBERT-v2 | [阅读](Jina/colbert-v2/Jina-ColBERT-v2详解.md) | [解读](Jina/colbert-v2/Jina-ColBERT-v2解读.md) | [中英对照](Jina/colbert-v2/zh.md) |
| Jina-reranker-v3 | [阅读](Jina/rerank-v3/jina-reranker-v3详解.md) | [解读](Jina/rerank-v3/jina-reranker-v3解读.md) | [中英对照](Jina/rerank-v3/zh.md) |
| Sentence-BERT | [阅读](Sentence-BERT/Sentence-BERT详解.md) | [解读](Sentence-BERT/Sentence-BERT解读.md) | [中英对照](Sentence-BERT/zh.md) |
| SPLADE | [阅读](SPLADE/SPLADE/SPLADE详解.md) | [解读](SPLADE/SPLADE/SPLADE解读.md) | [中英对照](SPLADE/SPLADE/zh.md) |
| SPLADE-v2 | [阅读](SPLADE/SPLADE-v2/SPLADE-v2详解.md) | [解读](SPLADE/SPLADE-v2/SPLADE-v2解读.md) | [中英对照](SPLADE/SPLADE-v2/zh.md) |
| MRL | [阅读](MRL/MRL详解.md) | [解读](MRL/MRL解读.md) | 待原文副本 / 译文许可 |
| GradCache | [阅读](GradCache/GradCache详解.md) | [解读](GradCache/GradCache解读.md) | [中英对照](GradCache/zh.md) |
| FollowIR | [阅读](FollowIR/FollowIR详解.md) | [解读](FollowIR/FollowIR解读.md) | [中英对照](FollowIR/zh.md) |
| HyDE | [阅读](HyDE/HyDE详解.md) | [解读](HyDE/HyDE解读.md) | [中英对照](HyDE/zh.md) |
| LongEmbed | [阅读](LongEmbed/LongEmbed详解.md) | [解读](LongEmbed/LongEmbed解读.md) | [中英对照](LongEmbed/zh.md) |
| DINOv2 | [阅读](DINO/DINOv2/DINOv2详解.md) | [解读](DINO/DINOv2/DINOv2解读.md) | [中英对照](DINO/DINOv2/zh.md) |
| DINOv3 | [阅读](DINO/DINOv3/DINOv3详解.md) | [解读](DINO/DINOv3/DINOv3解读.md) | [中英对照](DINO/DINOv3/zh.md) |
| ViDoRe-v2 | [阅读](ViDoRev2/ViDoRe-v2详解.md) | [解读](ViDoRev2/ViDoRe-v2解读.md) | [中英对照](ViDoRev2/zh.md) |
| Qwen3-VL-Embedding | [阅读](QAT/Qwen3-VL-Embedding/Qwen3-VL-Embedding详解.md) | [解读](QAT/Qwen3-VL-Embedding/Qwen3-VL-Embedding解读.md) | [中英对照](QAT/Qwen3-VL-Embedding/zh.md) |
| ColQwen | [阅读](ColQwen/ColQwen系列详解.md) | [解读](ColQwen/ColQwen系列解读.md) | [中英对照](ColQwen/zh.md) |

ColQwen 对应官方模型卡报告与对照，未假定它有独立源论文。ColBERT、MRL 和 Jina v1 已完成精炼版与解读；仓库没有其原文副本，尚未确认可再发布全文译文的许可，因此不把摘要替代品命名为全文对照。

## 专题与系列深读

| 文档 | 说明 |
|------|------|
| [BGE-M3三功能统一详解报告.md](BGE/M3/BGE-M3三功能统一详解报告.md) | BGE-M3 |
| [BGE-CPack详解.md](BGE/C-Pack/BGE-CPack详解.md) | [arXiv:2309.07597](https://arxiv.org/abs/2309.07597)（C-Pack / BGE 全家桶起点：C-MTEB + C-MTP + BGE 模型 + 三阶段训练） |
| [BGE-EN-ICL详解.md](BGE/EN-ICL/BGE-EN-ICL详解.md) | [arXiv:2409.15700](https://arxiv.org/abs/2409.15700)（Mistral-7B + causal + [EOS] + few-shot ICL 训练；MTEB 71.24 / 71.67） |
| [BGE-multilingual-gemma2详解.md](BGE/BGE-multilingual-gemma2详解.md) | Gemma-2-9B 骨干（附录 C of BGE-EN-ICL 论文）；MIRACL 74.1 SOTA / FR-MTEB / PL-MTEB SOTA |
| [BGE-Reranker详解.md](BGE/BGE-Reranker详解.md) | v2-m3 / v2-gemma / v2-minicpm-layerwise / **v2.5-gemma2-lightweight**（附录 D of BGE-EN-ICL 论文；深度+宽度双压缩） |
| [Embedding蒸馏技术详解.md](Embedding蒸馏技术详解.md) | Embedding 蒸馏专题 |
| [量化感知训练QAT详解.md](QAT/量化感知训练QAT详解.md) | QAT（Jacob/LSQ）vs 向量 QAT（Qwen3-VL）vs Jina v5 GOR；对齐文搜图索引 |
| [Jacob2018详解.md](QAT/Jacob2018/Jacob2018详解.md) | CVPR 2018 整数推理与 QAT 原论文：仿射量化、定点重缩放、BN 折叠与端侧实验；附英文全文和中文全译 |
| [Qwen3-VL-Embedding详解.md](QAT/Qwen3-VL-Embedding/Qwen3-VL-Embedding详解.md) | [arXiv:2601.04720v2](https://arxiv.org/abs/2601.04720v2)（多模态单向量召回与重排；数据合成、三阶段训练、MRL/QAT 与消融） |
| [难负例挖掘工业实践.md](难负例挖掘工业实践.md) | 难负例工业闭环 |
| [Embedding 损失函数精讲](对比学习与InfoNCE精讲.md) | 度量／对比／排序／回归／蒸馏大章：定义、样本与候选、InfoNCE 分母及梯度、多正例、MS／Circle、alignment／uniformity 与工程取舍 |
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

## 蒸馏基石

方法论总览见 [Embedding蒸馏技术详解.md](Embedding蒸馏技术详解.md)；下面两篇是被反复引用的原始论文，按论文结构写成研读笔记。

| 文档 | 论文 / 来源 |
|------|-------------|
| [Margin-MSE研读笔记.md](Margin-MSE/Margin-MSE研读笔记.md) | [arXiv:2010.02666](https://arxiv.org/abs/2010.02666)（跨架构蒸馏：只对齐正负例间隔，拼接式 BERT 教任意架构；GPL / R-GPL / `msmarco-*-margin-mse` 的源头） |
| [Multilingual-SBERT研读笔记.md](Multilingual-SBERT/Multilingual-SBERT研读笔记.md) | [arXiv:2004.09813](https://arxiv.org/abs/2004.09813) / [EMNLP 2020](https://aclanthology.org/2020.emnlp-main.365/)（多语蒸馏：平行句对 + 单条 MSE 把英文空间搬到 50 语；`paraphrase-multilingual-*` 的方法） |

## 领域专用 Embedding（2026-08 补齐）

面向文搜图等垂直场景：先读实践综述，再按领域看案例。总览的“代码与领域专用”一节只保留索引。

| 文档 | 说明 |
|------|------|
| [领域专用Embedding适配实践.md](领域专用Embedding适配实践.md) | Stage 2/4 课表 + 各领域正负例对照 + **对齐文搜图清单** |
| [SPECTER系列详解.md](SPECTER/SPECTER系列详解.md) | 引用图对比；SPECTER2 adapter / SciRepEval；[Ai2 博文](https://allenai.org/blog/specter2-adapting-scientific-document-embeddings-to-multiple-fields-and-task-formats-c95686c06567) |
| [MedCPT详解.md](MedCPT/MedCPT详解.md) | PubMed 点击 255M + 双塔/交叉编码器；生物医学零样本 |
| [代码检索Embedding合写.md](Code-IR/代码检索Embedding合写.md) | CodeBERT / UniXcoder / CoIR / BGE-Code（CodeR 合成） |
| [法律检索Embedding合写.md](Legal/法律检索Embedding合写.md) | LEGAL-BERT 领域 MLM + SAILER 结构感知；Hybrid 保条号 |
| [YouTube-DNN双塔详解.md](YouTube-DNN/YouTube-DNN双塔详解.md) | RecSys 2016 候选生成 = Two-Tower；负采样与精排分家 |

## 合写型专题（2026 补齐）

| 文档 | 覆盖内容 |
|------|---------|
| [LLM-Embedding冲榜路线_E5Mistral-NVEmbed-GritLM-SFR-Arctic-Stella.md](LLM-Embedding冲榜路线_E5Mistral-NVEmbed-GritLM-SFR-Arctic-Stella.md) | E5-Mistral / GritLM / NV-Embed-v2 / SFR-Embedding-2R / Arctic-Embed v1&2 / Stella-Jasper 六篇合写；6 个可插拔积木 + 消融交集 |
| [MLLM通用多模态嵌入_GME-VLM2Vec-BGEVL.md](MLLM通用多模态嵌入_GME-VLM2Vec-BGEVL.md) | GME [2412.16855](https://arxiv.org/abs/2412.16855) + VLM2Vec+MMEB [2410.05160](https://arxiv.org/abs/2410.05160) + BGE-VL/MegaPairs [2412.14475](https://arxiv.org/abs/2412.14475) 三篇合写 |
| [前沿短深读_LateChunking-Vec2Vec-ModernBERT-DINOv2v3-Qwen3-Seed-ViDoRev2.md](前沿短深读_LateChunking-Vec2Vec-ModernBERT-DINOv2v3-Qwen3-Seed-ViDoRev2.md) | Late Chunking [2409.04701](https://arxiv.org/abs/2409.04701) / Vec2Vec [2505.12540](https://arxiv.org/abs/2505.12540) / ModernBERT [2412.13663](https://arxiv.org/abs/2412.13663) / DINOv2 [2304.07193](https://arxiv.org/abs/2304.07193)+DINOv3 [2508.10104](https://arxiv.org/abs/2508.10104) / Qwen3-Embedding [2506.05176](https://arxiv.org/abs/2506.05176) / Seed1.5-Embedding / ViDoRe v2 [2505.17166](https://arxiv.org/abs/2505.17166) 七个方向合写 |

## 数据工程

| 文档 | 论文 / 来源 |
|------|-------------|
| [Doc2Query详解.md](Doc2Query/Doc2Query详解.md) | [arXiv:1904.08375](https://arxiv.org/abs/1904.08375)（预测查询拼回文档，再 BM25） |
| [Doc2Query++详解.md](Doc2Query++/Doc2Query++详解.md) | [arXiv:2510.09557](https://arxiv.org/abs/2510.09557)（主题覆盖；稠密双索引融合） |
| [GPL详解.md](GPL/GPL详解.md) | [arXiv:2112.07577](https://arxiv.org/abs/2112.07577)（造查询 + cross-encoder margin 伪标签） |
| [R-GPL详解.md](R-GPL/R-GPL详解.md) | [arXiv:2501.14434](https://arxiv.org/abs/2501.14434)（蒸馏中重挖难负例） |
| [InPars 论文精炼版](InPars/InPars详解.md) · [补充解读](InPars/InPars解读.md) · [全文中英对照](InPars/zh.md) | [arXiv:2202.05144](https://arxiv.org/abs/2202.05144)（少样本 LLM 造查询，训 monoT5） |
| [InPars-v2 论文精炼版](InPars-v2/InPars-v2详解.md) · [补充解读](InPars-v2/InPars-v2解读.md) · [全文中英对照](InPars-v2/zh.md) | [arXiv:2301.01820](https://arxiv.org/abs/2301.01820)（GPT-J + monoT5 过滤） |
| [Promptagator详解.md](Promptagator/Promptagator详解.md) | [arXiv:2209.11755](https://arxiv.org/abs/2209.11755)（≤8 例；一致性过滤后训双塔） |
| [Longtail-Product-Retrieval详解.md](Longtail-Product-Retrieval/Longtail-Product-Retrieval详解.md) | [arXiv:2602.23620](https://arxiv.org/abs/2602.23620)（离线多奖励改写蒸馏进商品检索） |
| [E5-Mistral合成数据详解.md](E5-Mistral/E5-Mistral合成数据详解.md) | [arXiv:2401.00368](https://arxiv.org/abs/2401.00368)（两步 prompt；全文中译见同目录 zh.md） |

## Late Interaction 族（文本 → 视觉文档）

| 文档 | 论文 / 来源 |
|------|-------------|
| [ColBERT详解.md](ColBERT/ColBERT详解.md) | [arXiv:2004.12832](https://arxiv.org/abs/2004.12832) |
| [ColBERTv2详解.md](ColBERTv2/ColBERTv2详解.md) | [arXiv:2112.01488](https://arxiv.org/abs/2112.01488) |
| [PLAID详解.md](PLAID/PLAID详解.md) | [arXiv:2205.09707](https://arxiv.org/abs/2205.09707)（阅读笔记：背景、质心交互与剪枝、四阶段流程、完整实验表和消融；8 张原图，中英翻译独立保留） |
| [ColPali详解.md](ColPali/ColPali详解.md) | [arXiv:2407.01449](https://arxiv.org/abs/2407.01449) |
| [ColQwen系列详解.md](ColQwen/ColQwen系列详解.md) | 同 ColPali 论文 + Vidore 模型卡（ColQwen2 / 2.5；社区 ColQwen3） |

## 稠密文本 Embedding 深读

| 文档 | 论文 / 来源 |
|------|-------------|
| [DPR详解.md](DPR/DPR详解.md) | [EMNLP 2020](https://aclanthology.org/2020.emnlp-main.550/)；双编码器、同批负例与 BM25 难负例，连接开放域问答和 RAG |
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
| [Jina-embeddings-v5-text详解.md](Jina/v5-text/Jina-embeddings-v5-text详解.md) | [arXiv:2602.15547v2](https://arxiv.org/abs/2602.15547v2)（学生空间蒸馏、四任务 LoRA、GOR 与二值化消融；附方法原图） |
| [Jina-embeddings-v5-omni详解.md](Jina/v5-omni/Jina-embeddings-v5-omni详解.md) | [arXiv:2605.08384](https://arxiv.org/abs/2605.08384) |
| [jina-reranker-v3详解.md](Jina/rerank-v3/jina-reranker-v3详解.md) | [arXiv:2509.25085](https://arxiv.org/abs/2509.25085)（LBNL 列表精排；共享因果上下文读出） |
| [Jina-ColBERT-v2详解.md](Jina/colbert-v2/Jina-ColBERT-v2详解.md) | [arXiv:2408.16672](https://arxiv.org/abs/2408.16672)（多语迟交互；多线性头 MRL） |
| [jina-reranker-v3.5详解.md](Jina/rerank-v3.5/jina-reranker-v3.5详解.md) | [arXiv:2607.18152](https://arxiv.org/abs/2607.18152)（LBNL + 3L2G 混合注意力 + 同尺寸自蒸馏） |

## 关联报告

| 文档 | 说明 |
|------|------|
| [../lora/README.md](../lora/README.md) | LoRA / PEFT（Jina v3 任务 LoRA、cloud_emb stage2 等） |
| [../distillation/知识蒸馏技术深度调研报告.md](../distillation/知识蒸馏技术深度调研报告.md) | 蒸馏总论 |

工程代码与训练数据在 **modelforge**：`scripts/embedding/`、`datasets/embedding/`、`tasks/cloud_emb/`。
