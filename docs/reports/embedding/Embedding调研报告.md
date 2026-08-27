# Embedding 调研报告

> **版本**: v1.2
> **日期**: 2026-07-20
> **范围**: 词/句/文档/多模态 Embedding 的发展历程、理论基础、训练方法、应用场景、评测体系与业界模型全景对比
> **本版增量**: §7–14 质量向重写——RAG 失败归因、Hybrid/RRF 机制、机制族谱替排行榜、多模态四类对齐图文综述、图搜图/文搜图、蒸馏版图、部署闭环、任务→表示决策

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [发展历程：从 BoW 到 LLM Embedding](#2-发展历程从-bow-到-llm-embedding)
   - 2.1 [完整时间线与 Model Card](#21-完整时间线)（含 BGE / Jina / Seed 等系列）
3. [理论基础与表示范式](#3-理论基础与表示范式)
   - 含：池化/读出、InfoNCE、三范式、Bi/Cross/Late、骨干≠交互
4. [架构分类体系](#4-架构分类体系)（四轴坐标）
5. [训练方法与数据工程](#5-训练方法与数据工程)
   - 含：损失选对、难负例/假负例、课表与判据
6. [评测体系与 Benchmark](#6-评测体系与-benchmark)
7. [应用场景全景](#7-应用场景全景) — 任务×表示×失败模式；RAG 归因
8. [检索 Pipeline 工程实践](#8-检索-pipeline-工程实践) — Hybrid/RRF/ANN/延迟
9. [文本 Embedding：机制族谱](#9-文本-embedding机制族谱非排行榜)
10. [多模态与专用 Embedding](#10-多模态与专用-embedding) — 四类路线；图搜图/文搜图
11. [Embedding 蒸馏与压缩](#11-embedding-蒸馏与压缩) — 版图（细节见专题）
12. [向量数据库与部署](#12-向量数据库与部署)
13. [前沿方向与开放问题](#13-前沿方向与开放问题)
14. [实践选型指南](#14-实践选型指南) — 任务→表示→交互
15. [参考文献](#15-参考文献)

---

## 1. 执行摘要

**Embedding（嵌入）** 是将离散符号（词、句、文档、图像）映射到连续向量空间的技术，使语义相近的对象在空间中距离更近。它是现代 NLP、信息检索、推荐系统与 RAG 的**语义基础设施**。

### 1.1 三代范式演进

| 代际                  | 时期       | 代表                               | 核心特征              | 局限                 |
| --------------------- | ---------- | ---------------------------------- | --------------------- | -------------------- |
| **稀疏统计**    | 1954–2012 | BoW, TF-IDF, LSA                   | 高维稀疏、可解释      | 无语义、维度爆炸     |
| **静态稠密**    | 2013–2018 | Word2Vec, GloVe, FastText          | 低维稠密、语义相似    | 一词一向量，无上下文 |
| **上下文/序列** | 2018–2022 | ELMo, BERT, SBERT                  | 动态上下文表征        | Encoder 规模受限     |
| **LLM 骨干**    | 2023–     | E5, BGE, NV-Embed, Qwen3-Emb       | 生成式 LLM + 对比学习 | 算力/部署成本高      |
| **统一多模态**  | 2024–     | CLIP, SigLIP 2, ColPali, Cohere v4 | 跨模态统一空间        | 模态对齐复杂         |

### 1.2 2026 年关键结论

1. **LLM 作 Embedding 骨干已成主流**：Qwen3-Embedding-8B（MTEB 70.6）、NV-Embed-v2（72.31）超越多数 API 模型
2. **生产 RAG 不只看 MTEB**：BGE-M3 以 568M 参数 + 三功能统一（Dense/Sparse/ColBERT）成为工程首选（机制深读见专题报告《[BGE-M3三功能统一详解报告](BGE-M3三功能统一详解报告.md)》）
3. **Hybrid 检索是标配**：Dense + BM25/Sparse + Reranker 三段式 Pipeline
4. **MRL 维度自适应**：OpenAI v3、Nomic、Jasper 均支持推理时截断维度
5. **多模态 Embedding 爆发**：ColPali 绕过 OCR，Cohere Embed v4 原生图文统一

---

## 2. 发展历程：从 BoW 到 LLM Embedding

### 2.1 完整时间线

本节先给**总览时间轴**，再按时代展开 **model card**（架构 / 模态 / 参数量级 / 维度 / 上下文 / 交互范式 / 要点）。
「BERT 之后」是重点：从句向量、Dense Retrieval，到指令化、LLM 骨干、系列化产品线（BGE / Jina / GTE / E5…）与多模态通用嵌入（Seed / Qwen3-VL-Emb / ColPali…）。表中参数/分数为公开文档与论文报道的**量级参考**，以厂商更新为准。

#### 2.1.1 总览时间轴


##### 1954 统计与稀疏检索

- **Harris 分布语义学** — 分布语义学理论奠基；词义即上下文共现分布

##### 1975

- **LSA / LSI** — SVD 分解共现矩阵 → 100–300 维稠密语义空间

##### 1985

- **WordNet** — 手工语义网络；共现统计与词典资源并行发展

##### 1994

- **BM25** — Okapi 概率检索；至今 Hybrid / RRF 稀疏腿基线

##### 2003

- **11 · Bengio 神经语言模型** — 连续词表示思想原型；NeurIPS 2003

##### 2013 静态词向量

- **10 · Word2Vec ★** — CBOW / Skip-gram + 负采样；类比运算里程碑

##### 2014

- **01 · GloVe** — 全局共现统计 + 最小二乘；与 Word2Vec 互补

##### 2016

- **07 · FastText** — 子词 n-gram；OOV 与形态丰富语言更稳

##### 2017 Transformer 底座

- **06 · Transformer** — Self-Attention 统一序列建模；后续 Embedding 共同底座

##### 2018 语境表示与预训练 Encoder

- **03 · ELMo** — 双向 LSTM 层加权；一词多义破冰
- **03 · Universal Sentence Encoder** — Transformer / DAN 通用句嵌入；工业部署开端
- **10 · BERT ★** — MLM+NSP；预训练→微调范式（原生非句向量）

##### 2019 句向量产业化

- **08 · Sentence-BERT ★** — Siamese BERT + pooling；句相似度速度 ×1000+
- **10 · DistilBERT** — 6 层蒸馏 BERT；Embedding 侧蒸馏早期实践
- **12 · USE-T5** — T5 Encoder 系句向量变体

##### 2020 Dense Retrieval 标准化

- **02 · REALM** — 检索增强预训练；检索进预训练循环
- **04 · DPR ★** — 双 BERT Q/D 塔 + FAISS；段落稠密检索范式
- **04 · ColBERT v1** — 每 token 向量 + MaxSim；Late Interaction 路线开创
- **05 · SPECTER** — SciBERT + 引用图对比；科学文献检索专用

##### 2021 对比学习与跨模态双塔

- **01 · ALIGN** — Google；十亿级噪声图文对；弱标注规模红利
- **02 · CLIP ★** — OpenAI ViT + 文本塔；零样本跨模态检索/分类
- **04 · SimCSE** — Dropout 自监督对比；无监督句向量里程碑
- **08 · ANCE** — RoBERTa + ANN 异步 hard negative 刷新
- **11 · Contriever** — Meta 无标注对比；无监督稠密检索
- **12 · ColBERT v2** — 蒸馏 + 索引压缩；Late Interaction 工程可用性上升

##### 2022 指令化、稀疏与 Matryoshka

- **02 · DiffCSE** — ELECTRA 式差异对比；无监督句向量增强
- **02 · Instructor** — 330+ 任务指令前缀；「Instruct: …」范式先导
- **02 · MRL** — Matryoshka 一向量多尺度（64…4096 可截断）
- **03 · SPLADE v2** — BERT → 词表稀疏权重；可学习稀疏检索 + Hybrid
- **05 · GTR** — Google T5 Encoder 检索族；Encoder-Decoder 做 Bi-Encoder
- **09 · RetroMAE** — 编码器-解码重建预训练；BGE 系 Stage-0 组件
- **10 · E5-base / E5-large** — MS 弱监督十亿对 → 监督微调流水线定型
- **12 · OpenAI ada-002** — 8K API Embedding；长上下文闭源事实标准
- **12 · SGPT** — GPT 系 last-token；证明 Decoder-only 可做句向量

##### 2023 开源系与商业 API 齐发

- **03 · Cohere embed-v3** — `search_query` / `search_document` 输入类型分化
- **06 · Voyage-lite** — 检索向优化 API；后有 voyage-code 分支
- **07 · GTE-base / GTE-large** — 阿里多阶段对比；后续 gte-Qwen2 切 LLM 骨干
- **08 · BGE v1** — BAAI FlagEmbedding 中英尺寸档开源
- **08 · bge-reranker** — Cross-Encoder 精排；与 BGE 嵌入配套
- **09 · BGE v1.5** — 相似度分布校准；HF 高下载工程默认档
- **10 · jina-embeddings-v2** — 8K 长上下文；en/de/es/zh 分语种卡
- **12 · E5-mistral-7b-instruct** — Mistral-7B + 指令；LLM 骨干嵌入早期标杆

##### 2024 三功能统一与 LLM 嵌入

- **01 · OpenAI embedding-3 ★** — small/large + MRL `dimensions` API；性价比跃升
- **01 · BGE-M3 ★** — Dense + Sparse + ColBERT；8K·100+ 语三功能统一
- **01 · GritLM-7B** — Mistral 统一生成 + 表示；同一模型双用途
- **02 · LLM2Vec** — 改 LLM→双向 + 对比；任意 Decoder 变编码器配方
- **02 · Nomic Embed Text v1.5** — 8K + MRL 开放权重；配套 Vision 分支
- **03 · BGE-ICL** — LLM + few-shot 示例；提示内塞例提升检索
- **04 · Snowflake Arctic Embed** — 小参数检索强；Apache-2.0 工程友好
- **04 · Stella / Jasper** — 多 Teacher 蒸馏 + MRL；社区小尺寸高分范式
- **04 · bge-reranker-v2** — 多语 / LLM Cross-Encoder；M3 精排配套
- **06 · NV-Embed-v1** — Mistral-7B 双向 + Latent Attention pooling
- **06 · bge-multilingual-gemma2** — Gemma-2-9B 多语 LLM 嵌入大参数档
- **06 · gte-Qwen2-1.5B / 7B** — Qwen2 Decoder 骨干；32K 开源中文/多语强势
- **07 · NV-Embed-v2** — Hard negative 工程强化；英 MTEB 领先档
- **07 · jina-clip-v2** — EVA02 + Jina-XLM-R；多语 CLIP-style 双塔
- **08 · Cohere embed-v3.5** — 商业 API 多语检索迭代
- **09 · jina-embeddings-v3** — 570M；retrieval / cls / cluster 任务 LoRA 适配器
- **09 · Voyage-3** — 检索向商业 API；large / lite 分支

##### 2025 MLLM 通用嵌入与文档多向量

- **01 · ColPali ★** — PaliGemma-3B + MaxSim；免 OCR 页级文档检索（ICLR）
- **02 · SigLIP 2** — Sigmoid loss + 自蒸馏 + 多语；常作 VLM 视觉塔
- **02 · VLM2Vec** — MLLM 对比训练框架；MMEB 基准路线开创
- **03 · ColQwen2** — Qwen2-VL + MaxSim；ColPali 族开源强跟随
- **03 · GME** — Qwen2-VL 通用多模态嵌入；图搜图 / 文搜图
- **03 · KaLM** — 小 LLM / EuroBERT 系；与 jina-v5-nano 同档竞争
- **04 · BGE-VL** — CLIP / LLaVA-NeXT 系；视觉检索、截图搜
- **04 · Conan-embedding-v2** — 中文向强检索；动态 hard negative
- **05 · BGE-Code-v1** — 代码专用骨干；Code IR / CoIR 向
- **05 · Cohere Embed v4** — 企业多模态 API；原生图+文统一 Transformer
- **06 · Seed-1.6-Embedding ★** — 文+图+视频混合检索；C-MTEB / MMEB-v2 领先档
- **06 · jina-embeddings-v4** — Qwen2.5-VL-3B；32K 多模态 + 任务 LoRA
- **06 · Qwen3-Embedding ★** — 0.6B / 4B / 8B；32K·MRL·指令前缀；开源综合榜前列
- **07 · Qwen3-VL-Embedding** — 通义多模态嵌入产品线；图文/视频文档
- **10 · Gemini Embedding** — Google 闭源检索向 API；生态绑定

##### 2026 小模型蒸馏与全模态 Omni

- **01 · Gemini Embedding 2** — Google 多模态检索 API 迭代
- **02 · jina-embeddings-v5 ★** — text-nano/small + v5-omni；蒸馏 Qwen3-Emb；Locked Aligned Towers
- **03 · Voyage 4** — 商业 API MoE 分支；长上下文检索向

#### 2.1.2 前深度学习与静态词向量（简卡）

| 模型 / 方法  | 年份       | 架构要点                       | 模态     | 典型维度   | 备注                   |
| ------------ | ---------- | ------------------------------ | -------- | ---------- | ---------------------- |
| BoW / TF-IDF | 1950s–    | 计数 / 加权                    | 文本     | \|V\| 稀疏 | 仍作 Hybrid 组件       |
| LSA / LSI    | 1975 / 90s | SVD 降维                       | 文本     | ~100–300  | 早期「语义空间」       |
| Word2Vec     | 2013       | CBOW / Skip-gram + NegSampling | 词       | 100–300   | 类比运算里程碑         |
| GloVe        | 2014       | 全局共现 + 最小二乘            | 词       | 50–300    | 统计 + 神经折中        |
| FastText     | 2016       | 子词 n-gram                    | 词       | 300        | OOV / 形态丰富语言更强 |
| ELMo         | 2018       | 双向 LSTM 层加权               | 上下文词 | ~1024      | 一词多义破冰           |
| USE          | 2018       | Transformer / DAN              | 句       | 512        | 通用句嵌入工业开端     |

#### 2.1.3 BERT 之后：句向量与 Dense Retrieval（2019–2021）

| 模型                    | 年份  | 骨干 / 架构                   | 模态            | 参数量级               | 维度          | 上下文         | 交互范式          | Model Card 要点                                                   |
| ----------------------- | ----- | ----------------------------- | --------------- | ---------------------- | ------------- | -------------- | ----------------- | ----------------------------------------------------------------- |
| **BERT**          | 2019  | Encoder Transformer；MLM+NSP  | 文本            | Base 110M / Large 340M | 隐层 768/1024 | 512            | —（预训练底座）  | 开启「预训练→微调」；原生非句向量，靠`[CLS]`/mean 勉强抽句向量 |
| **Sentence-BERT** | 2019  | Siamese BERT + pooling        | 文本            | ≈BERT                 | 768/1024      | 512            | Bi-Encoder        | NLI/三元组；句相似度速度 ×1000+；专用 Embedding 产业起点         |
| **DistilBERT**    | 2019  | 6 层学生蒸馏 BERT             | 文本            | ~66M                   | 768           | 512            | Encoder / 可做 Bi | 早期大规模 Embedding 侧蒸馏实践                                   |
| **DPR**           | 2020  | 双 BERT（Q/D 塔）             | 文本（Passage） | ~2×BERT-base          | 768           | 512            | Bi-Encoder        | Facebook；FAISS 稠密段落检索标准化                                |
| **REALM**         | 2020  | 检索增强预训练                | 文本            | —                     | —            | —             | Bi + 检索         | 检索进预训练循环                                                  |
| **ColBERT (v1)**  | 2020  | BERT + 每 token 向量          | 文本            | ~BERT                  | token×128    | 512            | Late Interaction  | MaxSim；精度/速度折中路线开创                                     |
| **SPECTER**       | 2020  | SciBERT + 引用图对比          | 科学文本        | ~110M                  | 768           | 512            | Bi-Encoder        | 论文检索 / 推荐专用                                               |
| **CLIP**          | 2021  | ViT + 文本 Transformer 双塔   | 图+文           | ViT-L/14 ≈400M 级     | 512/768       | 文本 77 tok 级 | Dual-/Bi-Encoder  | OpenAI；对比学习跨模态对齐；零样本分类                            |
| **ALIGN**         | 2021  | EfficientNet + BERT           | 图+文           | 十亿级噪声对           | —            | —             | Dual-Encoder      | Google；弱标注图文规模红利                                        |
| **SimCSE**        | 2021  | BERT/RoBERTa + Dropout 自监督 | 文本            | ≈BERT                 | 768           | 512            | Bi-Encoder        | 无监督对比学习句向量里程碑                                        |
| **Contriever**    | 2021  | BERT 无监督对比               | 文本            | ~110M                  | 768           | 512            | Bi-Encoder        | Meta；无标注稠密检索                                              |
| **ANCE**          | 2021  | RoBERTa + ANN 动态负例        | 文本            | ~125M                  | 768           | 512            | Bi-Encoder        | 异步 hard negative 更新                                           |
| **ColBERT v2**    | 2021  | BERT 多向量 + 蒸馏            | 文本            | ~110M                  | token×128    | 512            | Late Interaction  | 压缩索引 + 教师蒸馏，工程可用性上升                               |
| **RetroMAE**      | 2022* | 编码器-解码重建预训练         | 文本            | Encoder 级             | —            | 512+           | 预训练底座        | BGE 系 Stage-0；提升表示质量                                      |

\*RetroMAE 论文 2022，作为后续 BGE 预训练组件收录于此。

#### 2.1.4 指令化、稀疏与多维自适应（2022）

| 模型                    | 年份     | 骨干 / 架构              | 模态             | 参数量级                | 维度       | 上下文 | 交互范式                   | Model Card 要点                                 |
| ----------------------- | -------- | ------------------------ | ---------------- | ----------------------- | ---------- | ------ | -------------------------- | ----------------------------------------------- |
| **Instructor**    | 2022     | GTR/T5 系 + 任务指令前缀 | 文本             | 中等 Encoder            | 768        | 512    | Bi-Encoder                 | 330+ 任务统一指令嵌入；「Instruct: …」范式先导 |
| **GTR**           | 2021–22 | T5 Encoder               | 文本             | Base→XXL               | 768        | 512    | Bi-Encoder                 | Google；Encoder-Decoder 家族的检索变体          |
| **SGPT**          | 2022     | GPT 系 last-token        | 文本             | 125M–5.8B              | 随模型     | 2K 级  | Bi-Encoder（Decoder 骨干） | 证明 Decoder-only 也可做句向量                  |
| **SPLADE v2**     | 2022     | BERT → 词表稀疏权重     | 文本             | ~BERT                   | \|V\| 稀疏 | 512    | Lexical/Sparse             | 可学习稀疏检索；与 Dense Hybrid                 |
| **MRL**           | 2022     | 训练框架（非单模型）     | 任意             | —                      | 可截断     | —     | —                         | Matryoshka；一向量多尺度（64…4096）            |
| **DiffCSE**       | 2022     | ELECTRA 式差异对比       | 文本             | ≈BERT                  | 768        | 512    | Bi-Encoder                 | 无监督句向量增强变体                            |
| **E5-base/large** | 2022–23 | BERT/XLM-R               | 文本（多语大版） | Base~110M / Large~330M | 768/1024   | 512    | Bi-Encoder                 | MS：「弱监督十亿对 → 监督微调」流水线定型      |

#### 2.1.5 2023：开源系与商业 API 同时爆发

| 模型                                               | 年份       | 骨干 / 架构             | 模态                  | 参数量级                | 维度                | 上下文        | 交互范式                | Model Card 要点                                   |
| -------------------------------------------------- | ---------- | ----------------------- | --------------------- | ----------------------- | ------------------- | ------------- | ----------------------- | ------------------------------------------------- |
| **OpenAI text-embedding-ada-002**            | 2022末–23 | 未公开 Decoder 系       | 文本                  | API                     | 1536                | 8K            | Bi-Encoder              | 长上下文 API Embedding 事实标准                   |
| **OpenAI text-embedding-3-small/large**      | 2024.01*   | 未公开；支持 MRL        | 文本                  | API                     | 1536 / 3072（可截） | 8K            | Bi-Encoder              | 性价比与多语显著提升；`dimensions` API          |
| **Cohere embed-english / multilingual-v3.0** | 2023       | 私有 Encoder            | 文本                  | API                     | 1024                | 512           | Bi-Encoder              | 输入类型`search_query` / `search_document`    |
| **E5-mistral-7b-instruct**                   | 2023       | Mistral-7B + 指令       | 文本                  | 7B                      | 4096                | 4K–32K 级    | Bi-Encoder（LLM）       | LLM 骨干嵌入早期标杆；Query 侧加 instruct         |
| **GritLM-7B**                                | 2024*      | Mistral + 统一 Gen/Emb  | 文本                  | 7B                      | 4096                | 4K+           | Bi / 生成一体           | 「同一模型生成+表示」                             |
| **GTE-large / GTE-base**（Alibaba）          | 2023–24   | BERT / RoBERTa 系       | 文本                  | Base~0.1B / Large~0.3B | 768–1024           | 512→8K(v1.5) | Bi-Encoder              | 多阶段对比；后续**gte-Qwen2** 切入 LLM 骨干 |
| **gte-Qwen2-1.5B/7B-instruct**               | 2024       | Qwen2 Decoder           | 文本                  | 1.5B / 7B               | 1536+               | 32K           | Bi-Encoder              | 开源中文/多语强势                                 |
| **Nomic Embed Text v1 / v1.5**               | 2024       | 自定义 Encoder + MRL    | 文本                  | ~137M                   | 768（可截）         | 8K            | Bi-Encoder              | 长上下文开放权重；配套 Vision                     |
| **Snowflake arctic-embed-m/l**               | 2024       | BERT 系压缩训练         | 文本                  | M~110M / L~330M        | 768/1024            | 512           | Bi-Encoder              | 小参数检索强；Apache-2.0                          |
| **Voyage-2 / voyage-large-2**                | 2023–24   | API 专有                | 文本 / 代码分支       | API                     | 1024–1536          | 4K–16K       | Bi-Encoder              | 检索向优化；后有 voyage-code                      |
| **jina-embeddings-v2-base-***                | 2023–24   | JinaBERT / RoBERTa 变体 | 文本（en/de/es/zh…） | ~137–161M              | 768                 | **8K**  | Bi-Encoder              | 早期主打长上下文与分语言版                        |
| **LLM2Vec**                                  | 2024       | 改 LLM→双向 + 对比     | 文本                  | 随底座                  | 随底座              | 随底座        | Bi-Encoder              | 把任意 Decoder LLM 改造成编码器配方               |
| **UAE-Large-V1 / mxbai-embed**               | 2023–24   | BERT-large 系           | 文本                  | ~335M                   | 1024                | 512           | Bi-Encoder              | 社区蒸馏 / AnglE 损失路线                         |
| **bge-reranker-base/large**                  | 2023       | Cross-Encoder           | 文本                  | Base~0.1B / Large~0.5B | 标量分数            | 512           | **Cross-Encoder** | 与 BGE 嵌入配套的精排                             |

\*embedding-3 虽挂 2024 发布，常与 2023 ada 序列一起被产业对照，故列此表便于选型。

#### 2.1.6 BGE 全家桶（勿只记 M3）

BGE（BAAI General Embedding / FlagEmbedding）是**系列化**产品：中英尺寸档、多语三功能、ICL、多模态、代码、重排器齐全。

| 模型                                       | 发布    | 骨干                       | 模态          | 参数              | 维度         | 上下文       | 范式                 | 要点                                                                                                  |
| ------------------------------------------ | ------- | -------------------------- | ------------- | ----------------- | ------------ | ------------ | -------------------- | ----------------------------------------------------------------------------------------------------- |
| **bge-small/base/large-zh-v1.5**     | 2023    | BERT/RoBERTa 中文          | 文本-中       | 24M / 102M / 326M | 512/768/1024 | 512          | Bi                   | 中文检索工程默认档之一                                                                                |
| **bge-small/base/large-en-v1.5**     | 2023    | BERT 英                    | 文本-英       | 33M / 109M / 335M | 384/768/1024 | 512          | Bi                   | 相似度分布校准；HF 高下载                                                                             |
| **bge-m3**                           | 2024.01 | XLM-R 扩展                 | 文本·100+ 语 | **568M**    | 1024         | **8K** | Dense+Sparse+ColBERT | ★ Multi-Function / Lingual / Granularity；详见《[BGE-M3三功能统一详解](BGE-M3三功能统一详解报告.md)》 |
| **bge-m3-retromae / unsupervised**   | 2024    | 同系中间 ckpt              | 文本          | —                | —           | 8K           | 预训练/对比中间态    | 复现与二阶段训练用                                                                                    |
| **bge-multilingual-gemma2**          | 2024.06 | Gemma-2-9B                 | 文本·多语    | ~9B               | 3584         | 4K+          | Bi（LLM）            | 大参数多语开源档                                                                                      |
| **bge-en-icl**                       | 2024    | LLM + few-shot 示例        | 文本-英       | LLM 级            | 高维         | 长           | Bi + ICL             | 提示中塞示例提升检索                                                                                  |
| **bge-reranker-v2-m3**               | 2024    | 轻量 Cross / miniLM 变体等 | 文本·多语    | ~0.1–0.6B        | 分数         | 8K           | Cross                | 多语精排配套                                                                                          |
| **bge-reranker-v2-gemma / minicpm**  | 2024    | LLM Cross-Encoder          | 文本          | 2B–9B 级         | 分数         | 长           | Cross                | 高精度重排                                                                                            |
| **BGE-VL**（bge-vl-base / llava 系） | 2025    | CLIP / LLaVA-NeXT          | 图+文         | 0.4B–8B 级       | —           | —           | Dual / MLLM-Emb      | 视觉检索、截图搜                                                                                      |
| **BGE-Code-v1**                      | 2025    | 代码专用骨干               | 代码+文       | —                | —           | —           | Bi                   | Code IR / CoIR 向                                                                                     |

> **选型提示**：短中文句对优先 `bge-*-zh-v1.5`；多语长文 + Hybrid → `bge-m3`；要 ICL/指令 → `bge-en-icl` / LLM 档；图文 → `BGE-VL`，不要用纯文本 M3 硬撑跨模态。
>
> **专题深读**：若只熟悉 BERT / 稠密句向量、尚未建立 Sparse 直觉，请先读仓库专题《[BGE-M3三功能统一详解报告](BGE-M3三功能统一详解报告.md)》（含 Sparse 从零讲解、三 Head 公式、自知识蒸馏与 Hybrid 工程用法）。

#### 2.1.7 Jina Embeddings 迭代链（v2 → v5，含 CLIP / Omni）

| 模型                                            | 发布     | 骨干                      | 模态                  | 参数                                        | 维度（MRL）           | 上下文        | 范式                       | 要点                                                           |
| ----------------------------------------------- | -------- | ------------------------- | --------------------- | ------------------------------------------- | --------------------- | ------------- | -------------------------- | -------------------------------------------------------------- |
| **jina-embeddings-v2-base-en/de/es/zh…** | 2023–24 | JinaBERT                  | 文本（分语言）        | ~137–161M                                  | 768                   | 8K            | Bi                         | 长上下文 + 分语种卡                                            |
| **jina-clip-v1 / v2**                     | 2024     | EVA02 + Jina-XLM-R        | 图+文                 | v2≈0.87B                                   | 1024（可截）          | 文本长 / 图像 | Dual-Encoder               | 标准 CLIP-style；多语+ MRL                                     |
| **jina-embeddings-v3**                    | 2024.09  | jina-XLM-RoBERTa-24L      | 文本·89 语           | **570M**                              | 1024→32              | **8K**  | Bi +**任务 LoRA**    | retrieval / classification / clustering / text-matching 适配器 |
| **jina-embeddings-v4**                    | 2025.06  | **Qwen2.5-VL-3B**   | 文+图+PDF             | **3.8B**                              | 2048（亦可多向量）    | **32K** | Bi + Late Interaction 可选 | 统一多模态；task LoRA；Qwen Research License                   |
| **jina-embeddings-v5-text-nano**          | 2026.02  | EuroBERT-210M             | 文本·多语            | **239M**                              | 768→32               | 8K            | Bi + 任务适配              | 蒸馏自 Qwen3-Embedding-4B；小模型 SOTA 档                      |
| **jina-embeddings-v5-text-small**         | 2026.02  | **Qwen3-0.6B-Base** | 文本·119+ 语         | **677M**                              | 1024→32              | **32K** | Bi + 任务适配              | MTEB/MMTEB 小模型顶尖；last-token；Query/Doc 前缀              |
| **jina-embeddings-v5-omni-nano/small**    | 2026     | 锁定文本塔 + 视听投影     | **文/图/视/音** | nano≈文塔+编码器；small≈**1.7B** 级 | 与 text 对齐 768/1024 | 32K           | Locked Aligned Towers      | 文本索引、任意模态查询；几何保持对齐（arXiv:2605.08384）       |

> Jina 线索：**v2 长上下文 → v3 任务 LoRA → v4 MLLM 多模态 → v5 蒸馏小钢炮 + Omni 全模态**。选型时勿把 `jina-clip-v2`（双塔①类）与 `jina-embeddings-v4/v5-omni`（MLLM/后交互）混为一谈。

#### 2.1.8 2024–2026：LLM / MLLM 通用嵌入与 API、文档检索

| 模型                                                               | 年份              | 骨干 / 架构                                       | 模态                          | 参数量级        | 维度                  | 上下文                   | 交互范式         | Model Card 要点                                                                                           |
| ------------------------------------------------------------------ | ----------------- | ------------------------------------------------- | ----------------------------- | --------------- | --------------------- | ------------------------ | ---------------- | --------------------------------------------------------------------------------------------------------- |
| **NV-Embed-v1 / v2**                                         | 2024              | Mistral-7B；双向注意力 + Latent Attention pooling | 文本                          | 7B              | 4096                  | 512–4K+                 | Bi（LLM）        | v2 曾英 MTEB 领先（~72.31）；Hard Negative 工程强                                                         |
| **Stella / Jasper**                                          | 2024              | 多 Teacher 蒸馏 + MRL                             | 文本                          | 0.4B–1.5B 级   | 可变                  | 8K                       | Bi               | 小尺寸高分；社区蒸馏范式                                                                                  |
| **Snowflake arctic-embed-m-v1.5 等**                         | 2024–25          | 持续小模型迭代                                    | 文本                          | ≤330M          | 768+                  | 512                      | Bi               | 检索专用小模型持续刷分                                                                                    |
| **Voyage-3 / 3-large / 4**                                   | 2024–26          | API；部分 MoE                                     | 文本 / 多模态分支             | API             | 512–2048             | 32K                      | Bi               | 检索向商业 SOTA 竞争者                                                                                    |
| **Cohere Embed v4**                                          | 2025              | 统一 Transformer                                  | **图+文**（原生）       | API             | 1024+                 | 长（官方称 128K 级能力） | Bi / 多模态      | 企业多模态检索 API                                                                                        |
| **SigLIP / SigLIP 2**                                        | 2023 / 2025       | ViT + 文本塔；Sigmoid loss                        | 图+文                         | 视觉塔~0.4–2B  | —                    | —                       | Dual-Encoder     | CLIP 改进；SigLIP2 自蒸馏+多语，常作 VLM 视觉塔                                                           |
| **ColPali**                                                  | 2025 ICLR         | PaliGemma-3B + ColBERT MaxSim                     | **页面图像**            | ~3B             | patch×128            | 页级                     | Late Interaction | 免 OCR 文档页检索；ViDoRe                                                                                 |
| **ColQwen2 / ColQwen3**                                      | 2025              | Qwen2-VL / Qwen2.5-VL + MaxSim                    | 页面图像                      | ~2–3B          | 多向量                | 页级                     | Late Interaction | ColPali 族开源强跟随                                                                                      |
| **VLM2Vec / V2**                                             | 2024–25          | LLaVA / Qwen2-VL 等                               | 图文交错                      | 2B–7B          | —                    | —                       | MLLM Bi-Encoder  | MMEB 路线开创者之一                                                                                       |
| **GME / BGE-VL / E5-V**                                      | 2024–25          | Qwen2-VL / LLaVA-NeXT                             | 图+文                         | 2B–8B          | —                    | —                       | MLLM-Emb         | 通用多模态嵌入族                                                                                          |
| **Qwen3-Embedding-0.6B/4B/8B**                               | 2025              | **Qwen3** Decoder                           | 文本·100+ 语                 | 0.6–8B         | 32–4096 MRL          | 32K                      | Bi（LLM）        | ★ 开源综合榜前列；指令前缀                                                                               |
| **Qwen3-VL-Embedding**                                       | 2025–26          | Qwen3-VL                                          | 图文/视频文档                 | VL 级           | —                    | 长                       | MLLM Bi          | 通义多模态嵌入产品线                                                                                      |
| **Seed-1.5-Embedding**                                       | 2025              | Seed 系                                           | 文为主（及演变）              | API             | —                    | —                       | Bi               | 豆包向量前代；C-MTEB 竞争                                                                                 |
| **Seed-1.6-Embedding**（`doubao-embedding-vision-250615`） | **2025.06** | **Seed1.6-Flash**；双塔取 **[EOS]**   | **文+图+视频** 混合检索 | API（火山引擎） | **2048 / 1024** | 随底座                   | Dual/Bi（MLLM）  | ★ C-MTEB ~75.6 SOTA；MMEB-v2 Image 77.78 / Video 领先；见[官方说明](https://seed1-6-embedding.github.io/) |
| **Conan-embedding-v2**                                       | 2025              | 中文向强检索                                      | 文本-中                       | —              | —                    | —                       | Bi               | C-MTEB 前列；动态 hard negative                                                                           |
| **KaLM / gemma-embedding 等**                                | 2025–26          | 小 LLM / EuroBERT 系                              | 文本                          | <1B             | 可变                  | 8K–32K                  | Bi               | 与 jina-v5-nano 同档位竞赛                                                                                |
| **Gemini Embedding / Embedding 2**                           | 2025–26          | Google 专有                                       | 文本（及多模态延伸）          | API             | —                    | —                       | Bi               | 检索子集高分；生态绑定 Google                                                                             |
| **OpenAI / Google / 平台多模态 Embedding**                   | 2025–26          | 闭源                                              | 图文等                        | API             | 可变                  | 长                       | Bi               | 与开源 MLLM-Emb 并行                                                                                      |

#### 2.1.9 读时间线的正确方式

1. **先扫 §2.1.1**：按年扫一遍定位时代；★ 表分水岭；细节 model card 见 §2.1.2–2.1.8。
2. **系列 ≠ 单点**：谈 BGE 至少区分 v1.5 / M3 / ICL / VL / Code / Reranker；谈 Jina 至少区分 clip-v2 / v3 / v4 / v5-text / v5-omni。
3. **骨干 ≠ 交互**：Qwen3-Embedding、NV-Embed 骨干是 Decoder LLM，交互仍是 **Bi-Encoder**；ColPali 骨干是 VLM，交互是 **Late Interaction**。
4. **API 与开源分工**：OpenAI / Cohere / Voyage / Seed(Doubao) 吃工程便利；BGE / E5 / GTE / Qwen3-Emb / Jina 吃可私有化与可微调。
5. **多模态两派**：CLIP-style 双塔（①）vs MLLM 通用嵌入（③）vs 文档多向量（④）——同一年可并存，见《[图文 Embedding 模型技术综述](图文Embedding模型技术综述.md)》。

### 2.2 各阶段详解（演进因果，而不只是年表）

读本节时盯住一条链：**上一时代解决了什么 → 留下什么硬伤 → 下一时代用什么机制替换**。型号名单仍以 §2.1 的 model card 为准。

#### 2.2.1 稀疏统计时代（1954–2012）

| 方法              | 核心思想           | 向量特点                  | 典型应用         |
| ----------------- | ------------------ | ------------------------- | ---------------- |
| **BoW**     | 词频计数           | 维度 = 词表大小，极度稀疏 | 文本分类基线     |
| **TF-IDF**  | 词频 × 逆文档频率 | 抑制常见词权重            | 搜索引擎、LSH    |
| **LSA/LSI** | SVD 分解共现矩阵   | 约 100–300 维稠密        | 主题、去重       |
| **BM25**    | 概率检索打分       | 非稠密向量                | 至今 Hybrid 基线 |

**动机**：用可计算的词统计近似「相关性」。
**机制贡献**：可解释、可倒排、工程极便宜。
**硬伤**：同义/上下位无法对齐（「糖尿病」≠「血糖」字面）；维度随词表爆炸。
**为何没被淘汰**：精确关键词、SKU、错误码、法规条款号——Dense 至今仍会漏；2026 年生产检索的默认答案是 **Dense + BM25/Sparse**，而不是二选一。

#### 2.2.2 静态词嵌入时代（2013–2018）

分布假说（Harris）被神经方法做成低维稠密：Word2Vec 的 Skip-gram / CBOW + Negative Sampling，把「上下文预测」变成可反向传播的目标。

$$
\mathcal{L}_{\text{SkipGram}} \approx -\sum_{(w,c)}\log \sigma(v_w^\top v_c) - \sum_{(w,c^-)}\log \sigma(-v_w^\top v_{c^-})
$$

| 模型     | 创新                         | 维度     | 训练数据        |
| -------- | ---------------------------- | -------- | --------------- |
| Word2Vec | Skip-gram/CBOW + NegSampling | 100–300 | Google News 等  |
| GloVe    | 全局共现 + 加权最小二乘      | 50–300  | Wiki + Gigaword |
| FastText | 子词 n-gram                  | 300      | Common Crawl    |

**解决了什么**：语义类比、迁移到分类/NER 的初始化、OOV（FastText）缓解。
**硬伤**：一词一向量——「bank」银行/河岸共用一点；句子级只能平均词向量，句法与否定脆弱。
**如何被替代**：上下文模型（ELMo/BERT）让同一词型在不同句子中落到不同点。

#### 2.2.3 上下文嵌入与专用句向量（2018–2022）

**ELMo**：双向 LSTM 层加权 → 动态词向量，证明「上下文条件化」有效。
**BERT**：Encoder + MLM，成为通用编码器底座；但原生 `[CLS]`/mean **不是**为句相似与检索优化的。

**矛盾**：用 BERT 做句对相似度，若走 Cross-Encoder（两句拼进模型），精度高但无法预计算索引；若天真取 `[CLS]` 做 Bi-Encoder，精度塌方。

**Sentence-BERT（2019）** 的因果位置：用 Siamese/Triplet + 对比/回归损失，把 BERT 压成可 ANN 的句向量，推理相对 Cross-Encoder 快数量级，**专用 Embedding 产业从这里起步**。
```

[Sentence A] ── BERT ── Pooling ── u ──┐
                                        ├── CosineSimilarity(u, v)
[Sentence B] ── BERT ── Pooling ── v ──┘

```

随后分叉三条至今仍在用的线：

1. **Dense Retrieval**（DPR, 2020）：双塔 + 标注段落 + in-batch 负例，FAISS 进开放域 QA；
2. **Late Interaction**（ColBERT, 2020）：用多向量换回细粒度交互，精度逼近 CE、索引仍可预计算；
3. **无监督/弱监督对比**（SimCSE, Contriever）：降低对标注三元组的依赖。

**硬伤（进入 2022 前）**：512 上下文、跨语言与指令任务分裂、单向量对专有名词不稳、多模态另起炉灶（CLIP 2021 已开一条平行线）。

#### 2.2.4 指令化、稀疏神经检索与课表化训练（2022–2023）

- **Instructor / E5**：把「任务描述」写进 query 侧，一套权重服务多任务；弱监督十亿对 → 监督微调成为可复制课表。
- **SPLADE**：词法匹配神经化，Dense 丢关键词时的补丁变成可学习 Sparse。
- **MRL**：一向量多截断，部署侧按内存换精度。
- **商业 API**（ada-002 等）把 Embedding 变成基础设施计费项。

**因果**：问题从「有没有句向量」变成「**怎么规模化数据与接口契约**」。

#### 2.2.5 LLM 骨干与系列化产品（2023–）

| 维度   | BERT 系 Embedding         | LLM 系 Embedding               |
| ------ | ------------------------- | ------------------------------ |
| 骨干   | Encoder 110M–560M        | Decoder 0.6B–8B+              |
| 预训练 | MLM                       | CLM（再改编码/双向）           |
| 池化   | CLS / mean                | last token / EOS / latent attn |
| 数据   | 百万～亿对                | 十亿弱监督 + 合成 + 难负例     |
| 指令   | 可选                      | 多数成为契约                   |
| 上下文 | 512 → 8K（部分 Encoder） | 8K–32K 常见                   |

**为何换骨干**：生成式 LLM 的语义与长上下文迁移到表示任务；合成 query 便宜。
**并未改变的事实**：主流仍是 **Bi-Encoder Dense**；LLM 只是更强的 $f_\theta$，不是自动变成 Cross-Encoder。
**系列化**：BGE（v1.5→M3→ICL→VL/Code）、Jina（v2→v5）、GTE→gte-Qwen、Qwen3-Emb、Seed API——竞争从「单点刷榜」转为「产品线 + 可私有化 + 多模态」。

代表机制钩子（名单细节见 §2.1）：E5-mistral / NV-Embed（双向注意力与池化创新）、BGE-M3（三功能统一）、ColPali（视觉文档多向量）、Qwen3 / Seed（开源与 API 两端的多模态嵌入）。

#### 2.2.6 一张因果简图
```

BoW/BM25 ──(语义不足)──► Word2Vec ──(一词多义)──► ELMo/BERT
                              │
                              ▼
                     SBERT/DPR Dense 双塔
                      │           │
         (精度不够)    │           │ (关键词漏)
                      ▼           ▼
                 ColBERT/CE    SPLADE/Hybrid
                      │
         (长上下文/多任务/合成数据)
                      ▼
              Instruct + E5 课表 + LLM 骨干
                      │
         (文档版式 / 图文统一)
                      ▼
         CLIP 双塔 · MLLM-Emb · ColPali 多向量

```

---

## 3. 理论基础与表示范式

本章回答三个问题：**向量空间在优化什么**、**表示长什么样（Dense / Sparse / Multi-Vector）**、**Query 与 Document 何时交互（Bi / Cross / Late）**。后文训练、评测与选型都建立在这三套坐标系上。

### 3.1 Embedding 的数学定义

给定输入空间 $\mathcal{X}$（词、句、段落、图像、页面截图等），参数化映射

$$
f_\theta: \mathcal{X} \rightarrow \mathbb{R}^d, \quad v = f_\theta(x)
$$

把离散对象变成连续向量。写全一点，$f_\theta$ 通常是两段：

1. **骨干**产出 token（或 patch）级隐状态 $H\in\mathbb{R}^{n\times h}$；
2. **池化 / 读出头**把 $H$ 收成固定维 $v\in\mathbb{R}^d$（必要时再投影）。

检索与相似度任务通常在**归一化后的内积空间**上工作：先令 $\hat{v}=v/\|v\|_2$，再定义

$$
\text{sim}(x,y)=\hat{v}_x^\top \hat{v}_y=\cos(\theta_{xy}).
$$

此时余弦与点积数值相同；ANN 的 `IP` 索引与「先 L2 归一化再搜」是同一契约。训练未归一化、评测却用余弦（或反过来），会出现 loss 好看、线上召回崩——这是最常见的工程错位。

#### 3.1.1 池化：从 token 状态到一条向量（常被跳过的环节）

| 读出方式                       | 做法                              | 常见于                        | 要注意                                       |
| ------------------------------ | --------------------------------- | ----------------------------- | -------------------------------------------- |
| **[CLS] / 专用标记**     | 取位置 0 或特殊 token             | 早期 BERT 句向量              | 未为检索微调时表征弱                         |
| **Mean pooling**         | 对非 padding 位置取均值           | SBERT、多数 Encoder Embedding | 长文会被稀释；需正确 mask                    |
| **Last token / EOS**     | 取序列末隐状态                    | 多数 Decoder LLM Embedding    | 依赖因果掩码下「末位汇聚」；改双向后语义会变 |
| **Latent attention**     | 可学习 query 对$H$ 做注意力聚合 | NV-Embed                      | 多一套参数，常强于死板 mean                  |
| **多向量（不池化成 1）** | 保留每 token/ patch 向量          | ColBERT / ColPali             | 表示范式已变，见 §3.2.3                     |

**断环警告**：只说「用了 Qwen3 骨干」却不说读出方式，复现实验会对不齐。换 mean ↔ last-token 等于换了一个模型头。

#### 3.1.2 对比学习在优化什么（InfoNCE 在算什么）

对每个 query $q$，有正样本 $d^+$ 与负样本 $\{d^-_k\}$。令 $s_i=\text{sim}(q,d_i)/\tau$，InfoNCE 就是把「正确文档」当成类别的交叉熵：

$$
\mathcal{L}_{\text{InfoNCE}}=-\log\frac{e^{s_+}}{e^{s_+}+\sum_k e^{s_k^-}}=-\log\,p(d^+\!\mid q;\text{候选集}).
$$

要点：

- **优化的是相对排序，不是向量绝对值**。换负例池，同一正对的「最优嵌入」可以变。
- **$\tau$**：越小分布越尖，难负例梯度更大；过大则「谁都差不多」，细粒度分不开。
- **in-batch 负例**：把同一 batch 里别人的正文档当自己的负例，等价于用很大的候选集做上述 CE，几乎零额外编码成本。代价是：batch 里若混进近重复 / 同文档切块，会制造**假负例**（§5.3）。
- SigLIP 改成**逐对 sigmoid**，不做「全体负例上的 softmax」，大规模多模态时更稳；$\tau$ 与 InfoNCE **不可直接照搬**。

**错误用法**：把 Embedding 训练理解成「回归到某个固定教师向量」。可以做向量蒸馏（§5.2.5 / §11），但那是另一条损失；主路径对比学习并不要求绝对值对齐。

#### 3.1.3 非对称编码

生产检索里，Query 与 Document 往往**不应共用同一前缀**：

| 侧       | 常见处理                    | 原因                   |
| -------- | --------------------------- | ---------------------- |
| Query    | 任务指令 /`search_query`  | 短、意图强、要对齐任务 |
| Document | 纯正文 /`search_document` | 长、必须能离线预计算   |
| 两塔参数 | 共享骨干或 DPR 式双塔       | 共享省参；独立可专化   |

E5-instruct、Cohere embed-v3、多数 LLM Embedding 把非对称写成接口契约。模型若**按指令训练**，评测/线上漏 instruct，分数会明显掉——掉多少取决于该权重对指令的依赖，不是魔法常数。对称任务（STS、去重）通常**两侧同模板或都无指令**，与检索课表不要混用同一套字符串。

### 3.2 三种表示范式

表示范式回答：**一条文档在索引里存成什么**。它与「交互架构」（§3.3）正交：同一套骨干既可以读出 Dense，也可以接 Sparse Head 或保留多向量。
```

Dense        → 1 × d 稠密向量     （ANN 友好，信息瓶颈）
Sparse       → |V| 维词项权重     （倒排友好，精确匹配）
Multi-Vector → n × d' token 向量 （MaxSim，细粒度）

```

#### 3.2.1 Dense：单向量信息瓶颈

$$
\mathrm{TopK}(q)=\underset{d\in\mathcal{C}}{\operatorname{arg\,top\text{-}k}}\;\text{sim}(f(q),f(d)).
$$

**够用**：开放域段落、FAQ、多数 RAG chunk、聚类/去重。
**不够**：查询词必须命中（合规/SKU/条款号）、版式敏感扫描件、超长文档内定位 → Hybrid 或 Multi-Vector。

**反例**：384 维通用句向量搜「合同第 12.3 条」——近邻常是同主题错条款。加大 $d$ 往往解决不了，要加词法通道。

#### 3.2.2 Sparse：神经化的词法匹配

经典 BM25 用统计公式给词项加权。SPLADE 一类方法让 Encoder 对每个输入预测**词表上的权重**（常见实现：在 MLM 词表 logits 上做 $\mathrm{ReLU}(\cdot)$ / $\log(1+\mathrm{ReLU})$，再对 token 维聚合），得到稀疏向量 $w(d)\in\mathbb{R}^{|V|}$，分数为

$$
s_{\text{lex}}(q,d)=\sum_{t\in V} w_t(q)\,w_t(d).
$$

可进倒排，可解释（看哪些词项权重大）。BGE-M3 Sparse 同属此路，但与 Dense/ColBERT 头共享 Encoder——细节见《[BGE-M3三功能统一详解报告](BGE-M3三功能统一详解报告.md)》§3–§7。

**不是**「另一种 1024 维 Dense」。**错误用法**：把 $|V|$ 维稀疏向量丢进 FAISS 做 L2。

#### 3.2.3 Multi-Vector：为何是 MaxSim

Query / Doc 产出 $E_q\in\mathbb{R}^{|q|\times d'},\;E_d\in\mathbb{R}^{|d|\times d'}$。若对所有 token 对做全连接再求和，既贵又易被文档长度淹没。ColBERT 的 MaxSim：

$$
s(q,d)=\sum_{i=1}^{|q|}\max_{j=1}^{|d|} \hat{E}_q[i]^\top \hat{E}_d[j]
$$

含义：**每个 query token 在文档里找最匹配的一处**，再累加。这保留「谁对上了哪」的细粒度，又比 Cross-Encoder 便宜（文档向量可离线存）。ColPali 把 token 换成页面图像 patch，思想同一类。

相对 Dense：更贵的存储与检索算子（PLAID 等）。相对 Cross：文档仍可预计算。

### 3.3 三种交互架构

交互架构回答：**算分时，注意力有没有同时看见 q 与 d**。

| 架构                       | 编码           | 交互           | 速度       | 精度       | 代表               |
| -------------------------- | -------------- | -------------- | ---------- | ---------- | ------------------ |
| **Bi-Encoder**       | q、d 各算一次  | 向量点积/余弦  | ★★★★★ | ★★★     | E5, BGE, DPR, CLIP |
| **Cross-Encoder**    | `[q;d]` 联合 | 全层交叉注意力 | ★         | ★★★★★ | BGE-Reranker       |
| **Late Interaction** | 各算多向量     | MaxSim         | ★★★     | ★★★★   | ColBERT, ColPali   |

#### 3.3.1 为什么 Bi 通常打不过 Cross（机制，不是玄学）

Cross 在每一层都可以让「查询里的否定/数量/约束」直接 attend 到「文档里的对应片段」。Bi 必须在**互不看见对方**的前提下，把全部检索相关信息压进一个（或一组）向量，再靠点积碰运气。

具体失败型：query「不含糖的酸奶」vs 文档大谈「酸奶、蔗糖」——Cross 容易靠 token 级交互判不相关；Bi 的单向量常仍因「酸奶」主题靠近而排前。这就是生产上 **Bi/Hybrid 召回 + Cross 精排** 的根由，不是「多堆一个模型好看」。

**复杂度**：Cross 对全库打分是 $O(|\mathcal{C}|)$，不可扩展；只对召回 Top-K 精排才合理。

#### 3.3.2 FAQ：骨干 ≠ 表示 ≠ 交互

| 概念 | 问什么            | 例子                       |
| ---- | ----------------- | -------------------------- |
| 骨干 | 预训练谁          | BERT / Qwen / SigLIP       |
| 表示 | 索引存什么        | Dense / Sparse / Multi-Vec |
| 交互 | q–d 何时注意力   | Bi / Cross / Late          |
| 读出 | $H\to v$ 怎么做 | mean / EOS / latent        |

「上了 LLM Embedding」≠「变成了 Cross-Encoder」。Qwen3-Embedding 仍是 Bi+Dense；ColPali 骨干是 VLM，交互是 Late。

#### 3.3.3 Fusion ≈ Rerank

多模态 **Fusion Encoder**（图文 token 进同一 Transformer 出标量分）与文本 **Cross-Encoder Rerank** 是同一计算图：**联合编码 → 标量相关性**。差别在模态与数据，不在「算不算 Embedding」。可 ANN 召回 → 必须 Bi 或 Late；只打候选 → Cross/Fusion。

### 3.4 相似度度量与索引约定

| 度量              | 适用                       |
| ----------------- | -------------------------- |
| 余弦 / 归一化点积 | 默认检索                   |
| 未归一化点积      | 须与训练一致；幅度会进分数 |
| 欧氏距离          | 聚类等；与余弦空间不要混用 |
| MaxSim            | 多向量                     |
| 稀疏点积          | SPLADE / BGE-M3 Sparse     |

训练、评测、向量库三处契约必须一致。

### 3.5 小结

1. 先定任务：对称相似 vs 非对称检索 → 再定损失与模板（§5）。
2. 默认：Dense Bi +（可选）BM25/Sparse + Cross 精排。
3. 关键词/合规差 → 加 Sparse，而不是只加维。
4. 版式/无 OCR → Multi-Vector（ColPali 类），而不是迷信「先 OCR 再 Dense」。
5. 写清：骨干、读出、表示、交互四件套；缺一不可复现。

---

## 4. 架构分类体系

§3 已把**表示**与**交互**讲清。本章只补一张「选型坐标」：同一模型要在四个轴上同时定点，避免只报参数量。

### 4.1 四轴坐标（比单表有用）

| 轴             | 选项                                  | 决定什么                       |
| -------------- | ------------------------------------- | ------------------------------ |
| **骨干** | Encoder / Decoder-LLM / Enc-Dec / VLM | 上下文长度、语义上限、微调成本 |
| **读出** | CLS / mean / EOS / latent / 多向量    | 向量怎么从$H$ 来（§3.1.1）  |
| **表示** | Dense / Sparse / Multi-Vec（可组合）  | 索引形态与存储                 |
| **交互** | Bi / Cross / Late                     | 能否预计算、精排位置           |

**正交举例**：

| 模型            | 骨干        | 读出              | 表示                | 交互                    |
| --------------- | ----------- | ----------------- | ------------------- | ----------------------- |
| BGE-base        | Encoder     | CLS/mean 系       | Dense               | Bi                      |
| BGE-M3          | Encoder     | 多 Head           | Dense+Sparse+Multi  | Bi + Late（ColBERT 头） |
| Qwen3-Embedding | Decoder LLM | 指令 + 末位类读出 | Dense（常带 MRL）   | Bi                      |
| BGE-Reranker    | Encoder     | 分类头标量        | —（不算库向量）    | Cross                   |
| ColPali         | VLM         | patch 多向量      | Multi-Vec           | Late                    |
| CLIP            | ViT+Text    | 塔顶投影          | Dense（两模态对齐） | Bi（双塔）              |

### 4.2 骨干怎么选（机制，不是品牌）

| 骨干                           | 适合                                 | 不适合硬扛                                       |
| ------------------------------ | ------------------------------------ | ------------------------------------------------ |
| **Encoder ≤0.6B**       | 私有化 RAG、CPU/边缘、要 Sparse/多头 | 极长上下文、复杂指令跟随                         |
| **Decoder LLM 1.5B–8B** | 长上下文、指令式多任务、合成数据友好 | 极致 QPS / 端侧（除非蒸馏到学生）                |
| **VLM**                  | 图文统一、视觉文档                   | 纯文本检索的性价比（通常不如专用文本 Embedding） |

部署形态（API / 自托管 / 量化）是**运维轴**，不改变上面四轴；量化只动精度与吞吐，不把 Bi 变成 Cross。

---

## 5. 训练方法与数据工程

本章目标：读完后能搭出一条**可照着做的课表**——弱监督 → 监督 + hard negative →（可选）多任务 / 蒸馏 / MRL——并知道每一步在优化什么、常见翻车点在哪。难负例的工业细节（刷新节奏、假负例审计、评测回归）下沉专题《[难负例挖掘工业实践](难负例挖掘工业实践.md)》；此处给主文级深度。

### 5.1 训练 Pipeline 总览
```

Stage 0: 骨干预训练 (CLM/MLM/图文对比, 通用语料)
    ↓
Stage 1: 弱监督对比预训练 (十亿级 pairs, InfoNCE / Sigmoid)
    ↓
Stage 2: 监督微调 (MS MARCO, NLI, QA + Hard Negatives)
    ↓
Stage 3: 多任务混合 (Retrieval + STS + Classification + Clustering)
    ↓
Stage 4 (可选): 蒸馏 / MRL / 领域适配 / 指令对齐

```

| 阶段    | 数据噪声     | 负例策略               | 典型步数/规模 | 失败模式               |
| ------- | ------------ | ---------------------- | ------------- | ---------------------- |
| Stage 1 | 高（弱监督） | 以 in-batch 为主       | 亿～十亿对    | 假正例把空间学糊       |
| Stage 2 | 低           | BM25/CE/Teacher 难负例 | 百万级标注    | 假负例 → 过推、泛化差 |
| Stage 3 | 中           | 任务混合采样           | 视任务表      | 检索被 STS 稀释        |
| Stage 4 | —           | 教师分布 / 截维损失    | 较短          | 蒸馏温度与教师错配     |

**课表原则**：先用脏数据把空间「撑开」，再用干净数据 + 难负例「削尖」；领域适配尽量在 Stage 2/4 做，而不是一上来就用小领域数据从头训。

### 5.2 核心损失函数

先分清任务，再选损失——这是最容易选错的一步。

| 任务形态                        | 标签长什么样                      | 常用损失                          | 说明                                  |
| ------------------------------- | --------------------------------- | --------------------------------- | ------------------------------------- |
| **检索 / 「这一对是正」** | $(q,d^+)$，负例 implicit 或显式 | **MNRL / InfoNCE**          | 生产检索主路径                        |
| **相似度回归**            | 句对 + 连续分（0–1 / 0–5）      | **CosineSimilarityLoss** 等 | STS；**不是**开放域检索默认损失 |
| **三元组**                | $(q,d^+,d^-)$                   | Triplet / Margin                  | 负例质量决定上限                      |
| **教师蒸馏**              | 教师分数 / 序 / 向量              | KL / MSE / cosine                 | §5.2.5、§11                         |

**常见错配**：做 RAG 却用 CosineSimilarityLoss、且 batch 里没有检索负例 → 验证集 STS 涨、Recall 不涨；或只有连续相似度标签却硬上 MNRL。

#### 5.2.1 InfoNCE

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(q, d^+) / \tau)}{\exp(\text{sim}(q, d^+) / \tau) + \sum_{d^-} \exp(\text{sim}(q, d^-) / \tau)}
$$

即 §3.1.2「候选集上的 CE」。$\tau$ 文本检索常见 $0.01$–$0.05$。

#### 5.2.2 MNRL（in-batch 负例）

Batch 内配对 $(q_i,d_i)$：对 $q_i$，除 $d_i$ 外的 $\{d_j\}$ 皆为负。弱监督与检索微调主力。同文档切块 / 近重复进同一 batch → 假负例；要去重或分桶。

#### 5.2.3 Sigmoid（SigLIP 风格）

逐对 logistic，不做全体负例 softmax。多模态大 batch 更稳；$\tau$ 与 InfoNCE 不通用。

#### 5.2.4 MRL

$$
\mathcal{L}_{\text{MRL}} = \sum_{m \in \{64, 128, 256, 512, \ldots\}} w_m \cdot \mathcal{L}(v_{1:m})
$$

截断维是否够用，以自有检索集为准，不要背外部「保留百分之几」的传闻。

#### 5.2.5 蒸馏信号

$\mathrm{KL}(P_T\|P_S)$ / $\mathrm{MSE}(s_T,s_S)$ / $1-\cos(v_T,v_S)$：蒸排序、蒸分数、蒸向量。与 LLM token-KD 不是一套（§11）。

### 5.3 Hard Negative Mining

**难负例**：模型容易排高、业务判定不相关。
**假负例**：其实相关（或应相关），却被当负例推开。

没有难负例，Stage 2 往往只是在推「显而易见的不相关」。假负例过多：loss 降、邻域被撕碎，近义召回或 STS 变差。

| 方法                | 机制             | 主要风险      |
| ------------------- | ---------------- | ------------- |
| Random / BM25       | 易或偏词法       | 语义难例不足  |
| In-batch            | 便宜             | 假负例        |
| Teacher top-k       | 语义难例         | 混假负例      |
| CE / Positive-aware | 过滤更准         | 贵 / 阈值要调 |
| Dynamic / Cross-GPU | 训练中加难、扩池 | 工程复杂      |

**可操作一层（positive-aware）**：Teacher 候选里，若分数 ≥ 正样本分 − margin，则不当负例。每 query 留 2–4 个真正难负例通常够用。

**例子**：query「如何重置路由器」；正文档=官方重置步骤。「如何重置手机」主题近、答案错 → 好难负例。同型号另一篇官方 FAQ 变体被标负 → 假负例，推开会伤召回。

刷新节奏、假负例治理与评测回归 → 《[难负例挖掘工业实践](难负例挖掘工业实践.md)》。

### 5.4 数据

| 策略        | 来源             | 主要风险                             |
| ----------- | ---------------- | ------------------------------------ |
| Natural     | NLI / QA / 复述  | 与「检索相关」不同构                 |
| Weak        | 标题–正文、点击 | **假正例**（标题党）把空间学糊 |
| Synthetic   | LLM 写 query     | 模板塌缩；污染测试集                 |
| Self-mining | 自检索           | 自我确认                             |
| Multi-task  | 指令任务表       | 检索被 STS 采样稀释                  |

E5 压缩课表：弱监督十亿对 → 监督 + 负例 → Query 侧 instruction。BGE-M3：RetroMAE → 对比 → 三头 + Self-KD（见专题）。

合成数据至少核对：模板=线上；未用测试 doc 当条件；专有名词/长短 query 覆盖；多语言分层。

### 5.5 指令

```

Query:    Instruct: {task_description}
          Query: {query_text}
Document: {document_text}

```

按指令训练的权重，线上必须同构。领域说明进 instruct，或「通用检索指令 + 领域难负例」；禁止训练有指令、线上裸 query。

### 5.6 旋钮与判据

| 旋钮       | 起点                         | 怎么看                       |
| ---------- | ---------------------------- | ---------------------------- |
| $\tau$   | 0.01–0.05                   | 难例分不开略降；噪声过敏略升 |
| batch      | 尽量大                       | 更多 in-batch 负例           |
| hard neg/q | 2–4                         | 未过滤时宁少勿滥             |
| lr         | Encoder ~1e-5；LLM 更小/LoRA | 冲掉预训练则降               |
| 阶段切换   | 领域验证平台期               | 不要死磕固定 epoch           |

**最小可行**：开源 Embedding 续训 → 领域弱配对 → 正对 + 过滤难负例至验证饱和 → 可选 MRL/蒸馏。

**训练中看什么**：检索看 Recall@K / nDCG；只有 STS 标签才用相似度 Spearman。两者不能互相代替当「成功」判据。

### 5.7 进阶技巧（索引）

双向注意力（NV-Embed / LLM2Vec）、LoRA、RetroMAE、Latent Attention 池化、Query 扩展、Self-KD——都是在 §3–§5 主路径上的加成，不替代「损失选对 + 负例干净」。

---

## 6. 评测体系与 Benchmark

### 6.1 MTEB（Massive Text Embedding Benchmark）

| 属性     | 详情                                            |
| -------- | ----------------------------------------------- |
| 任务数   | 56（英文）/ 112+ 语言                           |
| 任务类型 | 8 类（见下表）                                  |
| 指标     | 各任务特定（MAP, nDCG, Accuracy, v-measure 等） |
| 平台     | HuggingFace MTEB Leaderboard                    |
| 版本     | v1 (56 tasks) / v2 (2026 重构，分数不可直接比)  |

**8 类任务各自在测什么**（平均分之前先看这个）：

| 任务类型                      | 测的能力                     | 和 RAG 检索的关系               | 典型翻车               |
| ----------------------------- | ---------------------------- | ------------------------------- | ---------------------- |
| **Retrieval**           | 非对称 q→d 排序（nDCG/MRR） | **最直接相关**            | 指令漏加、归一化不一致 |
| **Reranking**           | 对已给候选重排               | 测的是 Cross/精排，不是 Bi 建库 | 用 Bi 分数硬比 CE 任务 |
| **STS**                 | 对称句对相似度（Spearman）   | 相关但不等价；STS 涨≠Recall 涨 | 用 STS 早停做检索模型  |
| **Classification**      | 冻结构 + 线性头              | 测可分性                        | 与开放域检索弱相关     |
| **Clustering**          | 无监督簇纯度                 | 测空间几何                      | 检索强、聚类弱很常见   |
| **Pair Classification** | 句对关系（如重复）           | 偏对称匹配                      | —                     |
| **Summarization**       | 摘要与原文相关               | 窄                              | 权重低，别主导选型     |
| **Bitext Mining**       | 跨语对齐                     | 只在多语场景关键                | —                     |

公开榜的「平均分」会把上述能力揉在一起；业务若是 RAG，应主看 Retrieval（及自有集），其余当诊断。

### 6.2 BEIR（Zero-Shot Retrieval）

- 18 个检索数据集，零样本评估
- 涵盖 Bio, Finance, News, QA 等 9 个领域
- 指标：nDCG@10, Recall@100
- 现为 MTEB Retrieval 子集

### 6.3 其他 Benchmark

| Benchmark           | 范围         | 特点             |
| ------------------- | ------------ | ---------------- |
| **MMTEB**     | 250+ 语言    | 多语言综合       |
| **C-MTEB**    | 35 中文任务  | 中文专用         |
| **MTEB-Code** | 代码检索     | CodeSearchNet 等 |
| **ViDoRe**    | 视觉文档检索 | ColPali 主战场   |
| **AIR-Bench** | 动态更新     | 防数据泄露       |
| **CoIR**      | 代码 IR      | 代码搜索专用     |

### 6.4 MTEB 局限性（必读）

1. **与实际 RAG 性能不完全相关**：MTEB 高 ≠ 你的领域 RAG 好
2. **任务平均掩盖短板**：检索强 ≠ 聚类强；汇报时拆开 Retrieval / STS / Clustering
3. **版本不可比**：v1 vs v2 分数体系不同，禁止跨版本排行榜叙事
4. **数据污染风险**：合成数据与爬虫对若扫过测试文档，分数虚高
5. **缺少长文档/多模态**：需 BEIR 子集之外的业务集、ViDoRe、MMEB 等补充

### 6.5 自建评测协议（最小模板）

| 项       | 建议                                                          |
| -------- | ------------------------------------------------------------- |
| 文档池   | 冻结版本号；与训练挖负例语料隔离                              |
| Query 集 | 真实日志抽样 + 专家补难例；标明语言/长度/是否专有名词         |
| 标签     | 二值相关或 graded；双人标注抽一致性                           |
| 指标     | Recall@K、nDCG@K、MRR；业务侧加「人工 Top-5 可用率」          |
| 对照     | BM25、未微调开源 Embedding、当前线上模型                      |
| 防污染   | 训练前做 doc-id / 近邻重叠检查；合成数据禁止以测试 doc 为条件 |
| 发布     | 每次改模型附：数据版本、指令模板、是否归一化、索引类型        |

**建议路径**：公开榜粗筛 → 自有协议定胜负 → 端到端 RAG/点击 A/B。更细的污染防控可扩成专题《评测协议与污染防控》。

---

## 7. 应用场景全景

Embedding 不是「万能相似度」：不同任务对**对称性、粒度、索引形态、失败代价**的要求不同。先定任务，再定 §3 的四轴与 §5 的课表。

### 7.1 任务 × 表示 × 交互（比模型名单有用）

| 任务                        | 输入/输出      | 对称？          | 常用表示+交互               | 索引里存什么                | 典型失败                         |
| --------------------------- | -------------- | --------------- | --------------------------- | --------------------------- | -------------------------------- |
| **RAG 段落检索**      | query→chunk   | 否（q 短 d 长） | Dense Bi + Hybrid + CE 精排 | chunk 向量（+ 可选 sparse） | 切块切断语义；漏关键词；精排未开 |
| **FAQ / 问句匹配**    | q↔q'          | 是              | Dense Bi                    | 问题向量                    | 用检索非对称模板训 FAQ           |
| **语义搜索（站内）**  | query→doc     | 否              | Dense+BM25 Hybrid           | 全文 chunk                  | 只 Dense 漏 SKU/型号             |
| **去重 / 近重复**     | doc↔doc       | 是              | Dense Bi，阈值              | 全库或 LSH                  | 阈值过低误杀改版；过高漏洗稿     |
| **聚类 / 主题**       | doc→簇        | 是              | Dense                       | 向量                        | 用检索榜模型却不看 v-measure     |
| **推荐 Two-Tower**    | user, item→分 | 否              | Dense Bi（+ 特征塔）        | item 向量；user 在线算      | 冷启动；实时行为未进 user 塔     |
| **图搜图**            | image→image   | 是              | 视觉塔 Dense Bi             | 图像向量                    | 用 CLIP 文塔搜图；颜色/背景主导  |
| **文搜图**            | text→image    | 否              | 双塔对齐空间                | 图像向量                    | 训练图文对噪声大；指令未对齐     |
| **扫描件 / 版式文档** | text→page     | 否              | Multi-Vec Late              | 页级 patch 向量             | OCR→chunk→Dense 丢表格/图      |
| **代码搜索**          | nl/code↔code  | 可对称可非对称  | 代码专用 Bi                 | code snippet 向量           | 通用文本 Embedding 不懂 API 名   |
| **Agent 记忆**        | query→memory  | 否              | Dense Bi                    | 记忆条目向量                | 只存摘要丢细节；无 TTL/版本      |

### 7.2 RAG：Embedding 管哪一段、不管哪一段

```

离线:  Raw docs → 清洗 → Chunking → [可选 metadata] → Embed(doc) → 向量库 (+ 可选 BM25 索引)
在线:  Query → [可选改写/扩展] → Embed(q) → ANN/Hybrid 召回 Top-K
       → [可选 Rerank] → Top-N chunks → Prompt 拼装 → LLM

```

**Embedding 只保证「候选 chunk 在语义上像相关」**，不保证：事实正确、最新、可引用、无冲突。下游幻觉、过期、多跳推理失败，不能单靠换 Embedding 模型解决。

#### 7.2.1 Chunking 与向量的耦合

| 策略                         | 机制                       | 何时用           | 失败模式                                   |
| ---------------------------- | -------------------------- | ---------------- | ------------------------------------------ |
| 固定长度（如 512 tok）       | 实现简单                   | 通用 prose       | 法条/表格/步骤被拦腰切断 → 召回「半句话」 |
| 语义边界                     | 按段/标题切                | 结构化文档       | 切太碎丢上下文                             |
| Parent–Child                | 小块检索、大块进 prompt    | 长文 RAG         | 子块命中但父块未进上下文                   |
| Contextual chunk（dsRAG 等） | LLM 给每块加「文档级前缀」 | 强依赖块内自洽   | 前缀与训练分布不一致时 Embedding 漂移      |
| 页级视觉（ColPali）          | 不 OCR，整页多向量         | 扫描 PDF、幻灯片 | 纯文本 API 管道接不上                      |

**断环**：chunk 策略与 Embedding 训练粒度不一致（训练看整段、线上 128 tok 小块）时，Recall 上限会被 chunk 天花板卡住，换 8B 模型也无效。

#### 7.2.2 RAG 常见失败与归因

| 现象                    | 先查什么                | 常见根因                         |
| ----------------------- | ----------------------- | -------------------------------- |
| 召回空 / 全不相关       | Top-K、阈值、库是否更新 | 索引 stale；query 未加 instruct  |
| 召回有主题但答非所问    | chunk 内容、精排        | 切块错；无 Rerank；Bi 单向量瓶颈 |
| 专有名词搜不到          | Hybrid 分数             | 纯 Dense；未建 BM25/Sparse       |
| 英文 query 中文 doc 差  | 多语模型与模板          | 单语 Embedding；跨语未训         |
| MTEB 高、业务 Recall 低 | 自有验证集              | 榜与领域分布脱节；数据污染       |
| 加了 Rerank 更差        | Rerank 训练域           | CE 与领域不匹配；候选太少        |

### 7.3 FAQ、去重、推荐（短节）

**FAQ**：对称相似；两侧同一模板。误用检索的 `Instruct + Query` 非对称格式会把 FAQ 当「q 找长 doc」，Top-1 抖动。

**去重**：阈值在验证集上扫 Precision–Recall；近重复（改写、翻译）需要 harder 负例或更大模型，不是单纯调低阈值。

**推荐 Two-Tower**：与 Bi-Encoder 同构，但多了**特征工程、负采样（in-batch 曝光未点击）、实时 user 状态**。Embedding 文献里的 MNRL 课表要改成「点击为正、曝光未点击为负」，且假负例（用户其实喜欢但未点）更严重。

### 7.4 多模态检索在应用层的分叉

与 §10 四类路线对齐（详述见《[图文 Embedding 模型技术综述](图文Embedding模型技术综述.md)》）：

| 场景                | 应走的路线                          | 别走弯路                              |
| ------------------- | ----------------------------------- | ------------------------------------- |
| 商品图搜图          | ① 双塔（CLIP/SigLIP 或电商视觉塔） | 用文本 RAG 模型 encode 图像路径字符串 |
| 以文搜图            | ① 双塔，共享对齐空间               | 只用 DINO 视觉塔（无文本塔）          |
| 图文混合 Agent 记忆 | ③ MLLM 通用嵌入                    | 强 OCR 再当纯文本                     |
| PDF 报表 / 扫描合同 | ④ ColPali 类 Late                  | OCR→chunk→Dense 丢表头/合并单元格   |

**图搜图 vs 文搜图**：图搜图只需视觉塔在同一度量空间；文搜图必须**文本 query 与图像 doc 共空间**（CLIP/SigLIP/MLLM-Emb）。DINO 类自监督视觉塔语义强，但不自带文本对齐，不能直接文搜图，需另训对齐头或换 CLIP 系。

---

## 8. 检索 Pipeline 工程实践

Pipeline 设计 = 在**延迟预算**内最大化 **Recall@K（给 Rerank/LLM 的候选质量）**，不是堆模型个数。

### 8.1 三段式各自解决什么

| 阶段               | 算什么                                | 复杂度                    | 典型 K       | 失败时表现                                |
| ------------------ | ------------------------------------- | ------------------------- | ------------ | ----------------------------------------- |
| **Recall**   | Bi / Hybrid / Late 在**全库**   | $O(\log N)$ ANN 或倒排  | 50–200      | relevant 根本进不了候选 → 后面全白搭     |
| **Rerank**   | Cross / Late / 小 CE 在**候选** | $O(K \cdot L)$          | 5–20        | 召回够但 Top-N 仍含大量「主题对、答案错」 |
| **Generate** | LLM 读 context                        | $O(N \cdot \text{tok})$ | 3–10 chunks | context 太长/冲突/过时                    |

**原则**：Recall 阶段宁可多给（K 偏大），Rerank 收窄；省 Rerank 省的是延迟，丢的是 Precision。全库 Cross-Encoder 在百万级以上不可行（§3.3.1）。

### 8.2 Hybrid：为何有效、怎么融合

Dense 擅语义泛化；BM25/Sparse 擅**词面命中**（型号、法条号、错误码）。二者分数尺度不同，不能直接 $0.5 s_1 + 0.5 s_2$ 除非在验证集上标定。

#### 8.2.1 RRF（Reciprocal Rank Fusion）

$$
\text{RRF}(d) = \sum_{r \in \{\text{dense},\text{sparse},\ldots\}} \frac{1}{k + \text{rank}_r(d)}
$$

- 只用**名次**，不用原始分 → 不怕 Dense 余弦与 BM25 量纲不同。
- $k$ 常取 60（经验）；$k$ 越大，头部名次差异被抹平。
- **适用**：多路召回源（Dense + BM25 + 多模型）快速合并。
- **局限**：不利用「强相关 doc 分数领先幅度」；两路都排第 50 的 doc 可能被抬上来。

#### 8.2.2 加权分融合（BGE-M3 等）

$$
s = w_1 s_{\text{dense}} + w_2 s_{\text{lex}} + w_3 s_{\text{mul}}
$$

权重必须在**自有验证集**上网格搜索或学习排序；论文示例权重不可直接抄。Lexical 与 Dense 分数需各自归一化或校准后再加。

**工程路径**（BGE-M3 常见）：Dense ∪ Sparse 并集召回 → RRF 或加权 → 可选 ColBERT 头 / Cross-Encoder 精排。细节见《[BGE-M3三功能统一详解报告](BGE-M3三功能统一详解报告.md)》§10。

#### 8.2.3 何时 Hybrid 仍不够

- 表格/图在 PDF 内，文本通道完全丢失 → 上 ColPali（§10）。
- Query 极短且专有名词为主 → Sparse 权重要高，甚至 Sparse-only 兜底。
- 多语言 query 与单语 BM25 词表不匹配 → 靠多语 Dense，Sparse 仅作辅助。

### 8.3 Reranker 放哪、选谁

| 类型                              | 与 Recall 关系             | 何时选                             |
| --------------------------------- | -------------------------- | ---------------------------------- |
| **Cross-Encoder**           | 全新算$s(q,d)$，看全交互 | 文本候选 ≤200，延迟允许百 ms 级   |
| **Late（ColBERT/ColPali）** | 用预存多向量 MaxSim        | 已存多向量索引；要 Bi 与 CE 中间档 |
| **LLM Rerank（4B 级）**     | 大模型打分                 | 质量优先、候选很少                 |

Reranker 训练域与业务域不一致时，会出现「CE 把正确 doc 打下去」。此时优先**领域微调 Reranker** 或**领域 Bi 召回**，而不是换更大的通用 CE。

### 8.4 ANN 索引与召回质量

| 算法             | 机制             | 调什么                            | 典型坑                                 |
| ---------------- | ---------------- | --------------------------------- | -------------------------------------- |
| **HNSW**   | 图导航近似最近邻 | `efConstruction` / `efSearch` | `efSearch` 太小 → Recall 掉但延迟低 |
| **IVF**    | 先聚类再搜簇     | `nlist` / `nprobe`            | `nprobe` 太小漏簇                    |
| **PQ/OPQ** | 向量量化         | 码本数                            | 量化过猛伤精度；与 MRL 截维叠加要测    |
| **Flat**   | 暴力精确         | —                                | 仅小库或离线评测 gold                  |

**契约**：库内 metric（IP/L2）与 Embedding 是否 L2 归一化一致（§3.4）。换库不改归一化，会出现「离线评测 0.9、线上 0.6」的假象。

**增量更新**：doc 版本变更必须**同 id 覆盖或删后重插**；chunk 与 parent 版本不一致会导致 RAG 引用旧段落。

### 8.5 延迟与成本粗算（用于选型，非报价）

单次请求延迟 ≈ $T_{\text{embed}}(q) + T_{\text{ANN}}(K) + T_{\text{rerank}}(K') + T_{\text{LLM}}$。

- $T_{\text{embed}}(q)$：小 Encoder ~5–20ms GPU；7B LLM-Emb 更高，常 batch 摊薄。
- $T_{\text{ANN}}$：HNSW 毫秒级（百万级）；与 $K$ 弱相关。
- $T_{\text{rerank}}$：Cross 对 $K=100$ 常占 ** tens of ms–百 ms**，往往是瓶颈。

降本顺序：MRL 截维 → 小 Bi → 缩小 Rerank 输入 K → 量化 Embedding；**不要**先砍 Hybrid 再换 8B 模型。

---

## 9. 文本 Embedding：机制族谱（非排行榜）

选型应问：**要什么交互、什么数据课表、能否私有化**——MTEB 总分只做粗筛。分数随版本变化，下文**不列具体榜分**；需对照时查 HuggingFace MTEB Leaderboard 与模型卡。

### 9.1 机制族 A：Encoder + 对比学习（SBERT 血脉）

- **骨干**：BERT/RoBERTa/XLM-R（110M–560M）。
- **训练**：NLI/STS 起步 → MS MARCO 检索 + in-batch / BM25 负例。
- **读出**：mean / CLS。
- **代表**：SBERT、BGE v1.5 系、E5-base/large、GTE-base。
- **适合**：私有化 RAG、CPU/边缘、要快、要微调。
- **上限**：512–8K 上下文（视版本）；复杂 instruct 跟随弱于 LLM 系。

### 9.2 机制族 B：弱监督课表 + 指令（E5 / BGE 主流）

- **Stage 1**：十亿级弱对（标题–正文、QA）撑空间。
- **Stage 2**：MARCO/NLI + **hard negatives** + Query 侧 instruction。
- **代表**：multilingual-e5-instruct、BGE-M3（再叠加 Sparse/Late 头）。
- **关键差异**：BGE-M3 在族 B 上增加**三功能头 + RetroMAE 打底**（见专题）；不是「又一个 BERT」。
- **适合**：多语 Hybrid 生产、要 Sparse 兜底。
- **翻车点**：漏 instruct；弱监督假正例未用 Stage2 纠偏。

### 9.3 机制族 C：Decoder LLM 作 Bi-Encoder

- **骨干**：Mistral/Qwen/Llama 1.5B–8B。
- **改造**：双向注意力 / LLM2Vec 式改造 + last-token 或 latent pooling。
- **数据**：更大弱监督 + 合成 query + 长上下文。
- **代表**：E5-mistral、NV-Embed、Qwen3-Embedding、GTE-Qwen2。
- **适合**：长文档、多任务 instruct、追求榜上限。
- **代价**：推理与显存；必须蒸馏/量化才适合边缘（§11）。

### 9.4 机制族 D：蒸馏得到的小模型（Students）

- **机制**：大 Teacher（族 C 或 CE）→ 小 Student（0.6B–2B）对齐**排序/分数/向量**（§11）。
- **代表**：Jasper、Stella、KaLM、Qwen3-Embedding-0.6B 路线。
- **适合**：要接近大模型检索质量、但部署预算有限。
- **注意**：蒸馏目标须与线上一致（同一 instruct、同一 Rerank 链路）；只蒸 STS 不蒸检索会偏。

### 9.5 机制族 E：API 闭源 Embedding

- **机制黑盒**，强在运维与长上下文 SLA。
- **代表**：OpenAI v3、Cohere v3/v4、Voyage、Gemini Embedding。
- **适合**：快速验证、无 GPU、数据可出境。
- **不适合**：强领域微调、离线合规、要 Hybrid 三头可控。

### 9.6 族谱 → 场景（对应 §14，此处不堆品牌）

| 你的约束             | 优先机制族                                   |
| -------------------- | -------------------------------------------- |
| 边缘 / CPU / 低成本  | A 或 D（蒸馏小模型）                         |
| 多语 Hybrid 生产     | B（+ Reranker）                              |
| 长上下文 + 高质量    | C，或 D 蒸出来                               |
| 无 GPU、先验证       | E                                            |
| ≤0.6B 自训（图+文） | D + 多模态 §10 ③④；Teacher 用大 CLIP/MLLM |

**附录式对照**：若需参数/许可/维度，查 §2.1 model card 与 HuggingFace；**勿用本章做选型唯一依据**。

---

## 10. 多模态与专用 Embedding

术语与《[图文 Embedding 模型技术综述](图文Embedding模型技术综述.md)》一致：**① 双塔 ② Fusion ③ MLLM 通用嵌入 ④ 多向量后交互**。四类可组合（如 ③ 骨干 + ④ 文档头）。

### 10.1 四类路线对照（能力边界）

| 维度               | ① 双塔 CLIP/SigLIP | ② Fusion  | ③ MLLM-Emb         | ④ ColPali 类 Late     |
| ------------------ | ------------------- | ---------- | ------------------- | ---------------------- |
| **交互**     | Bi，图文各算一次    | 联合注意力 | Bi（last-token 等） | Late MaxSim            |
| **预计算**   | 图/文均可           | 仅候选打分 | 图/文均可           | 页向量可预计算         |
| **图搜图**   | ✅ 视觉塔           | ❌         | ✅（视觉编码）      | ✅（页/image patches） |
| **文搜图**   | ✅ 对齐空间         | ✅ 精排    | ✅ 指令化           | ✅ text query→page    |
| **扫描 PDF** | 弱（当普通图）      | 中         | 中                  | **强**           |
| **延迟**     | 低                  | 高         | 中–高              | 中（索引大）           |

### 10.2 图搜图 vs 文搜图（训练与索引）

**图搜图**：query 与 gallery 都是图像 → 同一视觉编码器 $f_v(\cdot)$，$s=\cos(f_v(q), f_v(d))$。可用 DINO/DINOv2 特征 + 小投影，或 CLIP 视觉塔。失败常见原因：域偏移（商品图 vs 用户手机拍）、颜色纹理主导、未做 hard negative（同款不同色）。

**文搜图**：query 是文本 → 需要 $f_t(\text{text})$ 与 $f_v(\text{image})$ **共空间**。必须走 CLIP/SigLIP 对比训练或 MLLM 图文对齐；不能直接把文本 Embedding 模型与 DINO 视觉特征拼在一起搜。

**≤0.6B 自训常见配方**（行动线细节见 `0.6B图搜图文搜图自训学习行动路线.md`）：

1. 大 Teacher：CLIP/SigLIP 或 2B+ MLLM-Emb 产生伪标签或分数；
2. 小 Student：0.3B–0.6B 双塔或共享视觉塔；
3. 损失：图文 InfoNCE/Sigmoid +（可选）图搜图同塔对比；
4. 数据：商品图+标题弱对 + 人工 hard 负（同款错色、相似类目）；
5. 评测：**分开报** Image→Image 与 Text→Image，不要只报一个「多模态平均」。

### 10.3 视觉文档（④ 类）

ColPali / ColQwen：**页面渲染图 → patch 多向量**，query 侧 text token 多向量，MaxSim 聚合。绕过 OCR 流水线，对表格、图表、多栏排版更稳。代价是索引体积（每页数百向量）与专用检索器。

**错误路径**：扫描 PDF → OCR 错字 → chunk → 文本 Embedding → 问「图 3 销售额」类问题。

### 10.4 代码与领域专用

| 类型                | 机制要点                    | 与通用文本差异                                                    |
| ------------------- | --------------------------- | ----------------------------------------------------------------- |
| **代码**      | 标识符、语法树、repo 上下文 | 通用文本 Embedding 对 API 名 OOV；用 code 预训练或 Voyage-code 类 |
| **法律/医疗** | 术语密度高、要精确引用      | 通用模型 + 领域微调 +**Hybrid 词面**；判例号靠 Sparse       |
| **科学文献**  | 长摘要、引用关系            | SPECTER 类引用图对比；可与 Dense 并用                             |

专用域**优先**领域数据微调 + 难负例，而不是换更大的通用 8B。

---

## 11. Embedding 蒸馏与压缩

主文只讲清**版图与何时用**；损失组合、多教师课表、多模态 Teacher 细节见《[Embedding蒸馏技术详解](Embedding蒸馏技术详解.md)》。与 [`../distillation/知识蒸馏技术深度调研报告.md`](../distillation/知识蒸馏技术深度调研报告.md) 的边界：**那边**以 LLM/VLM **生成式** KD 为主；**本章**只谈**表示与排序**。

### 11.1 三种蒸馏信号（不要混用）

| 信号                  | 监督什么                      | 典型损失           | 适用                              |
| --------------------- | ----------------------------- | ------------------ | --------------------------------- |
| **排序 / 分数** | Teacher 对$(q,d)$ 相关度    | KL / MSE on scores | 检索主任务；Teacher 是 CE 或强 Bi |
| **向量**        | $v_T(x)$ 与 $v_S(x)$ 对齐 | MSE /$1-\cos$    | 维数相同或线性投影后              |
| **关系**        | 样本间距离结构                | triplet / pairwise | 数据少、要保流形                  |

**与 LLM KD 的区别**：LLM KD 蒸 **token 分布**（logits）；Embedding KD 蒸 **几何与排序**。把 MiniLLM/OPD 当 Embedding 主损失通常不对题。

### 11.2 典型场景

| 场景                      | 做法                                        | 注意                                           |
| ------------------------- | ------------------------------------------- | ---------------------------------------------- |
| **8B → 0.6B 部署** | 大 Bi Teacher + 检索损失 + 可选 CE 分数蒸馏 | Student 与 Teacher**同一 instruct 模板** |
| **CE → Bi**        | CE 打候选分 → KL 到 Bi                     | 候选集要含 hard neg                            |
| **多 Teacher**      | 拼接分数或加权（Jasper/Stella 思路）        | Teachers 冲突时先统一口径再训                  |
| **Self-KD**         | EMA/ checkpoint 自身                        | BGE-M3 Stage3；无外部 Teacher                  |
| **无 Teacher 压缩** | **MRL 截维**、INT8/INT4 量化          | 不是蒸馏，但常一起做                           |

### 11.3 MRL 与量化（部署压缩，非知识迁移）

- **MRL**：训练时惩罚前缀维，推理截断 $d$ → 索引变小、ANN 更快；精度损失在自有集上扫 $d\in\{256,512,\ldots\}$。
- **量化**：INT8/INT4 减显存与带宽；Embedding 检索对量化通常比生成 LLM 更耐受，仍要测 Recall。

Qwen3-Emb-8B INT4 等案例说明「大模型可压到消费级 GPU」，但 **0.6B 自训目标**更常是「Student + 蒸馏」而不是量化 8B。

### 11.4 蒸馏实验最小协议

1. 固定验证集：Recall@K / nDCG，与 Teacher 同模板。
2. 报告：Student 参数量、维度、是否 MRL/量化。
3. 对照：Student 无蒸馏、Student 只蒸 STS（说明检索会偏）。
4. 失败信号：STS 升、Recall 不升 → 蒸馏目标错或假负例未控。

---

## 12. 向量数据库与部署

### 12.1 库 vs 库：按能力选，不按营销选

| 需求                  | 优先                                       |
| --------------------- | ------------------------------------------ |
| 纯研究 / 单进程       | FAISS（库，非服务）                        |
| 要 Hybrid BM25 + 向量 | Elasticsearch、Vespa、Milvus 2.x、Weaviate |
| 已有 PostgreSQL       | pgvector + 全文（Hybrid 需自拼）           |
| 原型 / 小团队         | Qdrant、Chroma                             |
| 十亿级、磁盘索引      | Milvus DiskANN、Vespa                      |
| 强 metadata 过滤      | 带 filter 的 ANN（多数商业库支持）         |

**Hybrid 在库里的含义**：是「同一引擎内 BM25+向量」，还是「应用层 RRF 两路索引」——二者运维复杂度差很多。

### 12.2 部署检查清单（与 §3.4、§8.4 闭环）

1. **Metric**：IP + L2 归一化 是否一致。
2. **维度**：MRL 截断后重新建库还是多集合。
3. **Metadata**：版本号、source、acl——过滤与 Embedding 正交但 RAG 必备。
4. **Batch 编码**：离线建库 batch 够大；在线 query 单条延迟单独测。
5. **服务**：TEI/vLLM/FlagEmbedding 用于 BGE/Qwen；Reranker 常单独进程防阻塞 ANN。
6. **监控**：Recall@K 抽样、索引 lag、embedding 模型版本号。

### 12.3 参考拓扑

```

Client → Gateway → [Embed query] → Vector DB (ANN ± BM25)
                 → [Rerank Top-K] → LLM
        离线: Doc pipeline → Embed batch → Upsert

```

---

## 13. 前沿方向与开放问题

每条附「可做什么实验」，避免空泛 trend 列表。

| 方向                         | 未解什么                | 可验证实验                                      |
| ---------------------------- | ----------------------- | ----------------------------------------------- |
| **MTEB ↔ RAG**        | 榜与业务 Recall 脱节    | 同一模型：MTEB Retrieval vs 自有 RAG 集相关系数 |
| **Unified Gen+Emb**    | 生成与嵌入是否互相伤    | GritLM/LLM2Vec：调 Gen/Emb loss 权重看 Recall   |
| **Visual Doc RAG**     | 何时胜过 OCR 管道       | 同 corpus：ColPali vs OCR+chunk+BGE 端到端      |
| **Long Context Emb**   | 32K 一次 embed vs chunk | 长文单向量 vs hierarchical chunk Recall         |
| **On-policy hard neg** | 动态负例是否稳          | 对比静态 mined neg：训练曲线与 STS 漂移         |
| **Agent Memory**       | 存 raw vs 摘要 vs 向量  | 多轮任务成功率 vs 记忆条数                      |
| **Graph+Vector**       | 何时 GraphRAG 值回票价  | 多跳 QA：纯向量 vs GraphRAG 成本/准确           |

**开放问题**（仍无共识）：嵌入反演与隐私、低资源语言公平性、多跳推理单向量瓶颈、增量索引与版本一致、端到端 RAG 评测标准缺失。

---

## 14. 实践选型指南

### 14.1 决策：任务 → 表示 → 交互 → 训练（不是先选品牌）

```

1) 对称还是非对称？
   对称（FAQ/去重/STS）→ 两侧同模板；损失可用 STS 或 MNRL
   非对称（RAG/搜索）→ Query instruct + Doc 无 instruct；MNRL + hard neg
2) 索引里存什么？
   单向量 Dense → 默认
   要词面命中 → + Sparse/BM25 Hybrid
   扫描件/版式 → Multi-Vec Late（ColPali）或 MLLM-Emb
3) 算分方式？
   全库 → Bi 或 Late；Cross 只打 Top-K
4) 资源？
   无 GPU → 族 A 小模型或 API
   要 ≤0.6B 自训 → 族 D + §10 多模态 Teacher
5) 验证？
   自有 Recall/nDCG → 再参考 MTEB Retrieval 子集

```

### 14.2 微调最小闭环（与 §5 一致）

1. 数据：$(q, d^+)$ + 过滤后 hard neg；禁止测试 doc 进训练。
2. 损失：**检索用 MNRL/InfoNCE**，不是 CosineSimilarityLoss（除非纯 STS）。
3. 模板：与线上一致。
4. 验证：**InformationRetrievalEvaluator** 类指标，不是只跑 STS Spearman。
5. 上线：Hybrid + 可选 Rerank；记录 embedding 与索引版本。

### 14.3 常见陷阱（机制版）

| 陷阱              | 机制原因                         |
| ----------------- | -------------------------------- |
| STS 高、Recall 低 | 优化对称相似，非 q→d 排序       |
| 换 8B 无提升      | chunk/负例/instruct 瓶颈在前     |
| Hybrid 无效       | 权重未标定；Sparse 索引未建      |
| Rerank 反效果     | CE 域不匹配；K 太小              |
| 文搜图失败        | 未对齐图文空间；只用 DINO 视觉塔 |
| MTEB 选型         | 平均掩盖 Retrieval 短板          |

---

## 15. 参考文献

### 奠基与综述

1. Mikolov et al. (2013). Efficient Estimation of Word Representations in Vector Space. *ICLR*.
2. Pennington et al. (2014). GloVe: Global Vectors for Word Representation. *EMNLP*.
3. Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL*.
4. Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.
5. Wang et al. (2024). Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5). *arXiv:2212.03533*.
6. Tao et al. (2024). LLMs are Also Effective Embedding Models: An In-depth Overview. *arXiv:2412.12591*.
7. Chen et al. (2024). Recent Advances in Text Embedding: MTEB Review. *arXiv:2406.01607*.

### 训练方法

8. Gao et al. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. *EMNLP*.
9. Kusupati et al. (2022). Matryoshka Representation Learning. *NeurIPS*.
10. Xiao / Chen et al. (2024). M3-Embedding (BGE-M3). *arXiv:2402.03216*.（本仓库专题解读：《[BGE-M3三功能统一详解报告](BGE-M3三功能统一详解报告.md)》）
11. Lee et al. (2024). NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models. *arXiv:2405.17400*.
12. Zhang et al. (2024). NV-Retriever: Hard-negative Mining. *arXiv:2407.15831*.
13. Conan-Embedding (2024). Dynamic Hard Negative Mining. *arXiv:2408.15710*.

### 模型

14. Li et al. (2024). Multilingual E5 Text Embeddings. *arXiv:2402.05672*.
15. Muennighoff et al. (2024). GritLM: Generative Representational Instruction Tuning. *ACL*.
16. Jasper & Stella (2024). Distillation of SOTA Embedding Models. *arXiv:2412.19048*.
17. Qwen Team (2025). Qwen3 Embedding Technical Report.

### 多模态

18. Radford et al. (2021). Learning Transferable Visual Models (CLIP). *ICML*.
19. Tschannen et al. (2025). SigLIP 2. *arXiv:2502.14786*.
20. Faysse et al. (2025). ColPali: Efficient Document Retrieval with VLMs. *ICLR*.

### 检索

21. Karpukhin et al. (2020). Dense Passage Retrieval (DPR). *EMNLP*.
22. Khattab & Zaharia (2020). ColBERT: Efficient and Effective Passage Search. *SIGIR*.
23. Formal et al. (2021). SPLADE v2: Sparse Lexical and Expansion Model. *SIGIR*.

---

## 附录 A：术语表

| 术语                       | 解释                                                           |
| -------------------------- | -------------------------------------------------------------- |
| Bi-Encoder                 | Query/Doc 独立编码，点积相似度                                 |
| Cross-Encoder              | Query+Doc 联合编码，精度高但慢                                 |
| Late Interaction           | 多向量 + MaxSim，ColBERT 范式（BGE-M3 mul 头同族，见专题报告） |
| Sparse / Lexical Embedding | 词表维稀疏权重；可学习版 ≈ 神经 BM25（BGE-M3 / SPLADE）       |
| InfoNCE                    | 对比学习损失，Info Noise-Contrastive Estimation                |
| MRL                        | Matryoshka Representation Learning，多维度嵌套                 |
| ANN                        | Approximate Nearest Neighbor，近似最近邻                       |
| HNSW                       | Hierarchical Navigable Small World，图索引                     |
| RRF                        | Reciprocal Rank Fusion，倒数排名融合                           |
| Hard Negative              | 与 anchor 相似但标签不同的困难负样本                           |
| MTEB                       | Massive Text Embedding Benchmark                               |
| BEIR                       | Benchmarking IR zero-shot                                      |
| RAG                        | Retrieval-Augmented Generation                                 |

## 附录 B：资源链接

### 本仓库专题报告（按主题）

**BGE 全家桶**（RetroMAE → C-Pack → M3 → EN-ICL → gemma2 → Reranker）

- [BGE-CPack详解](BGE-CPack详解.md)：C-Pack / BGE 全家桶起点，C-MTEB + C-MTP + 三阶段训练
- [BGE-M3三功能统一详解报告](BGE-M3三功能统一详解报告.md)：Dense + Sparse + Multi-vec 三头联合 + Self-KD
- [BGE-EN-ICL详解](BGE-EN-ICL详解.md)：Mistral-7B + causal + ICL few-shot 训练；MTEB 71.24
- [BGE-multilingual-gemma2详解](BGE-multilingual-gemma2详解.md)：Gemma-2-9B 骨干；MIRACL 74.1 SOTA
- [BGE-Reranker详解](BGE-Reranker详解.md)：v2-m3 / v2-gemma / v2.5-gemma2-lightweight（深度+宽度双压缩）

**基石短深读**

- [无监督对比检索三部曲](无监督对比检索三部曲_SimCSE-Contriever-Condenser.md)：SimCSE / Contriever / Condenser + coCondenser
- [RetroMAE与DupMAE详解](RetroMAE与DupMAE详解.md)：非对称 MAE + BoW 头
- [INSTRUCTOR详解](INSTRUCTOR详解.md)：指令化嵌入开山
- [CLIP详解](CLIP详解.md)：图文双塔基石，对称 InfoNCE + WIT 4 亿对
- [SigLIP与SigLIP2详解](SigLIP与SigLIP2详解.md)：sigmoid loss + LocCa + SILC/TIPS + NaFlex

**训练与损失**

- [对比学习与InfoNCE精讲](对比学习与InfoNCE精讲.md)：损失演化专题（Metric learning → InfoNCE → 各家变体）
- [难负例挖掘工业实践](难负例挖掘工业实践.md)：假负例治理与刷新
- [Embedding蒸馏技术详解](Embedding蒸馏技术详解.md)：CE→BE / vec / logit 三种蒸馏信号

**LLM-Embedding & MLLM-Embedding**

- [LLM-Embedding冲榜路线](LLM-Embedding冲榜路线_E5Mistral-NVEmbed-GritLM-SFR-Arctic-Stella.md)：E5-Mistral / GritLM / NV-Embed-v2 / SFR-2R / Arctic v1&2 / Stella 六篇合写
- [MLLM通用多模态嵌入](MLLM通用多模态嵌入_GME-VLM2Vec-BGEVL.md)：GME / VLM2Vec+MMEB / BGE-VL+MegaPairs 三篇合写

**前沿方向**

- [前沿短深读集合](前沿短深读_LateChunking-Vec2Vec-ModernBERT-DINOv2v3-Qwen3-Seed-ViDoRev2.md)：Late Chunking / Vec2Vec / ModernBERT / DINOv2+v3 / Qwen3-Embedding / Seed1.5 / ViDoRe v2

**Late Interaction 族**

- [ColBERT详解](ColBERT详解.md) · [ColBERTv2详解](ColBERTv2详解.md) · [ColPali详解](ColPali详解.md) · [ColQwen系列详解](ColQwen系列详解.md)

**稠密文本 Embedding**

- [E5详解](E5详解.md) · [GTE系列详解](GTE系列详解.md) · [Nomic-Embed详解](Nomic-Embed详解.md) · [LLM2Vec详解](LLM2Vec详解.md) · [Conan-embedding详解](Conan-embedding详解.md) · [Conan-embedding-v2详解](Conan-embedding-v2详解.md) · [QZhou-Embedding详解](QZhou-Embedding详解.md) · [Token-Prepending详解](Token-Prepending详解.md) · [Jasper-Token-Compression-600M详解](Jasper-Token-Compression-600M详解.md)

**Jina 系列**

- [Jina系列总览](Jina系列总览.md) · [v2](Jina-embeddings-v2详解.md) · [v3](Jina-embeddings-v3详解.md) · [jina-clip](jina-clip系列详解.md) · [v4](Jina-embeddings-v4详解.md) · [v5-text](Jina-embeddings-v5-text详解.md) · [v5-omni](Jina-embeddings-v5-omni详解.md)

**难负例算法与数据**

- [ANCE详解](ANCE详解.md) · [RocketQA详解](RocketQA详解.md) · [NV-Retriever详解](NV-Retriever详解.md) · [LLM-DA文本行人检索数据增强详解](LLM-DA文本行人检索数据增强详解.md) · [DeVE-QA稠密视频事件问答详解](DeVE-QA稠密视频事件问答详解.md) · [InternLM2数据处理与过滤详解](InternLM2数据处理与过滤详解.md)

### 外部资源

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)（含 MTEB v2 / MMTEB）
- [BEIR Leaderboard](https://github.com/beir-cellar/beir)
- [ViDoRe v2 Leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard)
- [MMEB Leaderboard](https://tiger-ai-lab.github.io/VLM2Vec/)
- [Sentence Transformers](https://www.sbert.net/)
- [FlagEmbedding (BGE)](https://github.com/FlagOpen/FlagEmbedding)
- [BGE-M3 官方文档](https://bge-model.com/bge/bge_m3.html)
- [MTEB GitHub](https://github.com/embeddings-benchmark/mteb)
- [ColPali / Vidore](https://github.com/illuin-tech/colpali)
- [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding)

---

*本报告基于 2024–2026 年公开论文、MTEB Leaderboard、技术博客与开源项目整理，仅供研究参考。*
