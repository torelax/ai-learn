# Jina Embedding 系列总览

> 谱系索引：从文本长上下文 Bi-Encoder，到 CLIP 双塔，再到 MLLM 统一多模态与 Locked Aligned Towers。
> 细节机制见各「详解」；本文只回答 **演进线索 / 架构类属 / 何时用谁**。

---

## 1. 一句话定位

Jina AI 的 Embedding 产品线不是单点刷榜，而是按 **文本 → 图文双塔 → 统一 MLLM → 蒸馏小模型 + 全模态锁定对齐** 迭代。选型时务必区分：`jina-clip-*`（CLIP dual-tower）≠ `jina-embeddings-v4`（MLLM，可 late interaction）≠ `jina-embeddings-v5-omni`（锁定文本塔 + 视听投影）。

---

## 2. 谱系时间线（v1 → v5-omni）

```text
v1 (2307.11224)          句向量起步；BERT 系短上下文
        │
        ▼
v2 (2310.19923)          ALiBi / JinaBERT；8K 长文档；分语种卡
        │
        ├──────────────────┐
        ▼                  ▼
v3 (2409.10173)      jina-clip v1/v2
  任务 LoRA + MRL      (2405.20204 / 2412.08802)
  多语 XLM-R           EVA02 + Jina-XLM-R 双塔
        │
        ▼
v4 (2506.18902)          Qwen2.5-VL-3B；文/图/PDF；单向量 + 可选多向量
        │
        ▼
v5-text (2602.15547)     蒸馏 Qwen3-Embedding；nano / small；任务适配
        │
        ▼
v5-omni (2605.08384)     GELATO：锁定 v5-text + 视听塔；文/图/视/音同空间
```

| 代际 | 代表模型 | 论文 | 骨干 / 要点 | 模态 | 参数量级 | 上下文 | 维度（MRL） |
|------|----------|------|-------------|------|----------|--------|-------------|
| **v1** | jina-embeddings-v1 | [2307.11224](https://arxiv.org/abs/2307.11224) | BERT 系句向量；奠定对比训练配方 | 文本 | 小–中 | ≤512 档 | — |
| **v2** | jina-embeddings-v2-base-* | [2310.19923](https://arxiv.org/abs/2310.19923) | JinaBERT + ALiBi；en/de/es/zh 分卡 | 文本 | ~137–161M | **8K** | 768 |
| **v3** | jina-embeddings-v3 | [2409.10173](https://arxiv.org/abs/2409.10173) | XLM-R-24L；**任务 LoRA**；89 语 | 文本 | **570M** | **8K** | 1024→32 |
| **CLIP** | jina-clip-v1 / v2 | [2405.20204](https://arxiv.org/abs/2405.20204) / [2412.08802](https://arxiv.org/abs/2412.08802) | EVA02 + Jina-XLM-R；图文+纯文兼顾 | 图+文 | v2≈0.87B | 文本 8K | 1024→64 |
| **v4** | jina-embeddings-v4 | [2506.18902](https://arxiv.org/abs/2506.18902) | **Qwen2.5-VL-3B**；任务 LoRA；单/多向量 | 文+图+PDF | **3.8B** | **32K** | 2048（多向量 128） |
| **v5-text** | v5-text-nano / small | [2602.15547](https://arxiv.org/abs/2602.15547) | EuroBERT / Qwen3-0.6B；**任务靶向蒸馏** | 文本 | 239M / 677M | 8K / **32K** | 768|1024→32 |
| **v5-omni** | v5-omni-nano / small | [2605.08384](https://arxiv.org/abs/2605.08384) | **GELATO** 锁定文本塔 + 视听投影 | 文/图/视/音 | ~1.0–1.7B 级 | 同 text | 与 text 对齐 |

**关于 v1**：无单独深读文件；机制与对比训练脉络在本总览与 [Jina-embeddings-v2详解.md](v2/Jina-embeddings-v2详解.md) 中覆盖即可。论文：[arXiv:2307.11224](https://arxiv.org/abs/2307.11224)。

---

## 3. 架构类属对照

四类交互范式（与《[图文Embedding模型技术综述](../图文Embedding模型技术综述.md)》一致）：

| 架构类 | 编码方式 | 打分 | Jina 代表 | 典型代价 |
|--------|----------|------|-----------|----------|
| **Bi-Encoder（文本）** | 单塔独立编码 $q$、$d$ → 单向量 | $\cos(q,d)$ / 点积 | v1 / v2 / v3 / v5-text | 索引便宜；细粒度弱 |
| **CLIP Dual-Tower** | 视觉塔 $f_I$、文本塔 $f_T$ 独立 | $\cos(f_I(I), f_T(T))$ | **jina-clip-v1/v2** | CPU/低成本图文检索友好 |
| **MLLM Bi / Late** | VLM 统一吃文/图 token；可出单向量或多向量 | 单向量点积；或多向量 MaxSim | **v4** | 质量高；算力与许可证（Qwen Research）需评估 |
| **Locked Aligned Towers** | **冻结**文本塔 + 冻结视听编码器；只训轻量 projector | 投影后与文本同空间 | **v5-omni（GELATO）** | 文本向量与 v5-text **bit-identical**；训练参数占比极低（论文约 $0.35\%$） |

```text
                    ┌── Bi-Encoder ──────── v1 / v2 / v3 / v5-text
Jina 产品线 ────────┼── CLIP Dual-Tower ─── jina-clip-v1 / v2
                    ├── MLLM（可 late）──── v4
                    └── Locked Towers ───── v5-omni
```

易混点：

1. **clip-v2 vs v4**：前者是标准双塔；后者是 Qwen2.5-VL 骨干的统一多模态嵌入，可切 late interaction。
2. **v4 vs v5-omni**：v4 会改动/适配 MLLM 本身；v5-omni **不改** v5-text 权重，只外挂投影，保证旧文本索引可复用。
3. **任务 LoRA**：自 v3 起成为产品线标配（retrieval / classification / clustering / text-matching）；推理时选适配器，而非只靠 instruction 前缀。

---

## 4. 何时用哪个模型

| 场景 | 优先选型 | 理由 |
|------|----------|------|
| 纯文本、长文档（合同/论文）、预算紧 | **v2** 或 **v5-text-nano** | v2 成熟 8K；nano 蒸馏后小模型性价比高 |
| 多语文本 + 检索/聚类/分类要分任务最优 | **v3** 或 **v5-text-small** | 任务 LoRA + MRL；small 上下文可达 32K |
| 只做图搜图 / 文搜图、要低延迟 | **jina-clip-v2** | CLIP 双塔；多语文本塔仍强 |
| PDF / 图表 / 扫描页 + 长文统一索引 | **v4**（必要时开多向量） | MLLM 吃页图；单/多向量可切换 |
| 已有 v5-text 文本库，要加图/视/音查询 | **v5-omni**（同档 nano/small） | 文本索引不用重建；几何保持对齐 |
| 遗留英文短句 / 对照论文 | v1 仅作历史；生产用 v2+ | v1 无独立深读必要 |

粗判决策树：

```text
只要文本？
  ├─ 要多模态页图/PDF 精度 → v4
  ├─ 要任务分适配 + 多语 → v3 / v5-text
  └─ 只要 8K 长文基线 → v2
只要图文、算力紧？ → jina-clip-v2
已有文本索引、加视听？ → v5-omni（对齐同档 text）
```

---

## 5. 代际机制速记

| 代际 | 核心机制（一句话） | 公式/直觉 |
|------|-------------------|-----------|
| v1 | 对比句向量 | InfoNCE 类：$\mathcal{L}=-\log\frac{e^{s(q,d^+)/\tau}}{\sum e^{s(q,d)/\tau}}$ |
| v2 | 长上下文位置偏置（ALiBi） | 注意力加距离偏置，使训练短、测试长 |
| v3 | 任务 LoRA + MRL | 同一底座，按 task id 挂低秩适配；$d$ 可截断至 32 |
| CLIP | 图文联合对比；文本侧不牺牲 MTEB | 同时优化 $I$–$T$ 与 $T$–$T$ |
| v4 | MLLM 统一编码；可选 late interaction | 单向量或 token/patch 多向量 MaxSim |
| v5-text | 任务靶向蒸馏 | 蒸馏 + 任务对比 $>$ 纯蒸馏 $>$ 纯对比 |
| v5-omni | GELATO 锁定塔 | 只训 projector；文本输出 $\equiv$ v5-text |

蒸馏细节另见《[Embedding蒸馏技术详解](../Embedding蒸馏技术详解.md)》§7。

---

## 6. 与相邻系列的边界

| 对照对象 | 差异（一句话） |
|----------|----------------|
| **BGE / E5 / GTE** | 同为文本 Bi-Encoder 族；Jina 更早押长上下文与 **任务 LoRA 产品化**，后走 MLLM / Omni |
| **Nomic Embed** | 同为 8K 开源长文；Nomic 强调数据可复现，Jina v2 同期竞品（见 [Nomic-Embed详解](../Nomic-Embed/Nomic-Embed详解.md)） |
| **ColPali / ColQwen** | 专攻页图多向量 MaxSim；v4 可 late，但定位是「统一多模态嵌入」而非纯视觉文档族 |
| **Qwen3-Embedding** | v5-text 的教师侧；学生侧用蒸馏换小参数，不是另起一套 LLM 微调范式 |
| **Seed / API 多模态** | 闭源便利 vs Jina 可私有化；架构上勿把 API 与 clip-v2 / v4 / omni 混为一谈 |

工程侧提醒：v4 权重常受 **Qwen Research License** 约束；上线前核对许可与商用条款。

---

## 7. 深读入口（本仓库）

| 文档 | 对应论文 | 状态 |
|------|----------|------|
| [Jina系列总览.md](Jina系列总览.md)（本文） | 全谱系索引 | ✅ |
| [Jina-embeddings-v2详解.md](v2/Jina-embeddings-v2详解.md) | [2310.19923](https://arxiv.org/abs/2310.19923)（含 v1 [2307.11224](https://arxiv.org/abs/2307.11224) 背景） | 深读 |
| [Jina-embeddings-v3详解.md](v3/Jina-embeddings-v3详解.md) | [2409.10173](https://arxiv.org/abs/2409.10173) | 深读 |
| [jina-clip系列详解.md](clip/jina-clip系列详解.md) | [2405.20204](https://arxiv.org/abs/2405.20204) / [2412.08802](https://arxiv.org/abs/2412.08802) | 深读 |
| [Jina-embeddings-v4详解.md](v4/Jina-embeddings-v4详解.md) | [2506.18902](https://arxiv.org/abs/2506.18902) | 深读 |
| [Jina-embeddings-v5-text详解.md](v5-text/Jina-embeddings-v5-text详解.md) | [2602.15547](https://arxiv.org/abs/2602.15547) | 深读 |
| [Jina-embeddings-v5-omni详解.md](v5-omni/Jina-embeddings-v5-omni详解.md) | [2605.08384](https://arxiv.org/abs/2605.08384) | 深读 |

总调研时间线亦见《[Embedding调研报告](../Embedding调研报告.md)》§2.1.7。

---

## 8. 参考文献（论文主键）

1. Günther et al. *Jina Embeddings*. [arXiv:2307.11224](https://arxiv.org/abs/2307.11224), 2023.  
2. Günther et al. *Jina Embeddings 2*. [arXiv:2310.19923](https://arxiv.org/abs/2310.19923), 2023.  
3. Sturua et al. *jina-embeddings-v3*. [arXiv:2409.10173](https://arxiv.org/abs/2409.10173), 2024.  
4. Koukounas et al. *Jina CLIP*. [arXiv:2405.20204](https://arxiv.org/abs/2405.20204), 2024.  
5. Koukounas et al. *jina-clip-v2*. [arXiv:2412.08802](https://arxiv.org/abs/2412.08802), 2024.  
6. Günther / Jina et al. *jina-embeddings-v4*. [arXiv:2506.18902](https://arxiv.org/abs/2506.18902), 2025.  
7. Akram et al. *jina-embeddings-v5-text*. [arXiv:2602.15547](https://arxiv.org/abs/2602.15547), 2026.  
8. *jina-embeddings-v5-omni* (GELATO). [arXiv:2605.08384](https://arxiv.org/abs/2605.08384), 2026.  
