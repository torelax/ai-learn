# 领域专用 Embedding 适配实践

> 读者前提：读过主文《[Embedding调研报告](Embedding调研报告.md)》§5 课表、§7.4 / §10 文搜图分叉。
> 读完收获：能把 SPECTER / MedCPT / 代码 IR / 法律 / 推荐双塔的案例，收成一套 **Stage 2/4 领域适配** 清单，并落到文搜图。
> 挂接：主文 §5.1 / §5.6「最小可行」、§10.4；各领域深读见文末索引。

主文已经写过两句课表，但没独立成篇：

- 「领域适配尽量在 Stage 2/4 做，而不是一上来就用小领域数据从头训」
- 「最小可行：开源 Embedding 续训 → 领域弱配对 → 正对 + 过滤难负例至验证饱和 → 可选 MRL/蒸馏」

下面用各领域论文把这两句拆开，并显式对齐 **文搜图**（text → image，共享向量空间）。电商图搜图、金融 FinBERT 没有单独成篇，放在本文短节。

---

## 领域专用在优化什么

通用 Embedding（E5 / BGE / GTE / CLIP）优化的是 **开放域平均相关**。领域专用要改的是三件事中的至少一件：

| 缺口 | 例子 | 只换更大通用模型够不够 |
| --- | --- | --- |
| **骨干词表 / 视觉先验不对** | 基因名、法条号、SKU、商品材质纹理 | 往往不够，要领域 MLM 或领域视觉塔 |
| **相关的定义不是开放域「像」** | 引用才算相关、判例法理相关、同款不同色不算命中 | 不够，要改正负例 |
| **任务格式撕裂同一个向量** | 分类 vs 近邻 vs 短 query 检索 | 不够，要 adapter / instruct / 分塔 |

SPECTER 改的是第二项（引用边）；LEGAL-BERT 改的是第一项；SPECTER2 改的是第三项；MedCPT 第一+第二（PubMedBERT + 点击）；CodeR 主要是第二项的数据覆盖；YouTube 是第二项在推荐分布下的负采样。

文搜图三项都会碰到：CLIP 开放域图文对 ≠ 商品标题搜主图；「像」≠ 同款；图搜图向量不能拿去给文本 query 用。

---

## 一张对照表：各领域的正例、难负例、评测

| 领域 | 骨干 | 正例从哪来 | 难负例 | 必须保留的稀疏 / 结构 | 权威评测 | 深读 |
| --- | --- | --- | --- | --- | --- | --- |
| 科学文献 | SciBERT | 引用边 | 引用的引用但未直连 | 论文 ID 可倒排 | SciDocs / SciRepEval | [SPECTER系列](SPECTER/SPECTER系列详解.md) |
| 生物医学 | PubMedBERT | 检索点击 | 双塔 MIPS local neg | PMID / 基因 ID | BEIR 生物医学、RELISH | [MedCPT](MedCPT/MedCPT详解.md) |
| 代码 | 代码预训练或通用+合成 | docstring、同语义、合成 query | PercPos 过滤后的近邻代码 | 符号精确匹配可选 | CoIR / CodeRAG | [代码IR合写](Code-IR/代码检索Embedding合写.md) |
| 法律 | LEGAL-BERT 再结构 | 先例标注 / 结构重建 | 事实像、法律关系不同 | **条号案号 BM25** | LeCaRD / COLIEE | [法律合写](Legal/法律检索Embedding合写.md) |
| 推荐召回 | ID embedding + MLP | 下次观看 / 点击 | 全库采样；精排才用曝光未点 | 无；冷启动靠内容塔 | 线上 Recall / 时长 | [YouTube DNN](YouTube-DNN/YouTube-DNN双塔详解.md) |
| 金融（短） | FinBERT 类 | 研报–公告、问答 | 同行业不同事件 | 代码、财报科目 | FiQA 等 | 见下节 |
| 电商文搜图 | CLIP / SigLIP 续训 | 标题–主图、搜索点击图 | 同款不同色、同品类不同 SPU | **SKU / 货号** | 自建 Text→Image Recall | 见「对齐文搜图」 |

规律：

1. **正例优先用领域里已经存在的边**（引用、点击、同款、观看），人工三元组是校准不是主粮。
2. **难负例必须落在任务相关的边界上**，不是随机。
3. **ID 类精确键永远 Hybrid**，Dense 不负责当主键。

---

## 方法论：为什么是 Stage 2/4

主文 Stage 0–4 是开放域课表。领域适配插在哪：

```text
Stage 0  通用或领域 MLM / 图文对比骨干     ← 可换 SciBERT / PubMedBERT / CLIP
Stage 1  开放域弱监督（可选，已有开源权重则跳）
Stage 2  领域弱配对 + 过滤难负例           ← 主战场
Stage 3  多任务（分类/STS）要小心稀释检索
Stage 4  短课：adapter、MRL、蒸馏、instruct
```

**不要 Stage 0 用 5 万条商品图从零训 ViT。** SPECTER 从 SciBERT 起步；MedCPT 从 PubMedBERT 起步；LEGAL-BERT 消融写明 continue-pretrain 通常优于 scratch；CodeR 也是在已有 embedder 上灌合成三元组。

**不要只有 Stage 3。** 把领域分类（品类、MeSH）混进检索对比，可能抬高分类、拉低 Recall——SPECTER2 干脆拆 adapter，就是为了不让一个 [CLS] 同时当分类特征和 ANN 键。

**最小可行（落到命令级直觉）**：

1. 选开源双塔（文搜图：CLIP / SigLIP / 中文 CLIP，不要 DINO-only）。
2. 接领域弱配对：标题–主图、搜索词–点击图、同款多图。
3. 挖难负例：同款不同色、同品类不同 SPU、视觉近邻但 SKU 不同；用正例分做 PercPos / margin 过滤假负。
4. 领域验证集饱和（Text→Image Recall@K **单独报**）再考虑蒸馏 / MRL / 加 Cross-Encoder 精排。
5. SKU、货号、条码走 BM25 / 倒排，和 Dense 做 Hybrid。

验证不饱和时加数据、改负例，不要加 7B。MedCPT 用 110M 级领域双塔零样本赢 GTR-XXL，已经把这条打过一次。

---

## 案例串读（只留可搬的机制）

### 科学：图结构 = 免费相关标注

SPECTER 把引用当正例、二跳未引当难负。文搜图对应：

- 正例：同 SPU、运营标的「同款」、搜索点击
- 难负：同品类、同色不同款、套图里的配饰图

SPECTER2 的 adapter：若你同时做 **品类分类、图搜图、文搜图**，不要共用一个投影头硬训——至少 query 侧 instruct 或分头。短文本搜图对应 SPECTER2 的 QRY adapter：标题塔 / 搜索词塔可以和「图搜图视觉塔」不是同一套投影。

### 生物医学：点击日志 = 百万弱监督

MedCPT 的 255M 点击对 = 你的搜索词–点击商品图。他们把「纯关键词点击」从精排数据里拿掉，避免 CE 重学 BM25。文搜图同样：只含货号的 query 不要主导对比损失，否则视觉塔学不会材质 / 款式。

双塔 in-batch + 精排 local hard neg 的两段课表，直接搬：先召回空间、再精排边界。

### 代码：评测集谎言与合成任务表

CoIR 证明 CodeSearchNet 上的虚高。文搜图不要只用「标题完全等于图 caption」的测试集——那是训练分布的克隆。要单列：短口语 query、属性 query（红色真丝）、否定（不要logo）、同款不同色。

CodeR 的 brainstorm：用 LLM 扩检索任务类型（按材质搜、按场景搜、按款搜），再生成 query，而不是把标题复制一千万遍。

### 法律：结构段 + Hybrid

SAILER 的 Fact / Reasoning 分家，对应商品 **标题 / 属性轴 / 主图**。只把主图和标题丢进 CLIP，属性「防泼水 / 尺码」进不了空间，就会出现「图对、货性不对」。条号 ≈ SKU：Dense 召回之后必须用稀疏打主键。

LEGAL-BERT 的 continue vs scratch：商品域视觉塔优先 **冻结或小 lr 续训 SigLIP**，不要随机初始化。

### 推荐：双塔服务形态与负例阶段

YouTube 候选生成告诉你：线上必须是 ANN 友好的 user/query 向量对 item 向量。文搜图 query 塔是文本，item 塔是图，服务形态已经是 Two-Tower。

负例：召回阶段用全库 in-batch / 随机难例；「曝光未点击」留给精排，并承认假负（看见没点可能是价格不是不相关）。Example age：上新、季节款要进特征或切时间窗。

---

## 金融与电商（不单独立篇的部分）

**FinBERT**（arXiv:1908.10063 等）：还是「领域语料 continue-pretrain」。金融检索额外有时序（同一公司不同季度不能当近邻正例）和必须命中的代码 / 科目。做法：FinBERT 当骨干 → 研报–公告 / 问句–段落对比 → Hybrid 保代码。不要用 MTEB 英文平均分代替 FiQA。

**电商图搜图**：视觉塔可 DINO / CLIP-vision；难负例是同款不同色、背景主导。这 **不能** 代替文搜图——没有文本塔就没有 text query。双任务时：图搜图同塔对比 + 文搜图跨塔对比，分开报指标（主文 §10.2）。Amazon 类双塔文献与 YouTube 同构，正例是点击 / 购买，域偏移是「棚拍 vs 买家秀」。

---

## 对齐文搜图：一份可执行清单

任务定义：query 为文本，gallery 为图像，分数 $\cos(f_t(q), f_v(x))$。路线必须是 CLIP / SigLIP / 中文 CLIP / MLLM-Emb，见主文 §10.2 与《[CLIP详解](CLIP/CLIP详解.md)》《[SigLIP与SigLIP2详解](SigLIP/SigLIP与SigLIP2详解.md)》。

| 步骤 | 做 | 不要 |
| --- | --- | --- |
| 骨干 | 开源图文双塔续训 | DINO-only；文本 RAG 模型 encode 图片路径 |
| 正例 | 标题–主图、搜索点击图、同款多图 | 纯货号 query 占满 batch |
| 难负 | 同款不同色、同品类不同 SPU、视觉 ANN 近邻 | 随机图；把同款多角度当负例 |
| 假负过滤 | PercPos / 正例 margin（NV-Retriever） | 把 top-k 全当负 |
| 结构 | 标题、属性、主图分字段或 instruct | 只 concat 成一句散文 |
| 稀疏 | SKU / 货号 BM25 Hybrid | 幻想 Dense 记住货号 |
| 课表 | Stage 2 领域对 → 可选 CE 精排 / 蒸馏 | 小数据从零训 ViT |
| 评测 | Text→Image Recall 与 Image→Image **分报** | 一个「多模态平均」混过去 |
| 服务 | 图像离线进 ANN，文本在线 | 每次把图和文拼成 Cross-Encoder 扫全库 |
| 多任务 | 图搜图 / 文搜图分头或分 instruct | 一个向量既当分类又当检索（SPECTER2 的反例） |

难负例刷新、假负审计的工程节奏仍看《[难负例挖掘工业实践](难负例挖掘工业实践.md)》；本文只规定 **领域边界上什么算难、什么算假**。

≤0.6B 学生 + 大教师伪标：仍按《[0.6B图搜图文搜图自训学习行动路线](0.6B图搜图文搜图自训学习行动路线.md)》，教师必须是已对齐图文空间的模型。

---

## 反模式

1. 用 MTEB / C-MTEB 平均分决定商品文搜图上线。
2. 领域只有 2 万对就从随机初始化训双塔。
3. 把图搜图验证集拿来报告文搜图。
4. Hard neg 用「随机另一个类目」——太易，梯度空。
5. 把同款三视图互相当负例——假负，撕碎邻域。
6. 训练有「以文搜图」instruct、线上裸 query（主文 §5.5）。
7. 指望 Embedding 记住 SKU；法律域已经证明条号必须 sparse。

---

## 深读与博客索引

| 类型 | 入口 |
| --- | --- |
| 科学 | 论文 [2004.07180](https://arxiv.org/abs/2004.07180)、[2211.13308](https://arxiv.org/abs/2211.13308)；[Ai2 SPECTER2 博文](https://allenai.org/blog/specter2-adapting-scientific-document-embeddings-to-multiple-fields-and-task-formats-c95686c06567)；[SPECTER系列详解](SPECTER/SPECTER系列详解.md) |
| 生物医学 | [2307.00589](https://arxiv.org/abs/2307.00589)；NCBI MedCPT 仓库；[MedCPT详解](MedCPT/MedCPT详解.md) |
| 代码 | CodeBERT / UniXcoder / CoIR / CodeR 四篇；[代码IR合写](Code-IR/代码检索Embedding合写.md) |
| 法律 | LEGAL-BERT、SAILER；[法律合写](Legal/法律检索Embedding合写.md) |
| 推荐 | RecSys 2016 YouTube DNN；[YouTube-DNN双塔详解](YouTube-DNN/YouTube-DNN双塔详解.md) |
| 通用课表 | 主文 §5；难负例专题；CLIP / SigLIP 详解 |

资料清单里的链接与本页同步维护。
