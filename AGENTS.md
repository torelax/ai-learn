# ai-learn Agent 开发规范

面向 Cursor / Agent 的调研与学习工程约定。与 `.cursor/rules/paper-research.mdc` 配合使用。

---

## 1. 项目职责

| 路径 | 职责 |
|------|------|
| `docs/reports/<topic>/` | 研究报告 Markdown（**权威正文**）与论文 PDF / 中译 / 配图，同一专题目录入库 |
| `embedding/`、`torch/`、`refs/` | 练习笔记 / 代码参考 |
| `agent/` | Agent 练习代码；`agent/phistory/` 为独立 GitHub 仓库，本仓库 `.gitignore` 不追踪 |

专题：`embedding`、`lora`、`distillation`、`Qwen` 等。

---

## 2. 专题目录布局

每个专题一份目录，深读 MD 与论文副本放在一起：

```
docs/reports/<topic>/
  README.md / 总览 / 合写 / 索引                # 跨多篇的总览仍在专题根
  figures/<短名>/                               # 报告嵌入图（Notion 上传用）
  <Name>/{详解.md, <Full Title>_<arxiv>.pdf, zh.md, figs/}
  <Series>/{系列详解.md, <Paper>/{PDF, zh.md, figs/}}
```

研究某篇论文时，优先在本地保留体积合理的 PDF：

- **文件夹**以方法名（若很知名）或文章名命名，**不要带 arXiv 编号**。同一方法/系列的多篇论文放在同一父文件夹下，每篇自含 PDF、中译、图片集。
- **PDF 文件名**用论文英文全名，有 arXiv 则在后缀加编号（如 `_2308.03281.pdf`）；**禁止**命名为 `paper.pdf`。无 arXiv 时只用全名。

示例：

- `docs/reports/embedding/LLM2Vec/LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders_2404.05961.pdf`
- `docs/reports/embedding/GTE/GTE/Towards General Text Embeddings with Multi-stage Contrastive Learning_2308.03281.pdf`
- `docs/reports/embedding/GTE/mGTE/mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval_2407.19669.pdf`

规则：

- 目录按专题分；不存在则创建
- 报告 MD **不要**写 `local PDF`、本地 PDF 路径、中译落盘路径；用 arXiv / ACL / 官方链接即可
- 配图优先从 **arXiv HTML / ar5iv** 下载原图到对应 `figs/`；无 HTML 原图时再 PDF 裁切兜底（见 `docs/reports/embedding/_fetch_html_figures.py`）
- **PDF 入库前先压**（见下）；按用户偏好，小文件保留，压缩后超过 5 MiB 的新论文 PDF 默认只保留官方下载链接，不强行入库

### 2.1 PDF 压缩（首次 commit 之前做）

arXiv 的 PDF 常把配图嵌到离谱的分辨率（DINOv3 原件里有 1515 dpi 的图，而屏幕阅读只需 ~150 dpi）。这类 PDF 里的**文字是矢量的**，降采样只动位图，正文、公式、图内标注逐像素不变。

```bash
cd docs/reports/embedding
python3 _shrink_pdf.py <新论文目录> --dry-run   # 先看一眼
python3 _shrink_pdf.py <新论文目录>             # 默认 /ebook
python3 _shrink_pdf.py DINO/DINOv3 --quality printer
```

- 默认 **`--quality ebook`**（图降到 150 dpi）。文字为主的论文和原件看不出区别，通常省 80–90%
- **图本身就是论文结论**时（DINOv2/DINOv3/CLIP 这类视觉论文）用 **`--quality printer`**（300 dpi）。ebook 会把照片细节明显糊掉
- 脚本只在「poppler 能读 + 页数不变 + 确实变小 + 图不低于 `--min-dpi`」时才替换；重复跑是空操作

**时机很关键：只压还没 commit 的 PDF。** 已入库的 PDF 压了反而让仓库变大——新 blob 加进去，旧的还留在历史里。存量大 PDF（`_shrink_pdf.py . --dry-run` 可列出，约 69 MB 可回收）留到将来重写历史时一次性处理。

---

## 3. 研究报告 → Notion（主动导入）

写完或重大更新 `docs/reports/**` 中的**论文阅读笔记**后，**主动**将该篇正文全文导入 Notion，并保证 **MD 与 Notion 正文一致**（格式可按 Notion 转换）。默认**一篇论文一个主阅读页，补充解读为其独立子页且入口置顶**；翻译仅保留在仓库，Notion 不放翻译正文、附件或链接。不自动导入研究过程、审核清单、选题依据，也不为这些资料创建子页或附件。**图必须同步**：Notion 页不得只有文字、漏掉报告中的关键图。

### 3.1 Embedding（已知落点，勿再问）

1. 全文写入 **Embedding&Rerank → Reference → Paper** 数据库
   （data source：`collection://0391fb9a-5dfb-4b28-b7dc-84cb27713890`）
2. 若属模型/算法卡片，在 **ModelCard** 对应 tab（如 Others）加 `<mention-page>` 引用
3. Paper 属性建议：`Type=Paper`，`DOMAIN` 含 `LLM`，`SubDomain` 含 `Embedding`，`URL`=论文链接，`简述`=一句话，`Readed` 按实情

### 3.2 非 Embedding（lora / distillation / Qwen 等）

**首次导入前必须询问用户** Notion 父页面 / 数据库落点；不得自行猜测。

已确认采用论文与技术主题两层组织（2026-10-19）：

- `docs/reports/lora/<Paper>/<方法名>详解.md` 是对应论文的中文译解与完整研读源，包含摘要译文、主文方法/实验/消融和必要的附录解释；保留该篇 PDF、原图及引用。原文译述与补充讲解分清，不把概要标为逐句全文翻译。
- LoRA、QLoRA、QA-LoRA 等论文页归入全局 **Paper**（`collection://0391fb9a-5dfb-4b28-b7dc-84cb27713890`），`Type=Paper`、`DOMAIN` 含 `LLM`、`SubDomain` 含 `Fine-Tune`、`TAG` 含 `Fine-Tuning` / `Peft`；`Readed` 按用户实际状态。
- **Fine-Tune**（page ID：`235a174cd8c68016b3d6c922d1a1d70e`）的 Ref 保留 Paper 链接视图，Paper tab 筛选论文类型及微调分类；技术总章在原页 LoRA 部分展开，用于跨论文讲解 rank、alpha、目标模块、优化、量化与部署，不局限于某篇原文。对应仓库源为 `docs/reports/lora/LoRA技术深度调研报告.md`。
- 提示微调跨论文综述继续保留既有 Fine-Tune 子页。技术总章与各篇原文研读相互引用；已有页面可按用户授权移入 Paper，保留 URL，避免重复创建。
- 本次及后续明确针对该总章的更新可局部替换 Fine-Tune 的 LoRA 技术内容；保留 Ref、其他技术章节和无关历史内容。已知落点无需重复询问。

### 3.3 Notion 删除禁令

- **禁止删除**任何既有 Notion 页面或他人/历史内容
- 删除（含自己误写的大块清理）须有用户明确授权；已有的具体清理指示可以直接执行，不重复询问。仅清理授权范围内的页面/内容
- 有用户正文时，首次补充接在其后，保留原有文字、图片、代码与子页；**后续只更新上次导入的范围，不能每次再追加整篇**。由 Agent 独立维护的主阅读页同样原位修订，保留页面 URL
- 优先定点更新；整页 `replace_content` 仅用于已确认由 Agent 维护的整页或用户明确授权的替换。更新前核对范围，保留所有既有子页/数据库的原生标签，更新后回读正文、公式、表格、图片与层级
- 用户已授权去重时，对照仓库历史和旧同步内容，合并确认由 Agent 重复导入的正文；保留用户新增、改写及无法确认归属的内容。明确授权不重复索要，不扩大清理到无关页面

---

## 4. 算法类 Notion / MD 版式

### 4.0 阅读笔记风格与文件职责

- 详细做法见 [论文阅读笔记指南](docs/paper-reading-guide.md)；标准样例是**仓库中的** `docs/reports/embedding/E5-Mistral/zh.md` 与 `docs/reports/embedding/Qwen3-Embedding/zh.md`，不是其 Notion 页或自行抽象出的通用模板。结合两篇原论文与各自笔记理解精炼的尺度
- 主阅读笔记是**基本保留原论文全部章节、小节与论证顺序的易读精炼版**。先读原文并对应各节，再压缩段落；原文独立讨论的数据构建、训练阶段、实验协议、消融、讨论等分别保留，不统一重排成固定章法，也不只补标题而省略技术内容
- 每节保留理解作者工作所需的定义、步骤、关键设置和证据；公式解释符号、维度与直觉，图表说明条件、比较与结果。必要的简短释义就近写清，跨论文分析、独立推导和实践建议放补充解读，不把主文写成观点文章、纠错清单或工作流水账
- `<方法名>详解.md` / 既有具名报告为忠于原文的论文精炼版及 Notion 主体同步源；另写 `<方法名>解读.md` 放补充分析。`zh.md` 为完整中英逐段对照，同段英文后紧接中文，不默认拆成英中各一篇。历史文件不因本规则自动改名或重排
- 采集、核查、同步日志和临时脚本留在仓库外的 `work/`；不创建或提交审核清单、选题笔记等过程文档，不在 Notion 创建相应页面。已有无关过程材料按用户授权清理
- 全文中英对照覆盖主文、附录、图表、脚注与参考文献，不能用摘要或阅读笔记代替；说明源版本与实际翻译/校订范围，保留公式和原始数字。Notion Paper 主阅读页以精炼版为主体，补充解读放在独立子页并将入口置于主页面开头；中英对照全文只在仓库保留，Notion 不放翻译正文、附件、折叠内容或链接

### 4.1 配图（之后新建/大改必做；存量不必整页补）

研究报告与同步到 Notion 的页面**必须有图**，至少包含：

1. **原图**：优先 arXiv HTML / ar5iv 抽出的论文原图（非 PDF 糊裁）；关键方法图、主结果图不得省略
2. **对应解释**：每张图下方（或紧邻）用中文说明「图在讲什么、和本文论点如何对应」——不要只贴图不写字，也不要只写「见图 N」而无图

落盘建议（便于入库与 Notion 上传）：

```
docs/reports/<topic>/figures/<短名>/fig01.png
```

从该篇 `figs/` 拷贝关键图到上述目录后嵌入报告；同步 Notion 时**上传同一批图**，保持 MD ↔ Notion 图文一致。

### 4.2 文首 Callout（精简）

算法/模型深读页及对应 MD 文首须有简短 callout：先用一句话说明方法思路或具体工作，再给论文、时间、代码等必要入口。按主题自然组织，可合并任务、骨干、语言等信息，不机械凑齐固定字段；详细规格放正文。

Callout 后紧接 1–2 个短段落的“论文简介”，约 200–350 字可作参考：背景/问题 → 核心方法步骤 → 代表性结果 → 主要局限。正文以作者叙事展开精炼全篇；个人补充解释另放独立解读。Notion 使用自动目录，不添加手写目录或 `table_of_contents`。

### 4.3 Embedding 类正文必答

正文（MD = Notion）须写清：

1. **训练数据**：用了哪些数据 / 合成策略 / 规模量级
2. **评测 benchmark**：MTEB / CMTEB / 词级 CoNLL 等
3. **对比方法**：和谁比、结论是什么
4. **数据集简介**：关键训练集或评测集用一两段说明（不必百科全文）

另按论文写清：方法步骤、损失/公式、训练与检索设置、实验和消融；跨论文与部署实践另放补充解读。

### 4.4 章节标题不加数字序号（增量）

**之后**新建或大改的报告 / Notion 页：标题用语义名即可（如 `## 实验用到的数据集`），**不要**写 `## 6. …`。交叉引用也用语义（「见『实验结果』注」），避免依赖序号。

存量页（已带序号）**不必**为去序号而整页重排；仅在该页后续有大改时顺带去掉即可。目的：插入章节时不必连带改前后标题，降低 Notion 同步代价。

### 4.5 MD ↔ Notion 一致

- **Markdown 为源文件**；Notion 为同步展示
- 导入后不得只留摘要版；用户要求全文则必须全文
- 改 MD 后若已存在对应 Notion 页，应同步更新 Notion
- 行内公式用 `$公式$`，公式内部不加反引号或中文引号；块级公式用独立行的 `$$`。转换时保护代码与公式，回读普通段落、列表、表格中的公式及主文是否重复

---

## 5. Agent 操作权限

| 操作 | 默认 |
|------|------|
| 读报告 / 写报告 MD / 下载 PDF 到 `docs/reports/<topic>/` | ✅ |
| 导入 / 更新 Notion（按 §3） | ✅ |
| 询问非 Embedding 的 Notion 落点 | ✅（必须先问） |
| 删除 Notion 页面或内容 | ❌ 除非用户明确同意 |
| `git commit` / `git push` | ❌ 除非用户明确要求 |
| 修改 git config | ❌ |

---

## 6. 相关文件

| 文件 | 用途 |
|------|------|
| [README.md](README.md) | 项目概览 |
| [docs/README.md](docs/README.md) | 文档索引 |
| [docs/reports/embedding/README.md](docs/reports/embedding/README.md) | Embedding 报告索引 |
| [.cursor/rules/paper-research.mdc](.cursor/rules/paper-research.mdc) | 论文研究强制规则 |
| [docs/paper-reading-guide.md](docs/paper-reading-guide.md) | 阅读笔记风格、文件职责与单页同步流程 |
| [.gitignore](.gitignore) | 忽略缓存 / 未完成目录等 |
