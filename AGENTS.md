# ai-learn Agent 开发规范

面向 Cursor / Agent 的调研与学习工程约定。与 `.cursor/rules/paper-research.mdc` 配合使用。

---

## 1. 项目职责

| 路径 | 职责 |
|------|------|
| `docs/reports/<topic>/` | 研究报告 Markdown（**权威正文**，入库） |
| `docs/papers/<topic>/` | 论文 PDF 本地副本（**整树不入库**） |
| `embedding/`、`torch/`、`refs/` | 练习笔记 / 代码参考 |

专题目录与报告对齐：`embedding`、`lora`、`distillation`、`Qwen` 等。

---

## 2. 论文 PDF 副本

研究某篇论文时，**必须**在本地保留 PDF：

```
docs/papers/<topic>/<Name>/{<Full Title>_<arxiv>.pdf, zh.md, figs/}
docs/papers/<topic>/<Series>/<Paper>/{<Full Title>_<arxiv>.pdf, zh.md, figs/}
```

- **文件夹**以方法名（若很知名）或文章名命名，**不要带 arXiv 编号**。同一方法/系列的多篇论文放在同一父文件夹下，每篇自含 PDF、中译、图片集。
- **PDF 文件名**用论文英文全名，有 arXiv 则在后缀加编号（如 `_2308.03281.pdf`）；**禁止**命名为 `paper.pdf`。无 arXiv 时只用全名。

示例：

- `docs/papers/embedding/LLM2Vec/LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders_2404.05961.pdf`
- `docs/papers/embedding/GTE/GTE/Towards General Text Embeddings with Multi-stage Contrastive Learning_2308.03281.pdf`
- `docs/papers/embedding/GTE/mGTE/mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval_2407.19669.pdf`

规则：

- 目录按报告专题分；不存在则创建
- **整树 `docs/papers/` 已在 `.gitignore`，禁止 `git add` PDF**
- 报告 MD **不要**写 `local PDF`、本地 PDF 路径、中译落盘路径；用 arXiv / ACL / 官方链接即可
- 配图优先从 **arXiv HTML / ar5iv** 下载原图到对应 `figs/`；无 HTML 原图时再 PDF 裁切兜底（见 `docs/papers/embedding/_fetch_html_figures.py`）

---

## 3. 研究报告 → Notion（主动导入）

写完或重大更新 `docs/reports/**` 中的**论文/算法深读**后，**主动**将全文导入 Notion，并保证 **MD 与 Notion 正文一致**（格式可按 Notion 转换，信息不得删减摘要化）。**图必须同步**：Notion 页不得只有文字、漏掉报告中的关键图。

### 3.1 Embedding（已知落点，勿再问）

1. 全文写入 **Embedding&Rerank → Reference → Paper** 数据库  
   （data source：`collection://0391fb9a-5dfb-4b28-b7dc-84cb27713890`）
2. 若属模型/算法卡片，在 **ModelCard** 对应 tab（如 Others）加 `<mention-page>` 引用
3. Paper 属性建议：`Type=Paper`，`DOMAIN` 含 `LLM`，`SubDomain` 含 `Embedding`，`URL`=论文链接，`简述`=一句话，`Readed` 按实情

### 3.2 非 Embedding（lora / distillation / Qwen 等）

**首次导入前必须询问用户** Notion 父页面 / 数据库落点；不得自行猜测。

### 3.3 Notion 删除禁令

- **禁止删除**任何既有 Notion 页面或他人/历史内容
- 任何删除（含自己误写的大块清理）**必须先问用户并得到明确同意**
- 默认只做：新建、追加、用全文 `replace_content` **覆盖自己负责导入的那一页**（覆盖前确认是本报告对应页）

---

## 4. 算法类 Notion / MD 版式

### 4.1 配图（之后新建/大改必做；存量不必整页补）

研究报告与同步到 Notion 的页面**必须有图**，至少包含：

1. **原图**：优先 arXiv HTML / ar5iv 抽出的论文原图（非 PDF 糊裁）；关键方法图、主结果图不得省略  
2. **对应解释**：每张图下方（或紧邻）用中文说明「图在讲什么、和本文论点如何对应」——不要只贴图不写字，也不要只写「见图 N」而无图  

落盘建议（便于入库与 Notion 上传）：

```
docs/reports/<topic>/figures/<短名>/fig01.png
```

从 `docs/papers/..._figs/` 拷贝关键图到上述目录后嵌入报告；同步 Notion 时**上传同一批图**，保持 MD ↔ Notion 图文一致。

### 4.2 文首 Callout（精简）

算法/模型深读页（及对应 MD 文首元信息）须有 callout（MD 可用引用块等价），**精简明了**，至少含：

| 字段 | 说明 |
|------|------|
| paper | 论文 URL（arXiv / ACL / PDF） |
| code / project | 官方仓库或项目页（若有） |
| refs | 极少数经典前置工作链接（若有，1–3 条） |
| backbone | 模型骨干 |
| date | 发行 / 论文时间 |
| modality | 文本 / 图像 / 多模态等 |
| languages | 支持语言（若可知） |

可按主题追加短字段（如 pooling、开源协议、参数量），仍保持 callout **短小**，细节放正文。

### 4.3 Embedding 类正文必答

正文（MD = Notion）须写清：

1. **训练数据**：用了哪些数据 / 合成策略 / 规模量级  
2. **评测 benchmark**：MTEB / CMTEB / 词级 CoNLL 等  
3. **对比方法**：和谁比、结论是什么  
4. **数据集简介**：关键训练集或评测集用一两段说明（不必百科全文）

另按论文写清：方法步骤、损失/公式、消融、可迁移实践。

### 4.4 章节标题不加数字序号（增量）

**之后**新建或大改的报告 / Notion 页：标题用语义名即可（如 `## 实验用到的数据集`），**不要**写 `## 6. …`。交叉引用也用语义（「见『实验结果』注」），避免依赖序号。

存量页（已带序号）**不必**为去序号而整页重排；仅在该页后续有大改时顺带去掉即可。目的：插入章节时不必连带改前后标题，降低 Notion 同步代价。

### 4.5 MD ↔ Notion 一致

- **Markdown 为源文件**；Notion 为同步展示
- 导入后不得只留摘要版；用户要求全文则必须全文
- 改 MD 后若已存在对应 Notion 页，应同步更新 Notion

---

## 5. Agent 操作权限

| 操作 | 默认 |
|------|------|
| 读报告 / 写报告 MD / 下载 PDF 到 `docs/papers/` | ✅ |
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
| [.gitignore](.gitignore) | 含 `docs/papers/` |
