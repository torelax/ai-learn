<!-- This is a placeholder for the README file -->

# AI-Learn

AI-Learn 是一个开源的人工智能学习项目，旨在帮助开发者和学习者快速入门并实践 AI 相关的知识与技术。

## 特性

- 覆盖主流 AI 领域（如机器学习、深度学习、自然语言处理等）
- 提供丰富的代码示例和实践项目
- 结构清晰，易于扩展和维护
- 适合初学者和进阶者

## 安装

```bash
git clone https://github.com/yourname/ai-learn.git
cd ai-learn
```

## 使用方法

1. 按照各子目录的 README 指引运行示例代码。
2. 可根据需求修改和扩展项目内容。

## 目录结构

```
ai-learn/
├── AGENTS.md                    # Agent / 论文研究与 Notion 同步规范
├── .archify/                    # 本地 Archify 图与 JSON 图源，按模块分目录，不入库
├── docs/                        # 知识库（Markdown 为源）：embedding/ · llm/ · post-training/
├── mkdocs.yml                   # 知识站点配置；产物 site/ 不入库
├── scripts/papers/              # 论文工具：配图、PDF 压缩、表格裁切
├── scripts/site/                # 站点构建 / 预览 / 部署（scripts/site/site.sh）
├── embedding/                   # 笔记与练习
├── agent/                       # Agent 练习；`agent/phistory/` 是独立 git 仓库（本仓库不追踪）
├── torch/                       # PyTorch 练习
├── transformers/                # Transformer、BERT 与 Hugging Face 练习
├── llm/                         # 大模型注意力、推理与 KV cache
├── ...
```

论文深读须把 PDF / 中译 / 配图与报告 MD 放在同一 `docs/<领域>/<专题>/` 目录；具名报告 MD 为阅读笔记权威正文，按 [阅读笔记指南](docs/paper-reading-guide.md) 组织，并主动全文同步到单个 Notion 阅读页（不附审核清单等子页）（Embedding 落点见 `AGENTS.md`；其他专题先问）。

Archify 交互图与可编辑 JSON 图源统一放在 `.archify/<模块路径>/`（如 `.archify/agent/hermes/`、`.archify/agent/pi/`），不纳入 Git 追踪。

## 知识站点

`docs/` 的 Markdown 构建成可浏览、可搜索的 HTML 站点（MkDocs Material，公式由 MathJax 渲染）：

```bash
scripts/site/site.sh serve   # 本地预览 http://127.0.0.1:8732
scripts/site/site.sh build   # 构建到 site/
```

只需 Python ≥ 3.10，依赖自动安装；部署到 GitHub Pages 等细节见 [scripts/README.md](scripts/README.md)。

## 贡献

欢迎提交 issue 和 pull request 共同完善项目！

## 许可证

本项目基于 MIT License 开源。
