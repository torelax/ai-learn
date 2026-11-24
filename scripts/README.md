# scripts

| 目录 | 用途 |
|------|------|
| [`papers/`](papers/) | 论文处理工具：配图抓取、PDF 裁图、表格裁切、PDF 压缩 |
| [`site/`](site/) | 知识站点：把 `docs/` 的 Markdown 构建成 HTML（MkDocs Material） |

## 知识站点

`docs/` 下的 Markdown 是唯一源文件。站点构建到仓库根的 `site/`，不入库。

```bash
scripts/site/site.sh serve     # 本地预览 http://127.0.0.1:8732 ，改 md 自动刷新
scripts/site/site.sh build     # 构建到 site/，可放到任意静态托管
scripts/site/site.sh check     # 严格构建：断链、锚点错误即失败
```

`serve` 自己在内存里构建，**不需要先 build，也和 deploy 无关**；`build` 只在要拿到静态文件（部署或离线拷贝）时用。端口可用参数或 `SITE_ADDR` 指定。在 SSH 远程机器上预览时，`127.0.0.1` 只有服务器自己能访问，二选一：

- 本机开端口转发：`ssh -N -L 8732:127.0.0.1:8732 <user>@<server>`，然后打开 http://127.0.0.1:8732/
- 监听内网：`scripts/site/site.sh serve 0.0.0.0:8732`，然后打开 `http://<server-ip>:8732/`（同网段的人都能访问）

clone 后只需要 Python ≥ 3.10：有 [uv](https://docs.astral.sh/uv/) 时在临时环境里装依赖，不碰项目 `.venv`；没有 uv 时自动在 `.venv-site/` 建 venv。依赖版本固定在 `site/requirements.txt`。

| 文件 | 作用 |
|------|------|
| `mkdocs.yml`（仓库根） | 站点配置：主题、Markdown 扩展、导航、排除项 |
| `site/hooks.py` | 构建期把 GFM 写法规范成 Python-Markdown 能正确渲染的形式，只处理内存中的文本，不改源文件 |
| `site/overrides/assets/` | 中文排版 CSS、MathJax 配置 |
| `site/requirements.txt` | 固定版本的构建依赖 |
| `site/site.sh` | build / serve / check / deploy / clean 入口 |

渲染方面的约定：

- 公式用 MathJax 3 渲染（`$…$`、`$$…$$`），从 jsDelivr CDN 加载；字体用系统字体，不请求 Google Fonts。
- `hooks.py` 修正的写法：2 或 3 格缩进的嵌套列表；紧贴段落、前面没有空行的列表、表格和 `$$`；`$$` 块内部的空行；表格单元格公式里的 `|`；没有被引用的脚注定义；指向 `docs/` 之外的链接（改写成 GitHub 链接）。
- 论文 PDF、没有被任何 Markdown 引用的图片不发布（约 100 MB）。
- 侧边栏按目录自动生成，页面标题取文中第一个 H1；`zh.md` / `en.md` 显示为「<论文> · 中英对照 / 英文原文」。

### 部署到 GitHub Pages

站点发布在 https://torelax.github.io/ ，源仓库是 [torelax/torelax.github.io](https://github.com/torelax/torelax.github.io)。构建产物只进那个仓库，不进本仓库历史。

```bash
scripts/site/site.sh deploy        # 构建并显示变更摘要，确认后推送；加 -y 跳过确认
```

- Pages 仓库的内容完全由 `site/` 决定，那边手工加的文件会在下次部署时被删除
- 脚本把 `site/` 同步到 `.site-deploy/`（Pages 仓库的本地克隆，已 gitignore），写入 `.nojekyll`，提交信息带上本仓库的 commit
- 走 HTTPS 推送，使用本机 git 凭据；换仓库或子路径时设置 `PAGES_REPO`、`PAGES_BRANCH`、`SITE_URL`
- 站点约 330 MB，在 GitHub Pages 1 GB 的上限之内；每次部署只有变动的页面和图片会新增到 Pages 仓库的历史里

## 论文工具

论文目录约定见 [AGENTS.md](../AGENTS.md)：`docs/<领域>/<专题>/<Name>/{<Full Title>_<arxiv>.pdf, zh.md, figs/}`。所有工具的路径参数可以写成相对当前目录，也可以写成相对 `docs/`，在仓库任意位置运行均可。

| 脚本 | 用途 | 外部依赖 |
|------|------|----------|
| `papers/fetch_figures.py <dir>...` | 从 arXiv HTML / ar5iv 下载论文原图到 `figs/figNN.png`；arXiv 号从 PDF 文件名读取，没有 PDF 时用 `--arxiv` | 无 |
| `papers/pdf_figures.py <dir>... [--embed zh.md]` | 兜底：按 “Figure N:” 图注从 PDF 裁图；默认不覆盖已有的 HTML 原图 | PyMuPDF |
| `papers/crop_tables.py <dir>... [--tables 2,3]` | 按表格横线从 PDF 裁出表格原排版到 `tables/tableNN.png` | poppler、Pillow |
| `papers/shrink_pdf.py <pdf或目录>... [--dry-run]` | 首次提交前压缩 PDF 中的位图（默认 150 dpi，视觉论文用 `--quality printer`） | ghostscript、poppler |

```bash
python3 scripts/papers/fetch_figures.py embedding/E5-Mistral
uv run --with pymupdf python3 scripts/papers/pdf_figures.py embedding/RocketQA/RocketQA
uv run --with pillow python3 scripts/papers/crop_tables.py embedding/Margin-MSE --tables 2,3
python3 scripts/papers/shrink_pdf.py embedding/DINO/DINOv3 --quality printer --dry-run
```

`papers/_paths.py` 是这几个脚本共用的路径解析与 arXiv 号识别。
