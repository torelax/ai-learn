# papers

布局：**一篇论文 / 一个方法 / 一个系列 → 一个文件夹**，文件夹用方法名或文章名，**不要带 arXiv 编号**。

```
docs/papers/<topic>/<Name>/{<Full Title>_<arxiv>.pdf, zh.md, figs/}
docs/papers/<topic>/<Series>/<Paper>/{<Full Title>_<arxiv>.pdf, zh.md, figs/}
```

- PDF：论文英文全名，有 arXiv 则后缀 `_YYMM.NNNNN.pdf`。**不要**命名为 `paper.pdf`。
- `zh.md`：全文中译（若有）
- `figs/`：图片集（优先 arXiv HTML / ar5iv 原图，否则 PDF 裁切）

系列示例：`embedding/GTE/{GTE,mGTE}/`、`embedding/BGE/{C-Pack,M3,EN-ICL}/`。

抓图脚本仍放在各 topic 根目录：`embedding/_fetch_html_figures.py`。
