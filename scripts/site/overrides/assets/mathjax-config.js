// pymdownx.arithmatex(generic) 输出 \( \) / \[ \]，只处理 .arithmatex 节点，避免误伤正文里的 $ 与反斜杠
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "ams",
    macros: {
      R: "\\mathbb{R}",
      E: "\\mathbb{E}",
      // 论文里常见的 bm 包写法，MathJax 默认不提供；等价于 boldsymbol
      bm: ["\\boldsymbol{#1}", 1],
    },
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
  chtml: { scale: 1.0, matchFontHeight: false },
};

// Material 的 instant navigation 切页时需要重新排版公式
if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    if (window.MathJax && MathJax.startup && MathJax.typesetPromise) {
      MathJax.startup.output.clearCache && MathJax.startup.output.clearCache();
      MathJax.typesetClear && MathJax.typesetClear();
      MathJax.texReset && MathJax.texReset();
      MathJax.typesetPromise();
    }
  });
}
