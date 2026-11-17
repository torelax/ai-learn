# 构建 HTML 阅读版

Markdown 是正文源。HTML、KaTeX字体和Mermaid脚本都写到 `../html/`，由本专题 `.gitignore` 排除。

先将固定版本的渲染依赖安装到一个临时或自选目录（不会修改仓库根依赖）：

```bash
npm install --prefix /tmp/agent-memory-render --ignore-scripts --no-audit --no-fund \
  markdown-it@14.1.0 katex@0.16.22 mermaid@10.9.3
node agent/memory/scripts/build-html.cjs /tmp/agent-memory-render/node_modules
```

脚本接受 node_modules 绝对路径；如果另选目录，请换成你的安装路径。构建后打开 `agent/memory/html/index.html`。字体、公式和绘图资源均为本地文件；在线引用仍需网络。构建器将相对MD链接改为对应HTML链接，保留本地图片来源，派生图片拷贝到html内。

数学公式在构建时用KaTeX生成，遇到无效公式直接失败；Mermaid在页面加载时渲染，浏览器验证还须检查绘图是否全部成功。这两个步骤不同，Node退出码为0不代表浏览器图一定正确。

`examples/` 的说明也会生成HTML。代码与JSON示例不作为公式解析，不执行代码块，不运行真实模型。渲染依赖属于派生工具链，应用所需版本已在本页固定；本脚本不自动联网安装或升级包。
