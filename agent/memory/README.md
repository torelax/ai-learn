# Agent Memory 专题报告

> **结论先行：**“存过”不代表“本轮使用了”。需要分别核查载体、写入策略、索引、读取触发者和注入位置。Pi 核心默认没有统一的跨会话 BM25/Embedding 记忆服务；Codex、Hermes、Claude Code 的原生路径也不相同。研究快照：2026-11-18。

## 阅读路线

| 章节 | 解决的问题 |
|---|---|
| [通用记忆架构与检索决策](01-architecture.md) | 内容分类、数据schema、写入/更新/遗忘、多路检索、谁触发读取、如何进入消息 |
| [四类原生 Agent](02-native-agents.md) | Claude Code、Codex、Hermes、Pi 分别保存什么文件或DB，何时写、怎样用 |
| [记忆插件、火山与腾讯项目](03-plugins-and-cn-systems.md) | claude-mem、OpenViking、MineContext、TencentDB Agent Memory、Hy-Memory 的具体管线 |
| [独立记忆系统](04-memory-systems.md) | Mem0 V3、当前 Letta/MemGPT、Graphiti、Hindsight、MemOS、LangMem |
| [Benchmark 与测试方案](05-benchmarks.md) | LoCoMo、LongMemEval/V2、MemoryAgentBench 等；公平对照、成本与故障归因 |
| [博客与前沿方向](06-reading-and-directions.md) | 上下文工程、长期任务交接、后台整理、程序性记忆与研究问题 |
| [可执行的词法记忆实验](examples/README.md) | 只用 Python 标准库，观察 SQLite/FTS5、时间更新、作用域与逻辑撤销 |

HTML 为这些 Markdown 的派生阅读版，输出在本目录 `html/`；由 `.gitignore` 排除，不入库。构建方法见 [渲染脚本](scripts/README.md)，输出首页为 `html/index.html`。公式用本地 KaTeX 渲染，Mermaid 转图，不依赖阅读时访问CDN。图的caption与正文标清官方图、源码机制图及建议架构。

## 先精确回答 Pi 的记忆问题

以下针对本地 coding-agent CLI 快照 `5f89cc0a7df669813650bfdac498f8ff865e8233`，不把 Durable 或外部插件的能力当默认。完整字段、路径和代码证据在[Pi 深读](02-native-agents.md#pi默认有持久会话与文件化知识没有统一的自动长期召回服务)。

| 载体 | 文件/结构 | 什么时候写 | 怎么取出和使用 | BM25/Embedding？ |
|---|---|---|---|---|
| 运行状态 | 进程内 Agent messages、队列等 | 本次运行状态变化 | 下一请求前以会话投影刷新 | 都不是 |
| 原始会话 | `~/.pi/agent/sessions/` 分目录JSONL；entry含id/parentId/type/message | 首次user/assistant使setup一起刷盘，后续完成条目追加 | 继续/选择会话，沿活动分支重放 | 不做默认跨会话语义召回 |
| 会话压缩 | 同一JSONL中的compaction，含summary/firstKeptEntryId/systemMessage | 手动或阈值/溢出恢复成功后 | 系统快照 + 摘要user + 保留尾部进入请求 | 选择边界+LLM摘要，不是检索索引 |
| 项目规则 | `AGENTS.override.md`、`AGENTS.md`、`CLAUDE.md` 等 | 人或工具实际修改文件 | 资源发现/加载后进入系统提示分区 | 路径发现、直接加载 |
| 技能 | `SKILL.md` 与references/scripts | 人或工具写入 | 名称描述先加载，主模型按需读全文 | 默认按描述选择，不是向量库 |
| 扩展状态 | JSONL `custom` 与 `custom_message`，插件可另建DB | 扩展显式append | custom需扩展读取并注入；custom_message可转换成模型消息 | 核心不为所有扩展统一实现 |

因此，问“Pi 有没有局外 LLM 每次判断是否查记忆”，对核心默认链路应答**没有这样的统一服务**。它有独立的摘要LLM调用，但摘要是写/维护会话上下文，不是跨会话检索路由器。你可以扩展一个 memory_search 工具让前台Agent决定调用，或做后台抽取/hook自动检索；这属于额外系统设计，需要定义来源、权限、索引和注入路径。

## 为什么不能用一个 Hybrid 标签比较所有系统

- Claude Code 本地 auto memory：Markdown 索引与按需topic读取。不要推断闭源内部一定有ANN。
- Codex 当前开源 v2：后台阶段提取/整合，摘要注入与文件工具；已核查的本地 search 是子串匹配，不是看见SQLite就称BM25。
- Hermes：小量冻结文件常驻，加FTS5/trigram历史检索、前台记忆工具、后台review；可选外部provider另论。
- claude-mem：插件hook、worker与存储/检索管线；不同版本/本地与云模式分别分析。
- OpenViking：层级文件表示不等于仍使用旧版目录递归搜索；当前源码及 find/search 路由见专章。
- Mem0：OSS V3 与 Platform V3分别解释；当前候选池、阈值与融合需要看真实代码，不照搬旧 ADD/UPDATE/DELETE 图。
- Graphiti、Hindsight、MemOS、LangMem：都是可嵌入能力；谁调用add/search/recall、什么时候把结果送主模型，仍需由宿主或工具协议决定。

## 报告的证据边界

项目章节尽可能固定源码commit；闭源产品使用官方公开文档，未将未公开数据库或模型角色写成事实。目录、字段与请求示例是教学构造，不来自你的私人聊天或真实记忆。版本快照、默认配置、可选能力、托管后端与开源实现分别标注。

“热门”不等于质量排名：正文只对已读取的GitHub star快照做带日期说明，没有按星数推出正确率或安全性。本文没有运行付费模型复现厂商排行榜；执行过的确定性实验与渲染检查，和资料中的作者自报成绩明确分开。

这里提供可学习的设计分析和复现实验方案，不是对全部开源项目的百科穷举。核心比较单位始终是：**存什么 → 谁写 → 何时写 → 存哪 → 谁查 → 怎么查 → 怎么送模 → 怎样纠正和删除。**
