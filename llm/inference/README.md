# LLM 推理与 KV cache

| 入口 | 内容 |
| --- | --- |
| [KV cache Notebook](kv-cache.ipynb) | 用小张量手写 KV 读写、RoPE 偏移、因果 mask、生成循环和 EOS，再对照 MiniMind 源码 |
| [CacheBlend](CacheBlend.md) | 结合图和伪代码，理解前文变化后怎样近似复用后缀块 |
| [Llama 模型参考](llama-reference/llama.py)与 [生成循环](llama-reference/llama_generate.py) | 对照原 Llama 2 实现的预分配 `cache_k/cache_v` 和 `start_pos`。这里保留的是源码参考，独立运行需要上游包、权重及相应环境 |

头部共享、可见范围和缓存表示的区别见 [注意力机制](../attention/README.md)。需要补 RoPE 时可看 [RoPE Notebook](../../transformers/basics/RoPE.ipynb)。
