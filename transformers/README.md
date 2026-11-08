# Transformer 学习笔记与代码

| 目录 | 内容 |
| --- | --- |
| [basics](basics/) | 自注意力与 Transformer 结构、最小语言模型示例、RoPE |
| [encoder-decoder](encoder-decoder/) | 编解码 Transformer 的独立示例，包含 `main.py`、配置和模型实现 |
| [bert](bert/) | BERT 参数量计算、文本分类和 NER |
| [huggingface](huggingface/) | Hugging Face 模型加载与数据集示例 |
| [LLM 推理](../llm/inference/README.md) | KV cache 代码导读和 CacheBlend 详解 |

学习推理时，先读 [RoPE 实现](basics/RoPE.ipynb)，再跑 KV cache Notebook，最后看 CacheBlend 如何在跨请求复用时处理变化的前文。

检索练习见 [Embedding Notebook](../embedding/embeddings.ipynb)和 [Rerank Notebook](../rerank/rerank.ipynb)。
