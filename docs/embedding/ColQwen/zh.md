# ColQwen 官方模型卡：中英逐段对照

> 来源为下列官方模型卡，采集于 2026-10-22；这不是一篇独立论文。英文段落后附中文机器初译，代码和数字表共用，未逐句人工审定。ColQwen2 卡片标注 Apache-2.0，ColQwen2.5 卡片标注 MIT，Tomoro 卡片标注 Apache-2.0；底座和 adapter 的具体使用许可分别以对应源文件为准。版本冲突与动态评测口径见同目录模型报告。

## ColQwen2

来源：https://huggingface.co/vidore/colqwen2-v1.0

```text
---
license: apache-2.0
library_name: colpali
base_model: vidore/colqwen2-base
language:
- en
tags:
- colpali
- vidore-experimental
- vidore
- sentence-transformers
- multi-vector
pipeline_tag: visual-document-retrieval
---
# ColQwen2: Visual Retriever based on Qwen2-VL-2B-Instruct with ColBERT strategy
```

**EN.** ### This is the base version trained with batch_size 256 instead of 32 for 5 epoch and with the updated pad token

**中.** ### 这是使用batch_size 256（而不是32）训练5个epoch并使用更新的pad令牌训练的基本版本

**EN.** ColQwen2 is a model based on a novel model architecture and training strategy based on Vision Language Models (VLMs) to efficiently index documents from their visual features.
It is a [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) extension that generates [ColBERT](https://arxiv.org/abs/2004.12832)- style multi-vector representations of text and images.
It was introduced in the paper [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449) and first released in [this repository](https://github.com/ManuelFay/colpali)

**中.** ColQwen2 是一个基于新颖模型架构和基于视觉语言模型 (VLM) 的训练策略的模型，可根据文档的视觉特征高效地索引文档。
它是 [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) 扩展，可生成 [ColBERT](https://arxiv.org/abs/2004.12832) 样式的文本和图像多向量表示。
它在论文[ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449)中介绍，并首次发布在[本存储库](https://github.com/ManuelFay/colpali)

```text
<p align="center"><img width=800 src="https://github.com/illuin-tech/colpali/blob/main/assets/colpali_architecture.webp?raw=true"/></p>
```

**EN.** ## Version specificity

**中.** ## 版本特异性

**EN.** This model takes dynamic image resolutions in input and does not resize them, changing their aspect ratio as in ColPali.
Maximal resolution is set so that 768 image patches are created at most. Experiments show clear improvements with larger amounts of image patches, at the cost of memory requirements.

**中.** 该模型在输入中采用动态图像分辨率，并且不会调整它们的大小，而是像 ColPali 中那样改变它们的纵横比。
设置最大分辨率，以便最多创建 768 个图像块。实验表明，大量图像补丁可以带来明显的改进，但会降低内存需求。

**EN.** This version is trained with `colpali-engine==0.3.1`.

**中.** 该版本使用 `colpali-engine==0.3.1` 进行训练。

**EN.** Data is the same as the ColPali data described in the paper.

**中.** 数据与论文中描述的ColPali数据相同。

**EN.** ## Model Training

**中.** ## 模型训练

**EN.** ### Dataset
Our training dataset of 127,460 query-page pairs is comprised of train sets of openly available academic datasets (63%) and a synthetic dataset made up of pages from web-crawled PDF documents and augmented with VLM-generated (Claude-3 Sonnet) pseudo-questions (37%).
Our training set is fully English by design, enabling us to study zero-shot generalization to non-English languages. We explicitly verify no multi-page PDF document is used both [*ViDoRe*](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d) and in the train set to prevent evaluation contamination.
A validation set is created with 2% of the samples to tune hyperparameters.

**中.** ### 数据集
我们的训练数据集包含 127,460 个查询页面对，其中包括公开学术数据集的训练集 (63%) 和合成数据集，该数据集由网络爬取的 PDF 文档页面组成，并用 VLM 生成的 (Claude-3 Sonnet) 伪问题进行了增强 (37%)。
我们的训练集完全是英语设计的，使我们能够研究对非英语语言的零样本泛化。我们明确核对，没有同一份多页 PDF 文档同时出现在 [*ViDoRe*](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d) 和训练集中，以防止评估污染。
使用 2% 的样本创建验证集来调整超参数。

**EN.** *Note: Multilingual data is present in the pretraining corpus of the language model and most probably in the multimodal training.*

**中.** *注意：多语言数据存在于语言模型的预训练语料库中，并且很可能存在于多模态训练中。*

**EN.** ### Parameters

**中.** ### 参数

**EN.** All models are trained for 1 epoch on the train set. Unless specified otherwise, we train models in `bfloat16` format, use low-rank adapters ([LoRA](https://arxiv.org/abs/2106.09685))
with `alpha=32`  and `r=32` on the transformer layers from the language model,
as well as the final randomly initialized projection layer, and use a `paged_adamw_8bit` optimizer.
We train on an 8 GPU setup with data parallelism, a learning rate of 5e-5 with linear decay with 2.5% warmup steps, and a batch size of 32.

**中.** 所有模型都在训练集上训练 1 epoch。除非另有说明，我们以 `bfloat16` 格式训练模型，使用低秩适配器（[LoRA](https://arxiv.org/abs/2106.09685)）
语言模型的 Transformer 层上有 `alpha=32` 和 `r=32`，
以及最终随机初始化的投影层，并使用 `paged_adamw_8bit` 优化器。
我们在具有数据并行性的 8 GPU 设置上进行训练，学习率为 5e-5，线性衰减，预热步骤为 2.5%，批处理大小为 32。

**EN.** ## Usage

**中.** ## 用法

**EN.** ### Sentence Transformers

**中.** ### 句子变形金刚

**EN.** ColQwen2 can be used as a multi-vector (ColBERT-style late interaction) retriever with Sentence Transformers via the `MultiVectorEncoder`:

**中.** ColQwen2 可以通过 `MultiVectorEncoder` 用作带有句子转换器的多向量（ColBERT 式后期交互）检索器：

```bash
pip install "sentence-transformers[image]>=6.0.0"
```

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("vidore/colqwen2-v1.0")

queries = [
    "What is the variable represented on the y-axis of the graph?",
    "Total outlay is maximum in which year?",
]
images = [
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc1.jpg",
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc2.jpg",
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc3.jpg",
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc4.jpg",
]

query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(images)
print(f"Query 0 shape:    {tuple(query_embeddings[0].shape)}")
print(f"Document 0 shape: {tuple(document_embeddings[0].shape)}")
# Query 0 shape:    (25, 128)
# Document 0 shape: (755, 128)

scores = model.similarity(query_embeddings, document_embeddings)
print(scores)
# tensor([[13.7060, 11.1911, 11.5526, 10.3418],
#         [ 7.2301, 16.2052,  6.9269,  6.3475]])
```

**EN.** To score only the image-patch tokens (which roughly halves the stored document index), enable the mask allowlist after loading:

**中.** 要仅对图像补丁标记进行评分（大约将存储的文档索引减半），请在加载后启用掩码白名单：

```python
model[-1].keep_only_token_ids = [151655]  # the <|image_pad|> token id
```

**EN.** ### ColPali Engine

**中.** ### ColPali 引擎

**EN.** > [!WARNING]
> Note: `colpali-engine` 0.3.13 and later no longer send the `"Query: "` query prefix that this checkpoint
> was trained with (illuin-tech/colpali#280 dropped it from `ColQwen2Processor`, and illuin-tech/colpali#339
> then changed the base class default it fell back on to `""`). The Sentence Transformers configuration in
> this repository reproduces the original training-time format, so its embeddings differ slightly from
> current `colpali-engine` output.

**中.** > [!警告]
> 注意：`colpali-engine` 0.3.13及更高版本不再发送该检查点的`"Query: "`查询前缀
> 接受过训练（illuin-tech/colpali#280 从 `ColQwen2Processor` 中删除了它，并且 illuin-tech/colpali#339
> 然后更改了基类默认值，它又回到了 `""`）。 Sentence Transformers 配置在
> 这个存储库重现了原始的训练时间格式，因此它的嵌入与
> 当前 `colpali-engine` 输出。

**EN.** Make sure `colpali-engine` is installed from source or with a version superior to 0.3.4.
`transformers` version must be > 4.46.1.

**中.** 确保 `colpali-engine` 从源代码安装或使用高于 0.3.4 的版本安装。
`transformers` 版本必须 > 4.46.1。

```bash
pip install git+https://github.com/illuin-tech/colpali
```

```python
import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2, ColQwen2Processor

model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")

# Your inputs
images = [
    Image.new("RGB", (128, 128), color="white"),
    Image.new("RGB", (64, 32), color="black"),
]
queries = [
    "Is attention really all you need?",
    "What is the amount of bananas farmed in Salvador?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
```

**EN.** ## Limitations

**中.** ## 限制

**EN.** - **Focus**: The model primarily focuses on PDF-type documents and high-ressources languages, potentially limiting its generalization to other document types or less represented languages.
 - **Support**: The model relies on multi-vector retreiving derived from the ColBERT late interaction mechanism, which may require engineering efforts to adapt to widely used vector retrieval frameworks that lack native multi-vector support.

**中.** - **焦点**：该模型主要关注 PDF 类型文档和高分辨率语言，可能会限制其对其他文档类型或代表性较少的语言的推广。
 - **支持**：该模型依赖于源自 ColBERT 后期交互机制的多向量检索，这可能需要工程工作来适应缺乏原生多向量支持的广泛使用的向量检索框架。

**EN.** ## License

**中.** ## 许可证

**EN.** ColQwen2's vision language backbone model (Qwen2-VL) is under `apache2.0` license. The adapters attached to the model are under MIT license.

**中.** ColQwen2 的视觉语言骨干模型（Qwen2-VL）采用 `apache2.0` 许可。模型附带的适配器已获得 MIT 许可。

**EN.** ## Contact

**中.** ## 联系方式

**EN.** - Manuel Faysse: manuel.faysse@illuin.tech
- Hugues Sibille: hugues.sibille@illuin.tech
- Tony Wu: tony.wu@illuin.tech

**中.** - 曼努埃尔·费塞：manuel.faysse@illuin.tech
- 休斯·西比尔：hugues.sibille@illuin.tech
- 吴托尼：tony.wu@illuin.tech

**EN.** ## Citation

**中.** ## 引文

**EN.** If you use any datasets or models from this organization in your research, please cite the original dataset as follows:

**中.** 如果您在研究中使用该组织的任何数据集或模型，请按如下方式引用原始数据集：

```bibtex
@misc{faysse2024colpaliefficientdocumentretrieval,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and Céline Hudelot and Pierre Colombo},
  year={2024},
  eprint={2407.01449},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2407.01449},
}
```

## ColQwen2.5

来源：https://huggingface.co/vidore/colqwen2.5-v0.2

```text
---
license: mit
library_name: colpali
base_model: vidore/colqwen2.5-base
language:
- en
tags:
- colpali
- vidore
- vidore-experimental
- sentence-transformers
- multi-vector
pipeline_tag: visual-document-retrieval
---
```

**EN.** # ColQwen2.5: Visual Retriever based on Qwen2.5-VL-3B-Instruct with ColBERT strategy

**中.** # ColQwen2.5：基于 Qwen2.5-VL-3B-Instruct 和 ColBERT 策略的视觉检索器

**EN.** ColQwen is a model based on a novel model architecture and training strategy based on Vision Language Models (VLMs) to efficiently index documents from their visual features.
It is a [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) extension that generates [ColBERT](https://arxiv.org/abs/2004.12832)- style multi-vector representations of text and images.
It was introduced in the paper [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449) and first released in [this repository](https://github.com/ManuelFay/colpali)

**中.** ColQwen 是一个基于新颖模型架构和基于视觉语言模型 (VLM) 的训练策略的模型，可根据文档的视觉特征高效地索引文档。
它是 [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) 扩展，可生成 [ColBERT](https://arxiv.org/abs/2004.12832) 样式的文本和图像多向量表示。
它在论文[ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449)中介绍，并首次发布在[本存储库](https://github.com/ManuelFay/colpali)

```text
<p align="center"><img width=800 src="https://github.com/illuin-tech/colpali/blob/main/assets/colpali_architecture.webp?raw=true"/></p>
```

**EN.** ## Version specificity

**中.** ## 版本特异性

**EN.** This model takes dynamic image resolutions in input and does not resize them, changing their aspect ratio as in ColPali.
Maximal resolution is set so that 768 image patches are created at most. Experiments show clear improvements with larger amounts of image patches, at the cost of memory requirements.

**中.** 该模型在输入中采用动态图像分辨率，并且不会调整它们的大小，而是像 ColPali 中那样改变它们的纵横比。
设置最大分辨率，以便最多创建 768 个图像块。实验表明，大量图像补丁可以带来明显的改进，但会降低内存需求。

**EN.** This version is trained with `colpali-engine==0.3.7`.

**中.** 该版本使用 `colpali-engine==0.3.7` 进行训练。

**EN.** Data is the same as the ColPali data described in the paper.

**中.** 数据与论文中描述的ColPali数据相同。

**EN.** ## Model Training

**中.** ## 模型训练

**EN.** ### Dataset
Our training dataset of 127,460 query-page pairs is comprised of train sets of openly available academic datasets (63%) and a synthetic dataset made up of pages from web-crawled PDF documents and augmented with VLM-generated (Claude-3 Sonnet) pseudo-questions (37%).
Our training set is fully English by design, enabling us to study zero-shot generalization to non-English languages. We explicitly verify no multi-page PDF document is used both [*ViDoRe*](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d) and in the train set to prevent evaluation contamination.
A validation set is created with 2% of the samples to tune hyperparameters.

**中.** ### 数据集
我们的训练数据集包含 127,460 个查询页面对，其中包括公开学术数据集的训练集 (63%) 和合成数据集，该数据集由网络爬取的 PDF 文档页面组成，并用 VLM 生成的 (Claude-3 Sonnet) 伪问题进行了增强 (37%)。
我们的训练集完全是英语设计的，使我们能够研究对非英语语言的零样本泛化。我们明确核对，没有同一份多页 PDF 文档同时出现在 [*ViDoRe*](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d) 和训练集中，以防止评估污染。
使用 2% 的样本创建验证集来调整超参数。

**EN.** *Note: Multilingual data is present in the pretraining corpus of the language model and most probably in the multimodal training.*

**中.** *注意：多语言数据存在于语言模型的预训练语料库中，并且很可能存在于多模态训练中。*

**EN.** ### Parameters

**中.** ### 参数

**EN.** All models are trained for 1 epoch on the train set. Unless specified otherwise, we train models in `bfloat16` format, use low-rank adapters ([LoRA](https://arxiv.org/abs/2106.09685))
with `alpha=32`  and `r=32` on the transformer layers from the language model,
as well as the final randomly initialized projection layer, and use a `paged_adamw_8bit` optimizer.
We train on an 8 GPU setup with data parallelism, a learning rate of 5e-5 with linear decay with 2.5% warmup steps, and a batch size of 32.

**中.** 所有模型都在训练集上训练 1 epoch。除非另有说明，我们以 `bfloat16` 格式训练模型，使用低秩适配器（[LoRA](https://arxiv.org/abs/2106.09685)）
语言模型的 Transformer 层上有 `alpha=32` 和 `r=32`，
以及最终随机初始化的投影层，并使用 `paged_adamw_8bit` 优化器。
我们在具有数据并行性的 8 GPU 设置上进行训练，学习率为 5e-5，线性衰减，预热步骤为 2.5%，批处理大小为 32。

**EN.** ## Usage

**中.** ## 用法

**EN.** ### Sentence Transformers

**中.** ### 句子变形金刚

**EN.** ColQwen2.5 can be used as a multi-vector (ColBERT-style late interaction) retriever with Sentence Transformers via the `MultiVectorEncoder`:

**中.** ColQwen2.5 可以通过 `MultiVectorEncoder` 用作带有句子转换器的多向量（ColBERT 式后期交互）检索器：

```bash
pip install "sentence-transformers[image]>=6.0.0"
```

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("vidore/colqwen2.5-v0.2")

queries = [
    "What is the variable represented on the y-axis of the graph?",
    "Total outlay is maximum in which year?",
]
images = [
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc1.jpg",
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc2.jpg",
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc3.jpg",
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc4.jpg",
]

query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(images)
print(f"Query 0 shape:    {tuple(query_embeddings[0].shape)}")
print(f"Document 0 shape: {tuple(document_embeddings[0].shape)}")
# Query 0 shape:    (25, 128)
# Document 0 shape: (755, 128)

scores = model.similarity(query_embeddings, document_embeddings)
print(scores)
# tensor([[13.8672, 12.3115, 12.1670, 11.0293],
#         [ 7.2012, 14.7207,  6.9414,  6.9746]])
```

**EN.** To score only the image-patch tokens (which drops the 11 non-image tokens per page), enable the mask allowlist after loading:

**中.** 要仅对图像补丁标记进行评分（每页删除 11 个非图像标记），请在加载后启用掩码白名单：

```python
model[-1].keep_only_token_ids = [151655]  # the <|image_pad|> token id
```

**EN.** ### ColPali Engine

**中.** ### ColPali 引擎

**EN.** > [!WARNING]
> Note: `colpali-engine` 0.3.13 and later no longer send the `"Query: "` query prefix that this checkpoint
> was trained with (illuin-tech/colpali#280 dropped it from `ColQwen2_5_Processor`, and illuin-tech/colpali#339
> then changed the base class default it fell back on to `""`). The Sentence Transformers configuration in
> this repository reproduces the original training-time format, so its embeddings differ slightly from
> current `colpali-engine` output.

**中.** > [!警告]
> 注意：`colpali-engine` 0.3.13及更高版本不再发送该检查点的`"Query: "`查询前缀
> 接受过训练（illuin-tech/colpali#280 从 `ColQwen2_5_Processor` 中删除了它，并且 illuin-tech/colpali#339
> 然后更改了基类默认值，它又回到了 `""`）。 Sentence Transformers 配置在
> 这个存储库重现了原始的训练时间格式，因此它的嵌入与
> 当前 `colpali-engine` 输出。

**EN.** Make sure `colpali-engine` is installed from source or with a version superior to 0.3.1.
`transformers` version must be > 4.45.0.

**中.** 确保 `colpali-engine` 从源代码安装或使用高于 0.3.1 的版本安装。
`transformers` 版本必须 > 4.45.0。

```bash
pip install git+https://github.com/illuin-tech/colpali
```

```python
import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

model = ColQwen2_5.from_pretrained(
        "vidore/colqwen2.5-v0.2",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # or "mps" if on Apple Silicon
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()
processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")

# Your inputs
images = [
    Image.new("RGB", (32, 32), color="white"),
    Image.new("RGB", (16, 16), color="black"),
]
queries = [
    "Is attention really all you need?",
    "What is the amount of bananas farmed in Salvador?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
```

**EN.** ## Limitations

**中.** ## 限制

**EN.** - **Focus**: The model primarily focuses on PDF-type documents and high-ressources languages, potentially limiting its generalization to other document types or less represented languages.
 - **Support**: The model relies on multi-vector retreiving derived from the ColBERT late interaction mechanism, which may require engineering efforts to adapt to widely used vector retrieval frameworks that lack native multi-vector support.

**中.** - **焦点**：该模型主要关注 PDF 类型文档和高分辨率语言，可能会限制其对其他文档类型或代表性较少的语言的推广。
 - **支持**：该模型依赖于源自 ColBERT 后期交互机制的多向量检索，这可能需要工程工作来适应缺乏原生多向量支持的广泛使用的向量检索框架。

**EN.** ## License

**中.** ## 许可证

**EN.** ColQwen2.5's vision language backbone model (Qwen2.5-VL) is under `Qwen RESEARCH LICENSE AGREEMENT` license. The adapters attached to the model are under MIT license.

**中.** ColQwen2.5的视觉语言骨干模型（Qwen2.5-VL）采用`Qwen RESEARCH LICENSE AGREEMENT`许可。模型附带的适配器已获得 MIT 许可。

**EN.** ## Contact

**中.** ## 联系方式

**EN.** - Manuel Faysse: manuel.faysse@illuin.tech
- Hugues Sibille: hugues.sibille@illuin.tech
- Tony Wu: tony.wu@illuin.tech

**中.** - 曼努埃尔·费塞：manuel.faysse@illuin.tech
- 休斯·西比尔：hugues.sibille@illuin.tech
- 吴托尼：tony.wu@illuin.tech

**EN.** ## Citation

**中.** ## 引文

**EN.** If you use any datasets or models from this organization in your research, please cite the original dataset as follows:

**中.** 如果您在研究中使用该组织的任何数据集或模型，请按如下方式引用原始数据集：

```bibtex
@misc{faysse2024colpaliefficientdocumentretrieval,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and Céline Hudelot and Pierre Colombo},
  year={2024},
  eprint={2407.01449},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2407.01449},
}
```

## Tomoro-ColQwen3

来源：https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b

```text
---
license: apache-2.0
license_name: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
tags:
- sentence-transformers
- multi-vector
- text
- image
- video
- multimodal-embedding
- vidore
- colpali
- colqwen3
- multilingual-embedding
language:
- multilingual
library_name: transformers
pipeline_tag: visual-document-retrieval
base_model:
- Qwen/Qwen3-VL-4B-Instruct
---
```

**EN.** # TomoroAI/tomoro-colqwen3-embed-4b

**中.** # TomoroAI/tomoro-colqwen3-embed-4b

**EN.** ## ⚡ Executive Summary

**中.** ## ⚡ 执行摘要

**EN.** **TomoroAI/tomoro-colqwen3-embed-4b** is a state-of-the-art [ColPali](https://arxiv.org/abs/2407.01449)-style multimodal embedding model. It maps text queries, visual documents (images, PDFs) or short videos into aligned multi-vector embeddings.

**中.** **TomoroAI/tomoro-colqwen3-embed-4b** 是最先进的 [ColPali](https://arxiv.org/abs/2407.01449) 式多模态嵌入模型。它将文本查询、视觉文档（图像、PDF）或短视频映射到对齐的多向量嵌入中。

**EN.** Built by merging **[Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)** with **[Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)**, this model inherits robust text retrieval capabilities while preserving a full vision stack. It has been fine-tuned on a curated mixture of [VDR](https://huggingface.co/datasets/vdr-multilingual-train), [ViDoRe-ColPali-Training](https://huggingface.co/datasets/vidore/colpali_train_set), [VisRAG-Ret-Train-Synthetic-data](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-Synthetic-data), and [VisRAG-Ret-Train-In-domain-data](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-In-domain-data). It achieves SOTA or competitive performance across **ViDoRe V1-V3** (English and Multilingual) while offering a significantly reduced embedding footprint compared to other full-dim Colpali model alternatives.

**中.** 该模型通过将 **[Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)** 与 **[Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)** 合并而构建，继承了强大的文本检索功能，同时保留了完整的视觉堆栈。它已在 [VDR](https://huggingface.co/datasets/vdr-multilingual-train)、[ViDoRe-ColPali-Training](https://huggingface.co/datasets/vidore/colpali_train_set)、[VisRAG-Ret-Train-Synthetic-data](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-Synthetic-data) 和[VisRAG-Ret-Train-域内数据](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-In-domain-data)。它在 **ViDoRe V1-V3**（英语和多语言）中实现了 SOTA 或具有竞争力的性能，同时与其他全暗 Colpali 模型替代品相比，显着减少了嵌入足迹。

**EN.** ## 🛠️ Model Specifications

**中.** ## 🛠️ 型号规格

| Feature<br>特点 | Detail<br>详情 |
| :--- | :--- |
| **Architecture**<br>**架构** | Qwen3-VL 4B (Encoder-only variant) + 320-dim Projection Head<br>Qwen3-VL 4B（仅编码器版本）+ 320 维投影头 |
| **Methodology**<br>**方法论** | ColPali-style Late Interaction (MaxSim scoring)<br>ColPali 风格的后期交互（MaxSim 评分） |
| **Token Budget**<br>**代币预算** | Up to 1,280 visual tokens per page or 5120 visual tokens per video (text prompts constrained only by the base context window)<br>每页最多 1,280 个视觉标记或每个视频最多 5120 个视觉标记（文本提示仅受基本上下文窗口限制） |
| **Context Window**<br>**上下文窗口** | 32k (inherited from base), typical usage < 2k tokens<br>32k（从基础继承），典型用法 < 2k 代币 |
| **Output**<br>**输出** | Multi-vector (Seq_Len × 320), L2-normalized<br>多向量 (Seq_Len × 320)，L2 归一化 |
| **Supported Modalities**<br>**支持的方式** | Text Queries, RGB Images, Synthetic Documents, Short Video (Frame-wise)<br>文本查询、RGB 图像、合成文档、短视频（逐帧） |
| **Precision**<br>**精度** | `bfloat16` weights, FlashAttention 2 enabled<br>`bfloat16` 权重，启用 FlashAttention 2 |

**EN.** ### Key Properties

**中.** ### 关键属性

**EN.** * **Merged Encoders:** Combines the Qwen3-VL vision encoder (patch-grid tokens with spatial merge) and language encoder.
* **Projection:** A custom 320-dim head projects every token (text or visual) into a vector.
* **Processing:**
    * **Queries:** Left-padded text sequences.
    * **Documents:** Rendered with a lightweight vision prompt and flattened into image tokens.
    * **Video:** Supports video retrieval by decoding videos into frames and processing via the vision stack (generalization capability, not explicitly fine-tuned; dedicated benchmark coming soon).
* **Storage Efficiency:**
    * *Baseline (NVIDIA Nemo-3B):* Stores 1,802 tokens @ 3,072 dims (≈10.3 TB for 1M images).
    * *Tomoro ColQwen3:* Stores max 1,280 tokens @ 320 dims (**≈0.82 TB for 1M images**).
    * **Result:** **13× smaller footprint** with higher performance.

**中.** * **合并编码器：** 结合了 Qwen3-VL 视觉编码器（具有空间合并的补丁网格标记）和语言编码器。
* **投影：** 定制的 320 维头将每个标记（文本或视觉）投影到矢量中。
* **处理：**
    * **查询：** 左侧填充的文本序列。
    * **文档：** 使用轻量级视觉提示渲染并扁平化为图像令牌。
    * **视频：** 通过将视频解码为帧并通过视觉堆栈进行处理来支持视频检索（泛化能力，未明确微调；专用基准即将推出）。
* **存储效率：**
    * *基线 (NVIDIA Nemo-3B)：* 存储 1,802 个令牌@ 3,072 个暗淡（100 万张图像约为 10.3 TB）。
    * *Tomoro ColQwen3：* 最多可存储 1,280 个令牌 @ 320 暗度（**≈0.82 TB，1M 图像**）。
    * **结果：** **占用空间小 13 倍**，性能更高。

**EN.** ---

**中.** ---

**EN.** ## 📊 Evaluation Results

**中.** ## 📊 评估结果

**EN.** We report results on the **ViDoRe** benchmark suite. The model sets new standards on multilingual and English splits on ViDoRe V2 and V3 while maintaining comparable high performance on ViDoRe V1.

**中.** 我们报告 **ViDoRe** 基准套件的结果。该模型在 ViDoRe V2 和 V3 上为多语言和英语分割设定了新标准，同时在 ViDoRe V1 上保持了相当的高性能。

**EN.** ### ViDoRe V3 (Latest)

**中.** ### ViDoRe V3（最新）

**EN.** **English nDCG@5**
| Model | CompSci | Energy | FinanceEn | FinanceFr | HR | Industrial | Pharma | Physics | **Avg** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[tomoro-colqwen3-8b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b)** | 0.7443 | **0.6491** | **0.6823** | **0.4546** | **0.6421** | 0.5766 | **0.6665** | **0.4747** | **0.6113** |
| [tomoro-colqwen3-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) | 0.7419 | 0.6023 | 0.6753 | 0.4202 | 0.6037 | **0.5787** | 0.6612 | 0.4640 | 0.5934 |
| [nemo-colembed-3b](https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v1) | 0.7514 | 0.5838 | 0.6712 | 0.3730 | 0.6256 | 0.5447 | 0.6524 | 0.4128 | 0.5769 |
| [jinaai/jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) | 0.7175 | 0.5842 | 0.6417 | 0.3859 | 0.6206 | 0.5443 | 0.6303 | 0.4191 | 0.5680 |
| [nomic-ai/colnomic-embed-multimodal-7b](https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b) | **0.7528** | 0.5824 | 0.6041 | 0.3877 | 0.6060 | 0.5229 | 0.6226 | 0.4423 | 0.5651 |

**中.** **英语 nDCG@5**
|型号|计算机科学 |能源|财经En |财经 |人力资源 |工业|制药 |物理学| **平均** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[tomoro-colqwen3-8b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b)** | 0.7443 | 0.7443 **0.6491** | **0.6823** | **0.4546** | **0.6421** | 0.5766 | **0.6665** | **0.4747** | **0.6113** |
| [tomoro-colqwen3-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) | 0.7419 | 0.7419 0.6023 | 0.6023 0.6753 | 0.6753 0.4202 | 0.6037 | 0.6037 **0.5787** | 0.6612 | 0.4640 | 0.5934 | 0.5934
| [nemo-colembed-3b](https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v1) | 0.7514 | 0.7514 0.5838 | 0.6712 | 0.3730 | 0.6256 | 0.6256 0.5447 | 0.5447 0.6524 | 0.6524 0.4128 | 0.4128 0.5769 | 0.5769
| [jinaai/jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) | 0.7175 | 0.7175 0.5842 | 0.6417 | 0.6417 0.3859 | 0.3859 0.6206 | 0.6206 0.5443 | 0.5443 0.6303 | 0.6303 0.4191 | 0.4191 0.5680 |
| [nomic-ai/colnomic-embed-multimodal-7b](https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b) | **0.7528** | 0.5824 | 0.6041 | 0.3877 | 0.3877 0.6060 | 0.5229 | 0.5229 0.6226 | 0.4423 | 0.4423 0.5651 |

**EN.** **Multilingual nDCG@5** (Excluding English Subsets)
| Model | CompSci | Energy | FinanceEn | FinanceFr | HR | Industrial | Pharma | Physics | **Avg** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[tomoro-colqwen3-8b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b)** | 0.7194 | **0.6619** | **0.6172** | **0.4570** | **0.6097** | **0.5164** | **0.6403** | **0.4706** | **0.5866** |
| [tomoro-colqwen3-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) | 0.7213 | 0.6374 | 0.6019 | 0.4305 | 0.5637 | 0.5131 | 0.6351 | 0.4636 | 0.5708 |
| [nemo-colembed-3b](https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v1) | 0.7216 | 0.5901 | 0.5646 | 0.4102 | 0.5504 | 0.4335 | 0.6170 | 0.4192 | 0.5383 |
| [jinaai/jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) | 0.6843 | 0.6036 | 0.5482 | 0.4249 | 0.5542 | 0.4732 | 0.6059 | 0.4381 | 0.5416 |
| [nomic-ai/colnomic-embed-multimodal-7b](https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b) | **0.7333** | 0.6160 | 0.5219 | 0.4169 | 0.5494 | 0.4764 | 0.5938 | 0.4449 | 0.5441 |

**中.** **多语言 nDCG@5** （不包括英语子集）
|型号|计算机科学 |能源|财经En |财经 |人力资源 |工业|制药 |物理学| **平均** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[tomoro-colqwen3-8b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b)** | 0.7194 | 0.7194 **0.6619** | **0.6172** | **0.4570** | **0.6097** | **0.5164** | **0.6403** | **0.4706** | **0.5866** |
| [tomoro-colqwen3-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) | 0.7213 | 0.7213 0.6374 | 0.6019 | 0.4305 | 0.4305 0.5637 | 0.5637 0.5131 | 0.5131 0.6351 | 0.4636 | 0.4636 0.5708 |
| [nemo-colembed-3b](https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v1) | 0.7216 | 0.7216 0.5901 | 0.5646 | 0.4102 | 0.5504 | 0.5504 0.4335 | 0.4335 0.6170 | 0.4192 | 0.5383 |
| [jinaai/jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) | 0.6843 | 0.6036 | 0.6036 0.5482 | 0.4249 | 0.4249 0.5542 | 0.4732 | 0.6059 | 0.6059 0.4381 | 0.5416 |
| [nomic-ai/colnomic-embed-multimodal-7b](https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b) | **0.7333** | 0.6160 | 0.5219 | 0.5219 0.4169 | 0.4169 0.5494 | 0.5494 0.4764 | 0.4764 0.5938 | 0.4449 | 0.4449 0.5441 |

**EN.** ### ViDoRe V2

**中.** ### ViDoRe V2

**EN.** **English nDCG@5**
| Model | BioMed | ESG HL | ESG Rpts | Economics | **Avg** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[tomoro-colqwen3-8b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b)** | **0.6784** | **0.7598** | **0.6549** | 0.6159 | **0.6772** |
| [tomoro-colqwen3-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) | 0.6718 | 0.7465 | 0.6300 | 0.5910 | 0.6598 |
| [nemo-colembed-3b](https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v1) | 0.6518 | 0.7538 | 0.6030 | **0.6619** | 0.6676 |
| [jinaai/jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) | 0.6359 | 0.6512 | 0.5194 | 0.5955 | 0.6005 |
| [nomic-ai/colnomic-embed-multimodal-7b](https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b) | 0.6479 | 0.6871 | 0.5498 | 0.5955 | 0.6201 |

**中.** **英语 nDCG@5**
|型号|生物医学| ESG HL | ESG 报告 |经济学 | **平均** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[tomoro-colqwen3-8b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b)** | **0.6784** | **0.7598** | **0.6549** | 0.6159 | 0.6159 **0.6772** |
| [tomoro-colqwen3-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) | 0.6718 | 0.7465 | 0.7465 0.6300 | 0.5910 | 0.6598 | 0.6598
| [nemo-colembed-3b](https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v1) | 0.6518 | 0.7538 | 0.6030 | **0.6619** | 0.6676 |
| [jinaai/jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) | 0.6359 | 0.6359 0.6512 | 0.5194 | 0.5194 0.5955 | 0.5955 0.6005 | 0.6005
| [nomic-ai/colnomic-embed-multimodal-7b](https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b) | 0.6479 | 0.6479 0.6871 | 0.5498 | 0.5498 0.5955 | 0.5955 0.6201 |

**EN.** **Multilingual nDCG@5**
| Model | BioMed | ESG Rpts | Economics | **Avg** |
| :--- | :--- | :--- | :--- | :--- |
| **[tomoro-colqwen3-8b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b)** | 0.6467 | 0.5911 | **0.5875** | **0.6085** |
| [tomoro-colqwen3-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) | **0.6478** | **0.6226** | 0.5536 | 0.6080 |
| [nemo-colembed-3b](https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v1) | 0.6187 | 0.5640 | 0.5506 | 0.5778 |
| [jinaai/jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) | 0.5994 | 0.5178 | 0.5364 | 0.5512 |
| [nomic-ai/colnomic-embed-multimodal-7b](https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b) | 0.6224 | 0.5336 | 0.5433 | 0.5664 |

**中.** **多语言 nDCG@5**
|型号|生物医学| ESG 报告 |经济学 | **平均** |
| :--- | :--- | :--- | :--- | :--- |
| **[tomoro-colqwen3-8b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b)** | 0.6467 | 0.6467 0.5911 | **0.5875** | **0.6085** |
| [tomoro-colqwen3-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) | **0.6478** | **0.6226** | 0.5536 | 0.5536 0.6080 |
| [nemo-colembed-3b](https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v1) | 0.6187 | 0.6187 0.5640 | 0.5506 | 0.5506 0.5778 |
| [jinaai/jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) | 0.5994 | 0.5994 0.5178 | 0.5364 | 0.5512 |
| [nomic-ai/colnomic-embed-multimodal-7b](https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b) | 0.6224 | 0.5336 | 0.5433 | 0.5664 |

**EN.** ### ViDoRe V1 (English nDCG@5)

**中.** ### ViDoRe V1（英语 nDCG@5）

| Model<br>型号 | ArxivQA | DocVQA<br>文档VQA | InfoVQA<br>信息质量保证 | Shift<br>转变 | Syn-AI<br>同步人工智能 | Syn-Eng<br>同步工程 | Syn-Gov<br>同步政府 | Syn-Health<br>同步健康 | TabFQuAD | Tatdqa<br>塔德卡 | **Avg**<br>**平均** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[tomoro-colqwen3-8b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b)** | **0.9115** | **0.6637** | 0.9448 | 0.8789 | 0.9926 | 0.9671 | **0.9758** | 0.9906 | 0.9423 | 0.8092 | 0.9076 |
| [tomoro-colqwen3-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) | 0.9066 | 0.6624 | 0.9429 | 0.8739 | 0.9926 | 0.9691 | 0.9717 | **0.9963** | 0.9433 | 0.7983 | 0.9057 |
| [nemo-colembed-3b](https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v1) | 0.8835 | 0.6621 | **0.9492** | 0.9070 | **0.9963** | 0.9663 | 0.9782 | 0.9926 | 0.9594 | 0.8057 | **0.9100** |
| [jinaai/jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) | 0.8846 | 0.6014 | 0.9379 | **0.9293** | 0.9926 | **0.9726** | 0.9659 | 0.9913 | 0.9560 | 0.8035 | 0.9035 |
| [nomic-ai/colnomic-embed-multimodal-7b](https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b) | 0.8832 | 0.6011 | 0.9221 | 0.8930 | 0.9876 | 0.9626 | 0.9592 | 0.9926 | **0.9596** | **0.8108** | 0.8972 |

**EN.** ### Video Retrieval Evaluation

**中.** ### 视频检索评估

**EN.** To demonstrate that Tomoro ColQwen3 strongly generalizes to video retrieval, we evaluated the models on the [CareBench](https://carebench.github.io/) for text to video (General Retrieval) task and [MMEB-V2](https://huggingface.co/datasets/TIGER-Lab/MMEB-V2) video_ret benchmark.

**中.** 为了证明 Tomoro ColQwen3 对视频检索的强泛化性，我们在 [CareBench](https://carebench.github.io/) 上针对文本到视频（一般检索）任务和 [MMEB-V2](https://huggingface.co/datasets/TIGER-Lab/MMEB-V2) video_ret 基准评估了模型。

**EN.** #### CareBench Evaluation

**中.** #### CareBench 评估

**EN.** For this evaluation, we utilized a **raw video encoding** approach: our models encoded the video files directly without any additional textual annotations or metadata inputs. This highlights the model's ability to perform retrieval based purely on visual semantics.

**中.** 对于本次评估，我们使用了**原始视频编码**方法：我们的模型直接对视频文件进行编码，无需任何额外的文本注释或元数据输入。这凸显了模型纯粹基于视觉语义执行检索的能力。

| Model<br>型号 | Recall@1<br>召回@1 | Recall@5<br>召回@5 | Recall@10<br>召回@10 |
| :--- | :--- | :--- | :--- |
| **[tomoro-colqwen3-8b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b)** | **0.8670** | **0.9590** | 0.9850 |
| [tomoro-colqwen3-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) | 0.8620 | 0.9570 | 0.9800 |
| [Care7B](https://huggingface.co/MCG-NJU/CaRe-7B) | 0.7700 | 0.9560 | **0.9870** |

**EN.** #### MMEB-V2 video_ret Evaluation

**中.** #### MMEB-V2 video_ret 评估

**EN.** All below evaluations are using Hit@1 metric.

**中.** 以下所有评估均使用 Hit@1 指标。

| Model<br>型号 | MSR-VTT | MSVD | DiDeMo<br>迪德莫 | VATEX<br>瓦泰克斯 | YouCook2<br>尤库克2 | Average<br>平均 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [tomoro-colqwen3-8b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b) | 50.3 | 71.2 | 58.8 | 48.0 | 27.8 | 51.2 |
| **[tomoro-colqwen3-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b)** | 51.1 | 72.3 | **59.5** | 49.0 | 26.6 | **51.7** |
| **[IFM-TTE-7B](https://interestfm-tte.github.io/)** | 52.7 | **73.1** | 49.7 | **51.5** | **31.6** | **51.7** |
| [seed-1.6-embedding](https://seed1-6-embedding.github.io/)<br>[种子-1.6-嵌入](https://seed1-6-embedding.github.io/) | **55.3** | 71.3 | 56.7 | 48.8 | 24.6 | 51.3 |

**EN.** `IFM-TTE-7B` and `seed-1.6-embedding` utilize video-text fine-tuning, whereas the Tomoro ColQwen series relies solely on image-text data.

**中.** `IFM-TTE-7B`和`seed-1.6-embedding`利用视频文本微调，而Tomoro ColQwen系列仅依赖图像文本数据。

**EN.** ---

**中.** ---

**EN.** ## 💻 Usage

**中.** ## 💻 用法

**EN.** The processor exposes `process_texts`, `process_images`, and `score_multi_vector`.

**中.** 该处理器公开了 `process_texts`、`process_images` 和 `score_multi_vector`。

**EN.** ### Prerequisites

**中.** ### 先决条件

**EN.** We strongly suggest `flash-attn` to be installed. If not, please change to `attention_impl="sdpa"`

**中.** 我们强烈建议安装`flash-attn`。如果不是，请改为`attention_impl="sdpa"`

**EN.** Currently we only support torch==2.8.0, for higher pytorch version, please build flash attention manually, otherwise performance throughput could be low.

**中.** 目前我们仅支持torch==2.8.0，对于更高的pytorch版本，请手动构建flashattention，否则性能吞吐量可能会较低。

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers pillow requests
pip install flash-attn --no-build-isolation
```

**EN.** ### Using Sentence Transformers

**中.** ### 使用句子转换器

**EN.** `tomoro-colqwen3-embed-4b` can be used as a multi-vector (ColBERT-style late interaction) retriever directly with Sentence Transformers via the `MultiVectorEncoder`.

**中.** `tomoro-colqwen3-embed-4b` 可用作多向量（ColBERT 式后期交互）检索器，通过 `MultiVectorEncoder` 直接与句子转换器配合使用。

```bash
pip install "sentence-transformers[image]>=6.0.0"
```

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("TomoroAI/tomoro-colqwen3-embed-4b", trust_remote_code=True)

queries = [
    "What is the variable represented on the y-axis of the graph?",
    "Total outlay is maximum in which year?",
]
documents = [
    f"https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc{i}.jpg"
    for i in range(1, 5)
]

query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(f"Query 0 shape:    {tuple(query_embeddings[0].shape)}")
print(f"Document 0 shape: {tuple(document_embeddings[0].shape)}")
# Query 0 shape:    (23, 320)
# Document 0 shape: (1251, 320)

# MaxSim late-interaction scoring (rows = queries, columns = images)
scores = model.similarity(query_embeddings, document_embeddings)
print(scores)
# tensor([[12.8291,  9.0850,  6.4121,  5.8818],
#         [ 4.5928, 10.7617,  4.7812,  5.3145]])
```

**EN.** > [!NOTE]
> Pages are tiled adaptively, so document embeddings vary in length (1251 tokens for the first three
> example pages, 1271 for the fourth). MaxSim handles that, and `model.similarity` masks the padding.

**中.** > [!注意]
> 页面自适应平铺，因此文档嵌入的长度各不相同（前三个为 1251 个标记）
> 示例页面，第四个为 1271）。 MaxSim 处理该问题，`model.similarity` 屏蔽填充。

**EN.** ### Using Transformers

**中.** ### 使用变形金刚

```python
import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO

# Configuration
MODEL_ID = "TomoroAI/tomoro-colqwen3-embed-4b"
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model & Processor
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    max_num_visual_tokens=1280,
)
model = AutoModel.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
    device_map=DEVICE,
).eval()

# Sample Data
queries = [
    "Retrieve the city of Singapore",
    "Retrieve the city of Beijing",
    "Retrieve the city of London",
]
docs = [
    "https://upload.wikimedia.org/wikipedia/commons/2/27/Singapore_skyline_2022.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/6/61/Beijing_skyline_at_night.JPG",
    "https://upload.wikimedia.org/wikipedia/commons/4/49/London_skyline.jpg",
]

def load_image(url: str) -> Image.Image:
    # Some CDNs (e.g., Wikimedia) expect a browser-like UA to avoid 403s.
    for headers in ({}, {"User-Agent": "Mozilla/5.0 (compatible; ColQwen3-demo/1.0)"}):
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 403:
            continue
        resp.raise_for_status()
        try:
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except UnidentifiedImageError as e:
            raise RuntimeError(f"Failed to decode image from {url}") from e
    raise RuntimeError(f"Could not fetch image (HTTP 403) from {url}; try downloading locally and loading from file path.")

# Helper Functions
def encode_queries(texts, batch_size=8):
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = processor.process_texts(texts=texts[start : start + batch_size])
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.inference_mode():
            out = model(**batch)
            vecs = out.embeddings.to(torch.bfloat16).cpu()
        outputs.extend(vecs)
    return outputs

def encode_docs(urls, batch_size=4):
    pil_images = [load_image(url) for url in urls]
    outputs = []
    for start in range(0, len(pil_images), batch_size):
        batch_imgs = pil_images[start : start + batch_size]
        features = processor.process_images(images=batch_imgs)
        features = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
        with torch.inference_mode():
            out = model(**features)
            vecs = out.embeddings.to(torch.bfloat16).cpu()
        outputs.extend(vecs)
    return outputs

# Execution
query_embeddings = encode_queries(queries)
doc_embeddings = encode_docs(docs)

# MaxSim Scoring
scores = processor.score_multi_vector(query_embeddings, doc_embeddings)
print(scores)
```

**EN.** #### 🎞️ Lightweight Video Retrieval

**中.** #### 🎞️ 轻量级视频检索

**EN.** ColQwen3 generalizes to short videos while learning from image-text retrieval task. This minimal example samples a clip with `torchvision`, encodes queries and frames, then pools frame embeddings with a per-dimension max before MaxSim scoring.

**中.** ColQwen3 在从图像文本检索任务中学习的同时，将其推广到短视频。这个最小的示例使用 `torchvision` 对剪辑进行采样，对查询和帧进行编码，然后在 MaxSim 评分之前将帧嵌入与每个维度的最大值进行池化。

**EN.** We recommand use of maximum 5120 visual tokens for video retrieval task for best performance.

**中.** 我们建议在视频检索任务中使用最多 5120 个视觉标记，以获得最佳性能。

```python
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor

MODEL_ID = "TomoroAI/tomoro-colqwen3-embed-4b"
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    max_num_visual_tokens=5120,
)
model = AutoModel.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
    device_map=DEVICE,
).eval()

queries = ["Retrieve the football video", "Find the basketball clip", "Find the swimming clip", "Find the wrestling clip"]
videos = ["/root/sample_videos/football.mp4", "/root/sample_videos/basketball.mp4", "/root/sample_videos/swimming.mp4",  "/root/sample_videos/wrestling.mp4"]


def encode_queries(texts):
    batch = processor.process_texts(texts=texts)
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    with torch.inference_mode():
        out = model(**batch)
    return out.embeddings.to(torch.bfloat16).cpu()

def encode_videos(paths):
    vids = [str(Path(p).expanduser()) for p in paths]
    feats = processor(
        videos=vids,
        padding="longest",
        return_tensors=None,  # keep metadata as Python objects until we drop it
        videos_kwargs={"return_metadata": True},
    )
    feats.pop("video_metadata", None)  # drop metadata before forwarding to the model
    feats = feats.convert_to_tensors(tensor_type="pt")
    feats = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in feats.items()}
    with torch.inference_mode():
        out = model(**feats)
    return out.embeddings.to(torch.bfloat16).cpu()

q_emb = encode_queries(queries)
v_emb = encode_videos(videos)
scores = processor.score_multi_vector(q_emb, v_emb)
print(scores)
```

**EN.** -----

**中.** -----

**EN.** ## ⚖️ Strengths & Limitations

**中.** ## ⚖️ 优点和局限性

**EN.** ### Strengths

**中.** ### 优势

**EN.** * **Performance:** State of the art retrieval performance on ViDoRe V2 & V3 dataset with excellent performance on multimodal document retrieval.
  * **Complex Layouts:** Excellent handling of chart-rich PDFs, domain-specific documents.
  * **End-to-end Retrieval:** Capable of OCR-free retrieval on unseen multimodal documents without using an intermediate vision LLM to generate summary for retrieval.
  * **Retrieval Task Transfer:** Inherited strong text retrieval performance from the merged vector of the Qwen3-Embedding-8B model.
  * **Multilingualism:** Strong performance on non-English document inputs.

**中.** * **性能：** ViDoRe V2 和 V3 数据集上最先进的检索性能，在多模态文档检索方面具有出色的性能。
  * **复杂布局：** 出色地处理图表丰富的 PDF、特定领域的文档。
  * **端到端检索：** 能够对未见过的多模态文档进行无 OCR 检索，无需使用中间视觉语言模型来生成检索摘要。
  * **检索任务转移：**继承了Qwen3-Embedding-8B模型的合并向量强大的文本检索性能。
  * **多语言能力：** 在非英语文档输入方面表现出色。

**EN.** ### Limitations

**中.** ### 限制

**EN.** * **Video Support:** The retrieval model generalizes to video retrieval on our preliminary findings, however it's not fine-tuned on large-scale video retrieval datasets, we plan to further improve this in the future.
  * **Storage Cost:** Still larger than single‑vector baselines despite the smaller token dimension.
  * **Retrieval Instructions:** The model currently is not fine-tuned with diverse system instructions similar to Qwen3-Embedding models, we intent to improve this with more synthetic dataset in the future.

**中.** * **视频支持：** 根据我们的初步研究结果，检索模型推广到视频检索，但是它没有在大规模视频检索数据集上进行微调，我们计划在未来进一步改进这一点。
  * **存储成本：** 尽管令牌维度较小，但仍然大于单向量基线。
  * **检索说明：** 该模型目前尚未使用类似于 Qwen3-Embedding 模型的各种系统指令进行微调，我们打算在未来通过更多合成数据集来改进这一点。

**EN.** ### License & Data

**中.** ### 许可证和数据

**EN.** Distributed under **Apache 2.0**.

**中.** 在 **Apache 2.0** 下分发。

**EN.** * **Weights:** Upstream Qwen checkpoints retain their community licenses; ensure compliance when mixing.
  * **Data:** Training data includes ViDoRe/MTEB corpora and synthetic VisRAG assets.

**中.** * **权重：** 上游 Qwen 检查站保留其社区许可证；确保混合时符合要求。
  * **数据：** 训练数据包括 ViDoRe/MTEB 语料库和合成 VisRAG 资产。

**EN.** ### Acknowledgement

**中.** ### 致谢

**EN.** We gratefully acknowledge the support of **[Tomoro AI](https://tomoro.ai/)**, a leading AI engineering firm dedicated to delivering high-quality enterprise solutions that accelerate complex R&D and business transformation. This work is directly applied to enhance Tomoro’s customized multimodal agentic RAG pipelines, empowering the autonomous agents to parse, reason over, and retrieve from large-scale enterprise **internal documentation**. By bridging the gap between vision and language, this model supports Tomoro AI's mission to **accelerate the delivery of high-quality** enterprise multimodal solutions and deploy robust, production-grade intelligence across high-stakes industries.

**中.** 我们衷心感谢**[Tomoro AI](https://tomoro.ai/)**的支持，这是一家领先的人工智能工程公司，致力于提供高质量的企业解决方案，加速复杂的研发和业务转型。这项工作直接应用于增强 Tomoro 的定制多模式代理 RAG 管道，使自主代理能够解析、推理和检索大型企业**内部文档**。通过弥合视觉和语言之间的差距，该模型支持 Tomoro AI 的使命，即**加速交付高质量**企业多模式解决方案，并在高风险行业部署强大的生产级智能。

**EN.** ## 📚 Citation

**中.** ## 📚 引文

**EN.** If you use this model, please cite:

**中.** 如果您使用此模型，请引用：

```bibtex
@misc{huang2025beyond,
  author = {Huang, Xin and Tan, Kye Min},
  title = {Beyond Text: Unlocking True Multimodal, End-to-end RAG with Tomoro ColQwen3},
  year = {2025},
  url = {https://tomoro.ai/insights/beyond-text-unlocking-true-multimodal-end-to-end-rag-with-tomoro-colqwen3},
  publisher = {Tomoro.ai}
}
```
