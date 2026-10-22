# ColPali: Efficient Document Retrieval with Vision Language Models / 全文中英对照

> 来源：[arXiv:2407.01449v6](https://arxiv.org/abs/2407.01449v6)。英文段落后紧接中文，覆盖所取 HTML 版本的主文、附录、图表、脚注与参考文献；公式、数字表和书目信息共用。中文以机器初译为底稿，进行了术语整理与结构核对，尚未逐句人工审定；遇到歧义请以紧邻英文和原论文为准。 原作版权归所列作者，许可见 [原作许可](http://creativecommons.org/publicdomain/zero/1.0/)；中文译文与版式改写在该许可条件下提供（涉及 SA 时沿用相同许可，涉及 NC 时仅限非商业使用）。

## Authors and affiliations / 作者与机构

Manuel Faysse [^id2] Affiliation: Illuin Technology Affiliation: CentraleSupélec, Paris-Saclay Email: manuel.faysse@centralesupelec.fr Hugues Sibille * Affiliation: Illuin Technology Affiliation: ETH Zürich Tony Wu * Affiliation: Illuin Technology Bilel Omrani Affiliation: Illuin Technology Gautier Viaud Affiliation: Illuin Technology Céline Hudelot Affiliation: CentraleSupélec, Paris-Saclay Pierre Colombo Affiliation: Equall.ai Affiliation: CentraleSupélec, Paris-Saclay

## Abstract / 摘要

**EN.** Documents are visually rich structures that convey information through text, but also figures, page layouts, tables, or even fonts. Since modern retrieval systems mainly rely on the textual information they extract from document pages to index documents -often through lengthy and brittle processes-, they struggle to exploit key visual cues efficiently. This limits their capabilities in many practical document retrieval applications such as Retrieval Augmented Generation (RAG). To benchmark current systems on visually rich document retrieval, we introduce the Visual Document Retrieval Benchmark ViDoRe , composed of various page-level retrieval tasks spanning multiple domains, languages, and practical settings. The inherent complexity and performance shortcomings of modern systems motivate a new concept; doing document retrieval by directly embedding the images of the document pages. We release ColPali , a Vision Language Model trained to produce high-quality multi-vector embeddings from images of document pages. Combined with a late interaction matching mechanism, ColPali largely outperforms modern document retrieval pipelines while being drastically simpler, faster and end-to-end trainable. We release models, data, code and benchmarks under open licenses at [https://hf.co/vidore](https://hf.co/vidore) .

**中.** 文档是视觉丰富的结构，通过文本、图形、页面布局、表格甚至字体来传达信息。由于现代检索系统主要依赖从文档页面提取的文本信息来索引文档（通常通过漫长而脆弱的过程），因此它们很难有效地利用关键视觉线索。这限制了它们在许多实际文档检索应用中的能力，例如检索增强生成（RAG）。为了对当前系统进行视觉丰富的文档检索进行基准测试，我们引入了视觉文档检索基准 ViDoRe，它由跨越多个领域、语言和实际设置的各种页面级检索任务组成。现代系统固有的复杂性和性能缺陷催生了一个新的概念；通过直接嵌入文档页面的图像来进行文档检索。我们发布了 ColPali，这是一种经过训练的视觉语言模型，可以从文档页面的图像生成高质量的多向量嵌入。与后期交互匹配机制相结合，ColPali 的性能在很大程度上优于现代文档检索管道，同时变得更加简单、更快和端到端可训练。我们在开放许可下发布模型、数据、代码和基准 [https://hf.co/vidore](https://hf.co/vidore)。

## Introduction / 简介

**EN.** Document Retrieval consists of matching a user query to relevant documents in a given corpus. It is central to many widespread industrial applications, either as a standalone ranking system (search engines) or as part of more complex information extraction or Retrieval Augmented Generation (RAG) pipelines.

**中.** 文档检索包括将用户查询与给定语料库中的相关文档进行匹配。它是许多广泛的工业应用的核心，无论是作为独立的排名系统（搜索引擎）还是作为更复杂的信息提取或检索增强生成（RAG）管道的一部分。

**EN.** Over recent years, pretrained language models have enabled large improvements in text embedding models. In practical industrial settings, however, the primary performance bottleneck for efficient document retrieval stems not from embedding model performance but from the prior data ingestion pipeline. Indexing a standard PDF document involves several steps. First, PDF parsers or Optical Character Recognition (OCR) systems are used to extract words from the pages. Document layout detection models can then be run to segment paragraphs, titles, and other page objects such as tables, figures, and headers. A chunking strategy is then defined to group text passages with some semantical coherence, and modern retrieval setups may even integrate a captioning step to describe visually rich elements in a natural language form, more suitable for embedding models. In our experiments ( Table 2 ), we typically find that optimizing the ingestion pipeline yields much better performance on visually rich document retrieval than optimizing the text embedding model.

**中.** 近年来，预训练的语言模型使文本嵌入模型有了很大的改进。然而，在实际工业环境中，高效文档检索的主要性能瓶颈不是源于嵌入模型性能，而是源于先前的数据摄取管道。为标准 PDF 文档建立索引涉及几个步骤。首先，使用 PDF 解析器或光学字符识别 (OCR) 系统从页面中提取单词。然后可以运行文档布局检测模型来分段段落、标题和其他页面对象，例如表格、图形和标题。然后定义分块策略，以具有一定语义一致性的文本段落进行分组，现代检索设置甚至可以集成字幕步骤，以自然语言形式描述视觉丰富的元素，更适合嵌入模型。在我们的实验中（表 2），我们通常发现优化摄取管道在视觉丰富的文档检索方面比优化文本嵌入模型产生更好的性能。

**EN.** Contribution 1: ViDoRe . In this work, we argue that document retrieval systems should not be evaluated solely on the capabilities of text embedding models ( Bajaj et al., 2016 ; Thakur et al., 2021 ; Muennighoff et al., 2022 ) , but should also consider the context and visual elements of the documents to be retrieved. To this end, we create and openly release ViDoRe , a comprehensive benchmark to evaluate systems on page-level document retrieval with a wide coverage of domains, visual elements, and languages. ViDoRe addresses practical document retrieval scenarios, where queries often necessitate both textual and visual understanding for accurate document matching. We highlight the shortcomings of current text-centric systems in these settings. [^footnote1]

**中.** 贡献一：ViDoRe。在这项工作中，我们认为文档检索系统不应仅根据文本嵌入模型的能力进行评估（Bajaj et al., 2016；Thakur et al., 2021；Muennighoff et al., 2022），还应考虑要检索的文档的上下文和视觉元素。为此，我们创建并公开发布 ViDoRe，这是一个全面的基准测试，用于评估广泛覆盖领域、视觉元素和语言的页面级文档检索系统。 ViDoRe 解决了实际的文档检索场景，在这些场景中，查询通常需要文本和视觉理解才能实现准确的文档匹配。我们强调当前以文本为中心的系统在这些设置中的缺点。 [^footnote1]

**EN.** Contribution 2: ColPali . We propose a novel concept and model architecture based on Vision Language Models (VLMs) to efficiently index documents purely from their visual features, allowing for subsequent fast query matching with late interaction mechanisms ( Khattab & Zaharia, 2020 ) . Our method, ColPali , significantly outperforms all other retrieval systems on ViDoRe while being fast and end-to-end trainable. These results demonstrate the potential and the many benefits of this novel Retrieval in Vision Space concept, which could significantly alter the way document retrieval is approached in the industry moving forward. We release all resources at [https://hf.co/vidore](https://hf.co/vidore) .

**中.** 贡献2：ColPali。我们提出了一种基于视觉语言模型（VLM）的新颖概念和模型架构，可以纯粹根据视觉特征有效地索引文档，从而允许后续快速查询匹配与后期交互机制（Khattab＆Zaharia，2020）。我们的方法 ColPali 显着优于 ViDoRe 上的所有其他检索系统，同时速度快且可进行端到端训练。这些结果证明了这种新颖的视觉空间检索概念的潜力和诸多好处，它可以显着改变行业未来文档检索的方式。我们将所有资源发布在 [https://hf.co/vidore](https://hf.co/vidore) 。

![Figure S1.F1](figs/S1-F1-01.webp)

**EN.** Figure 1: ColPali simplifies document retrieval w.r.t. standard retrieval methods while achieving stronger performances with better latencies. Latencies and results are detailed in section 5 and subsection B.4 .

**中.** 图 1：ColPali 简化了文档检索。标准检索方法，同时以更好的延迟实现更强的性能。第 5 节和 B.4 小节详细介绍了延迟和结果。

## Problem Formulation & Related Work / 问题阐述及相关工作

**EN.** Problem Setting. In our setting, a retrieval system scores how relevant a document $d$ from corpus $\mathcal{D}$ is with respect to a query $q$ . Computing the similarity score $s(q,d)\in\mathbb{R}$ for each of the $|\mathcal{D}|$ documents in the corpus creates a ranking we can use to extract the most relevant documents. In this work, we focus on page-level retrieval: given a query, is the correct document page retrieved by the system? For coherence with existing literature, we further use the term document to refer to individual pages, i.e. the atomic retrieved elements in our setting. As we focus on practical industrial retrieval applications (RAG, search engines) with potentially large corpora sizes, latency constraints are imposed on scoring systems. Most current retrieval systems can be decomposed into (1) an offline indexation phase in which a document index is built and (2) an online querying phase in which a query is matched to documents from the index and where low latency is vital to the user experience.

**中.** 问题设置。在我们的设置中，检索系统对语料库 $\mathcal{D}$ 中的文档 $d$ 与查询 $q$ 的相关程度进行评分。计算语料库中每个 $|\mathcal{D}|$ 文档的相似度得分 $s(q,d)\in\mathbb{R}$ 会创建一个排名，我们可以使用它来提取最相关的文档。在这项工作中，我们关注页面级检索：给定一个查询，系统是否检索到正确的文档页面？为了与现有文献保持一致，我们进一步使用术语文档来指代各个页面，即我们设置中的原子检索元素。由于我们专注于具有潜在大语料库大小的实际工业检索应用（RAG、搜索引擎），因此对评分系统施加了延迟限制。大多数当前的检索系统可以分解为（1）离线索引阶段，其中构建文档索引；以及（2）在线查询阶段，其中查询与索引中的文档相匹配，并且低延迟对于用户体验至关重要。

**EN.** Under these industrial constraints, we identify three main properties an efficient document retrieval systems should exhibit: (R1) strong retrieval performance , as measured by standard retrieval metrics; (R2) fast online querying , measured through average latencies; (R3) high throughput corpus indexation , ie. the number of pages that can be embedded in a given timeframe.

**中.** 在这些行业限制下，我们确定了高效的文档检索系统应表现出的三个主要属性：（R1）强大的检索性能，通过标准检索指标来衡量； (R2)快速在线查询，通过平均延迟来衡量； (R3) 高通量语料库索引，即。在给定时间范围内可以嵌入的页面数。

### Textual Retrieval Methods / 文本检索方法

**EN.** Document Retrieval in Text Space.

**中.** 文本空间中的文档检索。

**EN.** Statistical methods based on word frequency like TF-IDF ( Sparck Jones, 1972 ) and BM25 ( Robertson et al., 1994 ) are still widely used due to their simplicity and efficiency. More recently, neural embedding models based on fine-tuned large language models display state-of-the-art performance on a variety of text embedding tasks and top the retrieval leaderboards ( Muennighoff et al., 2022 ) .

**中.** 基于词频的统计方法，如 TF-IDF (Sparck Jones, 1972) 和 BM25 (Robertson et al., 1994) 由于其简单和高效仍然被广泛使用。最近，基于微调的大型语言模型的神经嵌入模型在各种文本嵌入任务上显示出最先进的性能，并在检索排行榜上名列前茅（Muennighoff 等人，2022）。

**EN.** Neural Retrievers. In bi-encoder models ( Reimers & Gurevych, 2019 ; Karpukhin et al., 2020 ; Wang et al., 2022 ) , documents are independently mapped offline to a dense vector space. Queries are embedded online and matched to documents through a fast cosine distance computation. A slower, but slightly more performant alternative, cross-encoder systems ( Wang et al., 2020 ; Cohere, 2024 ) concatenate query and document as a single input sequence and iteratively attribute matching scores to each possible combination. This enables full attention computation between query and document terms but comes at the cost of computational efficiency, as $|\mathcal{D}|$ encoding passes must be done online.

**中.** 神经检索器。在双编码器模型中（Reimers & Gurevych，2019；Karpukhin 等人，2020；Wang 等人，2022），文档独立地离线映射到密集向量空间。查询在线嵌入并通过快速余弦距离计算与文档匹配。一种速度较慢但性能稍高的替代方案是交叉编码器系统（Wang et al., 2020；Cohere, 2024）将查询和文档连接为单个输入序列，并迭代地将匹配分数归因于每个可能的组合。这使得查询和文档术语之间能够进行充分的注意力计算，但是以计算效率为代价的，因为 $|\mathcal{D}|$ 编码过程必须在线完成。

**EN.** Multi-Vector retrieval via late interaction. In the late interaction paradigm introduced by ColBERT ( Khattab & Zaharia, 2020 ) , an embedding is pre-computed and indexed per document token. At runtime, similarity can be computed with individual query token embeddings. The idea is to benefit from the rich interaction between individual query and document terms while taking advantage of the offline computation and fast query matching enabled by bi-encoders. See Appendix E for more details.

**中.** 通过后期交互进行多向量检索。在 ColBERT (Khattab & Zaharia, 2020) 引入的后期交互范例中，嵌入是根据文档标记预先计算和索引的。在运行时，可以使用单独的查询标记嵌入来计算相似性。这个想法是从单个查询和文档术语之间丰富的交互中受益，同时利用双编码器启用的离线计算和快速查询匹配。详细信息请参见附录 E。

**EN.** Retrieval Evaluation. Although benchmarks and leaderboards have been developed to evaluate text embedding models ( Thakur et al., 2021 ; Muennighoff et al., 2022 ) , much of the performance improvements in industrial use cases of embedding models stem from the prior data ingestion pipeline. While documents often rely on visual elements to more efficiently convey information to human readers, text-only systems barely tap into these visual cues. Other work has also independently studied table or chart retrieval systems through repurposed Question Answering datasets ( Zhang et al., 2019 ; Nowak et al., 2024 ) but only assessing specialized methods for each task.

**中.** 检索评估。尽管已经开发了基准和排行榜来评估文本嵌入模型（Thakur 等人，2021；Muennighoff 等人，2022），但嵌入模型的工业用例中的大部分性能改进都源于先前的数据摄取管道。虽然文档通常依赖视觉元素来更有效地向人类读者传达信息，但纯文本系统几乎没有利用这些视觉线索。其他工作还通过重新调整用途的问答数据集独立研究了表格或图表检索系统（Zhang et al., 2019；Nowak et al., 2024），但仅评估每个任务的专门方法。

**EN.** To our knowledge, no benchmark evaluates document retrieval systems in practical settings; in an end-to-end manner, across several document types and topics, and by evaluating the use of both textual and visual document features.

**中.** 据我们所知，没有基准测试在实际环境中评估文档检索系统。以端到端的方式，跨多种文档类型和主题，并评估文本和视觉文档功能的使用。

### Integrating Visual features / 整合视觉特征

**EN.** Contrastive Vision Language Models. Mapping latent representations of textual content to corresponding representations of visual content has been done by aligning disjoint visual and text encoders through contrastive losses ( Radford et al., 2021 ; Zhai et al., 2023 ) . While some OCR capabilities exist in these models, the visual component is often not optimized for text understanding.

**中.** 对比视觉语言模型。通过对比损失对齐不相交的视觉和文本编码器，将文本内容的潜在表示映射到视觉内容的相应表示（Radford 等人，2021；Zhai 等人，2023）。虽然这些模型中存在一些 OCR 功能，但视觉组件通常并未针对文本理解进行优化。

**EN.** The Fine-grained Interactive Language-Image Pre-training ( Yao et al., 2021 ) framework extends the late interaction mechanism to cross-modal Vision Language Models, relying on max similarity operations between text tokens and image patches.

**中.** 细粒度交互式语言-图像预训练（Yao et al., 2021）框架将后期交互机制扩展到跨模态视觉语言模型，依赖于文本标记和图像块之间的最大相似度操作。

**EN.** Visually Rich Document Understanding. To go beyond text, some document-focused models jointly encode text tokens alongside visual or document layout features ( Appalaraju et al., 2021 ; Kim et al., 2021 ; Huang et al., 2022 ; Tang et al., 2022 ) . Large Language transformer Models (LLMs) with strong reasoning capabilities have recently been combined with Vision Transformers (ViTs) ( Dosovitskiy et al., 2020 ) to create VLMs ( Alayrac et al., 2022 ; Liu et al., 2023 ; Bai et al., 2023 ; Laurençon et al., 2024b ) where image patch vectors from contrastively trained ViT models ( Zhai et al., 2023 ) are fed as input embeddings to the LLM and concatenated with the text-token embeddings.

**中.** 视觉丰富的文档理解。为了超越文本，一些以文档为中心的模型将文本标记与视觉或文档布局特征联合编码（Appalaraju 等人，2021；Kim 等人，2021；Huang 等人，2022；Tang 等人，2022）。具有强大推理能力的大型语言转换器模型（LLM）最近与视觉转换器（ViT）（Dosovitskiy 等人，2020）相结合来创建 VLM（Alayrac 等人，2022；Liu 等人，2023；Bai 等人，2023；Laurençon 等人，2024b），其中来自对比训练的 ViT 模型的图像块向量( Zhai et al., 2023 ) 作为输入嵌入提供给 LLM 并与文本标记嵌入连接起来。

**EN.** PaliGemma. The PaliGemma-3B model ( Beyer et al., 2024 ) extends concepts from Pali3 ( Chen et al., 2023 ) , and projects SigLIP-So400m/14 ( Alabdulmohsin et al., 2023 ) patch embeddings into Gemma-2B’s text vector space ( Gemma Team et al., 2024 ) . Along with its reasonable size w.r.t. other performant VLMs, an interesting property of PaliGemma’s text model is that it is fine-tuned with full-block attention on the prefix (instruction text and image tokens). See Appendix E for more details.

**中.** 巴利语杰玛。 PaliGemma-3B 模型（Beyer 等人，2024）扩展了 Pali3（Chen 等人，2023）的概念，并将 SigLIP-So400m/14（Alabdulmohsin 等人，2023）补丁嵌入到 Gemma-2B 的文本向量空间中（Gemma Team 等人，2024）。连同其合理的尺寸。与其他高性能 VLM 相比，PaliGemma 文本模型的一个有趣特性是它通过对前缀（指令文本和图像标记）的全块注意力进行微调。详细信息请参见附录 E。

**EN.** VLMs display enhanced capabilities in Visual Question Answering, captioning, and document understanding ( Yue et al., 2023 ) , but are not optimized for retrieval tasks.

**中.** VLM 在视觉问答、字幕和文档理解方面显示出增强的功能（Yue 等人，2023），但并未针对检索任务进行优化。

## The ViDoRe Benchmark / ViDoRe 基准

**EN.** Existing benchmarks for contrastive vision-language models primarily evaluate retrieval for natural images ( Lin et al., 2014 ; Borchmann et al., 2021 ; Thapliyal et al., 2022 ) . On the other hand, textual retrieval benchmarks ( Muennighoff et al., 2022 ) are evaluated at at textual passage level and are not tailored for document retrieval tasks. We fill the gap with ViDoRe , a comprehensive benchmark for document retrieval using visual features.

**中.** 对比视觉语言模型的现有基准主要评估自然图像的检索（Lin 等人，2014；Borchmann 等人，2021；Thapliyal 等人，2022）。另一方面，文本检索基准（Muennighoff et al., 2022）是在文本段落级别进行评估的，并不是针对文档检索任务而定制的。我们用 ViDoRe 填补了这一空白，这是一个使用视觉特征进行文档检索的综合基准。

### Benchmark Design / 基准设计

**EN.** ViDoRe is designed to comprehensively evaluate retrieval systems on their capacity to match queries to relevant documents at the page level. This benchmark encompasses multiple orthogonal subtasks, with focuses on various modalities - text, figures, infographics, tables; thematic domains - medical, business, scientific, administrative; or languages - English, French. Tasks also span varying levels of complexity, in order to capture signals from both weaker and stronger systems. As many systems require large amounts of time to index pages (captioning-based approaches can take dozens of seconds per page for instance), we limit the number of candidate documents for each retrieval task in order to evaluate even complex systems in a reasonable timeframe without sacrificing quality. For trainable retrieval systems, we provide a reference training set that can be used to facilitate comparisons.

**中.** ViDoRe 旨在全面评估检索系统在页面级别将查询与相关文档匹配的能力。该基准包含多个正交子任务，重点关注各种模式 - 文本、图形、信息图表、表格；主题领域——医学、商业、科学、行政；或语言 - 英语、法语。任务还涵盖不同程度的复杂性，以便从较弱和较强的系统捕获信号。由于许多系统需要大量时间来索引页面（例如，基于字幕的方法每页可能需要数十秒），因此我们限制每个检索任务的候选文档数量，以便在合理的时间范围内评估复杂的系统而不牺牲质量。对于可训练的检索系统，我们提供了一个参考训练集，可用于促进比较​​。

| Dataset<br>数据集 | Language<br>语言 | # Queries<br># 查询 | # Documents<br># 文件 | Description<br>描述 |
| --- | --- | --- | --- | --- |
| Academic Tasks<br>学术任务 |  |  |  |  |
| DocVQA<br>文档VQA | English<br>英语 | 500 | 500 | Scanned documents from UCSF Industry<br>UCSF Industry 的扫描文件 |
| InfoVQA<br>信息质量保证 | English<br>英语 | 500 | 500 | Infographics scrapped from the web<br>从网络上删除的信息图表 |
| TAT-DQA | English<br>英语 | 1600 | 1600 | High-quality financial reports<br>高质量的财务报告 |
| arXiVQA | English<br>英语 | 500 | 500 | Scientific Figures from arXiv<br>来自 arXiv 的科学数据 |
| TabFQuAD | French<br>法语 | 210 | 210 | Tables scrapped from the web<br>从网络上删除的表格 |
| Practical Tasks<br>实际任务 |  |  |  |  |
| Energy<br>能源 | English<br>英语 | 100 | 1000 | Documents about energy<br>有关能源的文件 |
| Government<br>政府 | English<br>英语 | 100 | 1000 | Administrative documents<br>行政文件 |
| Healthcare<br>医疗保健 | English<br>英语 | 100 | 1000 | Medical documents<br>医疗文件 |
| AI<br>人工智能 | English<br>英语 | 100 | 1000 | Scientific documents related to AI<br>与AI相关的科学文献 |
| Shift Project<br>轮班项目 | French<br>法语 | 100 | 1000 | Environmental reports<br>环境报告 |

**EN.** Table 1: ViDoRe comprehensively evaluates multimodal retrieval methods.

**中.** 表 1：ViDoRe 综合评估多模态检索方法。

**EN.** Academic Tasks. We repurpose widely used visual question-answering benchmarks for retrieval tasks: for each page-question-answer triplet, we use the question as the query, and the associated page as the gold document ( Table 1 ). These academic datasets either focus on single specific modalities ( Mathew et al., 2020 ; Mathew et al., 2021 ; Li et al., 2024 ) or target more varied visually rich documents ( Zhu et al., 2022 ) . Moreover, we consider TabFQuAD, a human-labeled dataset on tables extracted from French industrial PDF documents released with this work. Details can be found in subsection A.1 .

**中.** 学术任务。我们将广泛使用的视觉问答基准重新用于检索任务：对于每个页面-问题-答案三元组，我们使用问题作为查询，并将相关页面作为黄金文档（表 1）。这些学术数据集要么专注于单一特定模式（Mathew et al., 2020；Mathew et al., 2021；Li et al., 2024），要么针对更多样化、视觉效果丰富的文档（Zhu et al., 2022）。此外，我们还考虑了 TabFQuAD，这是一个人工标记的数据集，其表格是从随本工作发布的法国工业 PDF 文档中提取的。详细信息请参见 A.1 小节。

**EN.** Practical tasks. We construct topic-specific retrieval benchmarks spanning multiple domains to go beyond repurposed QA datasets and evaluate retrieval in more realistic industrial situations (e.g. RAG). To achieve this, we collect publicly accessible PDF documents and generate queries pertaining to document pages using Claude-3 Sonnet, a high-quality proprietary vision-language model ( Anthropic, 2024 ) . In total, we collect 1,000 document pages per topic, which we associate with 100 queries extensively filtered for quality and relevance by human annotators. The corpus topics are intentionally specific to maximize syntactic proximity between documents, creating more challenging retrieval tasks and covering an array of orthogonal domains ( Table 1 ). [^footnote2]

**中.** 实际任务。我们构建跨越多个领域的特定于主题的检索基准，以超越重新调整用途的 QA 数据集，并在更现实的工业情况（例如 RAG）中评估检索。为了实现这一目标，我们收集可公开访问的 PDF 文档，并使用 Claude-3 Sonnet（一种高质量的专有视觉语言模型）生成与文档页面相关的查询（Anthropic，2024）。我们总共收集了每个主题 1,000 个文档页面，并将其与由人工注释者根据质量和相关性进行广泛过滤的 100 个查询相关联。语料库主题有意特定，以最大限度地提高文档之间的语法相似性，创建更具挑战性的检索任务并覆盖一系列正交域（表 1）。 [^footnote2]

**EN.** Evaluation Metrics. We evaluate performance on our benchmark (Requirement R1) using standard metrics from the retrieval literature (nDCG, Recall@K, MRR). We report nDCG@5 values as the main performance metric in this work and release the complete sets of results along with the models [^footnote3] . To validate compliance with practical industrial requirements ( section 2 ), we also consider query latencies (R2) and indexing throughputs (R3).

**中.** 评估指标。我们使用检索文献（nDCG、Recall@K、MRR）中的标准指标来评估基准（要求 R1）的性能。我们报告 nDCG@5 值作为这项工作的主要性能指标，并发布完整的结果集以及模型 [^footnote3] 。为了验证是否符合实际工业要求（第 2 节），我们还考虑查询延迟 (R2) 和索引吞吐量 (R3)。

### Assessing Current Systems / 评估当前系统

**EN.** Unstructured. We evaluate retrieval systems representative of those found in standard industrial RAG pipelines. As is common practice, we rely on the Unstructured [^footnote4] off-the-shelf tool in the highest resolution settings to construct high-quality text chunks from PDF documents. Unstructured orchestrates the document parsing pipeline, relying on deep learning vision models to detect titles and document layouts ( Ge et al., 2021 ) , OCR engines ( Smith, 2007 ) to extract text in non-native PDFs, specialized methods or models to detect and reconstruct tables, and implements a chunking strategy ( by-title ) that leverages the detected document structure to preserve section boundaries when concatenating texts. As is common practice, in our simplest Unstructured configuration ( text-only ), only textual elements are kept and figures, images, and tables are considered noisy information and are filtered out.

**中.** 非结构化。我们评估标准工业 RAG 管道中具有代表性的检索系统。按照惯例，我们依靠最高分辨率设置下的非结构化 [^footnote4] 现成工具从 PDF 文档构建高质量的文本块。非结构化编排文档解析管道，依靠深度学习视觉模型来检测标题和文档布局（Ge et al., 2021），OCR引擎（Smith, 2007）来提取非原生PDF中的文本，专门的方法或模型来检测和重建表格，并实现分块策略（by-title），利用检测到的文档结构在连接文本时保留部分边界。按照惯例，在我们最简单的非结构化配置（仅文本）中，仅保留文本元素，而图形、图像和表格被视为噪声信息并被过滤掉。

**EN.** Unstructured + X. While Unstructured is a strong baseline by itself, we further augment Unstructured ’s output by integrating the visual elements. In ( + OCR ), tables, charts, and images are run through an OCR engine, processed by Unstructured, and chunked independently. In ( + Captioning ), we set up a fully-fledged captioning strategy ( Zhao et al., 2023 ) , in which we feed visual elements to a strong proprietary Vision Language Model (Claude-3 Sonnet ( Anthropic, 2024 ) ) to obtain highly detailed textual descriptions of the elements. Both strategies aim to integrate visual elements in the retrieval pipeline but incur significant latency and resource costs ( subsection 5.2 ).

**中.** 非结构化 + X。虽然非结构化本身就是一个强大的基线，但我们通过集成视觉元素进一步增强了非结构化的输出。在 ( + OCR ) 中，表格、图表和图像通过 OCR 引擎运行，由 Unstructed 处理，并独立分块。在（+ Captioning）中，我们建立了一个成熟的字幕策略（Zhao et al.，2023），其中我们将视觉元素提供给强大的专有视觉语言模型（Claude-3 Sonnet（Anthropic，2024））以获得元素的高度详细的文本描述。这两种策略都旨在将视觉元素集成到检索管道中，但会产生显着的延迟和资源成本（第 5.2 小节）。

**EN.** Embedding Model. To embed textual chunks, we evaluate Okapi BM25, the de facto standard sparse statistical retrieval method, and the dense encoder of BGE-M3 ( Chen et al., 2024 ) , a multilingual neural method with SOTA performance in its size category. Chunks are embedded and scored independently, and page-level scores are obtained by max-pooling over the page’s chunk scores. [^footnote5]

**中.** 嵌入模型。为了嵌入文本块，我们评估了 Okapi BM25（事实上的标准稀疏统计检索方法）和 BGE-M3 的密集编码器（Chen 等人，2024），这是一种在其大小类别中具有 SOTA 性能的多语言神经方法。块是独立嵌入和评分的，页面级分数是通过页面块分数的最大池化获得的。 [^footnote5]

**EN.** Contrastive VLMs. We also evaluate the strongest available vision-language embedding models; Jina CLIP ( Koukounas et al., 2024 ) , Nomic Embed Vision ( Nomic, 2024 ) , and SigLIP-So400m/14 ( Alabdulmohsin et al., 2023 ) .

**中.** 对比 VLM。我们还评估了最强大的可用视觉语言嵌入模型； Jina CLIP（Koukounas 等人，2024 年）、Nomic Embed Vision（Nomic，2024 年）和 SigLIP-So400m/14（Alabdulmohsin 等人，2023 年）。

**EN.** Results. From a performance perspective, best results are obtained by combining the Unstructured parser with visual information, either from captioning strategies or by running OCR on the visual elements ( Table 2 ). Little difference is seen between BM25 and BGE-M3 embeddings highlighting the visual information bottleneck. Contrastive VLMs lag behind. Beyond retrieval performance (R1), the indexing latencies (R2) reported in Figure 2 illustrate that PDF parsing pipelines can be very lengthy, especially when incorporating OCR or captioning strategies. Querying latencies at runtime (R3) are very good for all evaluated systems ( $\leq 22$ ms on a NVIDIA L4) due to fast query encoding and cosine similarity matching.

**中.** 结果。从性能角度来看，通过将非结构化解析器与视觉信息相结合，可以通过字幕策略或在视觉元素上运行 OCR 来获得最佳结果（表 2）。 BM25 和 BGE-M3 嵌入之间几乎没有什么区别，突出了视觉信息瓶颈。对比VLM 落后了。除了检索性能 (R1) 之外，图 2 中报告的索引延迟 (R2) 表明 PDF 解析管道可能非常长，尤其是在结合 OCR 或字幕策略时。由于快速查询编码和余弦相似性匹配，运行时 (R3) 的查询延迟对于所有评估的系统（在 NVIDIA L4 上为 $\leq 22$ ms）都非常好。

![Figure S3.F2](figs/S3-F2-01.png)

**EN.** Figure 2: Offline document indexing with ColPali is much simpler and faster compared to standard retrieval methods. The PDF Parser results are obtained following the Unstructured settings with BGE-M3 detailed in subsection 3.2 . All indexing speeds are averaged per-page latencies. More details in subsection B.4

**中.** 图 2：与标准检索方法相比，使用 ColPali 进行离线文档索引更加简单和快速。 PDF 解析器结果是按照第 3.2 小节中详细介绍的 BGE-M3 的非结构化设置获得的。所有索引速度都是每页延迟的平均值。更多详细信息请参见 B.4 小节

## Late interaction based Vision Retrieval / 基于后期交互的视觉检索

### Architecture / 建筑

**EN.** Vision-Language Models. Encouraged by their strong document understanding capabilities, we propose adapting recent VLMs for retrieval. The key concept is to leverage the alignment between output embeddings of text and image tokens acquired during multi-modal fine-tuning. To this extent, we introduce ColPali , a Paligemma-3B extension that is capable of generating ColBERT-style multi-vector representations of text and images ( Figure 1 ). PaliGemma-3B is a strong candidate due to its small size, the many released checkpoints fine-tuned for different image resolutions and tasks, and the promising performances on various document understanding benchmarks. We add a projection layer to map each of the language model’s output token embeddings (whether from text or image tokens) to a vector space of reduced dimension $D=128$ as used in the ColBERT paper ( Khattab & Zaharia, 2020 ) to keep lightweight bag-of-embedding representations.

**中.** 视觉语言模型。受到其强大的文档理解能力的鼓励，我们建议采用最新的 VLM 进行检索。关键概念是利用多模式微调过程中获得的文本和图像标记的输出嵌入之间的对齐。为此，我们引入了 ColPali，这是一个 Paligemma-3B 扩展，能够生成文本和图像的 ColBERT 风格的多向量表示（图 1）。 PaliGemma-3B 是一个强有力的候选者，因为它的尺寸小，许多已发布的检查点针对不同的图像分辨率和任务进行了微调，以及在各种文档理解基准上的良好表现。我们添加一个投影层来将每个语言模型的输出标记嵌入（无论是来自文本还是图像标记）映射到 ColBERT 论文（Khattab & Zaharia，2020）中使用的降维 $D=128$ 的向量空间，以保持轻量级的嵌入袋表示。

**EN.** Late Interaction. Given query $q$ and document $d$ , we denote as $\mathbf{E_{q}}\in\mathbb{R}^{{N_{q}}\times D}$ and $\mathbf{E_{d}}\in\mathbb{R}^{N_{d}\times D}$ their respective multi-vector representation in the common embedding space $\mathbb{R}^{D}$ , where $N_{q}$ and $N_{d}$ are respectively the number of vectors in the query and in the document page embeddings. The late interaction operator, $\text{LI}\left(q,d\right)$ , is the sum over all query vectors $\mathbf{E_{q}}^{(j)}$ , of its maximum dot product $\langle\cdot|\cdot\rangle$ with each of the $N_{d}$ document embedding vectors $\mathbf{E_{d}}_{(1:N_{d})}$ .

**中.** 后期互动。给定查询 $q$ 和文档 $d$ ，我们将它们在公共嵌入空间 $\mathbb{R}^{D}$ 中各自的多向量表示表示为 $\mathbf{E_{q}}\in\mathbb{R}^{{N_{q}}\times D}$ 和 $\mathbf{E_{d}}\in\mathbb{R}^{N_{d}\times D}$ ，其中 $N_{q}$ 和 $N_{d}$ 分别是查询和文档页面中的向量数量嵌入。后期交互运算符 $\text{LI}\left(q,d\right)$ 是所有查询向量 $\mathbf{E_{q}}^{(j)}$ 的总和，其最大点积 $\langle\cdot|\cdot\rangle$ 与每个 $N_{d}$ 文档嵌入向量 $\mathbf{E_{d}}_{(1:N_{d})}$ 。

$$
\text{LI}\left(q,d\right)=\sum_{i\in[|1,N_{q}|]}\max_{j\in[|1,N_{d}|]}\langle\mathbf{E_{q}}^{(i)}|\mathbf{E_{d}}^{(j)}\rangle\tag{1}
$$

**EN.** Contrastive Loss. The Late Interaction operation is fully differentiable, enabling backpropagation. Let a batch $\left\{q_{k},d_{k}\right\}_{k\in[|1,b|]}$ composed of $b$ query-page pairs, where for all $k\in[|1,b|]$ , the document page $d_{k}$ is the document corresponding to query $q_{k}$ . Following Khattab & Zaharia (2020) , we define our in-batch contrastive loss $\mathcal{L}$ as the softmaxed cross-entropy of the positive scores $s_{k}^{+}=\text{LI}\left(q_{k},d_{k}\right)$ w.r.t. to the maximal in-batch negative scores $s_{k}^{-}=\max\limits_{l,l\neq k}\hskip 8.53581pt\text{LI}\left(q_{k},d_{l}\right)$ [^footnote6] :

**中.** 对比损失。后期交互操作是完全可微分的，可实现反向传播。让一批 $\left\{q_{k},d_{k}\right\}_{k\in[|1,b|]}$ 由 $b$ 查询页面对组成，其中对于所有 $k\in[|1,b|]$ ，文档页面 $d_{k}$ 是与查询 $q_{k}$ 对应的文档。遵循 Khattab 和 Zaharia (2020) 的观点，我们将批量对比损失 $\mathcal{L}$ 定义为正分数 $s_{k}^{+}=\text{LI}\left(q_{k},d_{k}\right)$ 的 softmax 交叉熵。到最大批量负分数 $s_{k}^{-}=\max\limits_{l,l\neq k}\hskip 8.53581pt\text{LI}\left(q_{k},d_{l}\right)$ [^footnote6] ：

$$
\mathcal{L}=-\frac{1}{b}\sum_{k=1}^{b}\log\left[\frac{\exp\left(s_{k}^{+}\right)}{\exp\left(s_{k}^{+}\right)+\exp\left(s_{k}^{-}\right)}\right]=\frac{1}{b}\sum_{k=1}^{b}\log\left(1+\exp\left(s_{k}^{-}-s_{k}^{+}\right)\right)\tag{2}
$$

### Model training / 模型训练

**EN.** Dataset. Our training dataset of 118,695 query-page pairs is comprised of train sets of openly available academic datasets ( $63\%$ ) and a synthetic dataset made up of pages from web-crawled PDF documents and augmented with VLM-generated (Claude-3 Sonnet) pseudo-questions ( $37\%$ ). Dataset split details are given in subsection A.3 . Our training set is fully English by design, enabling us to study zero-shot generalization to non-English languages [^footnote7] . We explicitly verify no multi-page PDF document is used both ViDoRe and in the train set to prevent evaluation contamination. A validation set is created with $2\%$ of the samples to tune hyperparameters. We openly release the training dataset [^footnote8] for reproducibility and to encourage further research.

**中.** 数据集。我们的训练数据集包含 118,695 个查询页面对，由公开可用的学术数据集 ( $63\%$ ) 的训练集和合成数据集组成，该合成数据集由网络爬取的 PDF 文档页面组成，并用 VLM 生成的 (Claude-3 Sonnet) 伪问题 ( $37\%$ ) 进行增强。数据集分割详细信息在 A.3 小节中给出。我们的训练集是完全英语设计的，使我们能够研究非英语语言 [^footnote7] 的零样本泛化。我们明确验证 ViDoRe 和训练集中没有使用多页 PDF 文档，以防止评估污染。使用 $2\%$ 样本创建验证集来调整超参数。我们公开发布训练数据集 [^footnote8] 以实现可重复性并鼓励进一步研究。

**EN.** Parameters. All models are trained for 1 epoch on the train set. Unless specified otherwise, we train models in bfloat16 format, use low-rank adapters (LoRA, Hu et al. (2021) ) with $\alpha=32$ and $r=32$ on the transformer layers from the language model, as well as the final randomly initialized projection layer, and use a paged_adamw_8bit optimizer. We train on an 8 GPU setup with data parallelism, a learning rate of $5e-5$ with linear decay with 2.5% warmup steps, and a batch size of 32.

**中.** 参数。所有模型都在训练集上训练 1 epoch。除非另有说明，我们以 bfloat16 格式训练模型，在语言模型的转换器层以及最终随机初始化的投影层上使用低阶适配器（LoRA，Hu et al. (2021)）和 $\alpha=32$ 和 $r=32$，以及最终随机初始化的投影层，并使用 paged_adamw_8bit 优化器。我们在具有数据并行性的 8 GPU 设置上进行训练，学习率为 $5e-5$，具有 2.5% 预热步骤的线性衰减，批量大小为 32。

**EN.** Query Augmentation. As in Khattab & Zaharia (2020) , we append 5 <unused0> tokens to the query tokens to serve as a soft, differentiable query expansion or re-weighting mechanism.

**中.** 查询增强。正如 Khattab & Zaharia (2020) 中一样，我们将 5 个 <unused0> 标记附加到查询标记中，作为软性、可微分的查询扩展或重新加权机制。

## Results / 结果

|  | ArxivQ | DocQ<br>文档Q | InfoQ<br>信息Q | TabF<br>标签F | TATQ | Shift<br>转变 | AI<br>人工智能 | Energy<br>能源 | Gov.<br>政府。 | Health.<br>健康。 | Avg.<br>平均。 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unstructured text-only<br>仅非结构化文本 |  |  |  |  |  |  |  |  |  |  |  |
| - BM25 | - | 34.1 | - | - | 44.0 | 59.6 | 90.4 | 78.3 | 78.8 | 82.6 | - |
| - BGE-M3 | - | 28.4 $\downarrow$ 5.7 | - | - | 36.1 $\downarrow$ 7.9 | 68.5 $\uparrow$ 8.9 | 88.4 $\downarrow$ 2.0 | 76.8 $\downarrow$ 1.5 | 77.7 $\downarrow$ 1.1 | 84.6 $\uparrow$ 2.0 | - |
| Unstructured + OCR<br>非结构化+OCR |  |  |  |  |  |  |  |  |  |  |  |
| - BM25 | 31.6 | 36.8 | 62.9 | 46.5 | 62.7 | 64.3 | 92.8 | 85.9 | 83.9 | 87.2 | 65.5 |
| - BGE-M3 | 31.4 $\downarrow$ 0.2 | 25.7 $\downarrow$ 11.1 | 60.1 $\downarrow$ 2.8 | 70.8 $\uparrow$ 24.3 | 50.5 $\downarrow$ 12.2 | 73.2 $\uparrow$ 8.9 | 90.2 $\downarrow$ 2.6 | 83.6 $\downarrow$ 2.3 | 84.9 $\uparrow$ 1.0 | 91.1 $\uparrow$ 3.9 | 66.1 $\uparrow$ 0.6 |
| Unstructured + Captioning<br>非结构化+字幕 |  |  |  |  |  |  |  |  |  |  |  |
| - BM25 | 40.1 | 38.4 | 70.0 | 35.4 | 61.5 | 60.9 | 88.0 | 84.7 | 82.7 | 89.2 | 65.1 |
| - BGE-M3 | 35.7 $\downarrow$ 4.4 | 32.9 $\downarrow$ 5.4 | 71.9 $\uparrow$ 1.9 | 69.1 $\uparrow$ 33.7 | 43.8 $\downarrow$ 17.7 | 73.1 $\uparrow$ 12.2 | 88.8 $\uparrow$ 0.8 | 83.3 $\downarrow$ 1.4 | 80.4 $\downarrow$ 2.3 | 91.3 $\uparrow$ 2.1 | 67.0 $\uparrow$ 1.9 |
| Contrastive VLMs<br>对比 VLM |  |  |  |  |  |  |  |  |  |  |  |
| Jina-CLIP<br>吉纳-CLIP | 25.4 | 11.9 | 35.5 | 20.2 | 3.3 | 3.8 | 15.2 | 19.7 | 21.4 | 20.8 | 17.7 |
| Nomic-vision<br>规则愿景 | 17.1 | 10.7 | 30.1 | 16.3 | 2.7 | 1.1 | 12.9 | 10.9 | 11.4 | 15.7 | 12.9 |
| SigLIP (Vanilla)<br>SigLIP（原版） | 43.2 | 30.3 | 64.1 | 58.1 | 26.2 | 18.7 | 62.5 | 65.7 | 66.1 | 79.1 | 51.4 |
| Ours<br>我们的 |  |  |  |  |  |  |  |  |  |  |  |
| SigLIP (Vanilla)<br>SigLIP（原版） | 43.2 | 30.3 | 64.1 | 58.1 | 26.2 | 18.7 | 62.5 | 65.7 | 66.1 | 79.1 | 51.4 |
| BiSigLIP (+fine-tuning)<br>BiSigLIP（+微调） | 58.5 $\uparrow$ 15.3 | 32.9 $\uparrow$ 2.6 | 70.5 $\uparrow$ 6.4 | 62.7 $\uparrow$ 4.6 | 30.5 $\uparrow$ 4.3 | 26.5 $\uparrow$ 7.8 | 74.3 $\uparrow$ 11.8 | 73.7 $\uparrow$ 8.0 | 74.2 $\uparrow$ 8.1 | 82.3 $\uparrow$ 3.2 | 58.6 $\uparrow$ 7.2 |
| BiPali (+LLM)<br>比巴利语 (+法学硕士) | 56.5 $\downarrow$ -2.0 | 30.0 $\downarrow$ -2.9 | 67.4 $\downarrow$ -3.1 | 76.9 $\uparrow$ 14.2 | 33.4 $\uparrow$ 2.9 | 43.7 $\uparrow$ 17.2 | 71.2 $\downarrow$ -3.1 | 61.9 $\downarrow$ -11.7 | 73.8 $\downarrow$ -0.4 | 73.6 $\downarrow$ -8.8 | 58.8 $\uparrow$ 0.2 |
| ColPali (+Late Inter.)<br>ColPali（+国际米兰后期） | 79.1 $\uparrow$ 22.6 | 54.4 $\uparrow$ 24.5 | 81.8 $\uparrow$ 14.4 | 83.9 $\uparrow$ 7.0 | 65.8 $\uparrow$ 32.4 | 73.2 $\uparrow$ 29.5 | 96.2 $\uparrow$ 25.0 | 91.0 $\uparrow$ 29.1 | 92.7 $\uparrow$ 18.9 | 94.4 $\uparrow$ 20.8 | 81.3 $\uparrow$ 22.5 |

**EN.** Table 2: Comprehensive evaluation of baseline models and our proposed method on ViDoRe . Results are presented using nDCG@5 metrics, and illustrate the impact of different components. Text-only metrics are not computed for benchmarks with only visual elements.

**中.** 表 2：基线模型和我们在 ViDoRe 上提出的方法的综合评估。结果使用 nDCG@5 指标呈现，并说明了不同组件的影响。对于仅包含视觉元素的基准，不会计算纯文本指标。

### Performance (R1) / 性能（R1）

**EN.** We show performance is achieved iteratively through the combination of three factors; (1) a carefully crafted task-specific dataset, (2) pairing a pretrained LLM to a vision model to better leverage text semantics from the image, and (3) using multi-vector embeddings rather than a single vector representation to better capture the vast amount of visual information present in a document.

**中.** 我们证明性能是通过三个因素的组合迭代实现的； (1) 精心制作的特定于任务的数据集，(2) 将预训练的 LLM 与视觉模型配对，以更好地利用图像中的文本语义，以及 (3) 使用多向量嵌入而不是单个向量表示来更好地捕获文档中存在的大量视觉信息。

**EN.** Fine-tuning a Vision Model on a document retrieval oriented dataset: BiSigLIP . SigLIP [^footnote9] is a strong vision-language bi-encoder producing single vector embeddings, and pretrained on billions of image-text pairs from the English split of WebLI ( Chen et al., 2023 ) . Further fine-tuning the textual component of this model on our document-oriented dataset (BiSigLIP) yields clear improvements across the board, particularly on figure retrieval (ArxivQA) and table retrieval tasks (TabFQuAD).

**中.** 在面向文档检索的数据集上微调视觉模型：BiSigLIP。 SigLIP [^footnote9] 是一种强大的视觉语言双编码器，可生成单向量嵌入，并针对来自 WebLI 英语分割的数十亿个图像文本对进行预训练（Chen 等人，2023）。在我们的面向文档的数据集 (BiSigLIP) 上进一步微调该模型的文本组件，会产生全面的明显改进，特别是在图形检索 (ArxivQA) 和表格检索任务 (TabFQuAD) 方面。

**EN.** Feeding image patches to a LLM: BiPali . In the PaliGemma model architecture, SigLIP-generated patch embeddings are fed to a text language model and we can obtain LLM contextualized output patch embeddings. [^footnote10] This technique aligns the image token representations with the text token embeddings in the LLM’s embeddings space, and augments the vision model embeddings with the language model’s text understanding capabilities. We average pool these representations to obtain a single dense vector, effectively creating a PaliGemma bi-encoder model (BiPali). After fine-tuning on the training dataset, we obtain a model that performs slightly worse in English than the tuned BiSigLIP variant. [^footnote11] However, we see notable improvements in French tasks, indicating that BiPali’s LLM (Gemma 2B) helps multilingual text understanding. This is particularly notable as our training dataset does not contain non-English samples.

**中.** 将图像补丁提供给法学硕士：BiPali。在 PaliGemma 模型架构中，SigLIP 生成的补丁嵌入被馈送到文本语言模型，我们可以获得 LLM 上下文化输出补丁嵌入。 [^footnote10] 该技术将图像标记表示与 LLM 嵌入空间中的文本标记嵌入对齐，并通过语言模型的文本理解功能增强视觉模型嵌入。我们对这些表示进行平均池以获得单个密集向量，从而有效地创建 PaliGemma 双编码器模型（BiPali）。对训练数据集进行微调后，我们获得了一个模型，该模型在英语中的表现比调整后的 BiSigLIP 变体稍差。 [^footnote11] 然而，我们在法语任务中看到了显着的改进，这表明 BiPali 的 LLM (Gemma 2B) 有助于多语言文本理解。这是特别值得注意的，因为我们的训练数据集不包含非英语样本。

**EN.** Leveraging Multi-Vector Embeddings through Late Interaction: ColPali . One benefit of inputting image patch embeddings through a language model is that they are natively mapped to a latent space similar to the textual input (query). This enables leveraging the ColBERT strategy to construct one embedding per image patch token, and at inference compute all interactions between text tokens and image patches, resulting in a step-change improvement in performance compared to BiPali. Results in Table 2 show that our ColPali model also largely outperforms the strong baselines based on Unstructured and captioning, as well as all evaluated text-image embedding models. The difference is particularly stark on the more visually complex benchmark tasks, such as InfographicVQA, ArxivQA, and TabFQuAD, respectively representing infographics, figures, and tables. However, text-centric documents are also better retrieved by the ColPali models across all evaluated domains and languages, making our approach the overall best-performing document-retrieval model.

**中.** 通过后期交互利用多向量嵌入：ColPali。通过语言模型输入图像块嵌入的好处之一是它们本机映射到类似于文本输入（查询）的潜在空间。这使得能够利用 ColBERT 策略为每个图像补丁标记构建一个嵌入，并在推理时计算文本标记和图像补丁之间的所有交互，从而与 BiPali 相比，实现性能的逐步改进。表 2 中的结果表明，我们的 ColPali 模型在很大程度上优于基于非结构化和字幕的强大基线，以及所有评估的文本图像嵌入模型。这种差异在视觉上更复杂的基准测试任务上尤其明显，例如 InfographicVQA、ArxivQA 和 TabFQuAD，分别表示信息图表、图形和表格。然而，ColPali 模型在所有评估的领域和语言中也可以更好地检索以文本为中心的文档，这使得我们的方法成为整体性能最佳的文档检索模型。

**EN.** Negative Results. For extensiveness, we also train ColSigLIP, a late interaction variant of the BiSigLIP model but obtain abysmal performances. We attribute this to the large gaps w.r.t. SigLIP’s pre-training, in which only a pooled latent representation is used in the contrastive loss, which does not optimize the representations of individual patch and token embeddings. Similarly, we train a BiSigLIP PaliGemma variant, in which we retrieve the image representations from the SigLIP model that has been further updated by PaliGemma fine-tuning, and use the text representations from PaliGemma’s text model. After fine-tuning on our dataset, performance is severely inferior to SigLIP Vanilla which simply encodes with SigLIP’s original text and vision components. This indicates a logical misalignment between SigLIP embeddings, and Gemma embeddings after PaliGemma training. We detail these results in subsection C.1 .

**中.** 负面结果。为了广泛性，我们还训练了 ColSigLIP，这是 BiSigLIP 模型的后期交互变体，但获得了糟糕的性能。我们将此归因于巨大的差距。 SigLIP 的预训练，在对比损失中仅使用池化潜在表示，这并没有优化单个补丁和令牌嵌入的表示。类似地，我们训练 BiSigLIP PaliGemma 变体，其中我们从 SigLIP 模型中检索图像表示，并通过 PaliGemma 微调进一步更新，并使用 PaliGemma 文本模型中的文本表示。对我们的数据集进行微调后，性能严重低于仅使用 SigLIP 的原始文本和视觉组件进行编码的 SigLIP Vanilla。这表明 SigLIP 嵌入和 PaliGemma 训练后的 Gemma 嵌入之间存在逻辑错位。我们在 C.1 小节中详细介绍了这些结果。

### Latencies & Memory Footprint / 延迟和内存占用

**EN.** Online Querying. (R2) Logically, querying latencies differ between ColPali and a BGE-M3 embedding model. For BGE, encoding takes about $22$ ms for 15 tokens, while encoding a query with ColPali ’s language model takes about $30$ ms [^footnote12] . For smaller corpus sizes, computing the late interaction operation induces marginally small overheads ( $\approx 1$ ms per 1000 pages in the corpus), and the cosine similarity computation between bi-encoder vectors is even faster. Optimized late interaction engines ( Santhanam et al., 2022 ; Lee et al., 2023 ) enable to easily scale corpus sizes to millions of documents with reduced latency degradations.

**中.** 在线查询。 (R2) 从逻辑上讲，ColPali 和 BGE-M3 嵌入模型之间的查询延迟有所不同。对于 BGE，对 15 个标记进行编码大约需要 $22$ 毫秒，而使用 ColPali 的语言模型对查询进行编码大约需要 $30$ 毫秒 [^footnote12] 。对于较小的语料库大小，计算后期交互操作会产生稍微小的开销（语料库中每 1000 页 $\approx 1$ 毫秒），并且双编码器向量之间的余弦相似度计算甚至更快。优化的后期交互引擎（Santhanam 等人，2022；Lee 等人，2023）能够轻松地将语料库大小扩展到数百万个文档，同时减少延迟下降。

**EN.** Offline Indexing. (R3) Standard retrieval methods using bi-encoders represent each chunk as a single vector embedding, which is easy to store and fast to compute. However, processing a PDF to get the different chunks is the most time-consuming part (layout detection, OCR, chunking), and using captioning to handle multimodal data will only exacerbate this already lengthy process. On the other hand, ColPali directly encodes pages from their image representation. Although the model is larger than standard retrieval encoders, skipping the preprocessing allows large speedups at indexing [^footnote13] ( Figure 2 ). As pages are embedded end-to-end in single forward pass, the VRAM usage depends exclusively on the sequence length (number of patches per image) which is fixed as well, enabling efficient batching strategies to fully leverage hardware acceleration. ColPali also benefits from most LLM efficiency improvements introduced in the ecosystem such as Flash Attention ( Dao, 2023 ) .

**中.** 离线索引。 (R3) 使用双编码器的标准检索方法将每个块表示为单个向量嵌入，这易于存储且计算速度快。然而，处理 PDF 以获得不同的块是最耗时的部分（布局检测、OCR、分块），而使用字幕来处理多模式数据只会加剧这个本来就很漫长的过程。另一方面，ColPali 直接根据图像表示对页面进行编码。尽管该模型比标准检索编码器大，但跳过预处理可以大幅提高索引 [^footnote13] 的速度（图 2）。由于页面在单次前向传递中端到端嵌入，VRAM 的使用完全取决于序列长度（每个图像的补丁数量），该序列长度也是固定的，从而实现高效的批处理策略，以充分利用硬件加速。 ColPali 还受益于生态系统中引入的大多数 LLM 效率改进，例如 Flash Attention（Dao，2023）。

**EN.** Storage Footprint. Our method requires storing a vector per image patch, along with 6 extra text tokens “Describe the image” concatenated to image patches. We project each PaliGemma vector to a lower dimensional space ( $D=128$ ) to maximize efficiency, leading to a memory footprint of $257.5$ KB per page ( subsection B.3 ). Importantly, the memory footprint of the naive ColBERT indexing strategy can be drastically improved through compression and clustering mechanisms ( Santhanam et al., 2022 ; Clavié et al., 2024 ) .

**中.** 存储占用空间。我们的方法需要为每个图像块存储一个向量，以及连接到图像块的 6 个额外文本标记“描述图像”。我们将每个 PaliGemma 向量投影到较低维空间 ( $D=128$ ) 以最大限度地提高效率，从而导致每页内存占用为 $257.5$ KB (B.3 小节)。重要的是，朴素 ColBERT 索引策略的内存占用可以通过压缩和聚类机制得到显着改善（Santhanam 等人，2022 年；Clavié 等人，2024 年）。

**EN.** Token pooling. Token pooling ( Clavié et al., 2024 ) is a CRUDE-compliant method (document addition/deletion-friendly) that aims amountto reduce the amount of multi-vector embeddings. For ColPali, many image patches share redundant information, e.g. white background patches. By pooling these patches together, we can reduce the amount of embeddings while retaining most information. Retrieval performance with hierarchical mean token pooling on image embeddings is shown in Figure 3 (left). With a pool factor of 3, the total number of vectors is reduced by $66.7\%$ while $97.8\%$ of the original performance is maintained. We note that the Shift dataset—composed of the most text-dense documents—is a clear outlier, showcasing more information dense documents contain less redundant patches and may be prone to worse performance degradation with such pooling techniques.

**中.** 代币池。令牌池（Clavié et al., 2024）是一种符合 CRUDE 的方法（文档添加/删除友好），旨在减少多向量嵌入的数量。对于 ColPali，许多图像块共享冗余信息，例如白色背景补丁。通过将这些补丁集中在一起，我们可以减少嵌入量，同时保留大部分信息。图 3（左）显示了图像嵌入上分层平均标记池的检索性能。当池因子为3时，向量总数减少$66.7\%$，同时保持原始性能$97.8\%$。我们注意到，由文本最密集的文档组成的 Shift 数据集是一个明显的异常值，显示更多信息密集的文档包含更少的冗余补丁，并且使用这种池化技术可能会导致更严重的性能下降。

### Interpretability / 可解释性

![Figure S5.F3](figs/S5-F3-01.png)

![Figure S5.F3](figs/S5-F3-02.webp)

**EN.** Figure 3: (Left: Token Pooling ) Relative performance degradation when reducing the number of stored embeddings per document. (Right: Interpretability ) For each term in a user query, ColPali identifies the most relevant document image patches (highlighted zones) and computes a query-to-page matching score.

**中.** 图 3：（左：令牌池）减少每个文档存储的嵌入数量时相对性能下降。 （右：可解释性）对于用户查询中的每个术语，ColPali 会识别最相关的文档图像块（突出显示区域）并计算查询到页面的匹配分数。

**EN.** By superimposing the late interaction heatmap on top of the original image, we can visualize the most salient image patches with respect to each term of the query, yielding interpretable insights into model focus zones. As epitomized in Figure 3 (right), we observe ColPali exhibits strong OCR capabilities as both the words “hourly” and “hours” present a high similarity score with the query token <_hour> . We also note particular focus on other non-trivial image features such as the x-axis representing hours being salient. Other visualization examples are shown in Appendix D .

**中.** 通过将后期交互热图叠加在原始图像之上，我们可以可视化与查询的每个术语相关的最显着的图像块，从而产生对模型焦点区域的可解释的见解。如图 3（右）所示，我们观察到 ColPali 表现出强大的 OCR 功能，因为单词“hourly”和“hours”与查询标记 <_hour> 都呈现出很高的相似度分数。我们还注意到特别关注其他重要的图像特征，例如代表小时的 x 轴是显着的。其他可视化示例显示在附录 D 中。

## Ablation study / 消融研究

**EN.** We run various ablations to better understand the mechanisms at play. By default, result deltas reported below refer to nDCG@5 values averaged over all ViDoRe tasks. Detailed results in C.2 .

**中.** 我们进行了各种消融，以更好地了解起作用的机制。默认情况下，下面报告的结果增量是指所有 ViDoRe 任务的 nDCG@5 平均值。详细结果见C.2。

**EN.** Tradeoffs between model size and the number of image patches. We train a variant of PaliGemma with half the number of image patches (512). While we observe a clear performance degradation with respects to to the 1024-patch ColPali model ( $-24.8$ nDCG@5), memory usage is much lower. As an alternative to PaliGemma, we train Idefics2-8B ( Laurençon et al., 2024b ) , a VLM with a similar architecture and based on a Mistral-7B ( Jiang et al., 2023 ) language backbone and a SigLIP vision encoder paired with a perceiver resampler. The most notable differences with PaliGemma lie in the size of the language model (2B and 7B resp.) and the number of image patches (between 512 and 2048 for PaliGemma, and 64 post-resampling for Idefics2 [^footnote14] ). Our results suggest better language models enable more efficient representations of image embeddings - ColIdefics2 with 64 patches largely outperforms out ColPali with 512 patches (+20.1 nDCG@5). However ColIdefics2 (64) remains less accurate than ColPali (1024) ( $-4.7$ nDCG@5) while being about twice as slow in terms of training and inference latency. These results suggest there are tradeoffs between performance (R1), latencies during online querying (R2) and offline indexation phases (R3), and index memory size.

**中.** 模型大小和图像块数量之间的权衡。我们用一半数量的图像块（512）训练 PaliGemma 的变体。虽然我们观察到 1024 个补丁 ColPali 模型 ($-24.8$ nDCG@5) 的性能明显下降，但内存使用量却低得多。作为 PaliGemma 的替代方案，我们训练 Idefics2-8B（Laurençon 等人，2024b），这是一种具有类似架构的 VLM，基于 Mistral-7B（Jiang 等人，2023）语言主干和与感知器重采样器配对的 SigLIP 视觉编码器。与 PaliGemma 最显着的差异在于语言模型的大小（分别为 2B 和 7B）和图像块的数量（PaliGemma 为 512 到 2048 之间，Idefics2 [^footnote14] 为 64 后重采样）。我们的结果表明，更好的语言模型可以更有效地表示图像嵌入 - 具有 64 个补丁的 ColIdefics2 很大程度上优于具有 512 个补丁的 ColPali (+20.1 nDCG@5)。然而，ColIdefics2 (64) 的准确度仍然低于 ColPali (1024) ($-4.7$ nDCG@5)，同时在训练和推理延迟方面慢了大约两倍。这些结果表明，性能 (R1)、在线查询 (R2) 和离线索引阶段 (R3) 期间的延迟以及索引内存大小之间存在权衡。

**EN.** Unfreezing the vision component. We train a ColPali variant by also backpropagating through and updating the vision encoder and the projection layer. This leads to a slight performance degradation ( $-0.7$ nDCG@5). These conclusions may change with larger scales of training data.

**中.** 解冻视觉组件。我们还通过反向传播和更新视觉编码器和投影层来训练 ColPali 变体。这会导致性能轻微下降 ($-0.7$ nDCG@5)。这些结论可能会随着训练数据规模的扩大而改变。

**EN.** Impact of “query augmentation” tokens. In ColBERT, special tokens are concatenated to the input query to serve as soft query augmentation buffers. Training without these tokens, we observe no significant performance difference in the English benchmarks. However, performance on the French tasks seems to improve ( $+9.8$ nDCG@G on Shift, $+6.3$ nDCG@5 on TabFQuAD, Table 7 ).

**中.** “查询增强”令牌的影响。在 ColBERT 中，特殊标记被连接到输入查询以充当软查询增强缓冲区。在没有这些标记的情况下进行训练，我们观察到英语基准测试中没有显着的性能差异。然而，法语任务的性能似乎有所提高（Shift 上的 $+9.8$ nDCG@G，TabFQuAD 上的 $+6.3$ nDCG@5，表 7）。

**EN.** Impact of the Pairwise CE loss. Training with an in-batch negative contrastive loss, instead of the pairwise CE loss that only considers the hardest negative sample, leads to a slight performance degradation ( $-1.6$ nDCG@5) on the aggregated benchmark.

**中.** Pairwise CE 损失的影响。使用批量负对比损失进行训练，而不是仅考虑最困难的负样本的成对 CE 损失，会导致聚合基准上的轻微性能下降（$-1.6$ nDCG@5）。

**EN.** Adapting models to new tasks. Contrary to more complex multi-step retrieval pipelines, ColPali can be trained end-to-end, directly optimizing the downstream retrieval task which greatly facilitates fine-tuning to boost performance on specialized domains, multilingual retrieval, or specific visual elements the model struggles with. To demonstrate, we add 1552 samples representing French tables and associated queries to the training set. This represents the only French data in the training set, with all other examples being kept unchanged. We see clear nDCG@5 improvements ( $+2.6$ ) and even starker Recall@1 gains ( $+5$ ) on the TabFQuAD benchmark, with no performance degradation on the rest of the benchmark tasks ( $+0.4$ nDCG@5 overall).

**中.** 使模型适应新任务。与更复杂的多步骤检索管道相反，ColPali 可以进行端到端训练，直接优化下游检索任务，这极大地促进了微调，以提高模型在专业领域、多语言检索或特定视觉元素上的性能。为了进行演示，我们将代表法语表和相关查询的 1552 个样本添加到训练集中。这代表训练集中唯一的法语数据，所有其他示例保持不变。我们在 TabFQuAD 基准测试中看到了明显的 nDCG@5 改进 ( $+2.6$ ) 甚至更明显的 Recall@1 增益 ( $+5$ )，而其余基准测试任务的性能没有下降 (总体 $+0.4$ nDCG@5 )。

**EN.** Better VLMs lead to better visual retrievers. As improved VLMs are released, it is interesting to observe if improved performances on generative tasks translate once these models are adapted for image retrieval tasks through ColPali training strategies. We train the recently released Qwen2-VL 2B ( Wang et al., 2024b ) , a SOTA 2 billion parameter generative VLM, with the same data and training strategy, obtaining ColQwen2-VL. To approximately match ColPali’s memory requirements, we limit the number of image patches to 768, slightly less than ColPali’s 1024 patches. We observe clear performance improvements of $+5.3$ nDCG@5 values over ColPali showcasing clear performance correlations between generative benchmarks performance and retrieving metrics.

**中.** 更好的 VLM 会带来更好的视觉检索器。随着改进的 VLM 的发布，观察一旦这些模型通过 ColPali 训练策略适应图像检索任务，生成任务的性能改进是否会转化，这是很有趣的。我们使用相同的数据和训练策略训练最近发布的Qwen2-VL 2B（Wang et al., 2024b），这是一个SOTA 20亿参数生成VLM，得到ColQwen2-VL。为了大致匹配 ColPali 的内存要求，我们将图像补丁的数量限制为 768 个，略少于 ColPali 的 1024 个补丁。我们观察到 $+5.3$ nDCG@5 值相对于 ColPali 的明显性能改进，展示了生成基准性能和检索指标之间的明显性能相关性。

**EN.** Out-of-domain generalization. Some of the datasets in the ViDoRe benchmark have train sets, which we have integrated within the ColPali train set (eg. academic tasks). This is standard in embedding models ( Wang et al., 2024a ; Lee et al., 2024 ) , and while ColPali also exhibits strong performance on tasks in which this is not the case (French data is never seen by the model during training for instance), it remains interesting to evaluate model performance when training is done on a fully disjoint data distribution. We train a ColPali variant solely using the recent DocMatix dataset ( Laurençon et al., 2024a ) , a large scale, synthetically annotated visual document question answering dataset, which we subsample to obtain a comparably-sized train set. Results on ViDoRe show the performance drop is minor ( $-2.2$ nDCG@5), still outperforming the closest baseline method by over 12 points. These results showcase ColPali generalizes well outside of its training distribution, and demonstrate that our results are not unreasonably boosted with respect to baselines (BGE-M3) that cannot be fine-tuned on the same data [^footnote15]

**中.** 域外泛化。 ViDoRe 基准测试中的一些数据集具有训练集，我们已将其集成到 ColPali 训练集中（例如学术任务）。这是嵌入模型的标准（Wang et al., 2024a；Lee et al., 2024），虽然 ColPali 在非这种情况的任务中也表现出了强大的性能（例如，模型在训练期间从未见过法国数据），但在完全不相交的数据分布上进行训练时，评估模型性能仍然很有趣。我们仅使用最近的 DocMatix 数据集（Laurençon 等人，2024a）来训练 ColPali 变体，这是一个大规模的、综合注释的视觉文档问答数据集，我们对其进行二次采样以获得相当大小的训练集。 ViDoRe 上的结果显示性能下降很小（$-2.2$ nDCG@5），仍然比最接近的基线方法高出 12 个点以上。这些结果展示了 ColPali 在其训练分布之外的泛化能力，并证明我们的结果相对于无法在相同数据 [^footnote15] 上进行微调的基线 (BGE-M3) 而言并没有无理地提升

## Conclusions / 结论

**EN.** In this work, we introduced the Visual Document Retrieval Benchmark ( ViDoRe ), which evaluates document retrieval systems in realistic settings involving visually complex documents. We demonstrated that current retrieval pipelines and contrastive vision-language models struggle to efficiently exploit visual information embedded in documents, leading to suboptimal performance. To address this, we presented ColPali , a novel retrieval method that leverages Vision-Language Models to create high-quality, multi-vector embeddings purely from visual document features. ColPali largely outperforms the best existing document retrieval methods while enabling faster corpus indexing times and maintaining low querying latencies, thus circumventing many pain points of modern document retrieval applications. We hope to drive industrial adoption, and to encourage future work by publicly releasing the ViDoRe benchmark, the data, the codebase, and all models and baselines from our work.

**中.** 在这项工作中，我们引入了视觉文档检索基准（ViDoRe），它在涉及视觉复杂文档的现实环境中评估文档检索系统。我们证明了当前的检索管道和对比视觉语言模型难以有效地利用文档中嵌入的视觉信息，从而导致性能不佳。为了解决这个问题，我们提出了 ColPali，这是一种新颖的检索方法，它利用视觉语言模型纯粹根据视觉文档特征创建高质量的多向量嵌入。 ColPali 在很大程度上优于现有的最佳文档检索方法，同时实现更快的语料库索引时间并保持较低的查询延迟，从而规避了现代文档检索应用程序的许多痛点。我们希望通过公开发布 ViDoRe 基准、数据、代码库以及我们工作中的所有模型和基线来推动行业采用，并鼓励未来的工作。

**EN.** Future Work. Beyond performance improvements that could be obtained through better data, backbone models or training strategies, our vision at term is to combine visual retrieval systems and visually grounded query answering to create end-to-end RAG systems that purely function from image features. This idea is supported by concurrent work ( Ma et al., 2024 ) showcasing the strong promises of VLMs for visual QA, and may eventually become a new industrial standard for document processing. In this line of work, reliability is key, and confidence estimation techniques for Information Retrieval methods could become central to implement abstention mechanisms ( Gisserot-Boukhlef et al., 2024 ) , and are particularly interesting given the information rich multi-vector scoring mechanisms of late interaction systems. Expanding benchmarking efforts to cover more languages, modalities, and tasks is also a crucial future research direction ( Jiang et al., 2024 ) .

**中.** 未来的工作。除了可以通过更好的数据、主干模型或训练策略获得性能改进之外，我们的长期愿景是将视觉检索系统和基于视觉的查询应答相结合，创建纯粹根据图像特征运行的端到端 RAG 系统。这一想法得到了并行工作（Ma et al., 2024）的支持，展示了 VLM 对视觉 QA 的强大前景，并可能最终成为文档处理的新工业标准。在这方面的工作中，可靠性是关键，信息检索方法的置信度估计技术可能成为实现弃权机制的核心（Gisserot-Boukhlef 等人，2024），并且考虑到后期交互系统的信息丰富的多向量评分机制，该技术特别有趣。扩大基准测试工作以涵盖更多语言、模式和任务也是未来的一个重要研究方向（Jiang et al., 2024）。

## Reproducibility Statement / 再现性声明

**EN.** For transparency, reproducibility and to foster future work, we release our training data, model checkpoints (adapters), entire codebase, and complete evaluation benchmark under MIT licenses as detailed in the main paper. We also host a public ViDoRe leaderboard [^footnote16] to foster concurrent work in the space. The supplementary material further details training configurations for our models (also specified in HuggingFace model repositories), and dives into the process we used to generate synthetic data, how latency computations are performed, as well as provides further detailed evaluation results.

**中.** 为了透明度、可重复性和促进未来的工作，我们在 MIT 许可下发布了我们的训练数据、模型检查点（适配器）、整个代码库和完整的评估基准，如主论文中详述。我们还主办了一个公共 ViDoRe 排行榜 [^footnote16]，以促进该领域的并发工作。补充材料进一步详细介绍了我们模型的训练配置（也在 HuggingFace 模型存储库中指定），并深入探讨了我们用于生成合成数据的过程、如何执行延迟计算，并提供了进一步详细的评估结果。

## Acknowledgements / 致谢

**EN.** This work is partially supported by Illuin Technology, and by a grant from ANRT France. This work was performed using HPC resources from the CINES ADASTRA through Grant 2024-AD011015443 and from IDRIS with grant 2024-AD011015724R1. We extend our warm thanks to Jonathan Dong, Caio Corro, Victor Pellegrain and Ender Konukoglu for their valuable feedback on the paper.

**中.** 这项工作得到了 Illuin Technology 的部分支持，并得到了法国 ANRT 的资助。这项工作是使用来自 CINES ADASTRA 的 HPC 资源（通过拨款 2024-AD011015443）和来自 IDRIS 的 HPC 资源（通过拨款 2024-AD011015724R1）进行的。我们衷心感谢 Jonathan Dong、Caio Corro、Victor Pellegrain 和 Ender Konukoglu 对本文提出的宝贵反馈。

## References / 参考文献

- Alabdulmohsin et al. (2023) Ibrahim Alabdulmohsin, Xiaohua Zhai, Alexander Kolesnikov, and Lucas Beyer. Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design. 2023. doi: 10.48550/ARXIV.2305.13035 . URL [https://arxiv.org/abs/2305.13035](https://arxiv.org/abs/2305.13035) . Publisher: arXiv Version Number: 5.

- Alayrac et al. (2022) Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a Visual Language Model for Few-Shot Learning. 2022. doi: 10.48550/ARXIV.2204.14198 . URL [https://arxiv.org/abs/2204.14198](https://arxiv.org/abs/2204.14198) . Publisher: arXiv Version Number: 2.

- Anthropic (2024) Anthropic. The Claude 3 Model Family: Opus, Sonnet, Haiku, 2024. URL [https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf) .

- Appalaraju et al. (2021) Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, and R. Manmatha. DocFormer: End-to-End Transformer for Document Understanding, 2021. URL [https://arxiv.org/abs/2106.11539](https://arxiv.org/abs/2106.11539) . Version Number: 2.

- Bai et al. (2023) Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond. 2023. doi: 10.48550/ARXIV.2308.12966 . URL [https://arxiv.org/abs/2308.12966](https://arxiv.org/abs/2308.12966) . Publisher: arXiv Version Number: 3.

- Bajaj et al. (2016) Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. MS MARCO: A Human Generated MAchine Reading COmprehension Dataset, 2016. URL [https://arxiv.org/abs/1611.09268](https://arxiv.org/abs/1611.09268) . Version Number: 3.

- Beyer et al. (2024) Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Bošnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmsen, and Xiaohua Zhai. Paligemma: A versatile 3b vlm for transfer, 2024. URL [https://arxiv.org/abs/2407.07726](https://arxiv.org/abs/2407.07726) .

- Bloom (1970) Burton H. Bloom. Space/time trade-offs in hash coding with allowable errors. Commun. ACM , 13(7):422–426, July 1970. ISSN 0001-0782. doi: 10.1145/362686.362692 . URL [https://doi.org/10.1145/362686.362692](https://doi.org/10.1145/362686.362692) . Place: New York, NY, USA Publisher: Association for Computing Machinery.

- Borchmann et al. (2021) Łukasz Borchmann, Michał Pietruszka, Tomasz Stanislawek, Dawid Jurkiewicz, Michał Turski, Karolina Szyndler, and Filip Graliński. DUE: End-to-End Document Understanding Benchmark. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2) , 2021. URL [https://openreview.net/forum?id=rNs2FvJGDK](https://openreview.net/forum?id=rNs2FvJGDK) .

- Chen et al. (2024) Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation, 2024. URL [https://arxiv.org/abs/2402.03216](https://arxiv.org/abs/2402.03216) . Version Number: 3.

- Chen et al. (2023) Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, Daniel Salz, Xi Xiong, Daniel Vlasic, Filip Pavetic, Keran Rong, Tianli Yu, Daniel Keysers, Xiaohua Zhai, and Radu Soricut. PaLI-3 Vision Language Models: Smaller, Faster, Stronger, 2023. URL [https://arxiv.org/abs/2310.09199](https://arxiv.org/abs/2310.09199) . Version Number: 2.

- Clavié et al. (2024) Benjamin Clavié, Antoine Chaffin, and Griffin Adams. Reducing the Footprint of Multi-Vector Retrieval with Minimal Performance Impact via Token Pooling, 2024. URL [https://arxiv.org/abs/2409.14683](https://arxiv.org/abs/2409.14683) . Version Number: 1.

- Cohere (2024) Cohere. Introducing Rerank 3: A New Foundation Model for Efficient Enterprise Search & Retrieval, April 2024. URL [https://cohere.com/blog/rerank-3](https://cohere.com/blog/rerank-3) .

- Dao (2023) Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning, 2023. URL [https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691) .

- Darcet et al. (2023) Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision Transformers Need Registers. 2023. doi: 10.48550/ARXIV.2309.16588 . URL [https://arxiv.org/abs/2309.16588](https://arxiv.org/abs/2309.16588) . Publisher: [object Object] Version Number: 2.

- Devlin et al. (2018) Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, 2018. URL [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) . Version Number: 2.

- Dosovitskiy et al. (2020) Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. 2020. doi: 10.48550/ARXIV.2010.11929 . URL [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929) . Publisher: arXiv Version Number: 2.

- Ge et al. (2021) Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun. YOLOX: Exceeding YOLO Series in 2021, 2021. URL [https://arxiv.org/abs/2107.08430](https://arxiv.org/abs/2107.08430) . Version Number: 2.

- Gemma Team et al. (2024) Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, Léonard Hussenot, Pier Giuseppe Sessa, Aakanksha Chowdhery, Adam Roberts, Aditya Barua, Alex Botev, Alex Castro-Ros, Ambrose Slone, Amélie Héliou, Andrea Tacchetti, Anna Bulanova, Antonia Paterson, Beth Tsai, Bobak Shahriari, Charline Le Lan, Christopher A. Choquette-Choo, Clément Crepy, Daniel Cer, Daphne Ippolito, David Reid, Elena Buchatskaya, Eric Ni, Eric Noland, Geng Yan, George Tucker, George-Christian Muraru, Grigory Rozhdestvenskiy, Henryk Michalewski, Ian Tenney, Ivan Grishchenko, Jacob Austin, James Keeling, Jane Labanowski, Jean-Baptiste Lespiau, Jeff Stanway, Jenny Brennan, Jeremy Chen, Johan Ferret, Justin Chiu, Justin Mao-Jones, Katherine Lee, Kathy Yu, Katie Millican, Lars Lowe Sjoesund, Lisa Lee, Lucas Dixon, Machel Reid, Maciej Mikuła, Mateo Wirth, Michael Sharman, Nikolai Chinaev, Nithum Thain, Olivier Bachem, Oscar Chang, Oscar Wahltinez, Paige Bailey, Paul Michel, Petko Yotov, Rahma Chaabouni, Ramona Comanescu, Reena Jana, Rohan Anil, Ross McIlroy, Ruibo Liu, Ryan Mullins, Samuel L Smith, Sebastian Borgeaud, Sertan Girgin, Sholto Douglas, Shree Pandya, Siamak Shakeri, Soham De, Ted Klimenko, Tom Hennigan, Vlad Feinberg, Wojciech Stokowiec, Yu-hui Chen, Zafarali Ahmed, Zhitao Gong, Tris Warkentin, Ludovic Peran, Minh Giang, Clément Farabet, Oriol Vinyals, Jeff Dean, Koray Kavukcuoglu, Demis Hassabis, Zoubin Ghahramani, Douglas Eck, Joelle Barral, Fernando Pereira, Eli Collins, Armand Joulin, Noah Fiedel, Evan Senter, Alek Andreev, and Kathleen Kenealy. Gemma: Open Models Based on Gemini Research and Technology, 2024. URL [https://arxiv.org/abs/2403.08295](https://arxiv.org/abs/2403.08295) . Version Number: 4.

- Gisserot-Boukhlef et al. (2024) Hippolyte Gisserot-Boukhlef, Manuel Faysse, Emmanuel Malherbe, Céline Hudelot, and Pierre Colombo. Towards trustworthy reranking: A simple yet effective abstention mechanism, 2024. URL [https://arxiv.org/abs/2402.12997](https://arxiv.org/abs/2402.12997) .

- Hu et al. (2021) Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. 2021. doi: 10.48550/ARXIV.2106.09685 . URL [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) . Publisher: arXiv Version Number: 2.

- Huang et al. (2022) Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. 2022. doi: 10.48550/ARXIV.2204.08387 . URL [https://arxiv.org/abs/2204.08387](https://arxiv.org/abs/2204.08387) . Publisher: arXiv Version Number: 3.

- Jiang et al. (2023) Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7B. 2023. doi: 10.48550/ARXIV.2310.06825 . URL [https://arxiv.org/abs/2310.06825](https://arxiv.org/abs/2310.06825) . Publisher: arXiv Version Number: 1.

- Jiang et al. (2024) Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. Vlm2vec: Training vision-language models for massive multimodal embedding tasks, 2024. URL [https://arxiv.org/abs/2410.05160](https://arxiv.org/abs/2410.05160) .

- Karpukhin et al. (2020) Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense Passage Retrieval for Open-Domain Question Answering, 2020. URL [https://arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906) . Version Number: 3.

- Khattab & Zaharia (2020) Omar Khattab and Matei Zaharia. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. 2020. doi: 10.48550/ARXIV.2004.12832 . URL [https://arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832) .

- Kim et al. (2021) Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. OCR-free Document Understanding Transformer, 2021. URL [https://arxiv.org/abs/2111.15664](https://arxiv.org/abs/2111.15664) . Version Number: 5.

- Koukounas et al. (2024) Andreas Koukounas, Georgios Mastrapas, Michael Günther, Bo Wang, Scott Martens, Isabelle Mohr, Saba Sturua, Mohammad Kalim Akram, Joan Fontanals Martínez, Saahil Ognawala, Susana Guzman, Maximilian Werk, Nan Wang, and Han Xiao. Jina CLIP: Your CLIP Model Is Also Your Text Retriever, 2024. URL [https://arxiv.org/abs/2405.20204](https://arxiv.org/abs/2405.20204) . Version Number: 1.

- Laurençon et al. (2024a) Hugo Laurençon, Andrés Marafioti, Victor Sanh, and Léo Tronchon. Building and better understanding vision-language models: insights and future directions., 2024a.

- Laurençon et al. (2024b) Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models?, May 2024b. URL [http://arxiv.org/abs/2405.02246](http://arxiv.org/abs/2405.02246) . arXiv:2405.02246 [cs].

- Lee et al. (2024) Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nv-embed: Improved techniques for training llms as generalist embedding models, 2024. URL [https://arxiv.org/abs/2405.17428](https://arxiv.org/abs/2405.17428) .

- Lee et al. (2023) Jinhyuk Lee, Zhuyun Dai, Sai Meher Karthik Duddu, Tao Lei, Iftekhar Naim, Ming-Wei Chang, and Vincent Y. Zhao. Rethinking the Role of Token Retrieval in Multi-Vector Retrieval, 2023. URL [https://arxiv.org/abs/2304.01982](https://arxiv.org/abs/2304.01982) . Version Number: 3.

- Li et al. (2024) Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models, 2024.

- Lin et al. (2014) Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. Microsoft COCO: Common Objects in Context, 2014. URL [https://arxiv.org/abs/1405.0312](https://arxiv.org/abs/1405.0312) . Version Number: 3.

- Liu et al. (2023) Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual Instruction Tuning. 2023. doi: 10.48550/ARXIV.2304.08485 . URL [https://arxiv.org/abs/2304.08485](https://arxiv.org/abs/2304.08485) . Publisher: arXiv Version Number: 1.

- Lucas Beyer* et al. (2024) Lucas Beyer*, Andreas Steiner*, André Susano Pinto*, Alexander Kolesnikov*, Xiao Wang*, Xiaohua Zhai*, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Jeremiah Harmsen, Daniel Keysers, Neil Houlsby, Xi Chen, Emanuele Bugliarello, Thomas Unterthiner, Keran Rong, Matthias Minderer, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Julian Eisenschlos, Manoj Kumar, Matko Bošnjak, Matthias Bauer, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Paul Voigtlaender, Pinelopi Papalampidi, Olivier Henaff, Skanda Koppula, Xi Xiong, Radu Soricut, Model release contributors and general support, Tris Warkentin, Kat Black, Luiz Gustavo Martins, Glenn Cameron, Raj Gundluru, Manvinder Singh, Meg Risdal, Nilay Chauhan, Nate Keating, Nesh Devanathan, Elisa Bandy, Joe Fernandez, Antonia Paterson, Jenny Brennan, Tom Eccles, Pankil Botadra, Ben Bariach, Lav Rai, Minwoo Park, Dustin Luong, Daniel Vlasic, Bo Wu, Wenming Ye, Divyashree Sreepathihalli, Kiranbir Sodhia, Alek Andreev, Armand Joulin, Surya Bhupatiraju, Minh Giang, Joelle Barral, and Zoubin Ghahramani. PaliGemma, 2024. URL [https://www.kaggle.com/m/23393](https://www.kaggle.com/m/23393) .

- Ma et al. (2024) Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, Pan Zhang, Liangming Pan, Yu-Gang Jiang, Jiaqi Wang, Yixin Cao, and Aixin Sun. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations, 2024. URL [https://arxiv.org/abs/2407.01523](https://arxiv.org/abs/2407.01523) .

- Mathew et al. (2020) Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. DocVQA: A Dataset for VQA on Document Images. 2020. doi: 10.48550/ARXIV.2007.00398 . URL [https://arxiv.org/abs/2007.00398](https://arxiv.org/abs/2007.00398) .

- Mathew et al. (2021) Minesh Mathew, Viraj Bagal, Rubèn Pérez Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V Jawahar. InfographicVQA, 2021. URL [https://arxiv.org/abs/2104.12756](https://arxiv.org/abs/2104.12756) . Version Number: 2.

- Muennighoff et al. (2022) Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. MTEB: Massive Text Embedding Benchmark, 2022. URL [https://arxiv.org/abs/2210.07316](https://arxiv.org/abs/2210.07316) . Version Number: 3.

- Nomic (2024) Nomic. Nomic Embed Vision: Expanding The Nomic Latent Space, June 2024. URL [https://blog.nomic.ai/posts/nomic-embed-vision](https://blog.nomic.ai/posts/nomic-embed-vision) .

- Nowak et al. (2024) Averi Nowak, Francesco Piccinno, and Yasemin Altun. Multimodal chart retrieval: A comparison of text, table and image based approaches. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) , pp. 5488–5505, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.307 . URL [https://aclanthology.org/2024.naacl-long.307](https://aclanthology.org/2024.naacl-long.307) .

- Radford et al. (2021) Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision. 2021. doi: 10.48550/ARXIV.2103.00020 . URL [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020) . Publisher: arXiv Version Number: 1.

- Reimers & Gurevych (2019) Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks, 2019. URL [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084) . Version Number: 1.

- Robertson et al. (1994) Stephen E. Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu, and Mike Gatford. Okapi at TREC-3. In Donna K. Harman (ed.), Proceedings of The Third Text REtrieval Conference, TREC 1994, Gaithersburg, Maryland, USA, November 2-4, 1994 , volume 500-225 of NIST Special Publication , pp. 109–126. National Institute of Standards and Technology (NIST), 1994. URL [http://trec.nist.gov/pubs/trec3/papers/city.ps.gz](http://trec.nist.gov/pubs/trec3/papers/city.ps.gz) .

- Santhanam et al. (2022) Keshav Santhanam, Omar Khattab, Christopher Potts, and Matei Zaharia. PLAID: An Efficient Engine for Late Interaction Retrieval, 2022. URL [https://arxiv.org/abs/2205.09707](https://arxiv.org/abs/2205.09707) . Version Number: 1.

- Smith (2007) R. Smith. An Overview of the Tesseract OCR Engine. In Ninth International Conference on Document Analysis and Recognition (ICDAR 2007) Vol 2 , pp. 629–633, Curitiba, Parana, Brazil, September 2007. IEEE. ISBN 978-0-7695-2822-9. doi: 10.1109/ICDAR.2007.4376991 . URL [http://ieeexplore.ieee.org/document/4376991/](http://ieeexplore.ieee.org/document/4376991/) . ISSN: 1520-5363.

- Sparck Jones (1972) Karen Sparck Jones. A STATISTICAL INTERPRETATION OF TERM SPECIFICITY AND ITS APPLICATION IN RETRIEVAL. Journal of Documentation , 28(1):11–21, January 1972. ISSN 0022-0418. doi: 10.1108/eb026526 . URL [https://www.emerald.com/insight/content/doi/10.1108/eb026526/full/html](https://www.emerald.com/insight/content/doi/10.1108/eb026526/full/html) .

- Tang et al. (2022) Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit Bansal. Unifying Vision, Text, and Layout for Universal Document Processing, 2022. URL [https://arxiv.org/abs/2212.02623](https://arxiv.org/abs/2212.02623) . Version Number: 3.

- Thakur et al. (2021) Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models, 2021. URL [https://arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663) . Version Number: 4.

- Thapliyal et al. (2022) Ashish V. Thapliyal, Jordi Pont-Tuset, Xi Chen, and Radu Soricut. Crossmodal-3600: A Massively Multilingual Multimodal Evaluation Dataset, 2022. URL [https://arxiv.org/abs/2205.12522](https://arxiv.org/abs/2205.12522) . Version Number: 2.

- Wang et al. (2022) Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. Text Embeddings by Weakly-Supervised Contrastive Pre-training, 2022. URL [https://arxiv.org/abs/2212.03533](https://arxiv.org/abs/2212.03533) . Version Number: 2.

- Wang et al. (2024a) Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Improving text embeddings with large language models, 2024a. URL [https://arxiv.org/abs/2401.00368](https://arxiv.org/abs/2401.00368) .

- Wang et al. (2024b) Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution, 2024b. URL [https://arxiv.org/abs/2409.12191](https://arxiv.org/abs/2409.12191) .

- Wang et al. (2020) Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers, April 2020. URL [http://arxiv.org/abs/2002.10957](http://arxiv.org/abs/2002.10957) . arXiv:2002.10957 [cs].

- Yao et al. (2021) Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. FILIP: Fine-grained Interactive Language-Image Pre-Training, 2021. URL [https://arxiv.org/abs/2111.07783](https://arxiv.org/abs/2111.07783) . Version Number: 1.

- Yue et al. (2023) Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI, 2023. URL [https://arxiv.org/abs/2311.16502](https://arxiv.org/abs/2311.16502) . Version Number: 3.

- Zhai et al. (2023) Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid Loss for Language Image Pre-Training. 2023. doi: 10.48550/ARXIV.2303.15343 . URL [https://arxiv.org/abs/2303.15343](https://arxiv.org/abs/2303.15343) . Publisher: [object Object] Version Number: 4.

- Zhang et al. (2019) Li Zhang, Shuo Zhang, and Krisztian Balog. Table2vec: Neural word and entity embeddings for table population and retrieval. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval , SIGIR ’19. ACM, July 2019. doi: 10.1145/3331184.3331333 . URL [http://dx.doi.org/10.1145/3331184.3331333](http://dx.doi.org/10.1145/3331184.3331333) .

- Zhao et al. (2023) Ruochen Zhao, Hailin Chen, Weishi Wang, Fangkai Jiao, Xuan Long Do, Chengwei Qin, Bosheng Ding, Xiaobao Guo, Minzhi Li, Xingxuan Li, and Shafiq Joty. Retrieving Multimodal Information for Augmented Generation: A Survey, 2023. URL [https://arxiv.org/abs/2303.10868](https://arxiv.org/abs/2303.10868) . Version Number: 3.

- Zhu et al. (2022) Fengbin Zhu, Wenqiang Lei, Fuli Feng, Chao Wang, Haozhou Zhang, and Tat-Seng Chua. Towards Complex Document Understanding By Discrete Reasoning. 2022. doi: 10.48550/ARXIV.2207.11871 . URL [https://arxiv.org/abs/2207.11871](https://arxiv.org/abs/2207.11871) . Publisher: arXiv Version Number: 3.

## Benchmark Datasets / 基准数据集

### Academic Datasets / 学术数据集

**EN.** DocVQA ( Mathew et al., 2020 ) includes collected images from the UCSF Industry Documents Library. Questions and answers were manually annotated.

**中.** DocVQA（Mathew 等人，2020）包括从 UCSF 行业文档库收集的图像。问题和答案都是手动注释的。

**EN.** InfoVQA ( Mathew et al., 2021 ) includes infographics collected from the Internet using the search query “ infographics ”. Questions and answers were manually annotated.

**中.** InfoVQA（Mathew 等人，2021）包括使用搜索查询“infographics”从互联网收集的信息图表。问题和答案都是手动注释的。

**EN.** TAT-DQA ( Zhu et al., 2022 ) is a large-scale Document VQA dataset that was constructed from publicly available real-world financial reports. It focuses on rich tabular and textual content requiring numerical reasoning. Questions and answers were manually annotated by human experts in finance.

**中.** TAT-DQA（Zhu et al., 2022）是一个大规模文档 VQA 数据集，根据公开的真实世界财务报告构建。它侧重于需要数字推理的丰富表格和文本内容。问题和答案由人类金融专家手动注释。

**EN.** arXivQA ( Li et al., 2024 ) is a VQA dataset based on figures extracted from arXiv publications. The questions were generated synthetically using GPT-4 Vision.

**中.** arXivQA（Li et al., 2024）是基于从 arXiv 出版物中提取的数据的 VQA 数据集。这些问题是使用 GPT-4 Vision 综合生成的。

**EN.** TabFQuAD (Table French Question Answering Dataset) is designed to evaluate TableQA models in realistic industry settings. We create additional queries to augment the existing human-annotated ones using the same method described in subsection A.2 .

**中.** TabFQuAD（Table French Question Answering Dataset）旨在评估现实行业环境中的 TableQA 模型。我们使用 A.2 小节中描述的相同方法创建额外的查询来增强现有的人工注释查询。

### Practical Datasets / 实用数据集

**EN.** Methodology. Creating a relevant retrieval dataset close to real use cases is a major challenge as the dataset needs to be both sufficiently large for effective fine-tuning and sufficiently diverse to cover a broad range of modalities (full text, tables, charts, …), domains (industry, healthcare, …), and query-document interactions (extractive questions, open-ended questions, …). Our approach to building this dataset involves several steps: (1) we use a web crawler to collect publicly available documents on various themes and sources, (2) we convert these PDFs into a series of images, one per page, and (3) we generate queries related to each image using a VLM.

**中.** 方法论。创建接近真实用例的相关检索数据集是一项重大挑战，因为数据集需要足够大以进行有效的微调，并且足够多样化以涵盖广泛的模式（全文、表格、图表等）、领域（行业、医疗保健等）和查询文档交互（提取问题、开放式问题等）。我们构建此数据集的方法涉及几个步骤：(1) 我们使用网络爬虫收集有关各种主题和来源的公开文档，(2) 我们将这些 PDF 转换为一系列图像，每页一个，以及 (3) 我们使用 VLM 生成与每个图像相关的查询。

**EN.** Web-Crawler. We implemented a web crawler to efficiently collect large volumes of documents related to a given topic. The crawler is seeded with a user-defined query (e.g. ‘‘artificial intelligence”) and then uses GPT-3.5 Turbo to brainstorm related topics and subtopics. This query augmentation strategy aims at both broadening and deepening the search. GPT-3.5 Turbo is further used to generate diverse search queries from each subtopic. This query set is then consumed by a pool of parallel workers whose job is to fetch the associated most relevant documents. We use SerpAPI [^footnote17] along with a filetype filter (PDF documents only) to programmatically scrape Google Search rankings. Each file is hashed and stored in a Bloom filter ( Bloom, 1970 ) shared among workers to avoid duplicate documents in the final corpus. Unique scraped files are downloaded, and inserted into a SQLite database along with additional metadata.

**中.** 网络爬虫。我们实现了一个网络爬虫来有效地收集与给定主题相关的大量文档。爬虫以用户定义的查询（例如“人工智能”）为种子，然后使用 GPT-3.5 Turbo 集思广益相关主题和子主题。这种查询增强策略旨在扩大和深化搜索。GPT-3.5 Turbo 进一步用于从每个子主题生成不同的搜索查询。然后，该查询集由一组并行工作人员消耗，这些工作人员的工作是获取相关的最相关文档。我们使用 SerpAPI [^footnote17] 以及文件类型过滤器（仅限 PDF 文档）以编程方式抓取 Google 搜索排名。每个文件都经过哈希处理并存储在工作人员之间共享的 Bloom 过滤器（Bloom，1970）中，以避免最终语料库中出现重复文档。下载独特的抓取文件并将其与其他元数据一起插入到 SQLite 数据库中。

**EN.** Datamix. Using the web crawler, we collected approximately 100 documents for each of the following four seeds: “energy” , “government reports” , “healthcare industry” , and “artificial intelligence” . These seeds were meticulously hand-picked to align with real-use cases for retrieval models and visually rich pages. We also removed all documents containing any private information.

**中.** 数据混合。使用网络爬虫，我们为以下四个种子收集了大约 100 个文档：“能源”、“政府报告”、“医疗保健行业”和“人工智能”。这些种子经过精心挑选，与检索模型和视觉丰富页面的实际用例保持一致。我们还删除了包含任何私人信息的所有文档。

**EN.** Query Generation. To increase the efficiency of our query generation scheme and to limit API calls, we generate at most 3 questions per image. From all the documents collected, we randomly sample 10,000 images per theme and call Claude-3 Sonnet with the following prompt:

**中.** 查询生成。为了提高查询生成方案的效率并限制 API 调用，我们每张图像最多生成 3 个问题。从收集的所有文档中，我们为每个主题随机抽取 10,000 张图像，并使用以下提示调用 Claude-3 Sonnet：

**EN.** `You are an assistant specialized in Multimodal RAG tasks.`

**中.** `You are an assistant specialized in Multimodal RAG tasks.`

**EN.** ``

**中.** ``

**EN.** `The task is the following: given an image from a pdf page, you will have to`

**中.** `The task is the following: given an image from a pdf page, you will have to`

**EN.** `generate questions that can be asked by a user to retrieve information from`

**中.** `generate questions that can be asked by a user to retrieve information from`

**EN.** `a large documentary corpus.`

**中.** `a large documentary corpus.`

**EN.** `The question should be relevant to the page, and should not be too specific`

**中.** `The question should be relevant to the page, and should not be too specific`

**EN.** `or too general. The question should be about the subject of the page, and`

**中.** `or too general. The question should be about the subject of the page, and`

**EN.** `the answer needs to be found in the page.`

**中.** `the answer needs to be found in the page.`

**EN.** ``

**中.** ``

**EN.** `Remember that the question is asked by a user to get some information from a`

**中.** `Remember that the question is asked by a user to get some information from a`

**EN.** `large documentary corpus that contains multimodal data. Generate a question`

**中.** `large documentary corpus that contains multimodal data. Generate a question`

**EN.** `that could be asked by a user without knowing the existence and the content`

**中.** `that could be asked by a user without knowing the existence and the content`

**EN.** `of the corpus.`

**中.** `of the corpus.`

**EN.** `Generate as well the answer to the question, which should be found in the`

**中.** `Generate as well the answer to the question, which should be found in the`

**EN.** `page. And the format of the answer should be a list of words answering the`

**中.** `page. And the format of the answer should be a list of words answering the`

**EN.** `question.`

**中.** `question.`

**EN.** `Generate at most THREE pairs of questions and answers per page in a dictionary`

**中.** `Generate at most THREE pairs of questions and answers per page in a dictionary`

**EN.** `with the following format, answer ONLY this dictionary NOTHING ELSE:`

**中.** `with the following format, answer ONLY this dictionary NOTHING ELSE:`

**EN.** ``

**中.** ``

**EN.** `{`

**中.** `{`

**EN.** `"questions": [`

**中.** `"questions": [`

**EN.** `{`

**中.** `{`

**EN.** `"question": "XXXXXX",`

**中.** `"question": "XXXXXX",`

**EN.** `"answer": ["YYYYYY"]`

**中.** `"answer": ["YYYYYY"]`

**EN.** `},`

**中.** `},`

**EN.** `{`

**中.** `{`

**EN.** `"question": "XXXXXX",`

**中.** `"question": "XXXXXX",`

**EN.** `"answer": ["YYYYYY"]`

**中.** `"answer": ["YYYYYY"]`

**EN.** `},`

**中.** `},`

**EN.** `{`

**中.** `{`

**EN.** `"question": "XXXXXX",`

**中.** `"question": "XXXXXX",`

**EN.** `"answer": ["YYYYYY"]`

**中.** `"answer": ["YYYYYY"]`

**EN.** `},`

**中.** `},`

**EN.** `]`

**中.** `]`

**EN.** `}`

**中.** `}`

**EN.** `where XXXXXX is the question and [’YYYYYY’] is the corresponding list of answers`

**中.** `where XXXXXX is the question and [’YYYYYY’] is the corresponding list of answers`

**EN.** `that could be as long as needed.`

**中.** `that could be as long as needed.`

**EN.** ``

**中.** ``

**EN.** `Note: If there are no questions to ask about the page, return an empty list.`

**中.** `Note: If there are no questions to ask about the page, return an empty list.`

**EN.** `Focus on making relevant questions concerning the page.`

**中.** `Focus on making relevant questions concerning the page.`

**EN.** `Here is the page:`

**中.** `Here is the page:`

**EN.** ``

**中.** ``

**EN.** Human Validation. We manually validate every single synthetically created query in ViDoRe to ensure quality, query relevance, and consistency with the benchmark objective of evaluating retrieval in practical industrial settings. During this step, we randomly assign document-pair queries to 4 volunteer annotators and instruct them to filter out queries that do not fit the above-listed criteria. We also instruct annotators to flag any documents they deem to contain PII information or content not suited for an academic benchmark. No flag was raised during the entirety of the process, validating our prior PDF collection strategy. 100 queries per topic are collected in this manner. Annotators are colleagues and collaborators of the authors who volunteered to help. Each annotator spent approximately 3 hours filtering the larger query set down to 100 high-quality queries per topic.

**中.** 人类验证。我们手动验证 ViDoRe 中的每个综合创建的查询，以确保质量、查询相关性以及与实际工业环境中评估检索的基准目标的一致性。在此步骤中，我们将文档对查询随机分配给 4 名志愿者注释者，并指示他们过滤掉不符合上述条件的查询。我们还指示注释者标记他们认为包含 PII 信息或不适合学术基准的内容的任何文档。整个过程中没有出现任何异常情况，验证了我们之前的 PDF 收集策略。以这种方式收集每个主题 100 个查询。注释者是自愿提供帮助的作者的同事和合作者。每个注释者花费大约 3 小时将较大的查询集过滤为每个主题 100 个高质量查询。

### Training Dataset / 训练数据集

**EN.** The statistics of the train set are given in the following table. The creation of the train set follows the same methodology as in subsection A.2 . We made sure that a PDF document cannot have pages in both the training set and the test set to prevent data leakage and that there are no duplicate documents in each split.

**中.** 训练集的统计数据如下表所示。训练集的创建遵循与 A.2 小节中相同的方法。我们确保 PDF 文档不能同时包含训练集和测试集的页面，以防止数据泄漏，并且每个拆分中都没有重复的文档。

| Dataset Split<br>数据集分割 | Split Size<br>分体尺寸 | Language<br>语言 | Domain<br>域名 |
| --- | --- | --- | --- |
| DocVQA<br>文档VQA | 39,463 | English<br>英语 | Scanned documents from UCSF Industry<br>UCSF Industry 的扫描文件 |
| InfoVQA<br>信息质量保证 | 10,074 | English<br>英语 | Infographics scrapped from the web<br>从网络上删除的信息图表 |
| TATDQA<br>TATD质量保证 | 13,251 | English<br>英语 | High-quality financial reports<br>高质量的财务报告 |
| arXivQA | 10,000 | English<br>英语 | Scientific Scientific Figures from arXiv<br>来自 arXiv 的科学数据 |
| Scrapped PDFs<br>废弃的 PDF | 45,940 | English<br>英语 | Varied PDFs from 3885 distinct URL domains<br>来自 3885 个不同 URL 域的各种 PDF |
| TOTAL<br>总计 | 118,695 | English-only<br>仅英语 | Mixed<br>混合 |

**EN.** Table 3: Details on the different splits in the dataset used to train ColPali.

**中.** 表 3：用于训练 ColPali 的数据集中不同分割的详细信息。

## Implementation details / 实施细节

### Codebase / 代码库

**EN.** The codebase is written in PyTorch [^footnote18] and leverages HuggingFace tooling for model implementations and trainers [^footnote19] .

**中.** 该代码库是用 PyTorch [^footnote18] 编写的，并利用 HuggingFace 工具进行模型实现和训练器 [^footnote19] 。

### Hyperparameters / 超参数

**EN.** Hyperparameters are tuned on a validation split composed of $2\%$ of the training dataset. We find bi-encoder methods to be more sensible to learning rate variations than late interaction-based models and achieve the best performance for all models with a learning rate of $5e-5$ . We experiment with LoRA rank and $\alpha$ values and do not notice particular improvements past $r=\alpha=32$ . Per-device batch sizes are kept small due to long sequence lengths that complicate scaling past $b=4$ . We simulate larger batch sizes with multi-GPU training and train with a total batch size $b=32$ with no accumulation, for 1 epoch on our training set.

**中.** 超参数根据训练数据集 $2\%$ 组成的验证分割进行调整。我们发现双编码器方法比后期基于交互的模型对学习率变化更敏感，并且学习率为 $5e-5$ 的所有模型都实现了最佳性能。我们对 LoRA 等级和 $\alpha$ 值进行实验，没有注意到 $r=\alpha=32$ 相比的特殊改进。由于序列长度较长，使超过 $b=4$ 的扩展变得复杂，因此每个设备的批量大小保持较小。我们通过多 GPU 训练模拟更大的批量大小，并在训练集上使用总批量大小 $b=32$ 进行训练，无累积，持续 1 个周期。

### Embedding size / 嵌入尺寸

**EN.** Minimizing storage footprint can be essential to industrial retrieval systems if databases contain millions of documents. With this criterion in view, we have compared the embedding sizes of the models in our study. As shown in Table 4 , ColPali ’s embedding size is an order of magnitude larger than BM25 and two orders of magnitude larger than BGE-M3. However, in practical scenarios, pooling multi-vector embeddings by centroid cluster, or quantizing embeddings to binary representations [^footnote20] can reduce storage costs by two orders of magnitude ( Santhanam et al., 2022 ) with minimal performance hits, and make storage costs competitive with other systems.

**中.** 如果数据库包含数百万个文档，则最大限度地减少存储占用对于工业检索系统至关重要。考虑到这一标准，我们比较了研究中模型的嵌入大小。如表4所示，ColPali的嵌入大小比BM25大一个数量级，比BGE-M3大两个数量级。然而，在实际场景中，通过质心簇池化多向量嵌入，或将嵌入量化为二进制表示 [^footnote20] 可以将存储成本降低两个数量级（Santhanam et al., 2022），同时性能损失最小，并使存储成本与其他系统相比具有竞争力。

| Model<br>型号 | Embedding size (KB)<br>嵌入大小 (KB) |
| --- | --- |
| BGE-M3 | 8.60 |
| BM25 (dense emb.)<br>BM25（密嵌） | 3.00 |
| BM25 (sparse emb.)<br>BM25（稀疏嵌入） | 1.56 ± 0.51 |
| ColPali (float16)<br>科尔巴利 (float16) | 257.5 |

**EN.** Table 4: Comparison of the embedding sizes for the DocVQA test set from ViDoRe w.r.t. different retrieval models. The mean ± std size is given for the sparse embeddings. In general multiple vectors (2-5) per page are used for BGE-M3 and BM25.

**中.** 表 4：ViDoRe 中 DocVQA 测试集的嵌入大小比较不同的检索模型。稀疏嵌入的大小为平均值±标准差。一般来说，BGE-M3 和 BM25 每页使用多个向量 (2-5)。

### Latency computations / 延迟计算

**EN.** To ensure comparison fairness, the latencies of the different retrieval systems shown in Figure 2 are measured on the same g2-standard-8 GCP VM with a NVIDIA L4 GPU. Document pages are embedded using the highest settings of Unstructured with captioning (see subsection 3.2 ). SigLIP and ColPali are both loaded with bfloat16 parameter dtypes. The reported times in Table 5 are the average per-page latencies for each indexing operation on 1000 randomly chosen documents across all splits of the ViDoRe benchmark test set. A batch size of $8$ was used for the BGE-M3 model used with Unstructured , and a batch size of $4$ was used for SigLIP and ColPali .

**中.** 为了确保比较的公平性，图 2 中所示的不同检索系统的延迟是在具有 NVIDIA L4 GPU 的同一 g2-standard-8 GCP VM 上测量的。文档页面使用带标题的非结构化最高设置进行嵌入（请参阅第 3.2 小节）。 SigLIP 和 ColPali 都加载了 bfloat16 参数数据类型。表 5 中报告的时间是 ViDoRe 基准测试集所有分区中随机选择的 1000 个文档上每个索引操作的平均每页延迟。 $8$ 的批量大小用于与 Unstructed 一起使用的 BGE-M3 模型，而 $4$ 的批量大小用于 SigLIP 和 ColPali 。

| Indexing operation<br>索引操作 | Latency (s)<br>延迟时间 | Latency (s)<br>延迟时间 | Latency (s)<br>延迟时间 |
| --- | --- | --- | --- |
| Indexing operation<br>索引操作 | Unstructured<br>非结构化 | SigLIP<br>西格利普 | ColPali<br>巴利山口 |
| Layout detection<br>布局检测 | 0.81 | NA<br>不适用 | NA<br>不适用 |
| OCR<br>光学字符识别 | 2.67 | NA<br>不适用 | NA<br>不适用 |
| Captioning<br>字幕 | 3.71 | NA<br>不适用 | NA<br>不适用 |
| Page encoding<br>页面编码 | 0.03 | 0.12 | 0.39 |
| Total<br>总计 | 7.22 | 0.12 | 0.39 |

**EN.** Table 5: Page-level latencies for document indexing using various retrieval systems. SigLIP and ColPali are much faster than Unstructured because they don’t require the layout detection, OCR, and captioning operations.

**中.** 表 5：使用各种检索系统进行文档索引的页面级延迟。 SigLIP 和 ColPali 比 Unstructed 快得多，因为它们不需要布局检测、OCR 和字幕操作。

### Captioning / 字幕

**EN.** Examples of captions generated for visually rich document chunks with Claude-3 Sonnet are shown in Figure 5 and Figure 4 . The prompt used for generating the description is the following:

**中.** 图 5 和图 4 显示了使用 Claude-3 Sonnet 为视觉丰富的文档块生成的标题示例。用于生成描述的提示如下：

**EN.** You are an assistant specialized in document analysis. Given a table or a figure, you have to provide a detailed summary of the content in maximum 3000 characters. Your summary should be qualitative and not quantitative. Here is the table/figure to analyze: {image}. Answer ONLY with the caption of the table/figure.

**中.** 你是一名专门从事文档分析的助理。给定表格或图形，您必须提供最多 3000 个字符的详细内容摘要。你的总结应该是定性的而不是定量的。这是要分析的表格/图：{image}。仅回答表格/图的标题。

![Figure A2.F4](figs/A2-F4-01.webp)

**EN.** Figure 4: Example from the “Energy” test set. Caption: The image depicts the hourly energy generation profile, illustrating the contributions of various energy sources over 24 hours. The data is presented as a stacked bar chart, with the x-axis representing the hours of the day from 1 to 2, and the y-axis showing the average hourly generation in MW. The bars are segmented into different colors, each representing a distinct energy source: nuclear, bio, geothermal, solar, wind, hydro, natural gas, and other imports. The chart provides insights into the temporal variations in energy generation across different sources, highlighting the interplay between baseload and intermittent sources throughout the day.

**中.** 图 4：“能量”测试集的示例。图片说明：该图描绘了每小时的发电概况，说明了各种能源在 24 小时内的贡献。数据以堆叠条形图的形式呈现，x 轴代表一天中从 1 到 2 的小时，y 轴显示平均每小时发电量（以 MW 为单位）。这些条形图被分成不同的颜色，每种颜色代表一种不同的能源：核能、生物能源、地热能、太阳能、风能、水力、天然气和其他进口能源。该图表提供了对不同来源的能源产生的时间变化的深入了解，突出显示了全天基本负荷和间歇性来源之间的相互作用。

![Figure A2.F5](figs/A2-F5-01.webp)

**EN.** Figure 5: Example from the “Government Reports” test set. Caption: The image shows a table titled “System of Record” which outlines the different types of documents or records maintained across various systems or departments within an organization related to project management and construction. The rows list documents like project plans, budgets, schedules, contracts, purchase orders, invoices, change requests, bid submissions, drawings, manuals, meeting minutes, and reports. The columns indicate the system or department responsible for maintaining each record, such as County Servers, Project View, OnBase, CGI Advantage Financial System, and Purchasing Department. The table uses ”W” and ”T” markers to denote which system or department serves as the primary source (writer) or storage location (trailer) for each type of document.

**中.** 图 5：“政府报告”测试集的示例。图片说明：图像显示了一个标题为“记录系统”的表格，其中概述了组织内与项目管理和施工相关的各个系统或部门维护的不同类型的文档或记录。这些行列出了项目计划、预算、进度表、合同、采购订单、发票、变更请求、投标提交、图纸、手册、会议纪要和报告等文档。这些列表示负责维护每条记录的系统或部门，例如县服务器、项目视图、OnBase、CGI Advantage 财务系统和采购部门。该表使用“W”和“T”标记来表示哪个系统或部门充当每种类型文档的主要来源（作者）或存储位置（拖尾）。

## Additional results / 附加结果

### Other Metrics / 其他指标

|  | ArxivQ | DocQ<br>文档Q | InfoQ<br>信息Q | TabF<br>标签F | TATQ | Shift<br>转变 | AI<br>人工智能 | Energy<br>能源 | Gov.<br>政府。 | Health.<br>健康。 | Avg.<br>平均。 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unstructured text-only<br>仅非结构化文本 |  |  |  |  |  |  |  |  |  |  |  |
| BM25 | - | 26.6 | - | - | 34.6 | 45.0 | 86.0 | 70.0 | 68.0 | 74.0 | - |
| BGE-M3 | - | 22.8 $\downarrow$ 3.8 | - | - | 26.1 $\downarrow$ 8.5 | 51.0 $\uparrow$ 6.0 | 81.0 $\downarrow$ 5.0 | 72.0 $\uparrow$ 2.0 | 67.0 $\downarrow$ 1.0 | 77.0 $\uparrow$ 3.0 | - |
| Unstructured + OCR<br>非结构化+OCR |  |  |  |  |  |  |  |  |  |  |  |
| BM25 | 26.7 | 28.9 | 54.0 | 30.4 | 50.0 | 52.0 | 86.0 | 77.0 | 74.0 | 80.0 | 55.9 |
| BGE-M3 | 28.1 $\uparrow$ 1.4 | 22.9 $\downarrow$ 6.0 | 53.8 $\downarrow$ 0.2 | 55.7 $\uparrow$ 25.3 | 38.6 $\downarrow$ 11.4 | 56.0 $\uparrow$ 4.0 | 82.0 $\downarrow$ 4.0 | 79.0 $\uparrow$ 2.0 | 76.0 $\uparrow$ 2.0 | 83.0 $\uparrow$ 3.0 | 57.5 $\uparrow$ 1.6 |
| Unstructured + Captioning<br>非结构化+字幕 |  |  |  |  |  |  |  |  |  |  |  |
| BM25 | 35.5 | 30.2 | 61.5 | 24.3 | 49.0 | 47.0 | 79.0 | 76.0 | 75.0 | 81.0 | 55.9 |
| BGE-M3 | 29.3 $\downarrow$ 6.2 | 26.0 $\downarrow$ 4.2 | 62.1 $\uparrow$ 0.6 | 58.6 $\uparrow$ 34.3 | 30.6 $\downarrow$ 18.4 | 55.0 $\uparrow$ 8.0 | 80.0 $\uparrow$ 1.0 | 78.0 $\uparrow$ 2.0 | 69.0 $\downarrow$ 6.0 | 83.0 $\uparrow$ 2.0 | 57.2 $\uparrow$ 1.3 |
| Contrastive VLMs<br>对比 VLM |  |  |  |  |  |  |  |  |  |  |  |
| Jina-CLIP<br>吉纳-CLIP | 19.4 | 7.3 | 26.7 | 12.5 | 1.6 | 2.0 | 11.0 | 13.0 | 15.0 | 17.0 | 12.6 |
| Nomic-vision<br>规则愿景 | 10.4 | 6.7 | 22.1 | 9.6 | 1.6 | 0.0 | 9.0 | 9.0 | 7.0 | 13.0 | 8.8 |
| SigLIP (Vanilla)<br>SigLIP（原版） | 34.2 | 21.3 | 51.8 | 46.1 | 17.9 | 13.0 | 50.0 | 51.0 | 47.0 | 65.0 | 39.7 |
| Ours<br>我们的 |  |  |  |  |  |  |  |  |  |  |  |
| SigLIP (Vanilla)<br>SigLIP（原版） | 34.2 | 21.3 | 51.8 | 46.1 | 17.9 | 13.0 | 50.0 | 51.0 | 47.0 | 65.0 | 39.7 |
| BiSigLIP (+fine-tuning)<br>BiSigLIP（+微调） | 49.2 $\uparrow$ 15.0 | 23.8 $\uparrow$ 2.5 | 59.0 $\uparrow$ 7.2 | 52.1 $\uparrow$ 6.0 | 20.7 $\uparrow$ 2.8 | 16.0 $\uparrow$ 3.0 | 62.0 $\uparrow$ 12.0 | 61.0 $\uparrow$ 10.0 | 55.0 $\uparrow$ 8.0 | 72.0 $\uparrow$ 7.0 | 47.1 $\uparrow$ 7.4 |
| BiPali (+LLM)<br>比巴利语 (+法学硕士) | 46.4 $\downarrow$ -2.8 | 20.0 $\downarrow$ -3.8 | 54.6 $\downarrow$ -4.4 | 63.2 $\uparrow$ 11.1 | 20.4 $\downarrow$ -0.4 | 34.0 $\uparrow$ 18.0 | 59.0 $\downarrow$ -3.0 | 45.0 $\downarrow$ -16.0 | 57.0 $\uparrow$ 2.0 | 56.0 $\downarrow$ -16.0 | 45.6 $\downarrow$ -1.5 |
| ColPali (+Late Inter.)<br>ColPali（+国际米兰后期） | 72.4 $\uparrow$ 26.0 | 45.6 $\uparrow$ 25.6 | 74.6 $\uparrow$ 20.0 | 75.4 $\uparrow$ 12.1 | 53.1 $\uparrow$ 32.7 | 55.0 $\uparrow$ 21.0 | 93.0 $\uparrow$ 34.0 | 85.0 $\uparrow$ 40.0 | 85.0 $\uparrow$ 28.0 | 88.0 $\uparrow$ 32.0 | 72.7 $\uparrow$ 27.1 |

**EN.** Table 6: Comprehensive evaluation of baseline models and our proposed method on ViDoRe . Results are presented using Recall@1 metrics. Text-only metrics are not computed for benchmarks with only visual elements.

**中.** 表 6：基线模型和我们在 ViDoRe 上提出的方法的综合评估。结果使用 Recall@1 指标呈现。对于仅包含视觉元素的基准，不会计算纯文本指标。

### Model Variants / 型号变体

|  | ArxivQ | DocQ<br>文档Q | InfoQ<br>信息Q | TabF<br>标签F | TATQ | Shift<br>转变 | AI<br>人工智能 | Energy<br>能源 | Gov.<br>政府。 | Health.<br>健康。 | Avg.<br>平均。 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ColSigLIP (PaliGemma)<br>ColSigLIP（巴利语Gemma） | 3.1 | 3.0 | 5.1 | 6.2 | 2.5 | 1.0 | 3.4 | 3.4 | 2.3 | 2.2 | 3.2 |
| BiSigLIP (PaliGemma)<br>BiSigLIP（巴利语Gemma） | 18.5 | 14.6 | 33.4 | 39.5 | 16.1 | 5.2 | 27.6 | 32.6 | 36.6 | 35.7 | 26.0 |
| ColSigLIP (Original)<br>ColSigLIP（原始） | 2.6 | 2.2 | 2.3 | 5.7 | 1.8 | 1.0 | 2.6 | 4.1 | 1.4 | 1.5 | 2.5 |
| ColPali (No Q.A. Tokens)<br>ColPali（无问答代币） | 80.4 | 53.2 | 82.4 | 77.4 | 65.7 | 63.4 | 97.0 | 89.9 | 93.6 | 92.4 | 79.6 |
| ColPali (Docmatix) | 71.3 | 48.0 | 80.0 | 83.9 | 59.1 | 73.8 | 95.7 | 93.8 | 92.5 | 93.1 | 79.1 |
| ColPali (224)<br>科尔巴利 (224) | 71.0 | 37.4 | 62.3 | 65.7 | 28.6 | 20.4 | 65.7 | 66.8 | 73.9 | 73.0 | 56.5 |
| ColPali (Vision Trained)<br>ColPali（视觉训练） | 78.8 | 53.9 | 81.3 | 81.7 | 64.4 | 70.6 | 95.3 | 91.7 | 93.5 | 94.7 | 80.6 |
| ColPali (No Pairwise)<br>ColPali（无成对） | 79.0 | 53.0 | 82.1 | 85.3 | 63.2 | 66.2 | 94.9 | 88.9 | 92.7 | 92.1 | 79.7 |
| ColPali (+TabFQuAD training)<br>ColPali（+TabFQuAD 培训） | 77.6 | 54.7 | 82.6 | 86.5 | 65.4 | 73.9 | 94.8 | 92.4 | 94.2 | 94.8 | 81.7 |
| ColIdefics2 (64) | 73.6 | 48.0 | 82.4 | 81.6 | 63.0 | 57.2 | 95.5 | 86.9 | 86.6 | 91.2 | 76.6 |
| ColQwen2 (768) | 86.4 | 56.2 | 89.8 | 88.7 | 75.2 | 85.7 | 98.8 | 94.8 | 93.6 | 97.3 | 86.6 |
| ColPali (Reference: 448)<br>ColPali（参考：448） | 79.1 | 54.4 | 81.8 | 83.9 | 65.8 | 73.2 | 96.2 | 91.0 | 92.7 | 94.4 | 81.3 |

**EN.** Table 7: Benchmark scores for the “negative results” and various ablations on ViDoRe ; ColPali for reference. Results are presented using nDCG@5 metrics. Text-only metrics are not computed for benchmarks with only visual elements.

**中.** 表 7：ViDoRe 上“阴性结果”和各种消融的基准分数； ColPali供参考。结果使用 nDCG@5 指标呈现。对于仅包含视觉元素的基准，不会计算纯文本指标。

## More similarity maps / 更多相似度图

**EN.** In Figure 6 , ColPali assigns a high similarity to all patches with the word “Kazakhstan” when given the token <_Kazakhstan> . Moreover, our model seems to exhibit world knowledge capabilities as the patch around the word ”Kashagan” —an offshore oil field in Kazakhstan—also shows a high similarity score.

**中.** 在图 6 中，当给定标记 <_Kazakhstan> 时，ColPali 会为所有带有“Kazakhstan”一词的补丁分配高度相似性。此外，我们的模型似乎展示了世界知识能力，因为“Kashagan”（哈萨克斯坦的一个海上油田）一词周围的补丁也显示出很高的相似度。

![Figure A4.F6](figs/A4-F6-01.webp)

**EN.** Figure 6: Similarity of the image patches w.r.t. the underlined token in the user query. This example is from the Shift test set.

**中.** 图 6：图像块的相似度用户查询中带下划线的标记。此示例来自 Shift 测试集。

**EN.** It is also interesting to highlight that both this similarity map and the one displayed in Figure 3 (right) showcase a few white patches with high similarity scores. This behavior might first seem surprising as the white patches should not carry a meaningful signal from the original images. We believe the vectors associated with these patches share a similar role with the ViT registers ( Darcet et al., 2023 ) , i.e. these patches were repurposed for internal computations and stored the global information from the whole image.

**中.** 值得注意的是，该相似性图和图 3（右）中显示的相似性图都展示了一些具有高相似性分数的白色斑块。这种行为乍一看可能令人惊讶，因为白色斑块不应该携带来自原始图像的有意义的信号。我们认为与这些补丁相关的向量与 ViT 寄存器具有相似的作用（Darcet et al., 2023），即这些补丁被重新用于内部计算并存储整个图像的全局信息。

## Model glossary / 型号术语表

### SigLIP / 西格利普

**EN.** SigLIP (Sigmoid Loss for Language Image Pre-Training) builds upon CLIP (Contrastive Language-Image Pretraining)—a foundational model that aligns images and text by maximizing the similarity between correct image-text pairs while minimizing it for incorrect ones, leveraging a contrastive loss ( Zhai et al., 2023 ) . Unlike CLIP ( Radford et al., 2021 ) , which applies the softmax function to the logits, SigLIP uses the sigmoid activation function. This innovation eliminates the need for a global view of all pairwise similarities between images and texts within a batch, enabling more flexible batch size scaling (up to 1M items per batch, with an effective optimal batch size of 32k). This approach allows SigLIP to achieve state-of-the-art performance in zero-shot image classification tasks.

**中.** SigLIP（语言图像预训练的 Sigmoid 损失）建立在 CLIP（对比语言图像预训练）的基础上，CLIP 是一种基础模型，利用对比损失，通过最大化正确图像-文本对之间的相似性，同时最小化错误图像-文本对之间的相似性来对齐图像和文本（Zhai 等人，2023）。与将 softmax 函数应用于 logits 的 CLIP (Radford et al., 2021) 不同，SigLIP 使用 sigmoid 激活函数。这项创新消除了对批次内图像和文本之间所有成对相似性的全局视图的需要，从而实现更灵活的批次大小缩放（每批次最多 1M 个项目，有效最佳批次大小为 32k）。这种方法使 SigLIP 在零样本图像分类任务中实现了最先进的性能。

### PaliGemma / 巴利语杰玛

**EN.** PaliGemma is a 3B-parameter vision-language model. It integrates the SigLIP vision encoder with a Gemma-2B language decoder, connected via a multimodal linear projection layer ( Lucas Beyer* et al., 2024 ) . The model processes images by segmenting them into a fixed number of Vision Transformer ( Dosovitskiy et al., 2020 ) tokens, which are prepended to an optional text prompt.

**中.** PaliGemma 是一个 3B 参数视觉语言模型。它将 SigLIP 视觉编码器与 Gemma-2B 语言解码器集成，通过多模态线性投影层连接（Lucas Beyer* 等人，2024）。该模型通过将图像分割成固定数量的 Vision Transformer (Dosovitskiy et al., 2020) 标记来处理图像，这些标记被添加到可选的文本提示之前。

**EN.** A distinguishing feature of PaliGemma is its operation as a Prefix-Language Model (Prefix-LM). This design ensures full attention between image tokens and the user-provided input (prefix) while generating outputs auto-regressively (suffix). This architecture allows image tokens to access the task-specific query during processing, facilitating more effective task-dependent reasoning.

**中.** PaliGemma 的一个显着特征是它作为前缀语言模型（Prefix-LM）的操作。这种设计确保了图像标记和用户提供的输入（前缀）之间的充分关注，同时自动回归生成输出（后缀）。这种架构允许图像令牌在处理过程中访问特定于任务的查询，从而促进更有效的任务相关推理。

**EN.** PaliGemma was trained in four stages: unimodal pretraining with existing components, extended multimodal pretraining, short high-resolution pretraining, and task-specific fine-tuning.

**中.** PaliGemma 分四个阶段进行训练：使用现有组件的单模态预训练、扩展多模态预训练、简短的高分辨率预训练和针对特定任务的微调。

### ColBERT / 科尔伯特

**EN.** ColBERT (Contextualized Late Interaction over BERT) is a retrieval model designed to balance speed and effectiveness in information retrieval tasks ( Khattab & Zaharia, 2020 ) . Traditional retrieval models are typically categorized based on their type of interaction: either processing queries and documents independently for efficiency (bi-encoders) or jointly to capture rich contextual relationships (cross-encoders). ColBERT combines the advantages of both approaches through a novel late interaction mechanism.

**中.** ColBERT（Contextualized Late Interaction over BERT）是一种检索模型，旨在平衡信息检索任务的速度和有效性（Khattab & Zaharia，2020）。传统的检索模型通常根据交互类型进行分类：要么独立处理查询和文档以提高效率（双编码器），要么联合处理查询和文档以捕获丰富的上下文关系（跨编码器）。 ColBERT 通过一种新颖的后期交互机制结合了两种方法的优点。

**EN.** Queries and documents are encoded separately using BERT, enabling offline pre-computation of document representations for scalability. Instead of pooling embeddings into a single vector, ColBERT retains token-level embeddings and employs a MaxSim operator to compute fine-grained similarity scores. For each query token, the model determines the maximum similarity with document tokens, summing these scores to compute relevance.

**中.** 查询和文档使用 BERT 分别编码，从而实现文档表示的离线预计算以实现可扩展性。 ColBERT 没有将嵌入池化到单个向量中，而是保留了标记级嵌入，并采用 MaxSim 运算符来计算细粒度的相似度分数。对于每个查询标记，模型确定与文档标记的最大相似度，将这些分数相加以计算相关性。

**EN.** This architecture preserves the contextual richness of deep language models while significantly improving computational efficiency. By delaying the interaction step, ColBERT supports vector similarity indexing, facilitating end-to-end retrieval from large collections without prohibitive costs. Empirical evaluations on passage search datasets demonstrate that ColBERT achieves competitive effectiveness compared to existing BERT-based models ( Devlin et al., 2018 ) , while executing queries orders of magnitude faster and with drastically reduced computational requirements.

**中.** 这种架构保留了深度语言模型的上下文丰富性，同时显着提高了计算效率。通过延迟交互步骤，ColBERT 支持向量相似性索引，从而促进从大型集合中进行端到端检索，而无需高昂的成本。对段落搜索数据集的实证评估表明，与现有的基于 BERT 的模型相比，ColBERT 实现了竞争有效性（Devlin 等人，2018），同时执行查询的速度提高了几个数量级，并且计算要求大幅降低。

## Examples from the ViDoRe benchmark / ViDoRe 基准测试的示例

**EN.** Energy

**中.** 能源

![Figure A6.fig4](figs/A6-fig4-01.webp)

![Figure A6.fig4](figs/A6-fig4-02.webp)

![Figure A6.fig4](figs/A6-fig4-03.webp)

**EN.** Artificial Intelligence

**中.** 人工智能

![Figure A6.fig8](figs/A6-fig8-01.webp)

![Figure A6.fig8](figs/A6-fig8-02.webp)

![Figure A6.fig8](figs/A6-fig8-03.webp)

**EN.** Healthcare Industry

**中.** 医疗保健行业

![Figure A6.fig12](figs/A6-fig12-01.webp)

![Figure A6.fig12](figs/A6-fig12-02.webp)

![Figure A6.fig12](figs/A6-fig12-03.webp)

**EN.** Government Reports

**中.** 政府报告

![Figure A6.fig16](figs/A6-fig16-01.webp)

![Figure A6.fig16](figs/A6-fig16-02.webp)

![Figure A6.fig16](figs/A6-fig16-03.webp)

**EN.** Shift

**中.** 转变

![Figure A6.fig20](figs/A6-fig20-01.webp)

![Figure A6.fig20](figs/A6-fig20-02.webp)

![Figure A6.fig20](figs/A6-fig20-03.webp)

## Footnotes / 脚注

[^id2]: **EN.** Equal Contribution<br>**中.** 平等贡献

[^footnote1]: **EN.** The ViDoRe benchmark leaderboard is hosted publicly at [https://huggingface.co/spaces/vidore/vidore-leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard) to encourage further developments.<br>**中.** ViDoRe 基准排行榜在 [https://huggingface.co/spaces/vidore/vidore-leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard) 上公开托管，以鼓励进一步发展。

[^footnote2]: **EN.** Answers are generated alongside queries to (1) ground queries and improve their quality and (2) provide resources to foster future work.<br>**中.** 答案与查询一起生成，以 (1) 进行基础查询并提高其质量；(2) 提供资源以促进未来的工作。

[^footnote3]: **EN.** [https://huggingface.co/vidore](https://huggingface.co/vidore)<br>**中.** [https://huggingface.co/vidore](https://huggingface.co/vidore)

[^footnote4]: **EN.** [www.unstructured.io](https://www.unstructured.io)<br>**中.** [www.unstructured.io](https://www.unstructured.io)

[^footnote5]: **EN.** We empirically validated the max-pooling strategy over sub-page chunks to be more effective than concatenating all page chunks before embedding pagewise.<br>**中.** 我们凭经验验证了子页面块上的最大池策略比在逐页嵌入之前连接所有页面块更有效。

[^footnote6]: **EN.** We reformulate the loss to leverage the numerically stable softplus function where $\texttt{softplus(x)}=\log\left(1+\exp\left(x\right)\right)$<br>**中.** 我们重新表述损失以利用数值稳定的 softplus 函数，其中 $\texttt{softplus(x)}=\log\left(1+\exp\left(x\right)\right)$

[^footnote7]: **EN.** Multilingual data is present in the pretraining corpus of the language model (Gemma-2B) and potentially occurs during PaliGemma-3B’s multimodal training.<br>**中.** 多语言数据存在于语言模型 (Gemma-2B) 的预训练语料库中，并且可能在 PaliGemma-3B 的多模态训练期间出现。

[^footnote8]: **EN.** [https://huggingface.co/datasets/vidore/colpali_train_set](https://huggingface.co/datasets/vidore/colpali_train_set)<br>**中.** [https://huggingface.co/datasets/vidore/colpali_train_set](https://huggingface.co/datasets/vidore/colpali_train_set)

[^footnote9]: **EN.** [https://huggingface.co/google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)<br>**中.** [https://huggingface.co/google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)

[^footnote10]: **EN.** Note that the SigLIP model used in PaliGemma slightly differs in terms of number patches - 1024 patches for PaliGemma’s vision encoder, and 729 for the standalone SigLIP model.<br>**中.** 请注意，PaliGemma 中使用的 SigLIP 模型在补丁数量方面略有不同 - PaliGemma 视觉编码器有 1024 个补丁，独立 SigLIP 模型有 729 个补丁。

[^footnote11]: **EN.** This can be explained by the fact that contrary to SigLIP, the original PaliGemma is not trained on contrastive matching tasks, but rather on next token prediction. Our contrastive fine-tuning phase on 119K images to transform PaliGemma into a bi-encoder is 5 orders of magnitude smaller than SigLIP’s original contrastive training.<br>**中.** 这可以通过以下事实来解释：与 SigLIP 相反，原始的 PaliGemma 不是针对对比匹配任务进行训练，而是针对下一个标记预测进行训练。我们在 119K 图像上进行对比微调阶段，将 PaliGemma 转换为双编码器，比 SigLIP 原始对比训练小 5 个数量级。

[^footnote12]: **EN.** Computed for a batch size of $1$ (online), and averaged over 1000 queries. See subsection B.4<br>**中.** 针对 $1$ 的批量大小（在线）进行计算，并平均超过 1000 个查询。参见 B.4 小节

[^footnote13]: **EN.** Measures a NVIDIA L4 GPU, averaged on 100 pages, with a batch size of 4 pages for ColPali and 8 text chunks for Bi-Encoders. On average, a page is divided into 2.1 chunks. See subsection B.4 .<br>**中.** 测量 NVIDIA L4 GPU，平均为 100 页，ColPali 的批量大小为 4 页，双编码器的批量大小为 8 个文本块。平均一个页面被分为 2.1 个块。参见 B.4 小节。

[^footnote14]: **EN.** With the option of adding 4 sub-image crops of 64 tokens each to the sequence, for a total of 320 tokens<br>**中.** 可以选择将 4 个子图像裁剪（每个 64 个标记）添加到序列中，总共 320 个标记

[^footnote15]: **EN.** To train with data resembling the one BGE-M3 models would see at inference time would require running complex extraction pipelines for the more than 100K documents in the training set, notably relying on external proprietary captioning models which is both too costly and lengthy. This is not needed to train vision-based models.<br>**中.** 要使用类似于 BGE-M3 模型在推理时看到的数据进行训练，需要为训练集中超过 100K 的文档运行复杂的提取管道，特别是依赖外部专有字幕模型，这既昂贵又冗长。训练基于视觉的模型不需要这样做。

[^footnote16]: **EN.** [https://huggingface.co/spaces/vidore/vidore-leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard)<br>**中.** [https://huggingface.co/spaces/vidore/vidore-leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard)

[^footnote17]: **EN.** [https://serpapi.com/](https://serpapi.com/)<br>**中.** [https://serpapi.com/](https://serpapi.com/)

[^footnote18]: **EN.** [https://pytorch.org/](https://pytorch.org/)<br>**中.** [https://pytorch.org/](https://pytorch.org/)

[^footnote19]: **EN.** [https://huggingface.co](https://huggingface.co)<br>**中.** [https://huggingface.co](https://huggingface.co)

[^footnote20]: **EN.** [https://blog.vespa.ai/scaling-colpali-to-billions/](https://blog.vespa.ai/scaling-colpali-to-billions/)<br>**中.** [https://blog.vespa.ai/scaling-colpali-to-billions/](https://blog.vespa.ai/scaling-colpali-to-billions/)
