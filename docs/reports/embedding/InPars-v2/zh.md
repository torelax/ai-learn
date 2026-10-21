> 原文: [arXiv:2301.01820](https://arxiv.org/abs/2301.01820)（v4，2023-05-26）
> 说明：全文中英逐段对照，英文段落后紧接中文，保留原文完整结果表、脚注、致谢与参考文献。英文依 v4 原文转写，中文为辅助阅读译稿；原始数值与引用编号保留。

**预印本：** arXiv:2301.01820v4 [cs.IR]，2023 年 5 月 26 日。

**开源：** https://github.com/zetaalphavector/inPars/tree/master/legacy/inpars-v2

---

# InPars-v2: Large Language Models as Efficient Dataset Generators for Information Retrieval

# InPars-v2：大语言模型作为信息检索的高效数据集生成器

**Authors / 作者：** Vitor Jeronymo, Luiz Bonifacio, Hugo Abonizio, Marzieh Fadaee, Roberto Lotufo, Jakub Zavrel, Rodrigo Nogueira

**Affiliation / 单位：**

- Vitor Jeronymo、Luiz Bonifacio、Hugo Abonizio、Roberto Lotufo：NeuralMind, Brazil / 巴西 NeuralMind；FEEC-UNICAMP, Brazil / 巴西坎皮纳斯大学电气与计算机工程学院
- Marzieh Fadaee、Jakub Zavrel：Zeta Alpha, Netherlands / 荷兰 Zeta Alpha
- Rodrigo Nogueira：NeuralMind, Brazil；FEEC-UNICAMP, Brazil；Zeta Alpha, Netherlands

---

## Abstract / 摘要

**EN.** Recently, InPars introduced a method to efficiently use large language models (LLMs) in information retrieval tasks: via few-shot examples, an LLM is induced to generate relevant queries for documents. These synthetic query-document pairs can then be used to train a retriever. However, InPars and, more recently, Promptagator, rely on proprietary LLMs such as GPT-3 and FLAN to generate such datasets. In this work we introduce InPars-v2, a dataset generator that uses open-source LLMs and existing powerful rerankers to select synthetic query-document pairs for training. A simple BM25 retrieval pipeline followed by a monoT5 reranker finetuned on InPars-v2 data achieves new state-of-the-art results on the BEIR benchmark. To allow researchers to further improve our method, we open source the code, synthetic data, and finetuned models: https://github.com/zetaalphavector/inPars/tree/master/legacy/inpars-v2

**中.** 近日，InPars 提出一种在信息检索任务中高效使用大语言模型（large language model, LLM）的方法：通过少样本（few-shot）示例，诱导 LLM 为文档生成相关查询。这些合成的查询–文档对随后可以用来训练检索器。然而 InPars 以及更近的 Promptagator 依赖 GPT-3、FLAN 等专有 LLM 来生成这类数据集。本文提出 InPars-v2，一种数据集生成器：使用开源 LLM，并用已有的强重排器挑选用于训练的合成查询–文档对。一条简单的 BM25 检索流水线，再接在 InPars-v2 数据上微调的 monoT5 重排器，在 BEIR 基准上取得新的最优结果。为便于研究者继续改进本方法，开源代码、合成数据与微调模型：https://github.com/zetaalphavector/inPars/tree/master/legacy/inpars-v2

---

## Introduction and Background / 引言与背景（原文 §1）

**EN.** Data augmentation has been a reliable tool to improve the effectiveness of AI models in the face of the scarcity of high-quality in-domain training data, which is a common problem in practical applications. Previous work by Bonifacio et al. [1] and Dai et al. [2] successfully leveraged the few-shot capabilities of LLMs to generate reliable synthetic training data for information retrieval models. These training data helped their models achieve state-of-the-art (SOTA) results on the BEIR benchmark [6].

**中.** 面对高质量域内训练数据稀缺——这是实际应用中的常见问题——数据增强一直是提高 AI 模型效果的可靠工具。Bonifacio et al. [1] 与 Dai et al. [2] 的先前工作成功利用 LLM 的少样本能力，为信息检索模型生成可靠的合成训练数据。这些训练数据帮助他们的模型在 BEIR 基准 [6] 上达到当时最优（state-of-the-art, SOTA）。

**EN.** Bonifacio et al. [1] propose InPars where they generate queries from documents in the corpus using LLMs. Similarly to Bonifacio et al. [1], the recently published Promptagator [2] model also feeds prompts to LLMs in order to generate alternative queries for a given document in an unsupervised manner. It differs primarily from InPars in that it uses dataset-specific prompts, a larger LLM to generate queries, and a fully trainable retrieval pipeline with smaller models.

**中.** Bonifacio et al. [1] 提出 InPars：用 LLM 从语料中的文档生成查询。与 Bonifacio et al. [1] 类似，较近发表的 Promptagator [2] 也向 LLM 喂提示，以无监督方式为给定文档生成替代查询。它与 InPars 的主要差别在于：使用数据集专用提示、用更大的 LLM 生成查询，以及一条用更小模型、可全训练的检索流水线。

**EN.** This work extends the method of Bonifacio et al. [1] by using a reranker as a filtering mechanism to select the best synthetically generated examples and further improving retrieval effectiveness on BEIR. We also use an open-source query generator as opposed to the proprietary one used by Bonifacio et al. and provide the source code and data to reproduce our results on TPUs. We refer to Bonifacio et al. [1] model as Inpars-v1 and the model presented in this paper as Inpars-v2.

**中.** 本文扩展 Bonifacio et al. [1] 的方法：用重排器作为过滤机制，挑选最好的合成例，并进一步提高 BEIR 上的检索效果。相对 Bonifacio 等人使用的专有查询生成器，本文改用开源查询生成器，并提供可在 TPU 上复现结果的源代码与数据。本文将 Bonifacio et al. [1] 的模型称为 Inpars-v1，将本文提出的模型称为 Inpars-v2。

---

## Methodology / 方法（原文 §2）

**EN.** In this section, we explain the experiments we performed and how they differ from InPars-v1 [1].

**中.** 本节说明所做的实验，以及它们与 InPars-v1 [1] 的差别。

**EN.** To generate synthetic queries, we use the open-source GPT-J [8] with 6B parameters to replace OpenAI’s curie model used in InPars-v1. For each dataset in the BEIR benchmark, we sample 100k documents from its corpus and generate one synthetic query per document using GPT-J prompted with 3 examples from MS MARCO. We use greedy decoding and the “gbq” prompt template from InPars-v1. Some corpora in BEIR such as ArguAna [7] have less than 100k documents. In these cases, we generate as many synthetic queries as there are documents in the corpus. It takes on average 30 hours on an A100 GPU to generate 100k queries.

**中.** 为生成合成查询，用开源 GPT-J [8]（60 亿参数）替换 InPars-v1 所用的 OpenAI curie 模型。对 BEIR 基准中的每个数据集，从其语料采样 10 万篇文档，用带有 3 条 MS MARCO 示例的提示，让 GPT-J 为每篇文档生成一条合成查询。解码为贪心解码，提示模板采用 InPars-v1 的 “gbq”。BEIR 中有些语料不足 10 万篇，例如 ArguAna [7]。这些情况下，生成的合成查询数等于语料中的文档数。在一块 A100 GPU 上生成 10 万条查询平均需要 30 小时。

**EN.** Once the synthetic queries are generated, we apply a filtering step to select query-document pairs that are more likely to be relevant to each other. In InPars-v1, this filtering step consisted of selecting the top 10k query-document pairs with the highest log probabilities of generating a query given the 3-shot examples and the document as input. In InPars-v2, we use monoT5-3B [4] already finetuned on MS MARCO for one epoch to estimate a relevancy score for each of the 100k query-document pairs. Then, we keep only the top 10k pairs with the highest scores as our positive query-document pairs for training. It takes approximately 1.5 hours to score 100k query-document pairs on a TPU v3-8. It should take twice as much on a A100.

**中.** 合成查询生成之后，加一步过滤，选出更可能彼此相关的查询–文档对。在 InPars-v1 中，这一过滤是：在以 3-shot 示例加文档为输入的条件下，选取生成该查询的对数概率最高的 1 万对查询–文档。在 InPars-v2 中，用已经在 MS MARCO 上微调一个 epoch 的 monoT5-3B [4]，为这 10 万对查询–文档估计相关性分数。然后只保留分数最高的 1 万对，作为训练用的正查询–文档对。在 TPU v3-8 上为 10 万对打分大约需要 1.5 小时。在一块 A100 上应大约是两倍时间。脚注检查点：https://huggingface.co/castorini/monot5-3b-msmarco-10k

**EN.** To obtain negatives (i.e., non-relevant) query-document pairs, we randomly sample one document from the top 1000 retrieved by BM25 when issued the synthetic query. Thus, our training set consists of 10k positive query-document pairs and 10k negative query-document pairs.

**中.** 为得到负例（即不相关）查询–文档对，对合成查询用 BM25 检索，从其前 1000 篇中随机抽一篇文档。因此训练集由 1 万对正查询–文档和 1 万对负查询–文档组成。

**EN.** The rerankers are finetuned in the same manner as in InPars-v1: monoT5-3B is finetuned on MS MARCO for one epoch and then further finetuned for one epoch on the synthetic data. We use the Adafactor optimizer [5] with a constant learning rate of 1e-3. Each batch has 64 positive and 64 negative query-document pairs randomly sampled from the training dataset. We finetune one model on each synthetic dataset from BEIR, that is, we end up with 18 different rerankers, one per dataset, which are then evaluated on the corresponding test sets. Finetuning on each synthetic dataset takes less than 10 minutes on a TPU v3-8.

**中.** 重排器的微调方式与 InPars-v1 相同：monoT5-3B 先在 MS MARCO 上微调一个 epoch，再在合成数据上继续微调一个 epoch。优化器为 Adafactor [5]，学习率恒为 1e-3。每个 batch 有 64 对正查询–文档和 64 对负查询–文档，从训练数据集中随机抽取。在 BEIR 的每份合成数据集上各微调一个模型，因此得到 18 个不同的重排器，每个数据集一个，再在对应测试集上评测。在 TPU v3-8 上，每份合成数据集的微调不到 10 分钟。

**EN.** Evaluation is performed using the following pipeline: first we use Pyserini’s [3] flat indexes to retrieve a thousand documents for each query using BM25 with default parameters (k1=0.9, b=0.4), for each dataset. Then we use the finetuned monoT5-3B models to rerank these documents.

**中.** 评测使用如下流水线：先用 Pyserini [3] 的 flat 索引，对每个数据集用默认参数（k1=0.9，b=0.4）的 BM25 为每条查询检索一千篇文档。再用微调后的 monoT5-3B 模型重排这些文档。脚注：flat 索引，相对于 multifield 索引。

---

## Results / 结果（原文 §3）

**表 1 / Table 1.** nDCG@10 on BEIR. “Avg PrGator” is the average of datasets reported by Promptagator. / BEIR 上的 nDCG@10。“Avg PrGator” 是 Promptagator 所报告数据集上的平均。Promptagator uses a proprietary model, FLAN [9], to generate synthetic queries. The RankT5 model is a modified version of the monoT5 reranker, but its checkpoint and code are not published. In this work, we make the code, models, and data open-source and publicly available. / Promptagator 使用专有模型 FLAN [9] 生成合成查询。RankT5 是 monoT5 重排器的修改版，但其检查点与代码未发表。本文把代码、模型与数据开源并公开。

列依次为：BM25；monoT5-3B 仅 MS MARCO（MARCO）；monoT5-3B 再加 InPars-v1；monoT5-3B 再加 InPars-v2；Promptagator（PrGator）；RankT5。表中 “—” 为原文未报告。

| Dataset / 数据集 | BM25 | MARCO | +InPars-v1 | +InPars-v2 | PrGator | RankT5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TREC-Covid | 0.594 | 0.801 | 0.846 | 0.846 | 0.762 | 0.823 |
| Robust | 0.407 | 0.615 | 0.610 | 0.632 | — | — |
| FiQA | 0.236 | 0.509 | 0.492 | 0.509 | 0.494 | 0.493 |
| DBPedia | 0.318 | 0.472 | 0.494 | 0.498 | 0.434 | 0.459 |
| SciDocs | 0.149 | 0.197 | 0.206 | 0.208 | 0.201 | 0.191 |
| SciFact | 0.678 | 0.774 | 0.774 | 0.774 | 0.731 | 0.760 |
| NFCorpus | 0.321 | 0.383 | 0.385 | 0.385 | 0.370 | 0.399 |
| BioASQ | 0.522 | 0.566 | 0.607 | 0.595 | — | 0.579 |
| Natural Questions | 0.305 | 0.625 | 0.625 | 0.638 | — | 0.647 |
| HotpotQA | 0.633 | 0.760 | 0.790 | 0.791 | 0.736 | 0.753 |
| TREC-News | 0.395 | 0.477 | 0.458 | 0.490 | — | — |
| Quora | 0.788 | 0.835 | 0.874 | 0.845 | — | 0.819 |
| FEVER | 0.651 | 0.848 | 0.852 | 0.872 | 0.866 | 0.848 |
| Climate-FEVER | 0.165 | 0.288 | 0.287 | 0.323 | 0.241 | 0.275 |
| Signal | 0.328 | 0.302 | 0.319 | 0.308 | — | 0.319 |
| ArguAna | 0.397 | 0.379 | 0.371 | 0.369 | 0.630 | 0.406 |
| Touche | 0.442 | 0.309 | 0.260 | 0.291 | 0.381 | 0.486 |
| CQADupstack | 0.302 | 0.449 | 0.449 | 0.448 | — | — |
| Avg | 0.424 | 0.533 | 0.539 | 0.545 | — | — |
| Avg PrGator | 0.417 | 0.520 | 0.523 | 0.533 | 0.531 | 0.536 |

**EN.** Table 1 presents results for BM25 (2nd column), monoT5-3B finetuned on MS MARCO (3rd column), monoT5-3b finetuned on MS MARCO and further finetuned on InPars-v1 (4th column), and monoT5-3B finetuned on MS MARCO and then finetuned on InPars-v2 data (5th column). Compared to InPars-v1, our approach is substantially better on TREC-News, Climate-FEVER, Robust and Touche. Additionally, we compare our method with Promptagator [2] and RankT5 [10]. Taking into account the average of all BEIR datasets, these results represent a new state of the art on BEIR.

**中.** 表 1 给出 BM25（第 2 列）、在 MS MARCO 上微调的 monoT5-3B（第 3 列）、在 MS MARCO 上微调后再在 InPars-v1 上继续微调的 monoT5-3b（第 4 列），以及在 MS MARCO 上微调后再在 InPars-v2 数据上微调的 monoT5-3B（第 5 列）。相对 InPars-v1，本方法在 TREC-News、Climate-FEVER、Robust 和 Touche 上明显更好。此外，与 Promptagator [2] 和 RankT5 [10] 比较。就全部 BEIR 数据集的平均而言，这些结果代表 BEIR 上新的最优。

**EN.** Promptagator and RankT5 strive on datasets that monoT5 and InPars-v2 cannot even surpass BM25, such as Touche and ArguAna. Note that these datasets focus on argument retrieval, which is slightly different from other datasets in the BEIR benchmark. As a result, they benefit from using custom prompts. Promptagator does this without using supervised data from MS MARCO and using smaller T5 models with 110M parameters for the retrieval and reranking steps.

**中.** Promptagator 和 RankT5 在 monoT5 与 InPars-v2 甚至超不过 BM25 的数据集上表现强，例如 Touche 和 ArguAna。注意这些数据集聚焦论点检索，与 BEIR 基准中的其他数据集略有不同。因此它们受益于使用定制提示。Promptagator 做到这一点时不使用 MS MARCO 的监督数据，并且检索与重排步骤使用 1.1 亿参数的更小 T5 模型。

**EN.** Footnote. In preliminary experiments, we also observed an improvement of more than 10 nDCG@10 points on ArguAna by using a dataset-specific prompt to generate synthetic queries. More details and results on the full BEIR benchmark will appear in an upcoming paper.

**中.** 脚注。在初步实验中，用数据集专用提示生成合成查询，在 ArguAna 上也观察到超过 10 个 nDCG@10 点的提升。完整 BEIR 基准上的更多细节与结果将出现在后续论文中。

---

## Conclusion / 结论（原文 §4）

**EN.** In this work, we presented InPars-v2, an improved version of InPars [1] that uses a publicly available language model to generate queries and a better query-document pair selection process. Our results show that we achieve effectiveness on par with the state of the art on BEIR. The synthetic data and finetuned models were publicly released.

**中.** 本文提出 InPars-v2，它是 InPars [1] 的改进版：用公开可用的语言模型生成查询，并用更好的过程选择查询–文档对。结果表明，在 BEIR 上达到与当时最优相当的效果。合成数据与微调模型已公开发布。

---

## Acknowledgments / 致谢

**EN.** This research was partially supported by Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP) (project id 2022/01640-2). We also thank Centro Nacional de Processamento de Alto Desempenho (CENAPAD-SP) and Google Cloud for computing credits.

**中.** 本研究部分由圣保罗州研究基金会（Fundação de Amparo à Pesquisa do Estado de São Paulo, FAPESP）支持（项目号 2022/01640-2）。作者也感谢国家高性能处理中心（Centro Nacional de Processamento de Alto Desempenho, CENAPAD-SP）和 Google Cloud 提供的计算额度。

---

## References / 参考文献

正文引用编号与论文一致。下列为原文条目；中文是题名对照，不是改写后的引文。

1. L. Bonifacio, H. Abonizio, M. Fadaee, and R. Nogueira. Inpars: Data augmentation for information retrieval using large language models. arXiv preprint arXiv:2202.05144, 2022.  
   **中.** InPars：用大语言模型为信息检索做数据增强。
2. Z. Dai, V. Y. Zhao, J. Ma, Y. Luan, J. Ni, J. Lu, A. Bakalov, K. Guu, K. B. Hall, and M.-W. Chang. Promptagator: Few-shot dense retrieval from 8 examples. arXiv preprint arXiv:2209.11755, 2022.  
   **中.** Promptagator：从 8 个例子做少样本稠密检索。
3. J. Lin, X. Ma, S.-C. Lin, J.-H. Yang, R. Pradeep, and R. Nogueira. Pyserini: An easy-to-use python toolkit to support replicable ir research with sparse and dense representations. arXiv preprint arXiv:2102.10073, 2021.  
   **中.** Pyserini：用稀疏与稠密表示支持可复现 IR 研究的易用 Python 工具包。
4. R. Nogueira, Z. Jiang, R. Pradeep, and J. Lin. Document ranking with a pretrained sequence-to-sequence model. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings, pages 708–718, 2020.  
   **中.** 用预训练序列到序列模型做文档排序。
5. N. Shazeer and M. Stern. Adafactor: Adaptive learning rates with sublinear memory cost. In International Conference on Machine Learning, pages 4596–4604. PMLR, 2018.  
   **中.** Adafactor：内存开销次线性的自适应学习率。
6. N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych. Beir: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. arXiv preprint arXiv:2104.08663, 2021.  
   **中.** BEIR：信息检索模型零样本评测的异构基准。
7. H. Wachsmuth, S. Syed, and B. Stein. Retrieval of the best counterargument without prior topic knowledge. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 241–251, Melbourne, Australia, July 2018. Association for Computational Linguistics.  
   **中.** 在没有事先主题知识的情况下检索最佳反论点。
8. B. Wang and A. Komatsuzaki. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/mesh-transformer-jax, May 2021.  
   **中.** GPT-J-6B：一个 60 亿参数的自回归语言模型。
9. J. Wei, M. Bosma, V. Y. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021.  
   **中.** 微调后的语言模型是零样本学习者。
10. H. Zhuang, Z. Qin, R. Jagerman, K. Hui, J. Ma, J. Lu, J. Ni, X. Wang, and M. Bendersky. Rankt5: Fine-tuning t5 for text ranking with ranking losses. arXiv preprint arXiv:2210.10634, 2022.  
    **中.** RankT5：用排序损失微调 T5 做文本排序。
