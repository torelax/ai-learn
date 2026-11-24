# DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data / 全文中英对照

> 来源：[arXiv:2405.14333v1](https://arxiv.org/abs/2405.14333v1)。英文段落后紧接中文，覆盖所取 HTML 版本的主文、附录、图表、脚注与参考文献；公式、数字表和书目信息共用。中文以机器初译为底稿，进行了术语整理与结构核对，尚未逐句人工审定；遇到歧义请以紧邻英文和原论文为准。 原作版权归所列作者，许可见 [原作许可](http://creativecommons.org/licenses/by/4.0/)；中文译文与版式改写在该许可条件下提供（涉及 SA 时沿用相同许可，涉及 NC 时仅限非商业使用）。

## Authors and affiliations / 作者与机构

Huajian Xin Affiliation: DeepSeek Affiliation: Sun Yat-sen University Daya Guo Affiliation: DeepSeek Email: benjaminliu.eecs@gmail.com Zhihong Shao Affiliation: DeepSeek Email: wli8@ed.ac.uk Z.Z. Ren Qihao Zhu Bo Liu Affiliation: DeepSeek Email: xdliang328@gmail.com Chong Ruan Wenda Li Xiaodan Liang [^id1] Email: {xinhj, guoday, zhihongshao, rzz, zhuqh, chong.ruan}@deepseek.com, Affiliation: DeepSeek Affiliation: Sun Yat-sen University Affiliation: University of Edinburgh Affiliation: MBZUAI

## Abstract / 摘要

**EN.** Proof assistants like Lean have revolutionized mathematical proof verification, ensuring high accuracy and reliability. Although large language models (LLMs) show promise in mathematical reasoning, their advancement in formal theorem proving is hindered by a lack of training data. To address this issue, we introduce an approach to generate extensive Lean 4 proof data derived from high-school and undergraduate-level mathematical competition problems. This approach involves translating natural language problems into formal statements, filtering out low-quality statements, and generating proofs to create synthetic data. After fine-tuning the DeepSeekMath 7B model on this synthetic dataset, which comprises 8 million formal statements with proofs, our model achieved whole-proof generation accuracies of 46.3% with 64 samples and 52% cumulatively on the Lean 4 miniF2F test, surpassing the baseline GPT-4 at 23.0% with 64 samples and a tree search reinforcement learning method at 41.0%. Additionally, our model successfully proved 5 out of 148 problems in the Lean 4 Formalized International Mathematical Olympiad (FIMO) benchmark, while GPT-4 failed to prove any. These results demonstrate the potential of leveraging large-scale synthetic data to enhance theorem-proving capabilities in LLMs. Both the synthetic dataset and the model will be made available to facilitate further research in this promising field.

**中.** 像 Lean 这样的证明助手彻底改变了数学证明验证，确保了高精度和可靠性。尽管大型语言模型 (LLM) 在数学推理方面显示出前景，但由于缺乏训练数据，它们在形式定理证明方面的进步受到阻碍。为了解决这个问题，我们引入了一种方法来生成来自高中和本科水平数学竞赛问题的广泛 Lean 4 证明数据。这种方法涉及将自然语言问题转化为正式的陈述，过滤掉低质量的陈述，并生成证据以创建合成数据。在这个合成数据集（包含 800 万条带证明的正式语句）上对 DeepSeekMath 7B 模型进行微调后，我们的模型在 64 个样本中实现了 46.3% 的整体证明生成准确率，在 Lean 4 miniF2F 测试中累计达到 52%，超过了基线 GPT-4（在 64 个样本中为 23.0%）和树搜索强化学习方法的 41.0%。此外，我们的模型成功证明了 Lean 4 形式化国际数学奥林匹克 (FIMO) 基准中 148 个问题中的 5 个，而 GPT-4 未能证明任何问题。这些结果证明了利用大规模合成数据来增强LLM定理证明能力的潜力。合成数据集和模型都将可供使用，以促进这一前景广阔的领域的进一步研究。

## Introduction / 简介

**EN.** In modern mathematics, the increasing complexity of proofs presents substantial challenges for peer review. This complexity has led to the acceptance of erroneous proofs, with critical flaws often detected only after considerable time. To address these issues, formal mathematical languages such as Lean ( De Moura et al., 2015 ; Moura and Ullrich, 2021 ) , Isabelle ( Paulson, 1994 ) , and Coq ( The Coq Development Team, ) have been developed. These languages enable the creation of computer-verifiable proofs ( Avigad, 2023 ) . However, crafting formal proofs demands significant effort, specialized expertise, and poses challenges even for seasoned mathematicians. Consequently, the significance of automated theorem proving is on the rise ( Shulman, 2024 ) .

**中.** 在现代数学中，证明的日益复杂给同行评审带来了巨大的挑战。这种复杂性导致人们接受了错误的证明，而且往往要经过相当长的时间才能发现关键缺陷。为了解决这些问题，人们开发了 Lean（De Moura 等人，2015；Moura 和 Ullrich，2021）、Isabelle（Paulson，1994）和 Coq（The Coq 开发团队）等形式数学语言。这些语言可以创建计算机可验证的证明（Avigad，2023）。然而，制作形式证明需要付出巨大的努力和专业知识，甚至对经验丰富的数学家来说也是一个挑战。因此，自动化定理证明的重要性正在上升（Shulman，2024）。

**EN.** To reduce the effort involved in writing formal mathematical proofs, several approaches ( Polu and Sutskever, 2020 ; Jiang et al., 2021 ; Han et al., 2021 ; Polu et al., 2022 ; Lample et al., 2022 ; Jiang et al., 2022a ; Yang et al., 2024 ) have been developed, primarily focusing on search algorithms that explore potential solutions for proposed theorems. However, these methods often struggle with the vast search spaces required for complex theorems, rendering them ineffective for more intricate proofs ( Loos et al., 2017 ) . Recently, advances in large language models (LLMs) have introduced a novel strategy, utilizing pre-trained models to guide the search process. Although these new methods ( Jiang et al., 2022b ; Zhao et al., 2023 ; Xin et al., 2023 ) represent significant improvements, they still fall short of practical applicability due to the lack of parallel corpus. Unlike conventional programming languages such as Python or Java, formal proof languages are used by relatively few mathematicians, resulting in limited datasets. Recent advances in autoformalization ( Wu et al., 2022 ) allow more aligned data to be synthesized to train LLM-based automated theorem provers. Nevertheless, the resulting dataset remains too small to fully unleash the capabilities of LLMs.

**中.** 为了减少编写形式数学证明所需的工作量，人们开发了几种方法（Polu and Sutskever, 2020；Jiang et al., 2021；Han et al., 2021；Polu et al., 2022；Lample et al., 2022；Jiang et al., 2022a；Yang et al., 2024），主要侧重于探索潜力的搜索算法所提出定理的解决方案。然而，这些方法常常难以应对复杂定理所需的巨大搜索空间，导致它们对于更复杂的证明无效（Loos et al., 2017）。最近，大型语言模型（LLM）的进步引入了一种新颖的策略，利用预先训练的模型来指导搜索过程。尽管这些新方法（Jiang et al., 2022b；Zhao et al., 2023；Xin et al., 2023）代表了显着的改进，但由于缺乏平行语料库，它们仍然缺乏实际适用性。与 Python 或 Java 等传统编程语言不同，形式证明语言的数学家相对较少，因此数据集有限。自动形式化的最新进展（Wu et al., 2022）允许合成更多对齐的数据来训练基于 LLM 的自动定理证明者。然而，生成的数据集仍然太小，无法充分发挥LLM的能力。

**EN.** To address this issue, we propose a method for generating extensive Lean 4 proof data from informal mathematical problems. Our approach translates high-school and undergraduate-level mathematical competition problems into formal statements. We then automate proof generation using a large language model (LLM) and verify the correctness of these proofs within the Lean 4 environment. The primary challenge of this method is to ensure both the scale and quality of the synthetic data.

**中.** 为了解决这个问题，我们提出了一种从非正式数学问题生成广泛的 Lean 4 证明数据的方法。我们的方法将高中和本科水平的数学竞赛问题转化为正式的陈述。然后，我们使用大型语言模型 (LLM) 自动生成证明，并在 Lean 4 环境中验证这些证明的正确性。该方法的主要挑战是确保合成数据的规模和质量。

**EN.** Quality Assurance: We enhance the quality of generated proofs through a multi-step process. First, we filter out simple statements using a quality scoring model and exclude invalid statements via a hypothesis rejection strategy. Our novel iterative framework then improves proof quality by initially generating synthetic statements from informal math problems using an under-trained LLM fine-tuned on limited data. These statements are used to generate corresponding proofs, which are validated for correctness using a Lean 4 verifier. The correct theorem-proof pairs are subsequently used to further train the initial model. Through several iterations, the model trained on large-scale synthetic data becomes significantly more powerful than the originally under-trained LLMs, resulting in higher-quality theorem-proof pairs.

**中.** 质量保证：我们通过多步骤过程提高生成证明的质量。首先，我们使用质量评分模型过滤掉简单的陈述，并通过假设拒绝策略排除无效的陈述。然后，我们新颖的迭代框架首先使用训练不足的LLM（根据有限数据进行微调）从非正式数学问题生成综合陈述，从而提高证明质量。这些语句用于生成相应的证明，并使用 Lean 4 验证器验证其正确性。随后使用正确的定理证明对来进一步训练初始模型。通过多次迭代，在大规模合成数据上训练的模型变得比最初训练不足的 LLM 更强大，从而产生更高质量的定理证明对。

**EN.** Scale Assurance: To accelerate the proof generation process, our method addresses the challenge of the large search space for proofs. A significant cause of delays is the generation of unprovable statements that continue to be processed until they reach the time limit. To mitigate this, we propose proving negated statements in parallel. Once either the original statement or its negation is proved, the entire proving process is terminated.

**中.** 规模保证：为了加速证明生成过程，我们的方法解决了证明的大搜索空间的挑战。延迟的一个重要原因是生成无法证明的语句，这些语句将继续被处理，直到达到时间限制。为了缓解这种情况，我们建议并行证明否定陈述。一旦原命题或其否定被证明，整个证明过程就终止。

**EN.** We assess the effectiveness of our method on Lean 4 theorem proving using 488 problems from miniF2F ( Zheng et al., 2021 ) and 148 problems from the FIMO benchmarks ( Liu et al., 2023 ) . We utilize DeepSeekMath 7B ( Shao et al., 2024 ) , a state-of-the-art mathematical model, as our base. The results show that our iteratively trained model performs strongly, achieving 46.3% accuracy in whole-proof generation on the miniF2F-test benchmark with 64 samples, surpassing GPT-4 ( Achiam et al., 2023 ) at 23.0% and a reinforcement learning method at 41.0%. Additionally, our approach solved 4 out of 148 problems in the FIMO benchmark with 100 samples, while GPT-4 solved none, and our approach solved 5 with 4096 samples. Ablation experiments indicate that the model progressively solves more problems in miniF2F with each iteration. In summary, our paper makes the following contributions:

**中.** 我们使用 miniF2F 中的 488 个问题（Zheng et al., 2021）和 FIMO 基准中的 148 个问题（Liu et al., 2023）评估了我们的方法在 Lean 4 定理证明上的有效性。我们利用最先进的数学模型 DeepSeekMath 7B（Shao et al., 2024）作为我们的基础。结果表明，我们的迭代训练模型表现强劲，在 64 个样本的 miniF2F 测试基准上，整体证明生成的准确率达到 46.3%，超过了 GPT-4（Achiam 等人，2023）的 23.0% 和强化学习方法的 41.0%。此外，我们的方法用 100 个样本解决了 FIMO 基准中 148 个问题中的 4 个，而 GPT-4 没有解决任何问题，而我们的方法用 4096 个样本解决了 5 个问题。消融实验表明，模型每次迭代都会逐步解决 miniF2F 中的更多问题。总之，我们的论文做出了以下贡献：

**EN.** We introduce an iterative method to synthesize 8 million formal statements, each accompanied by a formal proof, from informal math problems. Experimental results demonstrate that this method significantly enhances both the scalability and quality of synthetic data.

**中.** 我们引入了一种迭代方法来从非正式数学问题中综合 800 万个正式陈述，每个陈述都附有正式证明。实验结果表明，该方法显着提高了合成数据的可扩展性和质量。

**EN.** Our model, trained on this synthetic dataset, achieves state-of-the-art performance on benchmarks, with whole-proof generation accuracies of 46.3% using 64 samples and 52% cumulatively on the Lean 4 miniF2F test. This surpasses the baseline GPT-4 at 23.0% with 64 samples and a tree search reinforcement learning method at 41.0%. Additionally, our model successfully proved 5 out of 148 problems in the Lean 4 Formalized International Mathematical Olympiad (FIMO) benchmark, while GPT-4 failed to prove any.

**中.** 我们的模型在这个合成数据集上进行训练，在基准测试中实现了最先进的性能，使用 64 个样本的整体证明生成精度为 46.3%，在 Lean 4 miniF2F 测试中累积精度为 52%。这超过了 64 个样本的基线 GPT-4 的 23.0% 和树搜索强化学习方法的 41.0%。此外，我们的模型成功证明了 Lean 4 形式化国际数学奥林匹克 (FIMO) 基准中 148 个问题中的 5 个，而 GPT-4 未能证明任何问题。

**EN.** We contribute to the mathematical and AI communities by creating and open-sourcing a large dataset of high-quality formal mathematical proofs, thereby fostering further research and development in automated theorem proving.

**中.** 我们通过创建和开源高质量形式数学证明的大型数据集，为数学和人工智能社区做出贡献，从而促进自动化定理证明的进一步研究和开发。

## Background and Related Works / 背景及相关工作

**EN.** Automated theorem proving has been a significant area of interest in artificial intelligence research since its inception ( Bibel, 2013 ) . Initial efforts were directed at simpler logical frameworks, which led to the development of highly efficient first-order theorem provers like E ( Schulz, 2002 ) and Vampire ( Kovács and Voronkov, 2013 ) . Nonetheless, these tools often fall short in handling complex theorems commonly found in modern proof assistants such as Lean ( De Moura et al., 2015 ) , Isabelle ( Paulson, 1994 ) , and Coq ( The Coq Development Team, ) . The advent of recent deep learning models and model-guided search techniques has reinvigorated the field ( Bansal et al., 2019 ) . This modern approach has not only enhanced the capabilities of ATP systems but also expanded their applicability in solving more intricate mathematical problems.

**中.** 自人工智能研究诞生以来，自动定理证明一直是人们感兴趣的重要领域（Bibel，2013）。最初的努力是针对更简单的逻辑框架，这导致了高效一阶定理证明器的开发，例如 E（Schulz，2002）和 Vampire（Kovács 和 Voronkov，2013）。尽管如此，这些工具在处理 Lean（De Moura 等人，2015）、Isabelle（Paulson，1994）和 Coq（Coq 开发团队）等现代证明助手中常见的复杂定理方面往往存在不足。最近深度学习模型和模型引导搜索技术的出现重振了该领域（Bansal 等人，2019）。这种现代方法不仅增强了 ATP 系统的能力，而且还扩展了其解决更复杂数学问题的适用性。

**EN.** ATP with Neural Models. With the development of deep learning, several approaches have been proposed to combine neural models with ATP ( Loos et al., 2017 ) . A series of ATP approaches adopts tree search algorithms guided by neural models ( Polu and Sutskever, 2020 ; Han et al., 2021 ; Polu et al., 2022 ; Jiang et al., 2022a ; Yang et al., 2024 ) . These approaches primarily utilize reinforcement learning techniques to enhance the accuracy of the model ( Kaliszyk et al., 2018 ; Crouse et al., 2021 ; Wu et al., 2021 ; Lample et al., 2022 ) . Since the search space is significantly large, the searching process consumes considerable time and computing resources.

**中.** ATP 与神经模型。随着深度学习的发展，人们提出了几种将神经模型与 ATP 相结合的方法（Loos et al., 2017）。一系列ATP方法采用神经模型引导的树搜索算法（Polu和Sutskever，2020；Han等，2021；Polu等，2022；Jiang等，2022a；Yang等，2024）。这些方法主要利用强化学习技术来提高模型的准确性（Kaliszyk et al., 2018; Crouse et al., 2021; Wu et al., 2021; Lample et al., 2022）。由于搜索空间非常大，搜索过程消耗大量的时间和计算资源。

**EN.** Another series of ATP approaches harnesses the power of large language models. These approaches typically involve language models that are fine-tuned with open-source proof data and interact with verifiers via a state-action transition program ( Polu and Sutskever, 2020 ; Jiang et al., 2021 ; Han et al., 2021 ; Polu et al., 2022 ; Lample et al., 2022 ; Jiang et al., 2022a ; Yang et al., 2024 ) . This process iteratively generates proof steps and verifies their correctness with formal verifiers. It then generates the next proof steps based on the proof states returned by the formal verifiers. Although these approaches achieve high performance, they are computationally intensive. To enhance efficiency, recent researches leverage language models to generate complete formal proofs directly ( First et al., 2023 ; Jiang et al., 2022b ; Zhao et al., 2023 ; Xin et al., 2023 ) , thus bypassing the iterative interaction during proof generation.

**中.** 另一系列 ATP 方法利用了大型语言模型的力量。这些方法通常涉及使用开源证明数据进行微调的语言模型，并通过状态-动作转换程序与验证者交互（Polu 和 Sutskever，2020；Jiang 等人，2021；Han 等人，2021；Polu 等人，2022；Lample 等人，2022；Jiang 等人，2022a；Yang 等人， 2024）。该过程迭代地生成证明步骤并通过形式验证器验证其正确性。然后，它根据形式验证者返回的证明状态生成下一个证明步骤。尽管这些方法实现了高性能，但它们是计算密集型的。为了提高效率，最近的研究利用语言模型直接生成完整的形式证明（First et al., 2023；Jiang et al., 2022b；Zhao et al., 2023；Xin et al., 2023），从而绕过证明生成过程中的迭代交互。

**EN.** Autoformalization for Formal Mathematics. Due to the limited availability of formal corpora for training, the performance of current large language models (LLMs) is also constrained. Thus, some approaches propose autoformalization ( Wu et al., 2022 ; Jiang et al., 2022b ) , which involves converting natural language descriptions into formal statements that can be verified by proof assistants. Several studies have generated synthetic datasets of formal proofs using rule-based transformations of existing theorems ( Wu et al., 2020 ; Wang and Deng, 2020 ; Xiong et al., 2023 ) . While effective, these methods are constrained by their reliance on predefined rules and lack flexibility for broader applications. Recent methodologies adopt large language models to translating natural language problems into formal statements ( Huang et al., 2024 ) . However, these datasets remain smaller than needed and are limited to small mathematical benchmarks, leading to only minor improvements in training outcomes for language models. In this paper, we aim to synthesise formal proofs via autoformalization at a much larger scale to boost the performance of a neural prover.

**中.** 形式数学的自动形式化。由于用于训练的正式语料库的可用性有限，当前大型语言模型（LLM）的性能也受到限制。因此，一些方法提出了自动形式化（Wu et al., 2022；Jiang et al., 2022b），其中涉及将自然语言描述转换为可以由证明助手验证的形式陈述。一些研究已经使用现有定理的基于规则的转换生成了形式证明的综合数据集（Wu et al., 2020；Wang and Deng, 2020；Xiong et al., 2023）。这些方法虽然有效，但因其对预定义规则的依赖而受到限制，并且缺乏更广泛应用的灵活性。最近的方法采用大型语言模型将自然语言问题转化为正式陈述（Huang et al., 2024）。然而，这些数据集仍然小于所需的数据集，并且仅限于小型数学基准，导致语言模型的训练结果仅略有改善。在本文中，我们的目标是通过更大规模的自动形式化来综合形式证明，以提高神经证明者的性能。

## Approach / 方法

![Figure S3.F1](figs/S3-F1-01.png)

**EN.** Figure 1 : An overview of our approach.

**中.** 图 1：我们的方法概述。

**EN.** In this section, we introduce our approach, which consists of four key processes as depicted in Figure 1 . The initial phase concentrates on generating formal mathematical statements from a broad collection of informal math problems, necessitating further proof. Next, the autoformalized statements are filtered through model scoring and hypothesis rejection methods to select high-quality statements. These statements are then proved by a model called DeepSeek-Prover, with their correctness verified by the formal verifier called Lean 4 [^footnote1] , yielding validated formal statements and proofs. These data serve as synthetic data for fine-tuning the DeepSeek-Prover. After enhancing DeepSeek-Prover, we repeat the entire previously described process. This cycle continues until the improvements in DeepSeek-Prover become marginal. Notably, to enhance proof efficiency, we prove concurrently both the original statements and their negations. This method has the advantage of swiftly discarding the original statement when it is invalid by proving its negation. The details of each phase will be described in the subsequent sections.

**中.** 在本节中，我们将介绍我们的方法，该方法由四个关键流程组成，如图 1 所示。初始阶段的重点是从大量非正式数学问题中生成正式数学陈述，需要进一步证明。接下来，通过模型评分和假设拒绝方法过滤自动形式化的语句，以选择高质量的语句。然后，这些陈述由名为 DeepSeek-Prover 的模型进行证明，其正确性由名为 Lean 4 [^footnote1] 的形式验证器验证，从而产生经过验证的形式陈述和证明。这些数据用作微调 DeepSeek-Prover 的合成数据。增强 DeepSeek-Prover 后，我们重复前面描述的整个过程。这个循环一直持续到 DeepSeek-Prover 的改进变得微乎其微为止。值得注意的是，为了提高证明效率，我们同时证明原始陈述及其否定。该方法的优点是，当原始语句无效时，可以通过证明其否定来快速丢弃原始语句。每个阶段的详细信息将在后续部分中描述。

### Autoformalization / 自动形式化

**EN.** The generation of formal proof data fundamentally relies on the availability of a substantial corpus of formal statements. In practice, however, amassing a large collection of manually crafted formal statements is challenging. Fortunately, the internet is replete with math-related problems expressed in natural language. By autoformalizing these informal mathematical problems, we can generate a vast repository of formal statements.

**中.** 形式证明数据的生成从根本上依赖于大量形式陈述语料库的可用性。然而，在实践中，收集大量手工制作的正式声明具有挑战性。幸运的是，互联网上充斥着用自然语言表达的数学相关问题。通过自动形式化这些非正式数学问题，我们可以生成大量正式陈述的存储库。

**EN.** We have observed that problems with explicit conditions and well-defined goals are typically easier to formalize compared to advanced mathematical topics that necessitate intricate definitions and constructions. Consequently, this paper primarily examines high school and undergraduate-level competition problems, with a particular emphasis on algebra and number theory, and to a lesser extent, combinatorics, geometry, and statistics. Despite their apparent simplicity, these problems often involve complex solution techniques, making them excellent candidates for constructing proof data to improve theorem-proving capabilities in Large Language Models (LLMs). To compile our dataset, we employed web scraping and careful data cleaning techniques to extract problems from online resources featuring high school and undergraduate exercises, exams, and competitions, resulting in a dataset of 869,659 high-quality natural language math problems.

**中.** 我们观察到，与需要复杂定义和构造的高级数学主题相比，具有明确条件和明确目标的问题通常更容易形式化。因此，本文主要研究高中和本科水平的竞赛问题，特别强调代数和数论，其次是组合学、几何和统计学。尽管这些问题表面上很简单，但它们通常涉及复杂的解决技术，这使得它们成为构造证明数据以提高大型语言模型（LLM）中定理证明能力的优秀候选者。为了编译我们的数据集，我们采用了网络抓取和仔细的数据清理技术，从包含高中和本科练习、考试和竞赛的在线资源中提取问题，从而生成了包含 869,659 个高质量自然语言数学问题的数据集。

**EN.** Specifically, we initialized the DeepSeek-Prover using the DeepSeekMath-Base 7B model ( Shao et al., 2024 ) . Initially, the model struggled to convert informal math problems into formal statements. To address this, we fine-tuned the DeepSeek-Prover model using the MMA dataset ( Jiang et al., 2023 ) , which comprises formal statements from Lean 4’s mathlib [^footnote2] that were back-translated into natural language problem descriptions by GPT-4. We then instructed the model to translate these natural language problems into formal statements in Lean 4 using a structured approach.

**中.** 具体来说，我们使用 DeepSeekMath-Base 7B 模型初始化 DeepSeek-Prover (Shao et al., 2024)。最初，该模型努力将非正式的数学问题转换为正式的陈述。为了解决这个问题，我们使用 MMA 数据集（Jiang et al., 2023）对 DeepSeek-Prover 模型进行了微调，该数据集包含来自 Lean 4 的 mathlib [^footnote2] 的正式语句，这些语句由 GPT-4 反向翻译为自然语言问题描述。然后，我们指示模型使用结构化方法将这些自然语言问题转化为 Lean 4 中的正式语句。

**EN.** Prompt : Mathematical Problem in Natural Language: {$informal_statement_with_answers} Translate the problem to Lean 4 (only the core declaration): ‘‘‘lean4

**中.** 提示：自然语言数学问题：{$informal_statement_with_answers} 将问题翻译为 Lean 4（仅核心声明）：‘‘‘lean4

**EN.** Response : {$formal_statement} ‘‘‘

**中.** 响应：{$formal_statement} ‘‘‘

### Quality Filtering / 质量过滤

**EN.** The quality of the autoformalized statements was found to be suboptimal due to two main issues. Firstly, many formal statements were overly simplistic. To address this, we developed scoring criteria and provided examples from miniF2F-valid as few-shot examples to guide the DeepSeek-Prover model in evaluating the content and quality of these statements using a chain-of-thought approach. Manual review of these scores confirmed that the model’s evaluations closely matched human intuition and expectations. Specifically, the model was instructed (see Appendix A.1 for the detailed prompt) to classify the quality of each formal statement into categories: "excellent," "good," "above average," "fair," or "poor." Statements rated as "fair" or "poor" were subsequently excluded.

**中.** 由于两个主要问题，自动形式化语句的质量被发现不是最佳的。首先，许多正式声明过于简单化。为了解决这个问题，我们制定了评分标准，并提供了 miniF2F-valid 中的示例作为少数样本，以指导 DeepSeek-Prover 模型使用思维链方法评估这些语句的内容和质量。对这些分数的手动审查证实，该模型的评估与人类的直觉和期望非常吻合。具体来说，模型被指示（详细提示见附录 A.1）将每个正式陈述的质量分为几类：“优秀”、“良好”、“高于平均水平”、“一般”或“差”。被评为“一般”或“差”的陈述随后被排除。

**EN.** The second issue pertains to formal statements that, although provable, are based on inconsistent hypotheses leading to vacuous conclusions, rendering the conclusions meaningless in mathematics. For example, consider the following model-generated statement:

**中.** 第二个问题涉及正式陈述，尽管这些陈述是可证明的，但其基于不一致的假设，导致空洞的结论，从而使结论在数学中毫无意义。例如，考虑以下模型生成的语句：

```text
example (θ : ℝ) (h₀ : ∀ z : ℂ, z ^ 2 = -1 ∧ z ^ 3 = -1 ∧ z ^ 6 = 1) (h₁ : Real.tan θ = 2 * Real.sqrt 3) : θ = 5 * Real.pi / 3
```

**EN.** Here, the hypothesis $z^{2}=-1\wedge z^{3}=-1\wedge z^{6}=1$ for all complex numbers is clearly false, making any derived conclusions meaningless. To eliminate such cases from our dataset, we implemented a hypothesis rejection method. This involves using the DeepSeek-Prover model to attempt proving the formal statement with ’False’ as the conclusion. A successful proof indicates an invalid hypothesis, prompting exclusion of the statement. An example is shown below:

**中.** 在这里，所有复数的假设 $z^{2}=-1\wedge z^{3}=-1\wedge z^{6}=1$ 显然是错误的，使得任何导出的结论都毫无意义。为了从我们的数据集中消除此类情况，我们实施了假设拒绝方法。这涉及使用 DeepSeek-Prover 模型尝试证明以“False”作为结论的正式陈述。成功的证明表明假设无效，从而促使该陈述被排除。示例如下所示：

```text
example (θ : ℝ) (h₀ : ∀ z : ℂ, z ^ 2 = -1 ∧ z ^ 3 = -1 ∧ z ^ 6 = 1) (h₁ : Real.tan θ = 2 * Real.sqrt 3) : False := by
  simpa using h₀ 1
```

**EN.** By applying this dual strategy of model scoring and hypothesis rejection, we curated a refined set of 712,073 high-quality formal statements, providing a robust foundation for further proof synthesis.

**中.** 通过应用模型评分和假设拒绝的双重策略，我们精心策划了一套包含 712,073 个高质量正式陈述的精炼集，为进一步的证明综合提供了坚实的基础。

### Statement Proving / 陈述证明

**EN.** After creating a substantial corpus of high-quality formal statements, we employed the model to search for proofs of these statements. Traditionally, language models have been used predominantly in a brute-force manner to prove theorems—repeatedly attempting until a valid proof is found or computational resources are exhausted. This approach is inefficient for our purposes. Typically, language models are applied to human-curated formal statements that are carefully crafted and generally true and provable; however, in our task of proving autoformalized statements, many of the statements produced by the model may be incorrect. Indeed, it is unreasonable to expect the model to validate a false proposition within any reliable proof system. This issue becomes more pronounced during large-scale autoformalization, where we observed that at least 20% of the formal statements generated by our model, even after quality filtering, were incorrect, leading to significant computational waste if addressed with brute force.

**中.** 在创建了大量高质量正式陈述的语料库后，我们使用该模型来搜索这些陈述的证据。传统上，语言模型主要以暴力方式来证明定理——反复尝试，直到找到有效的证明或耗尽计算资源。对于我们的目的来说，这种方法效率低下。通常，语言模型应用于人工策划的正式陈述，这些陈述是精心设计的，通常是真实的和可证明的；然而，在我们证明自动形式化陈述的任务中，模型产生的许多陈述可能是不正确的。事实上，期望模型在任何可靠的证明系统中验证伪命题是不合理的。这个问题在大规模自动形式化过程中变得更加明显，我们观察到，即使经过质量过滤，我们的模型生成的形式语句中至少有 20% 是不正确的，如果用暴力解决，会导致大量的计算浪费。

**EN.** To minimize resource wastage on unprovable statements and improve the efficiency of the proof search process, we exploited the logical symmetry between a statement and its negation to accelerate proof synthesis. We implemented dual concurrent proof searches for each synthetic statement—one for the statement $\Gamma\vdash P$ and another for its negation $\Gamma\vdash\neg P$ . The search terminates as soon as a valid proof is found for either, conclusively demonstrating the unprovability of the other. Each proof search stream attempts up to $k$ proofs unless a valid proof emerges sooner.

**中.** 为了最大限度地减少无法证明的陈述的资源浪费并提高证明搜索过程的效率，我们利用陈述及其否定之间的逻辑对称性来加速证明合成。我们对每个合成语句实现了双重并发证明搜索——一个用于语句 $\Gamma\vdash P$ ，另一个用于其否定 $\Gamma\vdash\neg P$ 。一旦找到其中一个的有效证据，搜索就会终止，最终证明另一个的不可证明性。每个证明搜索流最多尝试 $k$ 证明，除非更快出现有效证明。

**EN.** All validated proofs, whether they justify the original theorems or their negations, are then aggregated to further train the DeepSeek-Prover. Thus, this dual approach serves as a form of data augmentation, enriching the dataset with both propositions and their negations—even if the original propositions were not correctly formalized by the model.

**中.** 所有经过验证的证明，无论是证明原始定理还是否定定理，都会被汇总以进一步训练 DeepSeek-Prover。因此，这种双重方法可以作为数据增强的一种形式，通过命题及其否定来丰富数据集——即使模型没有正确地形式化原始命题。

### Iterative Enhancement / 迭代增强

**EN.** Since the entire pipeline heavily relies on the DeepSeek-Prover, enhancing the model’s performance after each iteration is crucial. To achieve this, we consistently fine-tune the model with newly generated data. The updated model is then utilized for subsequent autoformalization iterations. The key insight from this iterative process is that the model incrementally improves in strength and efficacy after each cycle of refinement and application. This iterative process continues until no further gains are observed. Consequently, the theorem-proof pairs generated by the model become increasingly higher in quality with each iteration. This method ensures that the DeepSeek-Prover consistently enhances its performance, ultimately producing superior theorem-proof pairs through continuous refinement.

**中.** 由于整个流程严重依赖 DeepSeek-Prover，因此在每次迭代后增强模型的性能至关重要。为了实现这一目标，我们不断地使用新生成的数据对模型进行微调。然后，更新的模型用于后续的自动形式化迭代。这个迭代过程的关键见解是，在每个细化和应用周期之后，模型的强度和功效逐渐提高。这个迭代过程一直持续到没有观察到进一步的增益为止。因此，模型生成的定理证明对的质量随着每次迭代而变得越来越高。这种方法确保 DeepSeek-Prover 不断增强其性能，最终通过不断的改进产生卓越的定理证明对。

## Experiments / 实验

### Experimental Setup / 实验装置

**EN.** DeepSeek-Prover is built upon DeepSeekMath-Base 7B model ( Shao et al., 2024 ) , a decoder-only transformer ( Vaswani et al., 2017 ) pre-trained on a corpus comprising 120 billion math-related tokens. We fine-tuned this model using a global batch size of 512 and a constant learning rate of $1\times 10^{-4}$ , incorporating 6,000 warmup steps with synthetic data. DeepSeek-Prover’s performance was evaluated against several baselines:

**中.** DeepSeek-Prover 基于 DeepSeekMath-Base 7B 模型（Shao 等人，2024）构建，这是一个仅解码器的转换器（Vaswani 等人，2017），在包含 1200 亿个数学相关标记的语料库上进行了预训练。我们使用全局批量大小 512 和恒定学习率 $1\times 10^{-4}$ 对该模型进行了微调，将 6,000 个预热步骤与合成数据结合在一起。 DeepSeek-Prover 的性能根据多个基准进行了评估：

**EN.** GPT-3.5 and GPT-4 ( Achiam et al., 2023 ) , developed by OpenAI, are advanced generative AI models known for their effectiveness in diverse tasks, including code generation. Although not explicitly designed for theorem proving, their extensive scale and parameter count confer significant capabilities. In contrast, DeepSeekMath is a specialized model, explicitly pre-trained for mathematical content. We utilized both GPT-4 (specifically the GPT-4-turbo 0409 version) and DeepSeekMath to generate complete proofs for given theorems using a methodology similar to ours.

**中.** 由 OpenAI 开发的 GPT-3.5 和 GPT-4（Achiam 等人，2023）是先进的生成式 AI 模型，以其在包括代码生成在内的各种任务中的有效性而闻名。尽管没有明确设计用于定理证明，但其广泛的规模和参数计数赋予了重要的功能。相比之下，DeepSeekMath 是一个专门的模型，针对数学内容进行了明确的预训练。我们利用 GPT-4（特别是 GPT-4-turbo 0409 版本）和 DeepSeekMath，使用与我们类似的方法来生成给定定理的完整证明。

**EN.** GPT-f ( Polu and Sutskever, 2020 ) , utilizing a GPT-2-inspired architecture ( Radford et al., 2019 ) , implements an iterative best-first search method to progressively generate and validate proof steps within a formal proof setting until a proof is either completed or resources are depleted. This methodology has been further advanced by Proof Artifact Co-Training ( Han et al., 2021 ) , ReProver ( Yang et al., 2024 ) , Llemma ( Azerbayev et al., 2023 ) , and COPRA ( Thakur et al., 2023 ) , which employ either specialized fine-tuned models or versatile general-purpose models such as GPT-3.5 and GPT-4 for the generation of proof steps.

**中.** GPT-f（Polu 和 Sutskever，2020）利用受 GPT-2 启发的架构（Radford 等人，2019）实现迭代最佳优先搜索方法，在正式证明设置中逐步生成和验证证明步骤，直到证明完成或资源耗尽。 Proof Artifact Co-Training (Han et al., 2021)、ReProver (Yang et al., 2024)、Llemma (Azerbayev et al., 2023) 和 COPRA (Thakur et al., 2023) 进一步推进了这种方法，它们采用专门的微调模型或多功能通用模型，例如 GPT-3.5 和GPT-4 用于生成证明步骤。

### Main Results / 主要结果

**EN.** This study addresses complex mathematical problems in algebra and number theory. We evaluate the theorem-proving efficacy of our model using the miniF2F ( Zheng et al., 2021 ) and FIMO ( Liu et al., 2023 ) benchmarks. The metric pass@k is employed to denote the scenario where at least one valid proof is discovered among the first k attempts generated by the model.

**中.** 这项研究解决了代数和数论中的复杂数学问题。我们使用 miniF2F（Zheng et al., 2021）和 FIMO（Liu et al., 2023）基准评估了我们模型的定理证明功效。度量 pass@k 用于表示在模型生成的前 k 次尝试中发现至少一个有效证明的场景。

| Method<br>方法 | Model size<br>型号尺寸 | Generation times<br>世代时代 | miniF2F-valid<br>miniF2F-有效 | miniF2F-test<br>miniF2F 测试 |
| --- | --- | --- | --- | --- |
| Tree Search Methods<br>树搜索方法 | Tree Search Methods<br>树搜索方法 | Tree Search Methods<br>树搜索方法 | Tree Search Methods<br>树搜索方法 | Tree Search Methods<br>树搜索方法 |
| COPRA (GPT-3.5) ( Thakur et al., 2023 )<br>COPRA (GPT-3.5)（Thakur 等人，2023） | - | $1\times 60$ | - | $9.0\%$ |
| COPRA (GPT-4) ( Thakur et al., 2023 )<br>COPRA (GPT-4)（Thakur 等人，2023） | - | $1\times 60$ | - | $26.6\%$ |
| Proof Artifact Co-Training ( Han et al., 2021 )<br>证明工件协同训练（Han et al., 2021） | 837M | $1\times 8\times 512$ | $23.9\%$ | $24.6\%$ |
| Proof Artifact Co-Training ( Han et al., 2021 )<br>证明工件协同训练（Han et al., 2021） | 837M | $8\times 8\times 512$ | $29.3\%$ | $29.2\%$ |
| ReProver ( Yang et al., 2024 )<br>ReProver（Yang 等人，2024） | 229M | $1\times 3751$ | - | $25.0\%$ |
| Llemma ( Azerbayev et al., 2023 )<br>Llemma（阿塞巴耶夫等人，2023） | 7B | $1\times 3200$ | - | $26.2\%$ |
| Llemma ( Azerbayev et al., 2023 )<br>Llemma（阿塞巴耶夫等人，2023） | 34B | $1\times 3200$ | - | $25.8\%$ |
| Curriculum Learning ( Polu et al., 2022 )<br>课程学习（Polu et al., 2022） | 837M | $1\times 8\times 512$ | $33.6\%$ | $29.6\%$ |
| Curriculum Learning ( Polu et al., 2022 )<br>课程学习（Polu et al., 2022） | 837M | $8\times 8\times 512$ | $41.2\%$ | $34.5\%$ |
| Curriculum Learning ( Polu et al., 2022 )<br>课程学习（Polu et al., 2022） | 837M | $64\times 8\times 512$ | $47.3\%$ | $36.6\%$ |
| Hypertree Proof Search ( Lample et al., 2022 )<br>超树证明搜索（Lample 等人，2022） | 600M | cumulative<br>累计 | $58.6\%$ | - |
| Hypertree Proof Search ( Lample et al., 2022 )<br>超树证明搜索（Lample 等人，2022） | 600M | $64\times 5000$ | - | $41.0\%$ |
| Whole-Proof Generation Methods<br>整体证明生成方法 | Whole-Proof Generation Methods<br>整体证明生成方法 | Whole-Proof Generation Methods<br>整体证明生成方法 | Whole-Proof Generation Methods<br>整体证明生成方法 | Whole-Proof Generation Methods<br>整体证明生成方法 |
| GPT-4-turbo 0409<br>GPT-4-涡轮0409 | - | 64 | $25.4\%$ | $23.0\%$ |
| DeepSeekMath-Base ( Shao et al., 2024 )<br>DeepSeekMath-Base（Shao 等人，2024） | 7B | 128 | $25.4\%$ | $27.5\%$ |
| DeepSeek-Prover<br>DeepSeek-证明者 | 7B | cumulative<br>累计 | 60.2% | 52.0% |
| DeepSeek-Prover<br>DeepSeek-证明者 | 7B | $1$ (greedy)<br>$1$（贪心） | - | $30.0\%$ |
| DeepSeek-Prover<br>DeepSeek-证明者 | 7B | $64$ | - | $46.3\%$ |
| DeepSeek-Prover<br>DeepSeek-证明者 | 7B | $128$ | - | $46.3\%$ |
| DeepSeek-Prover<br>DeepSeek-证明者 | 7B | $8192$ | - | $48.8\%$ |
| DeepSeek-Prover<br>DeepSeek-证明者 | 7B | $65536$ | - | $50.0\%$ |

**EN.** Table 1 : Comparing with state-of-the-arts on the miniF2F dataset.

**中.** 表 1：与 miniF2F 数据集上的最新技术进行比较。

**EN.** Results on MiniF2F. The miniF2F benchmark consists of 244 validation and 244 test problems, ranging from basic arithmetic to competition-level problems, e.g., problems from the American Invitational Mathematics Examination (AIME), the American Mathematics Competitions (AMC), and the International Mathematical Olympiad (IMO). We use the version of miniF2F in Lean 4, which was released by the LeanDojo project ( [https://github.com/yangky11/miniF2F-lean4](https://github.com/yangky11/miniF2F-lean4) ).

**中.** MiniF2F 上的结果。 miniF2F基准测试由244个验证问题和244个测试问题组成，范围从基础算术到竞赛级别的问题，例如美国数学邀请考试（AIME）、美国数学竞赛（AMC）和国际数学奥林匹克（IMO）的问题。我们使用Lean 4中的miniF2F版本，该版本由LeanDojo项目（[https://github.com/yangky11/miniF2F-lean4](https://github.com/yangky11/miniF2F-lean4）发布。

**EN.** Table 1 compares various state-of-the-art methods on the miniF2F dataset. DeepSeek-Prover outperforms all with cumulative scores of 60.2% on miniF2F-valid and 52.0% on miniF2F-test, significantly higher than other methods, including GPT-4 which scores 25.41% and 22.95%, respectively. Even the best tree search method, Hypertree Proof Search with a 600M model, achieves only up to 58.6% on miniF2F-valid and 41.0% on miniF2F-test. DeepSeek-Prover’s scalability is evident as its performance improves with increased computational resources, rising from 30.03% using a greedy approach to 50.0% at 65536 generation times, demonstrating its effectiveness in handling complex proof scenarios. Examples of proved theorems of MiniF2F can be found in Appendix A.3.1 .

**中.** 表 1 比较了 miniF2F 数据集上的各种最先进的方法。 DeepSeek-Prover 的表现优于所有方法，在 miniF2F 有效上的累积得分为 60.2%，在 miniF2F 测试上的累积得分为 52.0%，显着高于其他方法，包括 GPT-4，得分分别为 25.41% 和 22.95%。即使是最好的树搜索方法（具有 600M 模型的超树证明搜索），在 miniF2F 验证上也只能达到 58.6%，在 miniF2F 测试上只能达到 41.0%。 DeepSeek-Prover 的可扩展性是显而易见的，其性能随着计算资源的增加而提高，从使用贪婪方法的 30.03% 上升到 65536 代时的 50.0%，展示了其处理复杂证明场景的有效性。 MiniF2F 的证明定理的例子可以在附录 A.3.1 中找到。

**EN.** Results on FIMO. The FIMO benchmark comprises 149 formal problems which are sourced from the IMO shortlist translated into Lean 4. Our method successfully proved 4 theorems with 100 attempts per theorem, whereas GPT-4 failed to prove any. By increasing the number of attempts per theorem to 4,096, we successfully proved an additional theorem. Examples of proved theorems of FIMO can be found in Appendix A.3.2 .

**中.** FIMO 上的结果。 FIMO 基准包含 149 个形式问题，这些问题源自翻译为 Lean 4 的 IMO 候选名单。我们的方法成功证明了 4 个定理，每个定理尝试了 100 次，而 GPT-4 未能证明任何定理。通过将每个定理的尝试次数增加到 4,096 次，我们成功证明了另一个定理。 FIMO 定理证明的例子可以在附录 A.3.2 中找到。

### Ablation Studies / 消融研究

#### The Effectiveness of Large-scale Autoformalization / 大规模自动形式化的有效性

**EN.** To demonstrate the effectiveness of large-scale autoformalization, we conducted a comparative analysis as shown in Table 2 between our autoformalized dataset and conventional datasets using expert iteration ( Polu and Sutskever, 2020 ) . This iterative method entails generating formal proofs, fine-tune the model based on successful outcomes, and iterating this process until no additional enhancements are observed. The results indicate that models trained with our autoformalized data significantly outperform those trained solely with mathlib data.

**中.** 为了证明大规模自动形式化的有效性，我们使用专家迭代对我们的自动形式化数据集和传统数据集进行了如表 2 所示的比较分析（Polu 和 Sutskever，2020）。这种迭代方法需要生成正式的证明，根据成功的结果微调模型，并迭代此过程，直到观察不到额外的增强。结果表明，使用我们的自动形式化数据训练的模型明显优于仅使用 mathlib 数据训练的模型。

| Model<br>型号 | #Tokens<br>#代币 | miniF2F-valid<br>miniF2F-有效 | miniF2F-test<br>miniF2F 测试 |
| --- | --- | --- | --- |
| - | - | $25.4\%$ | $27.5\%$ |
| Mathlib<br>数学库 | $0.238$ B | $30.3\%$ | $31.2\%$ |
| Autoformalized Statements<br>自动形式化语句 | $3.108$ B | $48.8\%$ | $42.6\%$ |

**EN.** Table 2 : Improvement in pass rates for miniF2F at pass@128 in models trained on formal proofs, including those derived from human-authored theorems in Lean 4’s mathlib and automatically formalized theorems.

**中.** 表 2：在经过形式证明训练的模型中，miniF2F 在 pass@128 的通过率有所提高，包括那些源自 Lean 4 的 mathlib 中人类创作的定理和自动形式化定理的模型。

#### The Effectiveness of Formal Statements Scoring / 正式陈述评分的有效性

**EN.** To demonstrate the effectiveness of the model in filtering out low-quality statements, we fine-tuned the DeepSeekMath-Base model using an equal amount of high-score proof data and low-score proof data to verify the quality of the data, as shown in Table 3 . The table shows that the model trained on high-score proof data outperformed the model trained on low-score proof data by 4.5%. This enhancement underscores the utility of the model in accurately scoring and effectively filtering out lower-quality statements.

**中.** 为了证明模型过滤低质量语句的有效性，我们使用等量的高分证明数据和低分证明数据对 DeepSeekMath-Base 模型进行微调，以验证数据的质量，如表 3 所示。该表显示，在高分证明数据上训练的模型比在低分证明数据上训练的模型性能高出 4.5%。这一增强强调了该模型在准确评分和有效过滤低质量陈述方面的实用性。

| Scored Class<br>评分等级 | miniF2F-valid<br>miniF2F-有效 | miniF2F-test<br>miniF2F 测试 |
| --- | --- | --- |
| "excellent", "good" and "above average"<br>“优秀”、“良好”和“高于平均水平” | $48.8\%$ | $42.6\%$ |
| "fair" and "poor"<br>“一般”和“差” | $41.4\%$ | $38.1\%$ |

**EN.** Table 3 : Improvement in pass rates for miniF2F at pass@128 in models trained on differently scored proof data.

**中.** 表 3：在不同评分的证明数据上训练的模型中，miniF2F 在 pass@128 的通过率有所提高。

#### The Effectiveness of Iterative Enhancement / 迭代增强的有效性

**EN.** Table 4 demonstrates a distinct correlation between the number of iterations in data synthesis and enhanced performance in theorem proving. This evidence underscores the success of our iterative enhancement strategy in augmenting theorem-proving capabilities. Successive iterations not only refine the model’s ability to handle complex proofs but also significantly increase the quality and quantity of the synthetic data produced.

**中.** 表 4 展示了数据合成中的迭代次数与定理证明中的增强性能之间的明显相关性。这一证据强调了我们的迭代增强策略在增强定理证明能力方面的成功。连续的迭代不仅提高了模型处理复杂证明的能力，而且还显着提高了生成的合成数据的质量和数量。

| Model<br>型号 | miniF2F-valid<br>miniF2F-有效 | miniF2F-test<br>miniF2F 测试 |
| --- | --- | --- |
| iteration 0<br>迭代0 | $38.1\%$ | $34.0\%$ |
| iteration 1<br>迭代1 | $45.1\%$ | $39.3\%$ |
| iteration 2<br>迭代2 | $49.2\%$ | $41.4\%$ |
| iteration 3<br>迭代3 | $54.5\%$ | $45.1\%$ |
| iteration 4<br>迭代4 | $59.4\%$ | $46.3\%$ |

**EN.** Table 4 : Improvement in pass rates for miniF2F at pass@128 in models across successive training iterations, facilitated by the incremental integration of synthesized data via autoformalization.

**中.** 表 4：通过自动形式化对合成数据进行增量集成，模型中 miniF2F 在 pass@128 的通过率在连续训练迭代中得到提高。

#### The Effectiveness of Scaling Synthetic Theorem Proving Data / 扩展综合定理证明数据的有效性

**EN.** Our investigation into synthetic theorem proving data reveals a clear correlation between dataset size and model efficacy, as illustrated in Table 5 . By examining subsets of the eight million generated proof data points, we observed that performance on the miniF2F benchmark improves proportionally to the exponential increase in dataset size. This pattern highlights the pivotal importance of large-scale datasets for boosting model proficiency in automatically formalizing natural language questions. These findings emphasize the significant potential and necessity of systematic data construction for progressing in the field of automated theorem proving.

**中.** 我们对综合定理证明数据的调查揭示了数据集大小和模型功效之间的明显相关性，如表 5 所示。通过检查 800 万个生成的证明数据点的子集，我们观察到 miniF2F 基准测试的性能随着数据集大小的指数增长而按比例提高。这种模式凸显了大规模数据集对于提高模型自动形式化自然语言问题的熟练程度的至关重要性。这些发现强调了系统数据构建对于自动化定理证明领域进步的巨大潜力和必要性。

| Size<br>尺寸 | miniF2F-valid<br>miniF2F-有效 | miniF2F-test<br>miniF2F 测试 |
| --- | --- | --- |
| 1,000 | $22.95\%$ | $24.18\%$ |
| 10,000 | $32.79\%$ | $31.97\%$ |
| 100,000 | $36.07\%$ | $37.7\%$ |
| 1,000,000 | $39.34\%$ | $38.11\%$ |
| 8,066,621 | $42.62\%$ | $40.16\%$ |

**EN.** Table 5 : Improvement in pass rates for miniF2F at pass@128 in models trained with a larger fraction of synthesized data via autoformalization.

**中.** 表 5：通过自动形式化使用较大比例的合成数据训练的模型中，miniF2F 在 pass@128 的通过率有所提高。

## Case Studies / 案例研究

**EN.** This section presents two case studies to demonstrate the application of our methods in autoformalizing theorems. It showcases both successful proofs and the identification of inconsistencies during the Hypothesis Rejection stage.

**中.** 本节介绍两个案例研究来演示我们的方法在自动形式化定理中的应用。它展示了假设拒绝阶段的成功证明和不一致的识别。

### Autoformalized Theorem with Complete Proof / 具有完整证明的自动形式化定理

**EN.** Example a. Problem : Prove that the determinant of the following matrix is zero.

**中.** 示例a.问题：证明以下矩阵的行列式为零。

$$
\begin{bmatrix}1&\cos(a-b)&\cos(a)\\
\cos(a-b)&1&\cos(b)\\
\cos(a)&\cos(b)&1\end{bmatrix}
$$

**EN.** Autoformalized Theorem in Lean :

**中.** 精益中的自动形式化定理：

```text
example (a b : ℝ) :
  Matrix.det ![![1, Real.cos (a - b), Real.cos a], ![Real.cos (a - b), 1, Real.cos b], ![Real.cos a, Real.cos b, 1]] = 0
```

**EN.** This approach effectively translates the algebraic expression of the matrix and its determinant into a formal language using Lean. The autoformalization captures the essence of the original mathematical statement by defining a specific $3\times 3$ matrix dependent on real numbers $a$ and $b$ , and asserts that its determinant is zero. The formalization employs the Matrix.det function to compute the determinant, utilizing the ![...] notation for lists of lists in Lean to represent the matrix rows.

**中.** 这种方法使用 Lean 有效地将矩阵及其行列式的代数表达式转换为形式语言。自动形式化通过定义依赖于实数 $a$ 和 $b$ 的特定 $3\times 3$ 矩阵来捕获原始数学语句的本质，并断言其行列式为零。形式化使用 Matrix.det 函数来计算行列式，并利用 Lean 中列表列表的 ![...] 符号来表示矩阵行。

### Autoformalization of Theorem with Inconsistent Hypotheses / 具有不一致假设的定理的自动形式化

**EN.** Example b. Problem : Given a real number $D$ and the condition that for non-zero real numbers $a,b,c$ , the determinant of the matrix $\begin{bmatrix}a&b&c\\ 1&4&9\\ 3&1&2\end{bmatrix}$ equals $D$ , prove that $D^{2}=154$ .

**中.** 示例 B.问题：给定一个实数 $D$ ，并且对于非零实数 $a,b,c$ ，矩阵 $\begin{bmatrix}a&b&c\\ 1&4&9\\ 3&1&2\end{bmatrix}$ 的行列式等于 $D$ ，证明 $D^{2}=154$ 。

**EN.** Autoformalized Theorem in Lean :

**中.** 精益中的自动形式化定理：

```text
example (D : ℝ) (h₀ : ∀ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 →
  Matrix.det ![![a, b, c], ![1, 4, 9], ![3, 1, 2]] = D) : D ^ 2 = 154
```

**EN.** The initial autoformalization incorrectly assumes that the condition $D^{2}=154$ universally applies to all non-zero real numbers $a$ , $b$ , and $c$ . This assumption is not supported by the problem statement, which does not claim universal applicability. Instead, the formalization should aim to either identify specific values of $a$ , $b$ , and $c$ that satisfy $D^{2}=154$ or demonstrate that no such values exist.

**中.** 初始自动形式化错误地假设条件 $D^{2}=154$ 普遍适用于所有非零实数 $a$ 、 $b$ 和 $c$ 。问题陈述不支持这一假设，问题陈述并不声称具有普遍适用性。相反，形式化的目的应该是识别满足 $D^{2}=154$ 的 $a$ 、 $b$ 和 $c$ 的特定值，或者证明不存在此类值。

**EN.** The model successfully identifies this inconsistency and provides a counterexample to demonstrate the absurdity of the hypothesis:

**中.** 该模型成功地识别了这种不一致，并提供了一个反例来证明该假设的荒谬性：

```text
example (D : ℝ) (h₀ : ∀ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 →
  Matrix.det ![![a, b, c], ![1, 4, 9], ![3, 1, 2]] = D) : False := by
  have h₁ := h₀ 1 2 3
  have h₂ := h₀ 1 4 9
  simp [Matrix.det_fin_three] at h₁ h₂
  linarith
```

**EN.** A corrected version of the autoformalized theorem can be proposed as follows:

**中.** 自动形式化定理的修正版本可以提出如下：

```text
example (a b c : ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  let D := Matrix.det ![![a, b, c], ![1, 4, 9], ![3, 1, 2]];
  D ^ 2 = 154
```

**EN.** These examples illustrate the model’s capability to verify proofs and identify hypothesis inconsistencies effectively. Further details can be found in Appendix A.2 .

**中.** 这些例子说明了模型有效验证证据和识别假设不一致的能力。更多详细信息请参见附录 A.2。

## Conclusion / 结论

**EN.** In this paper, we presented a method to generate extensive synthetic proof data from high-school and undergraduate-level mathematical competition problems. By translating natural language problems into formal statements, filtering out low-quality ones, and using iterative proof generation, we created 8 million proof data points and significantly improved the DeepSeekMath 7B model’s performance in ATP when trained on this synthetic data. Our model outperforms GPT-4 and other methods on benchmarks like miniF2F and FIMO. By open-sourcing our dataset and model, we aim to advance research in automated theorem proving and enhance the capabilities of large language models in formal mathematical reasoning. Currently, our work mainly focuses on algebra and number theory at the middle school and undergraduate levels. In future work, we will aim to expand the diversity of mathematical problems addressed, enhancing the general applicability of our methods in ATP.

**中.** 在本文中，我们提出了一种从高中和本科水平的数学竞赛问题生成广泛的综合证明数据的方法。通过将自然语言问题转化为正式陈述、过滤掉低质量问题并使用迭代证明生成，我们创建了 800 万个证明数据点，并在使用这些合成数据进行训练时显着提高了 DeepSeekMath 7B 模型在 ATP 中的性能。我们的模型在 miniF2F 和 FIMO 等基准测试中优于 GPT-4 和其他方法。通过开源我们的数据集和模型，我们的目标是推进自动化定理证明的研究，并增强大型语言模型在形式数学推理中的能力。目前，我们的工作主要集中在中学和本科阶段的代数和数论。在未来的工作中，我们的目标是扩大所解决数学问题的多样性，增强我们的方法在 ATP 中的普遍适用性。

## Broader Impact / 更广泛的影响

**EN.** The research presented in this paper has the potential to significantly advance automated theorem proving by leveraging large-scale synthetic proof data generated from informal mathematical problems. This remarkable advancement can enhance the capabilities of large language models in formal theorem proving, contributing to more reliable mathematical proof verification and providing valuable educational resources for students and researchers. By directly releasing the code, model, and data, we aim to ensure the responsible use of our work, fostering further innovation and maintaining high standards of data privacy and intellectual property compliance.

**中.** 本文提出的研究有可能通过利用非正式数学问题生成的大规模综合证明数据来显着推进自动化定理证明。这一显着的进步可以增强大型语言模型在形式定理证明中的能力，有助于更可靠的数学证明验证，并为学生和研究人员提供宝贵的教育资源。通过直接发布代码、模型和数据，我们的目标是确保负责任地使用我们的工作，促进进一步创新并保持高标准的数据隐私和知识产权合规性。

## References / 参考文献

- Achiam et al. [2023] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 , 2023.

- Avigad [2023] J. Avigad. Mathematics and the formal turn, 2023.

- Azerbayev et al. [2023] Z. Azerbayev, H. Schoelkopf, K. Paster, M. D. Santos, S. McAleer, A. Q. Jiang, J. Deng, S. Biderman, and S. Welleck. Llemma: An open language model for mathematics. arXiv preprint arXiv:2310.10631 , 2023.

- Bansal et al. [2019] K. Bansal, S. Loos, M. Rabe, C. Szegedy, and S. Wilcox. Holist: An environment for machine learning of higher order logic theorem proving. In International Conference on Machine Learning , pages 454–463. PMLR, 2019.

- Bibel [2013] W. Bibel. Automated theorem proving . Springer Science & Business Media, 2013.

- Crouse et al. [2021] M. Crouse, I. Abdelaziz, B. Makni, S. Whitehead, C. Cornelio, P. Kapanipathi, K. Srinivas, V. Thost, M. Witbrock, and A. Fokoue. A deep reinforcement learning approach to first-order logic theorem proving. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 35, pages 6279–6287, 2021.

- De Moura et al. [2015] L. De Moura, S. Kong, J. Avigad, F. Van Doorn, and J. von Raumer. The lean theorem prover (system description). In Automated Deduction-CADE-25: 25th International Conference on Automated Deduction, Berlin, Germany, August 1-7, 2015, Proceedings 25 , pages 378–388. Springer, 2015.

- First et al. [2023] E. First, M. N. Rabe, T. Ringer, and Y. Brun. Baldur: Whole-proof generation and repair with large language models, 2023.

- Han et al. [2021] J. M. Han, J. Rute, Y. Wu, E. W. Ayers, and S. Polu. Proof artifact co-training for theorem proving with language models. arXiv preprint arXiv:2102.06203 , 2021.

- Huang et al. [2024] Y. Huang, X. Lin, Z. Liu, Q. Cao, H. Xin, H. Wang, Z. Li, L. Song, and X. Liang. Mustard: Mastering uniform synthesis of theorem and proof data. arXiv preprint arXiv:2402.08957 , 2024.

- Jiang et al. [2021] A. Q. Jiang, W. Li, J. M. Han, and Y. Wu. Lisa: Language models of isabelle proofs. In 6th Conference on Artificial Intelligence and Theorem Proving , pages 378–392, 2021.

- Jiang et al. [2022a] A. Q. Jiang, W. Li, S. Tworkowski, K. Czechowski, T. Odrzygóźdź, P. Miłoś, Y. Wu, and M. Jamnik. Thor: Wielding hammers to integrate language models and automated theorem provers. Advances in Neural Information Processing Systems , 35:8360–8373, 2022a.

- Jiang et al. [2022b] A. Q. Jiang, S. Welleck, J. P. Zhou, W. Li, J. Liu, M. Jamnik, T. Lacroix, Y. Wu, and G. Lample. Draft, sketch, and prove: Guiding formal theorem provers with informal proofs. arXiv preprint arXiv:2210.12283 , 2022b.

- Jiang et al. [2023] A. Q. Jiang, W. Li, and M. Jamnik. Multilingual mathematical autoformalization. arXiv preprint arXiv:2311.03755 , 2023.

- Kaliszyk et al. [2018] C. Kaliszyk, J. Urban, H. Michalewski, and M. Olšák. Reinforcement learning of theorem proving. Advances in Neural Information Processing Systems , 31, 2018.

- Kovács and Voronkov [2013] L. Kovács and A. Voronkov. First-order theorem proving and vampire. In International Conference on Computer Aided Verification , pages 1–35. Springer, 2013.

- Lample et al. [2022] G. Lample, T. Lacroix, M.-A. Lachaux, A. Rodriguez, A. Hayat, T. Lavril, G. Ebner, and X. Martinet. Hypertree proof search for neural theorem proving. Advances in neural information processing systems , 35:26337–26349, 2022.

- Liu et al. [2023] C. Liu, J. Shen, H. Xin, Z. Liu, Y. Yuan, H. Wang, W. Ju, C. Zheng, Y. Yin, L. Li, et al. Fimo: A challenge formal dataset for automated theorem proving. arXiv preprint arXiv:2309.04295 , 2023.

- Loos et al. [2017] S. Loos, G. Irving, C. Szegedy, and C. Kaliszyk. Deep network guided proof search. arXiv preprint arXiv:1701.06972 , 2017.

- Moura and Ullrich [2021] L. d. Moura and S. Ullrich. The lean 4 theorem prover and programming language. In Automated Deduction–CADE 28: 28th International Conference on Automated Deduction, Virtual Event, July 12–15, 2021, Proceedings 28 , pages 625–635. Springer, 2021.

- Paulson [1994] L. C. Paulson. Isabelle a Generic Theorem Prover . Springer Verlag, 1994.

- Polu and Sutskever [2020] S. Polu and I. Sutskever. Generative language modeling for automated theorem proving. arXiv preprint arXiv:2009.03393 , 2020.

- Polu et al. [2022] S. Polu, J. M. Han, K. Zheng, M. Baksys, I. Babuschkin, and I. Sutskever. Formal mathematics statement curriculum learning. arXiv preprint arXiv:2202.01344 , 2022.

- Radford et al. [2019] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog , 1(8):9, 2019.

- Schulz [2002] S. Schulz. E–a brainiac theorem prover. Ai Communications , 15(2-3):111–126, 2002.

- Shao et al. [2024] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, M. Zhang, Y. Li, Y. Wu, and D. Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 , 2024.

- Shulman [2024] M. Shulman. Strange new universes: Proof assistants and synthetic foundations, 2024.

- Thakur et al. [2023] A. Thakur, Y. Wen, and S. Chaudhuri. A language-agent approach to formal theorem-proving. arXiv preprint arXiv:2310.04353 , 2023.

- [29] The Coq Development Team. Coq. URL [https://coq.inria.fr](https://coq.inria.fr) .

- Vaswani et al. [2017] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems , 30, 2017.

- Wang and Deng [2020] M. Wang and J. Deng. Learning to prove theorems by learning to generate theorems. Advances in Neural Information Processing Systems , 33:18146–18157, 2020.

- Wu et al. [2021] M. Wu, M. Norrish, C. Walder, and A. Dezfouli. Tacticzero: Learning to prove theorems from scratch with deep reinforcement learning. Advances in Neural Information Processing Systems , 34:9330–9342, 2021.

- Wu et al. [2020] Y. Wu, A. Q. Jiang, J. Ba, and R. Grosse. Int: An inequality benchmark for evaluating generalization in theorem proving. arXiv preprint arXiv:2007.02924 , 2020.

- Wu et al. [2022] Y. Wu, A. Q. Jiang, W. Li, M. Rabe, C. Staats, M. Jamnik, and C. Szegedy. Autoformalization with large language models. Advances in Neural Information Processing Systems , 35:32353–32368, 2022.

- Xin et al. [2023] H. Xin, H. Wang, C. Zheng, L. Li, Z. Liu, Q. Cao, Y. Huang, J. Xiong, H. Shi, E. Xie, et al. Lego-prover: Neural theorem proving with growing libraries. arXiv preprint arXiv:2310.00656 , 2023.

- Xiong et al. [2023] J. Xiong, J. Shen, Y. Yuan, H. Wang, Y. Yin, Z. Liu, L. Li, Z. Guo, Q. Cao, Y. Huang, et al. Trigo: Benchmarking formal mathematical proof reduction for generative language models. arXiv preprint arXiv:2310.10180 , 2023.

- Yang et al. [2024] K. Yang, A. Swope, A. Gu, R. Chalamala, P. Song, S. Yu, S. Godil, R. J. Prenger, and A. Anandkumar. Leandojo: Theorem proving with retrieval-augmented language models. Advances in Neural Information Processing Systems , 36, 2024.

- Zhao et al. [2023] X. Zhao, W. Li, and L. Kong. Decomposing the enigma: Subgoal-based demonstration learning for formal theorem proving. arXiv preprint arXiv:2305.16366 , 2023.

- Zheng et al. [2021] K. Zheng, J. M. Han, and S. Polu. Minif2f: a cross-system benchmark for formal olympiad-level mathematics. arXiv preprint arXiv:2109.00110 , 2021.

## Appendix / supplemental material / 附录/补充材料

### Prompts / 提示

**EN.** Specifically, we use the following format for scoring for the quality of the formalized statements:

**中.** 具体来说，我们使用以下格式对形式化陈述的质量进行评分：

**EN.** Prompt : To evaluate whether a formal Lean4 statement will be of interest to the community, consider the following criteria: 1. Relevance to Current Research: Does the statement address a problem or concept that is actively being researched in mathematics or related fields? Higher relevance scores indicate greater potential interest. 2. Complexity and Depth: Is the statement complex enough to challenge existing theories and methodologies, yet deep enough to provide significant insights or advancements? Complexity and depth showcase Lean4’s capabilities and attract interest. 3. Interdisciplinary Potential: Does the statement offer opportunities for interdisciplinary research, connecting mathematics with other fields such as computer science, physics, or biology? Interdisciplinary projects often garner wide interest. 4. Community Needs and Gaps: Does the statement fill an identified need or gap within the Lean4 community or the broader mathematical community? Addressing these needs directly correlates with interest. 5. Innovativeness: How innovative is the statement? Does it propose new methods, concepts, or applications? Innovation drives interest and engagement. Customize your evaluation for each problem accordingly, assessing it as ‘excellent‘, ‘good‘, ‘above average‘, ‘fair‘ or ‘poor‘. You should respond in the following format for each statement: ‘‘‘ Translate the code to natural language: (Detailed explanation of the informal statement, including any relevant background information, assumptions, and definitions.) Analysis: (Provide a brief justification for each score, highlighting why the statement scored as it did across the criteria.) Assessment: (Based on the criteria, rate the statement as ‘excellent’, ‘good‘, ‘above average‘, ‘fair‘ or ‘poor‘.) ‘‘‘

**中.** 提示：要评估正式的 Lean4 声明是否会引起社区的兴趣，请考虑以下标准： 1. 与当前研究的相关性：该声明是否解决了数学或相关领域正在积极研究的问题或概念？相关性分数越高表明潜在兴趣越大。 2. 复杂性和深度：该陈述是否复杂到足以挑战现有的理论和方法，但又足够深入以提供重大见解或进步？复杂性和深度展示了 Lean4 的能力并吸引了人们的兴趣。 3. 跨学科潜力：该声明是否提供了跨学科研究的机会，将数学与计算机科学、物理学或生物学等其他领域联系起来？跨学科项目通常会引起广泛的兴趣。 4. 社区需求和差距：该声明是否填补了 Lean4 社区或更广泛的数学社区内已确定的需求或差距？满足这些需求与兴趣直接相关。 5. 创新性：该声明的创新性如何？它是否提出了新的方法、概念或应用？创新驱动兴趣和参与。相应地定制对每个问题的评估，将其评估为“优秀”、“良好”、“高于平均水平”、“一般”或“差”。您应该对每条语句按照以下格式进行回复： ‘‘’ 将代码翻译成自然语言：（非正式语句的详细解释，包括任何相关的背景信息、假设和定义。） 分析：（为每个分数提供简短的理由，突出显示该语句在标准中得分的原因。） 评估：（根据标准，将语句评级为“优秀”、“良好”、“高于平均水平”、“一般”或“差”。）“”“

### Case Studies of Autoformalization / 自动形式化的案例研究

**EN.** Example a. Problem in Natural Language : For a real number $a$ and a function $f$ defined on real numbers, where $f(x)=x^{3}-ax-1$ , if for all $x$ , $f(x)\leq 0$ implies $x$ is either less than $-1$ or greater than $1$ , then $a$ must equal $3$ .

**中.** 示例a.自然语言中的问题：对于实数 $a$ 和在实数上定义的函数 $f$ ，其中 $f(x)=x^{3}-ax-1$ ，如果对于所有 $x$ ， $f(x)\leq 0$ 意味着 $x$ 小于 $-1$ 或大于 $1$ ，则$a$ 必须等于 $3$ 。

**EN.** Autoformalized Theorems with Complete Proofs :

**中.** 具有完整证明的自动形式化定理：

```text
example (a : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = x ^ 3 - a * x - 1) :
    (∀ x, f x ≤ 0 → x ∈ Set.Iio (-1) ∪ Set.Ioi 1) → a = 3 := by
  intro h₁
  have h₂ := h₁ 0
  simp [h₀] at h₂
  have h₃ := h₁ 1
  simp [h₀] at h₃
  have h₄ := h₁ (-1)
  simp [h₀] at h₄
  linarith
```

**EN.** Analysis: This is a simple example to illustrate common cases in the autoformalization procedure. The model is required to recognize natural language descriptions in the problem statement and formalize them into proper mathematical operators. In this case, the statement " either less than $-1$ or greater than $1$ " describes a union of interval $(-\infty,-1)\cup(1,+\infty)$ , which has been accurately formalized to Set.Iio (-1) $\cup$ Set.Ioi 1 , where Set.Iio (-1) denotes an infinite-open interval ends at $-1$ , and Set.Ioi 1 denotes an open-infinite interval begins at $1$ .

**中.** 分析：这是一个简单的例子，说明自动形式化过程中的常见情况。该模型需要识别问题陈述中的自然语言描述并将其形式化为适当的数学运算符。在本例中，语句“小于 $-1$ 或大于 $1$ ”描述了区间 $(-\infty,-1)\cup(1,+\infty)$ 的并集，它已被精确形式化为 Set.Iio (-1) $\cup$ Set.Ioi 1 ，其中 Set.Iio (-1) 表示无限开区间结束于 $-1$ ，并且Set.Ioi 1 表示从 $1$ 开始的开放无限区间。

**EN.** Example b. Problem in Natural Language: Define two functions, $F$ and $f$ , where $F(k)$ is the binomial coefficient of $(k+2)$ choose 2, and $f(k)$ is the binomial coefficient of $(k+1)$ choose 1. For any natural number $n$ greater than or equal to 2, there exists a natural number $k$ such that $f(k)$ equals $(n-1)$ times $n$ divided by 2. Furthermore, there exists a natural number $m$ such that $F(m)$ equals $(n-1)$ times $n$ divided by 2, implying $m$ equals $k$ .

**中.** 示例 B.自然语言问题：定义两个函数， $F$ 和 $f$ ，其中 $F(k)$ 是 $(k+2)$ 的二项式系数，选择 2，$f(k)$ 是 $(k+1)$ 的二项式系数，选择 1。对于任何大于或等于的自然数 $n$ 2，存在一个自然数$k$，使得$f(k)$等于$(n-1)$乘以$n$除以2。此外，存在一个自然数$m$，使得$F(m)$等于$(n-1)$乘以$n$除以2，这意味着$m$ 等于 $k$ 。

**EN.** Autoformalized Theorems with Complete Proofs:

**中.** 具有完整证明的自动形式化定理：

```text
example :
  let F := fun k => Nat.choose (k + 2) 2;
  let f := fun k => Nat.choose (k + 1) 1;
  ∀ n : ℕ, 2 ≤ n → ∃ k : ℕ, f k = (n - 1) * n / 2 → (∃ m : ℕ, F m = (n - 1) * n / 2 → m = k) := by
  simp [Nat.choose, Nat.mul_sub_left_distrib, Nat.sub_sub, Nat.mul_one]
  aesop
```

**EN.** Analysis: The formalization for this combinatorial problem involving binomial coefficients is semantically faithful and effective. The use of $\mathtt{Nat.choose}$ accurately represents the binomial coefficients, and the logical structure with existential quantifiers ( $\exists$ ) and implications ( $\rightarrow$ ) aligns well with the problem’s requirements.

**中.** 分析：这个涉及二项式系数的组合问题的形式化在语义上是忠实且有效的。 $\mathtt{Nat.choose}$ 的使用准确地表示了二项式系数，并且具有存在量词（ $\exists$ ）和含义（ $\rightarrow$ ）的逻辑结构与问题的要求很好地吻合。

**EN.** Example c. Problem in Natural Language: Given three 3-dimensional vectors $a$ , $b$ , and $c$ , where the dot product of $a$ and $b$ equals $-3$ , the dot product of $a$ and $c$ equals $4$ , and the dot product of $b$ and $c$ equals $6$ . Prove that the dot product of $b$ with $7c-2a$ equals $48$ .

**中.** 示例 C.自然语言中的问题：给定三个3维向量 $a$ 、 $b$ 和 $c$ ，其中 $a$ 和 $b$ 的点积等于 $-3$ ，即 $a$ 和 $c$ 的点积等于 $4$ ，并且 $b$ 和 $c$ 的点积等于 $6$ 。证明 $b$ 与 $7c-2a$ 的点积等于 $48$ 。

**EN.** Autoformalized Theorems with Complete Proofs:

**中.** 具有完整证明的自动形式化定理：

```text
example (a b c : ℝ × ℝ × ℝ)
  (h₀ : a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2 = -3)
  (h₁ : a.1 * c.1 + a.2.1 * c.2.1 + a.2.2 * c.2.2 = 4)
  (h₂ : b.1 * c.1 + b.2.1 * c.2.1 + b.2.2 * c.2.2 = 6)
  : b.1 * (7 * c.1 - 2 * a.1) + b.2.1 * (7 * c.2.1 - 2 * a.2.1) + b.2.2 * (7 * c.2.2 - 2 * a.2.2) = 48 := by
  linarith [h₀, h₁, h₂]
```

**EN.** Analysis: This formalization result uses Cartesian product $\mathbb{R}\times\mathbb{R}\times\mathbb{R}$ to represent a 3-dimensional vector, where the components of a 3-dimensional Cartesian product a are denoted by a.1 , a.2.1 , and a.2.2 in Lean. The notation of dot products is expanded to the sum of element-wise products by definition.

**中.** 分析：该形式化结果使用笛卡尔积 $\mathbb{R}\times\mathbb{R}\times\mathbb{R}$ 来表示 3 维向量，其中 3 维笛卡尔积 a 的分量在 Lean 中表示为 a.1 、 a.2.1 和 a.2.2 。根据定义，点积的表示法被扩展为元素级积的总和。

### Example Lean Proofs Generated by Our Model / 我们的模型生成的精益证明示例

#### Results on MiniF2F-Test Dataset / MiniF2F 测试数据集的结果

**EN.** Example a. Problem in Natural Language: Show that for any real number $x$ and any natural number $n$ , if $x>-1$ , then $(1+nx)\leq(1+x)^{n}$ .

**中.** 示例a.自然语言问题：证明对于任何实数 $x$ 和任何自然数 $n$ ，如果 $x>-1$ ，则 $(1+nx)\leq(1+x)^{n}$ 。

**EN.** Formal Proof:

**中.** 形式证明：

```text
theorem induction_1pxpownlt1pnx (x : ℝ) (n : ℕ) (h₀ : -1 < x) (h₁ : 0 < n)
  : 1 + ↑n * x ≤ (1 + x) ^ (n : ℕ) := by
  induction' h₁ with k h₁ <;> simp_all [Nat.succ_eq_add_one, pow_add, mul_add, add_assoc, add_comm, add_left_comm]
  nlinarith [mul_self_nonneg (1 + x - 1)]
```

**EN.** Example b. Problem in Natural Language: Ms. Blackwell gives an exam to two classes. The mean of the scores of the students in the morning class is $84$ , and the afternoon class’s mean score is $70$ . The ratio of the number of students in the morning class to the number of students in the afternoon class is $\frac{3}{4}$ . What is the mean of the scores of all the students? Show that it is 76.

**中.** 示例 B.自然语言问题：布莱克威尔女士对两个班级进行了考试。上午班学生的平均成绩为 $84$ ，下午班学生的平均成绩为 $70$ 。上午班学生人数与下午班学生人数之比为$\frac{3}{4}$。所有学生成绩的平均值是多少？显示它是76。

**EN.** Formal Proof:

**中.** 形式证明：

```text
theorem amc12b_2021_p4 (m a : ℕ) (h₀ : 0 < m ∧ 0 < a)
  (h₁ : ↑m / ↑a = (3 : ℝ) / 4)
  : (84 * ↑m + 70 * ↑a) / (↑m + ↑a) = (76 : ℝ) := by
  have h₂ := h₀.1.ne'
  have h₃ := h₀.2.ne'
  field_simp at h₂ h₃ ⊢
  ring_nf
  norm_num
  rw [div_eq_inv_mul] at h₁
  field_simp at h₁
  linarith
```

**EN.** Example c. Problem in Natural Language: For how many positive integers $m$ does there exist at least one positive integer n such that $m\cdot n\leq m+n$ ? Show that it is infinitely many.

**中.** 示例 C.自然语言问题： 对于多少个正整数 $m$ ，是否存在至少一个正整数 n 使得 $m\cdot n\leq m+n$ ？证明它有无穷多个。

**EN.** Formal Proof:

**中.** 形式证明：

```text
theorem amc12a_2002_p6 (n : ℕ) (h₀ : 0 < n)
  : ∃ m, m > n ∧ ∃ p, m * p ≤ m + p := by
  simp_all only [ge_iff_le, gt_iff_lt, mul_one, mul_add, mul_comm, mul_assoc, mul_left_comm]
  use n + 1
  constructor
  exact Nat.lt_succ_self n
  use 1
  ring_nf
  nlinarith
```

#### Results on FIMO Dataset / FIMO 数据集结果

**EN.** Example a. Problem in Natural Language: Determine all functions $f$ from the set of positive integers into the set of positive integers such that for all $x$ and $y$ there exists a non degenerated triangle with sides of lengths $x$ , $f(y)$ and $f(y+f(x)-1)$ . The final answer is the identity function $f(x)=x$ .

**中.** 示例a.自然语言中的问题：确定从正整数集合到正整数集合的所有函数 $f$ ，使得对于所有 $x$ 和 $y$ ，存在一个边长为 $x$ 、 $f(y)$ 和 $f(y+f(x)-1)$ 的非退化三角形。最终答案是恒等函数 $f(x)=x$ 。

**EN.** Formal Proof:

**中.** 形式证明：

```text
theorem fimo_2009_algebra_p3
  (f : ℕ → ℕ)
  (h₀ : ∀ x y, ∃ (a b c : ℕ),
    a = x ∧
    b = f y ∧
    c = f (y + f x - 1) ∧
    a + b > c ∧
    a + c > b ∧
    b + c > a) :
  ∀ x, f x = x := by
  have h₁ : f 0 = 0 := by
    specialize h₀ 0 0
    obtain ⟨_, _, _, h₁, h₂, h₃⟩ := h₀
    simp at *
    linarith
  intro x
  induction x with
  | zero => exact h₁
  | succ x ih =>
    specialize h₀ (x + 1) 0
    obtain ⟨_, _, _, h₁, h₂, h₃⟩ := h₀
    simp at *
    linarith
```

**EN.** Example b. Problem in Natural Language: Prove that for every positive integer $n$ , there exists a fraction $\frac{a}{b}$ where $a$ and $b$ are integers satisfying $0<b\leqslant\sqrt{n}+1$ and $\sqrt{n}\leqslant\frac{a}{b}\leqslant\sqrt{n+1}$ .

**中.** 示例 B.自然语言问题：证明对于每个正整数 $n$ ，都存在一个分数 $\frac{a}{b}$ ，其中 $a$ 和 $b$ 是满足 $0<b\leqslant\sqrt{n}+1$ 和 $\sqrt{n}\leqslant\frac{a}{b}\leqslant\sqrt{n+1}$ 的整数。

**EN.** Formal Proof:

**中.** 形式证明：

```text
theorem fimo_2016_algebra_p5_1
  (n : ℕ)
  (h₀ : 0 < n) :
  ∃ a b : ℕ, 0 < b ∧ b ≤ n.sqrt + 1 ∧
    n.sqrt ≤ a / b ∧ a / b ≤ (n + 1).sqrt := by
  use n.sqrt, 1
  constructor <;>
  simp [h₀, Nat.sqrt_le_sqrt] <;>
  norm_num <;>
  linarith [Nat.sqrt_le_sqrt n]
```

### Formal Verification Details / 形式验证细节

**EN.** We verify the generated Lean 4 code with the following code as the prefix:

**中.** 我们验证生成的Lean 4代码，以下代码为前缀：

```text
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Algebra.Order.Floor
import Mathlib.Algebra.Associated
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.BigOperators.Pi
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Pi.Basic
import Mathlib.Algebra.Group.Commute.Basic
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.GroupPower.Identities
import Mathlib.Algebra.Order.Floor
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Asymptotics.AsymptoticEquivalent
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.GCD
import Mathlib.Data.Int.ModEq
import Mathlib.Data.Int.Parity
import Mathlib.Data.List.Intervals
import Mathlib.Data.List.Palindrome
import Mathlib.Data.Multiset.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Nat.Multiplicity
import Mathlib.Data.Nat.Parity
import Mathlib.Data.Nat.Prime
import Mathlib.Data.PNat.Basic
import Mathlib.Data.PNat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Polynomial.Eval
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Irrational
import Mathlib.Data.Real.NNReal
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Finite
import Mathlib.Data.Sym.Sym2
import Mathlib.Data.ZMod.Basic
import Mathlib.Dynamics.FixedPoints.Basic
import Mathlib.LinearAlgebra.AffineSpace.AffineMap
import Mathlib.LinearAlgebra.AffineSpace.Independent
import Mathlib.LinearAlgebra.AffineSpace.Ordered
import Mathlib.LinearAlgebra.FiniteDimensional
import Mathlib.Logic.Equiv.Basic
import Mathlib.Order.Filter.Basic
import Mathlib.Order.LocallyFinite
import Mathlib.Order.WellFounded
import Mathlib.Topology.Basic
import Mathlib.Topology.Instances.NNReal
import Aesop

set_option maxHeartbeats 0
set_option trace.aesop true
set_option trace.aesop.proof true

open Nat Real Rat BigOperators
```

## Footnotes / 脚注

[^id1]: **EN.** Corresponding author.<br>**中.** 通讯作者.

[^footnote1]: **EN.** $\mathtt{leanprover/lean4:v4.7.0-rc2}$<br>**中.** $\mathtt{leanprover/lean4:v4.7.0-rc2}$

[^footnote2]: **EN.** The specific mathlib commit used is $\mathtt{64528268b3c2cf578639bc479828882a9ecd3a82}$ .<br>**中.** 使用的具体 mathlib 提交是 $\mathtt{64528268b3c2cf578639bc479828882a9ecd3a82}$ 。
