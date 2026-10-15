# PLAID: An Efficient Engine for Late Interaction Retrieval

> Keshav Santhanam, Omar Khattab, Christopher Potts, Matei Zaharia. arXiv:2205.09707v1 (2022-05-19).

> Source: https://arxiv.org/html/2205.09707v1 · License: https://creativecommons.org/licenses/by/4.0/

> HTML-to-Markdown conversion: semantic headings, normalized tables and composed figure panels. Original equations and source inconsistencies are preserved; see the Chinese study notes for dimensional corrections.

## Abstract.

<!-- abstract1.1 -->
Pre-trained language models are increasingly important components across multiple information retrieval (IR) paradigms. Late interaction, introduced with the ColBERT model and recently refined in ColBERTv2, is a popular paradigm that holds state-of-the-art status across many benchmarks. To dramatically speed up the search latency of late interaction, we introduce the Performance-optimized Late Interaction Driver (PLAID). Without impacting quality, PLAID swiftly eliminates low-scoring passages using a novel centroid interaction mechanism that treats every passage as a lightweight bag of centroids. PLAID uses centroid interaction as well as centroid pruning, a mechanism for sparsifying the bag of centroids, within a highly-optimized engine to reduce late interaction search latency by up to 7 $\times$ on a GPU and 45 $\times$ on a CPU against vanilla ColBERTv2, while continuing to deliver state-of-the-art retrieval quality. This allows the PLAID engine with ColBERTv2 to achieve latency of tens of milliseconds on a GPU and tens or just few hundreds of milliseconds on a CPU at large scale, even at the largest scales we evaluate with 140M passages.

## Introduction

<!-- S1.p1.1 -->
Recent advances in neural information retrieval (IR) have led to notable gains on retrieval benchmarks and retrieval-based NLP tasks. Late interaction, introduced in ColBERT ( Khattab and Zaharia, 2020 ) , is a paradigm that delivers state-of-the-art quality in many of these settings, including passage ranking ( Santhanam et al., 2021 ; Hofstätter et al., 2022 ; Wang et al., 2021 ) , open-domain question answering ( Khattab et al., 2021b ; Li et al., 2021 ) , conversational tasks ( Paranjape et al., 2022 ; Minas Krasakis et al., 2022 ) , and beyond ( Khattab et al., 2021a ; Zhong et al., 2022 ) . ColBERT and its variants encode queries and documents into token-level vectors and conduct scoring via scalable yet fine-grained interactions at the level of tokens (Figure 1 ), alleviating the dot-product bottleneck of single-vector representations. The recent ColBERTv2 ( Santhanam et al., 2021 ) model demonstrates that late interaction models often considerably outperform recent single-vector and sparse representations within and outside the training domain, a finding echoed in several recent studies ( Lupart and Clinchant, 2022 ; Zhan et al., 2022 ; Thakur et al., 2021 ; Macdonald et al., 2021 ; Zeng et al., 2022 ; Thai et al., 2022 ) .

<!-- S1.p2.1 -->
Despite its strong retrieval quality, late interaction requires special infrastructure ( Khattab and Zaharia, 2020 ; Lin, 2022 ) for low-latency retrieval as it encodes each query and each document as a full matrix . Most IR models represent documents as a single vector, either sparse (e.g., BM25 ( Robertson et al., 1995 ) ; SPLADE ( Formal et al., 2021b ) ) or dense (e.g., DPR ( Karpukhin et al., 2020 ) ; ANCE ( Xiong et al., 2020 ) ), and thus mature sparse retrieval strategies like WAND ( Broder et al., 2003 ) or dense kNN methods like HNSW ( Malkov and Yashunin, 2018 ) cannot be applied directly or optimally to late interaction. While recent work ( Macdonald and Tonellotto, 2021 ; Tonellotto and Macdonald, 2021 ; Santhanam et al., 2021 ) has explored optimizing individual components of ColBERT’s pipeline, an end-to-end optimized engine has never been studied to our knowledge.

<!-- S1.p3.1 -->
We study how to optimize late-interaction search latency at a large scale, taking all steps of retrieval into account. We build on the state-of-the-art ColBERTv2 model. Besides improving quality with denoised supervision, ColBERTv2 aggressively compresses the storage footprint of late interaction. It reduces the index size by up to an order of magnitude using residual representations (§ 3.1 ). In those, each vector in a passage is encoded using the ID of its nearest centroid that approximates its token semantics—among tens or hundreds of thousands of centroids obtained through $k$ -means clustering—and a quantized residual vector.

<!-- S1.p4.1 -->
We introduce the P erformance-optimized La te I nteraction D river (PLAID), an efficient retrieval engine that reduces late interaction search latency by 2.5–7 $\times$ on GPU and 9–45 $\times$ on CPU against vanilla ColBERTv2 while retaining high quality. This allows the PLAID implementation of ColBERTv2, PLAID ColBERTv2, to achieve CPU-only latency of tens or just few hundreds of milliseconds and GPU latency of few tens of milliseconds at very large scale, even on 140M passages. Crucially, PLAID ColBERTv2 does so while continuing to deliver state-of-the-art retrieval quality.

<!-- S1.p5.1 -->
To dramatically speed up search, PLAID leverages the centroid component of the ColBERTv2 representations, which is a compact integer ID per token. Instead of exhaustively scoring all passages found with nearest-neighbor search, PLAID uses the centroids to identify high-scoring passages and eliminate weaker candidates without loading their larger residuals. We conduct this in a multi-stage pipeline and introduce centroid interaction , a scoring mechanism that treats every passage as a lightweight bag of centroid IDs . We show that this centroid-only multi-vector search exhibits high recall without using the vector residuals (§ 3.3 ), allowing us to reserve full scoring to a very small number of candidate passages. Because the centroids come from a fixed set (i.e., constitute a discrete vocabulary), the distance between the query vectors and all centroids can be computed once during search and re-used across all bag-of-centroids passage representations. This allows us to further leverage the centroid scores for centroid pruning , which sparsifies the bag of centroid representations in the earlier stages of retrieval by skipping centroid IDs that are distant from all query vectors.

<!-- S1.p6.1 -->
In the PLAID engine, we implement centroid interaction and centroid pruning and implement optimized yet modular kernels for the data movement, decompression, and scoring components of late interaction with the residual representations of ColBERTv2 (§ 4.5 ). We extensively evaluate the quality and efficiency of PLAID within and outside the training domain (on MS MARCO v1 ( Nguyen et al., 2016 ) and v2 ( Craswell et al., 2022 ) , Wikipedia, and LoTTE ( Santhanam et al., 2021 ) ) and across a wide range of corpus sizes (2M–140M passages), search depths ( $k$ =10, 100, 1000), and hardware settings with single- and multi-threaded CPU and with a GPU (§ 5.2 ). We also conduct a detailed ablation study to understand the empirical sources of gains among centroid interaction, centroid pruning, and our faster kernels (§ 5.3 ).

<!-- S1.p7.1 -->
In summary, we make the following contributions:

<!-- S1.I1.i1.p1.1 -->
We analyze centroid-only retrieval with ColBERTv2, showing that a pruned bag-of-centroids representation can support high-recall candidate generation (§ 3 ).

<!-- S1.I1.i2.p1.1 -->
We propose PLAID, a retrieval engine that introduces centroid interaction and centroid pruning as well as optimized implementations of these techniques for dramatically improving the latency of late-interaction search (§ 4 ).

<!-- S1.I1.i3.p1.1 -->
We extensively evaluate PLAID and conduct a large-scale evaluation up to 140M passages, the largest to our knowledge with late-interaction retrievers (§ 5 ).

![Figure 1](figs/fig01.png)

Figure 1 . The late interaction architecture, given a query and a passage. Diagram from Khattab
et al. (2021b) with permission.

## Related Work

### Neural IR

<!-- S2.SS1.p1.1 -->
The IR community has introduced many neural IR models based on pre-trained Transformers. Whereas early models were primarily cross-encoders ( Nogueira and Cho, 2019 ; MacAvaney et al., 2019 ) that attend jointly to queries and passages, many subsequent models target higher efficiency by producing independent representations for queries and passages. Some of those produce sparse term weights ( Dai and Callan, 2020 ; Mallia et al., 2021 ) , whereas others encode each passage or query into a single vector ( Xiong et al., 2020 ; Karpukhin et al., 2020 ; Qu et al., 2021 ) or multi-vector representation (the class we study; ( Khattab and Zaharia, 2020 ; Santhanam et al., 2021 ; Khattab et al., 2021b ; Humeau et al., 2020 ; Gao et al., 2021 ) ). These choices make different tradeoffs about efficiency and quality: whereas sparse term weights and single-vector models can be particularly lightweight in some settings, multi-vector late interaction ( Khattab and Zaharia, 2020 ) can often result in considerably stronger quality and robustness. Orthogonal to the choice of modeling query–document interactions, researchers have improved the supervision for neural models with harder negatives ( Khattab et al., 2021b ; Xiong et al., 2020 ; Zhan et al., 2020 ) as well as distillation and denoising ( Qu et al., 2021 ; Ren et al., 2021 ; Hofstätter et al., 2020 ) , among other approaches. Our work extends ColBERTv2 ( Santhanam et al., 2021 ) , which combines late interaction modeling with hard negative and denoising supervision to achieve state-of-the-art quality among standalone retrievers.

![Figure 2](figs/fig02.png)

Figure 2 . Latency breakdown of MS MARCO v1 dev queries run with vanilla ColBERTv2 and PLAID ColBERTv2 on a TITAN V GPU. Vanilla ColBERTv2 is overwhelmingly bottlenecked with the cost of index lookup and decompression, a challenge that PLAID addresses.

![Figure 3](figs/fig03.png)

Figure 3 . Recall of passages retrieved by a centroid-only version of ColBERTv2 with respect to the top $k$ passages retrieved by vanilla ColBERTv2. Centroids alone can identify virtually all of the top- $k$ passages retrieved with the full ColBERTv2 pipeline, within $10\cdot k$ or fewer candidates, motivating our centroid interaction strategy.

### Pruning for Sparse and Dense Retrieval

<!-- S2.SS2.p1.1 -->
For sparse retrieval models, traditional IR has a wealth of work on fast strategies for skipping documents for top- $k$ search. Strategies often keep metadata like term score upper bounds to skip lower-scoring candidates and most follow a Document-At-A-Time (DAAT) scoring approach ( Turtle and Flood, 1995 ; Broder et al., 2003 ; Ding and Suel, 2011 ; Dimopoulos et al., 2013 ; Mallia et al., 2017 ; Khattab et al., 2020 ) . Refer to Tonellotto et al. (2018) for a detailed treatment of recent methods. A key difference to our settings is that these all strategies expect a set of precomputed scores (particularly, useful upper bounds on every term–document pair), whereas with late interaction the term–document interaction (i.e., the MaxSim score) is only known at query time after a matrix-vector multiplication. Our observations about the utility of centroids for accelerating late interaction successfully moves the problem closer to classical IR, but poses the challenge that the query-to-centroid scores are only known at query time.

<!-- S2.SS2.p2.1 -->
For dense retrieval models that use single-vector representations, approximate $k$ -nearest neighbor (ANN) search is a well-studied problem ( Jegou et al., 2010 ; Johnson et al., 2019 ; Abuzaid et al., 2019 ; Malkov and Yashunin, 2018 ) . Our focus extends such work from a single vector to the late interaction of two matrices.

## Analysis of ColBERTv2 Retrieval

<!-- S3.p1.1 -->
We begin by a preliminary investigation of the latency (§ 3.2 ) and scoring patterns (§ 3.3 ) of ColBERTv2 retrieval that motivates our work on PLAID. To make this section self-contained, § 3.1 reviews the modeling, storage, and supervision of ColBERTv2.

### Modeling, Storage, and Retrieval

<!-- S3.SS1.p1.1 -->
PLAID optimizes retrieval for models using the late interaction architecture of ColBERT, which includes systems like ColBERTv2, Baleen ( Khattab et al., 2021a ) , Hindsight ( Paranjape et al., 2022 ) , and DrDecr ( Li et al., 2021 ) , among others. As depicted in Figure 1 , a Transformer encodes queries and passages independently into vectors at the token level. For scalability, passage representations are pre-computed offline. At search time, the similarity between a query $q$ and a passage $d$ is computed as the summation of “MaxSim” operations, namely, the largest cosine similarity between each vector in the query matrix and all of the passage vectors:

<!-- equation 1 -->
$$
S_{q,d}=\sum_{i=1}^{|Q|}\max_{j=1}^{|D|}Q_{i}\cdot D_{j}^{T}
$$

Equation (1).

<!-- S3.SS1.p1.2 -->
where $Q$ and $D$ are the matrix representations of the query and passage, respectively. In doing so, this scoring function aligns each query token with the “most similar” passage token and estimates relevance as the sum of these term-level scores. Refer to Khattab and Zaharia (2020) for a more complete discussion of late interaction.

<!-- S3.SS1.p2.1 -->
For storing the passage representations, we adopt the ColBERTv2 residual compression strategy, which reduces the index size by up to an order of magnitude over naive storage of late-interaction embeddings as vectors of 16-bit floating-point numbers. Instead, ColBERTv2’s compression strategy efficiently clusters all token-level vectors and encodes each vector using the ID of its nearest cluster centroid as well as a quantized residual vector, wherein each dimension is 1- or 2-bit encoding of the delta between the centroid and the original uncompressed vector. Decompressing a vector requires locating its centroid ID, encoded using 4 bytes, and its residual, which consume 16 or 32 bytes for 1- or 2-bit residuals, assuming the default 128-dimensional vectors.

<!-- S3.SS1.p3.1 -->
While we adopt ColBERTv2’s compression, we improve its retrieval strategy. We refer to the original retrieval strategy as “vanilla” ColBERTv2 retrieval. We refer to Santhanam et al. (2021) for details of compression and retrieval in ColBERTv2.

### ColBERTv2 Latency Breakdown

<!-- S3.SS2.p1.1 -->
Figure 2 presents a breakdown of query latency on MS MARCO Passage Ranking (v1) on a GPU, showing results for vanilla ColBERTv2 (Figure 2(a) ) against the new PLAID ColBERTv2 (Figure 2(b) ). Latency is divided between query encoding, candidate generation, index lookups (i.e., to gather the compressed vector representations for candidate passages), residual decompression, and finally scoring (i.e., the final MaxSim computations).

<!-- S3.SS2.p2.1 -->
For vanilla ColBERTv2, index lookup and residual decompression are overwhelming bottlenecks. Gathering vectors from the index is expensive because it consumes significant memory bandwidth: each vector in this setting is encoded with a 4-bit centroid ID and 32-byte residuals, each passage contains tens of vectors, and there can be up to $2^{16}$ candidate passages. Moreover, index lookup in vanilla ColBERTv2 also constructs padded tensors on the fly to deal with the variable length of passages. Decompression of residuals is comprised of several non-trivial operations such as unpacking bits and computing large sums, which can be expensive when ColBERTv2 produces a large initial candidate set (~10-40k passages) as is the case for MS MARCO v1. While it is possible to use a smaller candidate set, doing so reduces recall (§ 5 ).

![Figure 4](figs/fig04.png)

Figure 4 . Centroid score distribution for each query among a random sample of 15 MS MARCO v1 dev queries evaluated with ColBERTv2.

### Centroids Alone Identify Strong Candidates

<!-- S3.SS3.p1.1 -->
This breakdown in Figure 2(b) demonstrates that exhaustively scoring a large number of candidates passages, particularly gathering and decompressing their residuals, can amount to a considerable cost. Whereas ColBERTv2 ( Santhanam et al., 2021 ) exploits centroids to reduce the space footprint, our work demonstrates that the centroids can also accelerate search , while maintaining quality, by serving as proxies for the passage embeddings. Because of this, we can skip low-scoring passages without having to look up or decompress their residuals, adding some additional candidate generation overhead to achieve substantial savings in the subsequent stages (Figure 2(b) ).

<!-- S3.SS3.p2.1 -->
Effectively, we hypothesize that centroid-only retrieval can find the high-scoring passages otherwise retrieved by vanilla ColBERTv2. We test this hypothesis by comparing the top- $k$ passages retrieved by vanilla ColBERTv2 to a modified implementation that conducts retrieval using only the centroids and no residuals. We present the results in Figure 3 . At $k\in\{10,100,1000\}$ , the figure plots the average recall of the top- $k$ passages of vanilla ColBERTv2 within the passages retrieved by centroid-only ColBERTv2 at various depths. In other words, we report the fraction of the top- $k$ passages of vanilla ColBERTv2 that appear within the top- $k^{\prime}$ passages of centroid-only ColBERTv2, for $k^{\prime}\geq k$ .

<!-- S3.SS3.p3.1 -->
The results support our hypothesis, both in domain for MS MARCO v1 and out of domain using the LoTTE Pooled (dev) search queries ( Santhanam et al., 2021 ) . For instance, if we retrieve $10\cdot k$ passages using only centroids, those $10\cdot k$ passages still contain 99+% of the top $k$ passages retrieved by the vanilla ColBERTv2 full pipeline.

### Not All Centroids Are Important Per Query

<!-- S3.SS4.p1.1 -->
We further hypothesize that for a given query a small subset of the passage embedding clusters tend to be far more important than others in determining relevance. If this were in fact the case, then we could prioritize computation over these highly weighted centroids and discard the rest since we know they will not contribute significantly to the final ranking. We test this theory by randomly sampling 15 MS MARCO v1 queries and plotting an empirical CDF of each centroid’s maximum relevance score observed across all query tokens, as shown in Figure 4 . We do find that there is a small tail of highly weighted centroids whose relevance scores have far higher magnitude than all other centroids. While not shown in Figure 4 , we also repeated this experiment with LoTTE pooled queries and found a very similar score distribution.

## PLAID

![Figure 5](figs/fig05.png)

Figure 5 . The PLAID scoring pipeline. The first stage generates an initial set of candidate passages using the centroids. Next the second and third stages leverage centroid pruning and centroid interaction respectively to refine the candidate set. Then the last stage performs full residual decompression to obtain the final passage ranking. We use the hyperparameter ndocs to specify the number of candidates returned by Stage 2, and in our experiments we have Stage 3 output $\frac{\texttt{ndocs}}{4}$ passages.

<!-- S4.p1.1 -->
Figure 5 illustrates the PLAID scoring pipeline, which consists of multiple consecutive stages for retrieval, filtering, and ranking. The first stage produces an initial candidate set by computing relevance scores for each centroid with respect to the query embeddings. In the intermediate stages, PLAID uses the novel techniques of centroid interaction and centroid pruning to aggressively yet effectively filter the candidate passages. Finally, PLAID ranks the final candidate set using fully reconstructed passage embeddings. We discuss each of these modules in more depth as follows.

### Candidate Generation

<!-- S4.SS1.p1.1 -->
Given the query embedding matrix $Q$ and the list of centroid vectors $C$ in the index, PLAID computes the token-level query–centroid relevance scores $S_{c,q}$ as a matrix multiplication:

<!-- equation 2 -->
$$
S_{c,q}=C\cdot Q^{T}\\

$$

Equation (2).

<!-- S4.SS1.p1.2 -->
and then identifies the passages “close” to the top- $t$ centroids per query token as the initial candidate set. A passage is close to a centroid iff one or more of its tokens are assigned to that centroid by $k$ -means clustering during indexing. This value $t$ is referred to as nprobe in vanilla ColBERTv2 and we retain that terminology in PLAID ColBERTv2.

<!-- S4.SS1.p2.1 -->
The initial candidate generation in PLAID ColBERTv2 differs from the corresponding vanilla ColBERTv2 stage in two key aspects. First, while vanilla ColBERTv2 saves an inverted list mapping centroids to their corresponding embedding IDs, PLAID ColBERTv2 instead structures the inverted list as a map from centroids to the corresponding unique passage IDs . Storing passage IDs is advantageous over storing embedding IDs since there are far fewer passages than embeddings, meaning the inverted list has to store less information overall. This also enables PLAID ColBERTv2 to use 32-bit integers in the inverted list rather than potentially 64-bit longs. In practice, this translates to a space savings of 2.7 $\times$ in the MS MARCO v2 ( Craswell et al., 2022 ) inverted list (71 GB to 27 GB, with 140M passages).

<!-- S4.SS1.p3.1 -->
Second, and relatedly, if the initial candidate set was too large (as specified by the ncandidates hyperparameter) vanilla ColBERTv2 would prune it by scoring and ranking a subset of the candidate embedding vectors—in particular, the embeddings listed within the vanilla mapping from centroid IDs to embedding IDs—with full residual decompression, which is quite costly as we discuss in § 3.2 . In contrast, PLAID ColBERTv2 does not impose any limit on the initial candidate size because the subsequent stages can cheaply filter the candidate passages with centroid interaction and pruning.

### Centroid Interaction

<!-- S4.SS2.p1.1 -->
Centroid interaction cheaply approximates per-passage relevance by substituting each token’s embedding vector with its nearest centroid in the standard MaxSim formulation. By applying centroid interaction as an additional filtering stage, the scoring pipeline can skip the expensive embedding reconstruction process for a large fraction of the candidate passages. This results in significantly faster end-to-end retrieval. Intuitively, centroid interaction enables PLAID to emulate traditional bag-of-words retrieval wherein the centroid relevance scores take the role of the term relevance scores used in systems like BM25. However, because of its vector representations (of the query in particular), PLAID computes the centroid relevance scores at query time in contrast to the more traditional pre-computed term relevance scores.

<!-- S4.SS2.p2.1 -->
The procedure works as follows. Recall that $S_{c,q}$ from Equation 2 stores the relevance scores for each centroid with respect to the query tokens. Suppose $I$ is the list of the centroid indices mapped to each of the tokens in the candidate set. Furthermore, let $S_{c,q}[i]$ denote the $i$ -th row of $S_{c,q}$ . Then PLAID constructs the centroid-based approximate scores $\tilde{D}$ as

<!-- equation 3 -->
$$
\tilde{D}=\begin{bmatrix}S_{c,q}[I_{1}]\\
S_{c,q}[I_{2}]\\
\cdots\\
S_{c,q}[I_{|\tilde{D}|}]\end{bmatrix}
$$

Equation (3).

<!-- S4.SS2.p2.2 -->
Then to rank the candidate passages using $\tilde{D}$ , PLAID computes the MaxSim scores $S_{\tilde{D}}$ as

<!-- equation 4 -->
$$
S_{\tilde{D}}=\sum_{i}^{|Q|}\max_{j=1}^{|\tilde{D}|}\tilde{D}_{i,j}
$$

Equation (4).

<!-- S4.SS2.p2.3 -->
The top $k$ most relevant passages drawn from $S_{\tilde{D}}$ serve as the filtered candidate passage set.

<!-- S4.SS2.p3.1 -->
PLAID includes optimized kernels to efficiently deploy centroid interaction (and more generally MaxSim operations); we discuss these in § 4.5 .

### Centroid Pruning

<!-- S4.SS3.p1.1 -->
As an additional optimization, PLAID leverages the observation from § 3.3 to first prune low-magnitude centroid scores before constructing $\tilde{D}$ . In this filtering phase PLAID will only score tokens whose maximum corresponding centroid score meets the given threshold $t_{cs}$ . Concretely, $\tilde{D}$ will only be comprised of tokens whose corresponding centroid (suppose centroid $i$ ) meets the following condition:

<!-- equation 5 -->
$$
\max_{j=1}^{|Q|}S_{c,q_{i,j}}\geq t_{cs}
$$

Equation (5).

<!-- S4.SS3.p2.1 -->
We introduce the hyperparameter ndocs to refer to the number of candidate documents selected by Stage 2. We then found empirically that choosing $\frac{\texttt{ndocs}}{4}$ candidates from Stage 3 produced good results; we use this heuristic for all the results presented in § 5 .

### Scoring

<!-- S4.SS4.p1.1 -->
As in vanilla ColBERTv2, PLAID will reconstruct the original embeddings of the final candidate passage set via residual decompression and rank these using MaxSim. Let $D$ be the reconstructed embedding vectors for the final candidate set after decompression. Then the final scores $S_{q,d}$ are computed using Equation 1 .

<!-- S4.SS4.p2.1 -->
Section § 4.5 discusses fast kernels for accelerating the MaxSim and decompression operations.

### Fast Kernels: Padding-Free MaxSim & Optimized Decompression

<!-- S4.SS5.p1.1 -->
Figure 2(a) shows that index lookup operations are a large source of overhead for vanilla ColBERTv2. One reason these lookups are expensive is that they require reshaping and padding the 2D index tensors with an extra dimension representing the maximum passage length. The resulting 3D tensors facilitate batched MaxSim operations over ragged lists of token vectors. To avoid this padding, we instead implement custom C++ code that directly computes the MaxSim scores over the packed 2D index tensors (i.e., one where many 2D sub-tensors of various lengths are concatenated along the same dimension). Our kernel loops over each passage’s corresponding token vectors to compute the per-passage maximum scores with respect to each query token and then sums the per-passage maximum scores across all query tokens. This design is trivial to parallelize across passages, and also enables $O(|Q|)$ per-thread memory usage by allocating a single output vector to store the maximum scores per query token and repeatedly updating this vector in-place. In contrast, the padding-based approach requires $O(|D|\cdot|Q|)$ space. We have incorporated this design into optimized implementations of centroid interaction as well as the final MaxSim operation (stage 4 in Figure 5 ). PLAID only implements these kernels for CPU execution. Adding corresponding GPU kernels remains future work.

<!-- S4.SS5.p2.1 -->
ColBERTv2’s residual decompression scheme computes a list of centroid vectors, determines a fixed set of $2^{b}$ possible deltas from these centroids, and then stores the index into the set of deltas corresponding to each embedding vector. In particular, each compressed 8-bit value stores $\frac{8}{b}$ indices in the range $[0,2^{b})$ . ColBERTv2 incurs significant overhead due to residual decompression, as shown in Figure 2(a) . This is partially due to the naïve decompression implementation, which required explicitly unpacking bits from the compressed representation and performing expensive bit shift and sum operations to recover the original values. Instead, PLAID pre-computes all $2^{8}$ possible lists of indices encoded by an 8-bit packed value. These outputs are stored in a lookup table so that the decompression function can simply retrieve the indices from the table rather than manually unpacking the bits. We include optimized implementations of this lookup-based decompression for both CPU and GPU execution. The GPU implementation uses a custom CUDA kernel that allocates a separate thread to decompress each individual byte in the compressed residual tensor (the thread block size is computed as $\frac{b\cdot d}{8}$ for $d$ -dimensional embedding vectors). The CPU implementation instead parallelizes decompression at the granularity of individual passages.

## Evaluation

<!-- S5.p1.1 -->
Our evaluation seeks to answer the following research questions:

<!-- S5.I1.i1.p1.1 -->
How does PLAID affect end-to-end latency and retrieval quality across IR benchmarks? (§ 5.2 )

<!-- S5.I1.i2.p1.1 -->
How much do each of PLAID’s optimizations contribute to the performance speedups? (§ 5.3 )

<!-- S5.I1.i3.p1.1 -->
How well does PLAID scale with respect to the corpus size and the parallelism degree? (§ 5.4 )

### Setup

<!-- table 1 -->
| Dataset | Passages | Tokens | Queries | Vanilla index GiB | PLAID index GiB |
| --- | --- | --- | --- | --- | --- |
| MS MARCO v1 | 8.8M | 597.9M | 6980 | 24.6 | 21.6 |
| Wikipedia | 21.0M | 2.6B | 8757 | 105.2 | 92.0 |
| LoTTE pooled | 2.4M | 339.4M | 2931 | 14.0 | 12.3 |
| MS MARCO v2 | 138.4M | 9.4B | 3903 | 246.0 | 202.2 |

Table 1 . List of benchmarks used for evaluation with relevant statistics.

<!-- table 2 -->
| k | nprobe | t_cs | ndocs |
| --- | --- | --- | --- |
| 10 | 1 | 0.5 | 256 |
| 100 | 2 | 0.45 | 1024 |
| 1000 | 4 | 0.4 | 4096 |

Table 2 . PLAID hyperparameter configuration.

### PLAID Implementation

<!-- S5.SS1.SSS0.Px1.p1.1 -->
The PLAID engine subsumes centroid interaction as well as optimizations for residual decompression. We implement PLAID modularly as an extension to ColBERTv2’s PyTorch-based implementation, particularly its search components. For CPU execution, we implement the centroid interaction and decompression operations entirely in multithreaded C++ code. For GPUs, we implement centroid interaction in PyTorch and provide a CUDA kernel for fast decompression. Overall, PLAID constitutes roughly 300 lines of additional Python code and 700 lines of C++.

### Datasets

<!-- S5.SS1.SSS0.Px2.p1.1 -->
Our evaluation includes results from four different IR benchmarks, as listed in Table 1 . We perform in-domain evaluation on the MS MARCO v1 and Wikipedia Open QA benchmarks, with retrievers trained specifically for these tasks, and out-of-domain evaluation on the StackExchange-based LoTTE Santhanam et al. (2021) and the TREC 2021 Deep Learning Track ( Craswell et al., 2022 ) MS MARCO v2 benchmarks, with the ColBERTv2 retriever ( Santhanam et al., 2021 ) trained on MS MARCO v1. For evaluation on Wikipedia we use the December 2018 dump ( Karpukhin et al., 2020 ) with queries from the NaturalQuestions (NQ) dataset ( Kwiatkowski et al., 2019 ) . Our LoTTE ( Santhanam et al., 2021 ) evaluation uses the “pooled” dev dataset with “search”-style queries. For MS MARCO v2, we use the augmented passage version of the data ( Authors, 2021 ) and include passage titles while ignoring headings. As we evaluate several configurations of the models, all of our evaluation is performed using development set queries.

### Systems and hyperparameters

<!-- S5.SS1.SSS0.Px3.p1.1 -->
We report results for several systems for end-to-end results: vanilla ColBERTv2 and PLAID ColBERTv2 as well as ColBERT (v1) ( Khattab and Zaharia, 2020 ) , BM25 ( Robertson et al., 1995 ) , SPLADEv2 ( Formal et al., 2021a ) , and DPR ( Karpukhin et al., 2020 ) . For vanilla ColBERTv2, we use the specific hyperparameters reported in the ColBERTv2 paper for each benchmark dataset. We indicate these in the result tables with p ( nprobe ) and c ( ncandidates ). For PLAID ColBERTv2, we evaluate three different settings: $k=10$ , $k=100$ , and $k=1000$ . The $k$ parameter controls the final number of scored documents as well as the retrieval hyperparameters described in § 4 . Table 2 lists these hyperparameter configurations for each $k$ setting. We find empirically that ranking $\frac{\texttt{ndocs}}{4}$ documents for the final scoring stage produces strong results. For both vanilla ColBERTv2 and PLAID ColBERTv2, we compress all datasets to 2 bits per dimension, with the exception of MS MARCO v2 where we compress to 1 bit.

### Hardware

<!-- S5.SS1.SSS0.Px4.p1.1 -->
We conduct all experiments on servers with 28 Intel Xeon Gold 6132 2.6 GHz CPU cores (2 threads per core for a total of 56 threads) and 4 NVIDIA TITAN V GPUs each. Every server has two NUMA sockets with roughly 92 ns intra-socket memory latency, 142 ns inter-socket memory latency, 72 GBps intra-socket memory bandwidth, and 33 GBps inter-socket memory bandwidth. Each TITAN V GPU has 12 GB of high-bandwidth memory.

### Latency measurements

<!-- S5.SS1.SSS0.Px5.p1.1 -->
When measuring latency for end-to-end results, we compute the average latency of all queries (see Table 1 for query totals), and then report the minimum average latency across 3 trials. For other results we describe the specific measurement procedure in the relevant section. We discard the query encoding latency for neural models (ColBERTv1 ( Khattab and Zaharia, 2020 ) , vanilla ColBERTv2 ( Santhanam et al., 2021 ) , PLAID ColBERTv2, and SPLADEv2 ( Formal et al., 2021a ) ) following Mackenzie et al. (2021) ; prior work has shown that the cost of running the BERT model can be made negligible with standard techniques such as quantization, distillation, etc. ( Bergum, 2021 ) . We measure latency on an otherwise idle machine. We prepend commands with numactl --membind 0 to ensure intra-socket I/O operations. We do not do this for MS MARCO v2, since its large index may require both NUMA nodes. For GPU results we allow full usage of all 56 threads, but for CPU-only results we restrict usage to either 1 or 8 threads using torch.set_num_threads . For non-ColBERT systems we use the single-threaded latency numbers reported by Mackenzie et al. (2021) . Note that these numbers were measured on a different hardware setup and using a different implementation and are therefore simply meant to establish PLAID ColBERTv2’s competitive performance rather than serving as absolute comparisons.

### End-to-end Results

<!-- table 3 -->
| System | MRR@10 | R@100 | R@1k | 1-CPU ms | 8-CPU ms | GPU ms |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 (PISA; k=1000) | 18.7* | - | - | 8.3* | - | - |
| SPLADEv2 (PISA; k=1000) | 36.8* | - | 97.9* | 220.3* | - | - |
| ColBERTv1 | 36.1 | 87.3 | 95.2 | - | - | 54.3 |
| Vanilla ColBERTv2 (p=2,c=2^13) | 39.7 | 90.4 | 96.6 | 3485.1 | 921.8 | 53.4 |
| Vanilla ColBERTv2 (p=4,c=2^16) | 39.7 | 91.4 | 98.3 | - | 4568.5 | 259.6 |
| PLAID ColBERTv2 (k=10) | 39.4 | - | - | 185.5 | 31.5 | 11.5 |
| PLAID ColBERTv2 (k=100) | 39.8 | 90.6 | - | 222.3 | 52.9 | 20.2 |
| PLAID ColBERTv2 (k=1000) | 39.8 | 91.3 | 97.5 | 352.3 | 101.3 | 38.4 |

Table 3 . End-to-end in-domain evaluation on the MS MARCO v1 benchmark. Numbers marked with an asterisk are copied from Formal
et al. (2021b) for SPLADEv2 quality and Mackenzie
et al. (2021) for latencies.

<!-- table 4 -->
| System | Success@5 | Success@100 | 8-CPU ms | GPU ms |
| --- | --- | --- | --- | --- |
| DPR | 66.8 | 85.0 | - | - |
| ColBERT-QA uncompressed | 75.3 | 89.2 | - | - |
| Vanilla ColBERT-QA (p=4,c=2^15; residual compression) | 74.3 | 89.0 | 5077.9 | 204.1 |
| PLAID ColBERT-QA (k=10; residual compression) | 73.3 | - | 67.1 | 13.6 |
| PLAID ColBERT-QA (k=100; residual compression) | 74.1 | 88.0 | 120.1 | 26.9 |
| PLAID ColBERT-QA (k=1000; residual compression) | 74.4 | 88.9 | 228.4 | 55.3 |

Table 4 . End-to-end in-domain retrieval evaluation on the Wikipedia open-domain question answering benchmark. We use the NQ checkpoint of ColBERT-QA ( Khattab
et al., 2021b ) , and apply ColBERTv2 compression. We compare vanilla ColBERTv2 retrieval against PLAID ColBERTv2 retrieval. DPR results from Karpukhin et al. (2020) . We refer to Khattab
et al. (2021b) for details on OpenQA retrieval evaluation.

<!-- table 5 -->
| System | Success@5 | Success@100 | 8-CPU ms | GPU ms |
| --- | --- | --- | --- | --- |
| BM25 | 47.8* | 77.6* | - | - |
| SPLADEv2 | 67.0* | 89.0* | - | - |
| Vanilla ColBERTv2 (p=2,c=2^13) | 69.3 | 90.3 | 1508.4 | 66.9 |
| PLAID ColBERTv2 (k=10) | 69.1 | - | 35.5 | 9.2 |
| PLAID ColBERTv2 (k=100) | 69.4 | 89.9 | 64.8 | 17.4 |
| PLAID ColBERTv2 (k=1000) | 69.6 | 90.5 | 163.1 | 27.3 |

Table 5 . End-to-end out-of-domain evaluation on the (dev) pooled dataset of the LoTTE benchmark. Numbers marked with an asterisk were taken from Santhanam et al. (2021) .

<!-- table 6 -->
| System | MRR@100 | R@100 | R@1k | 8-CPU ms | GPU ms |
| --- | --- | --- | --- | --- | --- |
| BM25 (Anserini; augmented) | 8.7 | 40.3 | 69.3 | - | - |
| Vanilla ColBERTv2 (p=4,c=2^16) | 18.0 | 68.2 | 88.1 | 5228.5 | OOM |
| PLAID ColBERTv2 (k=10) | - | - | - | 136.4 | 47.1 |
| PLAID ColBERTv2 (k=100) | 17.9 | 67.0 | - | 181.9 | 96.1 |
| PLAID ColBERTv2 (k=1000) | 18.0 | 68.4 | 85.7 | 251.3 | OOM |

Table 6 . End-to-end out-of-domain evaluation on the MS MARCO v2 benchmark. BM25 results from ( Authors, 2022 ) .

<!-- S5.SS2.p1.1 -->
Table 3 presents in-domain results for the MS MARCO v1 benchmark. We observe that in the most conservative setting ( $k=1000$ ), PLAID ColBERTv2 is able to match the MRR@10 and Recall@100 achieved by vanilla ColBERTv2 while delivering speedups of 6.8 $\times$ on GPU and 45 $\times$ on CPU. For some minimal reduction in quality, PLAID ColBERTv2 can further increase the speedups over vanilla ColBERTv2 to 12.9–22.6 $\times$ on GPU and 86.4–145 $\times$ on CPU. PLAID ColBERTv2 also achieves competitive latency compared to other systems (within 1.6 $\times$ of SPLADEv2) while outperforming them on retrieval quality.

<!-- S5.SS2.p2.1 -->
We observe a similar trend with in-domain evaluation on the Wikipedia OpenQA benchmark as shown in Table 4 . PLAID ColBERTv2 achieves speedups of 3.7 $\times$ on GPU and 22 $\times$ on CPU with no quality loss compared to vanilla ColBERTv2, and speedups of 7.6–15 $\times$ on GPU and 42.3–75.7 $\times$ on CPU with minimal quality loss.

<!-- S5.SS2.p3.1 -->
We confirm PLAID works well in out-of-domain settings, as well, as demonstrated by our results on the LoTTE “pooled” dataset. We see in Table 5 that PLAID ColBERTv2 outperforms vanilla ColBERTv2 by 2.5 $\times$ on GPU and 9.2 $\times$ on CPU with $k=1000$ ; furthermore, this setting actually improves quality compared to vanilla ColBERTv2. With some quality loss PLAID ColBERTv2 can achieve speedups of 3.8–7.3 $\times$ on GPU and 23.2–42.5 $\times$ on CPU. Note that the CPU latencies achieved on LoTTE with PLAID ColBERTv2 are larger than those achieved on MS MARCO v1 because the average LoTTE passage length is roughly 2 $\times$ that of MS MARCO v1.

<!-- S5.SS2.p4.1 -->
Finally, Table 6 shows that PLAID ColBERTv2 scales effectively to MS MARCO v2, which is a large-scale dataset with 138M passages and 9.4B tokens (approximately 16 $\times$ bigger than MS MARCO v1). Continuing the trend we observe with other datasets, we find that PLAID ColBERTv2 is 20.8 $\times$ faster than vanilla ColBERTv2 on CPU with no quality loss up to 100 passages. We do find that when $k=1000$ both vanilla ColBERTv2 and PLAID ColBERTv2 run out of memory on GPU; we believe we can address this in PLAID by implementing custom padding-free MaxSim kernels for GPU execution as discussed in § 4.5 .

### Ablation

![Figure 6](figs/fig06.png)

Figure 6 . Ablation of performance optimizations included in PLAID.

<!-- S5.SS3.p1.1 -->
Figure 6 presents an ablation analysis to break down PLAID’s performance improvements for both GPU and CPU execution. Our measurements are taken from evaluation on a random sample of 500 MS MARCO v1 queries (note that this results in minor differences in the absolute numbers reported in Table 3 ). We consider vanilla ColBERTv2 as a baseline, and then add one stage of centroid interaction without pruning (stage 3 in Figure 5 ), followed by another stage of centroid interaction with centroid pruning (stage 2 in Figure 5 ), and then finally the optimized kernels described in § 4.5 . When applicable we use hyperparameters corresponding to the $k=1000$ setting described in Table 2 (i.e., the most conservative setting).

<!-- S5.SS3.p2.1 -->
We find that both the algorithmic improvements to the scoring pipeline as well as the implementation optimizations are key to PLAID’s performance. In particular, the centroid interaction stages alone deliver speedups of 5.2 $\times$ on GPU and 8.6 $\times$ on CPU, but adding the implementation optimizations result in additional speedups of 1.3 $\times$ on GPU and 4.9 $\times$ on CPU. Only enabling optimized C++ kernels on CPU without centroid interaction (not shown in Figure 6 ) results in an end-to-end speedup of just 3 $\times$ compared to 42.4 $\times$ with the complete PLAID.

### Scalability

![Figure 7](figs/fig07.png)

Figure 7 . End-to-end latency versus dataset size (as measured in number of embeddings) for each setting of $k$ (note the log-log scale). Dataset sizes are taken from Table 1 , and latency numbers are taken from Tables 3 , 4 , 5 , and 6 .

![Figure 8](figs/fig08.png)

Figure 8 . PLAID scaling behavior with respect to the number of available CPU threads.

<!-- S5.SS4.p1.1 -->
We evaluate PLAID’s scalability with respect to both the dataset size as well as the parallelism degree (on CPU).

<!-- S5.SS4.p2.1 -->
First, Figure 7 plots the end-to-end PLAID ColBERTv2 latencies we measured for each benchmark dataset versus the size of each dataset (measured in number of embeddings). While latencies across different datasets are not necessarily directly comparable (e.g due to different passage lengths), we nevertheless aim to analyze high-level trends from this figure. We find that in general, PLAID ColBERTv2 latencies appear to scale with respect to the square root of dataset size. This intuitively follows from the fact that ColBERTv2 sets the number of centroids proportionally to the square root of the number of embeddings, and the overhead of candidate generation is inversely correlated with the number of partitions.

<!-- S5.SS4.p3.1 -->
Next, Figure 8 plots the latency achieved by PLAID ColBERTv2 versus the number of available CPU threads, repeated for $k\in\{10,100,1000\}$ . We evaluate a random sample of 500 MS MARCO v1 queries to obtain the latency measurements. We observe that PLAID is able to take advantage of additional threads; in particular, executing with 16 threads results in a speedup of 4.9 $\times$ compared to single-threaded execution when $k=1000$ . While PLAID does not achieve perfect linear scaling, we speculate that possible explanations could include remaining inefficiencies in the existing vanilla ColBERTv2 candidate generation step (which we do not optimize at a low level for this work) or suboptimal load balancing between threads due to the non-uniform passage lengths. We defer more extensive profiling and potential solutions to future work.

## Conclusion

<!-- S6.p1.1 -->
In this work, we presented PLAID, an efficient engine for late interaction that accelerates retrieval by aggressively and cheaply filtering candidate passages. We showed that retrieval with only ColBERTv2 centroids retains high recall compared to vanilla ColBERTv2, and the distribution of centroid relevance scores skews toward lower magnitude scores. Using these insights, we introduced the technique of centroid interaction and incorporated centroid interaction into multiple stages of the PLAID ColBERTv2 scoring pipeline. We also described our highly optimized implementation of PLAID that includes custom kernels for padding-free MaxSim and residual decompression operations. We found in our evaluation across several IR benchmarks that PLAID ColBERTv2 provides speedups of 2.5–6.8 $\times$ on GPU and 9.2–45 $\times$ on CPU with virtually no quality loss compared to vanilla ColBERTv2 while scaling effectively to a dataset of 140 million passages.

## Original footnotes

* Santhanam and Khattab contributed equally.

1. Code maintained at https://github.com/stanford-futuredata/ColBERT. As of May 2022, PLAID lies under the branch fast_search but will soon be merged upstream.

2. The 32-bit passage-ID representation assumes no more than 2^32 (4 billion) passages, about 30 times the MS MARCO v2 scale.

## References (original bibliographic entries)

[1] Abuzaid et al . (2019) Firas Abuzaid, Geet Sethi, Peter Bailis, and Matei Zaharia. 2019. To index or not to index: Optimizing exact maximum inner product search. In 2019 IEEE 35th International Conference on Data Engineering (ICDE) . IEEE, 1250–1261.

[2] Authors (2021) Anserini GitHub Repo Authors. 2021. Passage Collection (Augmented). https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-v2.md#passage-collection-augmented

[3] Authors (2022) Anserini GitHub Repo Authors. 2022. Anserini Regressions: MS MARCO (V2) Passage Ranking. https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-v2-passage-augmented.md

[4] Bergum (2021) Jo Kristian Bergum. 2021. Pretrained Transformer Language Models for Search - part 3. https://blog.vespa.ai/pretrained-transformer-language-models-for-search-part-3/

[5] Broder et al . (2003) Andrei Z Broder, David Carmel, Michael Herscovici, Aya Soffer, and Jason Zien. 2003. Efficient query evaluation using a two-level retrieval process. In CIKM .

[6] Craswell et al . (2022) Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Jimmy Lin. 2022. Overview of the TREC 2021 deep learning track. In Text REtrieval Conference (TREC) . TREC. https://www.microsoft.com/en-us/research/publication/overview-of-the-trec-2021-deep-learning-track/

[7] Dai and Callan (2020) Zhuyun Dai and Jamie Callan. 2020. Context-Aware Term Weighting For First Stage Passage Retrieval. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020 , Jimmy Huang, Yi Chang, Xueqi Cheng, Jaap Kamps, Vanessa Murdock, Ji-Rong Wen, and Yiqun Liu (Eds.). ACM, 1533–1536. https://doi.org/10.1145/3397271.3401204

[8] Dimopoulos et al . (2013) Constantinos Dimopoulos, Sergey Nepomnyachiy, and Torsten Suel. 2013. Optimizing top-k document retrieval strategies for block-max indexes. In WSDM .

[9] Ding and Suel (2011) Shuai Ding and Torsten Suel. 2011. Faster top-k document retrieval using block-max indexes. In SIGIR .

[10] Formal et al . (2021a) Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. 2021a. SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval. arXiv preprint arXiv:2109.10086 (2021). https://arxiv.org/abs/2109.10086

[11] Formal et al . (2021b) Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021b. SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval . 2288–2292.

[12] Gao et al . (2021) Luyu Gao, Zhuyun Dai, and Jamie Callan. 2021. COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies . Association for Computational Linguistics, Online, 3030–3042. https://doi.org/10.18653/v1/2021.naacl-main.241

[13] Hofstätter et al . (2020) Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, and Allan Hanbury. 2020. Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation. arXiv preprint arXiv:2010.02666 (2020). https://arxiv.org/abs/2010.02666

[14] Hofstätter et al . (2022) Sebastian Hofstätter, Omar Khattab, Sophia Althammer, Mete Sertkan, and Allan Hanbury. 2022. Introducing Neural Bag of Whole-Words with ColBERTer: Contextualized Late Interactions using Enhanced Reduction. arXiv preprint arXiv:2203.13088 (2022).

[15] Humeau et al . (2020) Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. 2020. Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020 . OpenReview.net. https://openreview.net/forum?id=SkxgnnNFvH

[16] Jegou et al . (2010) Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence 33, 1 (2010), 117–128.

[17] Johnson et al . (2019) Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with gpus. IEEE Transactions on Big Data (2019).

[18] Karpukhin et al . (2020) Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) . Association for Computational Linguistics, Online, 6769–6781. https://doi.org/10.18653/v1/2020.emnlp-main.550

[19] Khattab et al . (2020) Omar Khattab, Mohammad Hammoud, and Tamer Elsayed. 2020. Finding the best of both worlds: Faster and more robust top-k document retrieval. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval . 1031–1040.

[20] Khattab et al . (2021a) Omar Khattab, Christopher Potts, and Matei Zaharia. 2021a. Baleen: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval. In Thirty-Fifth Conference on Neural Information Processing Systems .

[21] Khattab et al . (2021b) Omar Khattab, Christopher Potts, and Matei Zaharia. 2021b. Relevance-guided Supervision for OpenQA with ColBERT. Transactions of the Association for Computational Linguistics 9 (2021), 929–944.

[22] Khattab and Zaharia (2020) Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020 , Jimmy Huang, Yi Chang, Xueqi Cheng, Jaap Kamps, Vanessa Murdock, Ji-Rong Wen, and Yiqun Liu (Eds.). ACM, 39–48. https://doi.org/10.1145/3397271.3401075

[23] Kwiatkowski et al . (2019) Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A Benchmark for Question Answering Research. Transactions of the Association for Computational Linguistics 7 (2019), 452–466. https://doi.org/10.1162/tacl_a_00276

[24] Li et al . (2021) Yulong Li, Martin Franz, Md Arafat Sultan, Bhavani Iyer, Young-Suk Lee, and Avirup Sil. 2021. Learning Cross-Lingual IR from an English Retriever. arXiv preprint arXiv:2112.08185 (2021).

[25] Lin (2022) Jimmy Lin. 2022. A proposed conceptual framework for a representational approach to information retrieval. In ACM SIGIR Forum , Vol. 55. ACM New York, NY, USA, 1–29.

[26] Lupart and Clinchant (2022) Simon Lupart and Stéphane Clinchant. 2022. Toward A Fine-Grained Analysis of Distribution Shifts in MSMARCO. arXiv preprint arXiv:2205.02870 (2022).

[27] MacAvaney et al . (2019) Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. 2019. CEDR: Contextualized Embeddings for Document Ranking. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval . 1101–1104. https://doi.org/10.1145/3331184.3331317

[28] Macdonald and Tonellotto (2021) Craig Macdonald and Nicola Tonellotto. 2021. On approximate nearest neighbour selection for multi-stage dense retrieval. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management . 3318–3322.

[29] Macdonald et al . (2021) Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2021. On Single and Multiple Representations in Dense Passage Retrieval. arXiv preprint arXiv:2108.06279 (2021).

[30] Mackenzie et al . (2021) Joel Mackenzie, Andrew Trotman, and Jimmy Lin. 2021. Wacky Weights in Learned Sparse Representations and the Revenge of Score-at-a-Time Query Evaluation. arXiv preprint arXiv:2110.11540 (2021).

[31] Malkov and Yashunin (2018) Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE transactions on pattern analysis and machine intelligence 42, 4 (2018), 824–836.

[32] Mallia et al . (2021) Antonio Mallia, Omar Khattab, Torsten Suel, and Nicola Tonellotto. 2021. Learning passage impacts for inverted indexes. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval . 1723–1727.

[33] Mallia et al . (2017) Antonio Mallia, Giuseppe Ottaviano, Elia Porciani, Nicola Tonellotto, and Rossano Venturini. 2017. Faster blockmax wand with variable-sized blocks. In SIGIR .

[34] Mallia et al . (2019) Antonio Mallia, Michal Siedlaczek, Joel Mackenzie, and Torsten Suel. 2019. PISA: Performant indexes and search for academia. Proceedings of the Open-Source IR Replicability Challenge (2019).

[35] Minas Krasakis et al . (2022) Antonios Minas Krasakis, Andrew Yates, and Evangelos Kanoulas. 2022. Zero-shot Query Contextualization for Conversational Search. arXiv e-prints (2022), arXiv–2204.

[36] Nguyen et al . (2016) Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. MS MARCO: A Human-Generated MAchine Reading COmprehension Dataset. arXiv preprint arXiv:1611.09268 (2016). https://arxiv.org/abs/1611.09268

[37] Nogueira and Cho (2019) Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT. arXiv preprint arXiv:1901.04085 (2019). https://arxiv.org/abs/1901.04085

[38] Paranjape et al . (2022) Ashwin Paranjape, Omar Khattab, Christopher Potts, Matei Zaharia, and Christopher D Manning. 2022. Hindsight: Posterior-guided Training of Retrievers for Improved Open-ended Generation. In International Conference on Learning Representations . https://openreview.net/forum?id=Vr_BTpw3wz

[39] Qu et al . (2021) Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. 2021. RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies . Association for Computational Linguistics, Online, 5835–5847. https://doi.org/10.18653/v1/2021.naacl-main.466

[40] Ren et al . (2021) Ruiyang Ren, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Qiaoqiao She, Hua Wu, Haifeng Wang, and Ji-Rong Wen. 2021. RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking. arXiv preprint arXiv:2110.07367 (2021). https://arxiv.org/abs/2110.07367

[41] Robertson et al . (1995) Stephen E Robertson, Steve Walker, Susan Jones, Micheline M Hancock-Beaulieu, Mike Gatford, et al . 1995. Okapi at TREC-3. NIST Special Publication (1995).

[42] Santhanam et al . (2021) Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2021. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. arXiv preprint arXiv:2112.01488 (2021).

[43] Thai et al . (2022) Katherine Thai, Yapei Chang, Kalpesh Krishna, and Mohit Iyyer. 2022. RELIC: Retrieving Evidence for Literary Claims. arXiv preprint arXiv:2203.10053 (2022).

[44] Thakur et al . (2021) Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2) . https://openreview.net/forum?id=wCu6T5xFjeJ

[45] Tonellotto and Macdonald (2021) Nicola Tonellotto and Craig Macdonald. 2021. Query embedding pruning for dense retrieval. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management . 3453–3457.

[46] Tonellotto et al . (2018) Nicola Tonellotto, Craig Macdonald, Iadh Ounis, et al . 2018. Efficient Query Processing for Scalable Web Search. Foundations and Trends® in Information Retrieval (2018).

[47] Turtle and Flood (1995) Howard Turtle and James Flood. 1995. Query evaluation: strategies and optimizations. IP & M (1995).

[48] Wang et al . (2021) Xiao Wang, Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2021. Pseudo-relevance feedback for multiple representation dense retrieval. In Proceedings of the 2021 ACM SIGIR International Conference on Theory of Information Retrieval . 297–306.

[49] Xiong et al . (2020) Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. In International Conference on Learning Representations .

[50] Yang et al . (2018) Peilin Yang, Hui Fang, and Jimmy Lin. 2018. Anserini: Reproducible ranking baselines using Lucene. Journal of Data and Information Quality (JDIQ) 10, 4 (2018), 1–20.

[51] Zeng et al . (2022) Hansi Zeng, Hamed Zamani, and Vishwa Vinay. 2022. Curriculum Learning for Dense Retrieval Distillation. arXiv preprint arXiv:2204.13679 (2022).

[52] Zhan et al . (2020) Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Min Zhang, and Shaoping Ma. 2020. Learning To Retrieve: How to Train a Dense Retrieval Model Effectively and Efficiently. arXiv preprint arXiv:2010.10469 (2020). https://arxiv.org/abs/2010.10469

[53] Zhan et al . (2022) Jingtao Zhan, Xiaohui Xie, Jiaxin Mao, Yiqun Liu, Min Zhang, and Shaoping Ma. 2022. Evaluating Extrapolation Performance of Dense Retrieval. arXiv preprint arXiv:2204.11447 (2022).

[54] Zhong et al . (2022) Wei Zhong, Jheng-Hong Yang, and Jimmy Lin. 2022. Evaluating Token-Level and Passage-Level Dense Retrieval Models for Math Information Retrieval. arXiv preprint arXiv:2203.11163 (2022).
