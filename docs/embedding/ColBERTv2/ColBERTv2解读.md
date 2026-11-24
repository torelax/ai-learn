# ColBERTv2 补充解读

> 本页区分论文结果与进一步的理解、工程取舍。

## 质心与残差分工

质心给出同簇 token 的共享位置，残差保留具体上下文。只存质心会把同簇 token 完全混同，存高精度残差又失去压缩价值。1/2 bit 方案的有效性来自训练后 token 空间的结构和实验验证，不能推广到任意随机向量。

## 监督提升与压缩收益应分别测

将原版 ColBERT 与最终 v2 比较，同时改变了训练数据、教师和存储表示。要回答“量化损失多少”，应固定 checkpoint 比较压缩前后；要回答“蒸馏贡献多少”，应固定表示及候选设置。否则模型质量提升可能掩盖量化损失。

## 为什么还需要 PLAID

25 GiB 的索引比 154 GiB 更容易放下，但每次查询仍可能读取和解压上万篇候选。存储占用与在线带宽是两个瓶颈，PLAID 进一步让质心承担提前筛选的工作。

## 与现代 Embedding / 检索的关系

| 工作 | 关系 |
| ------ | ------ |
| ColBERT | 架构父亲；v2 = 压缩 + 现代监督 |
| RocketQAv2 / TAS-B | 同属蒸馏+难负例；学生换 late interaction |
| SPLADEv2 | 同「词级分解」竞品；稀疏词表维 vs 稠密多向量 |
| JPQ / RepCONC / BPR | 单向量量化/哈希；v2 证明 multi-vector 也可残差压 |
| PLAID | ColBERT 族检索系统加速 |
| ColPali | 视觉页继承 MaxSim；存储问题同样存在，可借鉴残差/token pooling |
| 当代单向量 SOTA（E5、BGE、GTE、Jasper…） | MTEB 主战场仍是单向量；**高精度检索 / RAG 召回 / 可解释 token 对齐** 时 ColBERT 族仍强 |

实践选型：

- 要 **MTEB 全能句向量** → 单向量蒸馏模型；
- 要 **Passage 检索 MRR / OOD 搜索** 且接受多向量索引 → ColBERTv2；
- 要 **PDF/扫描页、图表** → ColPali，再考虑 PLAID/压缩。

## 实践要点

| 步骤 | 建议 |
| ------ | ------ |
| 复现基线 | 先 triples ColBERT → 再 CE 蒸馏；不要指望一步到位 |
| 压缩位宽 | 默认 **2-bit**；磁盘极紧再试 1-bit 并回归 MRR |
| 质心规模 | 随 $\sqrt{n}$；采样 k-means 省内存 |
| 检索参数 | 调 $n_{\mathrm{probe}}$、$n_{\mathrm{candidate}}$ 做延迟–召回曲线 |
| 蒸馏教师 | 小 CE（MiniLM）足够且便宜；大 cross-encoder 未必划算 |
| 监控 | 压缩前后同 query 的 MaxSim 分数秩相关；异常再查量化 |

索引伪代码：

```text
C = kmeans(sample_embeddings(corpus))
for passage in corpus:
    D = encode(passage)           # multi-vector
    for v in D:
        t = nearest_centroid(v, C)
        store(id=t, residual=quantize(v - C[t], bits=b))
build_inverted_list(by centroid)
```
