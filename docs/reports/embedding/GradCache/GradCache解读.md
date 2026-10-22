# GradCache 补充解读：把“谁影响谁”与“怎么编码”分开

> 以下推导用于理解和实现检查，实验数字见主笔记。

## 缓存的不是旧文档向量

训练中的 memory bank 可能用之前迭代得到的向量当负例；GradCache 在一次更新里用**当前参数**先算完整 batch 表示，再计算每个表示应往哪个方向移动，最后重算当前参数编码器把这些方向传下去。它不会因为名叫 cache 就引入跨步陈旧表示。

## 一个两样本例子

假设大 batch 里有 $(q_1,d_1)$ 和 $(q_2,d_2)$。完整损失同时比较 $q_1$ 对 $d_1,d_2$，以及 $q_2$ 对 $d_1,d_2$。如果拆成两个只有一个正例的小 batch，分母可能只剩正例自身，loss 甚至退化为零。

普通梯度累积加总的是两个已经退化的目标，无法凭空产生 $q_1$ 对 $d_2$ 的排斥梯度。GradCache 先在两个文档都存在的表示层计算这些相互影响，再逐样本传入编码器，保留完整目标。

## 用代理标量理解最后一次反向

缓存得到 $a_i$ 后，可以对重算表示 $u_i(\theta)$ 构造

$$
\widetilde{\mathcal L}_i=\operatorname{stopgrad}(a_i)^\top u_i(\theta).
$$

这个标量的值本身没有检索意义，但它对参数的梯度正好等于所需的 $a_i^\top\partial u_i/\partial\theta$。所以实现可以直接给向量调用 backward，也可以对这一代理标量反传；缓存梯度必须视为常量。

## 四个常见实现错误

- 每个小 batch 都 optimizer.step：后面的重算参数已改变，破坏同一次完整更新。
- 缓存 loss 已经按大 batch 平均，反传时又除小 batch 数：梯度被重复缩小。
- 无图前向与重算 dropout 不一致：上游梯度对应的表示和实际 Jacobian 不匹配。
- 多卡 all-gather 只扩大前向分母，却漏掉远端 query 对本地 document 的梯度：看似用了全局负例，更新却不完整。

## 仍然可能 OOM 在哪里

如果编码器激活已被控制住，瓶颈可能移到 $B\times B$ 分数矩阵或多向量 late interaction 张量。此时还需要分块计算损失、流式 softmax 或更专门的实现。GradCache 与 activation checkpointing 也可以组合：前者拆 batch 依赖，后者拆层激活，两者代价都会增加重算。

参考：[原论文](https://aclanthology.org/2021.repl4nlp-1.31/)；[实现](https://github.com/luyug/GradCache)。
