# rsLoRA 技术详解

> 基于论文 [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/abs/2312.03732)（Damjan Kalajdzievski，Tenyx；arXiv:2312.03732，2023-11）。
> 本文把 **LoRA 缩放因子 γ=α/r 为何在高 rank 失效、rank-stabilized 的形式化定义、定理 3.2 的直觉与推论、γ=α/√r 的推导、rank 4→2048 的梯度塌缩实证、以及七组消融** 写全。
> 工程入口：HuggingFace PEFT 中 `LoraConfig(use_rslora=True)`。

---

## 1. 一句话定位

**rsLoRA（Rank-Stabilized LoRA）** 不改结构、不加参数，只改一个数：把 LoRA 的缩放因子从 **γ = α/r** 改为 **γ = α/√r**。这一改动消除了高 rank 下的**梯度塌缩（gradient collapse）**，让「加大 rank 换性能」第一次真正生效：

| 项 | 内容 |
| --- | --- |
| 问题 | LoRA 的 γ=α/r 在 rank 增大时把适配器输出/梯度压得趋零 → **高 rank ≈ 低 rank**，甚至略差 |
| 修正 | **γ = α/√r** 是唯一（模常数倍）能让适配器在任意 rank 下保持激活/梯度量级稳定的缩放 |
| 收益 | rank 4→2048 范围内，rsLoRA 微调损失随 rank **持续下降**；LoRA 基本平躺 |
| 成本 | **推理零额外开销**（同样可 merge）；只增加训练期计算 |
| 定位 | 把 LoRA 从「低秩小修小补」解锁为「训练算力 ↔ 性能的可调旋钮」 |
| 理论工具 | 借鉴 Yang & Hu 的 **abc-parametrization / 无穷宽极限学习轨迹分析** |
| 对社区的纠偏 | LoRA 原文「很低 rank（4/8/16）即够用」的结论，部分是 **γ=α/r 缩放导致高 rank 学不动** 的假象 |

一句话：**LoRA 说「rank 高了没用」，rsLoRA 说「是你的缩放把高 rank 摁死了」**。

---

## 2. 论文目录与结构导读

```text
§1  Introduction                     缩放因子被忽视；提出 rank-stabilized 缩放
§2  Background and Relevant Works
    2.1  Low-Rank Adapters (LoRA)    LoRA 形式化 + γ_r 的角色；顺带指出对 AdaLoRA 的适用性
    2.2  Scaling-Initialization-Update Schemes   Yang & Hu 的 abc-parametrization 框架
§3  rsLoRA: Rank-Stabilized Adapters
    定义 3.1  rank-stabilized 的两条稳定性条件
    定理 3.2  γ_r ∈ Θ(1/√r) 当且仅当 rank-stabilized
§4  Experimental Results             Llama-2 + OpenOrca，rank ∈ {4,8,32,128,512,2048}
§5  Conclusion                       计算↔性能旋钮；对「低内在秩」叙事的修正
附录 A  定理 3.2 完整证明
附录 B  七组消融：SGD 稳定性 / 换模型·优化器·数据 / 仅初始化缩放 / 低 rank 调 lr 不够 /
        仅 attention 适配 / 其它缩放函数 / 激活量级
```

阅读重点：§2.2 只需掌握框架直觉，**核心结论在定义 3.1 与定理 3.2**；§4 两张图（loss 曲线、梯度范数）是全部实验主张；附录 B.4（「低 rank 下单纯调大学习率不够」）是最容易被实践者问到的反例，值得细读。

---

## 3. 逐章节讲解

### 3.1 §1 Introduction：一个超参被当成常数

LoRA 的适配器写作 γ_r·BA，其中 γ_r 是随 rank 变化的缩放。实践中所有实现都照抄原文 **γ_r = α/r**（α 为常数），然后大家观察到「rank 提到 64/128 也没用」，于是「低内在秩」成了社区共识。

作者指出：这个共识里混进了一个**缩放伪影**。α/r 随 r 线性衰减，rank 翻倍，适配器输出先验地缩小一倍——梯度随之塌缩，高 rank 适配器「没学到东西」不是学不到，而是**被缩放摁住了**。本文三件事：

1. 证明唯一稳定的缩放是 γ_r = α/√r（定理 3.2）；
2. 实验展示修正后高 rank 持续带来收益；
3. 强调推理成本不变——**这是纯训练期的算力↔性能交换**。

### 3.2 §2 Background

#### 3.2.1 LoRA 形式化（论文版记号）

对预训练线性子模块 $x_{out} = W x_{in} + b$，LoRA 增广为

$$x_{out} = (W + \gamma_r B A)\, x_{in} + b$$

- $A \in \mathbb{R}^{r \times d_1}$，$B \in \mathbb{R}^{d_2 \times r}$，训练后合并存 $(W + \gamma_r B A)$；
- 初始化：$B = 0$，$A$ 元素 iid、均值 0、**方差与 r 无关**；
- 惯例实现：$\gamma_r = \alpha / r$。

本节还点名 **AdaLoRA**：它动态分配 rank，但沿用同一个 α/r 缩放——因此 rsLoRA 的修正对 AdaLoRA 同样适用（事实上任何含 γ_r 的 LoRA 变体都适用）。

并直接给出对 LoRA 原文的批评：Hu et al. 的 Table 6 显示 rank 64 相比 4/8/16 无提升，由此得出「很低 rank 即足够」——rsLoRA 认为这是缩放导致的**梯度塌缩假象**。

#### 3.2.2 abc-parametrization 框架（只需直觉）

Yang & Hu (2022) 研究无穷宽极限下「缩放-初始化-更新」三件套如何影响学习稳定性：把每层权重写成 $W^l = d^{-a_l} \cdot (W^l_{i,j})$，初始化方差 $d^{-b_l}$，学习率 $\eta d^{-c}$，分析 $(a,b,c)$ 取何值时宽网学习不塌缩。

rsLoRA 借用同一思想：把 **rank r 当作「宽度」**，问 γ_r 随 r 怎么缩放才能让适配器的**前向激活**与**反向梯度**在 r→∞ 时既不爆炸也不消失。结论：γ_r 必须按 1/√r 衰减。

### 3.3 §3 rsLoRA：定义与定理

#### 3.3.1 定义 3.1：什么叫「rank-stabilized」

适配器 γ_r BA 被称为 rank-stabilized，当且仅当两条成立：

1. **前向稳定**：若输入各元素的 m 阶矩是 $\Theta_r(1)$（与 r 无关的常数量级），则输出各元素的 m 阶矩也是 $\Theta_r(1)$；
2. **反向稳定**：若损失对适配器输出的梯度各元素是 $\Theta_r(1)$，则传回适配器输入的梯度各元素也是 $\Theta_r(1)$。

翻译：**无论 rank 多大，适配器既不放大也不缩小信号与梯度**。这正是一个「行为良好的层」该有的性质。

#### 3.3.2 定理 3.2：唯一解 γ_r ∈ Θ(1/√r)

在 $B=0$、$A$ 元素 iid 零均值、方差与 r 无关、且 γ_r→0（r→∞）的设定下，**在初始化的期望意义下，所有适配器在整个学习轨迹上 rank-stabilized，当且仅当**

$$\gamma_r \in \Theta_r\!\left(\frac{1}{\sqrt{r}}\right)$$

直觉推导（附录 A 的严格版）：$BAx$ 的每个输出元素是 r 个乘积项 $B_{\cdot i} A_{i \cdot} x$ 之和。由中心极限，r 个 iid 项之和的标准差按 **√r** 增长；要让输出量级与 r 无关，就得除以 √r。除以 r（LoRA 默认）等于**多压了 √r 倍**——r 越大压得越狠，梯度随之塌缩。

两个重要澄清（论文 §3 明确说明）：

- 定理只管**稳定性/塌缩**，不管「不同 rank 学到的特征质量」；高 rank 是否有用要实验验证（§4 给了肯定答案）；
- 中间层的梯度会经其它适配器间接依赖 r，但论文用**归纳法**说明：从输入层开始逐层前向、再从输出层逐层反向应用定理，链路中每一环都保持 Θ(1)。

#### 3.3.3 为什么是「交换旋钮」

- 高 rank 只增加**训练期** FLOPs 与适配器显存；合并后推理矩阵形状不变，**推理成本与 rank 无关**；
- 于是「rank」从「基本没用的旋钮」变成「预算够就往上拧」的旋钮——这正是论文副标题 *unlocking the potential* 的含义。

### 3.4 §4 Experimental Results

#### 3.4.1 设置

- 模型：**Llama-2**；数据：**OpenOrca** 指令集 20,000 条；
- 优化器：AdamW，lr 5e-5（HF 默认），常数 schedule；
- 适配位置：**所有线性层**（attention + MLP，非 LayerNorm）——沿用 Zhang et al. 附录 F 的结论「同预算下铺满最优」；
- rank 扫描：**r ∈ {4, 8, 32, 128, 512, 2048}**。

#### 3.4.2 主结果（图 2：微调困惑度）

- **LoRA**（铜色梯度）：各 rank 收敛到几乎相同的 loss；部分高 rank 甚至略差——α/r 把高 rank 摁死；
- **rsLoRA**（蓝绿梯度）：**rank 越大 loss 越低**，到 2048 仍在改善——「更多参数 → 更好微调」的预期（Ding et al., 2022）终于被兑现。

#### 3.4.3 机制验证（图 3：梯度范数）

- LoRA：随 r 增大，**初始梯度范数单调塌缩**（高 rank 线几乎贴地）；
- rsLoRA：**所有 rank 的初始梯度范数重合**，训练全程保持同一量级——定理 3.2 的直接可视化。

#### 3.4.4 消融一览（附录 B，§4 正文中点名）

| 消融 | 结论 |
| --- | --- |
| B.1 换 SGD | 梯度塌缩/稳定模式不变——不是 AdamW 的伪影 |
| B.2 换模型/优化器/数据集 | 模式复现 |
| B.3 仅初始化时缩放 | 不够——**整个训练过程**都需要 γ=α/√r |
| B.4 低 rank + 调大学习率 | **追不平** rsLoRA 高 rank——缩放修正 ≠ 调 lr |
| B.5 仅 attention 适配 | 结论不变，但铺满层仍是最优配置 |
| B.6 其它缩放函数 | α/r^β 扫描证实 β=1/2 最优区间 |
| B.7 激活量级 | rsLoRA 各 rank 激活量级一致，LoRA 随 r 衰减 |

B.4 特别值得记住：它封死了「我不改缩放，低 rank 下把学习率调大不就完了」的退路——**学习率补偿不了秩依赖的塌缩结构**，因为塌缩是逐层、随 r 变化的几何问题，不是全局步长问题。

### 3.5 §5 Conclusion

- γ=α/√r 是 LoRA 家族的**默认推荐缩放**（尤其计划用 r ≥ 32 时）；
- 「低内在秩」叙事需要重审：LoRA 原文「rank 64 无提升」很可能是缩放伪影；
- rsLoRA 与量化（QLoRA）、预算分配（AdaLoRA）、学习率拆分（LoRA+）、权重分解（DoRA）**全部正交**，可自由组合。

---

## 4. 附录 A：定理证明脉络（直觉版）

1. **单步前向**：输出元素 = r 个 iid 乘积项之和 → 方差 ∝ r → 需 γ_r ∝ 1/√r 归一；
2. **单步反向**：同构的求和结构 → 同一缩放；
3. **学习轨迹上**：对训练步归纳——只要每步更新量保持 Θ(1) 且各步方向不恶意对齐，量级结论在任意步成立；
4. **跨层传播**：对深度归纳——前向从第一层归到最后一层，反向反之；输入数据本身与 r 无关，提供了归纳基例。

严谨版处理了 $B$ 不再为 0（训练后）的情形：结论在期望意义下对任意学习步仍成立。

---

## 5. 实践指南

### 5.1 何时打开 rsLoRA

| 场景 | 建议 |
| --- | --- |
| r ≤ 16，快速验证 | 默认 LoRA 即可，rsLoRA 也无害 |
| r ≥ 32，追求更高拟合能力 | **开 rsLoRA**（α/√r），否则高 rank 大概率白搭 |
| 与 QLoRA 叠加 | 可以：量化只管基座存储，rsLoRA 只管适配器缩放 |
| 与 AdaLoRA 叠加 | 可以且推荐：动态 rank 分配同样需要稳定缩放 |
| 与 LoRA+ 叠加 | 可以：LoRA+ 管 A/B 学习率比，rsLoRA 管 rank 缩放，两个旋钮独立 |

### 5.2 超参换算

- HF PEFT：`LoraConfig(use_rslora=True, r=..., lora_alpha=...)`，缩放自动变为 α/√r；
- α 的经验起点：α ≈ r（LoRA 惯例）→ 换 rsLoRA 后 α 不需重调太狠；论文实验用 α 为常数（同一 α 跨 rank 扫描），结论对 α 选择稳健；
- 常见配方：r=64, α=64, rsLoRA → 有效缩放 64/8=8（比 LoRA 的 64/64=1 大 8 倍）——**第一次切换到 rsLoRA 时注意等效步长变大**，必要时同步降一点 lr 或 α。

### 5.3 排错信号

- 高 rank LoRA loss 与低 rank 无差 → 典型 α/r 塌缩，换 rsLoRA；
- 换 rsLoRA 后初期 loss 下降明显变快、偶发尖峰 → 等效缩放变大，微调 lr/α；
- 只想省显存不想动表达力 → rsLoRA 与 QLoRA 组合，rank 不必拉满。

---

## 6. 局限与开放问题

- **定理只管稳定不管质量**：高 rank 是否总更好取决于任务/数据；论文实证是指令微调单一场景；
- **α 与 lr 的联合最优**未系统刻画（B.6 只扫了 β）；
- **与初始化的交互**：PiSSA（SVD 初始化）、LoRA-GA（梯度对齐初始化）等改变 A/B 起点的方法，与 rank 缩放的联合理论尚缺；
- **超大 rank 的边际**：2048 之后是否仍单调，论文未探；
- **对「低内在秩」假说的冲击边界**：rsLoRA 证明的是「高 rank 学不动是缩放问题」，但「任务需要多大秩」仍是开放的经验问题。

---

## 7. 参考文献

- Kalajdzievski, D. **A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA.** arXiv:2312.03732, 2023.（Tenyx）
- Hu, E. et al. **LoRA: Low-Rank Adaptation of Large Language Models.** arXiv:2106.09685, 2021.
- Yang, G. & Hu, E. **Tensor Programs IVb: Adaptive Optimization in the Infinite-Width Limit.** arXiv:2208.01814, 2022.（abc-parametrization 框架）
- Zhang, Q. et al. **AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.** arXiv:2303.10512, 2023.
- Hayou, S. et al. **LoRA+: Efficient Low Rank Adaptation of Large Models.** arXiv:2402.12354, 2024.
- Dettmers, T. et al. **QLoRA: Efficient Finetuning of Quantized LLMs.** arXiv:2305.14314, 2023.
- HuggingFace PEFT 文档：[`use_rslora` 说明](https://huggingface.co/docs/peft/package_reference/lora)；博客 [Rank-Stabilized LoRA](https://huggingface.co/blog/damjan-k/rslora)
- Ding, N. et al. **Parameter-Efficient Fine-Tuning of Large-Scale Pre-Trained Language Models.** Nature Machine Intelligence, 2023.
