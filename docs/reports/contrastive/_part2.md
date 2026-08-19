## 关键要素
> **EN** Key Ingredients

### 大规模数据增强
> **EN** Heavy Data Augmentation

给定一个训练样本，需要采用数据增强技术为其创建噪声版本，作为正样本输入损失函数。恰当的数据增强设置对于学习良好且可泛化的嵌入特征至关重要。它在不修改语义含义的前提下，向样本引入非本质变化，从而鼓励模型学习表示中的本质部分。例如，SimCLR 的实验表明，随机裁剪与随机颜色扰动的组合对于学习图像视觉表示的良好性能至关重要。

> Given a training sample, data augmentation techniques are needed for creating noise versions of itself to feed into the loss as positive samples. Proper data augmentation setup is critical for learning good and generalizable embedding features. It introduces the non-essential variations into examples without modifying semantic meanings and thus encourages the model to learn the essential part of the representation. For example, experiments in SimCLR showed that the composition of random cropping and random color distortion is crucial for good performance on learning visual representation of images.

### 大批量大小
> **EN** Large Batch Size

在训练中使用大批量大小是许多对比学习方法（如 SimCLR、CLIP）成功的另一关键要素，尤其当方法依赖批内负例时。只有当批量足够大，损失函数才能覆盖足够多样化的负样本集合，对模型形成足够挑战，从而学习有意义的表示以区分不同样本。

> Using a large batch size during training is another key ingredient in the success of many contrastive learning methods (e.g. SimCLR, CLIP), especially when it relies on in-batch negatives. Only when the batch size is big enough, the loss function can cover a diverse enough collection of negative samples, challenging enough for the model to learn meaningful representation to distinguish different examples.

### 难负例挖掘
> **EN** Hard Negative Mining

难负样本应与锚样本具有不同标签，但其嵌入特征与锚嵌入非常接近。在有监督数据集中可访问真实标签时，识别任务相关的难负例较为容易。例如，在学习句子嵌入时，可将 NLI 数据集中标注为「矛盾」的句子对视为难负例对（如 SimCSE），或将 BM25 返回的、关键词匹配最多但排序靠前的错误候选作为难负样本（DPR；Karpukhin et al., 2020）。

> Hard negative samples should have different labels from the anchor sample, but have embedding features very close to the anchor embedding. With access to ground truth labels in supervised datasets, it is easy to identify task-specific hard negatives. For example when learning sentence embedding, we can treat sentence pairs labelled as "contradiction" in NLI datasets as hard negative pairs (e.g. SimCSE, or use top incorrect candidates returned by BM25 with most keywords matched as hard negative samples (DPR; Karpukhin et al., 2020).

然而，若希望保持无监督设定，难负例挖掘会变得棘手。增大训练批量或记忆库规模会隐式引入更多难负例，但也会带来大内存占用的副作用。

> However, it becomes tricky to do hard negative mining when we want to remain unsupervised. Increasing training batch size or memory bank size implicitly introduces more hard negative samples, but it leads to a heavy burden of large memory usage as a side effect.

Chuang 等人（2020）研究了对比学习中的采样偏差，并提出了去偏损失。在无监督设定下，由于不知道真实标签，我们可能意外采样到假负例。采样偏差可导致显著的性能下降。

> Chuang et al. (2020) studied the sampling bias in contrastive learning and proposed debiased loss. In the unsupervised setting, since we do not know the ground truth labels, we may accidentally sample false negative samples. Sampling bias can lead to significant performance drop.

![contrastive-sampling-bias](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/contrastive-sampling-bias.png)

对比学习中的采样偏差指假负例，可导致显著性能下降。（图源：Chuang et al., 2020）

> Sampling bias which refers to false negative samples in contrastive learning can lead to a big performance drop. (Image source: Chuang et al., 2020)

假设锚类别 $c$ 的概率均匀为 $\rho(c)=\eta^+$，观察到不同类别的概率为 $\eta^- = 1-\eta^+$。

> Let us assume the probability of anchor class $c$ is uniform $\rho(c)=\eta^+$ and the probability of observing a different class is $\eta^- = 1-\eta^+$.

- 对于 $\mathbf{x}$，观察到正例的概率为 $p^+_x(\mathbf{x}')=p(\mathbf{x}'\vert \mathbf{h}_{x'}=\mathbf{h}_x)$；
- 对于 $\mathbf{x}$，得到负样本的概率为 $p^-_x(\mathbf{x}')=p(\mathbf{x}'\vert \mathbf{h}_{x'}\neq\mathbf{h}_x)$。

> - The probability of observing a positive example for $\mathbf{x}$ is $p^+_x(\mathbf{x}')=p(\mathbf{x}'\vert \mathbf{h}_{x'}=\mathbf{h}_x)$;
> - The probability of getting a negative sample for $\mathbf{x}$ is $p^-_x(\mathbf{x}')=p(\mathbf{x}'\vert \mathbf{h}_{x'}\neq\mathbf{h}_x)$.

当采样 $\mathbf{x}^-$ 时，我们无法访问真实的 $p^-_x(\mathbf{x}^-)$，因此 $\mathbf{x}^-$ 可能以概率 $\eta^+$ 从（不希望的）锚类别 $c$ 中采样。实际采样数据分布变为：

> When we are sampling $\mathbf{x}^-$ , we cannot access the true $p^-_x(\mathbf{x}^-)$ and thus $\mathbf{x}^-$ may be sampled from the (undesired) anchor class $c$ with probability $\eta^+$. The actual sampling data distribution becomes:

$$ p(\mathbf{x}') = \eta^+ p^+_x(\mathbf{x}') + \eta^- p_x^-(\mathbf{x}') $$

因此，采样 $\mathbf{x}^-$ 时可用 $p^-_x(\mathbf{x}') = (p(\mathbf{x}') - \eta^+ p^+_x(\mathbf{x}'))/\eta^-$ 对损失去偏。给定从 $p$ 采样的 $N$ 个样本 $\{\mathbf{u}_i\}^N_{i=1}$ 以及从 $p^+_x$ 采样的 $M$ 个样本 $\{ \mathbf{v}_i \}_{i=1}^M$，可估计对比学习损失分母中第二项 $\mathbb{E}_{\mathbf{x}^-\sim p^-_x}[\exp(f(\mathbf{x})^\top f(\mathbf{x}^-))]$ 的期望：

> Thus we can use $p^-_x(\mathbf{x}') = (p(\mathbf{x}') - \eta^+ p^+_x(\mathbf{x}'))/\eta^-$ for sampling $\mathbf{x}^-$ to debias the loss. With $N$ samples $\{\mathbf{u}_i\}^N_{i=1}$ from $p$ and $M$ samples $\{ \mathbf{v}_i \}_{i=1}^M$ from $p^+_x$ , we can estimate the expectation of the second term $\mathbb{E}_{\mathbf{x}^-\sim p^-_x}[\exp(f(\mathbf{x})^\top f(\mathbf{x}^-))]$ in the denominator of contrastive learning loss:

$$ g(\mathbf{x}, \{\mathbf{u}_i\}^N_{i=1}, \{\mathbf{v}_i\}_{i=1}^M) = \max\Big\{ \frac{1}{\eta^-}\Big( \frac{1}{N}\sum_{i=1}^N \exp(f(\mathbf{x})^\top f(\mathbf{u}_i)) - \frac{\eta^+}{M}\sum_{i=1}^M \exp(f(\mathbf{x})^\top f(\mathbf{v}_i)) \Big), \exp(-1/\tau) \Big\} $$

其中 $\tau$ 为温度，$\exp(-1/\tau)$ 是 $\mathbb{E}_{\mathbf{x}^-\sim p^-_x}[\exp(f(\mathbf{x})^\top f(\mathbf{x}^-))]$ 的理论下界。

> where $\tau$ is the temperature and $\exp(-1/\tau)$ is the theoretical lower bound of $\mathbb{E}_{\mathbf{x}^-\sim p^-_x}[\exp(f(\mathbf{x})^\top f(\mathbf{x}^-))]$.

最终的去偏对比损失为：

> The final debiased contrastive loss looks like:

$$ \mathcal{L}^{N,M}_\text{debias}(f) = \mathbb{E}_{\mathbf{x},\{\mathbf{u}_i\}^N_{i=1}\sim p;\;\mathbf{x}^+, \{\mathbf{v}_i\}_{i=1}^M\sim p^+} \Big[ -\log\frac{\exp(f(\mathbf{x})^\top f(\mathbf{x}^+)}{\exp(f(\mathbf{x})^\top f(\mathbf{x}^+) + N g(x,\{\mathbf{u}_i\}^N_{i=1}, \{\mathbf{v}_i\}_{i=1}^M)} \Big] $$

![contrastive-debias-t-SNE](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/contrastive-debias-t-SNE.png)

去偏对比学习所学表示的 t-SNE 可视化。（图源：Chuang et al., 2020）

> t-SNE visualization of learned representation with debiased contrastive learning. (Image source: Chuang et al., 2020)

在上述记号基础上，Robinson 等人（2021）修改采样概率，通过按与锚样本的相似度对 $p^-_x(x')$ 加权，以针对难负例。新的采样概率 $q_\beta(x^-)$ 为：

> Following the above annotation, Robinson et al. (2021) modified the sampling probabilities to target at hard negatives by up-weighting the probability $p^-_x(x')$ to be proportional to its similarity to the anchor sample. The new sampling probability $q_\beta(x^-)$ is:

$$ q_\beta(\mathbf{x}^-) \propto \exp(\beta f(\mathbf{x})^\top f(\mathbf{x}^-)) \cdot p(\mathbf{x}^-) $$

其中 $\beta$ 为待调超参数。

> where $\beta$ is a hyperparameter to tune.

我们可用重要性采样估计分母中的第二项 $\mathbb{E}_{\mathbf{x}^- \sim q_\beta} [\exp(f(\mathbf{x})^\top f(\mathbf{x}^-))]$，其中配分函数 $Z_\beta$、$Z^+_\beta$ 均可经验估计。

> We can estimate the second term in the denominator $\mathbb{E}_{\mathbf{x}^- \sim q_\beta} [\exp(f(\mathbf{x})^\top f(\mathbf{x}^-))]$ using importance sampling where both the partition functions $Z_\beta, Z^+_\beta$ can be estimated empirically.

$$ \begin{aligned} \mathbb{E}_{\mathbf{u} \sim q_\beta} [\exp(f(\mathbf{x})^\top f(\mathbf{u}))] &= \mathbb{E}_{\mathbf{u} \sim p} [\frac{q_\beta}{p}\exp(f(\mathbf{x})^\top f(\mathbf{u}))] = \mathbb{E}_{\mathbf{u} \sim p} [\frac{1}{Z_\beta}\exp((\beta + 1)f(\mathbf{x})^\top f(\mathbf{u}))] \\ \mathbb{E}_{\mathbf{v} \sim q^+_\beta} [\exp(f(\mathbf{x})^\top f(\mathbf{v}))] &= \mathbb{E}_{\mathbf{v} \sim p^+} [\frac{q^+_\beta}{p}\exp(f(\mathbf{x})^\top f(\mathbf{v}))] = \mathbb{E}_{\mathbf{v} \sim p} [\frac{1}{Z^+_\beta}\exp((\beta + 1)f(\mathbf{x})^\top f(\mathbf{v}))] \end{aligned} $$

![contrastive-hard-negatives-code](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/contrastive-hard-negatives-code.png)

在 $M=1$ 时计算 NCE 损失、去偏对比损失与难负样本目标的伪代码。（图源：Robinson et al., 2021）

> Pseudo code for computing NCE loss, debiased contrastive loss, and hard negative sample objective when setting $M=1$. (Image source: Robinson et al., 2021)

## 视觉：图像嵌入
> **EN** Vision: Image Embedding

### 图像增强
> **EN** Image Augmentations

视觉领域的大多数对比表示学习方法依赖对样本施加一系列数据增强以创建噪声版本。增强应显著改变视觉外观，但保持语义含义不变。

> Most approaches for contrastive representation learning in the vision domain rely on creating a noise version of a sample by applying a sequence of data augmentation techniques. The augmentation should significantly change its visual appearance but keep the semantic meaning unchanged.

#### 基础图像增强
> **EN** Basic Image Augmentation

在保留语义含义的前提下修改图像的方式很多。可使用下列任一增强，或多种操作的组合。

> There are many ways to modify an image while retaining its semantic meaning. We can use any one of the following augmentation or a composition of multiple operations.

- 随机裁剪，再 resize 回原始尺寸。
- 随机颜色扰动
- 随机高斯模糊
- 随机颜色抖动
- 随机水平翻转
- 随机灰度转换
- 多裁剪增强：使用两个标准分辨率裁剪，并额外采样一组仅覆盖图像小区域的低分辨率裁剪。低分辨率裁剪降低计算成本。（SwAV）
- 以及更多……

> - Random cropping and then resize back to the original size.
> - Random color distortions
> - Random Gaussian blur
> - Random color jittering
> - Random horizontal flip
> - Random grayscale conversion
> - Multi-crop augmentation: Use two standard resolution crops and sample a set of additional low resolution crops that cover only small parts of the image. Using low resolution crops reduces the compute cost. (SwAV)
> - And many more …

#### 增强策略
> **EN** Augmentation Strategies

许多框架用于学习良好的数据增强策略（即多种变换的组合）。以下是若干常见方法。

> Many frameworks are designed for learning good data augmentation strategies (i.e. a composition of multiple transforms). Here are a few common ones.

- AutoAugment（Cubuk, et al. 2018）：受 NAS 启发，AutoAugment 将学习图像分类最佳数据增强操作（如剪切、旋转、反色等）的问题建模为 RL 问题，并搜索在验证集上准确率最高的组合。
- RandAugment（Cubuk et al., 2019）：RandAugment 用单一幅度参数控制不同变换操作的强度，大幅缩小 AutoAugment 的搜索空间。
- PBA（Population based augmentation；Ho et al., 2019）：PBA 将 PBT（Jaderberg et al, 2017）与 AutoAugment 结合，用进化算法并行训练一群体子模型以演化最佳增强策略。
- UDA（Unsupervised Data Augmentation；Xie et al., 2019）：在一组候选增强策略中，UDA 选择使无标签样本与其无标签增强版本上预测分布之间 KL 散度最小的策略。

> - AutoAugment(Cubuk, et al. 2018): Inspired by NAS, AutoAugment frames the problem of learning best data augmentation operations (i.e. shearing, rotation, invert, etc.) for image classification as an RL problem and looks for the combination that leads to the highest accuracy on the evaluation set.
> - RandAugment (Cubuk et al., 2019): RandAugment greatly reduces the search space of AutoAugment by controlling the magnitudes of different transformation operations with a single magnitude parameter.
> - PBA (Population based augmentation; Ho et al., 2019): PBA combined PBT (Jaderberg et al, 2017) with AutoAugment, using the evolutionary algorithm to train a population of children models in parallel to evolve the best augmentation strategies.
> - UDA (Unsupervised Data Augmentation; Xie et al., 2019): Among a set of possible augmentation strategies, UDA selects those to minimize the KL divergence between the predicted distribution over an unlabelled example and its unlabelled augmented version.

#### 图像混合
> **EN** Image Mixture

图像混合方法可从现有数据点构造新的训练样本。

> Image mixture methods can construct new training examples from existing data points.

- Mixup（Zhang et al., 2018）：在全局层面混合，对两张已有图像 $I_1$ 和 $I_2$ 做逐像素加权组合：$I_\text{mixup} \gets \alpha I_1 + (1-\alpha) I_2$，其中 $\alpha \in [0, 1]$。
- Cutmix（Yun et al., 2019）：在区域层面混合，将一张图像的局部区域与另一张图像的其余部分组合生成新样本：$I_\text{cutmix} \gets \mathbf{M}_b \odot I_1 + (1-\mathbf{M}_b) \odot I_2$，其中 $\mathbf{M}_b \in \{0, 1\}^I$ 为二值掩码，$\odot$ 为逐元素乘法。等价于用另一张图像的同一区域填充 cutout（DeVries & Taylor 2017）区域。
- MoCHi（「Mixing of Contrastive Hard Negatives」；Kalantidis et al. 2020）：给定查询 $\mathbf{q}$，MoCHi 维护 $K$ 个负特征的队列 $Q=\{\mathbf{n}_1, \dots, \mathbf{n}_K \}$，并按与查询的相似度 $\mathbf{q}^\top \mathbf{n}$ 降序排序。队列中前 $N$ 项视为最难负例 $Q^N$。合成难例可生成为 $\mathbf{h} = \tilde{\mathbf{h}} / |\tilde{\mathbf{h}}|$，其中 $\tilde{\mathbf{h}} = \alpha\mathbf{n}_i + (1-\alpha) \mathbf{n}_j$，$\alpha \in (0, 1)$。还可通过与查询特征混合得到更难样本：$\mathbf{h}' = \tilde{\mathbf{h}'} / |\tilde{\mathbf{h}'}|_2$，其中 $\tilde{\mathbf{h}'} = \beta\mathbf{q} + (1-\beta) \mathbf{n}_j$，$\beta \in (0, 0.5)$。

> - Mixup (Zhang et al., 2018): It runs global-level mixture by creating a weighted pixel-wise combination of two existing images $I_1$ and $I_2$: $I_\text{mixup} \gets \alpha I_1 + (1-\alpha) I_2$ and $\alpha \in [0, 1]$.
> - Cutmix (Yun et al., 2019): Cutmix does region-level mixture by generating a new example by combining a local region of one image with the rest of the other image. $I_\text{cutmix} \gets \mathbf{M}_b \odot I_1 + (1-\mathbf{M}_b) \odot I_2$, where $\mathbf{M}_b \in \{0, 1\}^I$ is a binary mask and $\odot$ is element-wise multiplication. It is equivalent to filling the cutout (DeVries & Taylor 2017) region with the same region from another image.
> - MoCHi ("Mixing of Contrastive Hard Negatives"; Kalantidis et al. 2020): Given a query $\mathbf{q}$, MoCHi maintains a queue of $K$ negative features $Q=\{\mathbf{n}_1, \dots, \mathbf{n}_K \}$ and sorts these negative features by similarity to the query, $\mathbf{q}^\top \mathbf{n}$, in descending order. The first $N$ items in the queue are considered as the hardest negatives, $Q^N$. Then synthetic hard examples can be generated by $\mathbf{h} = \tilde{\mathbf{h}} / |\tilde{\mathbf{h}}|$ where $\tilde{\mathbf{h}} = \alpha\mathbf{n}_i + (1-\alpha) \mathbf{n}_j$ and $\alpha \in (0, 1)$. Even harder examples can be created by mixing with the query feature, $\mathbf{h}' = \tilde{\mathbf{h}'} / |\tilde{\mathbf{h}'}|_2$ where $\tilde{\mathbf{h}'} = \beta\mathbf{q} + (1-\beta) \mathbf{n}_j$ and $\beta \in (0, 0.5)$.

### 并行增强
> **EN** Parallel Augmentation

该类方法对一张锚图像产生两个噪声版本，并学习使这两个增强样本共享相同嵌入的表示。

> This category of approaches produce two noise versions of one anchor image and aim to learn representation such that these two augmented samples share the same embedding.

#### SimCLR
> **EN** SimCLR

SimCLR（Chen et al, 2020）提出了一个用于对比学习视觉表示的简单框架。它通过在潜空间中用对比损失最大化同一样本不同增强视图之间的一致性，来学习视觉输入的表示。

> SimCLR (Chen et al, 2020) proposed a simple framework for contrastive learning of visual representations. It learns representations for visual inputs by maximizing agreement between differently augmented views of the same sample via a contrastive loss in the latent space.

![SimCLR](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/SimCLR.png)

用于对比学习视觉表示的简单框架。（图源：Chen et al, 2020）

> A simple framework for contrastive learning of visual representations. (Image source: Chen et al, 2020)

1. 随机采样大小为 $N$ 的小批量，每个样本施加两种不同的数据增强，共得到 $2N$ 个增强样本。

> 1. Randomly sample a minibatch of $N$ samples and each sample is applied with two different data augmentation operations, resulting in $2N$ augmented samples in total.

$$ \tilde{\mathbf{x}}_i = t(\mathbf{x}),\quad\tilde{\mathbf{x}}_j = t'(\mathbf{x}),\quad t, t' \sim \mathcal{T} $$

其中两个独立的数据增强算子 $t$ 和 $t'$ 从同一增强族 $\mathcal{T}$ 中采样。数据增强包括随机裁剪、带随机翻转的 resize、颜色扰动与高斯模糊。

> where two separate data augmentation operators, $t$ and $t'$, are sampled from the same family of augmentations $\mathcal{T}$. Data augmentation includes random crop, resize with random flip, color distortions, and Gaussian blur.

1. 给定一对正样本，其余 $2(N-1)$ 个数据点作为负样本。表示由基编码器 $f(.)$ 产生：

> 1. Given one positive pair, other $2(N-1)$ data points are treated as negative samples. The representation is produced by a base encoder $f(.)$:

$$ \mathbf{h}_i = f(\tilde{\mathbf{x}}_i),\quad \mathbf{h}_j = f(\tilde{\mathbf{x}}_j) $$

1. 对比学习损失用余弦相似度 $\text{sim}(.,.)$ 定义。注意损失作用在表示的额外投影层 $g(.)$ 上，而非直接作用在表示空间；但下游任务仅使用表示 $\mathbf{h}$。

> 1. The contrastive learning loss is defined using cosine similarity $\text{sim}(.,.)$. Note that the loss operates on an extra projection layer of the representation $g(.)$ rather than on the representation space directly. But only the representation $\mathbf{h}$ is used for downstream tasks.

$$ \begin{aligned} \mathbf{z}_i &= g(\mathbf{h}_i),\quad \mathbf{z}_j = g(\mathbf{h}_j) \\ \mathcal{L}_\text{SimCLR}^{(i,j)} &= - \log\frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)} \end{aligned} $$

其中 $\mathbb{1}_{[k \neq i]}$ 为指示函数：$k\neq i$ 时为 1，否则为 0。

> where $\mathbb{1}_{[k \neq i]}$ is an indicator function: 1 if $k\neq i$ 0 otherwise.

SimCLR 需要大批量以纳入足够负样本，才能获得良好性能。

> SimCLR needs a large batch size to incorporate enough negative samples to achieve good performance.

![SimCLR-algo](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/SimCLR-algo.png)

SimCLR 算法。（图源：Chen et al, 2020）

> The algorithm for SimCLR. (Image source: Chen et al, 2020).

#### Barlow Twins
> **EN** Barlow Twins

Barlow Twins（Zbontar et al. 2021）将同一样本的两个失真版本输入同一网络提取特征，并学习使两组输出特征之间的互相关矩阵接近单位阵。目标是使同一样本不同失真版本的表示向量相似，同时最小化这些向量之间的冗余。

> Barlow Twins (Zbontar et al. 2021) feeds two distorted versions of samples into the same network to extract features and learns to make the cross-correlation matrix between these two groups of output features close to the identity. The goal is to keep the representation vectors of different distorted versions of one sample similar, while minimizing the redundancy between these vectors.

![barlow-twins](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/barlow-twins.png)

Barlow Twins 学习流程示意。（图源：Zbontar et al. 2021）

> Illustration of Barlow Twins learning pipeline. (Image source: Zbontar et al. 2021).

设 $\mathcal{C}$ 为沿批量维度在两个相同网络输出之间计算的互相关矩阵。$\mathcal{C}$ 为方阵，大小与特征网络输出维度相同。矩阵中每个元素 $\mathcal{C}_{ij}$ 为网络输出向量在索引 $i, j$ 与批量索引 $b$ 处 $\mathbf{z}_{b,i}^A$ 与 $\mathbf{z}_{b,j}^B$ 的余弦相似度，取值在 -1（完全反相关）到 1（完全相关）之间。

> Let $\mathcal{C}$ be a cross-correlation matrix computed between outputs from two identical networks along the batch dimension. $\mathcal{C}$ is a square matrix with the size same as the feature network's output dimensionality. Each entry in the matrix $\mathcal{C}_{ij}$ is the cosine similarity between network output vector dimension at index $i, j$ and batch index $b$, $\mathbf{z}_{b,i}^A$ and $\mathbf{z}_{b,j}^B$, with a value between -1 (i.e. perfect anti-correlation) and 1 (i.e. perfect correlation).

$$ \begin{aligned} \mathcal{L}_\text{BT} &= \underbrace{\sum_i (1-\mathcal{C}_{ii})^2}_\text{invariance term} + \lambda \underbrace{\sum_i\sum_{i\neq j} \mathcal{C}_{ij}^2}_\text{redundancy reduction term} \\ \text{where } \mathcal{C}_{ij} &= \frac{\sum_b \mathbf{z}^A_{b,i} \mathbf{z}^B_{b,j}}{\sqrt{\sum_b (\mathbf{z}^A_{b,i})^2}\sqrt{\sum_b (\mathbf{z}^B_{b,j})^2}} \end{aligned} $$

Barlow Twins 在自监督学习上与 SOTA 方法具有竞争力。它自然避免平凡常数解（即表示坍缩），且对不同训练批量大小具有鲁棒性。

> Barlow Twins is competitive with SOTA methods for self-supervised learning. It naturally avoids trivial constants (i.e. collapsed representations), and is robust to different training batch sizes.

![barlow-twins-algo](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/barlow-twins-algo.png)

Barlow Twins 的 Pytorch 风格伪代码算法。（图源：Zbontar et al. 2021）

> Algorithm of Barlow Twins in Pytorch style pseudo code. (Image source: Zbontar et al. 2021).

#### BYOL
> **EN** BYOL

与上述方法不同，有趣的是，BYOL（Bootstrap Your Own Latent；Grill, et al 2020）声称在不使用负样本的情况下达到新的 SOTA 结果。它依赖两个神经网络——在线网络与目标网络——相互交互并彼此学习。目标网络（参数 $\xi$）与在线网络（参数 $\theta$）架构相同，但权重为 Polyak 平均：$\xi \leftarrow \tau \xi + (1-\tau) \theta$。

> Different from the above approaches, interestingly, BYOL (Bootstrap Your Own Latent; Grill, et al 2020) claims to achieve a new state-of-the-art results without using negative samples. It relies on two neural networks, referred to as online and target networks that interact and learn from each other. The target network (parameterized by $\xi$) has the same architecture as the online one (parameterized by $\theta$), but with polyak averaged weights, $\xi \leftarrow \tau \xi + (1-\tau) \theta$.

目标是学习可用于下游任务的表示 $y$。参数为 $\theta$ 的在线网络包含：

> The goal is to learn a presentation $y$ that can be used in downstream tasks. The online network parameterized by $\theta$ contains:

- 编码器 $f_\theta$；
- 投影器 $g_\theta$；
- 预测器 $q_\theta$。

> - An encoder $f_\theta$;
> - A projector $g_\theta$;
> - A predictor $q_\theta$.

目标网络架构相同，但参数为 $\xi$，通过 Polyak 平均 $\theta$ 更新：$\xi \leftarrow \tau \xi + (1-\tau) \theta$。

> The target network has the same network architecture, but with different parameter $\xi$, updated by polyak averaging $\theta$: $\xi \leftarrow \tau \xi + (1-\tau) \theta$.

![BYOL](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/BYOL.png)

BYOL 模型架构。训练后仅使用 $f_\theta$ 产生表示 $y=f_\theta(x)$，其余模块丢弃。$\text{sg}$ 表示 stop gradient。（图源：Grill, et al 2020）

> The model architecture of BYOL. After training, we only care about $f_\theta$ for producing representation, $y=f_\theta(x)$, and everything else is discarded. $\text{sg}$ means stop gradient. (Image source: Grill, et al 2020)

给定图像 $\mathbf{x}$，BYOL 损失构造如下：

> Given an image $\mathbf{x}$, the BYOL loss is constructed as follows:

- 创建两个增强视图：$\mathbf{v}=t(\mathbf{x}); \mathbf{v}'=t'(\mathbf{x})$，增强 $t \sim \mathcal{T}, t' \sim \mathcal{T}'$ 采样；
- 编码为表示：$\mathbf{y}_\theta=f_\theta(\mathbf{v}), \mathbf{y}'=f_\xi(\mathbf{v}')$；
- 投影到潜变量：$\mathbf{z}_\theta=g_\theta(\mathbf{y}_\theta), \mathbf{z}'=g_\xi(\mathbf{y}')$；
- 在线网络输出预测 $q_\theta(\mathbf{z}_\theta)$；
- 对 $q_\theta(\mathbf{z}_\theta)$ 与 $\mathbf{z}'$ 做 L2 归一化，得 $\bar{q}_\theta(\mathbf{z}_\theta) = q_\theta(\mathbf{z}_\theta) / | q_\theta(\mathbf{z}_\theta) |$ 与 $\bar{\mathbf{z}'} = \mathbf{z}' / |\mathbf{z}'|$；
- 损失 $\mathcal{L}^\text{BYOL}_\theta$ 为 L2 归一化预测 $\bar{q}_\theta(\mathbf{z})$ 与 $\bar{\mathbf{z}'}$ 之间的 MSE；
- 对称损失 $\tilde{\mathcal{L}}^\text{BYOL}_\theta$ 可通过交换 $\mathbf{v}'$ 与 $\mathbf{v}$ 得到，即将 $\mathbf{v}'$ 送入在线网络、$\mathbf{v}$ 送入目标网络；
- 最终损失为 $\mathcal{L}^\text{BYOL}_\theta + \tilde{\mathcal{L}}^\text{BYOL}_\theta$，仅优化参数 $\theta$。

> - Create two augmented views: $\mathbf{v}=t(\mathbf{x}); \mathbf{v}'=t'(\mathbf{x})$ with augmentations sampled $t \sim \mathcal{T}, t' \sim \mathcal{T}'$;
> - Then they are encoded into representations, $\mathbf{y}_\theta=f_\theta(\mathbf{v}), \mathbf{y}'=f_\xi(\mathbf{v}')$;
> - Then they are projected into latent variables, $\mathbf{z}_\theta=g_\theta(\mathbf{y}_\theta), \mathbf{z}'=g_\xi(\mathbf{y}')$;
> - The online network outputs a prediction $q_\theta(\mathbf{z}_\theta)$;
> - Both $q_\theta(\mathbf{z}_\theta)$ and $\mathbf{z}'$ are L2-normalized, giving us $\bar{q}_\theta(\mathbf{z}_\theta) = q_\theta(\mathbf{z}_\theta) / | q_\theta(\mathbf{z}_\theta) |$ and $\bar{\mathbf{z}'} = \mathbf{z}' / |\mathbf{z}'|$;
> - The loss $\mathcal{L}^\text{BYOL}_\theta$ is MSE between L2-normalized prediction $\bar{q}_\theta(\mathbf{z})$ and $\bar{\mathbf{z}'}$;
> - The other symmetric loss $\tilde{\mathcal{L}}^\text{BYOL}_\theta$ can be generated by switching $\mathbf{v}'$ and $\mathbf{v}$; that is, feeding $\mathbf{v}'$ to online network and $\mathbf{v}$ to target network.
> - The final loss is $\mathcal{L}^\text{BYOL}_\theta + \tilde{\mathcal{L}}^\text{BYOL}_\theta$ and only parameters $\theta$ are optimized.

与多数基于对比学习的流行方法不同，BYOL 不使用负样本对。多数自举方法依赖伪标签或簇索引，而 BYOL 直接自举潜表示。

> Unlike most popular contrastive learning based approaches, BYOL does not use negative pairs. Most bootstrapping approaches rely on pseudo-labels or cluster indices, but BYOL directly boostrapps the latent representation.

在没有负样本的情况下 BYOL 仍能工作良好，相当有趣且令人惊讶。后来我读到 Abe Fetterman 与 Josh Albrecht 的一篇博文，他们在复现 BYOL 时强调了两点意外发现：

> It is quite interesting and surprising that without negative samples, BYOL still works well. Later I ran into this post by Abe Fetterman & Josh Albrecht, they highlighted two surprising findings while they were trying to reproduce BYOL:

1. 去掉批归一化后，BYOL 通常不比随机更好。
2. 批归一化的存在隐式带来一种对比学习形式。他们认为使用负样本对避免模型坍缩（即：若对每个数据点都用全零表示会怎样？）很重要。批归一化隐式注入对负样本的依赖，因为无论一批输入多相似，数值都会被重分布（展开为 $\sim \mathcal{N}(0, 1$)），从而防止模型坍缩。若你从事该方向，强烈建议阅读全文。

> 1. BYOL generally performs no better than random when batch normalization is removed.
> 2. The presence of batch normalization implicitly causes a form of contrastive learning. They believe that using negative samples is important for avoiding model collapse (i.e. what if you use all-zeros representation for every data point?). Batch normalization injects dependency on negative samples inexplicitly because no matter how similar a batch of inputs are, the values are re-distributed (spread out $\sim \mathcal{N}(0, 1$) and therefore batch normalization prevents model collapse. Strongly recommend you to read the full article if you are working in this area.

### 记忆库
> **EN** Memory Bank

在每个批次中为大量负样本计算嵌入代价极高。常见做法是将表示存入记忆库，以数据陈旧性换取更低计算成本。

> Computing embeddings for a large number of negative samples in every batch is extremely expensive. One common approach is to store the representation in memory to trade off data staleness for cheaper compute.

#### 带记忆库的实例判别
> **EN** Instance Discrimination with Memoy Bank

实例对比学习（Wu et al, 2018）将类级监督推向极端，把每个实例视为独立的一类，意味着「类别」数与训练集样本数相同。因此无法训练具有如此多头的 softmax 层，而可用 NCE 近似。

> Instance contrastive learning (Wu et al, 2018) pushes the class-wise supervision to the extreme by considering each instance as a distinct class of its own. It implies that the number of "classes" will be the same as the number of samples in the training dataset. Hence, it is unfeasible to train a softmax layer with these many heads, but instead it can be approximated by NCE.

![instance-level-discrimination](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/instance-level-discrimination.png)

实例级对比学习的训练流程。所学嵌入经 L2 归一化。（图源：Wu et al, 2018）

> The training pipeline of instance-level contrastive learning. The learned embedding is L2-normalized. (Image source: Wu et al, 2018)

设 $\mathbf{v} = f_\theta(x)$ 为待学习的嵌入函数，向量归一化为 $|\mathbf{v}|=1$。非参数分类器以温度 $\tau$ 预测样本 $\mathbf{v}$ 属于类 $i$ 的概率：

> Let $\mathbf{v} = f_\theta(x)$ be an embedding function to learn and the vector is normalized to have $|\mathbf{v}|=1$. A non-parametric classifier predicts the probability of a sample $\mathbf{v}$ belonging to class $i$ with a temperature parameter $\tau$:

$$ P(C=i\vert \mathbf{v}) = \frac{\exp(\mathbf{v}_i^\top \mathbf{v} / \tau)}{\sum_{j=1}^n \exp(\mathbf{v}_j^\top \mathbf{v} / \tau)} $$

为避免每次都计算所有样本的表示，他们实现 Memory Bank，在数据库中存储过去迭代的样本表示。设 $V=\{ \mathbf{v}_i \}$ 为记忆库，$\mathbf{f}_i = f_\theta(\mathbf{x}_i)$ 为网络前向得到的特征。比较成对相似度时，可用记忆库中的 $\mathbf{v}_i$ 替代网络前向的 $\mathbf{f}_i$。

> Instead of computing the representations for all the samples every time, they implement an Memory Bank for storing sample representation in the database from past iterations. Let $V=\{ \mathbf{v}_i \}$ be the memory bank and $\mathbf{f}_i = f_\theta(\mathbf{x}_i)$ be the feature generated by forwarding the network. We can use the representation from the memory bank $\mathbf{v}_i$ instead of the feature forwarded from the network $\mathbf{f}_i$ when comparing pairwise similarity.

分母理论上需要访问所有样本的表示，但实践中代价过高。可用随机子集 $\{j_k\}_{k=1}^M$ 的 Monte Carlo 近似：

> The denominator theoretically requires access to the representations of all the samples, but that is too expensive in practice. Instead we can estimate it via Monte Carlo approximation using a random subset of $M$ indices $\{j_k\}_{k=1}^M$.

$$ P(i\vert \mathbf{v}) = \frac{\exp(\mathbf{v}^\top \mathbf{f}_i / \tau)}{\sum_{j=1}^N \exp(\mathbf{v}_j^\top \mathbf{f}_i / \tau)} \simeq \frac{\exp(\mathbf{v}^\top \mathbf{f}_i / \tau)}{\frac{N}{M} \sum_{k=1}^M \exp(\mathbf{v}_{j_k}^\top \mathbf{f}_i / \tau)} $$

由于每类仅一个实例，训练不稳定且波动大。为平滑训练，他们在基于近端优化的方法上为正样本引入额外项。最终 NCE 损失目标为：

> Because there is only one instance per class, the training is unstable and fluctuates a lot. To improve the training smoothness, they introduced an extra term for positive samples in the loss function based on the proximal optimization method. The final NCE loss objective looks like:

$$ \begin{aligned} \mathcal{L}_\text{instance} &= - \mathbb{E}_{P_d}\big[\log h(i, \mathbf{v}^{(t-1)}_i) - \lambda \|\mathbf{v}^{(t)}_i - \mathbf{v}^{(t-1)}_i\|^2_2\big] - M\mathbb{E}_{P_n}\big[\log(1 - h(i, \mathbf{v}'^{(t-1)})\big] \\ h(i, \mathbf{v}) &= \frac{P(i\vert\mathbf{v})}{P(i\vert\mathbf{v}) + MP_n(i)} \text{ where the noise distribution is uniform }P_n = 1/N \end{aligned} $$

其中 $\{ \mathbf{v}^{(t-1)} \}$ 为记忆库中上一迭代的嵌入。迭代间差异 $|\mathbf{v}^{(t)}_i - \mathbf{v}^{(t-1)}_i|^2_2$ 随嵌入收敛而逐渐消失。

> where $\{ \mathbf{v}^{(t-1)} \}$ are embeddings stored in the memory bank from the previous iteration. The difference between iterations $|\mathbf{v}^{(t)}_i - \mathbf{v}^{(t-1)}_i|^2_2$ will gradually vanish as the learned embedding converges.

#### MoCo 与 MoCo-V2
> **EN** MoCo & MoCo-V2

Momentum Contrast（MoCo；He et al, 2019）提供无监督学习视觉表示的框架，将动态字典检索结构化。字典为数据样本编码表示的大型 FIFO 队列。

> Momentum Contrast (MoCo; He et al, 2019) provides a framework of unsupervised learning visual representation as a dynamic dictionary look-up. The dictionary is structured as a large FIFO queue of encoded representations of data samples.

给定查询样本 $\mathbf{x}_q$，经编码器得到查询表示 $\mathbf{q} = f_q(\mathbf{x}_q)$。字典中的键表示列表 $\{\mathbf{k}_1, \mathbf{k}_2, \dots \}$ 由动量编码器编码：$\mathbf{k}_i = f_k (\mathbf{x}^k_i)$。假设其中仅有一个正键 $\mathbf{k}^+$ 与 $\mathbf{q}$ 匹配。论文中对 $\mathbf{x}_q$ 做不同增强的噪声副本得到 $\mathbf{k}^+$。然后在 1 个正样本与 $N-1$ 个负样本上使用温度 $\tau$ 的 InfoNCE 对比损失：

> Given a query sample $\mathbf{x}_q$, we get a query representation through an encoder $\mathbf{q} = f_q(\mathbf{x}_q)$. A list of key representations $\{\mathbf{k}_1, \mathbf{k}_2, \dots \}$ in the dictionary are encoded by a momentum encoder $\mathbf{k}_i = f_k (\mathbf{x}^k_i)$. Let's assume among them there is a single positive key $\mathbf{k}^+$ in the dictionary that matches $\mathbf{q}$. In the paper, they create $\mathbf{k}^+$ using a noise copy of $\mathbf{x}_q$ with different augmentation. Then the InfoNCE contrastive loss with temperature $\tau$ is used over one positive and $N-1$ negative samples:

$$ \mathcal{L}_\text{MoCo} = - \log \frac{\exp(\mathbf{q} \cdot \mathbf{k}^+ / \tau)}{\sum_{i=1}^N \exp(\mathbf{q} \cdot \mathbf{k}_i / \tau)} $$

与记忆库相比，MoCo 的基于队列的字典可复用紧邻前几个 mini-batch 的表示。

> Compared to the memory bank, a queue-based dictionary in MoCo enables us to reuse representations of immediately preceding mini-batches of data.

MoCo 字典作为队列不可微，因此不能靠反向传播更新键编码器 $f_k$。朴素做法是对 $f_q$ 与 $f_k$ 使用同一编码器。MoCo 则提出用动量系数 $m \in [0, 1)$ 的动量更新。设 $f_q$、$f_k$ 参数分别为 $\theta_q$、$\theta_k$：

> The MoCo dictionary is not differentiable as a queue, so we cannot rely on back-propagation to update the key encoder $f_k$. One naive way might be to use the same encoder for both $f_q$ and $f_k$. Differently, MoCo proposed to use a momentum-based update with a momentum coefficient $m \in [0, 1)$. Say, the parameters of $f_q$ and $f_k$ are labeled as $\theta_q$ and $\theta_k$, respectively.

$$ \theta_k \leftarrow m \theta_k + (1-m) \theta_q $$

![MoCo](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/MoCo.png)

Momentum Contrast（MoCo）学习视觉表示的示意。（图源：He et al, 2019）

> Illustration of how Momentum Contrast (MoCo) learns visual representations. (Image source: He et al, 2019)

相较 SimCLR，MoCo 的优势在于将批量大小与负样本数解耦；SimCLR 需要大批量以获得足够负样本，批量减小时性能会下降。

> The advantage of MoCo compared to SimCLR is that MoCo decouples the batch size from the number of negatives, but SimCLR requires a large batch size in order to have enough negative samples and suffers performance drops when their batch size is reduced.

SimCLR 的两项设计——（1）MLP 投影头与（2）更强的数据增强——被证明非常有效。MoCo V2（Chen et al, 2020）结合二者，在不依赖超大批量的情况下获得更好的迁移性能。

> Two designs in SimCLR, namely, (1) an MLP projection head and (2) stronger data augmentation, are proved to be very efficient. MoCo V2 (Chen et al, 2020) combined these two designs, achieving even better transfer performance with no dependency on a very large batch size.

#### CURL
> **EN** CURL

CURL（Srinivas, et al. 2020）将上述思想用于强化学习。它通过对原始观测 $o$ 的两个数据增强版本 $o_q$ 与 $o_k$ 用对比损失匹配嵌入来学习 RL 任务的视觉表示。CURL 主要依赖随机裁剪数据增强。键编码器实现为动量编码器，权重为查询编码器权重的 EMA，与 MoCo 相同。

> CURL (Srinivas, et al. 2020) applies the above ideas in Reinforcement Learning. It learns a visual representation for RL tasks by matching embeddings of two data-augmented versions, $o_q$ and $o_k$, of the raw observation $o$ via contrastive loss. CURL primarily relies on random crop data augmentation. The key encoder is implemented as a momentum encoder with weights as EMA of the query encoder weights, same as in MoCo.

RL 与有监督视觉任务的重要差异在于 RL 依赖连续帧之间的时间一致性。因此 CURL 对每帧栈一致地施加增强，以保留观测的时间结构信息。

> One significant difference between RL and supervised visual tasks is that RL depends on temporal consistency between consecutive frames. Therefore, CURL applies augmentation consistently on each stack of frames to retain information about the temporal structure of the observation.

![CURL](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/CURL.png)

CURL 架构。（图源：Srinivas, et al. 2020）

> The architecture of CURL. (Image source: Srinivas, et al. 2020)

### 特征聚类
> **EN** Feature Clustering

#### DeepCluster
> **EN** DeepCluster

DeepCluster（Caron et al. 2018）通过 k-means 迭代聚类特征，并以簇分配作为伪标签提供监督信号。

> DeepCluster (Caron et al. 2018) iteratively clusters features via k-means and uses cluster assignments as pseudo labels to provide supervised signals.

![deepcluster](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/deepcluster.png)

DeepCluster 方法示意：迭代聚类深度特征并以簇分配为伪标签。（图源：Caron et al. 2018）

> Illustration of DeepCluster method which iteratively clusters deep features and uses the cluster assignments as pseudo-labels. (Image source: Caron et al. 2018)

每轮迭代中，DeepCluster 用先前表示对数据点聚类，再将新簇分配作为新表示的分类目标。但该迭代过程易出现平凡解。虽不使用负样本对，却需要代价高的聚类阶段及专门措施以避免坍缩到平凡解。

> In each iteration, DeepCluster clusters data points using the prior representation and then produces the new cluster assignments as the classification targets for the new representation. However this iterative process is prone to trivial solutions. While avoiding the use of negative pairs, it requires a costly clustering phase and specific precautions to avoid collapsing to trivial solutions.

#### SwAV
> **EN** SwAV

SwAV（Swapping Assignments between multiple Views；Caron et al. 2020）是在线对比学习算法。它从一个增强版本计算图像的 code，并尝试用同一图像的另一增强版本预测该 code。

> SwAV (Swapping Assignments between multiple Views; Caron et al. 2020) is an online contrastive learning algorithm. It computes a code from an augmented version of the image and tries to predict this code using another augmented version of the same image.

![SwAV](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/SwAV.png)

SwAV 与[对比实例学习](#带记忆库的实例判别)的对比。（图源：Caron et al. 2020）

> Comparison of SwAV and [contrastive instance learning](#instance-discrimination-with-memoy-bank). (Image source: Caron et al. 2020)

给定两种不同增强下图像的特征 $\mathbf{z}_t$ 与 $\mathbf{z}_s$，SwAV 计算对应 code $\mathbf{q}_t$、$\mathbf{q}_s$，损失通过交换两个 code 并用 $\ell(.)$ 度量特征与 code 的拟合程度来量化。

> Given features of images with two different augmentations, $\mathbf{z}_t$ and $\mathbf{z}_s$, SwAV computes corresponding codes $\mathbf{q}_t$ and $\mathbf{q}_s$ and the loss quantifies the fit by swapping two codes using $\ell(.)$ to measure the fit between a feature and a code.

$$ \mathcal{L}_\text{SwAV}(\mathbf{z}_t, \mathbf{z}_s) = \ell(\mathbf{z}_t, \mathbf{q}_s) + \ell(\mathbf{z}_s, \mathbf{q}_t) $$

交换拟合预测依赖预测 code 与 $K$ 个可训练原型向量 $\mathbf{C} = \{\mathbf{c}_1, \dots, \mathbf{c}_K\}$ 之间的交叉熵。原型矩阵在不同 batch 间共享，表示每个实例应聚类到的锚簇。

> The swapped fit prediction depends on the cross entropy between the predicted code and a set of $K$ trainable prototype vectors $\mathbf{C} = \{\mathbf{c}_1, \dots, \mathbf{c}_K\}$. The prototype vector matrix is shared across different batches and represents anchor clusters that each instance should be clustered to.

$$ \ell(\mathbf{z}_t, \mathbf{q}_s) = - \sum_k \mathbf{q}^{(k)}_s\log\mathbf{p}^{(k)}_t \text{ where } \mathbf{p}^{(k)}_t = \frac{\exp(\mathbf{z}_t^\top\mathbf{c}_k / \tau)}{\sum_{k'}\exp(\mathbf{z}_t^\top \mathbf{c}_{k'} / \tau)} $$

在含 $B$ 个特征向量 $\mathbf{Z} = [\mathbf{z}_1, \dots, \mathbf{z}_B]$ 的 mini-batch 中，特征与原型向量的映射矩阵为 $\mathbf{Q} = [\mathbf{q}_1, \dots, \mathbf{q}_B] \in \mathbb{R}_+^{K\times B}$。希望最大化特征与原型的相似度：

> In a mini-batch containing $B$ feature vectors $\mathbf{Z} = [\mathbf{z}_1, \dots, \mathbf{z}_B]$, the mapping matrix between features and prototype vectors is defined as $\mathbf{Q} = [\mathbf{q}_1, \dots, \mathbf{q}_B] \in \mathbb{R}_+^{K\times B}$. We would like to maximize the similarity between the features and the prototypes:

$$ \begin{aligned} \max_{\mathbf{Q}\in\mathcal{Q}} &\text{Tr}(\mathbf{Q}^\top \mathbf{C}^\top \mathbf{Z}) + \varepsilon \mathcal{H}(\mathbf{Q}) \\ \text{where }\mathcal{Q} &= \big\{ \mathbf{Q} \in \mathbb{R}_{+}^{K \times B} \mid \mathbf{Q}\mathbf{1}_B = \frac{1}{K}\mathbf{1}_K, \mathbf{Q}^\top\mathbf{1}_K = \frac{1}{B}\mathbf{1}_B \big\} \end{aligned} $$

其中 $\mathcal{H}$ 为熵，$\mathcal{H}(\mathbf{Q}) = - \sum_{ij} \mathbf{Q}_{ij} \log \mathbf{Q}_{ij}$，控制 code 的平滑度。系数 $\epsilon$ 不宜过大，否则所有样本会被均匀分配到所有簇。$\mathbf{Q}$ 的候选解要求每行和为 $1/K$、每列和为 $1/B$，从而强制每个原型平均至少被选中 $B/K$ 次。

> where $\mathcal{H}$ is the entropy, $\mathcal{H}(\mathbf{Q}) = - \sum_{ij} \mathbf{Q}_{ij} \log \mathbf{Q}_{ij}$, controlling the smoothness of the code. The coefficient $\epsilon$ should not be too large; otherwise, all the samples will be assigned uniformly to all the clusters. The candidate set of solutions for $\mathbf{Q}$ requires every mapping matrix to have each row sum up to $1/K$ and each column to sum up to $1/B$, enforcing that each prototype gets selected at least $B/K$ times on average.

SwAV 依赖迭代 Sinkhorn-Knopp 算法（Cuturi 2013）求解 $\mathbf{Q}$。

> SwAV relies on the iterative Sinkhorn-Knopp algorithm (Cuturi 2013) to find the solution for $\mathbf{Q}$.

### 利用有监督数据集
> **EN** Working with Supervised Datasets

#### CLIP
> **EN** CLIP

CLIP（Contrastive Language-Image Pre-training；Radford et al. 2021）联合训练文本编码器与图像特征提取器，预训练任务为预测哪段 caption 与哪张图像配对。

> CLIP (Contrastive Language-Image Pre-training; Radford et al. 2021) jointly trains a text encoder and an image feature extractor over the pretraining task that predicts which caption goes with which image.

![CLIP](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/CLIP.png)

CLIP 在图文对上的对比预训练示意。（图源：Radford et al. 2021）

> Illustration of CLIP contrastive pre-training over text-image pairs. (Image source: Radford et al. 2021)

给定 $N$ 个（图像，文本）对的 batch，CLIP 计算该 batch 内全部 $N\times N$ 个（图像，文本）候选之间的稠密余弦相似度矩阵。文本与图像编码器联合训练，最大化 $N$ 对正确（图像，文本）关联的相似度，同时最小化 $N(N-1)$ 对错误关联的相似度，通过对稠密矩阵的对称交叉熵损失实现。

> Given a batch of $N$ (image, text) pairs, CLIP computes the dense cosine similarity matrix between all $N\times N$ possible (image, text) candidates within this batch. The text and image encoders are jointly trained to maximize the similarity between $N$ correct pairs of (image, text) associations while minimizing the similarity for $N(N-1)$ incorrect pairs via a symmetric cross entropy loss over the dense matrix.

CLIP 的 Numpy 风格伪代码见下图。

> See the numy-like pseudo code for CLIP in

![CLIP-algo](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/CLIP-algo.png)

CLIP 的 Numpy 风格伪代码算法。（图源：Radford et al. 2021）

> CLIP algorithm in Numpy style pseudo code. (Image source: Radford et al. 2021)

相较上述学习良好视觉表示的方法，CLIP 的特别之处在于「将自然语言作为训练信号」。它需要可获知哪段文本与哪张图像匹配的有监督数据集，在 4 亿（文本，图像）对上训练，数据来自互联网。查询列表包含英文维基百科中出现至少 100 次的全部词。有趣的是，他们发现基于 Transformer 的语言模型在零样本 ImageNet 分类上比词袋（BoW）文本编码器慢 3 倍。采用对比目标而非预测与图像关联的确切词（图像描述任务常用方法）可再将数据效率提升约 4 倍。

> Compared to other methods above for learning good visual representation, what makes CLIP really special is "the appreciation of using natural language as a training signal". It does demand access to supervised dataset in which we know which text matches which image. It is trained on 400 million (text, image) pairs, collected from the Internet. The query list contains all the words occurring at least 100 times in the English version of Wikipedia. Interestingly, they found that Transformer-based language models are 3x slower than a bag-of-words (BoW) text encoder at zero-shot ImageNet classification. Using contrastive objective instead of trying to predict the exact words associated with images (i.e. a method commonly adopted by image caption prediction tasks) can further improve the data efficiency another 4x.

![CLIP-efficiency](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/CLIP-efficiency.png)

词袋文本编码与对比训练目标可带来数倍数据效率提升。（图源：Radford et al. 2021）

> Using bag-of-words text encoding and contrastive training objectives can bring in multiple folds of data efficiency improvement. (Image source: Radford et al. 2021)

CLIP 产生良好的视觉表示，可非平凡地迁移到许多 CV 基准，结果与有监督基线具有竞争力。在测试的迁移任务中，CLIP 在非常细粒度分类以及计数物体数量等抽象或系统性任务上表现吃力。CLIP 模型的迁移性能与模型计算量平滑相关。

> CLIP produces good visual representation that can non-trivially transfer to many CV benchmark datasets, achieving results competitive with supervised baseline. Among tested transfer tasks, CLIP struggles with very fine-grained classification, as well as abstract or systematic tasks such as counting the number of objects. The transfer performance of CLIP models is smoothly correlated with the amount of model compute.

#### 有监督对比学习
> **EN** Supervised Contrastive Learning

交叉熵损失存在若干已知问题，如对噪声标签缺乏鲁棒性、可能出现较差间隔。对交叉熵的改进包括 curated 更好的训练数据，如标签平滑与数据增强。有监督对比损失（Supervised Contrastive Loss；Khosla et al. 2021）旨在比交叉熵更有效地利用标签信息，要求同类归一化嵌入彼此更接近，不同类比嵌入更远离。

> There are several known issues with cross entropy loss, such as the lack of robustness to noisy labels and the possibility of poor margins. Existing improvement for cross entropy loss involves the curation of better training data, such as label smoothing and data augmentation. Supervised Contrastive Loss (Khosla et al. 2021) aims to leverage label information more effectively than cross entropy, imposing that normalized embeddings from the same class are closer together than embeddings from different classes.

![sup-con](file:///data/zhangchangtian/project/ai-learn/docs/reports/contrastive/figures/LilianWeng_contrastive/sup-con.png)

有监督与自监督对比损失对比。有监督对比学习除增强版本外，还将同类不同样本视为正例。（图源：Khosla et al. 2021）

> Supervised vs self-supervised contrastive losses. Supervised contrastive learning considers different samples from the same class as positive examples, in addition to augmented versions. (Image source: Khosla et al. 2021)

给定随机采样的 $n$ 个（图像，标签）对 $\{\mathbf{x}_i, y_i\}_{i=1}^n$，对每个样本施加两次随机增强，得到 $2n$ 个训练对 $\{\tilde{\mathbf{x}}_i, \tilde{y}_i\}_{i=1}^{2n}$。

> Given a set of randomly sampled $n$ (image, label) pairs, $\{\mathbf{x}_i, y_i\}_{i=1}^n$, $2n$ training pairs can be created by applying two random augmentations of every sample, $\{\tilde{\mathbf{x}}_i, \tilde{y}_i\}_{i=1}^{2n}$.

有监督对比损失 $\mathcal{L}_\text{supcon}$ 利用多个正负样本，与 soft nearest-neighbor 损失非常相似：

> Supervised contrastive loss $\mathcal{L}_\text{supcon}$ utilizes multiple positive and negative samples, very similar to soft nearest-neighbor loss:

$$ \mathcal{L}_\text{supcon} = - \sum_{i=1}^{2n} \frac{1}{2 \vert N_i \vert - 1} \sum_{j \in N(y_i), j \neq i} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_j / \tau)}{\sum_{k \in I, k \neq i}\exp({\mathbf{z}_i \cdot \mathbf{z}_k / \tau})} $$

其中 $\mathbf{z}_k=P(E(\tilde{\mathbf{x}_k}))$，$E(.)$ 为编码网络（增强图像映射为向量），$P(.)$ 为投影网络（一向量映射为另一向量）。$N_i= \{j \in I: \tilde{y}_j = \tilde{y}_i \}$ 为标签 $y_i$ 的样本索引集。向集合 $N_i$ 纳入更多正样本可提升结果。

> where $\mathbf{z}_k=P(E(\tilde{\mathbf{x}_k}))$, in which $E(.)$ is an encoder network (augmented image mapped to vector) $P(.)$ is a projection network (one vector mapped to another). $N_i= \{j \in I: \tilde{y}_j = \tilde{y}_i \}$ contains a set of indices of samples with label $y_i$. Including more positive samples into the set $N_i$ leads to improved results.

根据其实验，有监督对比损失：

> According to their experiments, supervised contrastive loss:

- 优于基础交叉熵，但幅度较小。
- 在鲁棒性基准（ImageNet-C，对 ImageNet 施加常见自然扰动如噪声、模糊与对比度变化）上优于交叉熵。
- 对超参数变化更不敏感。

> - does outperform the base cross entropy, but only by a small amount.
> - outperforms the cross entropy on robustness benchmark (ImageNet-C, which applies common naturally occuring perturbations such as noise, blur and contrast changes to the ImageNet dataset).
> - is less sensitive to hyperparameter changes.
