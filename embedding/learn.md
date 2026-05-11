
## 训练

### 开始训练
```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 1. 加载预训练模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. 准备训练数据
train_examples = [
    InputExample(texts=['这是第一句话', '这是相似的句子'], label=0.8),
    InputExample(texts=['这是第二句话', '这是不相关的句子'], label=0.1)
]

# 3. 创建数据加载器
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# 4. 定义损失函数
train_loss = losses.CosineSimilarityLoss(model)

# 5. 开始训练
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=100,
    output_path='./my_embedding_model'
)
```
### 难负例挖掘

```python
train_loss = losses.MultipleNegativesRankingLoss(
    model,
    negative_mining_technique='hard',
    update_negative_embeddings=True
)

# 混合精度训练
model.fit(..., fp16=True)

# 梯度累积
model.fit(..., gradient_accumulation_steps=4)
```
### 超参优化
| 超参数       | 推荐范围       | 说明                |
|--------------|---------------|--------------------|
| batch_size   | 16-256        | 根据显存调整        |
| learning_rate| 1e-5到5e-5    | 预训练模型需小学习率|
| warmup_steps | 总步数的10%   | 避免初始震荡        |
| epoch        | 3-10          | 防止过拟合          |

### 评估
```python
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# 准备评估数据
evaluator = EmbeddingSimilarityEvaluator(
    sentences1, sentences2, scores)

# 运行评估
model.evaluate(evaluator)

```