# 概念性代码示例，具体API以官方库为准
import os
from dotenv import load_dotenv

load_dotenv('../../.env')  # 加载环境变量

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# 1. 模型加载与初始化 (Using Hugging Face Transformers)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
model.eval()  # 设置为评估模式

# 2. 输入处理
query = "什么是人工智能？"
document1 = "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
document2 = "深度学习是机器学习的一个领域，它本身就是人工智能的一个分支。"

# 构建查询-文档对
sentence_pairs = [[query, document1], [query, document2]]

# 3. 前向传播与得分计算
with torch.no_grad():  # 推理时不需要计算梯度
    inputs = tokenizer(
        sentence_pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )
    # inputs 包含 'input_ids', 'attention_mask', 'token_type_ids' (如果模型使用)

    outputs = model(**inputs)
    # outputs.logits 是原始得分 (通常是一个二维张量，形状为 [batch_size, num_labels])
    # 对于Reranker，通常num_labels=1 (回归任务) 或 num_labels=2 (分类任务，取相关类的概率)
    # BGE-Reranker 通常直接输出一个标量相关性分数，可能需要 sigmoid
    #
    # scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

    scores = torch.sigmoid(
        outputs.logits.squeeze(-1)
    )  # squeeze去掉最后一个维度，然后sigmoid

    # 或者直接使用 FlagEmbedding 库提供的 compute_score 方法，它封装了这些细节
    # from FlagEmbedding import FlagReranker
    # reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True) # use_fp16 可加速
    # scores = reranker.compute_score(sentence_pairs)

print(f"'{query}' 与 '{document1}' 的相关性得分: {scores[0].item()}")
print(f"'{query}' 与 '{document2}' 的相关性得分: {scores[1].item()}")
