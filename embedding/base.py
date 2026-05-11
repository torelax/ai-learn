# 概念性代码示例，具体API以官方库为准
import os
from dotenv import load_dotenv

load_dotenv('../.env')  # 加载环境变量

from sentence_transformers import SentenceTransformer


# 加载embedding模型
model_id = "bge-base-zh-v1.5"  # 替换成你想使用的模型ID
# model = SentenceTransformer(f'BAAI/{model_id}', device='cuda')
model = SentenceTransformer(f'BAAI/{model_id}', device='cuda')

embs = model.encode("你好，世界！")
print(embs)