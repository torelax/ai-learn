import requests

url = 'https://api.jina.ai/v1/embeddings'

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer xxxxxxx'  # 替换为你的API密钥
}

data = {
    "model": "jina-embeddings-v5-text-small",
    # "task": "text-matching",
    "task": "retrieval.query",
    "dimensions": 1024,
    "input": [
        {"text": "关闭名称为小红的灯"},
        {"text": "猫眼的红外灯可以关闭吗"},
        # {"text": "海滩上美丽的日落"},
        # {"text": "浜辺に沈む美しい夕日"}
    ]
}

response = requests.post(url, json=data, headers=headers)
print(response.json())