import os
from pathlib import Path

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

API_KEY = os.getenv("MASS_API_KEY")

try:
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://maas.hikvision.com.cn/v1"
    )
    completion = client.chat.completions.create(
    ##MODELNAME 信息：模型支持体验，则在模型广场-模型详情，模型名称下方复制按钮，复制填充；
    ##若模型不支持体验则需要自行部署，请前往部署服务部署该模型，部署成功后，复制部署服务model信息填充
 
        model="DeepSeek-V4-Flash-0731",  
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'ssh的—R可以让远程服务器对某一整个域名的访问都被本地机器代理吗？命令该怎么写？'}
        ]
    )
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"错误信息：{e}")
    print("请参考错误码表")