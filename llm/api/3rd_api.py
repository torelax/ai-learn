import random
import dashscope
import time

from abc import ABCMeta, abstractmethod
from http import HTTPStatus
from dashscope import Generation
from volcenginesdkarkruntime import Ark


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        results = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time-start_time} seconds to execute.")
        return results

    return wrapper


class BaseChat:
    __meta_class__ = ABCMeta

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        try:
            self.model_select = self.kwargs["model_select"]
            self.model_id = self.kwargs["model_id"]
            self.temperature = self.kwargs["temperature"]
            self.max_tokens = self.kwargs["max_new_tokens"]
        except Exception as e:
            raise ValueError(
                f"[{self.__class__.__name__}] parser model parmas error, reset from config please. {e}"
            )

    @abstractmethod
    def api(self):
        pass

    @abstractmethod
    def stream_api(self):
        pass


## AliQwenChat
dashscope.api_key = "xxx"


class AliQwenChat(BaseChat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @timer
    def api(self, messages, enable_search=False):
        response = Generation.call(
            model=self.model_id,
            messages=messages,
            # plus默认最大输出2000， turbo默认最大输出1500
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            enable_search=enable_search,
            # 设置随机数种子seed，如果没有设置，则随机数种子默认为1234
            seed=random.randint(1, 10000),
            # 将输出设置为"message"格式
            result_format="message",
        )
        if response.status_code != HTTPStatus.OK:
            print(f"[{self.model_select}]: \
                    Request id: {response.request_id}\
                    Status code: {response.status_code}\
                    error code: {response.code}\
                    error message: {response.message}", 40)
        print(
            f"[{self.model_select}] request_id: {response.request_id}, responses: {response}"
        )
        content = response.output.choices[0]["message"]["content"]
        return content

    def stream_api(self, messages, enable_search=False):
        responses = Generation.call(
            model=self.model_id,
            messages=messages,
            # plus默认最大输出2000， turbo默认最大输出1500
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            enable_search=enable_search,
            # 设置随机数种子seed，如果没有设置，则随机数种子默认为1234
            seed=random.randint(1, 10000),
            # 将输出设置为"message"格式
            result_format="message",
            incremental_output=True,
            stream=True,
        )
        for rsp in responses:
            flag = True
            if rsp.status_code != HTTPStatus.OK:
                raise TypeError(f"""[{self.model_select}]:
                    Request id: {rsp.request_id}
                    Status code: {rsp.status_code}
                    error code: {rsp.code}
                    error message: {rsp.message}""")
            content = rsp.output.choices[0]["message"]["content"]
            finish_reason = rsp.output.choices[0]["finish_reason"]
            fr = "stop" if finish_reason in ["stop", "length"] else finish_reason
            res = {
                "status": flag,
                "content": content,
                "finish_reason": fr,
                "request_id": rsp.request_id,
            }
            print(f"[{self.model_select}] {rsp}")
            yield res


## DoubaoChat
class DoubaoChat(BaseChat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = Ark(
            ak="xxx",
            sk="xxx",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )

    @timer
    def api(self, messages):
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=120,
        )
        print(
            f"[{self.model_select}] request_id: {completion.id}, responses: {completion}"
        )
        responses = completion.choices[0].message.content
        return responses

    def stream_api(self, messages, enable_search=False):
        responses = self.client.chat.completions.create(
            model=self.model_id, messages=messages, stream=True
        )
        for rsp in responses:
            flag = True
            if not rsp.choices:
                flag = False
            content = rsp.choices[0].delta.content
            finish_reason = rsp.choices[0].finish_reason
            fr = (
                "stop"
                if finish_reason in ["stop", "length", "content_filter"]
                else finish_reason
            )
            res = {
                "status": flag,
                "content": content,
                "finish_reason": fr,
                "request_id": rsp.id,
            }
            print(f"[{self.model_select}] {rsp}")
            yield res
