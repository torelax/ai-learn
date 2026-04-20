# mcp 应用于 llm
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(api_key="your-api-key", base_url="http://localhost:8000/v1")


async def run_mcp_client_and_chat():
    # 1. 配置 MCP 服务
    url = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse"
    headers = {"Authorization": "Bearer your-api-key"}

    async with sse_client(url=url, headers=headers) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # 2. 获取 MCP 工具列表
            mcp_tools = await session.list_tools()

            # 3. 将 MCP 工具转换为 OpenAI API 格式
            openai_tools = []
            for tool in mcp_tools.tools:
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )

            # 4. 构建对话消息
            messages = [
                {
                    "role": "system",
                    "content": "你是一个很有帮助的助手。请根据用户需求调用合适的工具。",
                },
                {"role": "user", "content": "请帮我搜索最新的科技新闻。"},
            ]

            # 5. 调用 LLM
            completion = client.chat.completions.create(
                model="Qwen3-30B-A3B-Instruct-2507-FP8",
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )

            # 6. 处理工具调用
            if completion.choices[0].message.tool_calls:
                tool_call = completion.choices[0].message.tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = eval(tool_call.function.arguments)

                print(f"模型请求调用工具: {tool_name}，参数: {tool_args}")

                # 7. 执行 MCP 工具调用
                result = await session.call_tool(tool_name, arguments=tool_args)
                print(f"工具执行结果: {result}")

                # 8. 将结果添加到对话
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": tool_call.id,
                    }
                )

                # 9. 获取最终回复
                final_completion = client.chat.completions.create(
                    model="Qwen3-30B-A3B-Instruct-2507-FP8",
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="none",
                )

                print("最终回复:")
                print(final_completion.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(run_mcp_client_and_chat())
