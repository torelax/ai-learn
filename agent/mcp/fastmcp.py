#!/usr/bin/env python3
"""
FastMCP 完整示例：包含Server和Client实现
FastMCP是一个轻量级的MCP(Model Context Protocol)服务构建库，可以快速开发AI代理可用的工具服务
"""

from fastmcp import FastMCP
import asyncio
import time
import platform
from typing import List


# ==============================
# 1. 服务端实现
# ==============================
def create_server() -> FastMCP:
    """
    创建MCP服务器实例，注册可用工具
    返回: 配置好的FastMCP服务器对象
    """
    # 初始化FastMCP服务器，设置服务名称
    mcp = FastMCP(name="示例工具服务")

    # 注册工具函数：使用@mcp.tool装饰器
    @mcp.tool
    def add(a: int, b: int) -> int:
        """
        两个整数相加
        参数:
            a: 第一个整数
            b: 第二个整数
        返回: 两个数的和
        """
        return a + b

    @mcp.tool
    def get_current_time() -> str:
        """
        获取当前系统时间
        返回: 格式化的当前时间字符串
        """
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    @mcp.tool
    def get_system_info() -> dict:
        """
        获取系统信息
        返回: 包含系统、主机名、Python版本等信息的字典
        """
        return {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
        }

    @mcp.tool
    def sort_list(numbers: List[int], reverse: bool = False) -> List[int]:
        """
        对整数列表进行排序
        参数:
            numbers: 要排序的整数列表
            reverse: 是否降序排列，默认False(升序)
        返回: 排序后的列表
        """
        return sorted(numbers, reverse=reverse)

    return mcp


# ==============================
# 2. 客户端实现
# ==============================
async def run_client():
    """
    MCP客户端示例：连接到本地MCP服务并调用工具
    """
    # 尝试不同的导入方式，兼容不同版本的fastmcp
    try:
        from fastmcp import Client
    except ImportError:
        try:
            from fastmcp.client import Client
        except ImportError:
            raise ImportError("请安装最新版本的fastmcp: pip install -U fastmcp")

    # 连接到本地运行的MCP服务
    async with Client.stdio("python", __file__, "server") as client:
        print("=== MCP 客户端示例 ===")
        print(f"已连接到服务: {client.server_info.name}")
        print("可用工具列表:")
        for tool in client.tools:
            print(f"  - {tool.name}: {tool.description}")
        print("\n" + "=" * 50 + "\n")

        # 示例1: 调用add工具
        print("1. 调用add(123, 456):")
        result = await client.call_tool("add", arguments={"a": 123, "b": 456})
        print(f"   结果: {result}")

        # 示例2: 调用get_current_time工具
        print("\n2. 调用get_current_time():")
        result = await client.call_tool("get_current_time")
        print(f"   结果: {result}")

        # 示例3: 调用get_system_info工具
        print("\n3. 调用get_system_info():")
        result = await client.call_tool("get_system_info")
        print("   结果:")
        for key, value in result.items():
            print(f"     {key}: {value}")

        # 示例4: 调用sort_list工具
        print("\n4. 调用sort_list([5, 2, 9, 1, 5, 6], reverse=True):")
        result = await client.call_tool(
            "sort_list", arguments={"numbers": [5, 2, 9, 1, 5, 6], "reverse": True}
        )
        print(f"   结果: {result}")


# ==============================
# 3. 入口逻辑
# ==============================
if __name__ == "__main__":
    import sys

    # 根据命令行参数判断是运行server还是client
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # 运行服务端: python fastmcp.py server
        server = create_server()
        # 以stdio模式运行服务(适合AI代理调用)
        server.run()
    elif len(sys.argv) > 1 and sys.argv[1] == "client":
        # 运行客户端: python fastmcp.py client
        asyncio.run(run_client())
    else:
        # 没有参数时显示帮助信息
        print("FastMCP 示例使用方法:")
        print("  运行服务端: python fastmcp.py server")
        print("  运行客户端: python fastmcp.py client")
        print("\n说明:")
        print("  服务端启动后会在stdio模式运行，可以被Claude等AI代理直接调用")
        print("  客户端会连接到本地服务端并演示调用所有注册的工具")
