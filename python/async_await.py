import asyncio
import time

async def visit_url(url, response_time):
    """访问 url"""
    await asyncio.sleep(response_time)
    return f"访问{url}, 已得到返回结果"

async def run_task():
    """收集子任务"""
    task = visit_url('http://wangzhen.com', 2)
    task_2 = visit_url('http://another', 3)
    await asyncio.run(task)
    await asyncio.run(task_2)
    
start_time = time.perf_counter()
asyncio.run(run_task())
print(f"消耗时间：{time.perf_counter() - start_time}")