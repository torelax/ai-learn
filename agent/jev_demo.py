#!/usr/bin/env python3
"""Jev 入门：一段客服工单，同时问 Choice、Score 和 Noul，再由 Python 分流。

仅使用 Python 3.10+ 标准库，不改项目依赖，不下载模型。

先看请求（不联网、不需要 key）：
    python agent/jev_demo.py --dry-run

真实调用（需要自己的 TypeSafe 账户及 API key，按服务价格计费）：
    export TYPESAFE_API_KEY='你的 API key'
    python agent/jev_demo.py
    python agent/jev_demo.py --text '我的服务一直报错，请转人工处理。'
    python agent/jev_demo.py --model jev-1.14.0

官方文档：https://docs.typesafe.ai/introduction/quickstart
接口：https://docs.typesafe.ai/api/system-one
概率与 confidence：https://docs.typesafe.ai/confidence

脚本只打印分流建议，不执行退款、发消息等操作。阈值只是演示值，未经业务校准。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_TEXT = "同一笔订阅被重复扣费了，请今天帮我处理退款。我已经联系三次了！"


def build_request(text: str, model: str) -> dict[str, Any]:
    """state 是待判断材料；三个问题互不依赖，可在同一次请求内评估。"""
    return {
        "model": model,
        "state": {"ticket": text},
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle the customer's main request?",
                "criteria": {
                    "billing": "Invoices, duplicate charges, payments, or refunds.",
                    "technical": "Bugs, service outages, or technical troubleshooting.",
                    "other": "Anything outside billing or technical support.",
                },
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": [
                    "Calm or neutral.",
                    "Frustrated but civil.",
                    "Very angry or explicitly threatening to leave.",
                ],
            },
            "wants_human": {
                "type": "noul",
                "instructions": "Does the customer explicitly ask for a human agent?",
            },
        },
    }


def evaluate(payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    request = Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    # 入门脚本发送一次请求。生产调用可用官方 SDK 的退避重试和连接复用。
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("API 响应应为 JSON 对象。")
    return result


def bounded_number(value: Any, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("预期数字字段，收到其他类型。")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= upper:
        raise ValueError(f"数字字段必须在 0 到 {upper:g} 之间。")
    return number


def routing_suggestion(result: dict[str, Any]) -> str:
    """读取原语的实际字段；Noul 没有单独的 confidence，Score 允许小数。"""
    answers = result["answers"]
    department = answers["department"]
    frustration = answers["frustration"]
    wants_human = answers["wants_human"]
    if (department["type"], frustration["type"], wants_human["type"]) != (
        "choice", "score", "noul"
    ):
        raise ValueError("响应问题类型与请求不一致。")
    team = department["choice"]
    if team not in {"billing", "technical", "other"}:
        raise ValueError("响应包含未声明的部门。")
    confidence = bounded_number(department["confidence"], 1)
    severity = bounded_number(frustration["score"], 2)
    severity_confidence = bounded_number(frustration["confidence"], 1)
    human_probability = bounded_number(wants_human["noul"], 1)

    # 这是解释 API 的示例策略；并不表示这些阈值已满足任何真实服务要求。
    if human_probability >= 0.8:
        return f"转人工：明确请求人工的概率较高；候选部门为 {team}。"
    if human_probability > 0.2:
        return "先澄清是否需要人工：Noul 概率位于演示策略的模糊区间。"
    if confidence < 0.75 or team == "other":
        return "转人工分诊：部门不够明确，或不在当前业务分类范围。"
    if severity >= 1.5 and severity_confidence >= 0.7:
        return f"交由 {team} 的人工坐席优先处理：情绪评分较高且较集中。"
    return f"将工单分配给 {team} 队列，由后续业务流程处理。"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", default=DEFAULT_TEXT, help="要评估的工单文本")
    parser.add_argument("--model", default="jev-latest", help="默认使用稳定别名；响应会显示实际版本")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP 超时秒数，默认 30")
    parser.add_argument("--dry-run", action="store_true", help="只打印请求，不调用模型")
    args = parser.parse_args()
    if not args.text.strip() or not args.model.strip():
        parser.error("--text 和 --model 不能为空。")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout 必须是有限正数。")

    payload = build_request(args.text, args.model)
    if args.dry_run:
        print("仅展示请求；未调用 Jev，没有模型预测结果：")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        print("请设置 TYPESAFE_API_KEY；无 key 时可先运行 --dry-run。", file=sys.stderr)
        return 2
    try:
        started = time.perf_counter()
        result = evaluate(payload, api_key, args.timeout)
        elapsed_ms = (time.perf_counter() - started) * 1000
        suggestion = routing_suggestion(result)
    except HTTPError as exc:
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        print(f"Jev API 返回 HTTP {exc.code}；请按官方文档检查账户、参数或限流。", file=sys.stderr)
        if retry_after:
            print(f"服务建议 Retry-After: {retry_after}", file=sys.stderr)
        return 1
    except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"调用或响应解析失败：{exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n本次客户端请求耗时：{elapsed_ms:.1f} ms（单次观测，不是性能基准）")
    print(f"实际模型：{result.get('model', '响应未提供版本')}")
    print(f"Python 分流建议：{suggestion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
