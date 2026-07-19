#!/usr/bin/env python
"""DeepSeek tool-calling 探针：实测 function calling 稳定性。

跑法
----
1. ``cp .env.example .env`` 并填入 ``DEEPSEEK_API_KEY``
2. ``uv run python experiments/deepseek_tool_probe.py``

输出每条 prompt 的命中情况（是否返回 tool_calls / 参数是否合法 / 延迟），
结尾给命中率。结果用于 ``docs/deepseek-tool-calling-eval.md`` 与 Phase 3 防御设计。
"""

from __future__ import annotations

import time

from customer_service.core.llm import LLM, Settings

TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "query_order",
        "description": "查询订单状态与物流。需要订单号。",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号，如 ORD20240001"},
            },
            "required": ["order_id"],
        },
    },
}]

SYSTEM = "你是订单客服。需要查询订单时必须调用 query_order，不要直接编造答案。"

PROMPTS = [
    "帮我查一下订单 ORD20240001 的物流",
    "订单 ORD20240002 发货了吗",
    "ORD20240003 到哪了",
    "我想看看 ORD20240001 现在什么状态",
    "查个单子 ORD20240002",
    "帮我看看 ORD20240003 发没发",
    "ORD20240001 现在在哪儿",
]


def main() -> None:
    settings = Settings()
    if not settings.api_key:
        print("⚠️ 未检测到 API key。请先: cp .env.example .env，填入 DEEPSEEK_API_KEY")
        return

    llm = LLM(settings)
    n_ok = 0
    for p in PROMPTS:
        t0 = time.perf_counter()
        resp = llm.chat(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
            tools=TOOL_SCHEMA,
        )
        dt = (time.perf_counter() - t0) * 1000
        tc = resp.tool_calls[0] if resp.tool_calls else None
        ok = tc is not None and tc.name == "query_order" and bool(tc.arguments.get("order_id"))
        n_ok += int(ok)
        mark = "✓" if ok else "✗"
        calls = [(c.name, c.arguments) for c in resp.tool_calls]
        print(f"[{mark}] {dt:6.0f}ms  {p}")
        print(f"      tool_calls={calls}  content={resp.content[:80]!r}")
    total = len(PROMPTS)
    print(f"\n命中率: {n_ok}/{total} = {n_ok / total:.0%}")


if __name__ == "__main__":
    main()
