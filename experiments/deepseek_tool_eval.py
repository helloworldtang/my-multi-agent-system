#!/usr/bin/env python
"""DeepSeek tool-calling 完整 eval（Phase 8）。

横向对比 3 种"让 LLM 调工具"的方式：
- ``tool_calling``：原生 ``tools`` 参数
- ``json_mode``：``response_format=json_object``，LLM 输出 ``{"tool", "args"}``
- ``prompt_fallback``：纯 prompt + 正则解析

每个 prompt 有明确期望（订单类→query_order / FAQ类→search_faq / 闲聊类→不调工具），
据此统计各模式成功率。结果用于 ``docs/deepseek-tool-calling-eval.md``。

跑法：``uv run python experiments/deepseek_tool_eval.py``（需 DEEPSEEK_API_KEY）
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from customer_service.core.llm import LLM, Settings

# 三种模式共用的工具清单
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "查询订单的状态与物流。需要订单号。",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_faq",
            "description": "从 FAQ 知识库检索答案。需要用户问题。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

SYS_TOOL_CALLING = "你是客服。根据用户问题决定是否调用工具，无需调用时直接回答。"

SYS_JSON_MODE = (
    "你是客服。可选工具：query_order(order_id) / search_faq(query)。\n"
    "如果要调工具，只输出 JSON：{\"tool\": \"query_order\", \"args\": {\"order_id\": \"...\"}}；\n"
    '不需要工具时输出 {"tool": null}。'
)

SYS_PROMPT = (
    "你是客服。可选工具：query_order / search_faq。\n"
    "如果要调工具，第一行输出 'TOOL: 工具名'，第二行输出 'ARGS: {json}'；\n"
    "不需要工具时只输出 'NONE'。"
)


@dataclass
class Case:
    text: str
    expect: str | None  # 期望工具名；None 表示不该调工具
    category: str


PROMPTS: list[Case] = [
    # 订单类 → query_order
    Case("查一下订单 ORD20240001", "query_order", "order"),
    Case("订单 ORD20240002 发货了吗", "query_order", "order"),
    Case("ORD20240003 到哪了", "query_order", "order"),
    Case("帮我看看 ORD20240001 的物流", "query_order", "order"),
    Case("查个单子 ORD20240002", "query_order", "order"),
    Case("ORD20240003 现在什么状态", "query_order", "order"),
    Case("订单 ORD20240001 物流信息", "query_order", "order"),
    Case("ORD20240002 还没到吗", "query_order", "order"),
    Case("帮我查 ORD20240003", "query_order", "order"),
    Case("订单号 ORD20240001 查一下", "query_order", "order"),
    Case("我的单子 ORD20240002 怎么样了", "query_order", "order"),
    Case("ORD20240001 配送到哪了", "query_order", "order"),
    Case("查 ORD20240003 的发货情况", "query_order", "order"),
    Case("ORD20240001 运单号多少", "query_order", "order"),
    Case("订单 ORD20240002 状态查询", "query_order", "order"),
    # FAQ 类 → search_faq
    Case("支持7天无理由退货吗", "search_faq", "faq"),
    Case("配送多久到", "search_faq", "faq"),
    Case("支持微信支付吗", "search_faq", "faq"),
    Case("运费多少", "search_faq", "faq"),
    Case("有会员折扣吗", "search_faq", "faq"),
    Case("支持花呗分期吗", "search_faq", "faq"),
    Case("你们卖什么商品", "search_faq", "faq"),
    Case("能货到付款吗", "search_faq", "faq"),
    Case("快速配送次日达吗", "search_faq", "faq"),
    Case("正品保障吗", "search_faq", "faq"),
    Case("退换货政策是什么", "search_faq", "faq"),
    Case("满多少免运费", "search_faq", "faq"),
    Case("怎么付款", "search_faq", "faq"),
    Case("金卡会员多少折扣", "search_faq", "faq"),
    Case("发顺丰吗", "search_faq", "faq"),
    # 闲聊类 → 不该调工具
    Case("今天天气真好", None, "chitchat"),
    Case("你叫什么名字", None, "chitchat"),
    Case("谢谢你", None, "chitchat"),
    Case("讲个笑话", None, "chitchat"),
    Case("你是机器人吗", None, "chitchat"),
    Case("现在几点了", None, "chitchat"),
    Case("帮我写首诗", None, "chitchat"),
    Case("今天星期几", None, "chitchat"),
    Case("你能做什么", None, "chitchat"),
    Case("再见", None, "chitchat"),
]


def _msgs(system: str, case: Case) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": case.text},
    ]


def mode_tool_calling(llm: LLM, case: Case) -> str | None:
    resp = llm.chat(_msgs(SYS_TOOL_CALLING, case), tools=TOOL_SCHEMAS)
    return resp.tool_calls[0].name if resp.tool_calls else None


def mode_json_mode(llm: LLM, case: Case) -> str | None:
    resp = llm.chat(_msgs(SYS_JSON_MODE, case), extra={"response_format": {"type": "json_object"}})
    try:
        data = json.loads(resp.content)
    except json.JSONDecodeError:
        return "__PARSE_ERROR__"
    tool = data.get("tool")
    return tool if isinstance(tool, str) else None


def mode_prompt_fallback(llm: LLM, case: Case) -> str | None:
    resp = llm.chat(_msgs(SYS_PROMPT, case))
    m = re.search(r"TOOL:\s*([A-Za-z_]+)", resp.content)
    return m.group(1) if m else None


MODES = {
    "tool_calling": mode_tool_calling,
    "json_mode": mode_json_mode,
    "prompt_fallback": mode_prompt_fallback,
}


def main() -> None:
    settings = Settings()
    if not settings.api_key:
        print("⚠️ 未检测到 DEEPSEEK_API_KEY")
        return
    llm = LLM(settings)

    results: dict[str, dict[str, int]] = {m: {"correct": 0, "total": 0} for m in MODES}
    by_cat: dict[str, dict[str, int]] = {m: {} for m in MODES}

    for i, case in enumerate(PROMPTS, 1):
        for mode, fn in MODES.items():
            try:
                got = fn(llm, case)
            except Exception as e:  # noqa: BLE001
                got = f"__ERR__{type(e).__name__}"
            ok = got == case.expect
            results[mode]["total"] += 1
            if ok:
                results[mode]["correct"] += 1
            cat = by_cat[mode].setdefault(case.category, {"correct": 0, "total": 0})
            cat["total"] += 1
            if ok:
                cat["correct"] += 1
            mark = "✓" if ok else "✗"
            print(
                f"[{i:2d}/{len(PROMPTS)}] {mode:15s} {mark} "
                f"expect={case.expect} got={got} | {case.text}"
            )
        time.sleep(0.1)  # 轻微限速，避免触发 rate limit

    print("\n===== 总成功率 =====")
    for mode in MODES:
        r = results[mode]
        print(f"{mode:15s} {r['correct']}/{r['total']} = {r['correct']/r['total']:.0%}")

    print("\n===== 按类别 =====")
    for mode in MODES:
        print(f"\n[{mode}]")
        for cat, r in by_cat[mode].items():
            print(f"  {cat:10s} {r['correct']}/{r['total']} = {r['correct']/r['total']:.0%}")


if __name__ == "__main__":
    main()
