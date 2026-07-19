"""router.py —— 意图识别（结构化分类，非 ReAct agent）。

解决什么
--------
判断用户问题该交给哪些 Agent。与 Phase 3 的 ReAct agent 不同，意图识别是"结构化
分类任务"——不需要工具循环，只需让 LLM 输出带置信度的 JSON。用 response_format
强制 JSON + pydantic 校验，避免老项目"关键词包含匹配"的脆弱兜底。

为什么支持多意图
----------------
真实客服里"我的订单 ORD20240001 还没到，我要退款"既是 ORDER 又是 COMPLAINT。
单意图 elif 会漏处理；多意图 + fanout 才对。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from customer_service.core.llm import LLM

INTENTS: tuple[str, ...] = ("FAQ", "ORDER", "COMPLAINT")

ROUTER_SYSTEM_PROMPT = (
    "你是客服意图识别专家。判断用户问题属于以下哪些意图（可多选）：\n"
    "- FAQ：常见咨询（产品/配送/支付/会员）\n"
    "- ORDER：订单相关（查询/取消/物流/退款）\n"
    "- COMPLAINT：投诉/情绪/不满\n\n"
    '只返回 JSON，格式：{"intents": [{"name": "FAQ", "score": 0.9}]}\n'
    "score 为 0-1 的置信度。没有匹配意图时返回空列表。"
    f" 可选意图名：{', '.join(INTENTS)}。"
)


class IntentHit(BaseModel):
    name: str
    score: float


class RouterResult(BaseModel):
    intents: list[IntentHit] = Field(default_factory=list)


@dataclass
class Classification:
    """分类结果：已过阈值、按 score 降序的 (意图, 置信度) 列表。"""

    intents: list[tuple[str, float]]
    raw: dict[str, Any] | None = None


def classify(llm: LLM, text: str, *, min_score: float = 0.5) -> Classification:
    """识别意图，返回过阈值的多意图（按置信度降序）。"""
    resp = llm.chat(
        [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        extra={"response_format": {"type": "json_object"}},
    )
    try:
        data = json.loads(resp.content)
    except json.JSONDecodeError:
        return Classification(intents=[])
    try:
        result = RouterResult.model_validate(data)
    except ValidationError:
        return Classification(intents=[])
    picked = [
        (h.name, h.score)
        for h in result.intents
        if h.name in INTENTS and h.score >= min_score
    ]
    picked.sort(key=lambda x: x[1], reverse=True)
    return Classification(intents=picked, raw=data)
