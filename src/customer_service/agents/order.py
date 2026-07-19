"""订单 Agent：处理订单查询、取消、退款。"""

from __future__ import annotations

from customer_service.core.agent import Agent, Confirmer
from customer_service.core.llm import LLM
from customer_service.core.tools import ToolRegistry
from customer_service.tools.order_tools import cancel_order, query_order
from customer_service.tools.refund_tools import create_refund

ORDER_SYSTEM_PROMPT = """你是电商订单客服助手，专业处理订单查询、取消与退款。

工作要求：
- 查询订单必须用 query_order 工具，绝不凭空编造订单信息。
- 取消订单用 cancel_order（仅在"待发货"可取消）。
- 退款用 create_refund。
- 把工具返回的关键信息组织成简洁、自然的中文回复给用户。"""


def make_order_agent(
    llm: LLM,
    *,
    max_steps: int = 5,
    confirmer: Confirmer | None = None,
) -> Agent:
    return Agent(
        name="OrderAgent",
        llm=llm,
        system_prompt=ORDER_SYSTEM_PROMPT,
        tools=ToolRegistry([query_order, cancel_order, create_refund]),
        max_steps=max_steps,
        confirmer=confirmer,
    )
