"""投诉 Agent：安抚情绪、收集信息、必要时升级人工。"""

from __future__ import annotations

from customer_service.core.agent import Agent, Confirmer
from customer_service.core.llm import LLM
from customer_service.core.tools import ToolRegistry
from customer_service.tools.refund_tools import escalate_to_human

COMPLAINT_SYSTEM_PROMPT = """你是投诉处理客服。

工作要求：
- 先共情、道歉，安抚用户情绪，不与用户争论。
- 需要订单号与问题描述，必要时用 escalate_to_human 升级给人工专员。
- 态度始终友好，回复简洁。"""


def make_complaint_agent(
    llm: LLM,
    *,
    max_steps: int = 5,
    confirmer: Confirmer | None = None,
) -> Agent:
    return Agent(
        name="ComplaintAgent",
        llm=llm,
        system_prompt=COMPLAINT_SYSTEM_PROMPT,
        tools=ToolRegistry([escalate_to_human]),
        max_steps=max_steps,
        confirmer=confirmer,
    )
