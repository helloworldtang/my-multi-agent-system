"""FAQ Agent：回答常见咨询，用 search_faq 检索知识库。"""

from __future__ import annotations

from customer_service.core.agent import Agent, Confirmer
from customer_service.core.llm import LLM
from customer_service.core.tools import ToolRegistry
from customer_service.tools.faq_tools import search_faq

FAQ_SYSTEM_PROMPT = """你是电商 FAQ 客服助手，负责产品、配送、支付、会员等常见咨询。

工作要求：
- 先用 search_faq 检索知识库，再依据检索到的答案组织回复。
- 检索不到的问题，礼貌告知并建议联系人工客服。
- 回答简洁友好，不要编造政策。"""


def make_faq_agent(
    llm: LLM,
    *,
    max_steps: int = 5,
    confirmer: Confirmer | None = None,
) -> Agent:
    return Agent(
        name="FAQAgent",
        llm=llm,
        system_prompt=FAQ_SYSTEM_PROMPT,
        tools=ToolRegistry([search_faq]),
        max_steps=max_steps,
        confirmer=confirmer,
    )
