"""system.py —— Multi-Agent 编排入口。

工作流
------
用户消息 → router 识别意图(可多选) → fanout 并行分发给对应 Agent → merge 合并回复。
跨轮对话历史由 system 维护并传给 Agent，让"它"等指代可解。

设计取舍
--------
- 不再有老项目的 AggregatorAgent：merge 用拼接，不再多调一次 LLM。
- 路由 / 分发 / 合并都是显式的纯函数与薄类，没有隐藏状态机。
"""

from __future__ import annotations

from customer_service.agents import make_complaint_agent, make_faq_agent, make_order_agent
from customer_service.core.agent import Agent, Confirmer
from customer_service.core.fanout import fanout, merge
from customer_service.core.llm import LLM
from customer_service.core.message import Message, Role
from customer_service.core.router import classify

_UNKNOWN_REPLY = (
    "抱歉，我暂时无法理解您的问题。能具体描述一下是关于"
    "产品咨询、订单查询，还是投诉建议吗？"
)


class MultiAgentSystem:
    """Multi-Agent 客服系统：router → fanout → merge，带跨轮记忆。"""

    def __init__(
        self,
        llm: LLM,
        *,
        max_steps: int = 5,
        confirmer: Confirmer | None = None,
        min_score: float = 0.5,
    ) -> None:
        self.llm = llm
        self.max_steps = max_steps
        self.confirmer = confirmer
        self.min_score = min_score
        self._agents: dict[str, Agent] = {
            "FAQ": make_faq_agent(llm, max_steps=max_steps, confirmer=confirmer),
            "ORDER": make_order_agent(llm, max_steps=max_steps, confirmer=confirmer),
            "COMPLAINT": make_complaint_agent(llm, max_steps=max_steps, confirmer=confirmer),
        }
        self.history: list[Message] = []

    def chat(self, user_message: str) -> str:
        cls = classify(self.llm, user_message, min_score=self.min_score)
        if not cls.intents:
            reply = _UNKNOWN_REPLY
        else:
            agents = [self._agents[name] for name, _ in cls.intents if name in self._agents]
            results = fanout(agents, user_message, history=self.history)
            reply = merge(results)
        self.history.append(Message(Role.USER, user_message))
        self.history.append(Message(Role.ASSISTANT, reply))
        return reply

    def reset(self) -> None:
        self.history.clear()
