"""fanout.py —— 并行调用多个 Agent + 合并结果（替代 aggregator）。

解决什么
--------
router 识别出多意图时，把请求并行分发给对应 Agent，再合并回复。这是真正的
Multi-Agent 协作点。

为什么用函数式 fan-out/fan-in 而非 actor 模型
----------------------------------------------
拓扑是静态 DAG（router → N agents → merge），不需要 actor 的动态订阅/邮箱/生命周期。
ThreadPoolExecutor 几行搞定；actor 要代码翻 3 倍（见 docs/design-decisions.md）。

为什么 merge 不再调 LLM
-----------------------
老项目的 AggregatorAgent 对单个结果又调一次 LLM，纯负价值。这里用简单拼接合并，
既快又省钱。
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from customer_service.core.agent import Agent, AgentResult
from customer_service.core.message import Message


def fanout(
    agents: Iterable[Agent],
    user_message: str,
    *,
    history: list[Message] | None = None,
    max_workers: int | None = None,
) -> list[AgentResult]:
    """并行调用多个 Agent，各自带上可选的跨轮历史。"""
    agents_list = list(agents)
    if not agents_list:
        return []
    workers = max_workers or min(len(agents_list), 4)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(a.run, user_message, history=history) for a in agents_list]
        return [f.result() for f in futures]


def merge(results: list[AgentResult]) -> str:
    """合并多个 Agent 的回复（不再调 LLM）。"""
    if not results:
        return "抱歉，我暂时无法理解您的问题，能具体描述一下吗？"
    if len(results) == 1:
        return results[0].content
    parts = [r.content for r in results if r.content]
    if not parts:
        return "（多个 Agent 均未给出有效回复）"
    return "\n\n".join(parts)
