"""fanout.py —— 并行调用多个 Agent + 合并结果（替代 aggregator）。

解决什么
--------
router 识别出多意图时，把请求并行分发给对应 Agent，再合并回复。这是真正的
Multi-Agent 协作点。

为什么用函数式 fan-out/fan-in 而非 actor 模型
----------------------------------------------
拓扑是静态 DAG（router → N agents → merge），不需要 actor 的动态订阅/邮箱/生命周期。
ThreadPoolExecutor 几行搞定；actor 要代码翻 3 倍（见 docs/design-decisions.md）。

merge 的分档（v2）
------------------
- 单 agent 或无结果：直接返回，零 LLM。
- 多 agent 且提供 ``llm``：调一次 LLM 把多条回复整合成连贯、去重的最终回复。
  修正了 v1 纯拼接在多意图时的信息重叠（demo 第 3 问曾出现退换货政策被答两遍），
  同时把额外 LLM 成本限制在"真有多 agent"的少数场景。
- 多 agent 但无 ``llm``：退化为拼接（兼容）。
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from customer_service.core.agent import Agent, AgentResult
from customer_service.core.llm import LLM
from customer_service.core.message import Message

MERGE_SYSTEM_PROMPT = (
    "你是客服回复整合专家。下面是多个专员客服对同一用户问题的各自回复。\n"
    "请整合成一条连贯、自然、去重的最终回复。\n"
    "要求：去掉重复信息；不要引入新信息或承诺；保留所有关键事实；用流畅中文。"
)


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


def merge(results: list[AgentResult], llm: LLM | None = None) -> str:
    """合并多个 Agent 的回复（分档规则见模块 docstring）。"""
    if not results:
        return "抱歉，我暂时无法理解您的问题，能具体描述一下吗？"
    if len(results) == 1:
        return results[0].content
    parts = [r.content for r in results if r.content]
    if not parts:
        return "（多个 Agent 均未给出有效回复）"
    if llm is None:
        return "\n\n".join(parts)
    resp = llm.chat(
        [
            {"role": "system", "content": MERGE_SYSTEM_PROMPT},
            {"role": "user", "content": "\n---\n".join(parts)},
        ],
    )
    return resp.content
