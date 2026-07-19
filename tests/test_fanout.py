"""fan-out / merge 测试。"""

from __future__ import annotations

from _fakes import FakeLLM, assistant_response
from customer_service.core.agent import Agent, AgentResult
from customer_service.core.fanout import fanout, merge


def test_fanout_collects_all() -> None:
    a1 = Agent("A1", FakeLLM(responses=[assistant_response("A1")]), "sys")
    a2 = Agent("A2", FakeLLM(responses=[assistant_response("A2")]), "sys")
    results = fanout([a1, a2], "hi")
    assert {r.content for r in results} == {"A1", "A2"}


def test_fanout_empty_agents() -> None:
    assert fanout([], "hi") == []


def test_merge_single_returns_content() -> None:
    assert merge([AgentResult(content="only")]) == "only"


def test_merge_multiple_without_llm_joins() -> None:
    r = merge([AgentResult(content="A"), AgentResult(content="B")])
    assert "A" in r and "B" in r


def test_merge_empty_clarifies() -> None:
    assert "无法理解" in merge([]) or "描述" in merge([])


def test_smart_merge_single_skips_llm() -> None:
    # 单 agent：不应消费 LLM 队列
    llm = FakeLLM(responses=[assistant_response("不应被调用")])
    assert merge([AgentResult(content="only")], llm=llm) == "only"
    assert len(llm.calls) == 0


def test_smart_merge_multiple_calls_llm_once() -> None:
    llm = FakeLLM(responses=[assistant_response("整合后的回复")])
    r = merge([AgentResult(content="A"), AgentResult(content="B")], llm=llm)
    assert r == "整合后的回复"
    assert len(llm.calls) == 1


def test_smart_merge_passes_all_parts_to_llm() -> None:
    llm = FakeLLM(responses=[assistant_response("ok")])
    merge([AgentResult(content="甲"), AgentResult(content="乙")], llm=llm)
    user_msg = llm.calls[0]["n_messages"]  # type: ignore[index]
    assert user_msg == 2  # system + 用户（拼接的两段）
