"""MultiAgentSystem 编排测试。"""

from __future__ import annotations

from _fakes import FakeLLM, assistant_response
from customer_service.core.message import ToolCall
from customer_service.system import MultiAgentSystem

_EMPTY = '{"intents": []}'


def _tc(name: str, args: dict[str, str], cid: str = "c1") -> ToolCall:
    return ToolCall(id=cid, name=name, arguments=args)


def test_no_intent_clarifies() -> None:
    system = MultiAgentSystem(FakeLLM(responses=[assistant_response(_EMPTY)]))
    reply = system.chat("今天天气怎么样")
    assert "无法理解" in reply or "描述" in reply


def test_single_order_intent() -> None:
    llm = FakeLLM(
        responses=[
            assistant_response('{"intents": [{"name": "ORDER", "score": 0.9}]}'),
            assistant_response("", tool_calls=[_tc("query_order", {"order_id": "ORD20240001"})]),
            assistant_response("您的订单 ORD20240001 已发货。"),
        ],
    )
    system = MultiAgentSystem(llm)
    assert "已发货" in system.chat("查 ORD20240001")


def test_below_threshold_treated_as_unknown() -> None:
    llm = FakeLLM(responses=[assistant_response('{"intents": [{"name": "FAQ", "score": 0.3}]}')])
    system = MultiAgentSystem(llm)
    reply = system.chat("嗯")
    assert "无法理解" in reply or "描述" in reply


def test_history_recorded_across_turns() -> None:
    llm = FakeLLM(responses=[assistant_response(_EMPTY), assistant_response(_EMPTY)])
    system = MultiAgentSystem(llm)
    system.chat("第一句")
    system.chat("第二句")
    assert len(system.history) == 4  # 两轮各 user + assistant


def test_reset_clears_history() -> None:
    system = MultiAgentSystem(FakeLLM(responses=[assistant_response(_EMPTY)]))
    system.chat("hi")
    system.reset()
    assert system.history == []
