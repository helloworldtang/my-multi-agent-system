"""业务 Agent 测试（FakeLLM 编排 tool_call → final）。"""

from __future__ import annotations

from _fakes import FakeLLM, assistant_response
from customer_service.agents.faq import make_faq_agent
from customer_service.agents.order import make_order_agent
from customer_service.core.message import ToolCall


def _tc(name: str, args: dict[str, str], cid: str = "c1") -> ToolCall:
    return ToolCall(id=cid, name=name, arguments=args)


def test_order_agent_queries_then_answers() -> None:
    llm = FakeLLM(
        responses=[
            assistant_response("", tool_calls=[_tc("query_order", {"order_id": "ORD20240001"})]),
            assistant_response("您的订单已发货，顺丰快递 SF1234567890。"),
        ],
    )
    agent = make_order_agent(llm)
    r = agent.run("查一下 ORD20240001")
    assert r.stopped_reason == "final"
    assert "顺丰" in r.content or "已发货" in r.content
    assert r.tool_calls[0].name == "query_order"


def test_faq_agent_searches_then_answers() -> None:
    llm = FakeLLM(
        responses=[
            assistant_response("", tool_calls=[_tc("search_faq", {"query": "退货政策"})]),
            assistant_response("我们支持 7 天无理由退货。"),
        ],
    )
    agent = make_faq_agent(llm)
    r = agent.run("能退货吗")
    assert r.stopped_reason == "final"
    assert r.tool_calls[0].name == "search_faq"


def test_order_agent_mounts_expected_tools() -> None:
    agent = make_order_agent(FakeLLM(responses=[]))
    names = set(agent.tools.names()) if agent.tools else set()
    assert names == {"query_order", "cancel_order", "create_refund"}
