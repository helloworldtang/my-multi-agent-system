"""Conversation / Message 单元测试。"""

from __future__ import annotations

import pytest

from customer_service.core.message import Conversation, Message, Role, ToolCall


def test_message_system_to_dict() -> None:
    m = Message(Role.SYSTEM, "you are helpful")
    assert m.to_dict() == {"role": "system", "content": "you are helpful"}


def test_message_assistant_with_tool_calls() -> None:
    tc = ToolCall(id="call_1", name="query_order", arguments={"order_id": "X1"})
    m = Message(Role.ASSISTANT, "", tool_calls=[tc])
    d = m.to_dict()
    assert d["role"] == "assistant"
    assert d["tool_calls"][0]["type"] == "function"
    assert d["tool_calls"][0]["function"]["name"] == "query_order"
    assert '"order_id": "X1"' in d["tool_calls"][0]["function"]["arguments"]


def test_message_tool_role_round_trip() -> None:
    m = Message(Role.TOOL, "result", tool_call_id="call_1", name="query_order")
    assert m.to_dict() == {
        "role": "tool",
        "content": "result",
        "tool_call_id": "call_1",
        "name": "query_order",
    }


def test_conversation_to_dicts_includes_system_first() -> None:
    conv = Conversation(system_prompt="sys")
    conv.user("hi")
    dicts = conv.to_dicts()
    assert dicts[0] == {"role": "system", "content": "sys"}
    assert dicts[1] == {"role": "user", "content": "hi"}


def test_conversation_truncate_keeps_recent() -> None:
    conv = Conversation(system_prompt="sys")
    for i in range(5):
        conv.user(f"m{i}")
    conv.truncate(2)
    assert len(conv) == 2
    assert [m.content for m in conv] == ["m3", "m4"]


def test_conversation_truncate_zero_clears() -> None:
    conv = Conversation(system_prompt="sys")
    conv.user("x")
    conv.truncate(0)
    assert len(conv) == 0


def test_conversation_truncate_negative_raises() -> None:
    conv = Conversation(system_prompt="sys")
    with pytest.raises(ValueError):
        conv.truncate(-1)


def test_conversation_truncate_noop_when_under_limit() -> None:
    conv = Conversation(system_prompt="sys")
    conv.user("only")
    conv.truncate(10)
    assert len(conv) == 1


def test_conversation_reset() -> None:
    conv = Conversation(system_prompt="sys")
    conv.user("hi")
    conv.reset()
    assert len(conv) == 0
