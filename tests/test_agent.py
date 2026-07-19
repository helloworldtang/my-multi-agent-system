"""Agent ReAct loop 测试：用 FakeLLM 编排预设响应，验证 tool 执行/回灌/防御。"""

from __future__ import annotations

from _fakes import FakeLLM, assistant_response
from customer_service.core.agent import Agent
from customer_service.core.message import ToolCall
from customer_service.core.tools import ToolRegistry, tool


@tool
def echo(text: str) -> str:
    """原样回声。

    Args:
        text: 要回声的文本。
    """
    return f"echo:{text}"


@tool(requires_confirmation=True)
def dangerous(action: str) -> str:
    """危险操作。

    Args:
        action: 操作名。
    """
    return f"done:{action}"


def _tc(name: str, args: dict[str, str], cid: str = "c1") -> ToolCall:
    return ToolCall(id=cid, name=name, arguments=args)


def test_no_tools_returns_final_content() -> None:
    llm = FakeLLM(responses=[assistant_response("你好")])
    agent = Agent("A", llm, "sys")
    r = agent.run("hi")
    assert r.content == "你好"
    assert r.stopped_reason == "final"
    assert r.steps == 1


def test_executes_tool_then_final() -> None:
    llm = FakeLLM(
        responses=[
            assistant_response("", tool_calls=[_tc("echo", {"text": "hello"})]),
            assistant_response("最终：echo:hello"),
        ],
    )
    agent = Agent("A", llm, "sys", tools=ToolRegistry([echo]))
    r = agent.run("say hello")
    assert r.content == "最终：echo:hello"
    assert r.stopped_reason == "final"
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "echo"
    assert len(llm.calls) == 2  # 两次 LLM 调用


def test_max_steps_stops() -> None:
    # 一直返回 tool_call，永不给 final
    responses = [assistant_response("", tool_calls=[_tc("echo", {"text": "x"}, f"c{i}")]) for i in range(10)]
    llm = FakeLLM(responses=responses)
    agent = Agent("A", llm, "sys", tools=ToolRegistry([echo]), max_steps=2)
    r = agent.run("loop")
    assert r.stopped_reason == "max_steps"
    assert r.steps == 2


def test_loop_detection_on_repeated_call() -> None:
    # 连续 3 次完全相同的 tool_call
    same = assistant_response("", tool_calls=[_tc("echo", {"text": "x"}, "c1")])
    llm = FakeLLM(responses=[same, same, same, same])
    agent = Agent("A", llm, "sys", tools=ToolRegistry([echo]), max_steps=10)
    r = agent.run("repeat")
    assert r.stopped_reason == "loop"


def test_confirmation_rejected_recovers() -> None:
    llm = FakeLLM(
        responses=[
            assistant_response("", tool_calls=[_tc("dangerous", {"action": "delete"})]),
            assistant_response("好的，未执行"),
        ],
    )
    agent = Agent(
        "A",
        llm,
        "sys",
        tools=ToolRegistry([dangerous]),
        confirmer=lambda _t, _a: False,
    )
    r = agent.run("do it")
    assert r.stopped_reason == "final"
    assert r.content == "好的，未执行"


def test_unknown_tool_recovered() -> None:
    llm = FakeLLM(
        responses=[
            assistant_response("", tool_calls=[_tc("nonexistent", {})]),
            assistant_response("该工具不存在，换种说法"),
        ],
    )
    agent = Agent("A", llm, "sys", tools=ToolRegistry([echo]))
    r = agent.run("x")
    assert r.stopped_reason == "final"
    assert r.content == "该工具不存在，换种说法"


def test_history_is_carried() -> None:
    # 带 history 时，前文进入对话，模型据此作答
    llm = FakeLLM(responses=[assistant_response("明白了")])
    agent = Agent("A", llm, "sys")
    from customer_service.core.message import Message, Role

    history = [Message(Role.USER, "前情"), Message(Role.ASSISTANT, "好的")]
    agent.run("继续", history=history)
    # 第一次（也是唯一一次）LLM 调用应含历史 3 条（system 不计）+ 当前 user
    assert llm.calls[0]["n_messages"] == 4
