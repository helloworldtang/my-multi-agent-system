"""测试 fixtures 与假对象。

提供两类替身：

- ``FakeLLM``：实现 ``customer_service.core.llm.LLM.chat`` 同款接口（入 messages/tools，
  出 ``ChatResponse``），队列式返回。用于 Agent / System 层测试，完全不触网。
- ``fake_openai_client`` / ``_resp``：模拟 OpenAI SDK 响应结构，用于 LLM 自身的
  重试 / 解析 / 日志测试（见 ``test_llm.py``）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from customer_service.core.llm import ChatResponse
from customer_service.core.message import ToolCall


@dataclass
class FakeLLM:
    """实现 LLM 接口的假 LLM（队列式 mock）。

    构造时传入按调用顺序排列的 ``ChatResponse``，每次 ``chat()`` 弹出一个。
    队列耗尽后再调用会抛错——这等价于一条隐式断言：实际调用次数 == 预期。
    """

    responses: list[ChatResponse] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ChatResponse:
        self.calls.append({"n_messages": len(list(messages)), "tools": tools, "temperature": temperature})
        if not self.responses:
            msg = "FakeLLM 响应队列已空但仍被调用——检查测试的预期调用次数"
            raise RuntimeError(msg)
        return self.responses.pop(0)


def assistant_response(
    content: str = "",
    *,
    tool_calls: list[ToolCall] | None = None,
) -> ChatResponse:
    """快速构造一个 assistant ``ChatResponse``。"""
    return ChatResponse(content=content, tool_calls=list(tool_calls or []))


@pytest.fixture
def fake_llm() -> FakeLLM:
    """空队列 FakeLLM；测试可自行 ``fake_llm.responses.append(...)`` 填充。"""
    return FakeLLM()
