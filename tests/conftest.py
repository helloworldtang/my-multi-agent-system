"""测试 fixtures 与假对象。

Phase 0 仅提供 ``FakeLLM`` 雏形。Phase 1 定义真正的 LLM 接口后，``FakeLLM``
会实现该接口，使测试在不触网、不消耗 API 额度的情况下验证 Agent 行为。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class FakeLLM:
    """假 LLM：按调用顺序弹出预设响应（队列式 mock）。

    队列耗尽后再调用会抛错——这等价于一条隐式断言：实际调用次数 == 预期。
    Phase 1 起将实现与 ``customer_service.core.llm.LLM`` 相同的接口。
    """

    responses: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def chat(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        if not self.responses:
            msg = "FakeLLM 响应队列已空但仍被调用——检查测试的预期调用次数"
            raise RuntimeError(msg)
        return self.responses.pop(0)


@pytest.fixture
def fake_llm() -> FakeLLM:
    """空队列 FakeLLM；测试可自行 ``.responses.extend([...])`` 填充。"""
    return FakeLLM()
