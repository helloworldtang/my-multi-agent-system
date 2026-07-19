"""pytest fixtures。假对象定义在 ``tests/_fakes.py``（便于测试显式 import）。"""

from __future__ import annotations

import pytest

from _fakes import FakeLLM


@pytest.fixture
def fake_llm() -> FakeLLM:
    """空队列 FakeLLM；测试可自行 ``fake_llm.responses.append(...)`` 填充。"""
    return FakeLLM()
