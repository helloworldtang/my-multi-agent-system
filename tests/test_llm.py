"""LLM 客户端测试：消息拼装、响应解析、重试、JSONL 日志。

全部用假 OpenAI client（SimpleNamespace 拼出最小结构），不触网、不依赖 API key。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from customer_service.core.llm import LLM, LLMError, Settings


def _resp(
    content: str = "hi",
    *,
    tool_calls: list[Any] | None = None,
    model: str = "deepseek-chat",
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _client(create_fn: Callable[..., Any]) -> SimpleNamespace:
    """构造最小假 OpenAI client：只有 ``chat.completions.create``。"""
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_fn)))


def test_settings_defaults() -> None:
    s = Settings(api_key="k")
    assert s.model == "deepseek-chat"
    assert s.base_url == "https://api.deepseek.com"
    assert s.max_retries == 3


def test_chat_parses_content_and_usage() -> None:
    seen: list[dict[str, Any]] = []

    def create(**kw: Any) -> SimpleNamespace:
        seen.append(kw)
        return _resp("hello")

    llm = LLM(Settings(api_key="k"), client=_client(create))
    r = llm.chat([{"role": "user", "content": "hi"}])
    assert r.content == "hello"
    assert r.usage == {"prompt_tokens": 10, "completion_tokens": 5}
    assert seen[0]["model"] == "deepseek-chat"
    assert seen[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_chat_parses_tool_calls() -> None:
    tc = SimpleNamespace(
        id="c1",
        function=SimpleNamespace(name="query_order", arguments='{"order_id": "X1"}'),
    )

    def create(**kw: Any) -> SimpleNamespace:
        return _resp("", tool_calls=[tc])

    llm = LLM(Settings(api_key="k"), client=_client(create))
    r = llm.chat(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f"}}],
    )
    assert r.tool_calls[0].name == "query_order"
    assert r.tool_calls[0].arguments == {"order_id": "X1"}


def test_chat_tolerates_bad_tool_args_json() -> None:
    tc = SimpleNamespace(
        id="c1",
        function=SimpleNamespace(name="f", arguments="not-json"),
    )

    def create(**kw: Any) -> SimpleNamespace:
        return _resp("", tool_calls=[tc])

    llm = LLM(Settings(api_key="k"), client=_client(create))
    r = llm.chat([{"role": "user", "content": "hi"}])
    assert r.tool_calls[0].arguments == {"_raw_arguments": "not-json"}


def test_chat_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LLM, "_RETRYABLE", (RuntimeError,))
    monkeypatch.setattr("customer_service.core.llm.time.sleep", lambda _s: None)
    seq: list[Any] = [RuntimeError("boom"), _resp("ok")]

    def create(**kw: Any) -> SimpleNamespace:
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    llm = LLM(Settings(api_key="k", max_retries=3), client=_client(create))
    r = llm.chat([{"role": "user", "content": "hi"}])
    assert r.content == "ok"


def test_chat_raises_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LLM, "_RETRYABLE", (RuntimeError,))
    monkeypatch.setattr("customer_service.core.llm.time.sleep", lambda _s: None)

    def create(**kw: Any) -> Any:
        raise RuntimeError("always")

    llm = LLM(Settings(api_key="k", max_retries=2), client=_client(create))
    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "hi"}])


def test_chat_writes_jsonl_log(tmp_path: pytest.TempPathHint) -> None:
    log = tmp_path / "llm.jsonl"

    def create(**kw: Any) -> SimpleNamespace:
        return _resp("hi")

    llm = LLM(Settings(api_key="k"), client=_client(create), log_path=log)
    llm.chat([{"role": "user", "content": "hi"}])
    record = json.loads(log.read_text(encoding="utf-8").strip())
    assert record["model"] == "deepseek-chat"
    assert record["prompt_tokens"] == 10
    assert record["n_messages"] == 1
