"""llm.py —— LLM 客户端封装（sync）。

解决什么
--------
把 OpenAI SDK 的调用收敛到一个入口：带超时、指数退避重试、JSONL 可观测日志、
可通过环境变量切换 provider。上层只关心 messages / tools，不处理网络抖动与散落配置。

为什么是 sync 而非 async
------------------------
客服场景的并行粒度是 agent 数（个位数），不是连接数（万级）。sync 栈可 pdb 单步、
博客易读；并行交给 ``fanout.py`` 的 ThreadPoolExecutor。未来若要流式 / 高并发，再包一层
AsyncLLM adapter——v1 不写（YAGNI）。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from customer_service.core.message import ToolCall

logger = logging.getLogger("customer_service.llm")


class LLMError(RuntimeError):
    """LLM 调用层错误（如重试耗尽）。"""


class Settings(BaseSettings):
    """从环境变量 / ``.env`` 读取的运行配置。

    约定：API key 兼容 ``DEEPSEEK_API_KEY``（沿用老项目习惯）与 ``CS_API_KEY``；
    其余配置统一用 ``CS_`` 前缀（如 ``CS_MODEL``）。
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="CS_", extra="ignore")

    api_key: str = Field(
        default="",
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "CS_API_KEY"),
    )
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.3
    timeout: float = 10.0
    max_retries: int = 3
    max_steps: int = 5  # Agent ReAct 循环上限（Phase 3 用）


@dataclass
class ChatResponse:
    """LLM 一次调用的结构化结果。"""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    raw: Any = None


def _strip_surrogates(s: str) -> str:
    """把 lone surrogate（U+D800–U+DFFF）替换成 U+FFFD。

    解决什么
    --------
    OpenAI SDK 序列化 payload 时走 ``json.dumps(ensure_ascii=False).encode('utf-8')``，
    lone surrogate 无法编码，会抛 ``UnicodeEncodeError: surrogates not allowed`` 直接崩。
    这些码点通常来自 ``surrogateescape`` 解码的非 UTF-8 字节——粘贴的文本、读取的
    FAQ/订单数据、检索拼接的上下文都可能混入，且在终端里不可见。

    为什么用 U+FFFD 而非 ``encode('utf-8', 'replace')``
    --------------------------------------------------
    后者把 surrogate 换成 ASCII ``?``，与真正的问号混淆；U+FFFD 是 Unicode 标准替换符，
    语义清晰。fast-path 用一次 encode 探测，绝大多数干净字符串零额外开销。
    """
    try:
        s.encode("utf-8")
        return s
    except UnicodeEncodeError:
        return "".join("�" if 0xD800 <= ord(c) <= 0xDFFF else c for c in s)


def _sanitize(obj: Any) -> Any:
    """递归清洗 payload 中的所有 str：移除 lone surrogate。

    只穿透 str / Mapping / list 三类；int / bool / None 等原样返回。
    """
    if isinstance(obj, str):
        return _strip_surrogates(obj)
    if isinstance(obj, Mapping):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


class LLM:
    """带重试 / 超时 / 日志的 LLM 客户端。

    ``client`` 参数用于测试注入假 OpenAI client；生产留空，内部用真 SDK。
    """

    # 可重试的瞬时异常。设为类属性，测试可 monkeypatch 成自定义异常，避免构造真实 SDK 异常。
    _RETRYABLE: tuple[type[BaseException], ...] = (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
    )

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any = None,
        log_path: Path | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self._client = client or OpenAI(
            api_key=self.settings.api_key or "missing-api-key",
            base_url=self.settings.base_url,
            timeout=self.settings.timeout,
        )
        self._log_path = log_path
        self._sleep = time.sleep  # 测试可替换为 no-op，跳过真实退避

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {"model": self.settings.model, "messages": list(messages)}
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools
        if extra:
            payload.update(extra)
        # 移除 messages/tools/extra 里可能混入的 lone surrogate，否则 SDK 序列化时会崩
        payload = _sanitize(payload)

        retryable = self._RETRYABLE
        last_err: BaseException | None = None
        for attempt in range(self.settings.max_retries):
            t0 = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(**payload)
            except retryable as e:
                # NOTE: 循环内捕获是预期行为（重试语义），PERF203 规则未启用
                last_err = e
                logger.warning(
                    "LLM 调用失败(%s)，第 %d/%d 次重试",
                    type(e).__name__,
                    attempt + 1,
                    self.settings.max_retries,
                )
                self._sleep(self._backoff(attempt))
                continue
            latency_ms = (time.perf_counter() - t0) * 1000
            chat = self._extract(resp)
            self._log_call(latency_ms, payload, chat)
            return chat

        raise LLMError(f"LLM 调用重试 {self.settings.max_retries} 次仍失败") from last_err

    @staticmethod
    def _backoff(attempt: int) -> float:
        # 0.5s, 1s, 2s, ...（测试用 monkeypatch time.sleep 跳过真实退避）
        seconds: float = 0.5 * (2**attempt)
        return seconds

    def _extract(self, resp: Any) -> ChatResponse:
        msg = resp.choices[0].message
        tool_calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            args_raw = tc.function.arguments
            try:
                args = json.loads(args_raw) if args_raw else {}
            except (json.JSONDecodeError, TypeError):
                # NOTE: 模型偶发返回非法 JSON，这里不崩，保留原始串供上层兜底解析
                args = {"_raw_arguments": args_raw}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage: dict[str, int] = {}
        if getattr(resp, "usage", None):
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
        return ChatResponse(
            content=getattr(msg, "content", None) or "",
            tool_calls=tool_calls,
            usage=usage,
            model=getattr(resp, "model", self.settings.model),
            raw=resp,
        )

    def _log_call(self, latency_ms: float, payload: Mapping[str, Any], chat: ChatResponse) -> None:
        record = {
            "model": chat.model,
            "n_messages": len(payload.get("messages", [])),
            "had_tools": bool(payload.get("tools")),
            "prompt_tokens": chat.usage.get("prompt_tokens"),
            "completion_tokens": chat.usage.get("completion_tokens"),
            "tool_calls": [tc.name for tc in chat.tool_calls],
            "latency_ms": round(latency_ms, 1),
        }
        line = json.dumps(record, ensure_ascii=False)
        if self._log_path:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        else:
            logger.info(line)
