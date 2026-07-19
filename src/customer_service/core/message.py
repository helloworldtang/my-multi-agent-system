"""message.py —— 消息类型与对话记忆。

解决什么
--------
把 OpenAI 风格的多角色消息 + 会话历史，抽象成类型安全、可截断的结构。上层 Agent
只管 ``append`` / ``to_dicts``，不必手拼消息列表，也不必担心上下文窗口溢出。

为什么自己定义而不用 SDK 的类型
--------------------------------
我们想让 provider 可切换（DeepSeek / OpenAI / 未来其它），所以消息层用自己的
``Message`` / ``Conversation``，只在送入 LLM 的前一刻才 ``to_dict()`` 成 OpenAI 格式。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """对话角色。继承 str 让 ``Role.USER == "user"`` 直接成立，方便序列化。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """模型发起的一次工具调用（参数已从 JSON 解析为 dict）。"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """一条对话消息。

    - assistant 消息可带 ``tool_calls``（模型决定调用工具时）；
    - tool 消息用 ``tool_call_id`` 与 ``name`` 回填工具执行结果。
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]
        if self.role is Role.TOOL:
            if self.tool_call_id is not None:
                d["tool_call_id"] = self.tool_call_id
            if self.name is not None:
                d["name"] = self.name
        return d


@dataclass
class Conversation:
    """一次会话：一个 system_prompt + 若干轮消息历史。

    truncate 采用滑动窗口（保留最近 N 条）。更精细的摘要式压缩是未来扩展点，
    见 ``docs/design-decisions.md``。
    """

    system_prompt: str
    messages: list[Message] = field(default_factory=list)

    def append(self, message: Message) -> Message:
        self.messages.append(message)
        return message

    def user(self, content: str) -> Message:
        return self.append(Message(Role.USER, content))

    def assistant(self, content: str, *, tool_calls: list[ToolCall] | None = None) -> Message:
        return self.append(Message(Role.ASSISTANT, content, tool_calls=tool_calls))

    def tool(self, content: str, *, tool_call_id: str, name: str) -> Message:
        return self.append(Message(Role.TOOL, content, tool_call_id=tool_call_id, name=name))

    def truncate(self, max_messages: int) -> None:
        """保留最近 ``max_messages`` 条（system_prompt 单独存放，不计入）。"""
        if max_messages < 0:
            msg = "max_messages must be non-negative"
            raise ValueError(msg)
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:] if max_messages else []

    def to_dicts(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = [{"role": Role.SYSTEM.value, "content": self.system_prompt}]
        result.extend(m.to_dict() for m in self.messages)
        return result

    def reset(self) -> None:
        self.messages.clear()

    def __iter__(self) -> Iterator[Message]:
        return iter(self.messages)

    def __len__(self) -> int:
        return len(self.messages)
