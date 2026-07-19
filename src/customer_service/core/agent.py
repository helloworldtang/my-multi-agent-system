"""agent.py —— 单 Agent 的 ReAct tool-calling 循环。

解决什么
--------
把"LLM 决定调工具 → 执行工具 → 把结果喂回 LLM → 直到给出最终答案"这个循环手写出来。
这是 ReAct 的本质，也是 Agent 框架（LangChain 等）的核心黑盒——我们把它摊开写清楚。

防御设计（依据 Phase 1.5 探针结论）
------------------------------------
DeepSeek tool calling 在简单场景稳定（100%），但复杂 / 多步场景仍需兜底：
- ``max_steps`` 硬上限，防死循环；
- 连续重复调用检测（>=3 次相同调用视为卡死）；
- 工具执行错误不崩，回灌给 LLM 让它换法子；
- ``requires_confirmation`` 的写操作经 ``confirmer`` 回调确认。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from customer_service.core.llm import LLM
from customer_service.core.message import Conversation, Message, ToolCall
from customer_service.core.tools import Tool, ToolError, ToolRegistry

# 写操作确认回调：(tool, args) -> 是否允许执行
Confirmer = Callable[[Tool, dict[str, Any]], bool]


@dataclass
class AgentResult:
    """一次 Agent 运行的结果与轨迹。"""

    content: str
    steps: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    stopped_reason: str = "final"  # final / max_steps / loop


class Agent:
    """单 Agent：一个 system_prompt + 一组工具 + ReAct 循环。"""

    def __init__(
        self,
        name: str,
        llm: LLM,
        system_prompt: str,
        tools: ToolRegistry | None = None,
        *,
        max_steps: int = 5,
        confirmer: Confirmer | None = None,
    ) -> None:
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_steps = max_steps
        self.confirmer = confirmer

    def run(self, user_message: str, *, history: list[Message] | None = None) -> AgentResult:
        conv = Conversation(system_prompt=self.system_prompt)
        if history:
            conv.messages.extend(history)
        conv.user(user_message)

        tools_schema = self.tools.to_openai_schemas() if self.tools else None
        made_calls: list[ToolCall] = []
        last_call_key: str | None = None
        repeat_count = 0

        for step in range(1, self.max_steps + 1):
            resp = self.llm.chat(conv.to_dicts(), tools=tools_schema)

            if not resp.tool_calls:
                conv.assistant(resp.content)
                return AgentResult(
                    content=resp.content, steps=step, tool_calls=made_calls, stopped_reason="final"
                )

            # 模型要调工具：记录 assistant 的 tool_calls，再逐个执行并把结果作为 tool 消息回灌
            conv.assistant(resp.content, tool_calls=resp.tool_calls)
            for tc in resp.tool_calls:
                made_calls.append(tc)
                key = f"{tc.name}:{_args_key(tc.arguments)}"
                repeat_count = repeat_count + 1 if key == last_call_key else 1
                last_call_key = key
                if repeat_count >= 3:
                    return AgentResult(
                        content="（检测到重复调用，已停止以防死循环）",
                        steps=step,
                        tool_calls=made_calls,
                        stopped_reason="loop",
                    )
                conv.tool(self._execute_tool(tc), tool_call_id=tc.id, name=tc.name)

        return AgentResult(
            content="（已达到最大步数，未能给出最终答案）",
            steps=self.max_steps,
            tool_calls=made_calls,
            stopped_reason="max_steps",
        )

    def _execute_tool(self, tc: ToolCall) -> str:
        if self.tools is None:
            return "错误：当前 Agent 未挂载工具"
        try:
            tool = self.tools.get(tc.name)
        except ToolError as e:
            return f"错误：{e}"
        if (
            tool.requires_confirmation
            and self.confirmer is not None
            and not self.confirmer(tool, tc.arguments)
        ):
            return "用户拒绝了该操作。"
        try:
            return tool.execute(tc.arguments)
        except ToolError as e:
            return f"工具执行失败：{e}"


def _args_key(arguments: dict[str, Any]) -> str:
    return repr(sorted(arguments.items()))
