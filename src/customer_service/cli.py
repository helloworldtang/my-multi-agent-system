"""cli.py —— 交互式 REPL 入口。

启动后进入多轮对话，支持命令：
- ``/quit`` / ``/exit``：退出
- ``/reset``：清空对话历史

用 rich 做高亮输出。安装后可直接 ``customer-service`` 启动。
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel

from customer_service.core.llm import LLM, Settings
from customer_service.system import MultiAgentSystem


def main() -> None:
    console = Console()
    settings = Settings()
    if not settings.api_key:
        console.print("[red]未检测到 DEEPSEEK_API_KEY，请先配置（见 .env.example）[/red]")
        sys.exit(1)

    system = MultiAgentSystem(LLM(settings))
    console.print(
        Panel(
            "Multi-Agent 智能客服（/quit 退出 · /reset 清空历史）",
            title="客服系统",
        )
    )
    while True:
        try:
            user = console.input("[bold green]你>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n再见！")
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            console.print("再见！")
            break
        if user == "/reset":
            system.reset()
            console.print("[yellow]已清空对话历史[/yellow]")
            continue
        reply = system.chat(user)
        console.print(Panel(reply, title="客服"))


if __name__ == "__main__":
    main()
