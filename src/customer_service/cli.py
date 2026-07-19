"""cli.py —— 交互式 REPL 入口。

启动后进入多轮对话，支持命令：
- ``/quit`` / ``/exit``：退出
- ``/reset``：清空对话历史

用 rich 做高亮输出。安装后可直接 ``customer-service`` 启动。
"""

from __future__ import annotations

import sys
import threading
from typing import Any

from rich.console import Console
from rich.panel import Panel

from customer_service.core.agent import Confirmer
from customer_service.core.llm import LLM, Settings
from customer_service.core.tools import Tool
from customer_service.system import MultiAgentSystem


def make_cli_confirmer(console: Console) -> Confirmer:
    """构造写操作确认回调（``cancel_order`` / ``create_refund`` 等执行前问 y/N）。

    解决什么
    --------
    不传 confirmer 时，``Agent`` 对写操作直接放行（见 ``agent.py`` 的
    ``requires_confirmation and confirmer is not None`` 判定）——用户一句"取消订单"
    就真取消了，没有二次确认。CLI 是给人用的，写操作该问一句。

    为什么内部加锁
    --------------
    ``fanout`` 用 ``ThreadPoolExecutor`` 并行跑多个 agent，ORDER 想取消、COMPLAINT
    想退款可能同时发生。两个线程并发调 ``console.input`` 会抢 stdin、提示交错。
    一把锁把确认步骤串行化——LLM 调用仍并行，只有 y/N 提示排队，性能无损。

    为什么默认拒绝（回车 / 非 y → False）
    -------------------------------------
    写操作误触代价高（真取消订单 / 真发起退款），安全默认；要执行必须显式输 y。
    """

    lock = threading.Lock()

    def confirm(tool: Tool, args: dict[str, Any]) -> bool:
        with lock:
            console.print(f"[yellow]⚠ 即将执行写操作：{tool.name}[/yellow]")
            console.print(f"  说明：{tool.description}")
            console.print(f"  参数：{args}")
            answer = console.input("[bold]是否允许？[y/N][/bold] ").strip().lower()
            return answer == "y"

    return confirm


def main() -> None:
    console = Console()
    settings = Settings()
    if not settings.api_key:
        console.print("[red]未检测到 DEEPSEEK_API_KEY，请先配置（见 .env.example）[/red]")
        sys.exit(1)

    system = MultiAgentSystem(LLM(settings), confirmer=make_cli_confirmer(console))
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
