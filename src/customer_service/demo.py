"""demo.py —— 预设对话脚本（README 截图素材）。

跑法：``uv run python -m customer_service.demo``
依次演示 FAQ / ORDER / 多意图 / 投诉 四类典型对话。
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from customer_service.core.llm import LLM, Settings
from customer_service.system import MultiAgentSystem

DEMO_QUESTIONS = [
    "你们支持7天无理由退货吗？",  # FAQ
    "帮我查一下订单 ORD20240001 的物流",  # ORDER
    "查一下 ORD20240002 还能取消吗，顺便问下退换货政策",  # 多意图 ORDER + FAQ
    "我买的东西质量太差了，要求退款",  # COMPLAINT
]


def main() -> None:
    console = Console()
    settings = Settings()
    if not settings.api_key:
        console.print("[red]未检测到 DEEPSEEK_API_KEY，请先配置（见 .env.example）[/red]")
        return

    system = MultiAgentSystem(LLM(settings))
    console.print(Panel("Multi-Agent 客服 Demo", title="demo"))
    for question in DEMO_QUESTIONS:
        console.print(f"\n[bold green]用户>[/bold green] {question}")
        reply = system.chat(question)
        console.print(Panel(reply, title="客服"))


if __name__ == "__main__":
    main()
