"""cli.py 的纯函数测试：``make_cli_confirmer`` 的 y/N 逻辑。

REPL 交互循环（``main``）靠手动 / demo 验证，这里只单测确认回调本身——
同意 / 拒绝 / 安全默认（回车拒绝）/ 提示文案四类路径。
"""

from __future__ import annotations

from typing import Any

from customer_service.cli import make_cli_confirmer
from customer_service.tools.order_tools import cancel_order


class _FakeConsole:
    """最小 rich Console 替身：只实现 confirmer 用到的 print / input。"""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.printed: list[str] = []

    def print(self, *args: Any, **kwargs: Any) -> None:
        self.printed.append(str(args[0]) if args else "")

    def input(self, *args: Any, **kwargs: Any) -> str:
        return self.answer


def test_confirmer_allows_on_yes() -> None:
    confirm = make_cli_confirmer(_FakeConsole("y"))
    assert confirm(cancel_order, {"order_id": "ORD20240002"}) is True


def test_confirmer_allows_on_uppercase_y() -> None:
    confirm = make_cli_confirmer(_FakeConsole("Y"))
    assert confirm(cancel_order, {"order_id": "ORD20240002"}) is True


def test_confirmer_rejects_on_no() -> None:
    confirm = make_cli_confirmer(_FakeConsole("n"))
    assert confirm(cancel_order, {"order_id": "ORD20240002"}) is False


def test_confirmer_rejects_on_default_enter() -> None:
    # 安全默认：回车（空输入）→ 拒绝，避免误触取消订单
    confirm = make_cli_confirmer(_FakeConsole(""))
    assert confirm(cancel_order, {"order_id": "ORD20240002"}) is False


def test_confirmer_rejects_on_unrelated_input() -> None:
    # 任何非 y（含中文）→ 拒绝
    confirm = make_cli_confirmer(_FakeConsole("随便"))
    assert confirm(cancel_order, {"order_id": "ORD20240002"}) is False


def test_confirmer_prints_tool_name_and_args() -> None:
    console = _FakeConsole("n")
    make_cli_confirmer(console)(cancel_order, {"order_id": "ORD20240002"})
    joined = "\n".join(console.printed)
    assert "cancel_order" in joined  # 展示工具名
    assert "ORD20240002" in joined  # 展示参数，用户能看清要执行什么
