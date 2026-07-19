"""冒烟测试：验证工具链就位（包可导入、版本号存在）。"""

from __future__ import annotations

from customer_service import __version__


def test_package_importable() -> None:
    assert isinstance(__version__, str) and __version__
