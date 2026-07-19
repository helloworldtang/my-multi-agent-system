"""订单工具测试。"""

from __future__ import annotations

from customer_service.tools.order_tools import cancel_order, query_order


def test_query_order_found() -> None:
    r = query_order.execute({"order_id": "ORD20240001"})
    assert "iPhone 15" in r
    assert "已发货" in r


def test_query_order_not_found() -> None:
    r = query_order.execute({"order_id": "NOPE"})
    assert "未找到" in r


def test_cancel_order_shipped_blocked() -> None:
    r = cancel_order.execute({"order_id": "ORD20240001"})
    assert "无法直接取消" in r


def test_cancel_order_pending_ok() -> None:
    r = cancel_order.execute({"order_id": "ORD20240002"})
    assert "已取消" in r


def test_cancel_requires_confirmation_flag() -> None:
    assert cancel_order.requires_confirmation is True
    assert query_order.requires_confirmation is False
