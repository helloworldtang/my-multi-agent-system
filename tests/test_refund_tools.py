"""退款工具测试。"""

from __future__ import annotations

from customer_service.tools.refund_tools import create_refund, escalate_to_human


def test_create_refund_requires_confirmation() -> None:
    assert create_refund.requires_confirmation is True


def test_create_refund_message() -> None:
    r = create_refund.execute({"order_id": "ORD20240001", "reason": "质量问题"})
    assert "ORD20240001" in r
    assert "质量问题" in r


def test_escalate_does_not_require_confirmation() -> None:
    assert escalate_to_human.requires_confirmation is False


def test_escalate_message() -> None:
    r = escalate_to_human.execute({"order_id": "ORD20240001", "issue": "商品损坏"})
    assert "ORD20240001" in r
    assert "专员" in r
