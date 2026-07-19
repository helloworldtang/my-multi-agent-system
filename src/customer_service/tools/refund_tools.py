"""退款与升级工具：发起退款（写操作，需确认）、升级人工。"""

from __future__ import annotations

from customer_service.core.tools import tool


@tool(requires_confirmation=True)
def create_refund(order_id: str, reason: str) -> str:
    """发起退款申请。

    Args:
        order_id: 订单号。
        reason: 退款原因。
    """
    return f"已为订单 {order_id} 创建退款申请（原因：{reason}），专员将在 24 小时内处理。"


@tool
def escalate_to_human(order_id: str, issue: str) -> str:
    """升级给人工客服专员（24 小时内回复）。

    Args:
        order_id: 订单号。
        issue: 问题描述。
    """
    return f"已升级人工专员（订单 {order_id}，问题：{issue}），24 小时内回复。"
