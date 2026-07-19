"""订单工具：查询与取消（数据迁移自老项目的硬编码 MOCK_ORDERS）。"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, cast

from customer_service.core.tools import tool


def _load_orders() -> dict[str, dict[str, Any]]:
    with resources.files("customer_service.data").joinpath("orders.json").open(
        "r", encoding="utf-8"
    ) as f:
        return cast(dict[str, dict[str, Any]], json.load(f))


@tool
def query_order(order_id: str) -> str:
    """查询订单的状态、商品、金额与物流信息。

    Args:
        order_id: 订单号，如 ORD20240001。
    """
    orders = _load_orders()
    o = orders.get(order_id)
    if o is None:
        return f"未找到订单 {order_id}，请确认订单号。"
    return (
        f"订单 {order_id}：商品「{o['product']}」，状态「{o['status']}」，"
        f"金额 {o['amount']} 元，物流「{o['logistics']}」。"
    )


@tool(requires_confirmation=True)
def cancel_order(order_id: str) -> str:
    """取消订单（仅"待发货"状态可取消；已发货需申请退货）。

    Args:
        order_id: 订单号。
    """
    orders = _load_orders()
    o = orders.get(order_id)
    if o is None:
        return f"未找到订单 {order_id}。"
    if o["status"] != "待发货":
        return f"订单 {order_id} 当前「{o['status']}」，已发货无法直接取消，请申请退货。"
    return f"订单 {order_id} 已取消。"
