"""FAQ TF-IDF 检索测试。"""

from __future__ import annotations

from customer_service.tools.faq_tools import search_faq


def test_search_refund_policy() -> None:
    r = search_faq.execute({"query": "7天能退货吗"})
    assert "7 天无理由" in r


def test_search_payment() -> None:
    r = search_faq.execute({"query": "怎么付款 支持微信吗"})
    assert "微信" in r


def test_search_top_k_limits_results() -> None:
    r = search_faq.execute({"query": "配送多久", "top_k": 1})
    assert r.count("Q：") == 1


def test_search_no_match_suggests_human() -> None:
    r = search_faq.execute({"query": "xyzqwerty unseen"})
    assert "未在 FAQ" in r or "人工客服" in r
