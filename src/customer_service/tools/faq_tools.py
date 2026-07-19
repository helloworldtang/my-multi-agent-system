"""FAQ 检索工具：通过 ``Retriever`` 协议检索知识库。

默认用零依赖的 ``TfidfRetriever``；换 ``EmbeddingRetriever`` 见 docs/design-decisions.md。
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, cast

from customer_service.core.retriever import FAQItem, Retriever, TfidfRetriever
from customer_service.core.tools import tool

_RETRIEVER: Retriever | None = None


def _load_faq() -> list[FAQItem]:
    with resources.files("customer_service.data").joinpath("faq.json").open(
        "r", encoding="utf-8"
    ) as f:
        raw = cast(list[dict[str, Any]], json.load(f))
    return [
        FAQItem(
            id=str(d.get("id", "")),
            question=str(d.get("question", "")),
            answer=str(d.get("answer", "")),
        )
        for d in raw
    ]


def _get_retriever() -> Retriever:
    """懒加载 retriever（可被测试替换为 EmbeddingRetriever 等）。"""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = TfidfRetriever(_load_faq())
    return _RETRIEVER


@tool
def search_faq(query: str, top_k: int = 3) -> str:
    """从 FAQ 知识库检索与问题最相关的若干条答案。

    Args:
        query: 用户的咨询问题。
        top_k: 返回的最多条数，默认 3。
    """
    items = _get_retriever().search(query, k=top_k)
    if not items:
        return "未在 FAQ 中找到相关问题，建议联系人工客服。"
    return "\n".join(f"Q：{i.question}\nA：{i.answer}" for i in items)
