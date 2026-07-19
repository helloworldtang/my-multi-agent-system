"""FAQ 检索工具：手写 TF-IDF + 余弦相似度，零依赖。

为什么自己写而非用向量库
------------------------
博客级作品追求"一键复现 + 零额外服务"。手写 TF-IDF（几十行）足以演示 RAG 的本质
（检索 + 拼上下文）；embedding 方案作为未来 ``Retriever`` 扩展点，见
``docs/design-decisions.md``。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from importlib import resources
from typing import Any, cast

from customer_service.core.tools import tool

_FAQ_CACHE: list[dict[str, Any]] | None = None


def _load_faq() -> list[dict[str, Any]]:
    global _FAQ_CACHE
    if _FAQ_CACHE is None:
        with resources.files("customer_service.data").joinpath("faq.json").open(
            "r", encoding="utf-8"
        ) as f:
            _FAQ_CACHE = cast(list[dict[str, Any]], json.load(f))
    return _FAQ_CACHE


def _tokenize(text: str) -> list[str]:
    """中英文混合分词：英文/数字按词，中文按单字。"""
    return re.findall(r"[a-z0-9]+|[一-龥]", text.lower())


def _term_freq(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {t: n / total for t, n in counts.items()}


def _build_index() -> tuple[list[dict[str, Any]], list[dict[str, float]], dict[str, float]]:
    docs = _load_faq()
    tokenized = [_tokenize(str(d.get("question", ""))) for d in docs]
    n_docs = len(docs)
    df: dict[str, int] = {}
    for toks in tokenized:
        for term in set(toks):
            df[term] = df.get(term, 0) + 1
    idf = {t: math.log((n_docs + 1) / (n + 1)) + 1.0 for t, n in df.items()}
    vecs = [
        {t: tf_v * idf.get(t, 0.0) for t, tf_v in _term_freq(toks).items()}
        for toks in tokenized
    ]
    return docs, vecs, idf


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(v * b.get(t, 0.0) for t, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (na * nb)


@tool
def search_faq(query: str, top_k: int = 3) -> str:
    """从 FAQ 知识库检索与问题最相关的若干条答案。

    Args:
        query: 用户的咨询问题。
        top_k: 返回的最多条数，默认 3。
    """
    docs, vecs, idf = _build_index()
    q_vec = {t: v * idf.get(t, 0.0) for t, v in _term_freq(_tokenize(query)).items()}
    scored = sorted(
        ((_cosine(q_vec, v), docs[i]) for i, v in enumerate(vecs)),
        key=lambda x: x[0],
        reverse=True,
    )
    top = [d for sim, d in scored[:top_k] if sim > 0]
    if not top:
        return "未在 FAQ 中找到相关问题，建议联系人工客服。"
    return "\n".join(f"Q：{d['question']}\nA：{d['answer']}" for d in top)
