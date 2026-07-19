"""retriever.py —— RAG 检索抽象（可替换入口）。

默认实现 ``TfidfRetriever`` 零依赖；``EmbeddingRetriever`` 作为扩展点（需另引 provider，
见 docs/design-decisions.md）。

为什么默认 TF-IDF
-----------------
FAQ 场景是少量（~10 条）封闭知识，TF-IDF 足够且零依赖。embedding 的语义优势要在大
规模、开放知识库才显著——10 条用 embedding 是杀鸡牛刀；且 DeepSeek 无 embedding 端点，
引入它会破坏"纯 DeepSeek + 一键复现"的定位。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol


@dataclass
class FAQItem:
    id: str
    question: str
    answer: str


class Retriever(Protocol):
    """检索协议：把 query 变成 top-k 相关 ``FAQItem``。"""

    def search(self, query: str, k: int = 3) -> list[FAQItem]:
        """返回与 query 最相关的至多 k 条 FAQItem（按相关度降序）。"""


def _tokenize(text: str) -> list[str]:
    """中英文混合分词：英文/数字按词，中文按单字。"""
    return re.findall(r"[a-z0-9]+|[一-龥]", text.lower())


def _term_freq(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {t: n / total for t, n in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(v * b.get(t, 0.0) for t, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (na * nb)


class TfidfRetriever:
    """手写 TF-IDF + 余弦相似度检索。零依赖。"""

    def __init__(self, items: list[FAQItem]) -> None:
        self._items = items
        tokenized = [_tokenize(i.question) for i in items]
        n_docs = len(items)
        df: dict[str, int] = {}
        for toks in tokenized:
            for term in set(toks):
                df[term] = df.get(term, 0) + 1
        self._idf = {t: math.log((n_docs + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self._vecs = [self._vec(_term_freq(toks)) for toks in tokenized]

    def _vec(self, tf: dict[str, float]) -> dict[str, float]:
        return {t: v * self._idf.get(t, 0.0) for t, v in tf.items()}

    def search(self, query: str, k: int = 3) -> list[FAQItem]:
        q_vec = self._vec(_term_freq(_tokenize(query)))
        scored = sorted(
            ((_cosine(q_vec, v), self._items[i]) for i, v in enumerate(self._vecs)),
            key=lambda x: x[0],
            reverse=True,
        )
        return [item for sim, item in scored[:k] if sim > 0]
