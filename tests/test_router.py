"""意图识别（router）测试。"""

from __future__ import annotations

from _fakes import FakeLLM, assistant_response
from customer_service.core.router import classify


def test_parses_multiple_intents_sorted() -> None:
    llm = FakeLLM(
        responses=[
            assistant_response(
                '{"intents": [{"name": "ORDER", "score": 0.6}, {"name": "FAQ", "score": 0.9}]}',
            ),
        ],
    )
    cls = classify(llm, "查订单顺便问退换货")
    assert [n for n, _ in cls.intents] == ["FAQ", "ORDER"]  # 按 score 降序


def test_filters_below_threshold() -> None:
    llm = FakeLLM(responses=[assistant_response('{"intents": [{"name": "FAQ", "score": 0.2}]}')])
    cls = classify(llm, "x", min_score=0.5)
    assert cls.intents == []


def test_handles_bad_json() -> None:
    llm = FakeLLM(responses=[assistant_response("not json")])
    assert classify(llm, "x").intents == []


def test_handles_empty_intents() -> None:
    llm = FakeLLM(responses=[assistant_response('{"intents": []}')])
    assert classify(llm, "x").intents == []


def test_drops_unknown_intent_name() -> None:
    llm = FakeLLM(responses=[assistant_response('{"intents": [{"name": "WEATHER", "score": 0.9}]}')])
    assert classify(llm, "x").intents == []
