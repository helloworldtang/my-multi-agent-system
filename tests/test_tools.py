"""Tool / ToolRegistry 单元测试：反射 schema、pydantic 校验、注册执行。"""

from __future__ import annotations

import pytest

from customer_service.core.tools import ToolError, ToolRegistry, tool


@tool
def add(a: int, b: int) -> int:
    """两数相加。

    Args:
        a: 第一个数。
        b: 第二个数。
    """
    return a + b


@tool(name="greet_user", requires_confirmation=True)
def greet(name: str, *, loud: bool = False) -> str:
    """打招呼。

    Args:
        name: 名字。
        loud: 是否大写。
    """
    return f"HELLO {name}" if loud else f"hello {name}"


def test_schema_has_name_and_description() -> None:
    schema = add.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "add"
    assert "两数相加" in schema["function"]["description"]


def test_schema_parameters_from_annotations() -> None:
    params = add.to_openai_schema()["function"]["parameters"]
    props = params["properties"]
    assert set(props) == {"a", "b"}
    assert props["a"]["type"] == "integer"
    assert set(params["required"]) == {"a", "b"}


def test_schema_optional_param_not_required() -> None:
    params = greet.to_openai_schema()["function"]["parameters"]
    assert params["properties"]["loud"]["type"] == "boolean"
    assert "loud" not in params.get("required", [])
    assert "name" in params["required"]


def test_schema_param_description_from_docstring() -> None:
    props = add.to_openai_schema()["function"]["parameters"]["properties"]
    assert "第一个数" in props["a"]["description"]


def test_execute_validates_and_calls() -> None:
    assert add.execute({"a": 1, "b": 2}) == "3"


def test_execute_stringifies_non_str_result() -> None:
    # add 返回 int，execute 统一成 str 供 LLM 消费
    assert isinstance(add.execute({"a": 1, "b": 2}), str)


def test_execute_rejects_bad_args() -> None:
    with pytest.raises(ToolError):
        add.execute({"a": "x", "b": 2})


def test_requires_confirmation_metadata() -> None:
    assert add.requires_confirmation is False
    assert greet.requires_confirmation is True


def test_custom_name() -> None:
    assert greet.name == "greet_user"


def test_registry_register_and_execute() -> None:
    reg = ToolRegistry([add])
    assert reg.names() == ["add"]
    assert reg.execute("add", {"a": 2, "b": 3}) == "5"


def test_registry_to_schemas() -> None:
    reg = ToolRegistry([add, greet])
    names = [s["function"]["name"] for s in reg.to_openai_schemas()]
    assert names == ["add", "greet_user"]


def test_registry_duplicate_raises() -> None:
    with pytest.raises(ToolError):
        ToolRegistry([add, add])


def test_registry_unknown_tool_raises() -> None:
    reg = ToolRegistry([add])
    with pytest.raises(ToolError):
        reg.execute("nope", {})


def test_registry_get_returns_tool() -> None:
    reg = ToolRegistry([add])
    assert reg.get("add") is add


def test_registry_len() -> None:
    assert len(ToolRegistry([add, greet])) == 2
