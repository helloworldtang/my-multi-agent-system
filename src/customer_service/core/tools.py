"""tools.py —— 工具注册与执行（手写 function calling）。

解决什么
--------
让业务函数用 ``@tool`` 一行声明成"可被 LLM 调用的工具"：自动从类型注解 + docstring
生成 OpenAI function schema，执行时用 pydantic 校验参数。function calling 的本质就是
"JSON schema + 派发"，这里没有任何黑盒。

为什么用反射生成 schema
-----------------------
这是本项目最大的卖点之一——几十行就把"工具注册"讲透了。复杂类型（Literal / 嵌套模型）
必要时可加 ``schema_override`` 兜底，但 v1 先覆盖绝大多数简单工具。
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from docstring_parser import parse as parse_docstring
from pydantic import BaseModel, Field, ValidationError, create_model


class ToolError(RuntimeError):
    """工具查找 / 参数校验 / 执行错误。"""


@dataclass
class Tool:
    """一个可被 LLM 调用的工具。"""

    name: str
    description: str
    func: Callable[..., Any]
    model: type[BaseModel]  # 由函数签名生成的参数校验模型
    requires_confirmation: bool = False  # 写操作（取消订单/退款）需上层先确认

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _clean_schema(self.model.model_json_schema()),
            },
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        try:
            validated = self.model(**arguments)
        except ValidationError as e:
            msg = f"工具 {self.name} 参数校验失败：{e}"
            raise ToolError(msg) from e
        result = self.func(**validated.model_dump())
        return result if isinstance(result, str) else str(result)


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """去掉 pydantic 生成的冗余 title，让 schema 更贴近 OpenAI 惯例。"""
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    requires_confirmation: bool = False,
) -> Any:
    """装饰器：把函数声明为 Tool。

    参数类型与必填从类型注解推断；参数描述从 Google-style docstring 的 ``Args`` 段解析；
    工具描述取 docstring 首段。
    """

    def make(f: Callable[..., Any]) -> Tool:
        sig = inspect.signature(f)
        hints = _safe_hints(f)
        doc = parse_docstring(inspect.getdoc(f) or "")
        param_docs = {p.arg_name: p.description or "" for p in doc.params}

        fields: dict[str, Any] = {}
        for pname, param in sig.parameters.items():
            annotation = hints.get(pname, str)
            description = param_docs.get(pname, "")
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[pname] = (annotation, Field(default=default, description=description))

        model = create_model(name or f.__name__, **fields)
        return Tool(
            name=name or f.__name__,
            description=(doc.short_description or f.__name__).strip(),
            func=f,
            model=model,
            requires_confirmation=requires_confirmation,
        )

    if func is None:
        return make
    return make(func)


def _safe_hints(f: Callable[..., Any]) -> dict[str, Any]:
    try:
        return get_type_hints(f)
    except Exception:  # noqa: BLE001
        # 反射失败时退化为无注解，不阻塞注册
        return {}


class ToolRegistry:
    """工具集合：注册、导出 schema、按名执行。"""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: Tool) -> None:
        if t.name in self._tools:
            msg = f"工具 {t.name} 已注册"
            raise ToolError(msg)
        self._tools[t.name] = t

    def get(self, name: str) -> Tool:
        t = self._tools.get(name)
        if t is None:
            msg = f"未知工具：{name}"
            raise ToolError(msg)
        return t

    def to_openai_schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        return self.get(name).execute(arguments)

    def names(self) -> list[str]:
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)
