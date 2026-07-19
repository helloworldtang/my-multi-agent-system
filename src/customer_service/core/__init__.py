"""core —— 纯手写的 Agent 核心抽象层。

本包是整个项目的"博客精华"：``llm`` / ``message`` / ``tools`` / ``agent`` / ``router``
/ ``fanout``。每个模块解决一个被 Agent 框架（LangChain 等）"藏起来"的底层问题，
并附设计说明（见各模块 docstring 与 ``docs/design-decisions.md``）。
"""
