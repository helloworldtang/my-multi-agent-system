"""入口模块可导入性测试（cli/demo 的交互逻辑靠手动/真实跑验证）。"""

from __future__ import annotations


def test_cli_main_callable() -> None:
    from customer_service import cli

    assert callable(cli.main)


def test_demo_questions_cover_scenarios() -> None:
    from customer_service import demo

    assert len(demo.DEMO_QUESTIONS) >= 3
