# customer-service

纯手写、**不依赖任何 Agent 框架**的 Multi-Agent 智能客服系统。

自己实现 ReAct / tool-calling 循环、工具注册、函数式 fan-out、多轮记忆——代码精良到能当博客逐篇读。

> 🚧 重写进行中（Phase 0 完成）。完整设计与文档见 `docs/`（随阶段补齐）。

## 快速开始

```bash
uv sync                          # 安装依赖
cp .env.example .env             # 填入 DEEPSEEK_API_KEY
uv run pytest                    # 跑测试（默认用 FakeLLM，不触网）
```
