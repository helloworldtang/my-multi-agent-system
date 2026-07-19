# 快速开始

## 前置

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)（安装：`curl -LsSf https://astral.sh/uv/install.sh | sh`）
- 一个 DeepSeek API Key（[申请](https://platform.deepseek.com/)）

## 1. 安装依赖

```bash
git clone <repo-url>
cd my-multi-agent-system
uv sync
```

`uv sync` 会自动按 `.python-version`（3.12）准备解释器并装好依赖。

## 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入：
# DEEPSEEK_API_KEY=sk-...
```

其它配置（`CS_MODEL` / `CS_MAX_STEPS` / `CS_TIMEOUT` 等）都有默认值，按需覆盖。

## 3. 运行

```bash
# 交互式 REPL（/quit 退出，/reset 清空历史）
uv run customer-service

# 或跑预设 demo（FAQ / 订单 / 多意图 / 投诉 四类）
uv run python -m customer_service.demo
```

## 4. 跑测试

```bash
uv run pytest            # 全部测试（用 FakeLLM，不触网、不耗额度）
uv run ruff check .      # lint
uv run mypy              # 类型检查（strict）
```

## 常见问题

- **报错"未检测到 DEEPSEEK_API_KEY"**：确认 `.env` 已填，或在 shell 里 `export DEEPSEEK_API_KEY=...`。
- **想换 provider（OpenAI 兼容）**：改 `.env` 的 `CS_BASE_URL` 与 `CS_MODEL`，并配 `DEEPSEEK_API_KEY`（或 `CS_API_KEY`）为对应 key。
