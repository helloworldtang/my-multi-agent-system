# DeepSeek tool-calling 探针结果（Phase 1.5）

> 脚本：`experiments/deepseek_tool_probe.py` · 模型 `deepseek-chat` · 2026-07

## 场景
单工具 `query_order`（参数 `order_id: str`），7 条自然语言 prompt（订单号 ORD20240001/2/3，措辞各异：查物流 / 发货了吗 / 到哪了 / 什么状态 / 查个单子 / 发没发 / 在哪儿）。

## 结果

| 指标 | 值 |
|------|-----|
| 命中率（返回 `tool_calls` 且参数合法） | **7/7 = 100%** |
| 平均延迟 | ~1.0s（713ms–1569ms） |
| 参数 JSON 解析失败 | 0 |
| 陷入循环 / 重复调用 | 0 |
| 未调 tool 直接编造答案 | 0 |

每条都正确：返回 `tool_calls=[('query_order', {'order_id': 'ORD...'})]`，订单号提取准确，content 表态"我来查"而非编造物流。

## 结论与对 Phase 3 的影响

- **简单单工具场景，DeepSeek function calling 非常稳定**——原生 `tools` 参数可用，无需 prompt-based fallback 作主路径。
- 但本次只覆盖**单工具 + 清晰实体**。复杂场景（多工具选择、模糊实体、多轮指代）未测，仍需防御：
  - `max_steps` 硬上限（防 ReAct 死循环）
  - tool args 的 JSON 容错解析（已在 `llm.py::_extract` 兜底非法 JSON）
  - 多工具时的 `tool_choice` 策略留待 Phase 3 / Phase 8 实测
- **防御强度结论：适度，不过度。** Phase 3 保留 `max_steps` + JSON 容错；暂不实现 prompt-based JSON fallback（除非后续实测发现需要）。

## 后续
Phase 8 将扩展为 100×prompt × 3 模式（`tool_calling` / `response_format=json_object` / prompt-fallback）的完整 eval，覆盖多工具与模糊场景。
