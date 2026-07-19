# 设计决策

记录每个关键取舍的"为什么"，包括被**否决**的方案。这本身是博客素材。

## 1. sync + ThreadPoolExecutor，而非 async

客服场景的并行度 = agent 数（个位数），不是连接数（万级）。sync 栈可 `pdb` 单步、博客易读；async 的真实价值（流式输出 / 长连接复用 / 高并发 server）本项目都没有。并行用 `ThreadPoolExecutor` 几行解决。

> **否决**：asyncio。复杂度翻倍但本项目收益为零。未来要流式再加 `AsyncLLM` adapter。

## 2. 引入 pydantic（非纯 stdlib）

`@tool` 参数校验 + DeepSeek 返回 JSON 校验 + 运行配置，pydantic 一举三得。"一致性 > 零依赖"——既然 tool args 要 pydantic，配置也顺手用它。

## 3. `@tool` 反射生成 schema（带 override 兜底）

从 `typing` 注解 + docstring 的 `Args` 段反射生成 OpenAI function schema，是本项目最大卖点之一——几十行讲透 function calling。复杂类型（`Literal` / 嵌套模型）必要时加 `schema_override`，但 v1 先覆盖绝大多数简单工具。

## 4. 函数式 fan-out/fan-in，否决 actor 模型 ⭐

拓扑是**静态 DAG**（router → N agents → merge），不需要 actor 的动态订阅 / 邮箱 / 消息序列化 / 生命周期管理。`fanout()` + `merge()` 两个函数 ~50 行搞定；actor 要代码翻 3 倍。

> **否决**：actor 模型。把"我们考虑过但否决了、理由是什么"写成文档，比硬塞一个用不上的抽象层更有价值。

## 5. FakeLLM 队列式测试

测试要验证**真行为**（tool_calls 回灌循环、max_steps、重复检测），而不是把 LLM mock 掉只测壳。`FakeLLM` 接受预设 `ChatResponse` 队列，每次 `chat()` 弹一个；队列耗尽即抛错——这等价于一条隐式断言：实际调用次数 == 预期。

## 6. 手写 TF-IDF RAG，Retriever 协议留 embedding 扩展

FAQ 场景是少量（~10 条）封闭知识，手写 TF-IDF + 余弦（`core/retriever.py::TfidfRetriever`，~40 行）足够且零依赖——绝佳博客素材。

检索层抽成 `Retriever` 协议（`core/retriever.py`），默认 `TfidfRetriever`。想换语义检索时，实现一个 `EmbeddingRetriever` 即可（需另引 provider：DeepSeek 无 embedding 端点，可用 OpenAI embeddings / 阿里 BGE / 本地 sentence-transformers）。**默认不引入 embedding**——10 条知识用向量是杀鸡牛刀，且会破坏"纯 DeepSeek + 一键复现"定位。

## 7. pyproject.toml + uv，砍掉 requirements.txt

uv 是 2025-26 Python 圈事实标准。`pyproject.toml` 一个文件管依赖 / entry point / ruff / mypy。

## DeepSeek tool calling 防御强度

依据 [Phase 1.5 探针](../experiments/deepseek_tool_probe.md)：简单单工具场景 **7/7 = 100% 命中**，原生 `tools` 参数可用，不需要 prompt-based fallback 当主路径。但复杂 / 多步 / 多工具场景未测，仍保留：

- `max_steps` 硬上限（防 ReAct 死循环）
- tool args JSON 容错解析（`llm.py::_extract` 兜底非法 JSON）
- 连续重复调用检测（≥3 次相同调用判卡死）

## merge：单 agent 零 LLM / 多 agent 一次摘要（v2）

`merge` 区分两种情况：

- **单 agent 或无结果**：直接返回，**零 LLM 调用**——修正了老项目 `AggregatorAgent` 对单个结果也调一次 LLM 的负价值设计。
- **多 agent 且提供 ``llm``**：调一次 LLM 把多条回复整合成一条连贯、去重的最终回复。

这个分档修正了 v1 纯拼接时"多意图信息重叠"的问题（demo 第 3 问曾出现退换货政策被答两遍），同时把额外 LLM 成本**限制在"真有多 agent"的少数场景**——大多数单意图请求仍零额外调用。

## 诚实声明：不为 Multi-Agent 硬造并行

多数真实客服 query 是**单意图**，`fanout` 主要为"查订单 + 问政策"这类多意图边缘场景而设。我们**没有**为了"看起来更像 Multi-Agent"而强行把单意图拆成并行——那样不可信。
