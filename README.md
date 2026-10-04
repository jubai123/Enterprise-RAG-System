# Enterprise RAG

Enterprise RAG 是一个面向企业文档的检索增强生成系统。项目使用混合检索、Cross Encoder
重排序和受约束的答案生成流程，在生成答案前尽可能找到与问题直接相关的文档证据。

系统基于 LangChain 组织模型与检索组件，使用 LangGraph 编排工作流，并使用 Qdrant 保存
文档向量。Dense 检索负责语义匹配，BM25 负责精确术语和技术标识符匹配，Cross Encoder
（默认走硅基流动在线 API，本地仅作后备）对候选 chunk / 父块打分。

## 核心能力

- 将结构化和非结构化企业文档转换为统一的 LangChain `Document`；
- 为文档生成稳定的文档标识和 chunk 标识，并按 (dsid, section) 预建受控长度的父块；
- 结合 Dense、BM25 与 RRF 进行多通道检索；
- 根据问题需求生成检索计划（含意图路由与每意图检索预算）；
- 使用 Cross Encoder 对候选打分并做确定性选择；
- 扩展相关父文档 / 父块上下文，控制喂给生成器的上下文宽度；
- 记录检索、模型调用和选择过程；
- 生成可复现的运行清单和诊断报告。

## 工作流程

两个图骨架共用同一条前缀：检索规划 → 混合检索，之后的取证据方式不同。

```text
问题
  -> 检索规划（plan_question）
  -> 混合检索（retrieve_queries）
  -> ── minimal：RRF doc 池 → CE 打分 → top-k doc → 父文档展开
     └─ chunkce：叶级 CE → 逐子查询取 top-K 叶升父块 → 父块粒度去重合并
  -> 答案生成（generate_answer）
```

**当前默认是 `minimal`**。`chunkce` 在 rag100_dev 100 题上测得更高（71.0 vs 66.0 raw），
但优势尚未通过显著性检验（McNemar p≈0.30），因此未设为默认——需要时用
`--config configs/rag100_dev_parentce_perquery.yaml` 显式选择。

Cross Encoder 在线模式下走硅基流动 `/rerank`（复用 `SILICONFLOW_API_KEY`），无需本地 GPU。
模型缺失、加载失败、推理异常或超时会直接终止运行，不会静默切换设备或降级。

工作流通过以下接口构建：

```python
from src.config import load_config
from src.graphs.dependencies import RagDependencies
from src.graphs.minimal import build_minimal_graph

config = load_config()
graph = build_minimal_graph(config, dependencies)
```

`RagDependencies` 统一封装 LLM、主检索器、父文档、父块索引和 Cross Encoder。

### 图变体

`graph.mode` 选择工作流骨架，两者共用同一份摄取、索引与评测入口：

| mode                | 构建器                  | 阶段                                                                  |
| ------------------- | ----------------------- | --------------------------------------------------------------------- |
| `minimal`（默认） | `build_minimal_graph` | 混合检索 → RRF doc 池 → CE top-k → 父文档展开 → 生成（3 阶段、宽上下文） |
| `chunkce`         | `build_chunkce_graph` | 混合检索 → **叶级** CE → 逐子查询取 top-K 叶升父块合并 → 生成      |

`chunkce` 与 `minimal` 的关键差别：CE 直接对检索命中的原始 chunk 打分（不经 RRF 折叠、
词法重排与窗口拼贴），父块展开发生在 CE **之后**，且取块预算是**每子查询**一份而非全局
一份，上下文宽度随子查询数增长。它需要索引层预建的父块 artifact：

```powershell
python -m scripts.build_parent_blocks --config configs/rag100_dev_parentce_perquery.yaml
```

产物路径由 `data.parent_blocks_file` 指定；文件缺失时退化为 identity 展开（无兄弟上下文）
并打 warning。

> 2026-09-21 起仓库只保留这两条线。此前用于承载 selection / judge / repair / 补充检索回环 /
> 实体链接扩展的 `full` 完整管线已整体移除——它在 rag100_dev 上测得 62~63%，低于这两条线。

## 项目结构

```text
configs/
  main.yaml                 完整运行配置（graph.mode 默认 minimal）
  rag100_dev_*.yaml         rag100 评测快照（含复现锁）
  rag100_frozen_*.yaml      45 题冻结集评测快照
scripts/
  ingest.py                 文档摄取
  build_index.py            构建向量索引
  build_parent_blocks.py    预建父块索引（chunkce 路径）
  preflight.py              运行前检查
  run_benchmark.py          启动工作流
  evaluate.py               评估与诊断
src/
  config/                   类型化配置加载与校验
  chains/                   LangChain 模型和 Prompt 组件
  graphs/                   LangGraph 状态、节点、图变体与工作流
  ingestion/                文档解析、切分、父块与 manifest
  indexing/                 Qdrant collection 与索引写入
  retrieval/                Dense、BM25、RRF 通道融合、Cross Encoder 重排序
  runtime/                  运行目录、依赖构建和逐条执行
  evaluation/               复现信息、指标、诊断和评估服务
  observability/            节点与模型调用遥测
results/                    各次评测的摘要入库（跨层汇总见 results/README.md）
tests/                      按领域组织的自动化检查
```

图骨架与共用件：

- `minimal.py` / `chunkce.py`：两条图骨架，均接受 `(config, dependencies)`；
- `dependencies.py`：`RagDependencies` 依赖容器（LLM、主检索器、父文档、父块索引、CE）；
- `state.py`：图状态定义。

图节点按职责拆分：

- `planning.py`：问题分析与检索任务规划（路由表在 `routing.py`，策略定义在 `retrieval_policy.py`）；
- `retrieval_nodes.py`：混合检索与父文档 / 父块扩展；
- `answer_nodes.py`：答案生成与一致性修复；
- `grounding.py`：全节点反捏造契约（Prompt 侧校验 + 输出侧标识符核查）；
- `node_utils.py`：图节点共用的纯函数；
- `prompt_registry.py`：统一登记影响运行行为的 Prompt。

## 环境准备

项目使用 Conda 管理环境：

```powershell
conda env create -f environment.yml
conda activate enterprise-rag-bench
Copy-Item .env.example .env
```

在 `.env` 中配置所需服务：

```text
DEEPSEEK_API_KEY=
SILICONFLOW_API_KEY=
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=enterprise_rag_bench
# 重排序默认 online（硅基流动），无需本地模型；provider=local 时才需要下方两项。
CROSS_ENCODER_PROVIDER=online
CROSS_ENCODER_MODEL_PATH=C:\path\to\cross-encoder
CROSS_ENCODER_MODEL_SHA256=
```

不要提交 `.env`、本地模型、原始数据、向量数据库文件或完整运行目录。

## 配置

`configs/main.yaml` 是完整配置真源。配置加载器会：

1. 读取主配置；
2. 深度合并可选的局部覆盖文件；
3. 展开环境变量；
4. 校验字段类型、范围和字段间关系。

配置通过冻结的 Pydantic `AppConfig` 暴露。未知字段、拼写错误和缺失的必需字段都会立即
报错，业务代码不维护额外的 Python 默认配置副本。

```python
from src.config import AppConfig, load_config

config: AppConfig = load_config()
```

使用局部配置覆盖：

```powershell
python -m scripts.run_benchmark --config configs/rag100_dev_minimal.yaml
```

使用单变量消融配置：

```powershell
python -m scripts.run_benchmark --variant dense_only
```

消融配置通过复制 `AppConfig` 修改单个因素，不会修改基础配置对象。

## 数据摄取与索引

数据路径、文档 manifest、Embedding 和 Qdrant collection 信息由主配置统一管理。

执行文档摄取：

```powershell
python -m scripts.ingest
```

构建或重建向量索引：

```powershell
python -m scripts.build_index
```

摄取流程会把源文档转换为标准 `Document`，按文档语义切分内容，并将处理后的 chunk
写入 manifest。索引流程读取 manifest、生成 Embedding，并使用稳定 ID 写入 Qdrant。

BM25 索引在运行进程启动时从 manifest 构建，不需要单独维护持久化文件。

## 运行

运行前检查环境、模型、数据和索引：

```powershell
python -m scripts.preflight --allow-dirty
```

执行工作流：

```powershell
python -m scripts.run_benchmark --run-name local_run
```

执行评估与诊断：

```powershell
python -m scripts.evaluate --run-dir runs/local_run
```

`scripts.run_benchmark` 只负责命令行参数和退出码。问题读取、运行目录、依赖构建、重试策略
和逐条执行位于 `src/runtime/benchmark.py`。外部评估进程由
`src/evaluation/official.py` 统一封装。

## 运行产物

每个运行目录包含以下主要文件：

- `answers.jsonl`：生成的答案与选中文档；
- `retrieved_docs.jsonl`：最终检索上下文；
- `graph_traces.jsonl`：规划、检索与 Cross Encoder 选择轨迹；
- `question_metrics.jsonl`：逐条运行指标；
- `run_summary.json`：运行汇总；
- `run_manifest.json`：源码、配置、Prompt、依赖、模型与数据签名；
- `official_results.json`：评估结果；
- `supplementary_metrics.json`：补充诊断指标；
- `failed_questions.jsonl`：需要进一步分析的条目；
- `recall_funnel.json`：检索阶段诊断。

运行目录默认只保存在本地，不进入 Git 仓库。

## 可复现性

运行清单记录以下信息：

- Git commit、dirty 状态和源码树 hash；
- 完整解析配置及其 hash；
- 所有运行 Prompt 的 hash；
- Python、Conda、Torch 和 CUDA 环境信息；
- Cross Encoder 标识、路径和模型目录 tree hash；
- 问题文件、文档 manifest、Embedding 和 Qdrant 信息；
- 请求模型标识和 API 实际返回的模型标识；
- 外部评估器 commit。

开发运行可以在 dirty 工作区执行，但清单会完整记录状态。需要严格复现时，应先提交相关
源码、配置和自动化检查，再运行不带 `--allow-dirty` 的预检。

## 实验与结果归档

每次评测的摘要入库 `results/<run_name>/`（`official_results.json`、`block_funnel.json`、
3× strict majority 复判结果）；全量运行目录（graph_traces、日志等）留在 gitignore 的
`runs/`，只本地留存。跨层汇总见 `results/README.md`。

为隔离不同索引层的实验，开发分支只保留当前开发线；被验证并否决的层以 **tag** 承载
（代码状态 + config + 结果摘要），不占分支命名空间。各层结论：

| 层                     | 切块 / Embedding                | 结论                       |
| ---------------------- | ------------------------------- | -------------------------- |
| old 语料层（采纳）     | 旧切块 / Qwen3-0.6B             | 59.0 → sb4 62.0 → ce6 63.0 |
| v2 结构恢复切块        | 结构恢复 / Qwen3-0.6B           | 56.0 否决                  |
| v3 叶粒度切块          | 叶粒度 / Qwen3-0.6B             | 55.0 否决                  |
| v3 + Qwen3-8B embed    | 叶粒度 / Qwen3-8B (4096)        | 58.0 否决                  |
| old 切块 + bge-m3      | 旧切块 / bge-m3                 | 58.0 否决                  |
| v3ov 叶粒度 + 邻接窗   | 叶粒度+邻接窗 / Qwen3-8B (4096) | 65.0 否决（Δ−1 落在噪声带） |

⚠️ 上表前五层的分数是在已移除的 `full` 管线上测的：作为历史结论仍成立，但用当前代码
无法复现。最后一层是唯一在 `minimal` 管线上测的层——同链基线 66.0%，Δ=−1.0 落在生成
非确定性噪声带内，无净增益，**切块形状 × embed 容量联合轴正式关闭**。

被否决层的 config 会显式钉死 `retrieval.single_document_budget`，防止继承 `main.yaml`
后来的默认值而让历史分数不可复现；改动该默认值时须同步检查这些复现锁。

## 质量检查

```powershell
python -m pytest -q
python -m compileall src scripts tests
python -m ruff check .
git diff --check
python -m scripts.preflight --allow-dirty
```

Ruff 只用于静态检查，不运行自动格式化。测试按 planning、retrieval、answer、runtime 和
evaluation 等领域组织，公共 fake 位于 `tests/rag_test_support.py`。

## 故障排查

配置加载失败：

- 检查 `configs/main.yaml` 和局部覆盖文件中的字段名称；
- 检查环境变量是否已展开；
- 根据 Pydantic 错误定位缺失字段或无效字段关系。

Cross Encoder 失败：

- 检查模型目录是否存在；
- 检查模型 tree hash 是否与配置一致；
- 检查 CUDA、Torch 和本机驱动环境；
- 查看异常状态和推理耗时记录。

检索异常：

- 检查 manifest 是否与当前数据一致；
- 检查 Qdrant collection、向量维度和数据数量；
- 查看 `retrieval_stage_history`、`rerank_history` 和 recall funnel；
- 确认 Dense、BM25 通道是否返回预期候选。

模型或评估请求失败：

- 检查 API 地址、密钥和模型配置；
- 检查运行清单中的请求模型与实际模型；
- 查看 `model_calls` 和节点遥测中的错误信息。

## 出处

本项目初始代码基座来自 [Batman0x0000001/RAG-Bench](https://github.com/Batman0x0000001/RAG-Bench)
（2026-07）；2026-08 起在其上继续开发，检索管线、图骨架、评测与复现基建等均已大幅改动。
