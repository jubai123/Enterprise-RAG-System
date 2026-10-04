# 索引层评测结果归档

为隔离不同索引层的实验，每个候选层独立承载（代码状态 + config + 结果摘要），避免相互
污染；本仓库只保留当前开发线的代码与结果摘要。原始 run 全量文件（graph_traces / logs 等）
在 gitignore 的 `runs/`，只本地留存；本目录只入库摘要（`official_results.json` +
`block_funnel.json` + 3× strict majority 复判结果）。

## 各索引层一览（rag100_dev 100 题 correctness）

⚠️ 下表前五层的数据是在**已被移除的 `full` 完整管线**上测的（2026-09-21 起只保留
`minimal` / `chunkce` 两个 graph.mode）。这些分数作为历史结论仍然成立，但**当前代码已无法
复现**——要复现需回到对应层的代码状态。

| 层 | Qdrant collection | 语料/切块 | Embedding | correctness | 判定 |
|---|---|---|---|---|---|
| old | `enterprise_rag_bench` | 旧语料 all_documents.jsonl / 旧切块 | Qwen3-0.6B | old 59.0 → sb4 62.0 → ce6 63.0 | ✅ 采纳 |
| v2 | `enterprise_rag_bench_v2` | data/processed/v2 / 结构恢复切块 | Qwen3-0.6B | 56.0 | ❌ 否决 |
| v3 | `enterprise_rag_bench_v3` | data/processed/v3 / 叶粒度切块 | Qwen3-0.6B | 55.0 | ❌ 否决 |
| v3q8b | `enterprise_rag_bench_v3q8b` | data/processed/v3 / 叶粒度切块 | Qwen3-8B(4096) | 58.0 | ❌ 否决 |
| bgem3 | `enterprise_rag_bench_bgem3` | 旧语料 / 旧切块 | bge-m3 | 58.0 | ❌ 否决 |
| v3ov8b | `enterprise_rag_bench_v3ov8b` | data/processed/v3ov / 叶粒度+邻接窗 | Qwen3-8B(4096) | 65.0(※) | ❌ 否决 |

※ v3ov8b 是唯一在 **minimal 管线**上测的层（2026-09-08）：同链 minimal + old 索引基线 = 66.0%，
Δ=−1.0 落在生成非确定性噪声带内，无净增益。**切块形状 × embed 容量联合轴正式关闭**。

## 现存结果

| 目录 | 线 | correctness(dev 100 / frozen 45) |
|---|---|---|
| `rag100_dev_parentce_perquery/` | chunkce per-query | 71.0 raw → **74.0**（3× strict majority 修正） |
| `rag100_frozen_parentce_perquery/` | chunkce per-query | 82.2 raw → **86.7**（修正） |
| `rag100_frozen_minimal/` | minimal | 修正跑产物（raw 基线 84.4） |

## 复现注意

否决层 config 曾各自钉死 `retrieval.single_document_budget`（v2/v3=1、v3q8b=4，bgem3 原 config=4），
防止继承 main.yaml 默认导致历史分数不可复现。语料/embedding 差异经 env
（`QDRANT_COLLECTION` / `SILICONFLOW_EMBEDDING_MODEL`）注入，见各层 config 注释。

当前 main.yaml 的默认档：`graph.mode: minimal` + `single_document_budget: 4` + `max_parent_chunks: 8`。
改这三个值都会让上表分数不可复现。
