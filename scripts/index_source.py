#!/usr/bin/env python3
"""增量索引：把单个 source 摄取、嵌入并合入累积 manifest。

用法:
    python -m scripts.index_source --source linear --config configs/rag100_dev.yaml

流程: 解压 archives -> 解析该 source chunks -> 写 per-source manifest ->
嵌入并 upsert 到 Qdrant（稳定 chunk_id，幂等）-> 按 chunk_id 去重合入累积 manifest。

幂等性: 重复运行会重新嵌入并 upsert（不损坏数据），累积 manifest 按 chunk_id 去重不变。
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.indexing.qdrant_index import index_documents
from src.ingestion.parse_documents import (
    extract_archives,
    parse_documents,
    read_manifest,
    write_manifest,
)
from src.retrieval.embeddings import build_embeddings
from src.utils.logging import setup_logging

DEFAULT_CUMULATIVE_MANIFEST = "data/processed/all_documents.jsonl"
FALLBACK_MANIFEST = "data/processed/github_documents.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally index a new corpus source.")
    parser.add_argument("--source", required=True, help="Source type to index, e.g. linear.")
    parser.add_argument("--config", default=None, help="Optional YAML/JSON config override file.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--manifest",
        default=DEFAULT_CUMULATIVE_MANIFEST,
        help="Cumulative manifest to merge into (default all_documents.jsonl).",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    # 1. 解压 + 解析该 source 的 chunks。
    extract_archives(config.data.archives_dir, config.data.documents_dir)
    try:
        docs = parse_documents(config.data.documents_dir, source_type=args.source)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Cannot parse source={args.source}: {exc}\n"
            f"Download its archives into {config.data.archives_dir} first "
            f"(e.g. linear_slice_*.zip), then retry."
        ) from exc
    logging.info("Parsed %d chunks for source=%s", len(docs), args.source)
    if not docs:
        raise SystemExit(f"No chunks parsed for source={args.source}.")

    # 2. per-source manifest（独立保存，便于单源重跑）。
    per_source = Path(config.data.manifest_file).with_name(f"{args.source}_documents.jsonl")
    write_manifest(docs, per_source)
    logging.info("Wrote per-source manifest %s (%d chunks)", per_source, len(docs))

    # 3. 向量增量写入：稳定 chunk_id -> Qdrant upsert，不 recreate。
    embeddings = build_embeddings(config.embedding)
    index_documents(docs, embeddings, config.qdrant, batch_size=args.batch_size)
    logging.info("Upserted %d vectors into Qdrant %s", len(docs), config.qdrant.collection)

    # 4. 合入累积 manifest：按 chunk_id 去重追加，写前备份。
    cumulative = Path(args.manifest)
    if cumulative.exists():
        base = cumulative
    else:
        base = Path(FALLBACK_MANIFEST)
        logging.info("Cumulative manifest missing; bootstrapping from %s", base)
    existing = read_manifest(base) if base.exists() else []
    if cumulative.exists():
        backup = cumulative.with_suffix(cumulative.suffix + ".bak")
        shutil.copy2(cumulative, backup)
        logging.info("Backed up cumulative manifest to %s", backup)
    seen = {doc.metadata.get("chunk_id") for doc in existing}
    added = [doc for doc in docs if doc.metadata.get("chunk_id") not in seen]
    combined = existing + added
    write_manifest(combined, cumulative)
    by_source = Counter(doc.metadata.get("source_type") for doc in combined)
    logging.info(
        "Cumulative manifest: %d chunks (%d new from %s); by source: %s",
        len(combined),
        len(added),
        args.source,
        dict(by_source),
    )


if __name__ == "__main__":
    main()
