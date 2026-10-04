#!/usr/bin/env python3
"""全量索引重建：recreate Qdrant collection，重切全部 source，重嵌并重写累积 manifest。

用法:
    python -m scripts.rebuild_index --config configs/rag100_dev.yaml
    python -m scripts.rebuild_index --config configs/rag100_dev.yaml --skip-embed  # 只重切，不嵌入

与 index_source（增量、单源）互补：本脚本用于解析/切片逻辑变更后的全量重做，
以稳定 chunk_id 重新 upsert（recreate 后无旧向量残留），并从头合并 per-source manifest。
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
from src.indexing.qdrant_index import index_documents, recreate_collection
from src.ingestion.parse_documents import (
    parse_documents,
    read_manifest,
    write_manifest,
)
from src.retrieval.embeddings import build_embeddings
from src.utils.logging import setup_logging

DEFAULT_MANIFEST = "data/processed/all_documents.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the full vector index from scratch.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Only re-chunk and rewrite manifests; do not embed or touch Qdrant.",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Cumulative manifest to rebuild (default data/processed/all_documents.jsonl).",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    sources = args.sources or config.data.source_scope or ["github"]
    cumulative = Path(args.manifest)

    embeddings = build_embeddings(config.embedding) if not args.skip_embed else None
    if embeddings is not None:
        recreate_collection(config.qdrant)
        logging.info("Recreated collection %s", config.qdrant.collection)

    per_source_files: list[Path] = []
    for source in sources:
        docs = parse_documents(config.data.documents_dir, source_type=source)
        logging.info("source=%s: %d chunks", source, len(docs))
        if not docs:
            raise SystemExit(f"No chunks parsed for source={source}.")
        per_source = Path(config.data.manifest_file).with_name(f"{source}_documents.jsonl")
        write_manifest(docs, per_source)
        per_source_files.append(per_source)
        if embeddings is not None:
            index_documents(docs, embeddings, config.qdrant, batch_size=args.batch_size)
            logging.info("Upserted %d vectors into %s", len(docs), config.qdrant.collection)

    if cumulative.exists():
        from datetime import datetime

        backup = cumulative.with_suffix(
            cumulative.suffix + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.copy2(cumulative, backup)
        logging.info("Backed up cumulative manifest to %s", backup)

    seen: set[str] = set()
    combined: list = []
    for per_source in per_source_files:
        for doc in read_manifest(per_source):
            chunk_id = str(doc.metadata.get("chunk_id"))
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            combined.append(doc)
    write_manifest(combined, cumulative)
    by_source = Counter(doc.metadata.get("source_type") for doc in combined)
    logging.info(
        "Cumulative manifest: %d chunks; by source: %s",
        len(combined),
        dict(by_source),
    )


if __name__ == "__main__":
    main()
