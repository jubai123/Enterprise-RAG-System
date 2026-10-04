"""离线预建 chunkce parent_block_llm 的父块索引 artifact。

纯离线变换：从现有 manifest 叶子（data/processed/all_documents.jsonl）重建 (dsid,section)
父块，长度按 retrieval.parent_max_chars 受控；产物 = data.parent_blocks_file。不动
Qdrant / embedding / manifest 本体，幂等可重建。

用法:
  python scripts/build_parent_blocks.py --config configs/rag100_dev_parentce.yaml
  python scripts/build_parent_blocks.py --config <yaml> --path <out.jsonl> --max-chars 8000
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.config import load_config
from src.ingestion.parent_blocks import (
    build_parent_blocks,
    load_parent_blocks,
    summarize,
    write_parent_blocks,
)
from src.ingestion.parse_documents import read_manifest
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build parent-block index from manifest.")
    parser.add_argument("--config", default=None, help="YAML config (defaults from main.yaml).")
    parser.add_argument("--path", default=None, help="Output jsonl path (overrides data.parent_blocks_file).")
    parser.add_argument("--max-chars", type=int, default=None, help="Parent block char cap.")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    manifest_file = Path(config.data.manifest_file)
    out_path = Path(args.path or config.data.parent_blocks_file)
    if not out_path or str(out_path) == ".":
        raise SystemExit(
            "No parent_blocks_file: set data.parent_blocks_file in config or pass --path."
        )
    max_chars = args.max_chars or config.retrieval.parent_max_chars

    print(f"[parent-blocks] manifest={manifest_file}", flush=True)
    documents = read_manifest(manifest_file)
    print(f"[parent-blocks] leaves={len(documents)}  max_chars={max_chars}", flush=True)
    blocks = build_parent_blocks(documents, max_chars)

    stats = summarize(blocks, max_chars)
    print(
        "[parent-blocks] summary: "
        f"blocks={stats['blocks']} leaves={stats['leaves']} "
        f"mean_chars={stats['mean_chars']} max_chars={stats['max_chars']} "
        f"multi_leaf_blocks={stats['multi_leaf_blocks']} "
        f"oversized_single_blocks={stats['oversized_single_blocks']}",
        flush=True,
    )

    # 覆盖校验:chunk_id → 父块 必须 1:1 覆盖全部叶子(无空洞、无重复归属)。
    covered: Counter[str] = Counter()
    for block in blocks:
        covered.update(block.child_chunk_ids)
    duplicate = [cid for cid, count in covered.items() if count > 1]
    if duplicate:
        raise SystemExit(f"[parent-blocks] chunk mapped to multiple blocks: {duplicate[:5]}")
    missing = [chunk.metadata.get("chunk_id") for chunk in documents
               if chunk.metadata.get("chunk_id") not in covered]
    if missing:
        raise SystemExit(
            f"[parent-blocks] {len(missing)} leaves have no parent block "
            f"(first 5: {missing[:5]})"
        )
    write_parent_blocks(out_path, blocks)
    print(f"[parent-blocks] wrote {out_path}", flush=True)

    # 复载自检
    reloaded = load_parent_blocks(out_path)
    assert len(reloaded) == len(blocks)
    print(f"[parent-blocks] reload ok: {len(reloaded)} blocks", flush=True)


if __name__ == "__main__":
    main()
