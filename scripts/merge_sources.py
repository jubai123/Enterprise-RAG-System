#!/usr/bin/env python3
"""Merge confluence+jira documents into the corpus manifest.

Keeps the existing github chunks (read from the current manifest) and appends
confluence/jira chunks parsed from data/raw/generated_data/sources. Writes the
combined manifest so scripts/build_index.py can re-index Qdrant with all sources.

Backs up the previous manifest to <manifest>.bak.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.ingestion.parse_documents import (
    parse_documents,
    read_manifest,
    write_manifest,
)

SOURCES = ("confluence", "jira")


def main() -> None:
    config = load_config("configs/main.yaml")
    manifest = Path(config.data.manifest_file)
    sources_dir = Path(config.data.documents_dir)

    existing = read_manifest(manifest)
    print(f"existing manifest: {len(existing)} chunks (github)")

    backup = manifest.with_suffix(manifest.suffix + ".bak")
    shutil.copy2(manifest, backup)
    print(f"backed up manifest to {backup}")

    combined = list(existing)
    for source in SOURCES:
        docs = parse_documents(sources_dir, source_type=source)
        print(f"parsed {source}: {len(docs)} chunks")
        combined.extend(docs)

    write_manifest(combined, manifest)
    from collections import Counter
    by_src = Counter(d.metadata.get("source_type") for d in combined)
    print(f"combined manifest: {len(combined)} chunks; by source: {dict(by_src)}")


if __name__ == "__main__":
    main()
