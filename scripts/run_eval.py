#!/usr/bin/env python3
"""One-click evaluation driver for the GitHub N39 development set.

Orchestrates the existing pipeline:
  1. Qdrant readiness + vector index (rebuild when missing/mismatched; or
     restore from a local .snapshot file when --snapshot is given)
  2. Certification preflight
  3. Benchmark (answer the 39 blind questions)
  4. Official evaluation (DeepSeek LLM as judge)
  5. Certification report

Every phase shells out to the existing `scripts/*.py` entry points so this
driver stays a thin coordinator. Run it with the project's virtualenv Python.

Snapshot workflow (reuse an existing index instead of rebuilding):
  source box:   python -m scripts.run_eval --create-snapshot <out.snapshot>
                (requires Qdrant up with the collection; writes a portable file)
  target box:   python -m scripts.run_eval --snapshot <out.snapshot> [...]
                (restores the collection from the local .snapshot file when the
                 index is missing/mismatched, else falls back to a rebuild)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Running the file directly (or via run_eval.sh's `exec`) puts only scripts/
# on sys.path, so `import src` fails; make the project root importable.
sys.path.insert(0, str(PROJECT_ROOT))
_self = str(Path(__file__).resolve())


def _log(message: str) -> None:
    print(f"\n===== {message} =====", flush=True)


def _run_module(module: str, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    _log(f"python -m {module} {' '.join(args)}")
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=PROJECT_ROOT,
        check=check,
    )


def _wait_qdrant(config, timeout: int = 120) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            _qdrant_client(config, timeout=2).get_collections()
            return
        except Exception:
            time.sleep(2)
    raise SystemExit(
        f"Qdrant did not become ready at {config.qdrant.url} within {timeout}s"
    )


def _start_qdrant(value: str, config) -> None:
    _log(f"Starting Qdrant ({value})")
    if value == "docker":
        subprocess.run(
            [
                "docker", "run", "-d", "--name", "qdrant-ragbench",
                "-p", "6333:6333", "-v", "qdrant_storage:/qdrant/storage",
                "qdrant/qdrant",
            ],
            check=False,
        )
    else:
        # Resolve to an absolute path: Popen execs a bare name with no "/" via
        # PATH lookup (which won't find a cwd-relative file), but is_file()
        # already checks relative to PROJECT_ROOT. This makes "qdrant" and
        # "./qdrant" both work.
        binary = Path(value).expanduser().resolve()
        if not binary.is_file():
            raise SystemExit(f"Qdrant binary not found: {binary}")
        subprocess.Popen(
            [str(binary), "--disable-telemetry"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    _wait_qdrant(config)


def _qdrant_point_count(config) -> int:
    return _qdrant_client(config, timeout=10).count(
        config.qdrant.collection, exact=True
    ).count


def _manifest_doc_count(config) -> int:
    path = PROJECT_ROOT / config.data.manifest_file
    if not path.is_file():
        return 0
    return sum(1 for line in path.open("r", encoding="utf-8") if line.strip())


def _qdrant_url(config) -> str:
    # Force IPv4 for localhost so the client never lands on ::1 (Qdrant listens
    # on IPv4; a transparent proxy answering on ::1 returns 502).
    url = config.qdrant.url
    for scheme in ("http", "https"):
        prefix = f"{scheme}://localhost"
        if url.startswith(prefix):
            return url.replace(prefix, f"{scheme}://127.0.0.1", 1)
    return url


def _qdrant_client(config, timeout: int):
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=_qdrant_url(config),
        api_key=config.qdrant.api_key or None,
        timeout=timeout,
    )


def _preflight(config_path: str, allow_dirty: bool) -> dict:
    cmd = ["scripts.preflight", "--config", config_path]
    if allow_dirty:
        cmd.append("--allow-dirty")
    result = subprocess.run(
        [sys.executable, "-m", *cmd],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"status": "FAIL", "failures": [f"preflight output unparsable (exit {result.returncode})"]}


def _count_lines(path: Path) -> int:
    return sum(1 for _ in path.open("r", encoding="utf-8"))


def _snapshot_roots() -> list[Path]:
    # Qdrant writes collection snapshots under <storage_parent>/snapshots, so
    # cover the layouts produced by _start_qdrant (cwd=PROJECT_ROOT) and by a
    # locally-run binary under qdrant_local/.
    return [
        PROJECT_ROOT / "snapshots",
        PROJECT_ROOT / "qdrant_local" / "snapshots",
        PROJECT_ROOT / "storage" / "snapshots",
    ]


def _find_snapshot_file(name: str) -> Path | None:
    for root in _snapshot_roots():
        direct = root / name
        if direct.is_file():
            return direct
        if root.is_dir():
            for sub in root.iterdir():
                candidate = sub / name
                if candidate.is_file():
                    return candidate
    return None


def _create_snapshot(config, output: Path) -> int:
    client = _qdrant_client(config, timeout=60)
    description = client.create_snapshot(config.qdrant.collection, wait=True)
    if description is None or not description.name:
        raise SystemExit("Qdrant did not return a snapshot name")
    source = _find_snapshot_file(description.name)
    if source is None:
        raise SystemExit(
            f"Created snapshot {description.name!r} but could not locate the file "
            f"under {[str(root) for root in _snapshot_roots()]!r}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    print(
        f"  {source.name}: {source.stat().st_size / 1_000_000:.1f} MB -> {output}",
        flush=True,
    )
    return _qdrant_point_count(config)


def _restore_snapshot(config, snapshot: Path) -> int:
    client = _qdrant_client(config, timeout=60)
    if client.collection_exists(config.qdrant.collection):
        print(f"  dropping existing collection {config.qdrant.collection}", flush=True)
        client.delete_collection(config.qdrant.collection)
    # Qdrant reads file:/// URIs from its own filesystem (same host).
    client.recover_snapshot(
        config.qdrant.collection, snapshot.resolve().as_uri(), wait=True
    )
    return _qdrant_point_count(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--official-repo", default="external/EnterpriseRAG-Bench")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Limit benchmark questions (smoke test).")
    parser.add_argument("--start-qdrant", default=None,
                        help="Start Qdrant if unreachable: 'docker' or a path to a qdrant binary.")
    parser.add_argument("--force-index", action="store_true", help="Always rebuild the Qdrant index.")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--snapshot", default=None,
                        help="Restore the Qdrant collection from a local .snapshot file "
                             "instead of rebuilding (falls back to a rebuild on failure/mismatch).")
    parser.add_argument("--create-snapshot", default=None,
                        help="Write the Qdrant collection snapshot to a local file, then exit. "
                             "Requires Qdrant to be running with the collection.")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--skip-certify", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Pass --resume to the official evaluation.")
    parser.add_argument("--official-correction", action="store_true",
                        help="Enable the three-judge gold-document correction flow.")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    from src.config import load_config  # requires the venv deps

    config = load_config(args.config)
    run_name = args.run_name or config.run_name
    run_dir = (PROJECT_ROOT / config.output.runs_dir / run_name).resolve()
    expected = config.data.expected_questions
    official_repo = (PROJECT_ROOT / args.official_repo).resolve()
    summary: dict[str, str] = {}
    ok = True

    # ------------------------------------------------------------------ phase 0: sanity
    if not args.skip_evaluate and not official_repo.is_dir():
        raise SystemExit(
            f"Official evaluator repo not found: {official_repo}\n"
            "Clone it first, e.g.:\n"
            "  git clone --depth 1 https://github.com/onyx-dot-app/EnterpriseRAG-Bench.git "
            "external/EnterpriseRAG-Bench\n"
            "(if direct GitHub is blocked, prefix the URL with a mirror such as "
            "https://ghfast.top/https://github.com/...)"
        )

    # ------------------------------------------------------------------ phase 1: qdrant + index
    manifest_count = _manifest_doc_count(config)
    print(f"\nExpected documents: {manifest_count}  |  questions: {expected}  |  run dir: {run_dir}")
    _log("Checking Qdrant")

    qdrant_up = False
    try:
        client = _qdrant_client(config, timeout=3)
        client.get_collections()
        qdrant_up = True
    except Exception:
        qdrant_up = False

    if not qdrant_up:
        if args.create_snapshot:
            raise SystemExit(
                f"Qdrant is not reachable at {config.qdrant.url}; --create-snapshot "
                "requires Qdrant running with the existing collection."
            )
        if args.start_qdrant:
            _start_qdrant(args.start_qdrant, config)
        else:
            raise SystemExit(
                f"Qdrant is not reachable at {config.qdrant.url}. Start it first "
                "(e.g. `docker run -d -p 6333:6333 qdrant/qdrant`) or pass --start-qdrant."
            )

    if args.create_snapshot:
        _log("Creating collection snapshot")
        output = Path(args.create_snapshot)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        count = _create_snapshot(config, output)
        print(f"  Snapshot written to {output} (collection has {count} points)")
        return

    indexed = 0
    if not args.skip_index:
        try:
            client.get_collection(config.qdrant.collection)
            indexed = _qdrant_point_count(config)
        except Exception:
            indexed = 0
        needs_build = args.force_index or indexed != manifest_count
        if needs_build:
            if indexed:
                print(f"  Qdrant has {indexed} points, manifest has {manifest_count}")
            else:
                print("  Qdrant collection missing or empty")
            restored = False
            if args.snapshot:
                snapshot = Path(args.snapshot)
                if not snapshot.is_absolute():
                    snapshot = PROJECT_ROOT / snapshot
                if snapshot.is_file():
                    print(f"  Restoring index from snapshot: {snapshot}", flush=True)
                    try:
                        restored_count = _restore_snapshot(config, snapshot)
                    except Exception as exc:
                        print(
                            f"  [warn] snapshot restore failed "
                            f"({type(exc).__name__}: {exc}); will rebuild"
                        )
                        restored_count = 0
                    if restored_count == manifest_count:
                        print(
                            f"  Snapshot restored {restored_count} points (matches manifest)"
                        )
                        indexed = restored_count
                        restored = True
                    else:
                        print(
                            f"  [warn] snapshot has {restored_count} points, "
                            f"manifest has {manifest_count}; will rebuild"
                        )
                else:
                    print(f"  [warn] snapshot file not found: {snapshot}; will rebuild")
            if not restored:
                _run_module("scripts.build_index", ["--config", args.config])
        else:
            print(f"  Qdrant index is current ({indexed} points)")

    # ------------------------------------------------------------------ phase 2: preflight
    if not args.skip_preflight:
        report = _preflight(args.config, args.allow_dirty)
        status = report.get("status")
        failures = report.get("failures", [])
        summary["preflight"] = status or "FAIL"
        if status != "PASS":
            for failure in failures:
                print(f"  [preflight] {failure}", file=sys.stderr, flush=True)
            if not args.allow_dirty:
                raise SystemExit(
                    "Preflight failed. Fix the issues above, or pass --allow-dirty to continue anyway "
                    "(note: a dirty tree makes certification fail)."
                )
            print("  Continuing despite preflight failures (--allow-dirty).")
    else:
        summary["preflight"] = "skipped"

    # ------------------------------------------------------------------ phase 3: benchmark
    if not args.skip_benchmark:
        cmd = ["scripts.run_benchmark", "--config", args.config, "--run-name", run_name]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        _run_module(cmd[0], cmd[1:])
        answers = run_dir / "answers.jsonl"
        if not answers.is_file():
            raise SystemExit(f"Benchmark did not produce {answers}")
        actual = _count_lines(answers)
        if actual != expected:
            raise SystemExit(
                f"Benchmark produced {actual}/{expected} answers; expected exactly {expected}."
            )
        summary["answers"] = f"{actual}/{expected}"
    else:
        summary["answers"] = "skipped"

    # ------------------------------------------------------------------ phase 4: evaluate
    if not args.skip_evaluate:
        cmd = [
            "scripts.evaluate",
            "--config", args.config,
            "--run-dir", str(run_dir),
            "--official-repo", str(official_repo),
            "--parallelism", str(args.parallelism),
        ]
        if args.resume:
            cmd.append("--resume")
        if args.official_correction:
            cmd.append("--official-correction")
        _run_module(cmd[0], cmd[1:])
        results_file = run_dir / "official_results.json"
        if not results_file.is_file():
            raise SystemExit(f"Evaluation did not produce {results_file}")
        summary["official_results"] = str(results_file)
    else:
        summary["official_results"] = "skipped"

    # ------------------------------------------------------------------ phase 5: certify
    if not args.skip_certify:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.certify", "--run-dir", str(run_dir)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        print(result.stdout, flush=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr, flush=True)
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError:
            report = {}
        summary["certification"] = report.get("status", "FAIL")
        ok = report.get("status") == "PASS"
    else:
        summary["certification"] = "skipped"

    # ------------------------------------------------------------------ summary
    print("\n===== SUMMARY =====")
    for key, value in summary.items():
        print(f"  {key:<18} {value}")
    print(f"  {'run_dir':<18} {run_dir}")
    for name in (
        "answers.jsonl",
        "retrieved_docs.jsonl",
        "official_results.json",
        "supplementary_metrics.json",
        "failed_questions.jsonl",
        "certification_report.json",
    ):
        path = run_dir / name
        if path.is_file():
            print(f"  {'output':<18} {path}")
    if args.skip_certify:
        print("  certification was skipped")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
