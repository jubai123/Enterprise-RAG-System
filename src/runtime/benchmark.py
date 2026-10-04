from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import httpx
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from openai import APIConnectionError
from qdrant_client.http.exceptions import ResponseHandlingException

from src.chains.langchain_rag import build_chat_model
from src.config import AppConfig
from src.evaluation.answer_writer import (
    append_answer,
    append_graph_trace,
    append_retrieved_docs,
)
from src.evaluation.benchmark_report import matches_source_type
from src.evaluation.metrics import build_question_metrics, write_question_metrics_outputs
from src.evaluation.reproducibility import (
    build_run_manifest,
    write_run_manifest,
)
from src.graphs.chunkce import build_chunkce_graph
from src.graphs.dependencies import RagDependencies
from src.graphs.minimal import build_minimal_graph
from src.graphs.prompt_registry import benchmark_prompt_texts
from src.ingestion.parent_blocks import ParentBlockIndex
from src.ingestion.parse_documents import read_manifest
from src.retrieval.cross_encoder import CrossEncoderScorer
from src.retrieval.embeddings import build_embeddings
from src.retrieval.vector_retriever import (
    build_hybrid_candidate_retriever,
    build_parent_document_store,
)

# 线上 API 瞬时断连的题目级退避重试集。注意 build_chat_model 走 langchain_anthropic
# （DeepSeek anthropic 端点），断连抛 anthropic.APIConnectionError，与 openai 客户端
# 同名类是不同类——两者都在列，否则单次 DeepSeek 掉线就终止整轮（85/100 题曾因此丢）。
CONNECTION_ERRORS = (
    APIConnectionError,
    AnthropicAPIConnectionError,
    httpx.ConnectError,
    httpx.ReadError,
    ResponseHandlingException,
)
RUN_FILENAMES = (
    "answers.jsonl",
    "retrieved_docs.jsonl",
    "graph_traces.jsonl",
    "question_metrics.jsonl",
    "run_summary.json",
    "run_manifest.json",
)


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def answers(self) -> Path:
        return self.root / "answers.jsonl"

    @property
    def retrieved_docs(self) -> Path:
        return self.root / "retrieved_docs.jsonl"

    @property
    def graph_traces(self) -> Path:
        return self.root / "graph_traces.jsonl"

    @property
    def manifest(self) -> Path:
        return self.root / "run_manifest.json"

    @classmethod
    def prepare(cls, config: AppConfig) -> RunPaths:
        root = Path(config.output.runs_dir) / config.run_name
        root.mkdir(parents=True, exist_ok=True)
        for filename in RUN_FILENAMES:
            target = root / filename
            if target.exists():
                target.unlink()
        return cls(root=root)


def invoke_question_with_retry(
    graph,
    question: dict,
    *,
    recursion_limit: int,
    attempts: int = 5,
    base_delay_seconds: float = 15.0,
) -> dict:
    """外部 API 瞬时断连时指数退避；其他异常立即终止。"""
    question_id = question["question_id"]
    for attempt in range(1, attempts + 1):
        try:
            return graph.invoke(
                {
                    "question": question["question"],
                    "source_types": question.get("source_types"),
                },
                config={"recursion_limit": recursion_limit},
            )
        except CONNECTION_ERRORS as exc:
            if attempt == attempts:
                raise
            delay = base_delay_seconds * (2 ** (attempt - 1))
            logging.warning(
                "Transient connection error on %s (attempt %d/%d), "
                "retrying in %.0fs: %s",
                question_id,
                attempt,
                attempts,
                delay,
                exc,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")


def iter_questions(
    path: str | Path,
    limit: int | None = None,
    source_type: str | None = None,
    question_type: str | None = None,
    include_mixed_sources: bool = False,
    include_question_ids: set[str] | None = None,
) -> Iterator[dict]:
    yielded = 0
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            question = json.loads(line)
            if include_question_ids is not None:
                if question.get("question_id") not in include_question_ids:
                    continue
            elif not matches_source_type(
                question,
                source_type,
                include_mixed_sources=include_mixed_sources,
            ):
                continue
            if question_type and question.get("question_type") != question_type:
                continue
            yield question
            yielded += 1
            if limit is not None and yielded >= limit:
                break


def build_dependencies(config: AppConfig) -> RagDependencies:
    embeddings = build_embeddings(config.embedding)
    llm = build_chat_model(config.llm)
    manifest_documents = read_manifest(config.data.manifest_file)
    parent_documents = build_parent_document_store(manifest_documents)
    logging.info("Building local BM25 index from %d chunks", len(manifest_documents))
    retriever = build_hybrid_candidate_retriever(
        config.qdrant,
        embeddings,
        manifest_documents,
        dense_k=config.retrieval.dense_candidate_k,
        bm25_k=config.retrieval.bm25_candidate_k,
        candidate_k=config.retrieval.hybrid_candidate_k,
        rrf_k=config.retrieval.channel_rrf_k,
        text_section_weight=config.retrieval.text_section_weight,
        mode=config.retrieval.mode,
    )
    cross_encoder = CrossEncoderScorer(
        provider=config.cross_encoder.provider,
        model_path=config.cross_encoder.model_path,
        model_id=config.cross_encoder.model_id,
        device=config.cross_encoder.device,
        base_url=config.cross_encoder.base_url,
        api_key=config.cross_encoder.api_key,
        batch_size=config.cross_encoder.batch_size,
        predict_timeout_seconds=config.cross_encoder.predict_timeout_seconds,
        max_retries=config.cross_encoder.max_retries,
        retry_backoff_seconds=config.cross_encoder.retry_backoff_seconds,
    )
    # chunkce 路径的父块索引(预建 artifact)。
    parent_blocks = None
    if config.graph.mode == "chunkce":
        blocks_file = config.data.parent_blocks_file
        if blocks_file and Path(blocks_file).exists():
            parent_blocks = ParentBlockIndex.from_path(blocks_file)
            logging.info(
                "Loaded parent-block index: %d blocks from %s",
                parent_blocks.block_count,
                blocks_file,
            )
        else:
            logging.warning(
                "graph.mode=chunkce but parent_blocks_file=%r not found; "
                "falling back to identity expansion (no sibling context)",
                blocks_file,
            )
    logging.info("Local BM25 index is ready")
    return RagDependencies(
        llm=llm,
        retriever=retriever,
        parent_documents=parent_documents,
        cross_encoder=cross_encoder,
        parent_blocks=parent_blocks,
    )


def execute_benchmark(
    config: AppConfig,
    *,
    limit: int | None = None,
    question_source_type: str | None = None,
    question_type: str | None = None,
    all_questions: bool = False,
    include_mixed_source_questions: bool = False,
    include_question_ids: set[str] | None = None,
) -> RunPaths:
    paths = RunPaths.prepare(config)
    source_type = None if all_questions else question_source_type or config.data.source_type
    dependencies = build_dependencies(config)
    if config.graph.mode == "minimal":
        graph = build_minimal_graph(config, dependencies)
    elif config.graph.mode == "chunkce":
        graph = build_chunkce_graph(config, dependencies)
    else:
        raise ValueError(
            f"unsupported graph.mode={config.graph.mode!r}; expected 'minimal' or 'chunkce'"
        )
    run_manifest = build_run_manifest(
        config,
        project_root=Path.cwd(),
        prompt_texts=benchmark_prompt_texts(),
    )
    write_run_manifest(paths.manifest, run_manifest)

    logging.info("Question source filter: %s", source_type or "all")
    question_metrics: list[dict] = []
    for question in iter_questions(
        config.data.questions_file,
        limit=limit,
        source_type=source_type,
        question_type=question_type,
        include_mixed_sources=include_mixed_source_questions,
        include_question_ids=include_question_ids,
    ):
        question_id = question["question_id"]
        logging.info("Answering %s", question_id)
        state = invoke_question_with_retry(
            graph,
            question,
            recursion_limit=config.graph.recursion_limit,
        )
        append_answer(
            paths.answers,
            question_id,
            state.get("answer", ""),
            state.get("document_ids", []),
        )
        append_retrieved_docs(
            paths.retrieved_docs,
            question_id,
            state.get("retrieved_docs", []),
        )
        append_graph_trace(paths.graph_traces, question_id, state)
        question_metrics.append(
            build_question_metrics(
                question_id,
                state,
                config.pricing,
            )
        )

    write_question_metrics_outputs(question_metrics, paths.root)
    run_manifest["observed_model_ids"] = sorted(
        {
            str(call.get("model"))
            for row in question_metrics
            for call in row.get("model_calls", [])
            if call.get("model")
        }
    )
    run_manifest["completed_question_count"] = len(question_metrics)
    write_run_manifest(paths.manifest, run_manifest)
    logging.info("Wrote answers to %s", paths.answers)
    return paths
