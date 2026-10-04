from __future__ import annotations

import logging
import threading
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Literal

import httpx
from langchain_core.documents import Document

from src.retrieval.candidate_text import extract_relevant_window, rank_candidate_chunks

logger = logging.getLogger(__name__)

_RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}


def _is_retryable_rerank_error(exc: Exception) -> bool:
    """连接层瞬时错误（ConnectError/ReadError/WriteError 等）与 429/5xx 可重试。"""
    if isinstance(exc, httpx.TransportError) and not isinstance(
        exc, httpx.TimeoutException
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_HTTP_STATUSES
    return False


class CrossEncoderScorer:
    """BGE reranker 打分器，支持本地 sentence-transformers 与硅基流动在线 API。

    任何失败都会立即终止当前 benchmark。provider 默认 local，pipeline 通过配置传入
    online；local 保留作 GPU 后备，online 走 {base_url}/rerank。
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        provider: Literal["local", "online"] = "local",
        model_id: str = "",
        device: str | None = None,
        base_url: str = "",
        api_key: str = "",
        chunk_chars: int = 400,
        passage_chars: int = 1000,
        batch_size: int = 32,
        predict_timeout_seconds: float = 120.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.5,
    ) -> None:
        self.provider = provider
        self.model_path = model_path
        self.model_id = model_id
        self.device = device
        self.base_url = base_url
        self.api_key = api_key
        self.chunk_chars = chunk_chars
        self.passage_chars = passage_chars
        self.batch_size = batch_size
        self.predict_timeout_seconds = predict_timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._model: Any | None = None
        self.last_status: str = "uninitialized"
        self.last_error: str | None = None
        self.last_latency_ms: float | None = None
        self.last_scored_count: int = 0

    @property
    def status(self) -> str:
        return self.last_status

    def _load(self) -> Any:
        if self.provider == "online":
            return None
        if self._model is not None:
            return self._model
        if not self.model_path or not Path(self.model_path).is_dir():
            self.last_status = "model_missing"
            self.last_error = f"model path not found: {self.model_path}"
            raise FileNotFoundError(self.last_error)
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_path, device=self.device)
            return self._model
        except Exception as exc:
            self.last_status = "load_failed"
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(self.last_error) from exc

    def _predict_with_timeout(
        self,
        model: Any,
        pairs: list[list[str]],
    ) -> list[float]:
        box: dict[str, Any] = {}
        errors: list[Exception] = []

        def _run() -> None:
            try:
                box["scores"] = model.predict(pairs, batch_size=self.batch_size)
            except Exception as exc:  # pragma: no cover - 在线程结束后重新抛出
                errors.append(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self.predict_timeout_seconds)
        if thread.is_alive():
            self.last_status = "inference_timeout"
            self.last_error = (
                f"predict exceeded {self.predict_timeout_seconds}s on {self.device}"
            )
            raise TimeoutError(self.last_error)
        if errors:
            raise errors[0]
        scores = box.get("scores")
        if scores is None:
            raise RuntimeError("cross-encoder prediction returned no scores")
        return scores

    def _predict_online(self, pairs: list[list[str]]) -> list[float]:
        """调用硅基流动 /rerank；按 batch_size 切批，响应按 results[].index 回填。

        对瞬时连接层错误（ConnectError/ReadError/WriteError 等）与 429/5xx 做
        指数退避重试——实测 SiliconFlow rerank 会偶发 SSL EOF 断连，重试可绕过；
        超时与非瞬时错误保持立即失败。
        """
        if not self.base_url or not self.api_key or not self.model_id:
            self.last_status = "inference_failed"
            self.last_error = (
                "online reranker requires base_url, api_key and model_id"
            )
            raise RuntimeError(self.last_error)
        query = pairs[0][0]
        documents = [passage for _query, passage in pairs]
        scores: list[float | None] = [None] * len(documents)
        url = f"{self.base_url.rstrip('/')}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for offset in range(0, len(documents), self.batch_size):
            batch = documents[offset : offset + self.batch_size]
            request_body = {
                "model": self.model_id,
                "query": query,
                "documents": batch,
                "top_n": len(batch),
            }
            response_payload: dict | None = None
            last_error: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    response = httpx.post(
                        url,
                        headers=headers,
                        json=request_body,
                        timeout=self.predict_timeout_seconds,
                    )
                    response.raise_for_status()
                    response_payload = response.json()
                    break
                except httpx.TimeoutException as exc:
                    self.last_status = "inference_timeout"
                    self.last_error = (
                        f"rerank request exceeded {self.predict_timeout_seconds}s"
                    )
                    raise TimeoutError(self.last_error) from exc
                except httpx.HTTPError as exc:
                    last_error = exc
                    if not _is_retryable_rerank_error(exc):
                        break
                    if attempt + 1 < self.max_retries:
                        logger.warning(
                            "rerank request failed (%s), retrying %d/%d",
                            exc,
                            attempt + 1,
                            self.max_retries - 1,
                        )
                        sleep(self.retry_backoff_seconds * (attempt + 1))
            if response_payload is None:
                self.last_status = "inference_failed"
                self.last_error = (
                    f"{type(last_error).__name__}: {last_error}"
                    if last_error is not None
                    else "rerank request failed"
                )
                raise RuntimeError(self.last_error) from last_error
            results = response_payload.get("results", [])
            for item in results:
                index = item.get("index")
                if isinstance(index, int) and 0 <= index < len(batch):
                    scores[offset + index] = item.get("relevance_score")
        missing = [index for index, score in enumerate(scores) if score is None]
        if missing:
            self.last_status = "inference_failed"
            self.last_error = f"rerank response missing scores for indexes {missing}"
            raise RuntimeError(self.last_error)
        return [score for score in scores if score is not None]

    def score_passages(self, question: str, texts: list[str]) -> list[float]:
        """直接对原始文本打分(chunk-CE 干净路径用)。

        不做任何 chunk 词法重排/抽句/窗口拼贴——texts 原样与 question 配对成
        (question, text) 交 CE。长度需与 texts 对齐返回。文本超模型窗时的截断是
        唯一允许的硬性操作,这里默认不截(语料 chunk ≤ ~3800 字符,在 bge-reranker-v2-m3
        输入窗内)。
        """
        started = perf_counter()
        self.last_status = "running"
        self.last_error = None
        self.last_latency_ms = None
        self.last_scored_count = 0
        if not texts:
            self.last_status = "success"
            self.last_latency_ms = (perf_counter() - started) * 1000
            return []
        pairs = [[question, text] for text in texts]
        try:
            if self.provider == "online":
                scores = self._predict_online(pairs)
            else:
                model = self._load()
                scores = self._predict_with_timeout(model, pairs)
            if len(scores) != len(texts) or any(score is None for score in scores):
                raise RuntimeError(
                    "cross-encoder score count mismatch: "
                    f"{len(scores)} != {len(texts)}"
                )
            self.last_status = "success"
            self.last_error = None
            self.last_latency_ms = (perf_counter() - started) * 1000
            self.last_scored_count = len(texts)
            return [float(score) for score in scores]
        except Exception as exc:
            if self.last_status not in {
                "model_missing",
                "load_failed",
                "inference_timeout",
            }:
                self.last_status = "inference_failed"
                self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_latency_ms = (perf_counter() - started) * 1000
            raise

    def score_chunks(
        self,
        question: str,
        chunks: list[Document],
    ) -> list[float]:
        """chunk-CE 干净路径:对检索命中的原始 chunk 直接打分(完整问题配对)。"""
        return self.score_passages(question, [chunk.page_content for chunk in chunks])

    def score_documents(
        self,
        question: str,
        documents_by_dsid: dict[str, list[Document]],
    ) -> dict[str, float]:
        started = perf_counter()
        self.last_status = "running"
        self.last_error = None
        self.last_latency_ms = None
        self.last_scored_count = 0
        try:
            pairs: list[list[str]] = []
            pair_dsids: list[str] = []
            for dsid, chunks in documents_by_dsid.items():
                if not chunks:
                    continue
                ordered = rank_candidate_chunks(question, [question], chunks)
                parts: list[str] = []
                total = 0
                for chunk in ordered:
                    window = extract_relevant_window(
                        chunk.page_content,
                        question,
                        [question],
                        self.chunk_chars,
                    )
                    parts.append(window)
                    total += len(window) + 3
                    if total >= self.passage_chars:
                        break
                passage = "\n".join(parts)[: self.passage_chars + 600]
                pairs.append([question, passage])
                pair_dsids.append(dsid)

            if not pairs:
                self.last_status = "success"
                self.last_latency_ms = (perf_counter() - started) * 1000
                return {}

            if self.provider == "online":
                scores = self._predict_online(pairs)
            else:
                model = self._load()
                scores = self._predict_with_timeout(model, pairs)
            if len(pair_dsids) != len(scores):
                raise RuntimeError(
                    "cross-encoder dsid/score count mismatch: "
                    f"{len(pair_dsids)} != {len(scores)}"
                )
            self.last_status = "success"
            self.last_error = None
            self.last_latency_ms = (perf_counter() - started) * 1000
            self.last_scored_count = len(pair_dsids)
            return {
                dsid: float(score)
                for dsid, score in zip(pair_dsids, scores)
                if score is not None
            }
        except Exception as exc:
            if self.last_status not in {
                "model_missing",
                "load_failed",
                "inference_timeout",
            }:
                self.last_status = "inference_failed"
                self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_latency_ms = (perf_counter() - started) * 1000
            raise
