from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from langchain_core.documents import Document

from src.retrieval.cross_encoder import CrossEncoderScorer


class _FakeCrossEncoderModel:
    def __init__(self, delay: float = 0.0, score: float = 1.0) -> None:
        self.delay = delay
        self.score = score

    def predict(self, pairs, batch_size=32):
        import time

        time.sleep(self.delay)
        return [self.score] * len(pairs)


def test_cross_encoder_timeout_aborts() -> None:
    scorer = CrossEncoderScorer(
        model_path="unused",
        device="cuda",
        predict_timeout_seconds=0.05,
    )
    scorer._model = _FakeCrossEncoderModel(delay=0.2)
    documents = {"a": [Document(page_content="evidence", metadata={"chunk_id": "a::0"})]}

    with pytest.raises(TimeoutError, match="predict exceeded"):
        scorer.score_documents("question", documents)

    assert scorer.last_status == "inference_timeout"


def test_cross_encoder_inference_failure_aborts() -> None:
    class BrokenModel:
        def predict(self, pairs, batch_size=32):
            raise RuntimeError("broken inference")

    scorer = CrossEncoderScorer(model_path="unused", device="cuda")
    scorer._model = BrokenModel()
    documents = {"a": [Document(page_content="evidence", metadata={"chunk_id": "a::0"})]}

    with pytest.raises(RuntimeError, match="broken inference"):
        scorer.score_documents("question", documents)

    assert scorer.last_status == "inference_failed"


def test_cross_encoder_missing_model_aborts(tmp_path: Path) -> None:
    scorer = CrossEncoderScorer(model_path=str(tmp_path / "missing"), device="cuda")
    documents = {"a": [Document(page_content="evidence", metadata={"chunk_id": "a::0"})]}

    with pytest.raises(FileNotFoundError, match="model path not found"):
        scorer.score_documents("question", documents)

    assert scorer.last_status == "model_missing"


def test_cross_encoder_load_failure_aborts(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    scorer = CrossEncoderScorer(model_path=str(model_dir), device="cuda")
    documents = {"a": [Document(page_content="evidence", metadata={"chunk_id": "a::0"})]}

    def fail_load(*args, **kwargs):
        raise RuntimeError("broken model")

    monkeypatch.setattr("sentence_transformers.CrossEncoder", fail_load)
    with pytest.raises(RuntimeError, match="broken model"):
        scorer.score_documents("question", documents)

    assert scorer.last_status == "load_failed"


def test_cross_encoder_resets_stale_telemetry_per_call() -> None:
    scorer = CrossEncoderScorer(model_path="unused", device="cuda")
    scorer.last_status = "inference_failed"
    scorer.last_error = "stale error"
    scorer.last_latency_ms = 999.0
    scorer.last_scored_count = 42

    scores = scorer.score_documents("question", {})

    assert scores == {}
    assert scorer.last_status == "success"
    assert scorer.last_error is None
    assert scorer.last_latency_ms is not None
    assert scorer.last_scored_count == 0


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _online_scorer(batch_size: int = 2) -> CrossEncoderScorer:
    return CrossEncoderScorer(
        provider="online",
        model_id="BAAI/bge-reranker-v2-m3",
        base_url="https://api.siliconflow.com/v1",
        api_key="sk-test",
        batch_size=batch_size,
    )


def _online_documents(count: int) -> dict[str, list[Document]]:
    return {
        f"dsid_{index}": [Document(page_content=f"evidence {index}")]
        for index in range(count)
    }


def test_cross_encoder_online_batches_and_orders_scores(monkeypatch) -> None:
    scorer = _online_scorer(batch_size=2)
    requests: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        requests.append(json)
        documents = json["documents"]
        results = [
            {"index": index, "relevance_score": len(documents) - index}
            for index in range(len(documents))
        ]
        return _FakeResponse({"results": results})

    monkeypatch.setattr(httpx, "post", fake_post)
    documents = _online_documents(5)

    scores = scorer.score_documents("question", documents)

    assert scorer.last_status == "success"
    assert scorer.last_scored_count == 5
    assert len(requests) == 3  # 5 docs / batch 2 → 2 + 2 + 1
    assert requests[0]["model"] == "BAAI/bge-reranker-v2-m3"
    assert requests[0]["query"] == "question"
    assert requests[0]["top_n"] == 2
    assert list(scores) == [f"dsid_{index}" for index in range(5)]
    assert scores["dsid_0"] > scores["dsid_4"]


def test_cross_encoder_online_timeout_aborts(monkeypatch) -> None:
    scorer = _online_scorer()

    def fake_post(url, headers=None, json=None, timeout=None):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(TimeoutError, match="exceeded"):
        scorer.score_documents("question", _online_documents(2))

    assert scorer.last_status == "inference_timeout"


def test_cross_encoder_online_http_error_aborts(monkeypatch) -> None:
    scorer = _online_scorer()

    def fake_post(url, headers=None, json=None, timeout=None):
        request = httpx.Request("POST", url)
        raise httpx.HTTPStatusError(
            "unauthorized", request=request, response=httpx.Response(401, request=request)
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(RuntimeError, match="unauthorized"):
        scorer.score_documents("question", _online_documents(2))

    assert scorer.last_status == "inference_failed"


def test_cross_encoder_online_requires_config(monkeypatch) -> None:
    scorer = _online_scorer()
    scorer.api_key = ""

    with pytest.raises(RuntimeError, match="requires base_url, api_key and model_id"):
        scorer.score_documents("question", _online_documents(1))

    assert scorer.last_status == "inference_failed"


def test_cross_encoder_online_transient_error_retries_then_succeeds(monkeypatch) -> None:
    scorer = _online_scorer(batch_size=2)
    scorer.max_retries = 3
    scorer.retry_backoff_seconds = 0
    calls = {"count": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["count"] += 1
        if calls["count"] < 3:
            raise httpx.ConnectError("connection dropped")
        return _FakeResponse(
            {
                "results": [
                    {"index": 0, "relevance_score": 1.0},
                    {"index": 1, "relevance_score": 0.5},
                ]
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    scores = scorer.score_documents("question", _online_documents(2))

    assert scorer.last_status == "success"
    assert scorer.last_scored_count == 2
    assert calls["count"] == 3  # 2 次瞬时失败 + 1 次成功
    assert set(scores) == {"dsid_0", "dsid_1"}


def test_cross_encoder_online_transient_error_exhausts_retries(monkeypatch) -> None:
    scorer = _online_scorer(batch_size=2)
    scorer.max_retries = 3
    scorer.retry_backoff_seconds = 0
    calls = {"count": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["count"] += 1
        raise httpx.ConnectError("connection dropped")

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(RuntimeError, match="connection dropped"):
        scorer.score_documents("question", _online_documents(2))

    assert scorer.last_status == "inference_failed"
    assert calls["count"] == 3  # 重试耗尽后终止 benchmark


def test_cross_encoder_online_server_error_retries(monkeypatch) -> None:
    scorer = _online_scorer(batch_size=2)
    scorer.retry_backoff_seconds = 0
    calls = {"count": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["count"] += 1
        request = httpx.Request("POST", url)
        raise httpx.HTTPStatusError(
            "server error", request=request, response=httpx.Response(503, request=request)
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(RuntimeError, match="server error"):
        scorer.score_documents("question", _online_documents(2))

    assert calls["count"] == 3  # 503 属于可重试 5xx，重试耗尽后终止


def test_score_passages_online_passes_texts_raw(monkeypatch) -> None:
    """score_passages 不得对文本做任何截断/拼贴——请求里的 documents 必须原样。"""
    scorer = _online_scorer(batch_size=10)
    requests: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        requests.append(json)
        documents = json["documents"]
        results = [
            {"index": index, "relevance_score": (len(documents) - index) / 100}
            for index in range(len(documents))
        ]
        return _FakeResponse({"results": results})

    monkeypatch.setattr(httpx, "post", fake_post)
    texts = ["long raw chunk text " + ("x" * 3000), "second chunk body"]

    scores = scorer.score_passages("question", texts)

    assert scorer.last_status == "success"
    assert scorer.last_scored_count == 2
    assert len(requests) == 1
    assert requests[0]["query"] == "question"
    assert requests[0]["documents"] == texts  # 原样传入，无 1000 字符截断
    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_score_passages_returns_empty_for_no_texts(monkeypatch) -> None:
    scorer = _online_scorer()

    def fake_post(url, headers=None, json=None, timeout=None):
        raise AssertionError("should not call API for empty texts")

    monkeypatch.setattr(httpx, "post", fake_post)

    assert scorer.score_passages("question", []) == []
    assert scorer.last_status == "success"
    assert scorer.last_scored_count == 0


def test_score_passages_local_aligns_scores() -> None:
    scorer = CrossEncoderScorer(model_path="unused", device="cuda")
    scorer._model = _FakeCrossEncoderModel(score=0.5)

    scores = scorer.score_passages("question", ["a", "b", "c"])

    assert scores == [0.5, 0.5, 0.5]
    assert scorer.last_scored_count == 3


def test_score_chunks_pairs_page_content(monkeypatch) -> None:
    scorer = CrossEncoderScorer(model_path="unused", device="cuda")
    calls = {"pairs": None}

    class RecordingModel:
        def predict(self, pairs, batch_size=32):
            calls["pairs"] = pairs
            return [1.0 - index / 100 for index in range(len(pairs))]

    scorer._model = RecordingModel()
    chunks = [
        Document(page_content=f"chunk body {index}", metadata={"chunk_id": f"d::{index}"})
        for index in range(3)
    ]

    scores = scorer.score_chunks("question", chunks)

    assert calls["pairs"] == [["question", f"chunk body {index}"] for index in range(3)]
    assert len(scores) == 3
    assert scores[0] > scores[2]
