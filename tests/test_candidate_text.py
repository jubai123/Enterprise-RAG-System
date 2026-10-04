from __future__ import annotations

from src.retrieval.candidate_text import (
    extract_exact_query_identifiers,
    extract_relevant_window,
)


def test_extract_exact_query_identifiers_drops_natural_slash_phrases() -> None:
    # 单斜杠全字母短语不是代码标识符：plan LLM 把它写进 requirement 后，
    # code-first judge 会要求证据含字面串 → 误判 INSUFFICIENT（qst_0290 类回归）。
    assert extract_exact_query_identifiers(
        "location/documentation of the advisory 0 to 100 score"
    ) == []
    assert extract_exact_query_identifiers("the change affects and/or its callers") == []


def test_extract_exact_query_identifiers_keeps_real_code_identifiers() -> None:
    assert extract_exact_query_identifiers("where is src/foo.py configured") == [
        "src/foo.py"
    ]
    assert extract_exact_query_identifiers("what does config/redis.yaml set") == [
        "config/redis.yaml"
    ]
    assert extract_exact_query_identifiers(
        "kernel_stability_threshold=0.92 default"
    ) == ["kernel_stability_threshold"]
    assert extract_exact_query_identifiers("nested src/foo/bar path") == [
        "src/foo/bar"
    ]
    assert extract_exact_query_identifiers("change --max-batch flag") == ["--max-batch"]


def test_extract_exact_query_identifiers_ignores_geography_and_generic_acronyms() -> None:
    # qst_0198：地理名/通用缩写不是代码标识符。code-first judge 要求证据含字面
    # "EU"/"CDN" 会误杀——gold 文档只写小写 "eu-west"，从不含大写 "EU"。
    assert extract_exact_query_identifiers(
        "In the EU West production setup where token streaming to browsers reconnects "
        "after a short handshake reset at the CDN layer"
    ) == []
    assert extract_exact_query_identifiers("SIEM privacy scan size estimate") == []


def test_extract_exact_query_identifiers_drops_hyphenated_slash_phrases() -> None:
    # qst_0221/qst_0268：带连字符的普通斜杠短语不是代码路径。连字符不能让
    # _is_code_path_like 返回 True，否则证据不含字面串时被 code-first 误杀。
    assert extract_exact_query_identifiers(
        "target turnaround time for new hire setup in Europe/Asia-Pacific"
    ) == []
    assert extract_exact_query_identifiers(
        "incident involving chopped/out-of-order packets"
    ) == []


def test_extract_relevant_window_prefers_mid_chunk_evidence() -> None:
    header = (
        "title: some-change\n"
        "section: description\n"
        "repo: redwood\n"
        "labels: observability, tracing\n"
        "\n"
        "Section: description\n"
        "Chunk: 0\n"
        "\n"
        "description:\n"
    )
    body = (
        "Motivation: this change improves tracing broadly and touches many "
        "observability surfaces across the runtime. "
        "The exact answer: the baseline token cost is computed using a rolling "
        "median of the first 32 tokens for the route/model combo. "
        "Other paragraphs only mention unrelated details and generic observability "
        "notes without the requested computation method."
    )
    content = header + body

    window = extract_relevant_window(
        content,
        question="How is the baseline token cost computed for the metric?",
        requirements=["baseline token cost computation"],
        chunk_chars=300,
    )

    assert "rolling median of the first 32 tokens" in window
    assert "title: some-change" not in window


def test_extract_relevant_window_keeps_short_content() -> None:
    content = "short chunk without a query marker"
    window = extract_relevant_window(
        content,
        question="How is the baseline token cost computed?",
        requirements=[],
        chunk_chars=300,
    )
    assert window == content
