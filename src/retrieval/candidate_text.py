from __future__ import annotations

import re

from langchain_core.documents import Document

from src.retrieval.lexical_retriever import tokenize_technical_text

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for",
    "from", "how", "in", "is", "it", "new", "of", "on", "or", "the",
    "to", "used", "what", "when", "which", "with",
}

EXACT_IDENTIFIER_PATTERN = re.compile(
    r"--[A-Za-z0-9][A-Za-z0-9_-]*|"
    r"[A-Za-z][A-Za-z0-9_-]*(?:[._/:][A-Za-z0-9_-]+)+|"
    r"\b[A-Z]{2,}[A-Z0-9-]*\b"
)
# 路径/配置键形标识符需含代码特征（数字/下划线/点），或多级路径（≥2 斜杠）。
# 全字母 + 单斜杠的短语（"location/documentation"、"and/or"）不是标识符——
# 若把它当作 identifier，code-first judge 会要求证据含字面串，导致误判 INSUFFICIENT。
# 连字符不算代码特征：地理名/普通短语里的连字符（"Europe/Asia-Pacific"、
# "chopped/out-of-order"）会骗过 _is_code_path_like，被当成必需标识符误杀。
_PATH_LIKE_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*(?:[._/:][A-Za-z0-9_-]+)+")


def _is_code_path_like(token: str) -> bool:
    return any(char.isdigit() or char in "_." for char in token) or token.count("/") >= 2

# 通用缩写不是精确标识符：HTTP/API/JSON 等出现在几乎所有技术文档里，
# 若把它纳入 identifier 信号，coverage 会误判（http_status 问题的 gold 文档
# 常不含字面 "HTTP"）。真正的标识符是 flags、路径、配置键这类专名。
# 地理码（EU/US/APAC）同样不是代码标识符：qst_0198 的 gold 文档只写小写
# "eu-west"，code-first judge 要求字面 "EU" 会误杀。注意保留 TP 这类 2 字母
# 简写——preserve_query_identifiers 靠它保真 follow-up 检索。
_GENERIC_ACRONYMS = frozenset({
    "API", "APAC", "CDN", "CLI", "CPU", "CRD", "CSS", "CSV", "DB", "DNS",
    "EMEA", "EU", "GPU", "HTML", "HTTP", "HTTPS", "ID", "IP", "JSON", "KMS",
    "KPI", "LLM", "OAuth", "PDF", "PII", "RAM", "RBAC", "RCA", "REST", "SDK",
    "SIEM", "SLA", "SLO", "SQL", "SSE", "SSH", "SSL", "SSO", "TCP", "TLS",
    "UDP", "UI", "UK", "URL", "US", "UX", "VPC", "XML", "YAML",
})


def extract_exact_query_identifiers(text: str) -> list[str]:
    identifiers = re.findall(r"`([^`]+)`", text)
    # 单个组合正则按原文位置序返回，保持 preserve_query_identifiers 的追加顺序。
    for match in EXACT_IDENTIFIER_PATTERN.findall(text):
        if _PATH_LIKE_PATTERN.fullmatch(match) and not _is_code_path_like(match):
            continue
        identifiers.append(match)
    return list(
        dict.fromkeys(
            identifier
            for identifier in identifiers
            if identifier and identifier.upper() not in _GENERIC_ACRONYMS
        )
    )


def required_evidence_signals(
    question: str,
) -> dict[str, re.Pattern[str] | list[str]]:
    lowered = question.lower()
    signals: dict[str, re.Pattern[str] | list[str]] = {}
    if "http status" in lowered or "status code" in lowered:
        signals["http_status"] = re.compile(
            r"(?:http(?:\s+status)?|status(?:\s+code)?|returns?|response)"
            r"\D{0,30}\b[1-5]\d{2}\b|"
            r"\b[1-5]\d{2}\b\D{0,30}(?:http|status|error)",
            flags=re.IGNORECASE,
        )
    asks_for_time_value = any(
        phrase in lowered
        for phrase in (
            "wait time",
            "waiting time",
            "timeout",
            "duration",
            "how long",
        )
    )
    if asks_for_time_value:
        signals["time_value"] = re.compile(
            r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds?|seconds?|minutes?)\b",
            flags=re.IGNORECASE,
        )
    if "size limit" in lowered or "size limits" in lowered:
        signals["size_value"] = re.compile(
            r"\b\d+(?:\.\d+)?\s*(?:b|kb|kib|mb|mib|gb|gib)\b",
            flags=re.IGNORECASE,
        )
    asks_for_three_items = bool(
        re.search(
            r"\b(?:three|3)\b.{0,50}\b"
            r"(?:modes?|settings?)\b",
            lowered,
        )
    )
    if asks_for_three_items:
        signals["three_item_enumeration"] = re.compile(
            r"\b(?:three|3)\b.{0,80}\b"
            r"(?:modes?|settings?)\b"
            r".{0,30}[:\-]",
            flags=re.IGNORECASE | re.DOTALL,
        )
    identifiers = extract_exact_query_identifiers(question)
    if identifiers:
        signals["identifier"] = identifiers
    return signals



def evidence_signal_matches(
    name: str,
    pattern: re.Pattern[str] | list[str],
    question: str,
    text: str,
) -> bool:
    if isinstance(pattern, list):
        # identifier 信号：全部精确标识符都出现才算命中，防止按主题相似误判。
        return all(identifier in text for identifier in pattern)
    matches = list(pattern.finditer(text))
    if name != "three_item_enumeration" or not matches:
        return bool(matches)

    topic_match = re.search(
        r"\b(?:three|3)\b(.{0,50}?)\b(?:modes?|settings?)\b",
        question.lower(),
    )
    topic_terms = {
        term
        for term in tokenize_technical_text(topic_match.group(1) if topic_match else "")
        if term not in STOPWORDS
        and term not in {"runtime", "mode", "modes", "setting", "settings"}
        and len(term) > 2
    }
    if not topic_terms:
        return True
    lowered = text.lower()
    for match in matches:
        nearby = lowered[max(0, match.start() - 120) : match.end() + 120]
        if topic_terms & set(tokenize_technical_text(nearby)):
            return True
    return False

def extract_relevant_window(
    content: str,
    question: str,
    requirements: list[str],
    chunk_chars: int,
) -> str:
    """提取 chunk 中与问题最相关的证据片段。

    长 chunk 的答案常常位于中后段，而开头是大量元数据；固定截取开头会让重排器
    看不到关键证据。这里跳过头部元数据，按句计算查询词加权命中，选择得分最高的
    若干句并按原文顺序拼接。
    """
    if len(content) <= chunk_chars:
        return content
    terms = {
        term
        for term in tokenize_technical_text(" ".join([question, *requirements]))
        if term not in STOPWORDS and len(term) > 1
    }
    if not terms:
        return content[:chunk_chars]
    # 跳过 chunk 头部的元数据块（title/section/repo/labels 等）
    body_start = 0
    chunk_marker = content.find("\nChunk: ")
    if chunk_marker >= 0:
        after_header = content.find("\n\n", chunk_marker)
        if after_header >= 0:
            body_start = after_header + 2
    body = content[body_start:]
    term_weights = {term: 2.0 if len(term) >= 6 else 1.0 for term in terms}
    parts = re.split(r"(?<=[.!?])\s+|\n+", body)
    scored: list[tuple[float, int, str, int]] = []
    offset = 0
    for part in parts:
        part_lower = part.lower()
        score = sum(
            weight for term, weight in term_weights.items() if term in part_lower
        )
        if score > 0:
            scored.append((score, len(part), part, offset))
        offset += len(part) + 1
    if not scored:
        return content[:chunk_chars]
    scored.sort(key=lambda item: (-item[0], -item[1]))
    sentence_cap = min(250, max(120, chunk_chars // 3))
    selected: list[tuple[int, str]] = []
    total = 0
    for _score, _length, part, offset in scored:
        if total >= chunk_chars or len(selected) >= 5:
            break
        snippet = part.strip()[:sentence_cap]
        selected.append((offset, snippet))
        total += len(snippet) + 3
    selected.sort(key=lambda item: item[0])
    return " … ".join(snippet for _offset, snippet in selected)

def rank_candidate_chunks(
    question: str,
    requirements: list[str],
    documents: list[Document],
) -> list[Document]:
    query_terms = {
        term
        for term in tokenize_technical_text(
            " ".join([question, *requirements])
        )
        if term not in STOPWORDS and len(term) > 1
    }
    patterns = required_evidence_signals(question)

    def _score(document: Document) -> tuple[int, int]:
        content = document.page_content
        direct_signals = sum(
            evidence_signal_matches(name, pattern, question, content)
            for name, pattern in patterns.items()
        )
        overlap = len(query_terms & set(tokenize_technical_text(content)))
        return direct_signals, overlap

    return sorted(documents, key=_score, reverse=True)
