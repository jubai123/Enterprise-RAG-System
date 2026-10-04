from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings

from src.config import EmbeddingConfig


def build_embeddings(config: EmbeddingConfig) -> Embeddings:
    provider = config.provider
    model = config.model

    if provider in {"openai", "openai_compatible", "siliconflow"}:
        # SiliconFlow 的 embedding API 兼容 OpenAI 格式，因此复用 LangChain 的 OpenAIEmbeddings。
        from langchain_openai import OpenAIEmbeddings

        api_key = config.api_key or None
        base_url = config.base_url or None
        if not api_key:
            raise ValueError("Missing embedding api_key. Please set SILICONFLOW_API_KEY in .env.")

        # 请求级超时:实测 SiliconFlow 偶发连接静默挂起,openai 客户端默认 600s 超时
        # 会让一个坏请求拖死整轮嵌入(曾出现 >1h 无任何 API 流量)。给到 ~90s 使挂起
        # 快速浮出并经 max_retries 重试;EMBED_REQUEST_TIMEOUT 可在外部按需调。
        timeout = float(os.environ.get("EMBED_REQUEST_TIMEOUT", "90"))
        return OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            # OpenAI-compatible 服务应让服务端使用目标模型自己的 tokenizer。
            check_embedding_ctx_length=False,
            # SiliconFlow 偶发 SSL 瞬时断连，提高客户端重试次数
            max_retries=6,
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")
