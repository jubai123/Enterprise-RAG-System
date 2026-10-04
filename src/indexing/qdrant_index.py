from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.config import QdrantConfig


def stable_document_ids(documents: list[Document]) -> list[str]:
    return [
        str(uuid5(NAMESPACE_URL, str(document.metadata["chunk_id"])))
        for document in documents
    ]


def recreate_collection(config: QdrantConfig) -> None:
    client = QdrantClient(url=config.url, api_key=config.api_key or None)
    distance = Distance[config.distance.upper()]
    collection_name = config.collection
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=config.vector_size,
            distance=distance,
        ),
    )


def index_documents(
    documents: list[Document],
    embeddings: Embeddings,
    qdrant_config: QdrantConfig,
    batch_size: int = 64,
) -> None:
    # 这里直接使用 LangChain 的 QdrantVectorStore，保证后续 retriever 使用同一套 metadata。
    # wait=true upsert 会同步等 HNSW 建索引;大 batch(512)偶发超过客户端默认读超时,
    # 造成 ResponseHandlingException: timed out,给足 300s。
    client = QdrantClient(
        url=qdrant_config.url,
        api_key=qdrant_config.api_key or None,
        timeout=300,
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=qdrant_config.collection,
        embedding=embeddings,
    )
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        # batch_size 必须继续透传给 add_texts:LangChain 内部默认 64 的嵌入批会把
        # CLI 的 512 静默切成 64/请求,请求开销主导把全量重建拖到 ~6h(实测 11 pts/s)。
        vector_store.add_documents(
            batch, ids=stable_document_ids(batch), batch_size=batch_size
        )
