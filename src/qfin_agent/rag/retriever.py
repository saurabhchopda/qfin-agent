"""Vector store retrieval utilities."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


class NewsRetriever:
    """FAISS-backed retriever for fundamental news context."""

    def __init__(self, vectorstore: FAISS):
        self._vectorstore = vectorstore

    @classmethod
    def from_documents(cls, documents: list[Document], embeddings: OpenAIEmbeddings) -> "NewsRetriever":
        """Create a retriever from in-memory documents."""
        vectorstore = FAISS.from_documents(documents, embeddings)
        return cls(vectorstore=vectorstore)

    @classmethod
    def load_local(cls, index_path: str, embeddings: OpenAIEmbeddings) -> "NewsRetriever":
        """Load a retriever from local disk."""
        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return cls(vectorstore=vectorstore)

    def save_local(self, index_path: str) -> None:
        """Persist retriever index to local disk."""
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(index_path)

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve top-k documents by semantic similarity."""
        return await self._vectorstore.asimilarity_search(query, k=top_k)
