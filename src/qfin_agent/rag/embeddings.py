"""Embedding utilities for the RAG pipeline."""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from qfin_agent.config import Settings


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    """Build OpenAI embeddings client.

    Args:
        settings: Runtime settings.

    Returns:
        OpenAI embeddings client.
    """

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for embeddings.")

    return OpenAIEmbeddings(
        model=settings.embeddings_model,
        api_key=settings.openai_api_key,
    )
