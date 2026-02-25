"""News ingestion and chunking for RAG."""

from __future__ import annotations

import asyncio
from datetime import datetime
from email.utils import parsedate_to_datetime

import feedparser
import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from qfin_agent.config import Settings


async def ingest_news_documents(
    ticker: str,
    settings: Settings,
    max_articles: int = 25,
) -> list[Document]:
    """Fetch and chunk news documents from RSS and optional Finnhub.

    Args:
        ticker: Equity ticker symbol.
        settings: Runtime settings.
        max_articles: Maximum number of top-level articles before chunking.

    Returns:
        Chunked documents suitable for vector indexing.
    """

    rss_docs = await _fetch_rss_news(ticker=ticker, feeds=settings.parsed_rss_feeds)
    finnhub_docs = await _fetch_finnhub_news(ticker=ticker, api_key=settings.finnhub_api_key)

    merged = (rss_docs + finnhub_docs)[:max_articles]
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    return splitter.split_documents(merged)


async def _fetch_rss_news(ticker: str, feeds: list[str]) -> list[Document]:
    docs: list[Document] = []

    async def parse_feed(url: str) -> list[Document]:
        formatted_url = url.format(ticker=ticker)
        parsed = await asyncio.to_thread(feedparser.parse, formatted_url)
        parsed_docs: list[Document] = []
        for entry in parsed.entries[:10]:
            published_at = _parse_published(entry.get("published"))
            summary = entry.get("summary", "")
            title = entry.get("title", "Untitled")
            link = entry.get("link", "")
            parsed_docs.append(
                Document(
                    page_content=f"{title}\n\n{summary}",
                    metadata={
                        "source": link,
                        "provider": "rss",
                        "title": title,
                        "published_at": published_at.isoformat() if published_at else None,
                        "ticker": ticker,
                    },
                )
            )
        return parsed_docs

    results = await asyncio.gather(*(parse_feed(feed) for feed in feeds), return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            continue
        docs.extend(result)
    return docs


async def _fetch_finnhub_news(ticker: str, api_key: str | None) -> list[Document]:
    if not api_key:
        return []

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": datetime.utcnow().date().isoformat(),
        "to": datetime.utcnow().date().isoformat(),
        "token": api_key,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    docs: list[Document] = []
    for item in data[:10]:
        title = item.get("headline", "Untitled")
        summary = item.get("summary", "")
        source = item.get("url", "")
        published_at = datetime.utcfromtimestamp(item["datetime"]) if item.get("datetime") else None
        docs.append(
            Document(
                page_content=f"{title}\n\n{summary}",
                metadata={
                    "source": source,
                    "provider": "finnhub",
                    "title": title,
                    "published_at": published_at.isoformat() if published_at else None,
                    "ticker": ticker,
                },
            )
        )
    return docs


def _parse_published(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
