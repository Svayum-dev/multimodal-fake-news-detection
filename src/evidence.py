"""News evidence retrieval for the multimodal fake-news research prototype.

Uses Google News RSS to discover related coverage, then attempts to extract
readable text from publisher pages. Relevance scores are lexical retrieval
scores only; they are NOT truth scores and do not establish support or
contradiction.
"""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus, urlparse
import re
import time
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
)


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _parse_date(value: str) -> str:
    if not value:
        return "Date unavailable"
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OverflowError):
        return value


def _google_news_results(query: str, limit: int, timeout: int) -> list[dict]:
    url = RSS_URL.format(query=quote_plus(query))
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)
    results = []
    for item in root.findall(".//item")[:limit]:
        title = _clean_text(item.findtext("title", default=""))
        link = _clean_text(item.findtext("link", default=""))
        description = item.findtext("description", default="")
        source_node = item.find("source")
        source = _clean_text(source_node.text if source_node is not None else "")
        pub_date = _parse_date(item.findtext("pubDate", default=""))
        if not title or not link:
            continue

        # RSS description contains HTML; extract plain text safely.
        snippet = _clean_text(BeautifulSoup(description, "html.parser").get_text(" "))
        results.append({
            "title": title,
            "url": link,
            "source": source or (urlparse(link).netloc.replace("www.", "")),
            "published": pub_date,
            "snippet": snippet,
            "article_text": "",
            "relevance": 0.0,
            "retrieval_status": "Search result only",
        })
    return results


def _extract_article_text(url: str, timeout: int = 12) -> tuple[str, str]:
    """Best-effort extraction. Some publishers block bots or require JavaScript."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            allow_redirects=True,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "html" not in content_type:
            return "", "Publisher did not return an HTML page"

        soup = BeautifulSoup(response.text, "html.parser")
        for node in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "aside"]):
            node.decompose()

        # Prefer article/main content; fall back to paragraphs.
        container = soup.find("article") or soup.find("main") or soup
        paragraphs = [
            _clean_text(p.get_text(" ", strip=True))
            for p in container.find_all("p")
        ]
        paragraphs = [p for p in paragraphs if len(p) >= 45]
        article_text = _clean_text(" ".join(paragraphs))[:12000]

        if article_text:
            return article_text, "Article text extracted"
        return "", "No readable article text found"
    except requests.RequestException as exc:
        return "", f"Publisher page unavailable: {type(exc).__name__}"


def _score_relevance(claim: str, results: list[dict]) -> None:
    documents = []
    for item in results:
        text = " ".join([item.get("title", ""), item.get("snippet", ""), item.get("article_text", "")])
        documents.append(text)

    if not documents:
        return

    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
        matrix = vectorizer.fit_transform([claim] + documents)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        for item, score in zip(results, scores):
            item["relevance"] = round(float(score), 3)
    except ValueError:
        for item in results:
            item["relevance"] = 0.0


def search_news_evidence(
    claim: str,
    limit: int = 6,
    extract_articles: bool = True,
    timeout: int = 15,
) -> dict:
    """Retrieve related news coverage for a claim.

    Returns article links, dates, snippets, optional extracted text, and a
    lexical relevance score. Does not label a claim true or false.
    """
    claim = _clean_text(claim)
    if len(claim) < 8:
        raise ValueError("Enter a more descriptive claim (at least 8 characters).")

    results = _google_news_results(claim, limit=max(1, min(limit, 10)), timeout=timeout)

    if extract_articles:
        for index, item in enumerate(results):
            article_text, status = _extract_article_text(item["url"], timeout=timeout)
            item["article_text"] = article_text
            item["retrieval_status"] = status
            # Avoid hammering publisher sites.
            if index < len(results) - 1:
                time.sleep(0.25)

    _score_relevance(claim, results)
    results.sort(key=lambda x: x["relevance"], reverse=True)

    return {
        "query": claim,
        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "provider": "Google News RSS",
        "results": results,
        "notice": (
            "Related coverage is not proof. Similarity ranks topical relevance only; "
            "the system does not automatically determine whether an article supports "
            "or contradicts the claim. Check the original source and full context."
        ),
    }
