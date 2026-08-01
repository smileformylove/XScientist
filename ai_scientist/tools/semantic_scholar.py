from __future__ import annotations

import os
import math
import time
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.utils.optional_dependencies import (
    import_backoff,
    import_optional_module,
    resolve_exception_types,
)

backoff = import_backoff()
requests = import_optional_module(
    "requests",
    install_hint="Install the 'requests' package to use Semantic Scholar search.",
)
try:
    _REQUEST_EXCEPTIONS_ROOT = getattr(requests, "exceptions")
except ModuleNotFoundError:
    _REQUEST_EXCEPTIONS_ROOT = object()
_SEMANTIC_SCHOLAR_RETRY_EXCEPTIONS = resolve_exception_types(
    _REQUEST_EXCEPTIONS_ROOT,
    ("HTTPError", "ConnectionError", "Timeout"),
)
_SEMANTIC_SCHOLAR_HTTP_ERROR = resolve_exception_types(
    _REQUEST_EXCEPTIONS_ROOT,
    ("HTTPError",),
)


def on_backoff(details: Dict) -> None:
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


def balanced_rank_papers(
    papers: List[Dict],
    *,
    limit: int | None = None,
    current_year: int | None = None,
) -> List[Dict]:
    """Balance API relevance, recency, and citations for discovery.

    Citation-only ordering suppresses emerging work. Semantic Scholar already
    returns a relevance ordering, so keep that as the strongest signal and use
    bounded recency/citation terms as secondary evidence.
    """

    year_now = current_year or datetime.now(timezone.utc).year
    ranked: List[Dict] = []
    for index, paper in enumerate(papers):
        row = dict(paper)
        try:
            year = int(row.get("year") or year_now)
        except (TypeError, ValueError):
            year = year_now
        try:
            citations = max(int(row.get("citationCount") or 0), 0)
        except (TypeError, ValueError):
            citations = 0
        age = max(year_now - year, 0)
        relevance = 1.0 / (1.0 + index)
        recency = 1.0 / (1.0 + 0.35 * age)
        citation_signal = min(math.log1p(citations) / math.log1p(1000), 1.0)
        row["discovery_score"] = round(
            0.55 * relevance + 0.30 * recency + 0.15 * citation_signal,
            6,
        )
        ranked.append(row)
    ranked.sort(
        key=lambda item: (
            float(item.get("discovery_score") or 0.0),
            int(item.get("year") or 0),
            int(item.get("citationCount") or 0),
        ),
        reverse=True,
    )
    return ranked[:limit] if limit is not None else ranked


class SemanticScholarSearchTool(BaseTool):
    def __init__(
        self,
        name: str = "SearchSemanticScholar",
        description: str = (
            "Search for relevant literature using Semantic Scholar. "
            "Provide a search query to find relevant papers."
        ),
        max_results: int = 10,
    ):
        parameters = [
            {
                "name": "query",
                "type": "str",
                "description": "The search query to find relevant papers.",
            }
        ]
        super().__init__(name, description, parameters)
        self.max_results = max_results
        self.S2_API_KEY = os.getenv("S2_API_KEY")
        if not self.S2_API_KEY:
            warnings.warn(
                "No Semantic Scholar API key found. Requests will be subject to stricter rate limits. "
                "Set the S2_API_KEY environment variable for higher limits."
            )

    def use_tool(self, query: str) -> Optional[str]:
        papers = self.search_for_papers(query)
        if papers:
            return self.format_papers(papers)
        else:
            return "No papers found."

    @backoff.on_exception(
        backoff.expo,
        _SEMANTIC_SCHOLAR_RETRY_EXCEPTIONS,
        on_backoff=on_backoff,
    )
    def search_for_papers(self, query: str) -> Optional[List[Dict]]:
        if not query:
            return None

        headers = {}
        if self.S2_API_KEY:
            headers["X-API-KEY"] = self.S2_API_KEY

        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params={
                "query": query,
                "limit": self.max_results,
                "fields": (
                    "paperId,title,authors,venue,year,publicationDate,abstract,"
                    "citationCount,url,externalIds,openAccessPdf"
                ),
            },
            timeout=30,
        )
        rsp.raise_for_status()
        results = rsp.json()
        total = results.get("total", 0)
        if total == 0:
            return None

        papers = results.get("data", [])
        return balanced_rank_papers(papers, limit=self.max_results)

    def format_papers(self, papers: List[Dict]) -> str:
        paper_strings = []
        for i, paper in enumerate(papers):
            authors = ", ".join(
                [author.get("name", "Unknown") for author in paper.get("authors", [])]
            )
            paper_strings.append(
                f"""{i + 1}: {paper.get("title", "Unknown Title")}. {authors}. {paper.get("venue", "Unknown Venue")}, {paper.get("year", "Unknown Year")}.
Number of citations: {paper.get("citationCount", "N/A")}
Paper ID: {paper.get("paperId", "N/A")}
URL: {paper.get("url", "N/A")}
Abstract: {paper.get("abstract", "No abstract available.")}"""
            )
        return "\n\n".join(paper_strings)


@backoff.on_exception(
    backoff.expo,
    _SEMANTIC_SCHOLAR_HTTP_ERROR,
    on_backoff=on_backoff,
    max_tries=8,
)
def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    S2_API_KEY = os.getenv("S2_API_KEY")
    headers = {}
    if not S2_API_KEY:
        warnings.warn(
            "No Semantic Scholar API key found. Requests will be subject to stricter rate limits."
        )
    else:
        headers["X-API-KEY"] = S2_API_KEY

    if not query:
        return None

    rsp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers=headers,
        params={
            "query": query,
            "limit": result_limit,
            "fields": (
                "paperId,title,authors,venue,year,publicationDate,abstract,"
                "citationStyles,citationCount,url,externalIds,openAccessPdf"
            ),
        },
        timeout=30,
    )
    rsp.raise_for_status()
    results = rsp.json()
    total = results["total"]
    if not total:
        return None

    papers = results["data"]
    return balanced_rank_papers(papers, limit=result_limit)
