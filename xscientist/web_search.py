"""Auditable, provider-neutral web discovery for XScientist.

Search results are discovery candidates.  They are deliberately kept separate
from source snapshots and passage evidence until a researcher selects and
freezes a source.  Providers return JSON metadata only; raw HTTP bodies never
leave this module.
"""

from __future__ import annotations

import json
import ipaddress
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from ai_scientist.protocol.canonical_json import canonical_content_hash

MAX_QUERY_CHARS = 1_000
MAX_LIMIT = 100
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 20.0
WEB_SEARCH_PROTOCOL = "xscientist.web-search.v1"


class WebSearchError(RuntimeError):
    """A provider could not execute a bounded search request."""


@dataclass(frozen=True)
class ProviderResponse:
    """Provider output before it is normalized into a research receipt."""

    candidates: tuple[Mapping[str, Any], ...] = ()
    complete: bool = True
    pagination: Mapping[str, Any] = field(default_factory=dict)
    errors: tuple[str, ...] = ()


class SearchProvider(Protocol):
    name: str
    version: str

    def search(
        self,
        query: str,
        *,
        vertical: str,
        limit: int,
        timeout: float,
    ) -> ProviderResponse: ...


def _text(value: Any, *, maximum: int = 1_000) -> str:
    return " ".join(str(value or "").split())[:maximum]


def _hash(value: Any) -> str:
    return canonical_content_hash(str(value or ""))


def canonicalize_url(value: Any) -> str:
    """Canonicalize a result URL while dropping common tracking parameters."""

    raw = _text(value, maximum=4_096)
    parsed = urlsplit(raw)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        return ""
    try:
        port = parsed.port
    except ValueError:
        return ""
    host = parsed.hostname.casefold().rstrip(".")
    if not host:
        return ""
    netloc = host
    if ":" in host and not host.startswith("["):
        netloc = f"[{host}]"
    if port and not (
        (parsed.scheme.casefold() == "https" and port == 443)
        or (parsed.scheme.casefold() == "http" and port == 80)
    ):
        netloc = f"{netloc}:{port}"
    ignored = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src"}
    query = [
        (key, val)
        for key, val in parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() not in ignored and not key.casefold().startswith("utm_")
    ]
    query.sort()
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.casefold(), netloc, path, urlencode(query), ""))


def _domain(url: str) -> str:
    return (urlsplit(url).hostname or "").casefold()


def _video_hint(url: str, title: str, snippet: str, vertical: str) -> bool:
    haystack = f"{url} {title} {snippet}".casefold()
    return vertical == "video" or any(
        marker in haystack
        for marker in ("youtube.com", "youtu.be", "bilibili.com", "vimeo.com", "video")
    )


def normalize_candidate(
    candidate: Mapping[str, Any],
    *,
    provider: str,
    rank: int,
    vertical: str,
) -> dict[str, Any] | None:
    """Convert provider-specific metadata to a bounded, hashable candidate."""

    url = canonicalize_url(candidate.get("url") or candidate.get("link"))
    title = _text(candidate.get("title"), maximum=500)
    snippet = _text(
        candidate.get("snippet") or candidate.get("description"), maximum=1_000
    )
    if not url or not title:
        return None
    published_at = _text(
        candidate.get("published_at")
        or candidate.get("publishedAt")
        or candidate.get("date"),
        maximum=80,
    )
    provider_result_id = _text(
        candidate.get("provider_result_id")
        or candidate.get("result_id")
        or candidate.get("id"),
        maximum=300,
    )
    media_type = _text(candidate.get("media_type") or candidate.get("type"), maximum=40)
    row: dict[str, Any] = {
        "rank": int(rank),
        "provider": _text(provider, maximum=120),
        "title": title,
        "url": url,
        "url_hash": _hash(url),
        "source_domain": _domain(url),
        "snippet": snippet,
        "snippet_hash": _hash(snippet),
        "media_type": media_type
        or ("video" if _video_hint(url, title, snippet, vertical) else "webpage"),
        "video_hint": _video_hint(url, title, snippet, vertical),
    }
    if published_at:
        row["published_at"] = published_at
    if provider_result_id:
        row["provider_result_id_hash"] = _hash(provider_result_id)
    return row


def merge_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Deduplicate by canonical URL and assign stable aggregate ranks."""

    by_url: dict[str, dict[str, Any]] = {}
    for row in rows:
        url = canonicalize_url(row.get("url"))
        if not url:
            continue
        existing = by_url.get(url)
        candidate = dict(row)
        candidate["url"] = url
        if existing is None:
            by_url[url] = candidate
            continue
        providers = list(existing.get("providers") or [existing.get("provider")])
        provider = candidate.get("provider")
        if provider and provider not in providers:
            providers.append(provider)
        existing["providers"] = providers
        existing["provider_count"] = len(providers)
        existing["provider_rank"] = min(
            int(existing.get("provider_rank") or existing.get("rank") or 1),
            int(candidate.get("rank") or 1),
        )
        if len(str(candidate.get("snippet") or "")) > len(
            str(existing.get("snippet") or "")
        ):
            existing["snippet"] = candidate["snippet"]
            existing["snippet_hash"] = candidate.get("snippet_hash") or _hash(
                candidate["snippet"]
            )
    merged = list(by_url.values())
    merged.sort(
        key=lambda item: (
            -int(item.get("provider_count") or 1),
            int(item.get("provider_rank") or item.get("rank") or 1),
            str(item.get("source_domain") or ""),
            str(item.get("url") or ""),
        )
    )
    for index, item in enumerate(merged[:limit], start=1):
        item["rank"] = index
        item.setdefault("provider_count", 1)
        item.setdefault("providers", [item.get("provider")])
    return merged[:limit]


def _json_request(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, Any] | None = None,
    timeout: float,
) -> Mapping[str, Any]:
    parsed = urlsplit(url)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise WebSearchError("search endpoint must be HTTPS or loopback HTTP")
    if parsed.scheme.casefold() == "http":
        try:
            is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            is_loopback = parsed.hostname.casefold() == "localhost"
        if not is_loopback:
            raise WebSearchError("HTTP search endpoints are allowed only on loopback")
    query = [
        (str(key), str(value))
        for key, value in (params or {}).items()
        if value not in (None, "")
    ]
    request_url = urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), "")
    )
    request = Request(
        request_url,
        headers={str(k): str(v) for k, v in (headers or {}).items()},
        method="GET",
    )
    try:
        with urlopen(request, timeout=min(max(float(timeout), 1.0), 60.0)) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise WebSearchError(f"search provider returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise WebSearchError("search provider request failed") from exc
    if len(body) > MAX_RESPONSE_BYTES:
        raise WebSearchError("search provider response exceeded the size limit")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WebSearchError("search provider returned invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise WebSearchError("search provider response must be a JSON object")
    return payload


class BraveSearchProvider:
    name = "brave"
    version = "api-v1"

    def __init__(
        self, *, api_key: str | None = None, endpoint: str | None = None
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("BRAVE_SEARCH_API_KEY")
            or os.getenv("XSCIENTIST_BRAVE_SEARCH_API_KEY")
        )
        self.endpoint = (
            endpoint
            or os.getenv("BRAVE_SEARCH_ENDPOINT")
            or "https://api.search.brave.com/res/v1"
        ).rstrip("/")

    def search(
        self, query: str, *, vertical: str, limit: int, timeout: float
    ) -> ProviderResponse:
        if not self.api_key:
            raise WebSearchError("provider brave requires BRAVE_SEARCH_API_KEY")
        resource = "videos/search" if vertical == "video" else "web/search"
        payload = _json_request(
            f"{self.endpoint}/{resource}",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
            params={"q": query, "count": limit},
            timeout=timeout,
        )
        raw = payload.get("results") or payload.get("videos") or []
        if not isinstance(raw, list):
            raise WebSearchError("provider brave returned an invalid results array")
        candidates = tuple(
            {
                "title": item.get("title"),
                "url": item.get("url") or item.get("link"),
                "snippet": item.get("description") or item.get("snippet"),
                "published_at": item.get("age") or item.get("published_at"),
                "provider_result_id": item.get("url"),
                "media_type": "video" if vertical == "video" else "webpage",
            }
            for item in raw
            if isinstance(item, Mapping)
        )
        return ProviderResponse(candidates=candidates, pagination={"count": len(raw)})


class YouTubeSearchProvider:
    name = "youtube"
    version = "data-api-v3"

    def __init__(
        self, *, api_key: str | None = None, endpoint: str | None = None
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("YOUTUBE_DATA_API_KEY")
            or os.getenv("XSCIENTIST_YOUTUBE_DATA_API_KEY")
        )
        self.endpoint = endpoint or "https://www.googleapis.com/youtube/v3/search"

    def search(
        self, query: str, *, vertical: str, limit: int, timeout: float
    ) -> ProviderResponse:
        if not self.api_key:
            raise WebSearchError("provider youtube requires YOUTUBE_DATA_API_KEY")
        payload = _json_request(
            self.endpoint,
            params={
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": limit,
                "key": self.api_key,
            },
            timeout=timeout,
        )
        raw = payload.get("items") or []
        if not isinstance(raw, list):
            raise WebSearchError("provider youtube returned an invalid items array")
        candidates = []
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            identifier = item.get("id") if isinstance(item.get("id"), Mapping) else {}
            video_id = identifier.get("videoId")
            snippet = (
                item.get("snippet") if isinstance(item.get("snippet"), Mapping) else {}
            )
            if video_id:
                candidates.append(
                    {
                        "title": snippet.get("title"),
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "snippet": snippet.get("description"),
                        "published_at": snippet.get("publishedAt"),
                        "provider_result_id": video_id,
                        "media_type": "video",
                    }
                )
        return ProviderResponse(
            candidates=tuple(candidates),
            pagination={"next_page_token": payload.get("nextPageToken", "")},
        )


class SearxngSearchProvider:
    """Use a self-hosted SearXNG instance for credential-free web discovery."""

    name = "searxng"
    version = "json-v1"

    def __init__(self, *, endpoint: str | None = None) -> None:
        self.endpoint = (
            endpoint or os.getenv("SEARXNG_URL") or os.getenv("XSCIENTIST_SEARXNG_URL")
        )

    def search(
        self, query: str, *, vertical: str, limit: int, timeout: float
    ) -> ProviderResponse:
        if not self.endpoint:
            raise WebSearchError("provider searxng requires SEARXNG_URL")
        endpoint = self.endpoint.rstrip("/") + "/search"
        payload = _json_request(
            endpoint,
            params={
                "q": query,
                "format": "json",
                "categories": "videos" if vertical == "video" else "general",
                "pageno": 1,
            },
            timeout=timeout,
        )
        raw = payload.get("results") or []
        if not isinstance(raw, list):
            raise WebSearchError("provider searxng returned an invalid results array")
        candidates = tuple(
            {
                "title": item.get("title"),
                "url": item.get("url") or item.get("link"),
                "snippet": item.get("content") or item.get("snippet"),
                "published_at": item.get("publishedDate"),
                "provider_result_id": item.get("url"),
                "media_type": "video" if vertical == "video" else "webpage",
            }
            for item in raw[:limit]
            if isinstance(item, Mapping)
        )
        return ProviderResponse(
            candidates=candidates,
            pagination={
                "number_of_results": payload.get("number_of_results", len(raw))
            },
        )


class SemanticScholarProvider:
    name = "semantic_scholar"
    version = "graph-api-v1"

    def __init__(self, *, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("S2_API_KEY")

    def search(
        self, query: str, *, vertical: str, limit: int, timeout: float
    ) -> ProviderResponse:
        fields = (
            "paperId,title,abstract,year,url,publicationDate,externalIds,openAccessPdf"
        )
        headers = {"X-API-KEY": self.api_key} if self.api_key else {}
        payload = _json_request(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params={"query": query, "limit": limit, "fields": fields},
            timeout=timeout,
        )
        raw = payload.get("data") or []
        if not isinstance(raw, list):
            raise WebSearchError(
                "provider semantic_scholar returned an invalid data array"
            )
        candidates = tuple(
            {
                "title": item.get("title"),
                "url": item.get("url") or (item.get("openAccessPdf") or {}).get("url"),
                "snippet": item.get("abstract"),
                "published_at": item.get("publicationDate") or item.get("year"),
                "provider_result_id": item.get("paperId"),
                "media_type": "paper",
            }
            for item in raw
            if isinstance(item, Mapping)
        )
        return ProviderResponse(
            candidates=candidates, pagination={"total": payload.get("total", len(raw))}
        )


def resolve_provider_names(
    providers: Sequence[str],
    *,
    vertical: str,
    extra_providers: Sequence[str] = (),
) -> list[str]:
    """Resolve ``auto`` deterministically from configured provider credentials."""

    requested = [
        str(item).strip().casefold() for item in providers if str(item).strip()
    ]
    if not requested:
        requested = ["auto"]
    allowed = {"brave", "youtube", "searxng", "semantic_scholar", *extra_providers}
    resolved: list[str] = []
    for name in requested:
        if name != "auto":
            if name not in allowed:
                raise WebSearchError(f"unknown web search provider: {name}")
            chosen = name
        elif vertical == "video" and (
            os.getenv("YOUTUBE_DATA_API_KEY")
            or os.getenv("XSCIENTIST_YOUTUBE_DATA_API_KEY")
        ):
            chosen = "youtube"
        elif os.getenv("SEARXNG_URL") or os.getenv("XSCIENTIST_SEARXNG_URL"):
            chosen = "searxng"
        elif os.getenv("BRAVE_SEARCH_API_KEY") or os.getenv(
            "XSCIENTIST_BRAVE_SEARCH_API_KEY"
        ):
            chosen = "brave"
        elif vertical == "video":
            chosen = "youtube"
        else:
            chosen = "semantic_scholar"
        if chosen not in resolved:
            resolved.append(chosen)
    return resolved


def provider_for(name: str) -> SearchProvider:
    providers: dict[str, Callable[[], SearchProvider]] = {
        "brave": BraveSearchProvider,
        "youtube": YouTubeSearchProvider,
        "searxng": SearxngSearchProvider,
        "semantic_scholar": SemanticScholarProvider,
    }
    try:
        return providers[name]()
    except KeyError as exc:
        raise WebSearchError(f"unknown web search provider: {name}") from exc


def search_web(
    query: str,
    *,
    providers: Sequence[str] = ("auto",),
    vertical: str = "web",
    limit: int = 10,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    provider_registry: Mapping[str, SearchProvider] | None = None,
) -> dict[str, Any]:
    """Execute bounded provider searches and return receipt-ready metadata."""

    normalized_query = _text(query, maximum=MAX_QUERY_CHARS)
    if not normalized_query:
        raise WebSearchError("web search query cannot be empty")
    if vertical not in {"web", "video"}:
        raise WebSearchError("web search vertical must be web or video")
    if limit < 1 or limit > MAX_LIMIT:
        raise WebSearchError(f"web search limit must be between 1 and {MAX_LIMIT}")
    resolved = resolve_provider_names(
        providers,
        vertical=vertical,
        extra_providers=tuple((provider_registry or {}).keys()),
    )
    registry = dict(provider_registry or {})
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    pagination: dict[str, Any] = {"providers": {}}
    provider_versions: dict[str, str] = {}
    complete = True
    for provider_name in resolved:
        provider = registry.get(provider_name) or provider_for(provider_name)
        provider_versions[provider_name] = (
            _text(getattr(provider, "version", "unknown"), maximum=120) or "unknown"
        )
        try:
            response = provider.search(
                normalized_query,
                vertical=vertical,
                limit=limit,
                timeout=timeout,
            )
        except WebSearchError as exc:
            errors.append(f"{provider_name}: {str(exc)[:300]}")
            complete = False
            continue
        except Exception:
            errors.append(f"{provider_name}: provider failed")
            complete = False
            continue
        provider_rows: list[dict[str, Any]] = []
        for rank, candidate in enumerate(response.candidates, start=1):
            normalized = normalize_candidate(
                candidate,
                provider=provider_name,
                rank=rank,
                vertical=vertical,
            )
            if normalized is not None:
                provider_rows.append(normalized)
        rows.extend(provider_rows)
        pagination["providers"][provider_name] = dict(response.pagination)
        if response.errors:
            errors.extend(
                f"{provider_name}: {_text(item, maximum=300)}"
                for item in response.errors
            )
        complete = complete and response.complete and not response.errors
    candidates = merge_candidates(rows, limit=limit)
    pagination["provider_count"] = len(resolved)
    pagination["returned_count"] = len(candidates)
    pagination["complete"] = complete
    return {
        "protocol": WEB_SEARCH_PROTOCOL,
        "query": normalized_query,
        "vertical": vertical,
        "providers": resolved,
        "provider_versions": provider_versions,
        "provider": "+".join(resolved),
        "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "candidates": candidates,
        "pagination": pagination,
        "complete": complete,
        "errors": errors,
        "request_hash": canonical_content_hash(
            {
                "query": normalized_query,
                "vertical": vertical,
                "providers": resolved,
                "limit": limit,
            }
        ),
    }


__all__ = [
    "BraveSearchProvider",
    "ProviderResponse",
    "SearchProvider",
    "SemanticScholarProvider",
    "SearxngSearchProvider",
    "YouTubeSearchProvider",
    "WebSearchError",
    "canonicalize_url",
    "merge_candidates",
    "normalize_candidate",
    "provider_for",
    "resolve_provider_names",
    "search_web",
]
