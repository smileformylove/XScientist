# Auditable Web Search

XScientist can discover web and video sources through the native research
protocol:

```bash
xscientist research web search "video evidence for intervention X" \
  --vertical video --provider auto --limit 20 --repo ./study
```

The command locks a `search_plan`, executes the configured provider(s), and
records a `search_receipt`. The receipt contains canonical URLs, URL and
snippet hashes, source domains, media/video hints, provider ranks, pagination,
retrieval time, and any provider failures. Search candidates remain pending
until a researcher selects one and records a `literature source` or `video add`
operation. A search result is therefore a discovery lead, not scientific
evidence by itself.

`auto` uses YouTube for the `video` vertical when `YOUTUBE_DATA_API_KEY` is
configured, a configured SearXNG instance for credential-free general web
search, Brave Search when `BRAVE_SEARCH_API_KEY` is configured, and Semantic
Scholar as the scholarly fallback for the `web` vertical. Providers can be
merged explicitly:

```bash
xscientist research web search "benchmark walkthrough" \
  --vertical video --provider brave --provider youtube \
  --repo ./study
```

Provider credentials are read from the environment and are never written to a
Research VCS object. The built-in clients use bounded HTTPS JSON requests. Raw
HTTP bodies, request headers, and credentials are discarded after parsing; only
bounded candidate metadata is retained. Use `--dry-run` to lock and inspect a
plan without making a network request.

Provider configuration:

- `brave`: `BRAVE_SEARCH_API_KEY` (optional `BRAVE_SEARCH_ENDPOINT` for a
  compatible HTTPS endpoint).
- `youtube`: `YOUTUBE_DATA_API_KEY`.
- `searxng`: `SEARXNG_URL` (an HTTPS endpoint, or a loopback HTTP endpoint for
  a local instance).
- `semantic_scholar`: `S2_API_KEY` is optional and raises the provider quota.

After selecting a candidate, freeze its content before using it in a claim:

```bash
xscientist research literature source RECEIPT_ID "Title" \
  --url "https://example.org/video" --content-hash sha256:... \
  --repo ./study
```

For video work, discovery does not download media automatically. Import the
selected URL through the existing video asset workflow, then index transcript,
OCR, captions, shots, and timestamped evidence so later claims point to a
stable asset and interval.
