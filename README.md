# WORX AI Search Worker

A Cloudflare Worker that provides semantic search and Q&A over site content using Cloudflare Vectorize (v2) and Workers AI. It’s multi-tenant (per-site) with API key protection, CORS controls, optional caching, robust logging, and several debug/admin utilities.

This README explains how it works, how to configure it, and all the ways to interact with it.


## Architecture overview
- Vector store: Cloudflare Vectorize (binding: `VECTORIZE`) containing document embeddings.
- Inference: Workers AI (binding: `AI`) for both embeddings and chat responses.
- State/config: Cloudflare KV (binding: `CONFIG`) for per-site configuration, document texts, and caches.
- Behavior system: 10 built-in response behaviors selected by custom intent detection.
- Search pipeline: single-pass scored intent detection (embedding + keyword + metadata, no priority list) with tiered confidence + content-floor gating, followed by three-pass re-ranking.
- Worker routes: REST endpoints for status, search, ask, embed, admin cache control, and debugging.


## Bindings and environment
Bindings are defined in `wrangler.jsonc`:
- **VECTORIZE**: Vectorize index (remote) used by the worker
- **AI**: Workers AI for embeddings and LLM inference
- **WORX_AI_CONFIG**: KV namespace for per-site configuration (`cfg:<site>`)
- **WORX_AI_CONTENT**: KV namespace for stored document texts (`doc:<site>:...`) from ingest pipeline
- **CONFIG** (legacy name for WORX_AI_CONFIG): Used for:
  - `ans:<hash>` — short-TTL answer cache for /ask
  - `qemb:<hash>` — short-TTL embedding cache for queries
- **ALLOWED_ORIGINS** (var): comma-separated list of origins allowed by CORS, used when a site config does not specify its own list.

Example (see existing `wrangler.jsonc` in the repo):

```
{
  "vectorize": [ { "binding": "VECTORIZE", "index_name": "worx-ai-index", "remote": true } ],
  "ai": { "binding": "AI" },
  "kv_namespaces": [ { "binding": "CONFIG", "id": "<prod-id>", "preview_id": "<preview-id>" } ],
  "vars": { "ALLOWED_ORIGINS": "http://localhost:5173,http://localhost:3000" }
}
```


## Per-site configuration (KV: cfg:<site>)
Create a JSON object in KV under key `cfg:<site>`. Required fields are marked.

```
{
  "site_key": "worxbranding-dev",
  "vectorize": { "index_name": "worx-ai-index", "dims": 1024, "metric": "cosine" },  // required
  "ai": { "embed_model": "@cf/baai/bge-large-en-v1.5" },                               // required
  "api_key": "dev-secret-123",                                                          // optional, can also set under search.api_key
  "default_behavior": "long_form_answer",    // fallback behavior when no intent matches (default: long_form_answer)
  "custom_intents": [ ... ],                 // array of custom intent objects (see Intent Detection section)
  "search": {
    "allowed_origins": ["http://localhost:5173", "https://your-frontend.app"],
    "api_key": "dev-secret-123",             // API key required on protected endpoints via header x-api-key
    "chat_model": "@cf/meta/llama-3.1-8b-instruct",
    "chat_temperature": 0.1,
    "system_prompt": "You are a helpful assistant...", // optional; worker provides a sensible default with URL restrictions
    "initial_topK": 15,             // results fetched from Vectorize for re-ranking (default 15)
    "final_topK": 3,                // results passed to behavior after re-ranking (default 3)
    "max_output_tokens": 1024,
    // Performance/cost tunables
    "max_context_docs": 6,           // max docs included in /ask prompt context
    "max_kv_text_chars": 3000,       // truncate KV text per doc to this many chars
    "answer_cache_ttl": 2592000,     // seconds for /ask answer cache (default: 30 days; cleared on ingest)
    "embed_cache_ttl": 7776000,      // seconds for query embedding cache (default: 90 days)
    "caching": true                  // default caching behavior (true/false); can be overridden per-request
  }
}
```

Notes:
- The worker must load cfg by `?site=<site>` in the URL on protected endpoints.
- For /ask specifically, the JSON body must also include `site` and `q`.


## CORS and authentication
- CORS: The worker checks the request Origin against `search.allowed_origins` for the site. If not provided, it falls back to the env var `ALLOWED_ORIGINS`.
- Auth: If `search.api_key` or top‑level `api_key` is set in the site config, protected endpoints require `x-api-key: <key>`.
- Protected endpoints: all except `GET /status` require API key when configured.


## Intent Detection & Behavior System (v2.0)
The worker uses a generic behavior framework where all intents are defined in site configuration (not hardcoded). Custom intents map user queries to response behaviors through hybrid detection.

### Behavior Framework
Ten built-in behaviors control how the worker generates responses. Each behavior has its own response strategy, token budget, and output format.

**LLM-based behaviors** (generate answers via Workers AI):
- `short_answer` - Quick facts, 1-2 sentences. Uses top 2 matches, metadata-only context (no KV text fetch). Max 150 tokens.
- `medium_answer` - Moderate detail, 2-4 sentences with supporting context.
- `long_form_answer` - Comprehensive answers with full KV text context. This is the **default behavior** when no intent matches.
- `detailed_explanation` - In-depth explanations with multiple source citations.
- `single_page_summary` - Summarizes a single page's content.
- `comparison` - Compares multiple results side-by-side.
- `navigation_help` - Guides users to the right page/section.

**List behaviors** (return `concreteDirective` for CMS rendering):
- `short_blurb_with_list` - LLM generates a 1-2 sentence blurb; CMS renders the list from its database via `concreteDirective`. 85% token reduction vs. having the LLM generate the list.
- `collection_overview` - Overview of a content collection with directive to render items.
- `recent_items` - Brief intro with directive to render recent content.

List behaviors return a `concreteDirective` object in the response instead of (or alongside) an LLM-generated answer. The CMS widget reads this directive and renders the list directly from the database, avoiding expensive token usage for enumerating items.

```json
{
  "answer": "Here are our services:",
  "concreteDirective": {
    "type": "render_children",
    "pageId": 42,
    "sortBy": "weight",
    "limit": 10
  },
  "behavior": "short_blurb_with_list",
  "intent": "services"
}
```

Directive types: `render_children`, `render_siblings`, `render_recent`.

### Custom Intents (in-band per request)
Intents come from the CMS in the request body (`body.intents`). Each intent maps detection rules to a response behavior, with optional model and prompt overrides. Each intent also carries a pre-computed `detection_embedding` (BGE-1024) of its name + description + examples + keywords.

```json
{
  "custom_intents": [
    {
      "name": "Services",
      "response_behavior": "medium_answer",
      "enabled": true,
      "is_system": false,
      "system_prompt": "...optional override...",
      "chat_model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
      "detection": {
        "keywords": ["services", "do you do", "do you offer", "..."],
        "description": "Plain-English description of what queries should hit this intent.",
        "examples": ["Do you do SEO?", "Tell me about your branding service", "..."],
        "metadata_matches": {
          "path_starts_with": "/services",
          "title_contains": ["service", "capability"]
        }
      },
      "detection_embedding": [0.012, -0.034, ...]
    }
  ]
}
```

Intent fields:
- `name` (required) - Unique identifier
- `response_behavior` (required) - One of the 10 behavior names
- `enabled` (optional, default true) - Set `false` to disable without deleting
- `is_system` (optional) - System intents (Default, Not Found) are excluded from scored detection and routed deterministically
- `system_prompt` (optional) - Appended to the site prompt for this intent
- `chat_model` / `answer_model` (optional) - Per-intent LLM override
- `detection.keywords` - Substring/word-boundary matches; multi-word phrases are stronger signals
- `detection.description`, `detection.examples` - Plain text fed into the embedding source
- `detection.metadata_matches` - Optional: `path_starts_with`, `collection`, `page_kind`, `title_contains` for evidence-based boosting
- `detection_embedding` - Pre-computed vector; the CMS auto-reembeds via `/admin/embed` whenever description/examples/keywords change

> `priority` is **deprecated** as of the score-based detection refactor. Old payloads with `priority` are ignored at routing time; the field is preserved on `CustomIntent` only for back-compat with stale KV writes.

When no custom intent matches with confidence, the worker falls back to the admin's Default system intent, or to `default_behavior` from site config.

### Score-Based Intent Detection (single pass)

The previous three-phase pipeline (pre-search keyword → post-search metadata → embedding similarity) has been replaced by a single scored pass that combines all three signals at once. No priority list — relevance evidence wins.

For each candidate intent (system intents excluded):

```
score = embedding_similarity                  // cosine(query, intent.detection_embedding)
      + capped_keyword_boost                  // up to KW_BOOST_CAP (0.30)
      + max_per_result_metadata_evidence      // max(result.score × meta_evidence) over top-3 vector results
```

Per-criterion metadata weights (asymmetric — path > collection > title > page_kind):

| Criterion | Weight |
|---|---|
| `path_starts_with` | 0.10 |
| `collection` | 0.05 |
| `title_contains` | 0.04 |
| `page_kind` | 0.025 |

Keyword boost weights (capped at 0.30 total):

| Match type | Weight |
|---|---|
| Multi-word substring (`"your process"` ⊂ query) | 0.10 |
| Multi-word loose (all tokens word-matched, gaps allowed: `"your design process"` matches `"your process"`) | 0.05 |
| Single-word, word-boundary match | 0.05 |

After scoring, the worker picks an outcome:

1. **High confidence** — top intent score ≥ `HIGH_CONFIDENCE` (0.68) → route there regardless of vector content. Lets query-pattern intents (e.g., Pricing — no indexed content by design) win on their own evidence.
2. **Content-backed** — top intent above `ABS_THRESHOLD` (0.55) AND top vector score ≥ `contentFloor` (0.6, configurable per-site via `search.not_found_threshold`) → route to top intent.
3. **No relevant content** — vector top score < `contentFloor` AND detection didn't hit high confidence → route to **Not Found** system intent.
4. **Ambiguous** — top1 score doesn't beat top2 by `AMBIGUITY_MARGIN` (0.02) → fall through to admin's **Default** system intent.

Tunables live in `src/search/detection.ts` and `src/search/pipeline.ts`. The `/search` response now includes a `detection` block (`reason`, `top_vector_score`, `score`, `top2` with components) for tuning.

### Three-Pass Re-Ranking
After vector search returns `initial_topK` results (default 15), three re-ranking passes refine the order before slicing to `final_topK` (default 3) for the behavior.

1. **Pass 1 - Metadata + Query-Token Overlap Boost:** Two boosts in one pass, applied to **all** matches (before the top-8 candidate slice):
   - Intent-driven: results whose metadata matches the intent's `metadata_matches` criteria (collection, page_kind, path prefix, title contains).
   - Query-driven: each non-stopword token from the user's query that appears in this page's `metadata.title` adds **+0.20** (cap **+0.40** total); a path-only hit adds **+0.05**. This catches "name in query equals page title" cases — e.g. `"Who is Shae?"` against the team-leaders collection — where pure embedding similarity could rank a different person's page higher and (with low `final_topK`) keep the right page out of the LLM's context. Because this runs before the candidate slice, a strong title-overlap match can be promoted into the top 8 even when its raw embedding rank was lower.
   - Fast, zero KV reads.
2. **Pass 2 - Full Text Fetch:** Fetches full document text from KV for the top 8 candidates after Pass 1.
3. **Pass 3 - Keyword Boost:** Boosts results whose full text contains query keywords and/or the detected intent's keywords. Uses combined, deduplicated keyword set from both sources.

The final re-ranked results are sliced to `final_topK` and passed to the behavior handler.

### Per-Intent Model & Prompt Overrides
Each custom intent can override the site-level LLM settings:
- `chat_model` on the intent overrides `search.chat_model` from site config
- `system_prompt` on the intent overrides the behavior's built-in default prompt

Fallback chain: intent setting → site config setting → system default.

### Response Format
All `/ask` and `/search` responses include `intent` and `behavior` fields. `/search` additionally returns a `detection` block for tuning visibility.

```json
{
  "ok": true,
  "q": "Show me your services",
  "answer": "We offer a range of professional services:",
  "behavior": "short_blurb_with_list",
  "intent": "Page Lists",
  "concreteDirective": { "type": "render_children", "pageId": 42, "sortBy": "weight" },
  "sources": [{ "title": "Services", "url": "/services", "score": 0.89 }],
  "stats": { "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "cached": false, "..." : "..." },
  "detection": {
    "reason": "matched",
    "top_vector_score": 0.682,
    "score": 0.914,
    "top2": [
      { "name": "Page Lists", "score": 0.914, "components": { "embedding": 0.699, "keyword": 0.20, "metadata": 0.016 } },
      { "name": "Services",   "score": 0.882, "components": { "embedding": 0.712, "keyword": 0.05, "metadata": 0.120 } }
    ]
  }
}
```

## Logging controls
- Centralized in `src/log.ts` with a simple flag:
  - Set `export const LOGGING_ENABLED = true;` to enable logs.
  - When enabled, logs include timestamps and duration metrics for key operations (embedding, Vectorize queries, KV reads/writes, handler/route times).


## Caching controls
Caching can be controlled at two levels with clear precedence:

1) Per-request override (highest precedence)
- Add `caching=1` to force caching ON for this request.
- Add `caching=0` to force caching OFF for this request.

2) Site config default (fallback)
- Set `search.caching: true | false` in `cfg:<site>` to define the default behavior when the URL has no `caching` parameter.

What is cached when ON:
- Query embedding cache `qemb:*` is used for /search and /ask (TTL default 90 days; configurable via `search.embed_cache_ttl`).
- /ask answer cache `ans:*` is checked and stored (TTL default 30 days / 2,592,000s; configurable via `search.answer_cache_ttl`). The long TTL is safe because the ingest worker automatically clears answer caches when content changes.
- **All cache keys include site metadata** for multi-tenant safe clearing and isolation.


## Endpoints
All routes are relative to your worker domain, e.g., `https://worx-search.worxbranding.workers.dev`.

General requirements:
- Include `?site=<site>` on all protected endpoints so the worker can load the site config.
- Provide `x-api-key: <key>` header if the site config requires it.

### 1) GET /status
Quick check of configuration for a site.

Request:
```
GET /status?site=<site>
```

Response:
```
{
  "ok": true,
  "site": "worxbranding-dev",
  "vectorize": "worx-ai-index",
  "dims": 1024,
  "embed_model": "@cf/baai/bge-large-en-v1.5",
  "chat_model": "@cf/meta/llama-3.1-8b-instruct",
  "requires_api_key": true
}
```


### 2) GET /search
Semantic search over your index.

Request:
```
GET /search?site=<site>&q=<query>&k=<topK>&debug=1&caching=1
```
Params:
- `site` (required): which tenant/site to query
- `q` (required): search text
- `k`, `topK`, `top_k`, or `limit` (optional): topK results (clamped 1-24); defaults to site config `topK`
- `debug=1` (optional): include brief debug block in response
- `caching=1` (optional): enable embedding cache

**Note:** Response includes `intent` and `behavior` fields showing the detected intent and response behavior used.

Example:
```
curl -s "https://<worker>/search?site=worxbranding-dev&q=hello&k=6&debug=1&caching=1" \
  -H "x-api-key: <key>" | jq .
```

Response (excerpt):
```
{
  "ok": true,
  "site": "worxbranding-dev",
  "q": "hello",
  "k": 6,
  "intent": "default",
  "behavior": "long_form_answer",
  "results": [ { "id": "worxbranding-dev:page:311", "score": 0.66, "metadata": { ... } }, ... ]
}
```


### 3) POST /ask
Retrieval‑augmented generation over your content.

Request:
```
POST /ask?site=<site>&debug=1&caching=1
Content-Type: application/json
x-api-key: <key>

{
  "site": "<site>",
  "q": "Who is Shae?",
  "k": 6,
  "systemPrompt": "Custom system prompt...",
  "temperature": 0.2
}
```
Body parameters:
- `site` (required): Must match query param
- `q` (required): User question
- `k`, `topK`, `top_k`, or `limit` (optional): Number of search results (clamped 1-24)
- `systemPrompt` or `system_prompt` (optional): Override default system prompt
- `temperature` or `chat_temperature` (optional): LLM temperature (clamped 0-1)

Notes:
- The `site` must be present both as a query param (for router to load config) and in the JSON body (validated by handler).
- `debug=1` includes an internal `_debug` block. Omitted otherwise.
- `caching=1` enables both embedding cache and answer cache (30-day TTL, cleared on ingest).
- Response includes `intent` and `behavior` fields showing the detected intent and response behavior.
- List behaviors may return a `concreteDirective` object instead of a full LLM-generated answer.

Response (excerpt):
```
{
  "ok": true,
  "q": "Who is Shae?",
  "k": 6,
  "answer": "…",
  "stats": {
    "question": "Who is Shae?",
    "found_index": true,
    "cached": false,
    "model": "@cf/meta/llama-3.1-8b-instruct",
    "tokens_input": 842,
    "tokens_output": 181,
    "total_tokens": 1023,
    "timestamp": "2025-10-23T13:44:00.000Z"
  }
}
```
On answer cache hit, `stats.cached = true` and token fields are null.


### 4) Admin: /admin/clear-cache (GET or POST)
Clear caches under `ans:*` and/or `qemb:*` prefixes in CONFIG KV. Requires API key.

**Multi-tenant safe:** Only deletes cache keys with matching site metadata, preventing cross-site cache interference.

Requests:
- Clear all:
```
GET /admin/clear-cache?site=<site>
```
- Clear only answers:
```
GET /admin/clear-cache?site=<site>&scope=answers
```
- Clear only embeddings:
```
GET /admin/clear-cache?site=<site>&scope=embeddings
```
- POST with JSON body also supported: `{ "scope": "answers" | "embeddings" | "all" }`

Response:
```
{
  "ok": true,
  "scope": "all",
  "results": [ { "prefix": "ans:", "total": 42, "queued": 42 }, { "prefix": "qemb:", "total": 120, "queued": 120 } ],
  "note": "Deletes queued asynchronously"
}
```


### 5) Admin: POST /admin/embed
Compute a BGE-1024 embedding vector for arbitrary text. Used by the CMS to re-embed an intent's `(name + description + examples + keywords)` source whenever any of those fields change, then write the resulting vector back to the intent record's `detectionEmbedding` column.

HMAC-protected (X-Worx-Timestamp + X-Worx-Signature). Same auth as every other route.

Request body:
```json
{ "site": "<site>", "text": "<source text to embed>" }
```

Response:
```json
{ "ok": true, "vector": [0.012, -0.034, ...], "dims": 1024 }
```

### 6) Debug: GET /debug/embed
Sanity‑check embed model/dimensions.

```
GET /debug/embed?site=<site>&q=<text>
```
Response example:
```
{ "ok": true, "q": "hello", "result": { "valid": true, "gotLen": 1024, "expected": 1024 } }
```


### 6) Debug: GET /debug/query-by-id
Look up nearest neighbors by a specific vector ID. Tries multiple ID variants and request shapes; includes a robust fallback that embeds KV text if by‑ID lookups fail.

```
GET /debug/query-by-id?site=<site>&id=<vectorId>&k=3&raw=1&doc=1&caching=1
```
Params:
- id (required): vector identifier to try. The handler will attempt combinations of:
  - `<site>:<id>`, `<id>`, stripped form without site prefix, and their `doc:`‑prefixed variants.
- k (optional): topK; default 3; robustly parsed from digits
- raw=1 (optional): only use the provided ID as‑is
- doc=1 (optional): only try `doc:`‑prefixed variants
- caching=1 (optional): used if the fallback path embeds KV text

If by‑ID lookup fails but a document text is found in KV, the worker embeds a snippet and returns neighbors with a note `fallback: embedded KV text via <doc_key>`.


### 7) Debug: GET /debug/list-ids
List candidate vector IDs by scanning KV document keys for the site.

```
GET /debug/list-ids?site=<site>&limit=100&cursor=<cursor>&stripped=1&validate=1
```
- stripped=1: include IDs without the site prefix
- validate=1: for each ID, try a lightweight Vectorize by‑ID query to see which resolve; adds validation summary to the response


## Request examples
- Search without caching (default):
```
curl -s "https://<worker>/search?site=<site>&q=hello" -H "x-api-key: <key>" | jq .
```
- Search with caching:
```
curl -s "https://<worker>/search?site=<site>&q=hello&caching=1" -H "x-api-key: <key>" | jq .
```
- Ask without caching (default):
```
curl -s -X POST "https://<worker>/ask?site=<site>" \
  -H "content-type: application/json" -H "x-api-key: <key>" \
  -d '{"site":"<site>","q":"..."}' | jq .
```
- Ask with caching:
```
curl -s -X POST "https://<worker>/ask?site=<site>&caching=1" \
  -H "content-type: application/json" -H "x-api-key: <key>" \
  -d '{"site":"<site>","q":"..."}' | jq .
```
- Clear cache (all):
```
curl -s "https://<worker>/admin/clear-cache?site=<site>" -H "x-api-key: <key>" | jq .
```
- List IDs for a site:
```
curl -s "https://<worker>/debug/list-ids?site=<site>" -H "x-api-key: <key>" | jq .
```
- Query by ID (with doc variants):
```
curl -s "https://<worker>/debug/query-by-id?site=<site>&id=page:311&k=3&doc=1" -H "x-api-key: <key>" | jq .
```


## Development
Prereqs: Node 18+, Wrangler CLI, Cloudflare account with Vectorize, Workers AI, and KV bound.

- Install deps: `npm i`
- Local dev (remote bindings): `npm run dev`
- Tail logs: `npm run monitor`
- Deploy: `npm run deploy`

Ensure your `wrangler.jsonc` has correct bindings and your KV contains the necessary `cfg:<site>` entries.


## Automatic Cache Invalidation
The search worker integrates with the ingest worker to automatically clear stale answer caches when content changes.

**When cache is cleared:**
- After ingest worker updates/adds content (via `/ingest/run`)
- After ingest worker deletes content (via `/ingest/delete`)
- After ingest worker clears site (via `/ingest/clear`)
- After queue processing completes (async operations)

**How it works:**
1. Ingest worker calls `/admin/clear-cache?site=<site>` after content operations
2. Search worker deletes all `ans:*` keys with matching site metadata
3. Query embeddings (`qemb:*`) are preserved (deterministic, content-independent)
4. Next /ask request generates fresh answer with updated content
5. New answer is cached with site metadata for future requests

**Benefits:**
- Users always see answers based on current content
- No manual cache clearing required after content updates
- Multi-tenant safe (only affects specified site's cache)
- Preserves query embedding cache for performance

**Note:** Cache clearing is automatic when using the ConcreteCMS package's event-driven auto-indexing or manual ingest operations.


## Troubleshooting
- CORS blocked: Verify the request Origin matches `search.allowed_origins` in the site config or `ALLOWED_ORIGINS` env var.
- 401 API key required: Ensure `x-api-key` header matches `search.api_key` (or top-level `api_key`) in the site config.
- Vectorize 40006 (invalid query vector, got 0 dimensions):
  - Use `/debug/query-by-id` with `&doc=1` or `&raw=1` to try different ID variants.
  - Use `/debug/list-ids` to enumerate likely IDs; `&validate=1` checks which resolve by ID.
  - The handler includes a fallback that embeds KV text when by‑ID lookup fails, returning nearest neighbors.
- /ask returns null token counts: That’s expected on cache hits; the answer was served from cache, so no new model usage was incurred.
- No caching effect: Remember caching is opt‑in per request. Add `&caching=1` to enable.


## License
Private/internal project unless otherwise specified.
