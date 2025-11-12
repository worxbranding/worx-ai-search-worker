# WORX AI Search Worker

A Cloudflare Worker that provides semantic search and Q&A over site content using Cloudflare Vectorize (v2) and Workers AI. It’s multi-tenant (per-site) with API key protection, CORS controls, optional caching, robust logging, and several debug/admin utilities.

This README explains how it works, how to configure it, and all the ways to interact with it.


## Architecture overview
- Vector store: Cloudflare Vectorize (binding: `VECTORIZE`) containing document embeddings.
- Inference: Workers AI (binding: `AI`) for both embeddings and chat responses.
- State/config: Cloudflare KV (binding: `CONFIG`) for per-site configuration, document texts, and caches.
- Worker routes: REST endpoints for status, search, ask, admin cache control, and debugging.


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
  "search": {
    "allowed_origins": ["http://localhost:5173", "https://your-frontend.app"],
    "api_key": "dev-secret-123",             // API key required on protected endpoints via header x-api-key
    "chat_model": "@cf/meta/llama-3.1-8b-instruct",
    "chat_temperature": 0.1,
    "system_prompt": "You are a helpful assistant...", // optional; worker provides a sensible default with URL restrictions
    "topK": 6,
    "max_output_tokens": 1024,
    // Performance/cost tunables
    "max_context_docs": 6,           // max docs included in /ask prompt context
    "max_kv_text_chars": 3000,       // truncate KV text per doc to this many chars
    "answer_cache_ttl": 600,         // seconds for /ask answer cache
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


## Intent Detection System
The worker includes an intelligent intent detection system that analyzes user queries to optimize search results and response formatting.

**Current Intent Categories (v1.0):**
- `person` - Biography/who questions (e.g., "Who is John Doe?")
- `service` - Services/capabilities queries (e.g., "What services do you offer?")
- `case_study` - Project/portfolio requests (e.g., "Show me case studies")
- `page_list` - List/index requests (e.g., "List all services")
- `how_to` - Process/procedure questions (e.g., "How do I contact support?")
- `company_info` - About/mission queries (e.g., "Tell me about the company")
- `contact` - Contact information requests (e.g., "What's your phone number?")
- `default` - General queries

**How It Works (Current):**
1. User question analyzed for keywords and patterns (hardcoded regex)
2. Intent category detected
3. Search topK adjusted based on intent
4. Results re-ranked using intent-specific algorithms
5. Response formatting optimized for intent type
6. Intent included in response metadata

**Example:**
```json
{
  "answer": "John Doe is our CEO...",
  "intent": "person",
  "sources": [...]
}
```

**🔜 Upcoming in v2.0: Generic Behavior Framework**

The intent system is being redesigned for maximum flexibility. See `/Users/shaeapland/ai_projects/INTENT_SYSTEM_REDESIGN.md` for full details.

Key changes:
- **Hardcoded intents replaced with generic behaviors** (short_blurb_with_list, long_form_answer, etc.)
- **Zero-code custom intent creation** via ConcreteCMS dashboard
- **Hybrid detection:** Keywords (fast path) + Metadata matching (semantic path)
- **Per-intent system prompts** editable without deployment
- All current intents will be migrated to the new system

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
- Query embedding cache `qemb:*` is used for /search and /ask.
- /ask answer cache `ans:*` is checked and stored (TTL default 600s; configurable via `search.answer_cache_ttl`).


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

**Note:** Response includes `intent` field showing detected question category.

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
- `caching=1` enables both embedding cache and short-TTL answer cache.
- Response includes `intent` field showing detected question category.

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


### 5) Debug: GET /debug/embed
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
