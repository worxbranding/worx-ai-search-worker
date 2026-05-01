import type { CustomIntent, Env, SiteConfig } from "../lib/types";

/**
 * Build the SiteConfig object the rest of the worker expects out of:
 *
 *   - environment-level wrangler vars (vectorize index/dims/metric, embed model)
 *   - the in-band request body posted by the CMS (search behaviour + intents)
 *
 * Per-site state used to live in a `cfg:<site>` KV row; that path is gone.
 * The `site` value is now just a routing key (used for Vectorize metadata
 * filtering and KV cache key prefixing), not authentication — auth happens
 * via HMAC at the entry point before this is ever called.
 */
export interface InBandSearchOverrides {
  system_prompt?: string;
  answer_model?: SiteConfig["search"] extends infer S
    ? (S extends { answer_model?: infer A } ? A : never)
    : never;
  chat_temperature?: number;
  initial_topK?: number;
  final_topK?: number;
  topK?: number;
  max_output_tokens?: number;
  max_context_docs?: number;
  max_kv_text_chars?: number;
  caching?: boolean;
  embed_cache_ttl?: number;
  answer_cache_ttl?: number;
  /** Legacy single-string model — promoted to answer_model.cloudflare. */
  chat_model?: string;
}

export interface InBandRequestBody {
  q?: string;
  site?: string;
  k?: number;
  search?: InBandSearchOverrides;
  intents?: CustomIntent[];
  default_behavior?: string;
  /**
   * Per-request cache controls. Either send `caching` for both layers,
   * or split with `cache_answer` / `cache_embedding`. The split fields
   * win over the single one — used by the Training tab to keep the
   * embedding cache hot while forcing fresh LLM calls.
   */
  cache_answer?: boolean;
  cache_embedding?: boolean;
  /**
   * Training-tab "route lock": skip intent detection and use this intent
   * by name. Useful when iterating on a prompt without the matching
   * surface shifting underfoot if the user tweaks keywords.
   */
  force_intent?: string;
}

export interface CacheFlags {
  answer: boolean;
  embedding: boolean;
}

/**
 * Construct the per-request SiteConfig from env vars + posted body. This
 * object is read-only inside the worker; mutating it has no effect on
 * persisted state because nothing is persisted any more.
 */
export function buildSiteConfig(env: Env, site: string, body: InBandRequestBody): SiteConfig {
  const dims = parseInt(env.VECTORIZE_DIMS || "0", 10);
  if (!dims || !Number.isFinite(dims)) {
    throw new Error("VECTORIZE_DIMS env var is missing or not a number");
  }
  if (!env.EMBED_MODEL) {
    throw new Error("EMBED_MODEL env var is required");
  }

  const metric = (env.VECTORIZE_METRIC || "cosine") as SiteConfig["vectorize"]["metric"];

  return {
    site_key: site,
    vectorize: {
      index_name: "worx-ai-index", // pinned by wrangler binding; surfaced for /status only
      dims,
      metric,
    },
    ai: {
      embed_model: env.EMBED_MODEL,
    },
    custom_intents: Array.isArray(body.intents) ? body.intents : [],
    default_behavior: body.default_behavior,
    search: body.search ?? {},
  };
}

/**
 * Pick an allowed CORS origin from the wrangler ALLOWED_ORIGINS list.
 * (Per-site CORS is owned by the CMS in front of the worker; this list is
 * just a defence-in-depth fallback for direct browser/CLI calls.)
 */
export function allowOrigin(origin: string | null, env: Env): string | null {
  const allowed = (env.ALLOWED_ORIGINS || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  if (!origin) return null;
  if (allowed.includes("*")) return "*";
  return allowed.includes(origin) ? origin : null;
}

/**
 * Decide whether the answer + embedding caches are enabled for a given
 * request, in priority order: body.cache_answer/cache_embedding (split)
 * → ?caching=0|1 query string (legacy) → cfg.search.caching → default on.
 *
 * The split fields exist so the Training tab can keep the embedding cache
 * hot while forcing the LLM to re-run on every "Try again."
 */
export function resolveCaching(url: URL, cfg: SiteConfig, body?: InBandRequestBody): CacheFlags {
  const fallback = cfg.search?.caching !== false;
  const queryRaw = (url.searchParams.get("caching") || "").trim();
  const querySays: boolean | null = queryRaw === "1" ? true : queryRaw === "0" ? false : null;
  const baseAll = querySays ?? fallback;

  return {
    answer:    typeof body?.cache_answer === "boolean"    ? body.cache_answer    : baseAll,
    embedding: typeof body?.cache_embedding === "boolean" ? body.cache_embedding : baseAll,
  };
}

/** Guard the vector topK value so callers cannot request outrageous limits. */
export function clampTopK(value: number, min = 1, max = 24): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(Math.max(Math.floor(value), min), max);
}

/**
 * Read the topK request value from several alternate query string names and
 * fall back to the provided configuration default when it is missing.
 */
export function resolveTopKFromQuery(url: URL, fallback: number): number {
  const candidates = [
    url.searchParams.get("k"),
    url.searchParams.get("topK"),
    url.searchParams.get("top_k"),
    url.searchParams.get("limit"),
  ];
  for (const raw of candidates) {
    if (!raw) continue;
    const n = Number(raw);
    if (!Number.isFinite(n)) continue;
    const clamped = clampTopK(n);
    if (clamped > 0) return clamped;
  }
  return clampTopK(fallback);
}

/** Keep the chat temperature within the model's supported bounds. */
export function clampTemperature(value: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(1, Math.max(0, Number(value)));
}
