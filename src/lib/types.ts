/**
 * Shared type definitions for the WORX AI search worker. Housing them in a
 * single module keeps dependencies tidy and makes it easier for future files
 * to import consistent shapes.
 */

/** Cloudflare Vectorize binding wrapper used by the worker. */
export interface VectorizeIndex {
  query(
    arg: number[] | { vectorId: string } | Record<string, unknown>,
    opts?: Record<string, unknown>
  ): Promise<{ matches?: Array<Record<string, unknown>> }>;
}

/** Cloudflare AI binding wrapper. The response shape varies by model. */
export interface Ai {
  run(model: unknown, input: unknown): Promise<unknown>;
}

/** Cloudflare execution context gives access to waitUntil for background tasks. */
export interface ExecutionContext {
  waitUntil(promise: Promise<unknown>): void;
}

/** Generic KV namespace binding. Only the methods we rely on are narrowed here. */
export interface KVNamespace {
  get<T = string>(key: string, type?: "text" | "json"): Promise<T | null>;
  put?(key: string, value: string, opts?: Record<string, unknown>): Promise<void>;
  delete?(key: string): Promise<void>;
  list?(
    opts?: { prefix?: string; limit?: number; cursor?: string }
  ): Promise<{
    keys: Array<{ name: string }>;
    list_complete: boolean;
    cursor?: string | null;
  }>;
}

/** Environment-binding bag passed to the Worker entrypoint. */
export interface Env {
  VECTORIZE: VectorizeIndex;
  AI: Ai;
  /** Optional response/embedding cache. Reads are best-effort; never fatal. */
  CACHE: KVNamespace;
  /** Pre-ingested document text chunks (read-only on the search side). */
  CONTENT: KVNamespace;
  /** Embedding model id used for queries — must match the indexed corpus. */
  EMBED_MODEL: string;
  /** Vectorize embedding dims. Stored as string because wrangler vars are strings. */
  VECTORIZE_DIMS: string;
  /** Vectorize distance metric (defaults to "cosine" if unset). */
  VECTORIZE_METRIC?: string;
  ALLOWED_ORIGINS?: string;
  AI_GATEWAY_ACCOUNT_ID?: string;
  AI_GATEWAY_SLUG?: string;
  OPENAI_API_KEY?: string;
  ANTHROPIC_API_KEY?: string;
  /** Current HMAC shared secret. Required. */
  WORX_HMAC_SECRET: string;
  /** Optional previous HMAC secret accepted during rotation overlap. */
  WORX_HMAC_SECRET_PREVIOUS?: string;
}

/** Multi-provider answer model selector. Stored per-site or per-intent. */
export type LlmProvider = "cloudflare" | "openai" | "anthropic";

export interface AnswerModel {
  provider: LlmProvider;
  model: string;
}

/** Custom intent configuration for the new behavior system. */
export interface CustomIntent {
  name: string;
  response_behavior: string;
  system_prompt?: string;
  chat_model?: string;
  answer_model?: AnswerModel;
  /** @deprecated No longer used by detection — kept optional for back-compat with old KV payloads. */
  priority?: number;
  enabled?: boolean;
  /** Package-owned fallback (Default, Not Found). UI hides delete; pipeline routes no-match queries here. */
  is_system?: boolean;
  /** Per-intent tuning overrides — undefined = inherit site default. */
  chat_temperature?: number;
  initial_topK?: number;
  final_topK?: number;
  max_output_tokens?: number;
  max_context_docs?: number;
  max_kv_text_chars?: number;
  detection: {
    keywords?: string[];
    description?: string;
    examples?: string[];
    metadata_matches?: {
      title_contains?: string[];
      page_kind?: string;
      collection?: string;
      path_starts_with?: string;
    };
  };
  /** Pre-computed embedding of (name + description + examples + keywords). */
  detection_embedding?: number[];
}

/** Per-site configuration retrieved from KV. */
export type SiteConfig = {
  site_key: string;
  vectorize: { index_name: string; dims: number; metric?: "cosine" | "euclidean" | "dot" };
  ai: { embed_model: string };
  api_key?: string;
  custom_intents?: CustomIntent[];
  default_behavior?: string;
  search?: {
    allowed_origins?: string[];
    api_key?: string;
    chat_model?: string;
    answer_model?: AnswerModel;
    chat_temperature?: number;
    system_prompt?: string;
    topK?: number;
    initial_topK?: number; // How many results to fetch from vectorize for re-ranking (default: 15)
    final_topK?: number; // How many results to pass to behavior after re-ranking (default: 3)
    max_output_tokens?: number;
    max_context_docs?: number;
    max_kv_text_chars?: number;
    answer_cache_ttl?: number;
    embed_cache_ttl?: number;
    caching?: boolean;
    /**
     * Cosine-similarity floor below which the worker treats a query as
     * "no relevant match found" and routes to the Not Found system
     * intent. Default 0.5.
     */
    not_found_threshold?: number;
    /**
     * Cosine-similarity floor for matching the query embedding against
     * an intent's pre-computed detection_embedding. Default 0.55.
     */
    intent_embedding_threshold?: number;

    /**
     * Per-behavior LLM `max_tokens` ceiling. Each behavior class enforces
     * its own cap on how many tokens the LLM is allowed to generate.
     * Per-intent `max_output_tokens` is clamped *by* these — so a 600-token
     * intent setting on a behavior with a 256 cap still produces only 256
     * tokens. Increase for sites with longer-form content; decrease for
     * snappy widget answers.
     *
     * Defaults reflect the original WORX tuning. Each value is **the maximum
     * the behavior will allow** — most behaviors then clamp the per-intent
     * `max_output_tokens` into a `[floor, ceiling]` range.
     */
    behavior_caps?: {
      /** ShortAnswer behavior — direct factual answers, kept very short. Default 150. */
      short_answer?: number;
      /** ShortBlurbWithList — the 1-2 sentence intro before the rendered child list. Default 256. */
      short_blurb_with_list?: number;
      /** ShortBlurbWithList fallback path (no children to render — answers inline). Default 384. */
      short_blurb_with_list_fallback?: number;
      /** MediumAnswer — typical Team Members / Services answers. Default 300. */
      medium_answer?: number;
      /** LongFormAnswer — the *ceiling* the behavior clamps the per-intent value into. Default 600. */
      long_form_answer?: number;
      /** LongFormAnswer floor — minimum tokens, prevents accidental ultra-short replies. Default 300. */
      long_form_answer_floor?: number;
      /** LongFormAnswer baseline default when no intent override is set. Default 500. */
      long_form_answer_default?: number;
      /** DetailedExplanation ceiling. Default 800. */
      detailed_explanation?: number;
      /** DetailedExplanation floor. Default 300. */
      detailed_explanation_floor?: number;
      /** DetailedExplanation default when no intent override. Default 600. */
      detailed_explanation_default?: number;
      /** SinglePageSummary ceiling. Default 500. */
      single_page_summary?: number;
      /** SinglePageSummary floor. Default 200. */
      single_page_summary_floor?: number;
      /** SinglePageSummary default. Default 400. */
      single_page_summary_default?: number;
      /** Comparison ceiling — comparison answers tend to run long. Default 1536. */
      comparison?: number;
      /** Comparison floor. Default 384. */
      comparison_floor?: number;
      /** Comparison default. Default 896. */
      comparison_default?: number;
      /** CollectionOverview — overview text rendered above a collection. Default 512. */
      collection_overview?: number;
      /** RecentItems — blurb intro before the rendered recent-items list. Default 256. */
      recent_items_blurb?: number;
      /** RecentItems fallback path (no list to render — answers inline). Default 384. */
      recent_items_fallback?: number;
      /** NavigationHelp — short pointer answers ("see X page"). Default 256. */
      navigation_help?: number;
    };

    /**
     * Re-rank boost weights applied during the three-pass re-rank
     * (metadata → fetch full text → keywords). Each match adds the
     * configured value to the candidate's score before re-sort. Tuning
     * these shifts which of (semantic similarity, exact path match,
     * title overlap, full-text keywords) dominates the final ranking.
     *
     * Defaults are tuned for WORX-style marketing-site content: short
     * pages with title-heavy semantics. A site with deep URL hierarchies
     * may want to raise the path weights; a site with very long pages
     * may want to lean more on `keyword_in_full_text`.
     */
    rerank_weights?: {
      /** Boost when match metadata.collection equals the intent's expected collection. Default 0.10. */
      metadata_collection?: number;
      /** Boost when metadata.page_kind equals the intent's expected kind. Default 0.05. */
      metadata_page_kind?: number;
      /** Strong boost when metadata.path EXACTLY equals the expected prefix. Default 0.50. */
      metadata_exact_path?: number;
      /** Boost when metadata.path starts-with the expected prefix (but isn't exact). Default 0.30. */
      metadata_path_prefix?: number;
      /** Boost when metadata.title contains any term from the intent's `title_contains` list. Default 0.15. */
      metadata_title_contains?: number;
      /** Per-token boost when a non-stopword from the user's query appears in metadata.title. Default 0.20. */
      query_token_in_title?: number;
      /** Per-token boost when a query token only appears in metadata.path (lower than title). Default 0.05. */
      query_token_in_path?: number;
      /** Cap on total query-token boost so it can't swamp intent-driven boosts. Default 0.40. */
      query_token_max_total?: number;
      /** Boost when an extracted query keyword appears in metadata.title. Default 0.25. */
      keyword_in_title?: number;
      /** Boost when a keyword appears in metadata.preview. Default 0.15. */
      keyword_in_preview?: number;
      /** Boost when a keyword appears in fetched full text. Default 0.20. */
      keyword_in_full_text?: number;
      /** Boost when a keyword appears in metadata.path only. Default 0.05. */
      keyword_in_path?: number;
      /** Per-keyword bonus when the result contains 2+ query keywords. Default 0.10. */
      multi_keyword_per_match?: number;
    };

    /**
     * Intent detection thresholds and slice sizes. Tuning these changes
     * how aggressive the worker is about routing low-confidence queries
     * to a real intent vs. falling through to Not Found / Default.
     */
    detection?: {
      /**
       * Detection score above which the worker accepts the routed intent
       * REGARDLESS of whether vector content also passes the content
       * floor. Genuine high-confidence matches (clear keyword + strong
       * embedding) survive even when the indexed content is thin.
       * Default 0.68.
       */
      high_confidence_score?: number;
      /**
       * Number of top candidates the keyword/full-text re-rank pass
       * operates on. Smaller = faster but may miss good matches whose
       * raw embedding rank is borderline. Default 8.
       */
      candidate_slice?: number;
    };
  };
};

/** Vector search match structure returned by Vectorize. */
export type SearchMatch = {
  id: string;
  score: number;
  metadata?: Record<string, unknown>;
};

/** Parsed child link from the Markdown stored alongside documents. */
export type ChildLink = {
  title: string;
  url: string;
  normalizedUrl: string;
};

/** Supported intent buckets used by prompt building. */
export type IntentKey =
  | "person"
  | "service"
  | "case_study"
  | "page_list"
  | "how_to"
  | "company_info"
  | "contact"
  | "default";

/** Intent detection result includes both the bucket and harvested keywords. */
export interface IntentResult {
  intent: IntentKey;
  keywords: string[];
}
