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
  priority: number;
  enabled?: boolean;
  /** Per-intent tuning overrides — undefined = inherit site default. */
  chat_temperature?: number;
  initial_topK?: number;
  final_topK?: number;
  max_output_tokens?: number;
  max_context_docs?: number;
  max_kv_text_chars?: number;
  detection: {
    keywords?: string[];
    metadata_matches?: {
      title_contains?: string[];
      page_kind?: string;
      collection?: string;
      path_starts_with?: string;
    };
  };
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
