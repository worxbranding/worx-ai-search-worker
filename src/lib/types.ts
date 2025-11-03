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
  CONFIG: KVNamespace;
  TRAINING_CACHE?: KVNamespace;
  ALLOWED_ORIGINS?: string;
}

/** Training-related configuration surfaced through the site config document. */
export type TrainingConfig = {
  enabled?: boolean;
  apiBase?: string;
  apiKey?: string;
  sessionEndpoint?: string;
  feedbackEndpoint?: string;
  publishEndpoint?: string;
  replayEndpoint?: string;
  policy?: {
    activeKey?: string;
    keyPrefix?: string;
    cacheTtlSeconds?: number;
    defaultPromptVariant?: string;
    weights?: Record<string, number>;
    metadata?: Record<string, unknown>;
  };
  logger?: {
    enabled?: boolean;
    endpoint?: string;
    headers?: Record<string, string>;
    cacheKeyPrefix?: string;
    ttlSeconds?: number;
    timeoutMs?: number;
  };
  [key: string]: unknown;
};

/** Per-site configuration retrieved from KV. */
export type SiteConfig = {
  site_key: string;
  vectorize: { index_name: string; dims: number; metric?: "cosine" | "euclidean" | "dot" };
  ai: { embed_model: string };
  api_key?: string;
  search?: {
    allowed_origins?: string[];
    api_key?: string;
    chat_model?: string;
    chat_temperature?: number;
    system_prompt?: string;
    topK?: number;
    max_output_tokens?: number;
    max_context_docs?: number;
    max_kv_text_chars?: number;
    answer_cache_ttl?: number;
    caching?: boolean;
  };
  training?: TrainingConfig;
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
