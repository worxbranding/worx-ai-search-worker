import type { Env, SearchMatch, SiteConfig, CustomIntent } from "../lib/types";

/**
 * Context provided to each behavior when executing.
 */
export interface BehaviorContext {
  query: string;
  matches: SearchMatch[];
  intent: CustomIntent | null;
  config: SiteConfig;
  env: Env;
}

/**
 * Response structure that behaviors can return.
 * Supports both traditional LLM answers and concrete directives.
 */
export interface BehaviorResponse {
  // Traditional LLM answer (most behaviors)
  answer?: string;

  // For list-based behaviors that delegate rendering to CMS
  blurb?: string;
  concreteDirective?: {
    type: "render_children" | "render_siblings" | "render_recent";
    pageId?: number;
    sortBy?: "weight" | "date" | "alpha";
    limit?: number;
  };

  // Metadata about the response
  intent?: string;
  behavior: string;
  sources?: Array<{
    title?: string;
    url?: string;
    score?: number;
  }>;

  // Token usage tracking
  tokens_input?: number | null;
  tokens_output?: number | null;
  total_tokens?: number | null;

  // Model and settings used
  model?: string;
  temperature?: number;
}

/**
 * Base interface that all behavior handlers must implement.
 */
export interface BehaviorHandler {
  /**
   * The unique identifier for this behavior (e.g., "short_blurb_with_list")
   */
  readonly name: string;

  /**
   * Execute this behavior and return a response.
   *
   * @param context - Contains query, search results, intent config, and environment
   * @returns Promise resolving to a behavior response
   */
  execute(context: BehaviorContext): Promise<BehaviorResponse>;
}
