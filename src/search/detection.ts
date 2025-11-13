import type { CustomIntent, SearchMatch } from "../lib/types";

/**
 * Hybrid Intent Detection System
 *
 * Two-phase detection:
 * 1. detectIntentFromQuery - Fast keyword-based pre-search detection
 * 2. detectIntentFromResults - Semantic metadata-based post-search detection
 */

/**
 * Phase 1: Pre-Search Keyword Matching (Fast Path)
 *
 * Check if query matches any intent keywords BEFORE running vector search.
 * Benefits:
 * - Fast detection without waiting for vector search
 * - Can optimize search parameters based on detected intent
 * - Catches direct questions: "Are you hiring?" → Careers intent
 *
 * @param query - User's query string
 * @param customIntents - Array of custom intent configurations
 * @returns Matched CustomIntent or null
 */
export function detectIntentFromQuery(
  query: string,
  customIntents: CustomIntent[]
): CustomIntent | null {
  if (!customIntents || customIntents.length === 0) {
    return null;
  }

  const queryLower = query.toLowerCase().trim();
  if (!queryLower) {
    return null;
  }

  // Sort by priority (highest first)
  const sortedIntents = [...customIntents]
    .filter((intent) => intent.enabled !== false)
    .sort((a, b) => (b.priority || 50) - (a.priority || 50));

  for (const intent of sortedIntents) {
    const keywords = intent.detection?.keywords;
    if (!keywords || keywords.length === 0) {
      continue;
    }

    // Check if any keyword matches
    for (const keyword of keywords) {
      if (!keyword) continue;

      const keywordLower = keyword.toLowerCase().trim();
      if (queryLower.includes(keywordLower)) {
        return intent; // Early match
      }
    }
  }

  return null; // No keyword match
}

/**
 * Phase 2: Post-Search Metadata Matching (Semantic Path)
 *
 * After vector search returns results, analyze top results for intent signals.
 * Benefits:
 * - Semantic understanding through vector search results
 * - Catches indirect questions: "What opportunities are available?" → /careers
 * - Works even when user doesn't use exact terminology
 *
 * @param results - Vector search results with metadata
 * @param customIntents - Array of custom intent configurations
 * @returns Matched CustomIntent or null
 */
export function detectIntentFromResults(
  results: SearchMatch[],
  customIntents: CustomIntent[]
): CustomIntent | null {
  if (!results || results.length === 0) {
    return null;
  }

  if (!customIntents || customIntents.length === 0) {
    return null;
  }

  // Sort by priority (highest first)
  const sortedIntents = [...customIntents]
    .filter((intent) => intent.enabled !== false)
    .sort((a, b) => (b.priority || 50) - (a.priority || 50));

  // Check top 3 results
  const topResults = results.slice(0, 3);

  for (const intent of sortedIntents) {
    const metadataConfig = intent.detection?.metadata_matches;
    if (!metadataConfig) {
      continue;
    }

    // Check if any of the top results match this intent's metadata criteria
    for (const result of topResults) {
      if (matchesMetadataCriteria(result.metadata, metadataConfig)) {
        return intent; // Result-based match
      }
    }
  }

  return null; // No metadata match
}

/**
 * Check if a result's metadata matches the intent's criteria.
 * Uses AND logic: ALL specified criteria must match.
 * If path_starts_with is specified, it MUST match (strongest signal).
 *
 * @param metadata - Result metadata
 * @param criteria - Intent's metadata matching configuration
 * @returns true if metadata matches ALL criteria
 */
function matchesMetadataCriteria(
  metadata: Record<string, unknown> | undefined,
  criteria: {
    title_contains?: string[];
    page_kind?: string;
    collection?: string;
    path_starts_with?: string;
  }
): boolean {
  if (!metadata || !criteria) {
    return false;
  }

  // If path_starts_with is specified, it MUST match (strongest signal)
  if (criteria.path_starts_with) {
    const path = (metadata.path as string) || (metadata.url as string) || "";
    if (!path.toLowerCase().startsWith(criteria.path_starts_with.toLowerCase())) {
      return false; // Path doesn't match - fail immediately
    }
  }

  // Check collection (if specified, must match)
  if (criteria.collection) {
    const collection = (metadata.collection as string) || "";
    if (collection.toLowerCase() !== criteria.collection.toLowerCase()) {
      return false;
    }
  }

  // Check page_kind (if specified, must match)
  if (criteria.page_kind) {
    const pageKind = (metadata.page_kind as string) || "";
    if (pageKind.toLowerCase() !== criteria.page_kind.toLowerCase()) {
      return false;
    }
  }

  // Check title_contains (if specified, at least one must match)
  if (criteria.title_contains && criteria.title_contains.length > 0) {
    const title = ((metadata.title as string) || "").toLowerCase();
    let titleMatches = false;
    for (const term of criteria.title_contains) {
      if (term && title.includes(term.toLowerCase())) {
        titleMatches = true;
        break;
      }
    }
    if (!titleMatches) {
      return false;
    }
  }

  // All specified criteria matched
  return true;
}

/**
 * Get the default intent configuration.
 * Used as fallback when no custom intent matches.
 */
export function getDefaultIntent(): CustomIntent {
  return {
    name: "default",
    response_behavior: "short_answer",
    priority: 0,
    enabled: true,
    detection: {},
  };
}
