import type { SearchMatch, SiteConfig } from "../lib/types";
import { log } from "../lib/logging";
import { rerankWeight } from "../lib/configDefaults";

/**
 * Extract meaningful keywords from query (excluding common stop words)
 */
export function extractKeywords(query: string): string[] {
  const stopWords = new Set([
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
    'could', 'can', 'may', 'might', 'must', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
    'when', 'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'who', 'what', 'which', 'this', 'that', 'these',
    'those', 'am', 'as', 'if', 'or', 'because', 'until', 'while', 'but', 'and'
  ]);

  return query
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(word => word.length > 2 && !stopWords.has(word));
}

/**
 * Apply metadata-based boosting (collection, page_kind, path, title).
 * This pass is fast and doesn't require fetching full text from KV.
 *
 * Also applies a per-result *query-token* title/path overlap boost: if any
 * non-stopword token from the user's query appears in this page's title or
 * URL path, the page gets a meaningful boost. This is what catches "the name
 * in the query equals the page title" cases like "Who is Shae?" — where pure
 * embedding similarity could otherwise rank a different team-member page
 * higher and (with low top_k) keep the right page out of the LLM's context.
 *
 * This pass runs on ALL matches before the top-8 candidate slice, so the
 * boost can pull a relevant page into the candidate set even when its raw
 * embedding rank was outside the top 8.
 */
export function applyMetadataBoost(
  matches: SearchMatch[],
  metadataMatches: Record<string, any>,
  queryTokens: string[] = [],
  config?: SiteConfig
): SearchMatch[] {
  // Resolve per-site weights once (defaults from configDefaults if absent).
  const W = {
    collection:       rerankWeight(config, 'metadata_collection'),
    pageKind:         rerankWeight(config, 'metadata_page_kind'),
    exactPath:        rerankWeight(config, 'metadata_exact_path'),
    pathPrefix:       rerankWeight(config, 'metadata_path_prefix'),
    titleContains:    rerankWeight(config, 'metadata_title_contains'),
    queryTokenTitle:  rerankWeight(config, 'query_token_in_title'),
    queryTokenPath:   rerankWeight(config, 'query_token_in_path'),
    queryTokenMaxTotal: rerankWeight(config, 'query_token_max_total'),
  };

  return matches.map((match) => {
    const metadata = (match.metadata || {}) as Record<string, any>;
    let boost = 0;
    const boostDetails: string[] = [];

    if (metadataMatches.collection && metadata.collection === metadataMatches.collection) {
      boost += W.collection;
      boostDetails.push(`collection match (+${W.collection.toFixed(2)})`);
    }

    if (metadataMatches.page_kind && metadata.page_kind === metadataMatches.page_kind) {
      boost += W.pageKind;
      boostDetails.push(`page_kind match (+${W.pageKind.toFixed(2)})`);
    }

    if (metadataMatches.path_starts_with && typeof metadata.path === 'string') {
      if (metadata.path === metadataMatches.path_starts_with) {
        boost += W.exactPath;
        boostDetails.push(`exact path match (+${W.exactPath.toFixed(2)})`);
      } else if (metadata.path.startsWith(metadataMatches.path_starts_with)) {
        boost += W.pathPrefix;
        boostDetails.push(`path prefix match (+${W.pathPrefix.toFixed(2)})`);
      }
    }

    if (metadataMatches.title_contains && Array.isArray(metadataMatches.title_contains) && typeof metadata.title === 'string') {
      const titleLower = metadata.title.toLowerCase();
      for (const term of metadataMatches.title_contains) {
        if (titleLower.includes(String(term).toLowerCase())) {
          boost += W.titleContains;
          boostDetails.push(`title contains "${term}" (+${W.titleContains.toFixed(2)})`);
          break;
        }
      }
    }

    // Query-token overlap with this page's title / path. Title hits weigh
    // more than path hits. Capped so it can promote a missed page into
    // the candidate set without swamping intent-driven boosts.
    if (queryTokens.length > 0) {
      const titleLower = String(metadata.title || '').toLowerCase();
      const pathLower  = String(metadata.path  || '').toLowerCase();
      let queryBoost = 0;
      const titleHits: string[] = [];
      const pathHits: string[] = [];
      for (const tok of queryTokens) {
        if (!tok) continue;
        if (titleLower.includes(tok)) {
          queryBoost += W.queryTokenTitle;
          titleHits.push(tok);
        } else if (pathLower.includes(tok)) {
          queryBoost += W.queryTokenPath;
          pathHits.push(tok);
        }
      }
      if (queryBoost > W.queryTokenMaxTotal) queryBoost = W.queryTokenMaxTotal;
      if (queryBoost > 0) {
        boost += queryBoost;
        if (titleHits.length) boostDetails.push(`query token(s) in title [${titleHits.join(',')}] (+${queryBoost.toFixed(2)})`);
        else if (pathHits.length) boostDetails.push(`query token(s) in path [${pathHits.join(',')}] (+${queryBoost.toFixed(2)})`);
      }
    }

    return {
      ...match,
      score: match.score + boost,
      _boostDetails: boostDetails,
      _originalScore: match.score,
      _metadataBoost: boost,
    };
  }).sort((a, b) => b.score - a.score);
}

/**
 * Apply keyword-based boosting using FULL TEXT content from KV.
 * Requires matches to have _fullText property already fetched.
 * This is more expensive but more accurate than metadata-only boosting.
 */
export function applyKeywordBoost(
  matches: SearchMatch[],
  queryKeywords: string[],
  config?: SiteConfig
): SearchMatch[] {
  if (queryKeywords.length === 0) {
    log("[KeywordBoost] No keywords to boost");
    return matches;
  }

  const W = {
    title:    rerankWeight(config, 'keyword_in_title'),
    preview:  rerankWeight(config, 'keyword_in_preview'),
    fullText: rerankWeight(config, 'keyword_in_full_text'),
    path:     rerankWeight(config, 'keyword_in_path'),
    multi:    rerankWeight(config, 'multi_keyword_per_match'),
  };

  return matches.map((match) => {
    const metadata = (match.metadata || {}) as Record<string, any>;
    const fullText = (match as any)._fullText as string | null;

    let boost = 0;
    const boostDetails = Array.isArray((match as any)._boostDetails)
      ? [...(match as any)._boostDetails]
      : [];

    const title = String(metadata.title || '').toLowerCase();
    const preview = String(metadata.preview || '').toLowerCase();
    const path = String(metadata.path || '').toLowerCase();
    const fullTextLower = fullText ? fullText.toLowerCase() : '';

    let keywordMatchCount = 0;
    for (const keyword of queryKeywords) {
      // Priority 1: Title match (strongest signal)
      if (title.includes(keyword)) {
        keywordMatchCount++;
        boost += W.title;
        boostDetails.push(`keyword "${keyword}" in title (+${W.title.toFixed(2)})`);
      }
      // Priority 2: Preview match
      else if (preview.includes(keyword)) {
        keywordMatchCount++;
        boost += W.preview;
        boostDetails.push(`keyword "${keyword}" in preview (+${W.preview.toFixed(2)})`);
      }
      // Priority 3: Full text match
      else if (fullTextLower && fullTextLower.includes(keyword)) {
        keywordMatchCount++;
        boost += W.fullText;
        boostDetails.push(`keyword "${keyword}" in full text (+${W.fullText.toFixed(2)})`);
      }
      // Priority 4: Path match (weakest)
      else if (path.includes(keyword)) {
        keywordMatchCount++;
        boost += W.path;
        boostDetails.push(`keyword "${keyword}" in path (+${W.path.toFixed(2)})`);
      }
    }

    // Extra boost if page contains multiple keywords from query
    if (keywordMatchCount >= 2) {
      const multiBoost = W.multi * keywordMatchCount;
      boost += multiBoost;
      boostDetails.push(`multiple keywords (${keywordMatchCount}) (+${multiBoost.toFixed(2)})`);
    }

    const previousBoost = (match as any)._metadataBoost || 0;
    const totalBoost = previousBoost + boost;

    return {
      ...match,
      score: (match as any)._originalScore + totalBoost,
      _boostDetails: boostDetails,
      _keywordBoost: boost,
      _totalBoost: totalBoost,
    };
  }).sort((a, b) => b.score - a.score);
}
