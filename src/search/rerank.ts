import type { SearchMatch } from "../lib/types";
import { log } from "../lib/logging";

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
 */
export function applyMetadataBoost(
  matches: SearchMatch[],
  metadataMatches: Record<string, any>
): SearchMatch[] {
  return matches.map((match) => {
    const metadata = (match.metadata || {}) as Record<string, any>;
    let boost = 0;
    const boostDetails: string[] = [];

    // Boost if collection matches
    if (metadataMatches.collection && metadata.collection === metadataMatches.collection) {
      boost += 0.1;
      boostDetails.push('collection match (+0.1)');
    }

    // Boost if page_kind matches
    if (metadataMatches.page_kind && metadata.page_kind === metadataMatches.page_kind) {
      boost += 0.05;
      boostDetails.push('page_kind match (+0.05)');
    }

    // Boost if path EXACTLY matches expected prefix (strongest signal)
    if (metadataMatches.path_starts_with && typeof metadata.path === 'string') {
      if (metadata.path === metadataMatches.path_starts_with) {
        boost += 0.5;
        boostDetails.push('exact path match (+0.5)');
      } else if (metadata.path.startsWith(metadataMatches.path_starts_with)) {
        boost += 0.3;
        boostDetails.push('path prefix match (+0.3)');
      }
    }

    // Boost if title contains expected terms
    if (metadataMatches.title_contains && Array.isArray(metadataMatches.title_contains) && typeof metadata.title === 'string') {
      const titleLower = metadata.title.toLowerCase();
      for (const term of metadataMatches.title_contains) {
        if (titleLower.includes(String(term).toLowerCase())) {
          boost += 0.15;
          boostDetails.push(`title contains "${term}" (+0.15)`);
          break;
        }
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
  queryKeywords: string[]
): SearchMatch[] {
  if (queryKeywords.length === 0) {
    log("[KeywordBoost] No keywords to boost");
    return matches;
  }

  return matches.map((match) => {
    const metadata = (match.metadata || {}) as Record<string, any>;
    const fullText = (match as any)._fullText as string | null;

    let boost = 0;
    const boostDetails = Array.isArray((match as any)._boostDetails)
      ? [...(match as any)._boostDetails]
      : [];

    // Check metadata fields (fast, but less comprehensive)
    const title = String(metadata.title || '').toLowerCase();
    const preview = String(metadata.preview || '').toLowerCase();
    const path = String(metadata.path || '').toLowerCase();

    // Check full text (slow, but comprehensive) - ONLY if we have it
    const fullTextLower = fullText ? fullText.toLowerCase() : '';

    let keywordMatchCount = 0;
    for (const keyword of queryKeywords) {
      let matchedInField = false;

      // Priority 1: Title match (strongest signal)
      if (title.includes(keyword)) {
        keywordMatchCount++;
        boost += 0.25; // Increased from 0.2
        boostDetails.push(`keyword "${keyword}" in title (+0.25)`);
        matchedInField = true;
      }
      // Priority 2: Preview match
      else if (preview.includes(keyword)) {
        keywordMatchCount++;
        boost += 0.15;
        boostDetails.push(`keyword "${keyword}" in preview (+0.15)`);
        matchedInField = true;
      }
      // Priority 3: Full text match (NEW - most important!)
      else if (fullTextLower && fullTextLower.includes(keyword)) {
        keywordMatchCount++;
        boost += 0.20; // Strong boost for full text match
        boostDetails.push(`keyword "${keyword}" in full text (+0.20)`);
        matchedInField = true;
      }
      // Priority 4: Path match (weakest)
      else if (path.includes(keyword)) {
        keywordMatchCount++;
        boost += 0.05;
        boostDetails.push(`keyword "${keyword}" in path (+0.05)`);
        matchedInField = true;
      }
    }

    // Extra boost if page contains multiple keywords from query
    if (keywordMatchCount >= 2) {
      const multiBoost = 0.1 * keywordMatchCount;
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
