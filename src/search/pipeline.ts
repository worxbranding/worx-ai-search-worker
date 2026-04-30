import type { Env, ExecutionContext, SiteConfig, SearchMatch, CustomIntent } from "../lib/types";
import { log, time } from "../lib/logging";
import { cachedEmbed, filterToSite } from "./vectorize";
import { detectIntentFromQuery, detectIntentFromResults, getDefaultIntent } from "./detection";
import { extractKeywords, applyMetadataBoost, applyKeywordBoost } from "./rerank";
import { fetchFullTextForMatches } from "../utils/fullText";

/**
 * Shared search pipeline used by both /search and /ask endpoints.
 *
 * This ensures both endpoints use identical intent detection, re-ranking,
 * and result filtering logic.
 *
 * Flow:
 * 1. Detect intent from query keywords (fast path)
 * 2. Run vector search (fetch initial_topK)
 * 3. Detect intent from results metadata (semantic path)
 * 4. Re-rank Pass 1: Metadata boosting (collection, path, page_kind)
 * 5. Re-rank Pass 2: Fetch full text for top candidates
 * 6. Re-rank Pass 3: Keyword boosting (query + intent keywords)
 * 7. Slice to final_topK for LLM/response
 */
export async function executeSearchPipeline(
  query: string,
  site: string,
  cfg: SiteConfig,
  env: Env,
  ctx: ExecutionContext | undefined,
  cache: { answer: boolean; embedding: boolean }
): Promise<{
  matches: SearchMatch[];
  intent: CustomIntent;
  behaviorName: string;
  queryKeywords: string[];
  intentKeywords: string[];
  allKeywords: string[];
  initial_topK: number;
  final_topK: number;
}> {
  // Get custom intents from config
  const customIntents = cfg.custom_intents || [];

  // Phase 1: Try keyword-based detection (fast path)
  let detectedIntent: CustomIntent | null = detectIntentFromQuery(query, customIntents);
  if (detectedIntent) {
    log("[Intent] Detected via keywords:", detectedIntent.name);
  }

  // Prepare vector search with two-stage filtering. Per-intent overrides
  // take precedence when present — keyword-pre-detected intent overrides
  // initial_topK before the fetch; the final intent (whether keyword or
  // metadata-detected) overrides final_topK at slice time below.
  const initial_topK = Number(
    (detectedIntent?.initial_topK !== undefined ? detectedIntent.initial_topK : cfg.search?.initial_topK) ?? 15
  );
  let final_topK = Number(cfg.search?.final_topK ?? 3);

  log("[TopK] initial_topK:", initial_topK, "final_topK:", final_topK);

  // Generate embedding and run vector search
  const embedTtl = Math.max(86400, Math.min(31536000, Number(cfg.search?.embed_cache_ttl ?? 7776000)));
  const vector = await time("cachedEmbed", () =>
    cachedEmbed(env, cfg.ai.embed_model, cfg.vectorize.dims, query, ctx, cache.embedding, embedTtl, site)
  );

  // @ts-ignore Workers typing for VECTORIZE.query is permissive
  const res = await time("VECTORIZE.query", () =>
    env.VECTORIZE.query(vector, { topK: initial_topK, includeMetadata: true, returnMetadata: true })
  );
  let matches: SearchMatch[] = filterToSite(site, res?.matches || []);
  log("[VectorSearch] Fetched", matches.length, "results from vectorize");

  // Phase 2: If no keyword match, try metadata-based detection (semantic path)
  if (!detectedIntent && matches.length > 0) {
    detectedIntent = detectIntentFromResults(matches, customIntents);
    if (detectedIntent) {
      log("[Intent] Detected via metadata:", detectedIntent.name);
    }
  }

  // Use detected intent or fall back to default
  const intent = detectedIntent || getDefaultIntent();
  const behaviorName = intent.response_behavior || cfg.default_behavior || "long_form_answer";

  // Apply intent's final_topK override if set (post-detection refinement
  // — works whether the intent came from keyword or metadata detection).
  if (intent.final_topK !== undefined) {
    final_topK = Number(intent.final_topK);
    log("[TopK] final_topK overridden by intent:", intent.name, "->", final_topK);
  }

  // Extract keywords from query
  const queryKeywords = extractKeywords(query);
  log("[Keywords] Extracted from query:", JSON.stringify(queryKeywords));

  // Combine query keywords with intent keywords for comprehensive re-ranking
  const intentKeywords = (detectedIntent && detectedIntent.detection?.keywords) || [];
  const allKeywords = [...new Set([...queryKeywords, ...intentKeywords])]; // Deduplicate
  log("[Keywords] Combined (query + intent):", JSON.stringify(allKeywords));

  // THREE-PASS RE-RANKING SYSTEM
  if (matches.length > 0) {
    const originalScores = matches.slice(0, 3).map(m => ({
      title: (m.metadata as any)?.title,
      score: m.score.toFixed(3)
    }));

    // PASS 1: Metadata-only boosting (fast, 0 KV reads)
    const metadataMatches = (detectedIntent && detectedIntent.name !== "default")
      ? ((detectedIntent as any).detection?.metadata_matches || {})
      : {};

    matches = applyMetadataBoost(matches, metadataMatches);
    log("[Pass1:Metadata] Applied metadata boosting");

    // Slice to top 8 candidates for text fetching (saves KV reads)
    const candidateCount = Math.min(8, matches.length);
    const candidates = matches.slice(0, candidateCount);
    log("[Pass1:Metadata] Top", candidateCount, "candidates after metadata boost");

    // PASS 2: Fetch full text from KV for candidates
    const candidatesWithText = await time("fetchFullText", () =>
      fetchFullTextForMatches(env, candidates)
    );
    log("[Pass2:FetchText] Fetched full text for", candidatesWithText.length, "candidates");

    // PASS 3: Keyword boosting using full text (with combined query + intent keywords)
    const reranked = applyKeywordBoost(candidatesWithText, allKeywords);
    log("[Pass3:Keywords] Applied keyword boosting with", allKeywords.length, "combined keywords");

    const afterScores = reranked.slice(0, 3).map(m => ({
      title: (m.metadata as any)?.title,
      score: m.score.toFixed(3),
      metadataBoost: ((m as any)._metadataBoost || 0).toFixed(3),
      keywordBoost: ((m as any)._keywordBoost || 0).toFixed(3),
      totalBoost: ((m as any)._totalBoost || 0).toFixed(3)
    }));

    log("[Re-rank] Original:", JSON.stringify(originalScores));
    log("[Re-rank] Final:", JSON.stringify(afterScores));

    // Replace matches with reranked results
    matches = reranked;
  }

  // Slice to final_topK for behavior/LLM
  const finalMatches = matches.slice(0, Math.min(final_topK, matches.length));
  log("[Final] Passing", finalMatches.length, "results (from", matches.length, "candidates)");

  log("[Behavior] Using:", behaviorName, "for intent:", intent.name);

  return {
    matches: finalMatches,
    intent,
    behaviorName,
    queryKeywords,
    intentKeywords,
    allKeywords,
    initial_topK,
    final_topK,
  };
}
