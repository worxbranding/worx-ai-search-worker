import type { Env, ExecutionContext, SiteConfig, SearchMatch, CustomIntent } from "../lib/types";
import { log, time } from "../lib/logging";
import { cachedEmbed, filterToSite } from "./vectorize";
import { detectIntent, getDefaultIntent } from "./detection";
import { extractKeywords, applyMetadataBoost, applyKeywordBoost } from "./rerank";
import { fetchFullTextForMatches } from "../utils/fullText";
import { detectionValue } from "../lib/configDefaults";

/**
 * Shared search pipeline used by both /search and /ask endpoints.
 *
 * Flow:
 * 1. Embed the query and run vector search.
 * 2. Not-Found short-circuit when results are empty / low-spread.
 * 3. Single-pass scored intent detection (no priority list).
 * 4. Three-pass re-ranking (metadata → fetch full text → keywords).
 * 5. Slice to final_topK for the answer behavior.
 */
export async function executeSearchPipeline(
  query: string,
  site: string,
  cfg: SiteConfig,
  env: Env,
  ctx: ExecutionContext | undefined,
  cache: { answer: boolean; embedding: boolean },
  forceIntent?: string
): Promise<{
  matches: SearchMatch[];
  intent: CustomIntent;
  behaviorName: string;
  queryKeywords: string[];
  intentKeywords: string[];
  allKeywords: string[];
  initial_topK: number;
  final_topK: number;
  /** Detection telemetry surfaced to /search and /ask for tuning. */
  detection?: {
    reason: string;
    top_vector_score: number;
    score?: number;
    threshold?: number;
    margin?: number;
    top_intents?: Array<{ name: string; score: number; components: { embedding: number; keyword: number; metadata: number } }>;
  };
}> {
  const customIntents = cfg.custom_intents || [];

  // Route-lock from caller (Training tab "Lock to <intent>" toggle).
  let detectedIntent: CustomIntent | null = null;
  if (forceIntent) {
    const forced = customIntents.find((i) => i.name === forceIntent);
    if (forced) {
      detectedIntent = forced;
      log("[Intent] Forced by request:", forced.name);
    } else {
      log("[Intent] force_intent did not match any intent:", forceIntent);
    }
  }

  // Pre-search topK. Without a pre-search keyword phase we can't know the
  // routed intent before fetching, so we take the max of the cfg default
  // and any per-intent override — fetch a little extra rather than miss
  // candidates that an intent-specific override would have requested.
  const intentMaxInitial = customIntents
    .filter((i) => i.enabled !== false && i.initial_topK !== undefined)
    .reduce((m, i) => Math.max(m, Number(i.initial_topK)), 0);
  const initial_topK = Math.max(intentMaxInitial, Number(cfg.search?.initial_topK ?? 15));
  let final_topK = Number(cfg.search?.final_topK ?? 3);

  log("[TopK] initial_topK:", initial_topK, "final_topK:", final_topK);

  // Embed once; reused for both vector search and intent scoring.
  const embedTtl = Math.max(86400, Math.min(31536000, Number(cfg.search?.embed_cache_ttl ?? 7776000)));
  const vector = await time("cachedEmbed", () =>
    cachedEmbed(env, cfg.ai.embed_model, cfg.vectorize.dims, query, ctx, cache.embedding, embedTtl, site)
  );

  // @ts-ignore Workers typing for VECTORIZE.query is permissive
  const res = await time("VECTORIZE.query", () =>
    env.VECTORIZE.query(vector, { topK: initial_topK, includeMetadata: true, returnMetadata: true })
  );
  let matches: SearchMatch[] = filterToSite(site, res?.matches || []);
  const topVectorScore = matches.length > 0 ? Number(matches[0]?.score ?? 0) : 0;
  log("[VectorSearch] Fetched", matches.length, "results from vectorize, top=", topVectorScore.toFixed(4));

  // Content-existence floor on the *raw* top vector score. BGE produces
  // ~0.6–0.75 for genuinely relevant content and ~0.4–0.55 for nonsense.
  const contentFloor = Number(cfg.search?.not_found_threshold ?? 0.6);
  const noRelevantContent = matches.length === 0 || topVectorScore < contentFloor;

  // Run detection. Tiered confidence rule:
  //   - HIGH_CONFIDENCE detection wins regardless of content (pricing, etc.)
  //   - Lower-confidence detection only wins if content also passes the
  //     floor — otherwise we fall through to Not Found
  // This stops fuzzy BGE similarity from letting nonsense queries pick
  // the closest intent when there's no real content backing.
  // Detection score above this is trusted regardless of vector content.
  // Tuned so genuine high-quality intent matches (clear keyword + strong
  // embedding) survive without indexed content (Pricing, Team Members on
  // a thinly-indexed CEO page) while fuzzy junk-vs-Services matches —
  // which after metadata halving cap at ~0.59 — don't.
  // Configurable via cfg.search.detection.high_confidence_score.
  const HIGH_CONFIDENCE = detectionValue(cfg, 'high_confidence_score');
  let detectionReason: string | null = null;
  let detectionScore = 0;
  let detection: ReturnType<typeof detectIntent> = { intent: null, reason: "no_intents" };
  if (!detectedIntent) {
    // Short openers like "Who are you?" / "What is WORX?" can't carry enough
    // signal to clear the standard threshold even when an intent's examples
    // include them — there's just not enough text to embed against. Relax
    // the bar for ≤4-word queries so they don't fall through to Default
    // when an intent is plainly relevant.
    const wordCount = query.trim().split(/\s+/).filter(Boolean).length;
    const isShort = wordCount > 0 && wordCount <= 4;
    const shortQueryThreshold = isShort ? 0.45 : undefined;
    // Short queries also tend to tie multiple intents within ~0.01 because
    // there isn't enough text to break a tie. Drop the ambiguity margin so
    // a near-tie doesn't blackhole an obviously-relevant intent.
    const shortQueryMargin = isShort ? 0.001 : undefined;
    const cfgThreshold = cfg.search?.intent_embedding_threshold;
    detection = detectIntent(query, vector as unknown as number[], matches, customIntents, {
      threshold: shortQueryThreshold ?? cfgThreshold,
      margin: shortQueryMargin,
    });
    detectionReason = detection.reason;
    detectionScore = detection.score ?? 0;
    log(
      "[IntentDetect] reason=", detection.reason,
      "topScore=", detection.score?.toFixed(4),
      "components=", JSON.stringify(detection.components),
      "top=", JSON.stringify(detection.top_intents)
    );
    if (detection.intent) {
      const highConf = detectionScore >= HIGH_CONFIDENCE;
      if (highConf || !noRelevantContent) {
        detectedIntent = detection.intent;
        log("[Intent] Routed to:", detectedIntent.name, highConf ? "(high-confidence)" : "(content-backed)");
      } else {
        log("[Intent] Borderline detection (", detectionScore.toFixed(3), ") with no content — falling through");
      }
    }
  }

  // No-confidence routing. Falls through to Not Found when:
  //   (1) no intent picked AND vector content also weak
  //   (2) detection returned below_threshold (no intent above floor)
  //   (3) borderline detection that didn't survive the content check
  // Ambiguous detection (have content, no clear intent) falls through to
  // admin's Default intent.
  let intent: CustomIntent | null = detectedIntent;
  if (!intent && !forceIntent) {
    const wantsNotFound = noRelevantContent || detectionReason === "below_threshold";
    if (wantsNotFound) {
      const notFound = customIntents.find((i) => i.is_system === true && i.name === "Not Found");
      if (notFound) {
        log("[Intent] Routing to Not Found (", noRelevantContent ? "no_relevant_content" : detectionReason, ")");
        return {
          matches: [],
          intent: notFound,
          behaviorName: notFound.response_behavior || "short_answer",
          queryKeywords: [],
          intentKeywords: [],
          allKeywords: [],
          initial_topK,
          final_topK: 0,
          detection: {
            reason: detectionReason || "no_relevant_content",
            top_vector_score: topVectorScore,
            score: detectionScore,
            threshold: detection.threshold,
            margin: detection.margin,
            top_intents: detection.top_intents,
          },
        };
      }
    }
  }

  // Fall back to admin's Default system intent (real KV-backed entry with
  // its own system_prompt + behavior) before the synthesized fallback.
  if (!intent) {
    const adminDefault = customIntents.find((i) => i.is_system === true && i.name === "Default");
    intent = adminDefault || getDefaultIntent(cfg.default_behavior);
  }
  const behaviorName = intent.response_behavior || cfg.default_behavior || "long_form_answer";

  // Per-intent final_topK override (post-detection refinement).
  if (intent.final_topK !== undefined) {
    final_topK = Number(intent.final_topK);
    log("[TopK] final_topK overridden by intent:", intent.name, "->", final_topK);
  }

  const queryKeywords = extractKeywords(query);
  log("[Keywords] Extracted from query:", JSON.stringify(queryKeywords));

  const intentKeywords = (detectedIntent && detectedIntent.detection?.keywords) || [];
  const allKeywords = [...new Set([...queryKeywords, ...intentKeywords])];
  log("[Keywords] Combined (query + intent):", JSON.stringify(allKeywords));

  // THREE-PASS RE-RANKING SYSTEM
  if (matches.length > 0) {
    const originalScores = matches.slice(0, 3).map((m) => ({
      title: (m.metadata as any)?.title,
      score: m.score.toFixed(3),
    }));

    const metadataMatches = (detectedIntent && detectedIntent.name !== "default")
      ? ((detectedIntent as any).detection?.metadata_matches || {})
      : {};
    matches = applyMetadataBoost(matches, metadataMatches, queryKeywords, cfg);
    log("[Pass1:Metadata] Applied metadata + query-token-overlap boosting");

    // Candidate slice for the keyword/full-text pass. Configurable via
    // cfg.search.detection.candidate_slice. Smaller = faster but may miss
    // relevant pages whose raw embedding rank is borderline.
    const candidateSlice = detectionValue(cfg, 'candidate_slice');
    const candidateCount = Math.min(candidateSlice, matches.length);
    const candidates = matches.slice(0, candidateCount);
    log("[Pass1:Metadata] Top", candidateCount, "candidates after metadata boost");

    const candidatesWithText = await time("fetchFullText", () =>
      fetchFullTextForMatches(env, candidates)
    );
    log("[Pass2:FetchText] Fetched full text for", candidatesWithText.length, "candidates");

    const reranked = applyKeywordBoost(candidatesWithText, allKeywords, cfg);
    log("[Pass3:Keywords] Applied keyword boosting with", allKeywords.length, "combined keywords");

    const afterScores = reranked.slice(0, 3).map((m) => ({
      title: (m.metadata as any)?.title,
      score: m.score.toFixed(3),
      metadataBoost: ((m as any)._metadataBoost || 0).toFixed(3),
      keywordBoost: ((m as any)._keywordBoost || 0).toFixed(3),
      totalBoost: ((m as any)._totalBoost || 0).toFixed(3),
    }));

    log("[Re-rank] Original:", JSON.stringify(originalScores));
    log("[Re-rank] Final:", JSON.stringify(afterScores));

    matches = reranked;
  }

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
    detection: {
      reason: detectionReason || "matched",
      top_vector_score: topVectorScore,
      score: detectionScore,
      threshold: detection.threshold,
      margin: detection.margin,
      top_intents: detection.top_intents,
    },
  };
}
