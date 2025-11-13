import { json } from "../http/response";
import { log, startTimer, time } from "../lib/logging";
import { resolveCaching } from "../config/siteConfig";
import { cachedEmbed, filterToSite } from "../search/vectorize";
import { detectIntentFromQuery, detectIntentFromResults, getDefaultIntent } from "../search/detection";
import { extractKeywords, applyMetadataBoost, applyKeywordBoost } from "../search/rerank";
import { fetchFullTextForMatches } from "../utils/fullText";
import type { Env, ExecutionContext, SearchMatch, SiteConfig, CustomIntent } from "../lib/types";

/**
 * Vector search endpoint with preview/debug mode.
 * This endpoint mirrors the /ask logic but stops before calling the LLM.
 *
 * Purpose: Let users preview search results, re-ranking, and context
 * before executing the full /ask pipeline.
 */
export async function handleSearch(
  req: Request,
  env: Env,
  cfg: SiteConfig,
  ctx?: ExecutionContext
): Promise<Response> {
  const stop = startTimer("handleSearch");
  const url = new URL(req.url);
  const site = (url.searchParams.get("site") || "").trim();
  const q = (url.searchParams.get("q") || "").trim();
  const wantDebug = (url.searchParams.get("debug") || "") === "1";
  const wantCaching = resolveCaching(url, cfg);

  if (!site) {
    stop();
    return json({ ok: false, error: "Missing ?site=" }, { status: 400 });
  }
  if (!q) {
    stop();
    return json({ ok: false, error: "Missing ?q=" }, { status: 400 });
  }

  // Use same two-stage filtering as /ask
  const initial_topK = Number(cfg.search?.initial_topK ?? 15);
  const final_topK = Number(cfg.search?.final_topK ?? 3);

  // Get custom intents from config
  const customIntents = cfg.custom_intents || [];

  // Phase 1: Keyword-based intent detection
  let detectedIntent: CustomIntent | null = detectIntentFromQuery(q, customIntents);
  const detectionMethod = detectedIntent ? 'keyword' : null;

  // Fetch vector embedding and query vectorize
  const vector = await time("cachedEmbed(search)", () =>
    cachedEmbed(env, cfg.ai.embed_model, cfg.vectorize.dims, q, ctx, wantCaching)
  );

  // @ts-ignore Cloudflare Workers types are permissive
  const out = await time("VECTORIZE.query(search)", () =>
    env.VECTORIZE.query(vector, { topK: initial_topK, returnMetadata: true })
  );
  let matches = filterToSite(site, (out?.matches || []) as SearchMatch[]);

  // Phase 2: Metadata-based intent detection (if no keyword match)
  if (!detectedIntent && matches.length > 0) {
    detectedIntent = detectIntentFromResults(matches, customIntents);
  }

  const intent = detectedIntent || getDefaultIntent();
  const behaviorName = intent.response_behavior || cfg.default_behavior || "long_form_answer";

  // Extract keywords
  const queryKeywords = extractKeywords(q);

  // THREE-PASS RE-RANKING (same as /ask)
  const beforeRank = matches.slice(0, 5).map(m => ({
    title: (m.metadata as any)?.title,
    score: m.score.toFixed(3)
  }));

  // PASS 1: Metadata-only boosting (fast, 0 KV reads)
  const metadataMatches = (detectedIntent && detectedIntent.name !== "default")
    ? ((detectedIntent as any).detection?.metadata_matches || {})
    : {};

  matches = applyMetadataBoost(matches, metadataMatches);

  // Slice to top 8 candidates for text fetching
  const candidateCount = Math.min(8, matches.length);
  const candidates = matches.slice(0, candidateCount);

  // PASS 2: Fetch full text from KV for candidates
  const candidatesWithText = await time("fetchFullText", () =>
    fetchFullTextForMatches(env, candidates)
  );

  // PASS 3: Keyword boosting using full text
  const reranked = applyKeywordBoost(candidatesWithText, queryKeywords);

  const afterRank = reranked.slice(0, 5).map(m => ({
    title: (m.metadata as any)?.title,
    score: m.score.toFixed(3),
    metadataBoost: ((m as any)._metadataBoost || 0).toFixed(3),
    keywordBoost: ((m as any)._keywordBoost || 0).toFixed(3),
    totalBoost: ((m as any)._totalBoost || 0).toFixed(3),
    boostDetails: (m as any)._boostDetails
  }));

  // Replace matches with reranked results
  matches = reranked;

  // Slice to final results
  const finalMatches = matches.slice(0, Math.min(final_topK, matches.length));

  // Build response with detailed pipeline info
  const body: Record<string, unknown> = {
    ok: true,
    site,
    q,
    initial_topK,
    final_topK,
    total_results: matches.length,
    keywords: queryKeywords,
    intent: {
      name: intent.name,
      priority: intent.priority,
      behavior: behaviorName,
      detection_method: detectionMethod || 'default',
    },
    pipeline: {
      stage1_vectorize: {
        description: "Fetch results from vectorize (vector similarity)",
        fetched: initial_topK,
        top5_before_rerank: beforeRank,
      },
      stage2_metadata_boost: {
        description: "Apply intent metadata boosting (collection, path, page_kind)",
        candidates_selected_for_text_fetch: candidateCount,
        metadata_rules_applied: Object.keys(metadataMatches).length > 0,
      },
      stage3_fetch_text: {
        description: "Fetch full KV text for top candidates",
        kv_reads: candidateCount,
      },
      stage4_keyword_boost: {
        description: "Apply keyword boosting using full text content",
        keywords_applied: queryKeywords,
        top5_after_rerank: afterRank,
      },
      stage5_final: {
        description: "Select final results for LLM context",
        selected: finalMatches.length,
      }
    },
    results: finalMatches.map(m => ({
      id: m.id,
      score: m.score,
      original_score: (m as any)._originalScore,
      boost_breakdown: {
        metadata_boost: (m as any)._metadataBoost || 0,
        keyword_boost: (m as any)._keywordBoost || 0,
        total_boost: (m as any)._totalBoost || 0,
      },
      boost_details: (m as any)._boostDetails || [],
      title: (m.metadata as any)?.title,
      url: (m.metadata as any)?.url,
      preview: (m.metadata as any)?.preview,
      collection: (m.metadata as any)?.collection,
      page_kind: (m.metadata as any)?.page_kind,
      path: (m.metadata as any)?.path,
      has_full_text: !!(m as any)._fullText,
      full_text_length: (m as any)._fullText ? (m as any)._fullText.length : 0,
    })),
  };

  if (wantDebug) {
    body._debug = {
      all_results: matches.map(m => ({
        title: (m.metadata as any)?.title,
        score: m.score.toFixed(3),
        boost: (m as any)._boost?.toFixed(3),
      }))
    };
  }

  stop();
  return json(body);
}
