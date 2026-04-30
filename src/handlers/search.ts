import { json } from "../http/response";
import { startTimer } from "../lib/logging";
import { resolveCaching, type InBandRequestBody } from "../config/siteConfig";
import { executeSearchPipeline } from "../search/pipeline";
import type { Env, ExecutionContext, SiteConfig } from "../lib/types";

/**
 * Vector search endpoint with preview/debug mode.
 * Mirrors /ask but stops before calling the LLM. Body has already been
 * HMAC-verified and parsed by the entry point.
 */
export async function handleSearch(
  req: Request,
  env: Env,
  cfg: SiteConfig,
  body: InBandRequestBody,
  ctx?: ExecutionContext
): Promise<Response> {
  const stop = startTimer("handleSearch");
  const url = new URL(req.url);
  const site = (body.site || cfg.site_key || "").trim();
  const q = (body.q || url.searchParams.get("q") || "").trim();
  const wantCaching = resolveCaching(url, cfg);

  if (!site) {
    stop();
    return json({ ok: false, error: "Missing ?site=" }, { status: 400 });
  }
  if (!q) {
    stop();
    return json({ ok: false, error: "Missing ?q=" }, { status: 400 });
  }

  // Execute shared search pipeline (same logic as /ask endpoint)
  const {
    matches: finalMatches,
    intent,
    behaviorName,
    queryKeywords,
    intentKeywords,
    allKeywords,
    initial_topK,
    final_topK,
  } = await executeSearchPipeline(q, site, cfg, env, ctx, wantCaching);

  // Build response with detailed pipeline info
  const responseBody: Record<string, unknown> = {
    ok: true,
    site,
    q,
    initial_topK,
    final_topK,
    total_results: finalMatches.length,
    keywords: {
      query_keywords: queryKeywords,
      intent_keywords: intentKeywords,
      combined_keywords: allKeywords,
    },
    intent: {
      name: intent.name,
      priority: intent.priority,
      behavior: behaviorName,
    },
    pipeline: {
      description: "Search + re-ranking pipeline (shared with /ask endpoint)",
      stage1: "Detect intent from query keywords",
      stage2: "Run vector search (fetch initial_topK)",
      stage3: "Detect intent from result metadata",
      stage4: "Metadata boosting (collection, path, page_kind)",
      stage5: "Fetch full text for top candidates",
      stage6: "Keyword boosting using query + intent keywords",
      stage7: "Select final_topK for response",
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

  stop();
  return json(responseBody);
}
