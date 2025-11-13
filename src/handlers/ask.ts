import { json } from "../http/response";
import { log, startTimer, time } from "../lib/logging";
import { clampTemperature, clampTopK, resolveCaching } from "../config/siteConfig";
import { detectIntentFromQuery, detectIntentFromResults, getDefaultIntent } from "../search/detection";
import { getBehavior } from "../behaviors";
import type { BehaviorResponse } from "../behaviors";
import { ensureMarkdown } from "../search/context";
import { cachedEmbed, filterToSite } from "../search/vectorize";
import { sha1Hex } from "../utils/crypto";
import { isNoAnswer } from "../utils/isNoAnswer";
import { extractKeywords, applyMetadataBoost, applyKeywordBoost } from "../search/rerank";
import { fetchFullTextForMatches } from "../utils/fullText";
import type { Env, ExecutionContext, SiteConfig, SearchMatch, CustomIntent } from "../lib/types";


/**
 * Conversation endpoint for WORX AI using the new behavior system.
 *
 * Flow:
 * 1. Detect intent from query (keyword-based, fast path)
 * 2. Run vector search
 * 3. Detect intent from results (metadata-based, semantic path)
 * 4. Re-rank results based on intent metadata (if detected)
 * 5. Execute appropriate behavior
 * 6. Cache response if enabled
 */
export async function handleAsk(
  req: Request,
  env: Env,
  cfg: SiteConfig,
  ctx?: ExecutionContext
): Promise<Response> {
  const stop = startTimer("handleAsk");
  const url = new URL(req.url);
  const wantDebug = (url.searchParams.get("debug") || "") === "1";
  const wantCaching = resolveCaching(url, cfg);

  const body = (await req.json().catch(() => ({}))) as {
    site?: string;
    q?: string;
    k?: number;
    systemPrompt?: string;
    system_prompt?: string;
    chatTemperature?: number;
    temperature?: number;
    chat_temperature?: number;
  };

  const site = (body.site || "").trim();
  const q = (body.q || "").trim();

  if (!site) {
    stop();
    return json({ ok: false, error: "Missing field 'site'" }, { status: 400 });
  }
  if (!q) {
    stop();
    return json({ ok: false, error: "Missing field 'q'" }, { status: 400 });
  }

  // Get custom intents from config
  const customIntents = cfg.custom_intents || [];

  // Phase 1: Try keyword-based detection (fast path)
  let detectedIntent: CustomIntent | null = detectIntentFromQuery(q, customIntents);
  if (detectedIntent) {
    log("[Intent] Detected via keywords:", detectedIntent.name);
  }

  // Prepare vector search with two-stage filtering
  // Stage 1: Fetch many results from vectorize for re-ranking
  const initial_topK = Number(cfg.search?.initial_topK ?? 15);
  // Stage 2: After re-ranking, pass fewer results to behavior
  const final_topK = Number(cfg.search?.final_topK ?? 3);

  log("[TopK] initial_topK:", initial_topK, "final_topK:", final_topK);

  const vector = await time("cachedEmbed(ask)", () =>
    cachedEmbed(env, cfg.ai.embed_model, cfg.vectorize.dims, q, ctx, wantCaching)
  );

  const runQuery = async (topK: number) => {
    // @ts-ignore Workers typing for VECTORIZE.query is permissive
    const res = await time("VECTORIZE.query(ask)", () =>
      env.VECTORIZE.query(vector, { topK, includeMetadata: true, returnMetadata: true })
    );
    return filterToSite(site, res?.matches || []);
  };

  // Run vector search - fetch initial_topK for re-ranking
  let matches: SearchMatch[] = await runQuery(initial_topK);
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

  // Extract keywords from query for re-ranking
  const queryKeywords = extractKeywords(q);
  log("[Keywords] Extracted from query:", JSON.stringify(queryKeywords));

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

    // PASS 3: Keyword boosting using full text
    const reranked = applyKeywordBoost(candidatesWithText, queryKeywords);
    log("[Pass3:Keywords] Applied keyword boosting with full text");

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

  // Phase 3: Slice to final_topK for behavior/LLM
  const finalMatches = matches.slice(0, Math.min(final_topK, matches.length));
  log("[Final] Passing", finalMatches.length, "results to behavior (from", matches.length, "candidates)");

  log("[Behavior] Using:", behaviorName, "for intent:", intent.name);

  // Check cache before executing behavior
  const ansKeyRaw = JSON.stringify({
    site,
    q,
    initial_k: initial_topK,
    final_k: final_topK,
    behavior: behaviorName,
    intent: intent.name,
    customPromptHash: intent.system_prompt ? await sha1Hex(intent.system_prompt) : null,
  });
  const ansKey = `ans:${await sha1Hex(ansKeyRaw)}`;

  if (wantCaching) {
    try {
      const cachedRaw = await env.WORX_AI_CONFIG.get<string>(ansKey, "text");
      if (cachedRaw && cachedRaw.trim()) {
        const cachedResponse = JSON.parse(cachedRaw) as BehaviorResponse;
        log("[cachedAnswer] HIT", ansKey);

        const nowIso = new Date().toISOString();
        const noAnswer = cachedResponse.answer ? isNoAnswer(cachedResponse.answer) : false;
        const foundIndex = matches.length > 0 && !noAnswer;

        const stats = {
          question: q,
          found_index: foundIndex,
          cached: true,
          model: cachedResponse.model || null,
          tokens_input: null,
          tokens_output: null,
          total_tokens: null,
          timestamp: nowIso,
          temperature: cachedResponse.temperature || null,
          intent: intent.name,
          behavior: behaviorName,
        };

        const bodyOut: Record<string, unknown> = {
          ok: true,
          q,
          k: final_topK,
          initial_k: initial_topK,
          ...cachedResponse, // Include answer, blurb, concreteDirective, etc.
          stats,
        };

        if (wantDebug) {
          bodyOut._debug = {
            matches: finalMatches.slice(0, 3),
            allMatches: matches.slice(0, 5),
            cache: "hit",
            intent: intent.name
          };
        }

        const response = json(bodyOut);
        stop();
        return response;
      }
    } catch (error) {
      log("[cachedAnswer] READ-ERROR", ansKey, String((error as Error)?.message || error));
    }
  }

  // Execute behavior with sliced results
  const behavior = getBehavior(behaviorName);
  const behaviorResponse = await time("behavior.execute", () =>
    behavior.execute({
      query: q,
      matches: finalMatches,
      intent,
      config: cfg,
      env,
    })
  );

  // Cache the response
  const ansTtl = Math.max(60, Math.min(86400, Number(cfg.search?.answer_cache_ttl ?? 600)));
  if (wantCaching) {
    try {
      const cacheValue = JSON.stringify(behaviorResponse);
      if (ctx?.waitUntil && env.WORX_AI_CONFIG.put) {
        ctx.waitUntil(env.WORX_AI_CONFIG.put(ansKey, cacheValue, { expirationTtl: ansTtl }));
        log("[cachedAnswer] STORE-QUEUED", ansKey, `ttl=${ansTtl}`);
      } else if (env.WORX_AI_CONFIG.put) {
        await env.WORX_AI_CONFIG.put(ansKey, cacheValue, { expirationTtl: ansTtl });
        log("[cachedAnswer] STORED", ansKey, `ttl=${ansTtl}`);
      }
    } catch (error) {
      log("[cachedAnswer] STORE-ERROR", ansKey, String((error as Error)?.message || error));
    }
  }

  // Build response
  const nowIso = new Date().toISOString();
  const noAnswer = behaviorResponse.answer ? isNoAnswer(behaviorResponse.answer) : false;
  const foundIndex = matches.length > 0 && !noAnswer;

  const stats = {
    question: q,
    found_index: foundIndex,
    cached: false,
    model: behaviorResponse.model || null,
    tokens_input: behaviorResponse.tokens_input || null,
    tokens_output: behaviorResponse.tokens_output || null,
    total_tokens: behaviorResponse.total_tokens || null,
    timestamp: nowIso,
    temperature: behaviorResponse.temperature || null,
    intent: intent.name,
    behavior: behaviorName,
  };

  const bodyOut: Record<string, unknown> = {
    ok: true,
    q,
    k: final_topK, // Number of results passed to behavior
    initial_k: initial_topK, // Number of results fetched from vectorize
    ...behaviorResponse, // Include answer, blurb, concreteDirective, sources, etc.
    stats,
  };

  if (wantDebug) {
    bodyOut._debug = {
      matches: finalMatches.slice(0, 3), // Show the final matches passed to behavior
      allMatches: matches.slice(0, 5), // Show top 5 from full set for comparison
      cache: "miss",
      intent: intent.name,
      behavior: behaviorName,
      detectionMethod: detectedIntent ? (customIntents.indexOf(detectedIntent) >= 0 ? "custom" : "default") : "default",
    };
  }

  const response = json(bodyOut);
  stop();
  return response;
}
