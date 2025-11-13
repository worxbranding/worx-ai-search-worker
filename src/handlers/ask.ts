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
import type { Env, ExecutionContext, SiteConfig, SearchMatch, CustomIntent } from "../lib/types";

/**
 * Re-rank search results by boosting matches that align with intent metadata.
 * This ensures that when an intent is detected, the most relevant page is selected
 * even if vector search didn't rank it highest.
 */
function reRankByIntentMetadata(matches: SearchMatch[], metadataMatches: Record<string, any>): SearchMatch[] {
  return matches
    .map((match) => {
      const metadata = (match.metadata || {}) as Record<string, any>;
      let boost = 0;

      // Boost if collection matches
      if (metadataMatches.collection && metadata.collection === metadataMatches.collection) {
        boost += 0.1;
      }

      // Boost if page_kind matches
      if (metadataMatches.page_kind && metadata.page_kind === metadataMatches.page_kind) {
        boost += 0.05;
      }

      // Boost if path EXACTLY matches expected prefix (strongest signal)
      if (metadataMatches.path_starts_with && typeof metadata.path === 'string') {
        if (metadata.path === metadataMatches.path_starts_with) {
          // Exact path match - this is almost certainly the right page
          boost += 0.5;
        } else if (metadata.path.startsWith(metadataMatches.path_starts_with)) {
          // Path starts with - still strong
          boost += 0.3;
        }
      }

      // Boost if title contains expected terms
      if (metadataMatches.title_contains && Array.isArray(metadataMatches.title_contains) && typeof metadata.title === 'string') {
        const titleLower = metadata.title.toLowerCase();
        for (const term of metadataMatches.title_contains) {
          if (titleLower.includes(String(term).toLowerCase())) {
            boost += 0.15;
            break; // Only boost once for title match
          }
        }
      }

      // Return match with adjusted score
      return {
        ...match,
        score: match.score + boost,
      };
    })
    .sort((a, b) => b.score - a.score); // Re-sort by adjusted score
}

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

  // Prepare vector search
  const baseTopK = clampTopK(Number(body.k || cfg.search?.topK || 6));
  // Fetch more results when intent is detected to ensure target page is included
  const initialK = detectedIntent ? Math.min(10, baseTopK + 5) : Math.min(baseTopK, 6);

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

  // Run vector search (fetch more results if intent was detected)
  let matches: SearchMatch[] = await runQuery(initialK);

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

  // Re-rank matches ONLY if intent was detected by keywords (not by metadata)
  // This prevents false positives where metadata matches trigger wrong intents
  if (detectedIntent && detectedIntent.name !== "default" && matches.length > 0 && (detectedIntent as any).detection?.metadata_matches) {
    const beforeScores = matches.slice(0, 3).map(m => ({ path: (m.metadata as any)?.path, score: m.score }));
    matches = reRankByIntentMetadata(matches, (detectedIntent as any).detection.metadata_matches);
    const afterScores = matches.slice(0, 3).map(m => ({ path: (m.metadata as any)?.path, score: m.score }));
    log("[Re-rank] Before:", JSON.stringify(beforeScores));
    log("[Re-rank] After:", JSON.stringify(afterScores));
    log("[Re-rank] Boosted matches based on intent metadata for:", detectedIntent.name);
  }

  log("[Behavior] Using:", behaviorName, "for intent:", intent.name);

  // Check cache before executing behavior
  const ansKeyRaw = JSON.stringify({
    site,
    q,
    k: initialK,
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
          k: initialK,
          ...cachedResponse, // Include answer, blurb, concreteDirective, etc.
          stats,
        };

        if (wantDebug) {
          bodyOut._debug = { matches: matches.slice(0, 3), cache: "hit", intent: intent.name };
        }

        const response = json(bodyOut);
        stop();
        return response;
      }
    } catch (error) {
      log("[cachedAnswer] READ-ERROR", ansKey, String((error as Error)?.message || error));
    }
  }

  // Execute behavior
  const behavior = getBehavior(behaviorName);
  const behaviorResponse = await time("behavior.execute", () =>
    behavior.execute({
      query: q,
      matches,
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
    k: initialK,
    ...behaviorResponse, // Include answer, blurb, concreteDirective, sources, etc.
    stats,
  };

  if (wantDebug) {
    bodyOut._debug = {
      matches: matches.slice(0, 3),
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
