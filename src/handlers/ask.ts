import { json } from "../http/response";
import { log, startTimer, time } from "../lib/logging";
import { resolveCaching, type InBandRequestBody } from "../config/siteConfig";
import { getBehavior } from "../behaviors";
import type { BehaviorResponse } from "../behaviors";
import { sha1Hex } from "../utils/crypto";
import { isNoAnswer } from "../utils/isNoAnswer";
import { ensureSourceLink } from "../utils/ensureSourceLink";
import { providerForModel } from "../utils/providerForModel";
import { executeSearchPipeline } from "../search/pipeline";
import { mergeIntentTuning } from "../lib/llm";
import { buildSystemPrompt } from "../lib/prompts";
import type { Env, ExecutionContext, SiteConfig } from "../lib/types";


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
 *
 * Body has already been HMAC-verified and JSON-parsed by the entry point.
 */
export async function handleAsk(
  req: Request,
  env: Env,
  cfg: SiteConfig,
  body: InBandRequestBody,
  ctx?: ExecutionContext
): Promise<Response> {
  const stop = startTimer("handleAsk");
  const url = new URL(req.url);
  const wantDebug = (url.searchParams.get("debug") || "") === "1";
  const cache = resolveCaching(url, cfg, body);

  const site = (body.site || cfg.site_key || "").trim();
  const q = (body.q || "").trim();

  if (!site) {
    stop();
    return json({ ok: false, error: "Missing field 'site'" }, { status: 400 });
  }
  if (!q) {
    stop();
    return json({ ok: false, error: "Missing field 'q'" }, { status: 400 });
  }

  // Execute shared search pipeline (same logic as /search endpoint).
  // body.force_intent (if any) skips intent detection — used by the
  // Training tab's "Lock to <intent>" toggle so iteration stays anchored.
  const {
    matches: finalMatches,
    intent,
    behaviorName,
    initial_topK,
    final_topK,
  } = await executeSearchPipeline(q, site, cfg, env, ctx, cache, body.force_intent);

  // Not-Found short-circuit. The pipeline routes to the Not Found system
  // intent when vector search has nothing relevant. Emit its system_prompt
  // verbatim — no LLM call, no token spend, deterministic voice.
  if (intent.is_system === true && intent.name === "Not Found") {
    const cannedAnswer = (intent.system_prompt || "I couldn't find that in WORX content. Try rephrasing your question.").trim();
    const stats = {
      question: q,
      found_index: false,
      cached: false,
      provider: null,
      model: null,
      tokens_input: 0,
      tokens_output: 0,
      total_tokens: 0,
      timestamp: new Date().toISOString(),
      temperature: null,
      intent: intent.name,
      behavior: behaviorName,
    };
    const bodyOut: Record<string, unknown> = {
      ok: true,
      q,
      k: final_topK,
      initial_k: initial_topK,
      answer: cannedAnswer,
      stats,
    };
    if (wantDebug) {
      bodyOut._debug = { matches: [], cache: "miss", intent: intent.name, short_circuit: "not_found" };
    }
    const response = json(bodyOut);
    stop();
    return response;
  }

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

  if (cache.answer) {
    try {
      const cachedRaw = await env.CACHE.get<string>(ansKey, "text");
      if (cachedRaw && cachedRaw.trim()) {
        const cachedResponse = JSON.parse(cachedRaw) as BehaviorResponse;
        log("[cachedAnswer] HIT", ansKey);

        const nowIso = new Date().toISOString();
        const noAnswer = cachedResponse.answer ? isNoAnswer(cachedResponse.answer) : false;
        const foundIndex = finalMatches.length > 0 && !noAnswer;

        const stats = {
          question: q,
          found_index: foundIndex,
          cached: true,
          provider: providerForModel(cachedResponse.model || null),
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

  // Execute behavior with sliced results. Per-intent tuning overrides
  // (chat_temperature, topK, max_*, etc.) are merged into config.search
  // here so behaviors can keep reading config.search.* without caring
  // whether the value came from the intent or the site row.
  const cfgForBehavior = mergeIntentTuning(cfg, intent);
  const behavior = getBehavior(behaviorName);
  const behaviorResponse = await time("behavior.execute", () =>
    behavior.execute({
      query: q,
      matches: finalMatches,
      intent,
      config: cfgForBehavior,
      env,
    })
  );

  // Drive traffic: every answer should link the user to the page it cites.
  // Small models sometimes skip the link even when prompted to include one,
  // so guarantee at least one link by appending one if the answer doesn't
  // already contain a markdown link to one of the sources. Skip when:
  //   - The behavior produced a `concreteDirective` (the CMS will render a
  //     full list of links separately — adding another would be redundant).
  //   - There are no sources to link to.
  //   - The answer is the canned "no information" / Not Found response.
  if (
    behaviorResponse.answer &&
    !behaviorResponse.concreteDirective &&
    Array.isArray(behaviorResponse.sources) &&
    behaviorResponse.sources.length > 0 &&
    !isNoAnswer(behaviorResponse.answer)
  ) {
    behaviorResponse.answer = ensureSourceLink(behaviorResponse.answer, behaviorResponse.sources);
  }

  // Cache the response
  // Default: 30 days (2592000s). Will be cleared on ingest for fresh answers.
  const ansTtl = Math.max(60, Math.min(31536000, Number(cfg.search?.answer_cache_ttl ?? 2592000)));
  if (cache.answer) {
    try {
      const cacheValue = JSON.stringify(behaviorResponse);
      if (ctx?.waitUntil && env.CACHE.put) {
        ctx.waitUntil(env.CACHE.put(ansKey, cacheValue, { expirationTtl: ansTtl, metadata: { site } }));
        log("[cachedAnswer] STORE-QUEUED", ansKey, `ttl=${ansTtl}`, `site=${site}`);
      } else if (env.CACHE.put) {
        await env.CACHE.put(ansKey, cacheValue, { expirationTtl: ansTtl, metadata: { site } });
        log("[cachedAnswer] STORED", ansKey, `ttl=${ansTtl}`, `site=${site}`);
      }
    } catch (error) {
      log("[cachedAnswer] STORE-ERROR", ansKey, String((error as Error)?.message || error));
    }
  }

  // Build response
  const nowIso = new Date().toISOString();
  const noAnswer = behaviorResponse.answer ? isNoAnswer(behaviorResponse.answer) : false;
  const foundIndex = finalMatches.length > 0 && !noAnswer;

  const stats = {
    question: q,
    found_index: foundIndex,
    cached: false,
    provider: providerForModel(behaviorResponse.model || null),
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
    // What the worker actually used after intent overrides were merged
    // into the site config. Surfaces in the Training UI so the user can
    // see "this is what got sent" vs the bare site row. system_prompt
    // is shown post-combination (site + intent appended) so the value
    // matches what the LLM saw.
    _resolved_search: {
      ...(cfgForBehavior.search ?? {}),
      system_prompt: buildSystemPrompt(
        cfg.search?.system_prompt,
        intent?.system_prompt,
        cfgForBehavior.search?.system_prompt ?? "",
      ),
    },
    _resolved_intent: intent && intent.name !== "default" ? {
      name: intent.name,
      system_prompt: intent.system_prompt ?? null,
      answer_model: intent.answer_model ?? (intent.chat_model
        ? { provider: "cloudflare", model: intent.chat_model }
        : null),
      response_behavior: intent.response_behavior ?? null,
    } : null,
  };

  if (wantDebug) {
    bodyOut._debug = {
      matches: finalMatches.slice(0, 3), // Show the final matches passed to behavior
      cache: "miss",
      intent: intent.name,
      behavior: behaviorName,
    };
  }

  const response = json(bodyOut);
  stop();
  return response;
}
