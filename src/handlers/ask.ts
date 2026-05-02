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
 * Marketing-shaped diagnosis surfaced to the Training tab so the team can
 * tell at a glance whether a bad answer is a routing problem or a content
 * problem — without reading the technical scoring table. Severity reflects
 * the likely fix surface, not just routing success.
 *
 *   bad   → no relevant content (Not Found OR vector hit < 0.30)
 *   soft  → routed but with a soft signal (default fallback, close call,
 *           thin content)
 *   good  → matched cleanly with content backing
 *
 * Stays loose enough to evolve — add signals, don't break the contract.
 */
/**
 * Cheap "confabulation hint": does the query mention a distinctive content
 * word that the matched sources don't appear to cover at all? When the
 * answer is clean but the pages don't reference the asked-about topic, the
 * LLM is probably inventing — a high-confidence routing miss this signal
 * can catch.
 *
 * Only counts words ≥4 characters and not on a small stopword list, so
 * filler words don't drag the signal. URL slugs are checked alongside
 * titles because page slugs often spell out the topic.
 */
const TOPIC_STOPWORDS = new Set([
  "what", "whats", "when", "where", "which", "whose", "with", "your", "yours",
  "youre", "have", "this", "that", "from", "tell", "about", "much", "many",
  "long", "does", "their", "they", "them", "there", "would", "could", "should",
  "make", "give", "take", "want", "need", "show", "find", "help", "into", "more",
  "some", "than", "then", "very", "well",
]);
function topicCoveredInSources(
  query: string,
  sources: Array<{ title?: string; url?: string }>
): boolean {
  const tokens = query
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length >= 4 && !TOPIC_STOPWORDS.has(w));
  if (tokens.length === 0) return true; // nothing distinctive — can't judge
  if (!sources || sources.length === 0) return false;
  const hay = sources
    .map((s) => `${s.title || ""} ${s.url || ""}`)
    .join(" ")
    .toLowerCase();
  return tokens.some((t) => hay.includes(t));
}

function buildDiagnosis(args: {
  query: string;
  intentName: string;
  isNotFound: boolean;
  detection?: {
    reason?: string;
    top_vector_score?: number;
    top_intents?: Array<{ name: string; score: number }>;
  };
  topSourceTitle?: string | null;
  sources?: Array<{ title?: string; url?: string }>;
  /** Whether the matched intent has a curated appended prompt. When true, the
   * intent author has explicitly shaped the response — so topic-not-covered
   * is almost certainly intentional (e.g. Pricing intent always redirects
   * to the contact form regardless of source content). Skip the signal. */
  intentHasAppendedPrompt?: boolean;
}): {
  severity: "good" | "soft" | "bad";
  signals: string[];
  headline: string;
  explanation: string;
  footer?: string;
  query: string;
  closest_intent?: string;
  runner_up_intent?: string;
  top_source_title?: string;
} {
  const { query, intentName, isNotFound, detection, topSourceTitle, sources } = args;
  const topVec = Number(detection?.top_vector_score ?? 0);
  const top = detection?.top_intents?.[0];
  const runnerUp = detection?.top_intents?.[1];
  const reason = detection?.reason || "matched";
  const signals: string[] = [];

  if (isNotFound || (topVec > 0 && topVec < 0.3)) {
    signals.push("no_content");
  } else if (reason === "below_threshold" || reason === "ambiguous" || intentName === "default" || intentName === "Default") {
    signals.push("no_intent");
  } else {
    if (top && runnerUp && top.score - runnerUp.score < 0.10) signals.push("close_call");
    if (topVec > 0 && topVec < 0.5) signals.push("thin_content");
    // Confabulation hint: routing matched but the topic isn't in any source.
    // Skip when we already raised thin_content (same fix surface) or when
    // there are zero sources to evaluate against.
    if (
      !signals.includes("thin_content") &&
      !args.intentHasAppendedPrompt &&
      sources && sources.length > 0 &&
      !topicCoveredInSources(query, sources)
    ) {
      signals.push("topic_not_covered");
    }
    if (signals.length === 0) signals.push("strong_match");
  }

  let severity: "good" | "soft" | "bad";
  let headline: string;
  let explanation: string;
  let footer: string | undefined;

  if (signals.includes("no_content")) {
    severity = "bad";
    headline = "No relevant content found";
    explanation =
      "WORX content doesn't seem to cover this question yet. Add a page about this topic, wait for the next ingest cycle, then re-ask.";
  } else if (signals.includes("no_intent")) {
    severity = "soft";
    headline = "No intent matched — used Default fallback";
    explanation = top
      ? `Closest was "${top.name}" but the score wasn't confident enough. Either add a new intent for this question shape, or add this phrasing to "${top.name}" examples.`
      : "Nothing in your intent library was confident. Add a new intent that covers this question shape.";
  } else if (signals.includes("topic_not_covered")) {
    severity = "soft";
    headline = `${intentName} matched, but the topic isn't in the pages we found`;
    explanation =
      `${intentName} routed confidently, but none of the matched pages mention the topic of this question. The answer may be partly invented. Check whether WORX has a page covering this topic — if not, that's a content gap.`;
  } else if (signals.includes("thin_content")) {
    severity = "soft";
    headline = `Matched ${intentName}, but the closest page is thin`;
    explanation = topSourceTitle
      ? `${intentName} routed correctly, but the top page "${topSourceTitle}" doesn't strongly cover this question. Consider expanding that page, or check whether a different page should rank higher.`
      : `${intentName} routed correctly, but the top page didn't strongly cover this question. The answer may be vague.`;
  } else if (signals.includes("close_call")) {
    severity = "soft";
    headline = `Close call — ${intentName} won by a small margin`;
    explanation = runnerUp
      ? `${intentName} matched correctly, but "${runnerUp.name}" was a near tie. The answer is fine; only act if a different phrasing routes wrong.`
      : `${intentName} matched, but a slightly different phrasing could route somewhere else.`;
    footer = "Yellow = working but could be sharpened. Only fix it if you're seeing the wrong intent route on similar questions.";
  } else {
    severity = "good";
    headline = `Matched ${intentName} cleanly`;
    explanation = "";
  }

  // Soft-severity panels (other than close_call which already has its own) get
  // a generic clarification footer so a marketer doesn't treat every yellow
  // as broken.
  if (severity === "soft" && !footer) {
    footer = "Yellow = something to look at. Green = locked in.";
  }

  return {
    severity,
    signals,
    headline,
    explanation,
    footer,
    query,
    closest_intent: top?.name,
    runner_up_intent: runnerUp?.name,
    top_source_title: topSourceTitle || undefined,
  };
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
    detection,
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
      _intent_detection: detection ?? null,
      _diagnosis: buildDiagnosis({
        query: q,
        intentName: intent.name,
        isNotFound: true,
        detection,
        topSourceTitle: null,
        sources: [],
        intentHasAppendedPrompt: !!(intent.system_prompt && intent.system_prompt.trim()),
      }),
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
          _intent_detection: detection ?? null,
          _diagnosis: buildDiagnosis({
            query: q,
            intentName: intent.name,
            isNotFound: false,
            detection,
            topSourceTitle: (cachedResponse.sources && cachedResponse.sources[0]?.title) || null,
            sources: cachedResponse.sources || [],
            intentHasAppendedPrompt: !!(intent.system_prompt && intent.system_prompt.trim()),
          }),
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
    _intent_detection: detection ?? null,
    _diagnosis: buildDiagnosis({
      query: q,
      intentName: intent.name,
      isNotFound: false,
      detection,
      topSourceTitle: (behaviorResponse.sources && behaviorResponse.sources[0]?.title) || null,
      sources: behaviorResponse.sources || [],
      intentHasAppendedPrompt: !!(intent.system_prompt && intent.system_prompt.trim()),
    }),
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
