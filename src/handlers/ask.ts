import { json } from "../http/response";
import { log, startTimer, time } from "../lib/logging";
import { clampTemperature, clampTopK, resolveCaching } from "../config/siteConfig";
import {
  DEFAULT_SYSTEM_PROMPT,
  INTENT_GUIDANCE,
  detectIntent,
  pickMatchesByIntent,
} from "../search/intent";
import {
  buildDocContext,
  ensureMarkdown,
  limitChildrenForIntent,
  normalizeUrl,
  parseChildrenMd,
} from "../search/context";
import { cachedEmbed, filterToSite } from "../search/vectorize";
import { sha1Hex } from "../utils/crypto";
import { isNoAnswer } from "../utils/isNoAnswer";
import type { Env, ExecutionContext, SiteConfig, SearchMatch } from "../lib/types";

/**
 * Conversation endpoint for WORX AI. It builds a tailored prompt, caches both
 * embeddings and answers when enabled, and provides intent-specific guidance.
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

  const intentInfo = detectIntent(q);
  const baseTopK = clampTopK(Number(body.k || cfg.search?.topK || 6));
  const initialK = Math.min(baseTopK, 3);
  const fallbackK = Math.max(initialK + 3, Math.min(baseTopK, 6));

  const promptOverrideRaw =
    typeof body.systemPrompt === "string"
      ? body.systemPrompt
      : typeof body.system_prompt === "string"
      ? body.system_prompt
      : "";
  const promptOverride = promptOverrideRaw ? String(promptOverrideRaw).trim() : "";
  const configPrompt = cfg.search?.system_prompt || DEFAULT_SYSTEM_PROMPT;

  const temperatureCandidateRaw =
    body.chatTemperature ??
    body.temperature ??
    body.chat_temperature ??
    null;
  const configTemperature = Number(cfg.search?.chat_temperature ?? 0.1);
  const temperatureCandidate =
    temperatureCandidateRaw !== null &&
    temperatureCandidateRaw !== undefined &&
    !Number.isNaN(Number(temperatureCandidateRaw))
      ? Number(temperatureCandidateRaw)
      : configTemperature;
  const temperature = clampTemperature(temperatureCandidate, configTemperature);

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

  let matches: SearchMatch[] = pickMatchesByIntent(
    await runQuery(initialK),
    intentInfo.intent,
    intentInfo.keywords
  );
  if (!matches.length && fallbackK > initialK) {
    matches = pickMatchesByIntent(await runQuery(fallbackK), intentInfo.intent, intentInfo.keywords);
  }

  const resolvedK = matches.length ? Math.min(matches.length, baseTopK) : initialK;
  const maxDocs = Math.max(1, Math.min(10, Number(cfg.search?.max_context_docs ?? 6)));
  const maxChars = Math.max(200, Math.min(4000, Number(cfg.search?.max_kv_text_chars ?? 2000)));
  const selected = matches.slice(0, Math.min(matches.length, maxDocs));

  const allowedUrlSet = new Set<string>();
  const MAX_ALLOWED_URLS = 40;
  const addAllowedUrl = (candidate: unknown) => {
    if (typeof candidate !== "string") return;
    if (allowedUrlSet.size >= MAX_ALLOWED_URLS) return;
    const trimmed = candidate.trim();
    if (!trimmed) return;
    const normalized = normalizeUrl(trimmed);
    if (normalized) {
      allowedUrlSet.add(normalized);
    } else {
      allowedUrlSet.add(trimmed);
    }
  };

  const docStop = startTimer("buildDocContext(filtered)");
  const contexts = await Promise.all(
    selected.map(async (match, idx) => {
      const metadata = (match.metadata || {}) as Record<string, unknown>;
      addAllowedUrl(metadata["url"]);
      addAllowedUrl(metadata["canonical"]);
      const childrenAll = parseChildrenMd(metadata["children_md"]);
      const baseUrl = normalizeUrl((metadata["url"] as string) || (metadata["canonical"] as string) || "");
      const childrenFiltered =
        intentInfo.intent === "page_list" && baseUrl
          ? childrenAll.filter((child) => {
              const childNormalized = child.normalizedUrl || normalizeUrl(child.url);
              return childNormalized !== baseUrl;
            })
          : childrenAll;
      const children = limitChildrenForIntent(childrenFiltered, intentInfo.intent);
      for (const child of children) {
        addAllowedUrl(child.url);
        if (child.normalizedUrl) addAllowedUrl(child.normalizedUrl);
      }
      const docContext = await buildDocContext(
        env,
        metadata,
        maxChars,
        intentInfo.keywords,
        intentInfo.intent,
        children
      );
      return `[#${idx + 1}] ${docContext}`;
    })
  );
  docStop();

  const allowedUrls = Array.from(allowedUrlSet);
  const linkHints = ["Use only these URLs when linking:", ...allowedUrls.map((u) => `- ${u}`)].join("\n");

  const basePrompt = (promptOverride !== "" ? promptOverride : configPrompt).trim();
  const intentGuide = (INTENT_GUIDANCE[intentInfo.intent] || INTENT_GUIDANCE.default).trim();
  const system = `${basePrompt}

Link Guidance:
${linkHints}

Intent focus: ${intentInfo.intent}
${intentGuide}`.trim();

  const user = `Question: ${q}\n\nContext:\n${contexts.join("\n\n")}`;
  const chatModel = cfg.search?.chat_model || "@cf/meta/llama-3.1-8b-instruct";
  const max_output_tokens = Math.max(128, Math.min(2048, Number(cfg.search?.max_output_tokens ?? 1024)));

  const ansKeyRaw = JSON.stringify({
    site,
    q,
    k: resolvedK,
    chatModel,
    temperature,
    max_output_tokens,
    intent: intentInfo.intent,
    systemHash: await sha1Hex(system),
  });
  const ansKey = `ans:${await sha1Hex(ansKeyRaw)}`;
  if (wantCaching) {
    try {
      const cachedAns = await env.WORX_AI_CONFIG.get<string>(ansKey, "text");
      if (cachedAns && cachedAns.trim()) {
        const sanitizedCached = ensureMarkdown(cachedAns);
        log("[cachedAnswer] HIT", ansKey);
        const nowIso = new Date().toISOString();
        const noAnswer = isNoAnswer(sanitizedCached);
        const foundIndex = Array.isArray(matches) && matches.length > 0 && !noAnswer;
        const stats = {
          question: q,
          found_index: foundIndex,
          cached: true,
          model: chatModel,
          tokens_input: null,
          tokens_output: null,
          total_tokens: null,
          timestamp: nowIso,
          temperature,
          intent: intentInfo.intent,
        };
        const bodyOut: Record<string, unknown> = {
          ok: true,
          q,
          k: resolvedK,
          answer: sanitizedCached,
          stats,
          intent: intentInfo.intent,
        };
        if (wantDebug) {
          bodyOut._debug = { matches: matches.slice(0, 3), cache: "hit", intent: intentInfo.intent };
        }
        const response = json(bodyOut);
        stop();
        return response;
      }
    } catch (error) {
      log("[cachedAnswer] READ-ERROR", ansKey, String((error as Error)?.message || error));
    }
  }

  const chat = await time("AI.run(chat)", () =>
    env.AI.run(chatModel as any, {
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature,
      max_output_tokens,
    } as any)
  );

  const rawAnswer =
    (chat as any).response ||
    "I couldn’t locate that information in the current WORX content. Try a different phrasing or explore the site for more context.";
  const answer = ensureMarkdown(rawAnswer as string);
  const noAnswer = isNoAnswer(answer);
  const ansTtl = Math.max(60, Math.min(86400, Number(cfg.search?.answer_cache_ttl ?? 600)));
  if (wantCaching) {
    try {
      if (ctx?.waitUntil && env.WORX_AI_CONFIG.put) {
        ctx.waitUntil(env.WORX_AI_CONFIG.put(ansKey, answer, { expirationTtl: ansTtl }));
        log("[cachedAnswer] STORE-QUEUED", ansKey, `ttl=${ansTtl}`);
      } else if (env.WORX_AI_CONFIG.put) {
        const stopPut = startTimer("KV put ans");
        await env.WORX_AI_CONFIG.put(ansKey, answer, { expirationTtl: ansTtl });
        stopPut();
        log("[cachedAnswer] STORED", ansKey, `ttl=${ansTtl}`);
      }
    } catch {
      log("[cachedAnswer] STORE-SKIP", ansKey, "KV not writable or put() unavailable");
    }
  }

  const usage = (chat as any)?.usage || (chat as any)?.meta?.usage || {};
  const tokens_input = (usage?.input_tokens ?? usage?.prompt_tokens ?? usage?.inputTokens ?? null) as number | null;
  const tokens_output = (usage?.output_tokens ?? usage?.completion_tokens ?? usage?.outputTokens ?? null) as number | null;
  const total_tokens = (usage?.total_tokens ??
    (tokens_input != null && tokens_output != null ? tokens_input + tokens_output : null)) as number | null;
  const nowIso2 = new Date().toISOString();
  const foundIndex2 = Array.isArray(matches) && matches.length > 0 && !noAnswer;
  const stats = {
    question: q,
    found_index: foundIndex2,
    cached: false,
    model: chatModel,
    tokens_input,
    tokens_output,
    total_tokens,
    timestamp: nowIso2,
    temperature,
    intent: intentInfo.intent,
  };
  const bodyOut: Record<string, unknown> = {
    ok: true,
    q,
    k: resolvedK,
    answer,
    stats,
    intent: intentInfo.intent,
  };
  if (wantDebug) {
    bodyOut._debug = { matches: matches.slice(0, 3), cache: "miss", intent: intentInfo.intent };
  }
  const response = json(bodyOut);
  stop();
  return response;
}
