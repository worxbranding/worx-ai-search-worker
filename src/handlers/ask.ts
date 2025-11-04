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
import { cachedEmbed, filterToSite, reRankMatches } from "../search/vectorize";
import { sha1Hex } from "../utils/crypto";
import { isNoAnswer } from "../utils/isNoAnswer";
import type { Env, ExecutionContext, SiteConfig, SearchMatch } from "../lib/types";
import { coerceBoolean } from "../utils/coerce";
import {
  getPolicySnapshot,
  getReRankWeights,
  pickPromptVariant,
} from "../training/policyCache";
import {
  matchesToTrainingDocs,
  TrainingAskMetadata,
  TrainingPolicyPromptVariant,
  TrainingPolicySnapshot,
} from "../training/types";

export interface AskExecutionInput {
  site: string;
  question: string;
  topK: number;
  promptOverride: string;
  temperature: number;
  wantCaching: boolean;
  training: boolean;
  promptVariant?: string | null;
  disableCaching?: boolean;
  retryContext?: {
    previous_answer?: string;
    feedback?: string;
    instruction?: string;
  } | null;
}

interface AskStats {
  question: string;
  found_index: boolean;
  cached: boolean;
  model: string;
  tokens_input: number | null;
  tokens_output: number | null;
  total_tokens: number | null;
  timestamp: string;
  temperature: number;
  intent: string;
}

export interface AskExecutionResult {
  question: string;
  resolvedK: number;
  answer: string;
  stats: AskStats;
  intent: string;
  matches: SearchMatch[];
  selected: SearchMatch[];
  contexts: string[];
  allowedUrls: string[];
  linkHints: string;
  systemPrompt: string;
  chatModel: string;
  maxOutputTokens: number;
  cacheState: "hit" | "miss";
  promptVariant?: TrainingPolicyPromptVariant | null;
  policySnapshot?: TrainingPolicySnapshot | null;
  trainingMetadata?: TrainingAskMetadata;
}

function buildTrainingMetadata(
  enabled: boolean,
  site: string,
  contexts: string[],
  selected: SearchMatch[],
  systemPrompt: string,
  chatModel: string,
  temperature: number,
  intent: string,
  policySnapshot: TrainingPolicySnapshot | null,
  promptVariant: TrainingPolicyPromptVariant | null,
  fromCache: boolean
): TrainingAskMetadata | undefined {
  if (!enabled) return undefined;
  return {
    policyVersion: policySnapshot?.version ?? null,
    policySnapshot,
    promptVariant,
    matches: matchesToTrainingDocs(selected, contexts, site),
    contexts,
    systemPrompt,
    temperature,
    model: chatModel,
    intent,
    fromCache,
  };
}

export async function executeAsk(
  env: Env,
  cfg: SiteConfig,
  input: AskExecutionInput,
  ctx?: ExecutionContext
): Promise<AskExecutionResult> {
  const intentInfo = detectIntent(input.question);
  const baseTopK = clampTopK(input.topK);
  const initialK = Math.min(baseTopK, 3);
  const fallbackK = Math.max(initialK + 3, Math.min(baseTopK, 6));

  const vector = await time("cachedEmbed(ask)", () =>
    cachedEmbed(env, cfg.ai.embed_model, cfg.vectorize.dims, input.question, ctx, input.wantCaching)
  );
  const runQuery = async (topK: number) => {
    // @ts-ignore Workers typing for VECTORIZE.query is permissive
    const res = await time("VECTORIZE.query(ask)", () =>
      env.VECTORIZE.query(vector, { topK, includeMetadata: true, returnMetadata: true })
    );
    return filterToSite(input.site, res?.matches || []);
  };

  let matches: SearchMatch[] = pickMatchesByIntent(
    await runQuery(initialK),
    intentInfo.intent,
    intentInfo.keywords
  );
  if (!matches.length && fallbackK > initialK) {
    matches = pickMatchesByIntent(await runQuery(fallbackK), intentInfo.intent, intentInfo.keywords);
  }

  let policySnapshot: TrainingPolicySnapshot | null = null;
  try {
    policySnapshot = await getPolicySnapshot(env, cfg, input.site);
  } catch (error) {
    log("[policyCache] load error", input.site, String((error as Error)?.message || error));
  }
  const weights = getReRankWeights(cfg, policySnapshot);
  if (weights && Object.keys(weights).length) {
    matches = reRankMatches(matches, weights);
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
      const baseUrl = normalizeUrl(
        (metadata["url"] as string) || (metadata["canonical"] as string) || ""
      );
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
  const linkHints = ["Use only these URLs when linking:", ...allowedUrls.map((u) => `- ${u}`)].join(
    "\n"
  );

  const configPrompt = (cfg.search?.system_prompt || DEFAULT_SYSTEM_PROMPT).trim();
  const policyDefaultVariant =
    typeof cfg.training?.policy?.defaultPromptVariant === "string"
      ? cfg.training?.policy?.defaultPromptVariant
      : null;
  const promptVariant = pickPromptVariant(
    policySnapshot,
    input.promptVariant || policyDefaultVariant
  );
  const variantPrompt = promptVariant?.systemPrompt?.trim();
  const basePrompt = (input.promptOverride || "").trim() || variantPrompt || configPrompt;
  const intentGuide = (INTENT_GUIDANCE[intentInfo.intent] || INTENT_GUIDANCE.default).trim();

  // Extract approved examples from policy (few-shot learning)
  const examples = policySnapshot?.prompts?.examples || [];
  const examplesText =
    examples.length > 0
      ? "\n\nHere are examples of correct answers:\n\n" +
        examples
          .map((ex: any) => `Q: ${ex.question}\nA: ${ex.answer}`)
          .join("\n\n")
      : "";

  // Extract guardrail rules from policy
  const guardrails = policySnapshot?.prompts?.guardrails || [];
  const guardrailsText =
    guardrails.length > 0
      ? "\n\nIMPORTANT GUIDELINES:\n" +
        guardrails.map((rule: string, i: number) => `${i + 1}. ${rule}`).join("\n")
      : "";

  // Add retry context if this is a retry attempt
  const retryContextText = input.retryContext
    ? `\n\nPREVIOUS ATTEMPT FEEDBACK:
Your previous answer: "${input.retryContext.previous_answer || 'N/A'}"
Feedback: ${input.retryContext.feedback || 'N/A'}
${input.retryContext.instruction || 'Please try again and correct the mistake.'}`
    : "";

  let system = `${basePrompt}

Link Guidance:
${linkHints}

Intent focus: ${intentInfo.intent}
${intentGuide}${retryContextText}${examplesText}${guardrailsText}`.trim();

  const user = `Question: ${input.question}\n\nContext:\n${contexts.join("\n\n")}`;
  const chatModel = cfg.search?.chat_model || "@cf/meta/llama-3.1-8b-instruct";
  const max_output_tokens = Math.max(
    128,
    Math.min(2048, Number(cfg.search?.max_output_tokens ?? 1024))
  );

  const cachingEnabled = input.wantCaching && !input.disableCaching;
  let cacheState: "hit" | "miss" = "miss";
  let ansKey: string | null = null;
  let answer: string;
  let stats: AskStats;

  if (cachingEnabled) {
    const systemHash = await sha1Hex(system);
    const ansKeyRaw = JSON.stringify({
      site: input.site,
      q: input.question,
      k: resolvedK,
      chatModel,
      temperature: input.temperature,
      max_output_tokens,
      intent: intentInfo.intent,
      systemHash,
      promptVariant: promptVariant?.key || null,
    });
    ansKey = `ans:${await sha1Hex(ansKeyRaw)}`;
    try {
      const cachedAns = await env.CONFIG.get<string>(ansKey, "text");
      if (cachedAns && cachedAns.trim()) {
        const sanitizedCached = ensureMarkdown(cachedAns);
        log("[cachedAnswer] HIT", ansKey);
        const timestamp = new Date().toISOString();
        const noAnswer = isNoAnswer(sanitizedCached);
        const foundIndex = Array.isArray(matches) && matches.length > 0 && !noAnswer;
        stats = {
          question: input.question,
          found_index: foundIndex,
          cached: true,
          model: chatModel,
          tokens_input: null,
          tokens_output: null,
          total_tokens: null,
          timestamp,
          temperature: input.temperature,
          intent: intentInfo.intent,
        };
        cacheState = "hit";
        const trainingMetadata = buildTrainingMetadata(
          input.training,
          input.site,
          contexts,
          selected,
          system,
          chatModel,
          input.temperature,
          intentInfo.intent,
          policySnapshot,
          promptVariant,
          true
        );
        return {
          question: input.question,
          resolvedK,
          answer: sanitizedCached,
          stats,
          intent: intentInfo.intent,
          matches,
          selected,
          contexts,
          allowedUrls,
          linkHints,
          systemPrompt: system,
          chatModel,
          maxOutputTokens: max_output_tokens,
          cacheState,
          promptVariant,
          policySnapshot,
          trainingMetadata,
        };
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
      temperature: input.temperature,
      max_output_tokens,
    } as any)
  );

  const rawAnswer =
    (chat as any).response ||
    "I couldn’t locate that information in the current WORX content. Try a different phrasing or explore the site for more context.";
  answer = ensureMarkdown(rawAnswer as string);
  const noAnswer = isNoAnswer(answer);
  const ansTtl = Math.max(60, Math.min(86400, Number(cfg.search?.answer_cache_ttl ?? 600)));

  if (cachingEnabled && ansKey) {
    try {
      if (ctx?.waitUntil && env.CONFIG.put) {
        ctx.waitUntil(env.CONFIG.put(ansKey, answer, { expirationTtl: ansTtl }));
        log("[cachedAnswer] STORE-QUEUED", ansKey, `ttl=${ansTtl}`);
      } else if (env.CONFIG.put) {
        const stopPut = startTimer("KV put ans");
        await env.CONFIG.put(ansKey, answer, { expirationTtl: ansTtl });
        stopPut();
        log("[cachedAnswer] STORED", ansKey, `ttl=${ansTtl}`);
      }
    } catch {
      log("[cachedAnswer] STORE-SKIP", ansKey, "KV not writable or put() unavailable");
    }
  }

  const usage = (chat as any)?.usage || (chat as any)?.meta?.usage || {};
  const tokens_input = (usage?.input_tokens ??
    usage?.prompt_tokens ??
    usage?.inputTokens ??
    null) as number | null;
  const tokens_output = (usage?.output_tokens ??
    usage?.completion_tokens ??
    usage?.outputTokens ??
    null) as number | null;
  const total_tokens = (usage?.total_tokens ??
    (tokens_input != null && tokens_output != null ? tokens_input + tokens_output : null)) as
    | number
    | null;
  const timestamp = new Date().toISOString();
  const foundIndex = Array.isArray(matches) && matches.length > 0 && !noAnswer;
  stats = {
    question: input.question,
    found_index: foundIndex,
    cached: false,
    model: chatModel,
    tokens_input,
    tokens_output,
    total_tokens,
    timestamp,
    temperature: input.temperature,
    intent: intentInfo.intent,
  };

  const trainingMetadata = buildTrainingMetadata(
    input.training,
    input.site,
    contexts,
    selected,
    system,
    chatModel,
    input.temperature,
    intentInfo.intent,
    policySnapshot,
    promptVariant,
    false
  );

  return {
    question: input.question,
    resolvedK,
    answer,
    stats,
    intent: intentInfo.intent,
    matches,
    selected,
    contexts,
    allowedUrls,
    linkHints,
    systemPrompt: system,
    chatModel,
    maxOutputTokens: max_output_tokens,
    cacheState,
    promptVariant,
    policySnapshot,
    trainingMetadata,
  };
}

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
  const wantDebug = (url.searchParams.get("debug") || "").trim() === "1";
  const wantCaching = resolveCaching(url, cfg);
  const body = (await req.json().catch(() => ({}))) as Record<string, any>;

  const site = (typeof body.site === "string" ? body.site : "").trim();
  const q = (typeof body.q === "string" ? body.q : "").trim();
  if (!site) {
    stop();
    return json({ ok: false, error: "Missing field 'site'" }, { status: 400 });
  }
  if (!q) {
    stop();
    return json({ ok: false, error: "Missing field 'q'" }, { status: 400 });
  }

  const queryTraining = coerceBoolean(url.searchParams.get("training"));
  const modeTraining = (url.searchParams.get("mode") || "").toLowerCase() === "training";
  const headerTraining = [req.headers.get("x-worx-training"), req.headers.get("x-training-mode")]
    .map(coerceBoolean)
    .some((val) => val === true);
  const bodyTraining = coerceBoolean(body.training);
  const trainingFlag = modeTraining || queryTraining === true || headerTraining || bodyTraining === true;

  const baseTopK = clampTopK(Number(body.k || cfg.search?.topK || 6));

  let promptOverrideRaw =
    typeof body.systemPrompt === "string"
      ? body.systemPrompt
      : typeof body.system_prompt === "string"
      ? body.system_prompt
      : "";
  if (!promptOverrideRaw && typeof body.promptOverride === "string") {
    promptOverrideRaw = body.promptOverride;
  }
  const promptOverride = promptOverrideRaw ? String(promptOverrideRaw).trim() : "";

  const temperatureCandidateRaw =
    body.chatTemperature ?? body.temperature ?? body.chat_temperature ?? null;
  const configTemperature = Number(cfg.search?.chat_temperature ?? 0.1);
  const temperatureCandidate =
    temperatureCandidateRaw !== null &&
    temperatureCandidateRaw !== undefined &&
    !Number.isNaN(Number(temperatureCandidateRaw))
      ? Number(temperatureCandidateRaw)
      : configTemperature;
  const temperature = clampTemperature(temperatureCandidate, configTemperature);

  const promptVariantCandidates = [
    typeof body.promptVariant === "string" ? body.promptVariant : null,
    typeof body.prompt_variant === "string" ? body.prompt_variant : null,
    url.searchParams.get("prompt_variant"),
    url.searchParams.get("promptVariant"),
  ];
  const promptVariant =
    promptVariantCandidates.find((value) => typeof value === "string" && value.trim())?.trim() ||
    null;

  const disableCachingInput = coerceBoolean(body.disableCaching);
  let disableCaching = disableCachingInput ?? false;
  if (trainingFlag && disableCachingInput === undefined) {
    disableCaching = true;
  }

  // Extract retry context for iterative training
  const retryContext =
    body.retry_context && typeof body.retry_context === "object"
      ? {
          previous_answer: body.retry_context.previous_answer,
          feedback: body.retry_context.feedback,
          instruction: body.retry_context.instruction,
        }
      : null;

  const result = await executeAsk(
    env,
    cfg,
    {
      site,
      question: q,
      topK: baseTopK,
      promptOverride,
      temperature,
      wantCaching,
      training: trainingFlag,
      promptVariant,
      disableCaching,
      retryContext,
    },
    ctx
  );

  const bodyOut: Record<string, unknown> = {
    ok: true,
    q: result.question,
    k: result.resolvedK,
    answer: result.answer,
    stats: result.stats,
    intent: result.intent,
  };

  if (wantDebug) {
    bodyOut._debug = {
      matches: result.matches.slice(0, 3),
      cache: result.cacheState,
      intent: result.intent,
    };
  }

  if (result.trainingMetadata) {
    const trainingOut: Record<string, unknown> = {
      intent: result.trainingMetadata.intent,
      policyVersion: result.trainingMetadata.policyVersion ?? null,
      promptVariant: result.trainingMetadata.promptVariant?.key || null,
      prompt: {
        system: result.trainingMetadata.systemPrompt,
        temperature: result.trainingMetadata.temperature,
        model: result.trainingMetadata.model,
        linkHints: result.linkHints,
        allowedUrls: result.allowedUrls,
      },
      documents: result.trainingMetadata.matches,
      contexts: result.trainingMetadata.contexts,
      cache: result.trainingMetadata.fromCache ? "hit" : "miss",
    };
    if (result.trainingMetadata.promptVariant?.metadata) {
      trainingOut.promptVariantMetadata = result.trainingMetadata.promptVariant.metadata;
    }
    if (result.trainingMetadata.policySnapshot) {
      trainingOut.policy = result.trainingMetadata.policySnapshot;
    }
    bodyOut._training = trainingOut;
  }

  stop();
  return json(bodyOut);
}
