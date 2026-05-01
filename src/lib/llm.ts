import type { AnswerModel, CustomIntent, Env, LlmProvider, SiteConfig } from "./types";
import { extractResponse } from "../utils/extractResponse";

export interface LlmMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface LlmRequest {
  provider: LlmProvider;
  model: string;
  messages: LlmMessage[];
  temperature?: number;
  max_tokens?: number;
}

export interface LlmUsage {
  tokens_input: number | null;
  tokens_output: number | null;
  total_tokens: number | null;
}

export interface LlmResponse {
  answer: string;
  usage: LlmUsage;
  raw: unknown;
}

const DEFAULT_ANSWER_MODEL: AnswerModel = {
  provider: "cloudflare",
  model: "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
};

/**
 * Merge per-intent tuning overrides on top of site-level config.search, so
 * downstream behaviors only need to read `config.search.*` and get the
 * resolved value. Returns a new SiteConfig — original is not mutated.
 *
 * Per-intent fields supported: chat_temperature, initial_topK, final_topK,
 * max_output_tokens, max_context_docs, max_kv_text_chars. answer_model
 * stays handled by resolveAnswerModel because it has its own resolution
 * chain (intent.answer_model → intent.chat_model → cfg.search.answer_model).
 */
export function mergeIntentTuning(
  config: SiteConfig,
  intent: CustomIntent | null | undefined,
): SiteConfig {
  if (!intent) return config;
  const search = { ...(config.search ?? {}) };
  if (intent.chat_temperature   !== undefined) search.chat_temperature   = intent.chat_temperature;
  if (intent.initial_topK       !== undefined) search.initial_topK       = intent.initial_topK;
  if (intent.final_topK         !== undefined) search.final_topK         = intent.final_topK;
  if (intent.max_output_tokens  !== undefined) search.max_output_tokens  = intent.max_output_tokens;
  if (intent.max_context_docs   !== undefined) search.max_context_docs   = intent.max_context_docs;
  if (intent.max_kv_text_chars  !== undefined) search.max_kv_text_chars  = intent.max_kv_text_chars;
  // Reflect the intent's model override on cfg.search.answer_model so
  // _resolved_search in the response shows what actually ran. Behaviors
  // still get the model via resolveAnswerModel, which has the same
  // intent-first precedence — this is purely for UI fidelity.
  if (intent.answer_model?.provider && intent.answer_model?.model) {
    search.answer_model = intent.answer_model;
  } else if (intent.chat_model) {
    search.answer_model = { provider: "cloudflare", model: intent.chat_model };
  }
  return { ...config, search };
}

/**
 * Resolve which provider+model to use for a given request, applying the
 * intent override → site config → built-in default fallback chain.
 *
 * Backwards compat: legacy `chat_model` strings are treated as cloudflare.
 */
export function resolveAnswerModel(
  intent: CustomIntent | null | undefined,
  config: SiteConfig | null | undefined,
): AnswerModel {
  if (intent?.answer_model?.provider && intent.answer_model.model) {
    return intent.answer_model;
  }
  if (intent?.chat_model) {
    return { provider: "cloudflare", model: intent.chat_model };
  }
  if (config?.search?.answer_model?.provider && config.search.answer_model.model) {
    return config.search.answer_model;
  }
  if (config?.search?.chat_model) {
    return { provider: "cloudflare", model: config.search.chat_model };
  }
  return DEFAULT_ANSWER_MODEL;
}

/**
 * Run a chat completion against the resolved provider. Returns a uniform
 * { answer, usage, raw } shape regardless of which provider was used.
 *
 * Throws on configuration errors (missing secret, missing gateway vars) and
 * on non-2xx responses from the gateway. Behaviors decide whether to surface
 * the error to the caller or render a fallback message.
 */
export async function runChat(env: Env, req: LlmRequest): Promise<LlmResponse> {
  if (req.provider === "cloudflare") {
    return callCloudflare(env, req);
  }
  if (req.provider === "openai" || req.provider === "anthropic") {
    return callViaGateway(env, req);
  }
  throw new Error(`Unsupported LLM provider: ${String(req.provider)}`);
}

async function callCloudflare(env: Env, req: LlmRequest): Promise<LlmResponse> {
  const raw = await env.AI.run(req.model as any, {
    messages: req.messages,
    temperature: req.temperature,
    max_tokens: req.max_tokens,
  } as any);

  return {
    answer: extractResponse(raw) ?? "",
    usage: extractUsage(raw),
    raw,
  };
}

async function callViaGateway(env: Env, req: LlmRequest): Promise<LlmResponse> {
  const accountId = env.AI_GATEWAY_ACCOUNT_ID;
  const slug = env.AI_GATEWAY_SLUG;
  if (!accountId || !slug) {
    throw new Error(
      "AI Gateway not configured (missing AI_GATEWAY_ACCOUNT_ID or AI_GATEWAY_SLUG)",
    );
  }

  const apiKey = req.provider === "openai" ? env.OPENAI_API_KEY : env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    throw new Error(
      `Missing ${req.provider === "openai" ? "OPENAI_API_KEY" : "ANTHROPIC_API_KEY"} secret`,
    );
  }

  const url = `https://gateway.ai.cloudflare.com/v1/${accountId}/${slug}/compat/chat/completions`;
  const body: Record<string, unknown> = {
    model: `${req.provider}/${req.model}`,
    messages: req.messages,
  };
  // OpenAI's GPT-5 family and reasoning (o1/o3/o4) models have stricter
  // params: they reject `max_tokens` (must be `max_completion_tokens`) and
  // only accept the default temperature, so we omit it entirely.
  // Anthropic accepts `max_tokens` and any temperature (via Cloudflare's
  // compat endpoint).
  const isOpenAiRestricted =
    req.provider === "openai" && /^(gpt-5|o1|o3|o4-)/i.test(req.model);

  if (req.temperature != null && !isOpenAiRestricted) {
    body.temperature = req.temperature;
  }
  if (req.max_tokens != null) {
    if (isOpenAiRestricted) {
      // GPT-5 / reasoning models count hidden reasoning tokens against
      // this budget. Per-behavior clamps (e.g. LongFormAnswer caps at
      // 600) get fully consumed by reasoning and leave nothing for the
      // visible answer — the model then returns an empty string and the
      // behavior falls back to a "no content" message. Floor the budget
      // to give reasoning headroom + room for the actual answer.
      body.max_completion_tokens = Math.max(req.max_tokens, 4000);
    } else {
      body.max_tokens = req.max_tokens;
    }
  }

  const resp = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(
      `${req.provider} gateway call failed (${resp.status}): ${text.slice(0, 500)}`,
    );
  }

  const raw = (await resp.json()) as Record<string, unknown>;
  return {
    answer: extractResponse(raw) ?? "",
    usage: extractUsage(raw),
    raw,
  };
}

function extractUsage(raw: unknown): LlmUsage {
  const u = (raw as any)?.usage ?? (raw as any)?.meta?.usage ?? {};
  const tokens_input =
    (u.input_tokens ?? u.prompt_tokens ?? u.inputTokens ?? null) as number | null;
  const tokens_output =
    (u.output_tokens ?? u.completion_tokens ?? u.outputTokens ?? null) as number | null;
  const total_tokens = (u.total_tokens ??
    (tokens_input != null && tokens_output != null
      ? tokens_input + tokens_output
      : null)) as number | null;
  return { tokens_input, tokens_output, total_tokens };
}
