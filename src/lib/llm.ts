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
  if (req.temperature != null) body.temperature = req.temperature;
  if (req.max_tokens != null) body.max_tokens = req.max_tokens;

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
