import { startTimer } from "../lib/logging";
import { json } from "../http/response";
import type { Env } from "../lib/types";

/**
 * Status endpoint: reports what the worker is configured with at the env
 * level. No KV reads, no body required — just confirms the wrangler vars
 * + secrets are wired.
 */
export async function handleStatus(req: Request, env: Env): Promise<Response> {
  const stop = startTimer("handleStatus");
  const dims = parseInt(env.VECTORIZE_DIMS || "0", 10);
  const res = json({
    ok: true,
    embed_model: env.EMBED_MODEL || null,
    vectorize: {
      dims: Number.isFinite(dims) ? dims : null,
      metric: env.VECTORIZE_METRIC || "cosine",
    },
    ai_gateway: env.AI_GATEWAY_SLUG
      ? { account_id: env.AI_GATEWAY_ACCOUNT_ID, slug: env.AI_GATEWAY_SLUG }
      : null,
    secrets_present: {
      openai: !!env.OPENAI_API_KEY,
      anthropic: !!env.ANTHROPIC_API_KEY,
      hmac: !!env.WORX_HMAC_SECRET,
      hmac_previous: !!env.WORX_HMAC_SECRET_PREVIOUS,
    },
    cors_fallback_origins: (env.ALLOWED_ORIGINS || "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean),
  });
  stop();
  return res;
}
