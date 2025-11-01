import { startTimer } from "../lib/logging";
import { json } from "../http/response";
import { loadSiteConfig, wantApiKey } from "../config/siteConfig";
import type { Env, SiteConfig } from "../lib/types";

/**
 * Status endpoint: returns the resolved configuration so operators can confirm
 * bindings are wired correctly.
 */
export async function handleStatus(req: Request, env: Env): Promise<Response> {
  const stop = startTimer("handleStatus");
  const url = new URL(req.url);
  const site = (url.searchParams.get("site") || "").trim();
  if (!site) {
    stop();
    return json({ ok: false, error: "Missing ?site=" }, { status: 400 });
  }
  const cfg: SiteConfig = await loadSiteConfig(env, site);
  const res = json({
    ok: true,
    site,
    vectorize: cfg.vectorize.index_name,
    dims: cfg.vectorize.dims,
    embed_model: cfg.ai.embed_model,
    chat_model: cfg.search?.chat_model || "@cf/meta/llama-3.1-8b-instruct",
    system_prompt: cfg.search?.system_prompt || null,
    chat_temperature: cfg.search?.chat_temperature ?? null,
    requires_api_key: !!wantApiKey(cfg),
  });
  stop();
  return res;
}
