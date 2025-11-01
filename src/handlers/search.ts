import { json } from "../http/response";
import { startTimer, time } from "../lib/logging";
import { clampTopK, resolveCaching, resolveTopKFromQuery } from "../config/siteConfig";
import { cachedEmbed, filterToSite } from "../search/vectorize";
import { detectIntent } from "../search/intent";
import type { Env, ExecutionContext, SearchMatch, SiteConfig } from "../lib/types";

/**
 * Vector search endpoint. It reuses the cached embedding logic, so `/search`
 * benefits from KV caching when enabled.
 */
export async function handleSearch(
  req: Request,
  env: Env,
  cfg: SiteConfig,
  ctx?: ExecutionContext
): Promise<Response> {
  const stop = startTimer("handleSearch");
  const url = new URL(req.url);
  const site = (url.searchParams.get("site") || "").trim();
  const q = (url.searchParams.get("q") || "").trim();
  const intentInfo = detectIntent(q);
  const cfgTopK = Number(cfg.search?.topK ?? 6);
  const defaultK = clampTopK(Number.isFinite(cfgTopK) ? cfgTopK : 6);
  const k = resolveTopKFromQuery(url, defaultK);
  const wantDebug = (url.searchParams.get("debug") || "") === "1";
  const wantCaching = resolveCaching(url, cfg);
  if (!site) {
    stop();
    return json({ ok: false, error: "Missing ?site=" }, { status: 400 });
  }
  if (!q) {
    stop();
    return json({ ok: false, error: "Missing ?q=" }, { status: 400 });
  }

  const vector = await time("cachedEmbed(search)", () =>
    cachedEmbed(env, cfg.ai.embed_model, cfg.vectorize.dims, q, ctx, wantCaching)
  );
  // @ts-ignore Cloudflare Workers types are permissive for VECTORIZE.query
  const out = await time("VECTORIZE.query(search)", () =>
    env.VECTORIZE.query(vector, { topK: k, returnMetadata: true })
  );
  const pre = (out?.matches || []) as SearchMatch[];
  const post = filterToSite(site, pre);

  const body: Record<string, unknown> = { ok: true, site, q, k, results: post, intent: intentInfo.intent };
  if (wantDebug) {
    body._debug = {
      total_pre: pre.length,
      total_post: post.length,
      sample_ids: post.slice(0, 3).map((m) => m.id),
    };
  }
  stop();
  return json(body);
}
