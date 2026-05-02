import { json } from "../http/response";
import { log, startTimer, time } from "../lib/logging";
import type { Env, ExecutionContext, KVNamespace, SiteConfig } from "../lib/types";
import { embed } from "../search/vectorize";

/**
 * Delete all KV keys that share the provided prefix. When waitUntil is
 * available we queue the deletes so the request can return quickly.
 *
 * If siteFilter is provided, only deletes keys whose metadata.site matches.
 */
async function kvDeleteByPrefix(
  ns: KVNamespace,
  prefix: string,
  ctx?: ExecutionContext,
  siteFilter?: string
): Promise<{ prefix: string; total: number; queued: number; filtered: number }> {
  if (!ns.list || !ns.delete) throw new Error("KV list/delete not available on this binding");
  let cursor: string | undefined = undefined;
  let total = 0;
  let queued = 0;
  let filtered = 0;

  do {
    const page = await ns.list!({ prefix, limit: 1000, cursor });
    const keys = page.keys || [];
    total += keys.length;

    // Filter by site if specified
    let namesToDelete = keys.map((k) => k.name);
    if (siteFilter) {
      namesToDelete = keys
        .filter((k) => {
          // Primary: check metadata.site for exact match
          const meta = (k as any).metadata;
          if (meta && meta.site === siteFilter) return true;
          // Fallback: check key name for exact prefix match
          // Key format is "prefix:site:hash", so after the prefix the site
          // must be followed by ":" to avoid substring false positives
          const afterPrefix = k.name.indexOf(":") + 1;
          const rest = k.name.substring(afterPrefix);
          if (rest === siteFilter || rest.startsWith(siteFilter + ":")) return true;
          return false;
        })
        .map((k) => k.name);
      filtered = keys.length - namesToDelete.length;
    }

    if (namesToDelete.length) {
      const delAll = Promise.all(namesToDelete.map((n) => ns.delete!(n).catch(() => {})));
      if (ctx?.waitUntil) {
        ctx.waitUntil(delAll);
        queued += namesToDelete.length;
      } else {
        await delAll;
        queued += namesToDelete.length;
      }
    }
    cursor = (page as any).cursor || undefined;
    if (page.list_complete) break;
  } while (cursor);
  return { prefix, total, queued, filtered };
}

/**
 * Admin endpoint to clear cached answers and embeddings.
 *
 * Query params:
 * - scope: "answers" | "embeddings" | "all" (default: "answers")
 * - site: optional site key to filter (e.g., "worxbranding-dev")
 *
 * Body (JSON):
 * - scope: same as query param
 * - site: same as query param
 */
export async function handleClearCache(
	req: Request,
	env: Env,
	body: { scope?: string; site?: string } | undefined,
	ctx?: ExecutionContext
): Promise<Response> {
  // Body is parsed once by the entry point (HMAC reads the raw bytes there);
  // passing it back in here avoids the "stream already consumed" bug where
  // req.json() returns {} and scope silently defaults to "answers".
  const stop = startTimer("handleClearCache");
  const url = new URL(req.url);
  const qsScope = (url.searchParams.get("scope") || "").trim();
  const qsSite = (url.searchParams.get("site") || "").trim();
  const b = body ?? {};
  const scope = String(b.scope || qsScope || "answers").toLowerCase();
  const site = String(b.site || qsSite || "").trim();

  if (!env.CACHE.list || !env.CACHE.delete) {
    stop();
    return json(
      { ok: false, error: "CACHE KV does not support list/delete in this environment" },
      { status: 501 }
    );
  }

  const prefixes =
    scope === "answers" ? ["ans:"] :
    scope === "embeddings" ? ["qemb:"] :
    scope === "all" ? ["ans:", "qemb:"] :
    ["ans:"]; // Default to answers only

  const results: Array<{ prefix: string; total: number; queued: number; filtered: number }> = [];
  for (const p of prefixes) {
    try {
      const r = await time(`KV clear ${p}`, () => kvDeleteByPrefix(env.CACHE, p, ctx, site || undefined));
      results.push(r);
      log("[clear-cache]", p, r, site ? `(site: ${site})` : "(all sites)");
    } catch (error: any) {
      results.push({ prefix: p, total: 0, queued: 0, filtered: 0 });
      log("[clear-cache] ERROR", p, String(error?.message || error));
    }
  }

  stop();
  return json({
    ok: true,
    scope,
    site: site || "all",
    results,
    note: ctx?.waitUntil ? "Deletes queued asynchronously" : "Deletes completed synchronously",
  });
}

/**
 * Embed a single text string with the same model the search pipeline
 * uses, return the raw vector. The CMS calls this when an intent is saved
 * so a fresh detection embedding gets stored. HMAC-protected by the
 * worker's standard signing path (same as /admin/clear-cache).
 */
export async function handleAdminEmbed(
  req: Request,
  env: Env,
  cfg: SiteConfig,
  body: { text?: string } | undefined
): Promise<Response> {
  const stop = startTimer("handleAdminEmbed");
  const text = String((body?.text ?? "")).trim();
  if (text === "") {
    stop();
    return json({ ok: false, error: "Missing field 'text'" }, { status: 400 });
  }
  try {
    const vector = await embed(env, cfg.ai.embed_model, cfg.vectorize.dims, text);
    stop();
    return json({
      ok: true,
      model: cfg.ai.embed_model,
      dims: cfg.vectorize.dims,
      vector,
    });
  } catch (error: any) {
    stop();
    return json({ ok: false, error: String(error?.message || error) }, { status: 500 });
  }
}
