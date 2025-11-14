import { json } from "../http/response";
import { log, startTimer, time } from "../lib/logging";
import type { Env, ExecutionContext, KVNamespace } from "../lib/types";

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
          // For answer cache (ans:*), check if key contains site in hash
          // This is a simple contains check - may need refinement
          const meta = k.metadata as any;
          if (meta && meta.site === siteFilter) return true;
          // Fallback: check if key name contains site identifier
          if (k.name.includes(siteFilter)) return true;
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
	ctx?: ExecutionContext
): Promise<Response> {
  const stop = startTimer("handleClearCache");
  const url = new URL(req.url);
  const qsScope = (url.searchParams.get("scope") || "").trim();
  const qsSite = (url.searchParams.get("site") || "").trim();
  const body = (await req.json().catch(() => ({}))) as { scope?: string; site?: string };
  const scope = String(body.scope || qsScope || "answers").toLowerCase();
  const site = String(body.site || qsSite || "").trim();

  if (!env.WORX_AI_CONFIG.list || !env.WORX_AI_CONFIG.delete) {
    stop();
    return json(
      { ok: false, error: "CONFIG KV does not support list/delete in this environment" },
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
      const r = await time(`KV clear ${p}`, () => kvDeleteByPrefix(env.WORX_AI_CONFIG, p, ctx, site || undefined));
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
