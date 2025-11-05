import { json } from "../http/response";
import { log, startTimer, time } from "../lib/logging";
import type { Env, ExecutionContext, KVNamespace } from "../lib/types";

/**
 * Delete all KV keys that share the provided prefix. When waitUntil is
 * available we queue the deletes so the request can return quickly.
 */
async function kvDeleteByPrefix(
  ns: KVNamespace,
  prefix: string,
  ctx?: ExecutionContext
): Promise<{ prefix: string; total: number; queued: number }> {
  if (!ns.list || !ns.delete) throw new Error("KV list/delete not available on this binding");
  let cursor: string | undefined = undefined;
  let total = 0;
  let queued = 0;
  do {
    const page = await ns.list!({ prefix, limit: 1000, cursor });
    const names = (page.keys || []).map((k) => k.name);
    total += names.length;
    if (names.length) {
      const delAll = Promise.all(names.map((n) => ns.delete!(n).catch(() => {})));
      if (ctx?.waitUntil) {
        ctx.waitUntil(delAll);
        queued += names.length;
      } else {
        await delAll;
        queued += names.length;
      }
    }
    cursor = (page as any).cursor || undefined;
    if (page.list_complete) break;
  } while (cursor);
  return { prefix, total, queued };
}

/** Admin endpoint to clear cached answers and embeddings. */
export async function handleClearCache(
	req: Request,
	env: Env,
	ctx?: ExecutionContext
): Promise<Response> {
  const stop = startTimer("handleClearCache");
  const url = new URL(req.url);
  const qsScope = (url.searchParams.get("scope") || "").trim();
  const body = (await req.json().catch(() => ({}))) as { scope?: string };
  const scope = String(body.scope || qsScope || "all").toLowerCase();

  if (!env.WORX_AI_CONFIG.list || !env.WORX_AI_CONFIG.delete) {
    stop();
    return json(
      { ok: false, error: "CONFIG KV does not support list/delete in this environment" },
      { status: 501 }
    );
  }

  const prefixes =
    scope === "answers" ? ["ans:"] : scope === "embeddings" ? ["qemb:"] : ["ans:", "qemb:"];

  const results: Array<{ prefix: string; total: number; queued: number }> = [];
  for (const p of prefixes) {
    try {
      const r = await time(`KV clear ${p}`, () => kvDeleteByPrefix(env.WORX_AI_CONFIG, p, ctx));
      results.push(r);
      log("[clear-cache]", p, r);
    } catch (error: any) {
      results.push({ prefix: p, total: 0, queued: 0 });
      log("[clear-cache] ERROR", p, String(error?.message || error));
    }
  }

  stop();
  return json({
    ok: true,
    scope,
    results,
    note: ctx?.waitUntil ? "Deletes queued asynchronously" : "Deletes completed synchronously",
  });
}
