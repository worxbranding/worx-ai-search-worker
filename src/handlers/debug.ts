import { json } from "../http/response";
import { log, startTimer, time } from "../lib/logging";
import { resolveCaching, type InBandRequestBody } from "../config/siteConfig";
import {
  cachedEmbed,
  embed,
  filterToSite,
  sitePrefixedId,
  vectorizeQueryById,
} from "../search/vectorize";
import type { Env, SiteConfig } from "../lib/types";

/** Quick endpoint to inspect raw embeddings generated for a prompt. */
export async function handleDebugEmbed(
  req: Request,
  env: Env,
  cfg: SiteConfig,
  body: InBandRequestBody
): Promise<Response> {
  const stop = startTimer("handleDebugEmbed");
  const url = new URL(req.url);
  const q = (body.q || url.searchParams.get("q") || "").trim();
  if (!q) {
    stop();
    return json({ ok: false, error: "Missing field 'q'" }, { status: 400 });
  }
  try {
    const arr = await embed(env, cfg.ai.embed_model, cfg.vectorize.dims, q);
    const res = json({
      ok: true,
      q,
      result: { valid: Array.isArray(arr) && arr.length === cfg.vectorize.dims, gotLen: arr.length, expected: cfg.vectorize.dims },
    });
    stop();
    return res;
  } catch (error: any) {
    stop();
    return json({ ok: false, q, error: String(error?.message || error) }, { status: 500 });
  }
}

/**
 * Provide a robust Vectorize ID lookup that tries multiple id shapes and
 * falls back to embedding the KV content when necessary.
 */
export async function handleDebugQueryById(
  req: Request,
  env: Env,
  cfg: SiteConfig,
  body: InBandRequestBody
): Promise<Response> {
  const stop = startTimer("handleDebugQueryById");
  const url = new URL(req.url);
  const site = (body.site || cfg.site_key || url.searchParams.get("site") || "").trim();
  const idRaw = (url.searchParams.get("id") || "").trim();
  const kRaw = (url.searchParams.get("k") || "").trim();
  const kDigits = kRaw.match(/\d+/)?.[0] || "";
  const kParsed = parseInt(kDigits, 10);
  const k = Math.max(1, Math.min(24, Number.isFinite(kParsed) ? kParsed : 3));
  const useRaw = ["1", "true", "yes"].includes((url.searchParams.get("raw") || "").trim().toLowerCase());
  const forceDoc = ["1", "true", "yes"].includes((url.searchParams.get("doc") || "").trim().toLowerCase());
  const wantCaching = resolveCaching(url, cfg);
  if (!site) {
    stop();
    return json({ ok: false, error: "Missing ?site=" }, { status: 400 });
  }
  if (!idRaw) {
    stop();
    return json({ ok: false, error: "Missing ?id" }, { status: 400 });
  }

	const prefixed = sitePrefixedId(site, idRaw);
	const stripped = idRaw.startsWith(`${site}:`) ? idRaw.slice(site.length + 1) : null;

	let candidateIds: string[];
  if (useRaw) {
    candidateIds = [idRaw];
  } else if (forceDoc) {
    candidateIds = [
      `doc:${prefixed}`,
      `doc:${idRaw}`,
      ...(stripped ? [`doc:${stripped}`] : []),
    ];
  } else {
    candidateIds = [
      prefixed,
      idRaw,
      ...(stripped ? [stripped] : []),
      `doc:${prefixed}`,
      `doc:${idRaw}`,
      ...(stripped ? [`doc:${stripped}`] : []),
    ];
  }

  const tryIds = Array.from(new Set(candidateIds.filter(Boolean)));

  let lastErr: unknown = null;

  for (const vectorId of tryIds) {
    try {
      const out = await vectorizeQueryById(env, vectorId, k);
      const post = filterToSite(site, out?.matches || []);
      if ((out?.matches && out.matches.length > 0) || post.length > 0) {
        log("[debug/query-by-id] SUCCESS with", vectorId);
        const res = json({ ok: true, id: vectorId, k, results: post });
        stop();
        return res;
      }
      lastErr = lastErr || new Error("No matches returned for this id");
    } catch (error: any) {
      lastErr = error;
      const msg = String(error?.message || error);
      if (msg.includes("40006") || msg.toLowerCase().includes("invalid query vector")) {
        log("[debug/query-by-id] attempt failed for", vectorId, "->", msg);
        continue;
      }
      break;
    }
  }

  try {
    const kvCandidates = Array.from(
      new Set(
        tryIds.flatMap((vid) => {
          const arr: string[] = [];
          if (vid.startsWith("doc:")) arr.push(vid);
          arr.push(`doc:${vid}`);
          if (!vid.startsWith(`${site}:`)) arr.push(`doc:${site}:${vid}`);
          return arr;
        })
      )
    );

    for (const kvKey of kvCandidates) {
      try {
        const txt = await env.CONTENT.get<string>(kvKey, "text");
        if (txt && txt.trim()) {
          const snippet = txt.slice(0, 3000);
          const vec = await cachedEmbed(env, cfg.ai.embed_model, cfg.vectorize.dims, snippet, undefined, wantCaching);
          // @ts-ignore Workers typing
          const out = await env.VECTORIZE.query(vec, { topK: k, returnMetadata: true, includeMetadata: true });
          const post = filterToSite(site, out?.matches || []);
          if (post.length > 0) {
            log("[debug/query-by-id] FALLBACK success via KV doc", kvKey);
            const res = json({
              ok: true,
              id: idRaw,
              k,
              results: post,
              note: `fallback: embedded KV text via ${kvKey}`,
            });
            stop();
            return res;
          }
        }
      } catch {
        // ignore and try next
      }
    }
  } catch (error) {
    log("[debug/query-by-id] fallback error", String((error as any)?.message || error));
  }

  const hints = [
    "The provided id might not exist in the index.",
    "Try adding &doc=1 to query using doc:-prefixed ids (e.g., doc:<site>:page:<id>).",
    "If your vectors are stored without a site prefix, try adding &raw=1 to use the id exactly as provided.",
    `Tried ids: ${tryIds.join(", ")}`,
    "Note: We also attempted a fallback by embedding the KV document text when available.",
  ];

  const errMsg = lastErr ? String((lastErr as any)?.message || lastErr) : "No matches returned for any attempted id variant";
  stop();
  return json({ ok: false, error: errMsg, site, id: idRaw, k, hints, tried: tryIds }, { status: 400 });
}

/** List the doc:* ids stored in KV so operators can inspect ingest output. */
export async function handleDebugListIds(
	req: Request,
	env: Env
): Promise<Response> {
  const stop = startTimer("handleDebugListIds");
  const url = new URL(req.url);
  const site = (url.searchParams.get("site") || "").trim();
  const limit = Math.max(1, Math.min(1000, Number(url.searchParams.get("limit") || 100)));
  const cursor = url.searchParams.get("cursor") || undefined;
  const wantStripped = ["1", "true", "yes"].includes((url.searchParams.get("stripped") || "").trim().toLowerCase());
  const wantValidate = ["1", "true", "yes"].includes((url.searchParams.get("validate") || "").trim().toLowerCase());
  if (!site) {
    stop();
    return json({ ok: false, error: "Missing ?site=" }, { status: 400 });
  }
  if (!env.CONTENT.list) {
    stop();
    return json({ ok: false, error: "CONTENT KV does not support list() in this environment" }, { status: 501 });
  }

  const prefix = `doc:${site}:`;
  const page = await time(`KV list ${prefix}`, () => env.CONTENT.list!({ prefix, limit, cursor }));
  const names = (page.keys || []).map((k) => k.name);
  const idsPrefixed = names.map((n) => (n.startsWith("doc:") ? n.slice(4) : n));
  const idsStripped = idsPrefixed.map((v) => (v.startsWith(`${site}:`) ? v.slice(site.length + 1) : v));

  let validatedOk: string[] = [];
  let validatedFail: string[] = [];

  if (wantValidate && idsPrefixed.length) {
    for (const vid of idsPrefixed) {
      try {
        await vectorizeQueryById(env, vid, 1);
        validatedOk.push(vid);
      } catch (error: any) {
        const msg = String(error?.message || error);
        if (msg.includes("40006") || msg.toLowerCase().includes("invalid query vector")) {
          validatedFail.push(vid);
        } else {
          log("[debug/list-ids] validate error for", vid, "->", msg);
          validatedFail.push(vid);
        }
      }
    }
  }

  const body: Record<string, unknown> = {
    ok: true,
    site,
    prefix,
    limit,
    total_page: names.length,
    list_complete: !!(page as any).list_complete,
    next_cursor: (page as any).cursor || null,
    ids_prefixed: idsPrefixed,
    ids_stripped: idsStripped,
  };
  body.ids = wantStripped ? idsStripped : idsPrefixed;
  if (wantValidate) {
    body.validation = {
      ok: validatedOk.length,
      fail: validatedFail.length,
      fail_sample: validatedFail.slice(0, 20),
    };
  }

  stop();
  return json(body);
}
