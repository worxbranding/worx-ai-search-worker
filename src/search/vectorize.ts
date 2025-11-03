import { log, startTimer, time } from "../lib/logging";
import { sha1Hex } from "../utils/crypto";
import type { Env, ExecutionContext, SearchMatch } from "../lib/types";

/** Prefix a document id with the site code when it is missing. */
export function sitePrefixedId(site: string, id: string): string {
  return id.startsWith(`${site}:`) ? id : `${site}:${id}`;
}

function coerceMatch(raw: unknown): SearchMatch | null {
  if (!raw || typeof raw !== "object") return null;
  const obj = raw as Record<string, unknown>;
  const id = typeof obj.id === "string" ? obj.id : null;
  if (!id) return null;
  const score = typeof obj.score === "number" && Number.isFinite(obj.score) ? obj.score : 0;
  const metadata =
    obj.metadata && typeof obj.metadata === "object" ? (obj.metadata as Record<string, unknown>) : undefined;
  return { id, score, metadata };
}

/** Remove matches that clearly belong to other tenants and coerce the payload shape. */
export function filterToSite(site: string, matches: unknown[]): SearchMatch[] {
  const filtered: SearchMatch[] = [];
  for (const raw of matches || []) {
    const match = coerceMatch(raw);
    if (!match) continue;
    if (match.id.startsWith(`${site}:`)) {
      filtered.push(match);
      continue;
    }
    const siteMeta = (match.metadata as any)?.site;
    if (siteMeta === site) {
      filtered.push(match);
    }
  }
  return filtered;
}

/**
 * Query Vectorize by id while trying multiple payload shapes. Vectorize v2 has
 * evolved over time, so this wrapper defends against subtle API changes.
 */
export async function vectorizeQueryById(
  env: Env,
  id: string,
  k: number
): Promise<{ matches?: Array<Record<string, unknown>> } | null> {
  const options = { topK: k, returnMetadata: true, includeMetadata: true } as Record<string, unknown>;
  const shapes: Array<{ desc: string; payload: Record<string, unknown> }> = [
    { desc: "{ id }", payload: { id, ...options } },
    { desc: "{ vectorId }", payload: { vectorId: id, ...options } },
    { desc: "{ vector: { id } }", payload: { vector: { id }, ...options } },
  ];

  let lastErr: unknown = null;
  for (const shape of shapes) {
    try {
      // @ts-ignore Vectorize typings are permissive in Workers
      const output = await time(`VECTORIZE.query(byId:${shape.desc})`, () => env.VECTORIZE.query(shape.payload));
      if (output && Array.isArray(output.matches)) return output;
      lastErr = lastErr || new Error("No matches returned");
    } catch (error: any) {
      lastErr = error;
      const msg = String(error?.message || error);
      if (msg.includes("40006") || msg.toLowerCase().includes("invalid query vector")) {
        log("[vectorizeQueryById] failed", shape.desc, "for id", id, "->", msg);
        continue;
      }
      throw error;
    }
  }
  if (lastErr) throw lastErr;
  return null;
}

function isNumArray(value: unknown): value is number[] {
  return Array.isArray(value) && value.every((x) => typeof x === "number" && Number.isFinite(x));
}

function pluckEmbedding(raw: unknown, dims: number): number[] | null {
  if (Array.isArray(raw) && isNumArray(raw) && raw.length === dims) return raw;
  if (raw && typeof raw === "object") {
    const data = (raw as any).data;
    if (Array.isArray(data) && isNumArray(data) && data.length === dims) return data;
    if (Array.isArray(data) && data.length > 0 && Array.isArray(data[0]) && isNumArray(data[0]) && data[0].length === dims) return data[0];
    if (Array.isArray(data) && data.length > 0 && typeof data[0] === "object") {
      const embedding = (data[0] as any).embedding;
      if (Array.isArray(embedding) && isNumArray(embedding) && embedding.length === dims) return embedding;
    }
    const embeddings = (raw as any).embeddings;
    if (Array.isArray(embeddings) && isNumArray(embeddings) && embeddings.length === dims) return embeddings;
    if (Array.isArray(embeddings) && embeddings.length > 0 && Array.isArray(embeddings[0]) && isNumArray(embeddings[0]) && embeddings[0].length === dims) return embeddings[0];
  }
  return null;
}

/** Use the Cloudflare AI binding to generate an embedding vector. */
export async function embed(env: Env, model: string, dims: number, text: string): Promise<number[]> {
  const raw = await time("AI.run(embed)", () => env.AI.run(model as any, { text }));
  const arr = pluckEmbedding(raw, dims);
  if (!arr) throw new Error("Embedding dims mismatch or invalid shape");
  return arr;
}

/**
 * Try to reuse embeddings from KV when enabled, otherwise fall back to the live
 * model call. The waitUntil branch queues writes when running inside Workers.
 */
export async function cachedEmbed(
  env: Env,
  model: string,
  dims: number,
  text: string,
  ctx?: ExecutionContext,
  useCache = false
): Promise<number[]> {
  const key = `qemb:${await sha1Hex(text)}`;

  if (!useCache) {
    return await time("embed(cachedEmbed)", () => embed(env, model, dims, text));
  }

  try {
    const cached = await env.CONFIG.get<number[]>(key, "json");
    if (cached && Array.isArray(cached) && cached.length === dims) {
      log("[cachedEmbed] HIT", key);
      return cached;
    }
  } catch (error) {
    log("[cachedEmbed] READ-ERROR", key, String((error as Error)?.message || error));
  }

  const embedding = await time("embed(cachedEmbed)", () => embed(env, model, dims, text));
  log("[cachedEmbed] MISS", key, "(computed)");

  try {
    if (ctx?.waitUntil && env.CONFIG.put) {
      ctx.waitUntil(env.CONFIG.put(key, JSON.stringify(embedding), { expirationTtl: 86400 }));
      log("[cachedEmbed] STORE-QUEUED", key, "ttl=86400");
    } else if (env.CONFIG.put) {
      const stop = startTimer("KV put qemb");
      await env.CONFIG.put(key, JSON.stringify(embedding), { expirationTtl: 86400 });
      stop();
      log("[cachedEmbed] STORED", key, "ttl=86400");
    }
  } catch {
    log("[cachedEmbed] STORE-SKIP", key, "KV not writable or put() unavailable");
  }

  return embedding;
}

/** Reorder matches according to weight-based scoring rules. */
export function reRankMatches(matches: SearchMatch[], weights?: Record<string, number>): SearchMatch[] {
  if (!weights || !Object.keys(weights).length) return matches;
  const baseWeight = Number(weights["base"] ?? weights["score"] ?? 1);
  if (!Number.isFinite(baseWeight) || baseWeight === 0) {
    return matches;
  }

  const scored = matches.map((match) => {
    const metadata = (match.metadata || {}) as Record<string, unknown>;
    let extra = 0;
    for (const [key, weightRaw] of Object.entries(weights)) {
      if (key === "base" || key === "score") continue;
      const weight = Number(weightRaw);
      if (!Number.isFinite(weight) || weight === 0) continue;
      const value = metadata[key];
      if (typeof value === "number" && Number.isFinite(value)) {
        extra += value * weight;
      } else if (typeof value === "string") {
        const parsed = Number(value);
        if (Number.isFinite(parsed)) extra += parsed * weight;
      }
    }
    const totalScore = match.score * baseWeight + extra;
    return { match, totalScore };
  });

  scored.sort((a, b) => b.totalScore - a.totalScore);
  return scored.map((item) => item.match);
}
