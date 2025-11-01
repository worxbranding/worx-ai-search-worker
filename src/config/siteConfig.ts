import { json } from "../http/response";
import type { Env, SiteConfig } from "../lib/types";

/**
 * Load the site configuration document from KV and perform a couple of sanity
 * checks so downstream code can assume required fields exist.
 */
export async function loadSiteConfig(env: Env, site: string): Promise<SiteConfig> {
  const key = `cfg:${site}`;
  const cfg = await env.CONFIG.get<SiteConfig>(key, "json");
  if (!cfg) throw new Error(`Missing CONFIG KV entry for ${key}`);
  if (!cfg.vectorize?.dims) throw new Error(`CONFIG ${key} missing vectorize.dims`);
  if (!cfg.ai?.embed_model) throw new Error(`CONFIG ${key} missing ai.embed_model`);
  return cfg;
}

/**
 * Pick an allowed CORS origin by combining site-level overrides and package
 * defaults. Returns null when nothing should be appended to the response.
 */
export function allowOrigin(origin: string | null, cfg: SiteConfig, env: Env): string | null {
  const fromCfg = cfg.search?.allowed_origins || [];
  let allowed = fromCfg.length
    ? fromCfg
    : (env.ALLOWED_ORIGINS || "")
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
  if (!origin) return null;
  if (allowed.includes("*")) return "*";
  return allowed.includes(origin) ? origin : null;
}

/**
 * Decide whether caching is enabled for a given request. Query string toggles
 * take precedence over stored configuration, defaulting to false.
 */
export function resolveCaching(url: URL, cfg: SiteConfig): boolean {
  const raw = (url.searchParams.get("caching") || "").trim();
  if (raw === "1") return true;
  if (raw === "0") return false;
  return !!cfg.search?.caching;
}

/** Return the API key expected for search endpoints, if any. */
export function wantApiKey(cfg: SiteConfig): string {
  return (cfg.search?.api_key || cfg.api_key || "").trim();
}

/**
 * Enforce the API key header when a key is configured. Returns a response that
 * callers can send directly when the key is missing or incorrect.
 */
export async function requireApiKey(req: Request, cfg: SiteConfig): Promise<Response | null> {
  const want = wantApiKey(cfg);
  if (!want) return null;
  const got = (req.headers.get("x-api-key") || "").trim();
  if (got === want) return null;
  return json({ ok: false, error: "API key required or invalid" }, { status: 401 });
}

/** Guard the vector topK value so callers cannot request outrageous limits. */
export function clampTopK(value: number, min = 1, max = 24): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(Math.max(Math.floor(value), min), max);
}

/**
 * Read the topK request value from several alternate query string names, and
 * fall back to the provided configuration default when it is missing.
 */
export function resolveTopKFromQuery(url: URL, fallback: number): number {
  const candidates = [
    url.searchParams.get("k"),
    url.searchParams.get("topK"),
    url.searchParams.get("top_k"),
    url.searchParams.get("limit"),
  ];
  for (const raw of candidates) {
    if (!raw) continue;
    const n = Number(raw);
    if (!Number.isFinite(n)) continue;
    const clamped = clampTopK(n);
    if (clamped > 0) return clamped;
  }
  return clampTopK(fallback);
}

/**
 * Keep the chat temperature within the model's supported bounds while allowing
 * per-request overrides.
 */
export function clampTemperature(value: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(1, Math.max(0, Number(value)));
}
