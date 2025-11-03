import { log } from "../lib/logging";
import type { Env, ExecutionContext, SiteConfig } from "../lib/types";
import type { TrainingSessionLogPayload } from "./types";
import {
  resolveTrainingEndpoint,
  resolveTrainingHeaders,
  resolveTrainingTimeoutMs,
} from "./config";

function resolveCacheTtl(cfg: SiteConfig): number {
  const loggerCfg = (cfg.training?.logger || {}) as Record<string, unknown>;
  const raw = Number(loggerCfg["ttlSeconds"] ?? 3600);
  if (Number.isFinite(raw) && raw > 0) return raw;
  return 3600;
}

function resolveCachePrefix(cfg: SiteConfig): string | null {
  const loggerCfg = (cfg.training?.logger || {}) as Record<string, unknown>;
  const raw = typeof loggerCfg["cacheKeyPrefix"] === "string" ? (loggerCfg["cacheKeyPrefix"] as string) : "";
  return raw.trim() || null;
}

async function persistToCache(
  env: Env,
  cfg: SiteConfig,
  payload: TrainingSessionLogPayload,
  ctx?: ExecutionContext
): Promise<void> {
  const ns = env.TRAINING_CACHE || env.CONFIG;
  if (!ns?.put) return;
  const prefix = resolveCachePrefix(cfg);
  if (!prefix) return;

  const key = `${prefix}${payload.site}:${Date.now()}:${Math.random().toString(16).slice(2)}`;
  const ttl = resolveCacheTtl(cfg);
  const body = JSON.stringify(payload);

  const write = async () => {
    try {
      await ns.put!(key, body, { expirationTtl: ttl });
      log("[sessionLogger] cached", key, `ttl=${ttl}`);
    } catch (error) {
      log("[sessionLogger] cache error", key, String((error as Error)?.message || error));
    }
  };

  if (ctx?.waitUntil) {
    ctx.waitUntil(write());
  } else {
    await write();
  }
}

async function dispatchToEndpoint(
  endpoint: string,
  headers: Record<string, string>,
  payload: TrainingSessionLogPayload,
  timeoutMs: number
): Promise<void> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!res.ok) {
      log("[sessionLogger] upstream non-2xx", endpoint, `status=${res.status}`);
    }
  } catch (error) {
    log("[sessionLogger] upstream error", endpoint, String((error as Error)?.message || error));
  } finally {
    clearTimeout(timeout);
  }
}

export async function logTrainingSession(
  env: Env,
  cfg: SiteConfig,
  payload: TrainingSessionLogPayload,
  ctx?: ExecutionContext
): Promise<void> {
  await persistToCache(env, cfg, payload, ctx);

  const endpoint = resolveTrainingEndpoint(cfg, "session", "/api/training/session-log");
  if (!endpoint) {
    log("[sessionLogger] skip: endpoint not configured");
    return;
  }

  const headers = resolveTrainingHeaders(cfg, "session");
  const timeoutMs = resolveTrainingTimeoutMs(cfg, "session");

  const send = async () => {
    await dispatchToEndpoint(endpoint, headers, payload, timeoutMs);
  };

  if (ctx?.waitUntil) {
    ctx.waitUntil(send());
    return;
  }
  await send();
}
