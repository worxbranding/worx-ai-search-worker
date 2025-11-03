import type { SiteConfig } from "../lib/types";
import type { TrainingEndpointKey } from "./config";
import {
  resolveTrainingEndpoint,
  resolveTrainingHeaders,
  resolveTrainingTimeoutMs,
} from "./config";

export interface TrainingForwardResult<T = unknown> {
  ok: boolean;
  status: number;
  upstream: T | string | null;
}

export async function forwardTrainingRequest<T = unknown>(
  cfg: SiteConfig,
  key: TrainingEndpointKey,
  fallbackPath: string,
  payload: unknown
): Promise<TrainingForwardResult<T> | null> {
  const endpoint = resolveTrainingEndpoint(cfg, key, fallbackPath);
  if (!endpoint) return null;

  const headers = resolveTrainingHeaders(cfg, key);
  const timeoutMs = resolveTrainingTimeoutMs(cfg, key);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const status = res.status;
    const text = await res.text();
    let upstream: T | string | null = null;
    if (text) {
      try {
        upstream = JSON.parse(text) as T;
      } catch {
        upstream = text;
      }
    }
    return { ok: res.ok, status, upstream };
  } finally {
    clearTimeout(timeout);
  }
}
