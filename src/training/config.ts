import type { SiteConfig } from "../lib/types";

export type TrainingEndpointKey = "session" | "feedback" | "publish" | "replay" | "policy";

function coerceHeaders(value: unknown): Record<string, string> | null {
  if (!value || typeof value !== "object") return null;
  const output: Record<string, string> = {};
  for (const [key, raw] of Object.entries(value as Record<string, unknown>)) {
    if (typeof raw === "string") {
      output[key] = raw;
    }
  }
  return Object.keys(output).length ? output : null;
}

export function resolveTrainingEndpoint(
  cfg: SiteConfig,
  key: TrainingEndpointKey,
  fallbackPath: string
): string | null {
  const training = cfg.training || {};
  const directKey = `${key}Endpoint`;
  const directRaw = typeof (training as any)[directKey] === "string" ? (training as any)[directKey] : "";
  if (directRaw && directRaw.trim()) {
    return directRaw.trim();
  }

  if (key === "session" && training.logger && typeof training.logger === "object") {
    const loggerEndpoint =
      typeof (training.logger as Record<string, unknown>)["endpoint"] === "string"
        ? ((training.logger as Record<string, unknown>)["endpoint"] as string)
        : "";
    if (loggerEndpoint && loggerEndpoint.trim()) {
      return loggerEndpoint.trim();
    }
  }

  const apiBaseRaw = typeof training.apiBase === "string" ? training.apiBase : "";
  if (!apiBaseRaw || !apiBaseRaw.trim()) return null;
  try {
    const baseString = apiBaseRaw.trim().endsWith("/")
      ? apiBaseRaw.trim()
      : `${apiBaseRaw.trim()}/`;
    const baseUrl = new URL(baseString);
    const normalized = (fallbackPath || "").trim();
    if (!normalized) {
      return baseUrl.toString();
    }
    const resolved = new URL(normalized, baseUrl);
    return resolved.toString();
  } catch {
    return null;
  }
}

export function resolveTrainingHeaders(
  cfg: SiteConfig,
  key?: TrainingEndpointKey,
  overrides?: Record<string, string>
): Record<string, string> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  const training = cfg.training || {};

  const shared = coerceHeaders((training as any).headers);
  if (shared) {
    Object.assign(headers, shared);
  }

  if (key === "session" && training.logger && typeof training.logger === "object") {
    const loggerHeaders = coerceHeaders((training.logger as Record<string, unknown>)["headers"]);
    if (loggerHeaders) {
      Object.assign(headers, loggerHeaders);
    }
  }

  if (key) {
    const specific = coerceHeaders((training as any)[`${key}Headers`]);
    if (specific) {
      Object.assign(headers, specific);
    }
  }

  const apiKey = typeof training.apiKey === "string" ? training.apiKey.trim() : "";
  if (apiKey) {
    headers["x-api-key"] = apiKey;
  }

  const basicAuth = (training as any).basicAuth as { user?: string; pass?: string } | undefined;
  if (basicAuth && basicAuth.user && basicAuth.pass) {
    headers["Authorization"] = "Basic " + btoa(`${basicAuth.user}:${basicAuth.pass}`);
  } else {
    const devBasic = (cfg as any).dev_basic_auth as
      | { enabled?: boolean; user?: string; pass?: string }
      | undefined;
    if (devBasic && devBasic.enabled && devBasic.user && devBasic.pass) {
      headers["Authorization"] = "Basic " + btoa(`${devBasic.user}:${devBasic.pass}`);
    }
  }

  if (overrides) {
    Object.assign(headers, overrides);
  }

  return headers;
}

export function resolveTrainingTimeoutMs(cfg: SiteConfig, key?: TrainingEndpointKey): number {
  const training = cfg.training || {};
  if (key) {
    const specific = Number((training as any)[`${key}TimeoutMs`]);
    if (Number.isFinite(specific) && specific > 0) return specific;
  }

  const general = Number((training as any).timeoutMs);
  if (Number.isFinite(general) && general > 0) return general;

  const loggerTimeout = Number(training.logger?.timeoutMs);
  if (Number.isFinite(loggerTimeout) && loggerTimeout > 0) return loggerTimeout;

  return 5000;
}
