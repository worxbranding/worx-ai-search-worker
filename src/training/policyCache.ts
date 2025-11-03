import { log } from "../lib/logging";
import type { Env, SiteConfig } from "../lib/types";
import type {
  TrainingPolicyCacheOptions,
  TrainingPolicyPromptVariant,
  TrainingPolicySnapshot,
} from "./types";

type PolicyCacheEntry = {
  value: TrainingPolicySnapshot | null;
  expiresAt: number;
};

const POLICY_CACHE = new Map<string, PolicyCacheEntry>();
const DEFAULT_CACHE_TTL_MS = 60_000;

function resolvePolicyKey(site: string, cfg: SiteConfig): string {
  const policyCfg = (cfg.training?.policy || {}) as Record<string, unknown>;
  const activeKeyRaw = typeof policyCfg["activeKey"] === "string" ? (policyCfg["activeKey"] as string) : "";
  if (activeKeyRaw.trim()) {
    return activeKeyRaw.replace(/{site}/gi, site);
  }
  const keyPrefixRaw = typeof policyCfg["keyPrefix"] === "string" ? (policyCfg["keyPrefix"] as string) : "";
  const prefix = keyPrefixRaw.trim() || "policy:";
  let key = `${prefix}${site}`;
  if (!key.includes(":active")) {
    key = key.endsWith(":") ? `${key}active` : `${key}:active`;
  }
  return key;
}

function resolveCacheTtlMs(cfg: SiteConfig, overrides?: TrainingPolicyCacheOptions): number {
  const fromOverrides = overrides?.cacheTtlMs;
  if (typeof fromOverrides === "number" && Number.isFinite(fromOverrides) && fromOverrides > 0) {
    return fromOverrides;
  }
  const fromCfg = Number((cfg.training?.policy || {})["cacheTtlSeconds"] ?? 60);
  if (Number.isFinite(fromCfg) && fromCfg > 0) {
    return fromCfg * 1000;
  }
  return DEFAULT_CACHE_TTL_MS;
}

async function fetchPolicySnapshot(
  env: Env,
  cfg: SiteConfig,
  site: string
): Promise<TrainingPolicySnapshot | null> {
  const key = resolvePolicyKey(site, cfg);
  try {
    const snapshot = await env.CONFIG.get<TrainingPolicySnapshot>(key, "json");
    if (!snapshot) {
      log("[policyCache] miss (KV empty)", key);
    }
    return snapshot || null;
  } catch (error) {
    log("[policyCache] fetch error", key, String((error as Error)?.message || error));
    return null;
  }
}

export async function getPolicySnapshot(
  env: Env,
  cfg: SiteConfig,
  site: string,
  opts: TrainingPolicyCacheOptions = {}
): Promise<TrainingPolicySnapshot | null> {
  const cacheKey = site || cfg.site_key || "default";
  const ttlMs = resolveCacheTtlMs(cfg, opts);
  const now = Date.now();
  const cached = POLICY_CACHE.get(cacheKey);
  if (!opts.forceRefresh && cached && cached.expiresAt > now) {
    return cached.value;
  }

  const value = await fetchPolicySnapshot(env, cfg, site);
  POLICY_CACHE.set(cacheKey, { value, expiresAt: now + ttlMs });
  return value;
}

export function pickPromptVariant(
  snapshot: TrainingPolicySnapshot | null,
  candidate?: string | null
): TrainingPolicyPromptVariant | null {
  if (!snapshot?.prompts?.variants) return null;
  const variants = snapshot.prompts.variants;
  const normalizedCandidate = candidate?.trim();
  let resolvedKey: string | undefined;
  if (normalizedCandidate && variants[normalizedCandidate]) {
    resolvedKey = normalizedCandidate;
  } else if (snapshot.prompts.default && variants[snapshot.prompts.default]) {
    resolvedKey = snapshot.prompts.default;
  } else {
    const keys = Object.keys(variants);
    resolvedKey = keys.length ? keys[0] : undefined;
  }
  if (!resolvedKey) return null;
  const variant = variants[resolvedKey] || null;
  if (!variant) return null;
  return { ...variant, key: variant.key || resolvedKey };
}

export function getReRankWeights(
  cfg: SiteConfig,
  snapshot: TrainingPolicySnapshot | null
): Record<string, number> {
  const fromSnapshot = snapshot?.rerank?.weights;
  if (fromSnapshot && Object.keys(fromSnapshot).length) {
    return fromSnapshot;
  }
  const fromCfg = (cfg.training?.policy?.weights || {}) as Record<string, number>;
  if (fromCfg && Object.keys(fromCfg).length) {
    return fromCfg;
  }
  return {};
}
