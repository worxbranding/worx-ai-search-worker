import type { SiteConfig } from "./types";

/**
 * Default values for the tunable knobs in `cfg.search.behavior_caps`,
 * `cfg.search.rerank_weights`, and `cfg.search.detection`. Centralized
 * here so behaviors / pipeline / rerank don't each redeclare them, and so
 * a single edit changes the cluster-wide default for every site that
 * doesn't override the value in its KV config row.
 *
 * Each helper reads `cfg.search.<group>?.<key>` and falls back to the
 * default below. Per-intent overrides are applied earlier (via
 * `mergeIntentTuning` in lib/llm.ts) — by the time these helpers run,
 * the per-intent value has already been folded into `cfg.search`.
 */

/** Behavior token caps — see SiteConfig.search.behavior_caps for documentation. */
export const BEHAVIOR_CAP_DEFAULTS = {
  short_answer: 150,
  short_blurb_with_list: 256,
  short_blurb_with_list_fallback: 384,
  medium_answer: 300,
  long_form_answer: 600,
  long_form_answer_floor: 300,
  long_form_answer_default: 500,
  detailed_explanation: 800,
  detailed_explanation_floor: 300,
  detailed_explanation_default: 600,
  single_page_summary: 500,
  single_page_summary_floor: 200,
  single_page_summary_default: 400,
  comparison: 1536,
  comparison_floor: 384,
  comparison_default: 896,
  collection_overview: 512,
  recent_items_blurb: 256,
  recent_items_fallback: 384,
  navigation_help: 256,
} as const;

/** Re-rank weights — see SiteConfig.search.rerank_weights for documentation. */
export const RERANK_WEIGHT_DEFAULTS = {
  metadata_collection: 0.10,
  metadata_page_kind: 0.05,
  metadata_exact_path: 0.50,
  metadata_path_prefix: 0.30,
  metadata_title_contains: 0.15,
  query_token_in_title: 0.20,
  query_token_in_path: 0.05,
  query_token_max_total: 0.40,
  keyword_in_title: 0.25,
  keyword_in_preview: 0.15,
  keyword_in_full_text: 0.20,
  keyword_in_path: 0.05,
  multi_keyword_per_match: 0.10,
} as const;

/** Intent detection knobs — see SiteConfig.search.detection for documentation. */
export const DETECTION_DEFAULTS = {
  high_confidence_score: 0.68,
  candidate_slice: 8,
} as const;

/** Resolve a behavior_caps value from cfg with default fallback. */
export function behaviorCap(
  cfg: SiteConfig | null | undefined,
  key: keyof typeof BEHAVIOR_CAP_DEFAULTS,
): number {
  const v = cfg?.search?.behavior_caps?.[key];
  return typeof v === "number" && Number.isFinite(v) ? v : BEHAVIOR_CAP_DEFAULTS[key];
}

/** Resolve a rerank_weights value from cfg with default fallback. */
export function rerankWeight(
  cfg: SiteConfig | null | undefined,
  key: keyof typeof RERANK_WEIGHT_DEFAULTS,
): number {
  const v = cfg?.search?.rerank_weights?.[key];
  return typeof v === "number" && Number.isFinite(v) ? v : RERANK_WEIGHT_DEFAULTS[key];
}

/** Resolve a detection value from cfg with default fallback. */
export function detectionValue(
  cfg: SiteConfig | null | undefined,
  key: keyof typeof DETECTION_DEFAULTS,
): number {
  const v = cfg?.search?.detection?.[key];
  return typeof v === "number" && Number.isFinite(v) ? v : DETECTION_DEFAULTS[key];
}

/**
 * Resolve max_tokens for the long-form / multi-paragraph behaviors.
 *
 * Precedence:
 *   1. `cfg.search.max_output_tokens` (already merged from per-intent
 *      override via mergeIntentTuning) — if the CMS user explicitly set
 *      this on the intent, it wins.
 *   2. `cfg.search.behavior_caps.<behavior>_default` (config-driven baseline)
 *      → falls back to BEHAVIOR_CAP_DEFAULTS.
 *
 * Then clamped into `[<behavior>_floor, <behavior>]` so a wildly out-of-
 * range value (e.g. someone typing 99999 into the CMS) doesn't crash the
 * provider. The floor/ceiling pair are themselves config-overridable, so
 * a site that wants to allow really long answers can lift the ceiling.
 */
export function resolveBehaviorMaxTokens(
  cfg: SiteConfig | null | undefined,
  key: "long_form_answer" | "detailed_explanation" | "single_page_summary" | "comparison",
): number {
  const ceilingKey = key;
  const floorKey = `${key}_floor` as keyof typeof BEHAVIOR_CAP_DEFAULTS;
  const defaultKey = `${key}_default` as keyof typeof BEHAVIOR_CAP_DEFAULTS;
  const ceiling = behaviorCap(cfg, ceilingKey);
  const floor = behaviorCap(cfg, floorKey);
  const dflt = behaviorCap(cfg, defaultKey);
  const requested = Number(cfg?.search?.max_output_tokens ?? dflt);
  return Math.max(floor, Math.min(ceiling, requested));
}

/**
 * Resolve max_tokens for "single-cap" behaviors (ShortAnswer, ShortBlurb,
 * MediumAnswer, NavigationHelp, RecentItems, CollectionOverview). These
 * don't have a floor/ceiling range — they're style-defining ("ShortAnswer
 * is short"), and the cap is just a default the per-intent value
 * overrides.
 *
 * Precedence:
 *   1. `cfg.search.max_output_tokens` (per-intent override, when set)
 *   2. `cfg.search.behavior_caps.<key>` → BEHAVIOR_CAP_DEFAULTS[key]
 */
export function resolveSimpleBehaviorMaxTokens(
  cfg: SiteConfig | null | undefined,
  key: keyof typeof BEHAVIOR_CAP_DEFAULTS,
): number {
  return Number(cfg?.search?.max_output_tokens ?? behaviorCap(cfg, key));
}
