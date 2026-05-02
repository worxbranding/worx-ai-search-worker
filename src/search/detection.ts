import type { CustomIntent, SearchMatch } from "../lib/types";

/**
 * Score-based intent detection.
 *
 * Each candidate intent gets a single score combining three signals:
 *   score = embeddingSimilarity            // 0..1, primary
 *         + cappedKeywordBoost             // up to KW_BOOST_CAP
 *         + maxResultMetadataEvidence      // up to ~META_WEIGHTS.path_starts_with
 *
 * The argmax wins iff it clears an absolute threshold AND beats the
 * runner-up by an ambiguity margin. Otherwise we fall back to Default.
 *
 * No priority field. Different intents answer different questions; the
 * tiebreak should be query-relevance evidence, not an admin-set integer.
 *
 * The not-found short-circuit (top vector score below floor / low spread)
 * lives in pipeline.ts and runs before this — system intents like
 * "Not Found" and "Default" are never scored here.
 */

const ABS_THRESHOLD = 0.55;
const AMBIGUITY_MARGIN = 0.02;

// Per-criterion metadata weights. Intentionally asymmetric: a path-prefix
// match is strong evidence; a generic page_kind is weak. Halved from
// initial values because BGE vector hits are typically ~0.6, so a full
// evidence stack (~0.18) was producing 0.10+ metadata boosts that swamped
// the embedding/keyword signal for queries that touched a topic-specific
// page even tangentially.
const META_WEIGHTS = {
  path_starts_with: 0.10,
  collection: 0.05,
  title_contains: 0.04,
  page_kind: 0.025,
} as const;

// Keyword boost is bounded so a verbose intent with 30 keywords can't
// swamp a tighter intent's embedding similarity advantage. Cap raised
// to 0.30 so explicit-pattern intents (like Page Lists with "show me
// your X" matches) can outweigh a topic-specific intent's metadata pull.
const KW_WEIGHT_MULTI = 0.10;   // multi-word phrase match (substring)
const KW_WEIGHT_LOOSE = 0.05;   // multi-word with all tokens word-matched but not contiguous
const KW_WEIGHT_SINGLE = 0.05;  // single-word match (with word boundaries)
const KW_BOOST_CAP = 0.30;

interface ScoredIntent {
  intent: CustomIntent;
  score: number;
  components: { embedding: number; keyword: number; metadata: number };
}

export interface IntentDetectionResult {
  intent: CustomIntent | null;
  reason: "matched" | "below_threshold" | "ambiguous" | "no_intents";
  score?: number;
  components?: { embedding: number; keyword: number; metadata: number };
  /** Threshold + margin used in this evaluation (so the dashboard can show why something fell short). */
  threshold?: number;
  margin?: number;
  /** Top scored candidates for telemetry / dashboard diagnostics. */
  top_intents?: Array<{ name: string; score: number; components: ScoredIntent["components"] }>;
}

/** Cosine similarity between two equal-length vectors. */
function cosineSim(a: number[], b: number[]): number {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length === 0 || a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function matchesAsWord(haystack: string, needle: string): boolean {
  let pos = haystack.indexOf(needle);
  while (pos !== -1) {
    const before = pos === 0 || /[\s\p{P}]/u.test(haystack[pos - 1]);
    const afterIdx = pos + needle.length;
    const after = afterIdx >= haystack.length || /[\s\p{P}]/u.test(haystack[afterIdx]);
    if (before && after) return true;
    pos = haystack.indexOf(needle, pos + 1);
  }
  return false;
}

function keywordBoost(queryLower: string, intent: CustomIntent): number {
  const keywords = intent.detection?.keywords;
  if (!keywords || keywords.length === 0) return 0;
  let multi = 0;
  let loose = 0;
  let single = 0;
  for (const kw of keywords) {
    if (!kw) continue;
    const k = kw.toLowerCase().trim();
    if (!k) continue;
    if (k.includes(" ")) {
      // Multi-word: prefer contiguous substring match. Otherwise count
      // it as "loose" if every token in the keyword phrase appears as a
      // word in the query (handles "your process" → "your design process").
      if (queryLower.includes(k)) {
        multi += 1;
      } else {
        const tokens = k.split(/\s+/).filter(Boolean);
        if (tokens.length >= 2 && tokens.every((t) => matchesAsWord(queryLower, t))) {
          loose += 1;
        }
      }
    } else {
      if (matchesAsWord(queryLower, k)) single += 1;
    }
  }
  return Math.min(
    KW_BOOST_CAP,
    multi * KW_WEIGHT_MULTI + loose * KW_WEIGHT_LOOSE + single * KW_WEIGHT_SINGLE
  );
}

/**
 * Per-criterion graded evidence for one (intent, result) pair.
 * Unlike the previous implementation, this returns *partial credit* rather
 * than a binary AND across all criteria — so a strong path-prefix signal
 * survives even when collection/page_kind don't line up.
 */
function metadataEvidence(intent: CustomIntent, metadata: Record<string, unknown> | undefined): number {
  const c = intent.detection?.metadata_matches;
  if (!c || !metadata) return 0;
  let evidence = 0;
  if (c.path_starts_with) {
    const path = String((metadata.path as string) || (metadata.url as string) || "").toLowerCase();
    if (path && path.startsWith(c.path_starts_with.toLowerCase())) evidence += META_WEIGHTS.path_starts_with;
  }
  if (c.collection) {
    const col = String(metadata.collection || "").toLowerCase();
    if (col === c.collection.toLowerCase()) evidence += META_WEIGHTS.collection;
  }
  if (c.title_contains && c.title_contains.length > 0) {
    const t = String(metadata.title || "").toLowerCase();
    for (const s of c.title_contains) {
      if (s && t.includes(s.toLowerCase())) {
        evidence += META_WEIGHTS.title_contains;
        break;
      }
    }
  }
  if (c.page_kind) {
    const pk = String(metadata.page_kind || "").toLowerCase();
    if (pk === c.page_kind.toLowerCase()) evidence += META_WEIGHTS.page_kind;
  }
  return evidence;
}

/**
 * Metadata boost for an intent: max over the top-3 results of
 * (resultScore × metadataEvidence). Weighting by the result's own vector
 * score is what stops a marginal vector hit with the right `page_kind`
 * from hijacking routing.
 */
function metadataBoost(intent: CustomIntent, results: SearchMatch[]): number {
  if (!results || results.length === 0) return 0;
  if (!intent.detection?.metadata_matches) return 0;
  let max = 0;
  const top = results.slice(0, 3);
  for (const r of top) {
    const e = metadataEvidence(intent, r.metadata);
    if (e === 0) continue;
    const contribution = Number(r.score ?? 0) * e;
    if (contribution > max) max = contribution;
  }
  return max;
}

/**
 * Run the unified scored detection against a list of intents.
 * System intents (`is_system === true`) and disabled intents are excluded
 * from scoring — Default and Not Found are routed deterministically.
 */
export function detectIntent(
  query: string,
  queryEmbedding: number[] | null | undefined,
  results: SearchMatch[],
  customIntents: CustomIntent[],
  options?: { threshold?: number; margin?: number }
): IntentDetectionResult {
  if (!customIntents || customIntents.length === 0) {
    return { intent: null, reason: "no_intents" };
  }

  const candidates = customIntents.filter((i) => i.enabled !== false && i.is_system !== true);
  if (candidates.length === 0) {
    return { intent: null, reason: "no_intents" };
  }

  const threshold = options?.threshold ?? ABS_THRESHOLD;
  const margin = options?.margin ?? AMBIGUITY_MARGIN;
  const queryLower = query.toLowerCase().trim();

  const scored: ScoredIntent[] = candidates.map((intent) => {
    const emb =
      queryEmbedding &&
      Array.isArray(intent.detection_embedding) &&
      intent.detection_embedding.length === queryEmbedding.length
        ? cosineSim(queryEmbedding, intent.detection_embedding as number[])
        : 0;
    const kw = keywordBoost(queryLower, intent);
    const meta = metadataBoost(intent, results);
    return { intent, score: emb + kw + meta, components: { embedding: emb, keyword: kw, metadata: meta } };
  });

  scored.sort((a, b) => b.score - a.score);
  const top1 = scored[0];
  const top2 = scored[1];
  const topSummary = scored.slice(0, 5).map((s) => ({
    name: s.intent.name,
    score: Number(s.score.toFixed(4)),
    components: {
      embedding: Number(s.components.embedding.toFixed(4)),
      keyword: Number(s.components.keyword.toFixed(4)),
      metadata: Number(s.components.metadata.toFixed(4)),
    },
  }));

  if (!top1 || top1.score < threshold) {
    return {
      intent: null,
      reason: "below_threshold",
      score: top1?.score,
      components: top1?.components,
      threshold,
      margin,
      top_intents: topSummary,
    };
  }
  if (top2 && top1.score - top2.score < margin) {
    return {
      intent: null,
      reason: "ambiguous",
      score: top1.score,
      components: top1.components,
      threshold,
      margin,
      top_intents: topSummary,
    };
  }
  return {
    intent: top1.intent,
    reason: "matched",
    score: top1.score,
    components: top1.components,
    threshold,
    margin,
    top_intents: topSummary,
  };
}

/**
 * Default fallback intent. Behavior defaults to long_form_answer when the
 * site config does not specify one — pipeline.ts passes
 * cfg.default_behavior so the admin's setting wins.
 */
export function getDefaultIntent(behavior?: string): CustomIntent {
  return {
    name: "default",
    response_behavior: behavior || "long_form_answer",
    priority: 0,
    enabled: true,
    detection: {},
  };
}
