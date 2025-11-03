import type { SearchMatch } from "../lib/types";

/** Supported training feedback votes. */
export type TrainingFeedbackVote = "up" | "down" | "neutral";

/** Request payload accepted by the training session endpoint. */
export interface TrainingSessionRequest {
  site: string;
  q: string;
  k?: number;
  promptOverride?: string;
  prompt_variant?: string;
  metadata?: Record<string, unknown>;
  trainerId?: string;
  trainer_id?: string;
  tags?: string[];
  disableCaching?: boolean;
  training?: boolean;
}

/** Minimal document summary captured for training session logs. */
export interface TrainingDocumentMatch {
  id: string;
  rank: number;
  score: number;
  metadata?: Record<string, unknown>;
  context?: string | null;
  source?: string | null;
}

/** Shape of the payload sent to Concrete CMS for training session logging. */
export interface TrainingSessionLogPayload {
  uid?: string;
  site: string;
  question: string;
  answer: string;
  intent?: string;
  matches: TrainingDocumentMatch[];
  stats: Record<string, unknown>;
  prompt: {
    system: string;
    variant?: string | null;
    temperature: number;
    model: string;
    policyVersion?: string | null;
    linkHints?: string;
  };
  contexts?: string[];
  trainerId?: string;
  note?: string;
  metadata?: Record<string, unknown>;
  fromCache?: boolean;
}

/** Payload forwarded from the training UI when casting a vote. */
export interface TrainingFeedbackPayload {
  sessionId: string;
  site: string;
  vote: TrainingFeedbackVote;
  note?: string;
  tags?: string[];
  metadata?: Record<string, unknown>;
  trainerId?: string;
}

/** Prompt variant definition contained within a policy snapshot. */
export interface TrainingPolicyPromptVariant {
  key?: string;
  systemPrompt?: string;
  description?: string;
  metadata?: Record<string, unknown>;
}

/** Policy snapshot persisted in CONFIG KV by the ingested worker. */
export interface TrainingPolicySnapshot {
  version?: string;
  site?: string;
  publishedAt?: string;
  prompts?: {
    default?: string;
    variants?: Record<string, TrainingPolicyPromptVariant>;
  };
  rerank?: {
    weights?: Record<string, number>;
    defaultVariant?: string;
    metadata?: Record<string, unknown>;
  };
  guardrails?: {
    riskRules?: Record<string, unknown>;
    metadata?: Record<string, unknown>;
  };
  metadata?: Record<string, unknown>;
}

/** Options for interacting with the local policy cache helper. */
export interface TrainingPolicyCacheOptions {
  forceRefresh?: boolean;
  cacheTtlMs?: number;
}

/** Result returned from the `/ask` helper when training metadata is requested. */
export interface TrainingAskMetadata {
  policyVersion?: string | null;
  policySnapshot?: TrainingPolicySnapshot | null;
  promptVariant?: TrainingPolicyPromptVariant | null;
  matches: Array<TrainingDocumentMatch>;
  contexts: string[];
  systemPrompt: string;
  temperature: number;
  model: string;
  intent: string;
  fromCache: boolean;
}

/** Convenience helper for mapping matches to training document summaries. */
export function matchesToTrainingDocs(
  matches: SearchMatch[],
  contexts: string[],
  site: string
): TrainingDocumentMatch[] {
  return matches.map((match, idx) => ({
    id: match.id,
    rank: idx + 1,
    score: match.score,
    metadata: (match.metadata || {}) as Record<string, unknown>,
    context: contexts[idx] || null,
    source: site,
  }));
}
