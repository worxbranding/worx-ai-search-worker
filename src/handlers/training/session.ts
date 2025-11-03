import { json } from "../../http/response";
import { clampTemperature, clampTopK } from "../../config/siteConfig";
import { startTimer } from "../../lib/logging";
import type { Env, ExecutionContext, SiteConfig } from "../../lib/types";
import { executeAsk } from "../ask";
import { logTrainingSession } from "../../training/sessionLogger";
import { coerceBoolean } from "../../utils/coerce";
import {
  matchesToTrainingDocs,
  type TrainingSessionLogPayload,
  type TrainingSessionRequest,
} from "../../training/types";

export async function handleTrainingSession(
  req: Request,
  env: Env,
  cfg: SiteConfig,
  ctx?: ExecutionContext
): Promise<Response> {
  const sessionUid = typeof crypto.randomUUID === "function"
    ? crypto.randomUUID().replace(/-/g, "")
    : Array.from(crypto.getRandomValues(new Uint8Array(16)))
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");
  const stop = startTimer("handleTrainingSession");
  const body = (await req.json().catch(() => ({}))) as TrainingSessionRequest & {
    promptOverride?: string;
    promptVariant?: string;
    prompt_variant?: string;
    note?: string;
    metadata?: Record<string, unknown>;
    useCache?: boolean;
    disableCaching?: boolean;
  };

  const site = (body.site || "").trim();
  const question = (body.q || "").trim();
  if (!site) {
    stop();
    return json({ ok: false, error: "Missing field 'site'" }, { status: 400 });
  }
  if (!question) {
    stop();
    return json({ ok: false, error: "Missing field 'q'" }, { status: 400 });
  }

  const baseTopK = clampTopK(Number(body.k || cfg.search?.topK || 6));

  let promptOverride = "";
  if (typeof body.promptOverride === "string" && body.promptOverride.trim()) {
    promptOverride = body.promptOverride.trim();
  } else if (typeof body.promptOverride === "string") {
    promptOverride = body.promptOverride;
  }
  if (!promptOverride && typeof body.promptOverride !== "string") {
    if (typeof (body as any).systemPrompt === "string") {
      promptOverride = ((body as any).systemPrompt as string).trim();
    } else if (typeof (body as any).system_prompt === "string") {
      promptOverride = ((body as any).system_prompt as string).trim();
    }
  }

  const promptVariant =
    typeof body.promptVariant === "string"
      ? body.promptVariant.trim()
      : typeof body.prompt_variant === "string"
      ? body.prompt_variant.trim()
      : null;

  const trainerId =
    typeof (body as any).trainerId === "string"
      ? ((body as any).trainerId as string).trim()
      : typeof (body as any).trainer_id === "string"
      ? ((body as any).trainer_id as string).trim()
      : undefined;

  const temperatureCandidateRaw =
    (body as any).chatTemperature ??
    (body as any).temperature ??
    (body as any).chat_temperature ??
    null;
  const configTemperature = Number(cfg.search?.chat_temperature ?? 0.1);
  const temperatureCandidate =
    temperatureCandidateRaw !== null &&
    temperatureCandidateRaw !== undefined &&
    !Number.isNaN(Number(temperatureCandidateRaw))
      ? Number(temperatureCandidateRaw)
      : configTemperature;
  const temperature = clampTemperature(temperatureCandidate, configTemperature);

  const useCache = coerceBoolean((body as any).useCache) ?? false;
  const disableCachingInput = coerceBoolean(body.disableCaching);
  const disableCaching = disableCachingInput ?? !useCache;

  const result = await executeAsk(
    env,
    cfg,
    {
      site,
      question,
      topK: baseTopK,
      promptOverride,
      temperature,
      wantCaching: useCache,
      training: true,
      promptVariant,
      disableCaching,
    },
    ctx
  );

  const trainingDetails = result.trainingMetadata;
  const metadata: Record<string, unknown> = {};
  if (body.metadata && typeof body.metadata === "object" && !Array.isArray(body.metadata)) {
    Object.assign(metadata, body.metadata);
  }
  if (Array.isArray(body.tags)) {
    metadata.tags = body.tags;
  }
  if (promptVariant) {
    metadata.promptVariant = promptVariant;
  }

  const statsRecord: Record<string, unknown> = { ...result.stats };
  const logPayload: TrainingSessionLogPayload = {
    uid: sessionUid,
    site,
    question: result.question,
    answer: result.answer,
    intent: result.intent,
    matches: trainingDetails?.matches || matchesToTrainingDocs(result.selected, result.contexts, site),
    stats: statsRecord,
    prompt: {
      system: trainingDetails?.systemPrompt || result.systemPrompt,
      variant: trainingDetails?.promptVariant?.key || promptVariant,
      temperature: trainingDetails?.temperature ?? temperature,
      model: trainingDetails?.model ?? result.chatModel,
      policyVersion: trainingDetails?.policyVersion ?? null,
      linkHints: result.linkHints,
    },
    contexts: trainingDetails?.contexts || result.contexts,
    trainerId,
    note: typeof body.note === "string" ? body.note : undefined,
    metadata: Object.keys(metadata).length ? metadata : undefined,
    fromCache: trainingDetails?.fromCache ?? false,
  };

  await logTrainingSession(env, cfg, logPayload, ctx);

  const responsePayload: Record<string, unknown> = {
    ok: true,
    sessionId: sessionUid,
    question: result.question,
    answer: result.answer,
    stats: result.stats,
    intent: result.intent,
    documents: trainingDetails?.matches || [],
    contexts: trainingDetails?.contexts || result.contexts,
    prompt: logPayload.prompt,
    policy: trainingDetails?.policySnapshot || null,
    cache: trainingDetails?.fromCache ? "hit" : "miss",
    allowedUrls: result.allowedUrls,
    linkHints: result.linkHints,
  };
  if (trainerId) {
    responsePayload.trainerId = trainerId;
  }
  if (Object.keys(metadata).length) {
    responsePayload.metadata = metadata;
  }

  stop();
  return json(responsePayload);
}
