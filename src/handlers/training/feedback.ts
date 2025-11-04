import { json } from "../../http/response";
import { startTimer } from "../../lib/logging";
import type { Env, ExecutionContext, SiteConfig } from "../../lib/types";
import type { TrainingFeedbackPayload } from "../../training/types";
import { forwardTrainingRequest } from "../../training/proxy";

function parseString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function parseVote(value: unknown): string {
  const normalized = parseString(value).toLowerCase();
  if (["up", "down", "neutral"].includes(normalized)) return normalized;
  return "";
}

export async function handleTrainingFeedback(
  req: Request,
  _env: Env,
  cfg: SiteConfig,
  _ctx?: ExecutionContext
): Promise<Response> {
  const stop = startTimer("handleTrainingFeedback");
  const body = (await req.json().catch(() => ({}))) as TrainingFeedbackPayload & {
    session_id?: string;
    vote?: string;
    trainer_id?: string;
    metadata?: Record<string, unknown>;
  };

  const url = new URL(req.url);
  let site = parseString(body.site);
  if (!site) site = parseString(url.searchParams.get("site"));
  let sessionId = parseString(body.sessionId || body.session_id);
  if (!sessionId) sessionId = parseString(url.searchParams.get("sessionId") || url.searchParams.get("session_id"));
  let vote = parseVote(body.vote);
  if (!vote) vote = parseVote(url.searchParams.get("vote"));
  if (!site || !sessionId || !vote) {
    stop();
    return json(
      { ok: false, error: "Fields 'site', 'sessionId', and 'vote' are required" },
      { status: 400 }
    );
  }

  const forwardPayload: Record<string, unknown> = {
    site,
    sessionId,
    vote,
  };
  if (typeof body.note === "string" && body.note.trim()) {
    forwardPayload.note = body.note.trim();
  }
  if (Array.isArray(body.tags)) {
    forwardPayload.tags = body.tags;
  }
  if (body.metadata && typeof body.metadata === "object" && !Array.isArray(body.metadata)) {
    forwardPayload.metadata = body.metadata;
  }
  const trainerId = parseString(body.trainerId) || parseString(body.trainer_id);
  if (trainerId) {
    forwardPayload.trainerId = trainerId;
  }

  try {
    const result = await forwardTrainingRequest(cfg, "feedback", "api/training/feedback", forwardPayload);
    if (!result) {
      stop();
      return json({ ok: false, error: "Training feedback endpoint not configured" }, { status: 501 });
    }
    stop();
    return json(
      {
        ok: result.ok,
        status: result.status,
        upstream: result.upstream,
      },
      { status: result.status }
    );
  } catch (error) {
    stop();
    return json(
      { ok: false, error: String((error as Error)?.message || error) },
      { status: 502 }
    );
  }
}
