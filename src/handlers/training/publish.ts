import { json } from "../../http/response";
import { startTimer } from "../../lib/logging";
import type { Env, ExecutionContext, SiteConfig } from "../../lib/types";
import { forwardTrainingRequest } from "../../training/proxy";

function parseString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

export async function handleTrainingPublish(
  req: Request,
  _env: Env,
  cfg: SiteConfig,
  _ctx?: ExecutionContext
): Promise<Response> {
  const stop = startTimer("handleTrainingPublish");
  const body = (await req.json().catch(() => ({}))) as Record<string, unknown>;
  const site = parseString(body["site"]);
  if (!site) {
    stop();
    return json({ ok: false, error: "Field 'site' is required" }, { status: 400 });
  }

  const forwardPayload: Record<string, unknown> = { ...body, site };
  try {
    const result = await forwardTrainingRequest(cfg, "publish", "/api/training/publish", forwardPayload);
    if (!result) {
      stop();
      return json({ ok: false, error: "Training publish endpoint not configured" }, { status: 501 });
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
