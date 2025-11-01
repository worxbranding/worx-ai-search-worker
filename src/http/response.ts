// Lightweight HTTP helpers shared by the worker routes.

/** Serialize JSON with consistent headers so responses look identical. */
export const json = (payload: unknown, init: ResponseInit = {}): Response =>
  new Response(JSON.stringify(payload, null, 2), {
    headers: { "content-type": "application/json; charset=utf-8" },
    ...init,
  });

/** Apply optional CORS headers to a response while preserving the payload. */
export function withCors(originAllow: string | null, res: Response): Response {
  const headers = new Headers(res.headers);
  if (originAllow) headers.set("Access-Control-Allow-Origin", originAllow);
  headers.set("access-control-allow-credentials", "true");
  headers.set("access-control-allow-headers", "content-type, authorization, x-api-key");
  headers.set("access-control-allow-methods", "GET, POST, OPTIONS");
  return new Response(res.body, { status: res.status, headers });
}
