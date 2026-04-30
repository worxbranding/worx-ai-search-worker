import { json, withCors } from "./http/response";
import { log, time } from "./lib/logging";
import { allowOrigin, buildSiteConfig, type InBandRequestBody } from "./config/siteConfig";
import { verifyHmac } from "./lib/hmac";
import { handleStatus } from "./handlers/status";
import { handleSearch } from "./handlers/search";
import { handleAsk } from "./handlers/ask";
import { handleClearCache } from "./handlers/admin";
import {
  handleDebugEmbed,
  handleDebugListIds,
  handleDebugQueryById,
} from "./handlers/debug";
import type { Env, ExecutionContext } from "./lib/types";

/**
 * Entry point: verifies HMAC, parses the in-band body once, builds a
 * per-request SiteConfig from env + body, and dispatches to handlers.
 *
 * Site identity is the routing key (used for Vectorize metadata filtering
 * and KV cache prefixing); HMAC is what proves the request came from the
 * trusted CMS.
 */
export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(req.url);
    const { pathname } = url;
    const site = (url.searchParams.get("site") || "").trim();
    const origin = req.headers.get("Origin");
    const originAllow = allowOrigin(origin, env);
    log("REQUEST", req.method, pathname, site ? `(site=${site})` : "");

    if (req.method === "OPTIONS") {
      if (origin && !originAllow) {
        return json({ ok: false, error: "CORS: origin not allowed" }, { status: 403 });
      }
      return withCors(originAllow || "*", new Response(null, { status: 200 }));
    }

    try {
      // Read raw body once. HMAC is computed over the exact bytes; subsequent
      // JSON.parse runs on the same string.
      const rawBody = req.method === "GET" || req.method === "HEAD" ? "" : await req.text();

      const verify = await verifyHmac(req, rawBody, {
        current: env.WORX_HMAC_SECRET,
        previous: env.WORX_HMAC_SECRET_PREVIOUS,
      });
      if (!verify.ok) {
        log("[hmac]", req.method, pathname, verify.reason);
        return withCors(
          originAllow,
          json({ ok: false, error: `unauthorized: ${verify.reason}` }, { status: verify.status })
        );
      }

      let body: InBandRequestBody = {};
      if (rawBody) {
        try {
          body = JSON.parse(rawBody) as InBandRequestBody;
        } catch {
          return withCors(originAllow, json({ ok: false, error: "invalid JSON body" }, { status: 400 }));
        }
      }

      // /status doesn't require a site routing key.
      if (req.method === "GET" && pathname === "/status") {
        const res = await time("route:/status", () => handleStatus(req, env));
        return withCors(originAllow, res);
      }

      if (!site) {
        return withCors(
          originAllow,
          json({ ok: false, error: "Missing or invalid site" }, { status: 400 })
        );
      }

      const cfg = buildSiteConfig(env, site, body);

      if (req.method === "POST" && pathname === "/search") {
        return withCors(originAllow, await time("route:/search", () => handleSearch(req, env, cfg, body, ctx)));
      }
      if (req.method === "POST" && pathname === "/ask") {
        return withCors(originAllow, await time("route:/ask", () => handleAsk(req, env, cfg, body, ctx)));
      }
      if (req.method === "POST" && pathname === "/admin/clear-cache") {
        return withCors(
          originAllow,
          await time("route:/admin/clear-cache", () => handleClearCache(req, env, ctx))
        );
      }
      if (req.method === "POST" && pathname === "/debug/embed") {
        return withCors(originAllow, await time("route:/debug/embed", () => handleDebugEmbed(req, env, cfg, body)));
      }
      if (req.method === "POST" && pathname === "/debug/query-by-id") {
        return withCors(
          originAllow,
          await time("route:/debug/query-by-id", () => handleDebugQueryById(req, env, cfg, body))
        );
      }
      if (req.method === "POST" && pathname === "/debug/list-ids") {
        return withCors(
          originAllow,
          await time("route:/debug/list-ids", () => handleDebugListIds(req, env))
        );
      }

      return withCors(originAllow, new Response("Not found", { status: 404 }));
    } catch (error: any) {
      log("[ERROR]", req.method, pathname, String(error?.message || error));
      return withCors(
        originAllow,
        json({ ok: false, error: String(error?.message || error) }, { status: 500 })
      );
    }
  },
};
