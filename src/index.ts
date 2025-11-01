import { json, withCors } from "./http/response";
import { log, time } from "./lib/logging";
import {
  allowOrigin,
  loadSiteConfig,
  requireApiKey,
} from "./config/siteConfig";
import { handleStatus } from "./handlers/status";
import { handleSearch } from "./handlers/search";
import { handleAsk } from "./handlers/ask";
import { handleClearCache } from "./handlers/admin";
import {
  handleDebugEmbed,
  handleDebugListIds,
  handleDebugQueryById,
} from "./handlers/debug";
import type { Env, ExecutionContext, SiteConfig } from "./lib/types";

/** Entry point: routes incoming Worker requests to the modularised handlers. */
export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(req.url);
    const { pathname } = url;
    const site = (url.searchParams.get("site") || "").trim();
    log("REQUEST", req.method, pathname, site ? `(site=${site})` : "");

    let cfg: SiteConfig | undefined;
    if (site) {
      try {
        cfg = await time("loadSiteConfig", () => loadSiteConfig(env, site));
      } catch (error: any) {
        return json({ ok: false, error: String(error?.message || error) }, { status: 500 });
      }
    }

    const origin = req.headers.get("Origin");
    const originAllow = cfg ? allowOrigin(origin, cfg, env) : null;

    if (req.method === "OPTIONS") {
      if (origin && !originAllow) {
        return json({ ok: false, error: "CORS: origin not allowed" }, { status: 403 });
      }
      return withCors(originAllow || "*", new Response(null, { status: 200 }));
    }

    try {
      if (!cfg) {
        return withCors(originAllow, json({ ok: false, error: "Missing or invalid site" }, { status: 400 }));
      }

      const auth = await requireApiKey(req, cfg);
      if (auth) return withCors(originAllow, auth);

      if (req.method === "GET" && pathname === "/status") {
        const res = await time("route:/status", () => handleStatus(req, env));
        return withCors(originAllow, res);
      }
      if (req.method === "GET" && pathname === "/search") {
        return withCors(originAllow, await time("route:/search", () => handleSearch(req, env, cfg!, ctx)));
      }
      if (req.method === "POST" && pathname === "/ask") {
        return withCors(originAllow, await time("route:/ask", () => handleAsk(req, env, cfg!, ctx)));
      }
      if (req.method === "POST" && pathname === "/admin/clear-cache") {
        return withCors(
          originAllow,
          await time("route:/admin/clear-cache", () => handleClearCache(req, env, cfg!, ctx))
        );
      }
      if (req.method === "GET" && pathname === "/admin/clear-cache") {
        return withCors(
          originAllow,
          await time("route:/admin/clear-cache", () => handleClearCache(req, env, cfg!, ctx))
        );
      }
      if (req.method === "GET" && pathname === "/debug/embed") {
        return withCors(originAllow, await time("route:/debug/embed", () => handleDebugEmbed(req, env, cfg!)));
      }
      if (req.method === "GET" && pathname === "/debug/query-by-id") {
        return withCors(
          originAllow,
          await time("route:/debug/query-by-id", () => handleDebugQueryById(req, env, cfg!))
        );
      }
      if (req.method === "GET" && pathname === "/debug/list-ids") {
        return withCors(
          originAllow,
          await time("route:/debug/list-ids", () => handleDebugListIds(req, env, cfg!))
        );
      }

      return withCors(originAllow, new Response("Not found", { status: 404 }));
    } catch (error: any) {
      log("[ERROR]", req.method, pathname, String(error?.message || error));
      return withCors(originAllow, json({ ok: false, error: String(error?.message || error) }, { status: 500 }));
    }
  },
};
