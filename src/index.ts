export interface Env {
    VECTORIZE: VectorizeIndex;
    CONFIG: KVNamespace;
    AI: Ai;

    // fallbacks (used if config is missing pieces)
    DEFAULT_ALLOWED_ORIGINS?: string; // csv
    DEFAULT_TOPK?: string;
    DEFAULT_EMBED_MODEL?: string;
    DEFAULT_CHAT_MODEL?: string;
    DEFAULT_CHAT_TEMPERATURE?: string;
}

type SiteSearchConfig = {
    site_key: string;

    // CORS
    allowed_origins?: string[]; // explicit list; if omitted, default var used

    // Optional simple auth
    api_key?: string; // if present, require x-worx-key == api_key

    // AI
    embed_model?: string; // defaults to env.DEFAULT_EMBED_MODEL
    chat_model?: string;  // defaults to env.DEFAULT_CHAT_MODEL
    chat_temperature?: number; // defaults to env.DEFAULT_CHAT_TEMPERATURE (number)
    topK?: number; // default if client omits k

    // Answering
    system_prompt?: string; // custom per-site
};

const DIMS = 768;

// ---------- tiny utils ----------
const json = (o: unknown, init: ResponseInit = {}) =>
    new Response(JSON.stringify(o, null, 2), {
        headers: { "content-type": "application/json; charset=utf-8" },
        ...init,
    });

function csvToList(s?: string): string[] {
    return (s || "")
        .split(",")
        .map((x) => x.trim())
        .filter(Boolean);
}

function isNumArray(a: any): a is number[] {
    return Array.isArray(a) && a.every((x) => typeof x === "number" && Number.isFinite(x));
}

function ensureDims(v: number[]) {
    if (!Array.isArray(v) || v.length !== DIMS) {
        throw new Error(`Invalid embedding dims: got ${Array.isArray(v) ? v.length : 0}, need ${DIMS}`);
    }
}

function fmtCitation(m: Record<string, any>): string {
    const title = m?.title ?? m?.path ?? m?.url ?? "(untitled)";
    const url = m?.url ?? "";
    const ts = m?.updated_at ? ` (updated ${m.updated_at})` : "";
    return `${title} — ${url}${ts}`;
}

// ---------- config load ----------
async function loadSiteCfg(env: Env, site: string): Promise<SiteSearchConfig> {
    const key = `cfg:${site}`;
    const cfg = await env.CONFIG.get<SiteSearchConfig>(key, "json");
    if (!cfg) throw new Error(`Missing CONFIG KV entry for ${key}`);
    if (cfg.site_key !== site) throw new Error(`CONFIG ${key} site_key mismatch`);

    // apply defaults
    if (!cfg.embed_model) cfg.embed_model = env.DEFAULT_EMBED_MODEL || "@cf/baai/bge-base-en-v1.5";
    if (!cfg.chat_model) cfg.chat_model = env.DEFAULT_CHAT_MODEL || "@cf/meta/llama-3.1-8b-instruct";
    if (cfg.chat_temperature == null) {
        cfg.chat_temperature = Number(env.DEFAULT_CHAT_TEMPERATURE || "0.1");
    }
    if (!cfg.allowed_origins || !cfg.allowed_origins.length) {
        cfg.allowed_origins = csvToList(env.DEFAULT_ALLOWED_ORIGINS || "");
    }
    if (!cfg.topK) cfg.topK = Number(env.DEFAULT_TOPK || "6");
    return cfg;
}

// ---------- CORS ----------
function pickCorsOrigin(req: Request, cfg: SiteSearchConfig): string | null {
    const origin = req.headers.get("Origin");
    if (!origin) return null; // allow server-to-server curls; browser needs Origin
    if (cfg.allowed_origins.includes(origin)) return origin;
    return null;
}

function withCors(res: Response, origin: string | null) {
    const h = new Headers(res.headers);
    h.set("access-control-allow-credentials", "true");
    h.set("access-control-allow-headers", "content-type, authorization");
    h.set("access-control-allow-methods", "GET, POST, OPTIONS");
    h.set("Access-Control-Allow-Origin", origin ?? "null");
    return new Response(res.body, { ...res, headers: h });
}

// ---------- auth ----------
function checkApiKey(req: Request, cfg: SiteSearchConfig) {
    if (!cfg.api_key) return true; // no key required for this tenant
    const got = req.headers.get("x-worx-key");
    return got === cfg.api_key;
}

// ---------- robust embed shape plucker ----------
function pluckEmbedding(raw: any): number[] | null {
    if (Array.isArray(raw) && isNumArray(raw) && raw.length === DIMS) return raw;
    if (raw && typeof raw === "object") {
        const d = (raw as any).data;
        if (Array.isArray(d) && isNumArray(d) && d.length === DIMS) return d;
        if (Array.isArray(d) && d.length > 0 && Array.isArray(d[0]) && isNumArray(d[0]) && d[0].length === DIMS) return d[0];
        if (Array.isArray(d) && d.length > 0 && d[0] && Array.isArray(d[0].embedding) && isNumArray(d[0].embedding)) {
            if (d[0].embedding.length === DIMS) return d[0].embedding;
        }
        const e = (raw as any).embeddings;
        if (Array.isArray(e) && isNumArray(e) && e.length === DIMS) return e;
        if (Array.isArray(e) && e.length > 0 && Array.isArray(e[0]) && isNumArray(e[0]) && e[0].length === DIMS) return e[0];
    }
    return null;
}

async function embed(env: Env, model: string, text: string): Promise<number[]> {
    const raw = await env.AI.run(model as any, { text });
    const arr = pluckEmbedding(raw);
    if (!arr) {
        const hint = raw && typeof raw === "object" ? { keys: Object.keys(raw).slice(0, 6) } : { typeof: typeof raw };
        throw new Error(`Embedding unexpected shape: ${JSON.stringify(hint)}`);
    }
    ensureDims(arr);
    return arr;
}

// ---------- endpoints ----------
async function handleStatus(req: Request, env: Env) {
    const u = new URL(req.url);
    const site = u.searchParams.get("site") || "(none)";
    return json({
        ok: true,
        siteParam: site,
        defaults: {
            allowed_origins: env.DEFAULT_ALLOWED_ORIGINS,
            topK: Number(env.DEFAULT_TOPK || "6"),
            embed_model: env.DEFAULT_EMBED_MODEL,
            chat_model: env.DEFAULT_CHAT_MODEL,
            chat_temperature: Number(env.DEFAULT_CHAT_TEMPERATURE || "0.1"),
        },
        endpoints: {
            status: "GET /status?site=...",
            search: "GET /search?site=...&q=...&k=8",
            ask: "POST /ask  { site, q, k }",
            debug_query_by_id: "GET /debug/query-by-id?site=...&id=...&k=8",
        }
    });
}

async function handlePreflight(req: Request, cfg: SiteSearchConfig) {
    const origin = pickCorsOrigin(req, cfg);
    if (!origin) return json({ ok: false, error: "CORS: origin not allowed" }, { status: 403 });
    return withCors(new Response(null, { status: 200 }), origin);
}

// Vectorize filter helper (multi-tenant partition)
function siteFilter(site: string) {
    // Vectorize v2 metadata filter: match exact field value
    return { must: [{ key: "site", match: site }] };
}

async function handleSearch(req: Request, env: Env, cfg: SiteSearchConfig) {
    const u = new URL(req.url);
    const site = cfg.site_key;
    const origin = pickCorsOrigin(req, cfg);
    if (!origin) return json({ ok: false, error: "CORS: origin not allowed" }, { status: 403 });

    if (!checkApiKey(req, cfg)) return withCors(json({ ok: false, error: "Unauthorized" }, { status: 401 }), origin);

    const q = u.searchParams.get("q") || "";
    const k = Number(u.searchParams.get("k") || cfg.topK || 6);
    if (!q) return withCors(json({ ok: false, error: "Missing q" }, { status: 400 }), origin);

    try {
        const vec = await embed(env, cfg.embed_model!, q);

        const out = await env.VECTORIZE.query(vec, {
            topK: isFinite(k) && k > 0 ? k : (cfg.topK || 6),
            // per-tenant filter
            filter: siteFilter(site),
            returnMetadata: true
        } as any);

        const matches = out.matches ?? [];
        return withCors(json({
            ok: true,
            site,
            q,
            k,
            results: matches.map((m: any) => ({
                id: m.id,
                score: m.score,
                metadata: m.metadata
            }))
        }), origin);
    } catch (e: any) {
        return withCors(json({ ok: false, error: String(e?.message || e) }, { status: 500 }), origin);
    }
}

async function handleAsk(req: Request, env: Env, cfg: SiteSearchConfig) {
    const origin = pickCorsOrigin(req, cfg);
    if (!origin) return json({ ok: false, error: "CORS: origin not allowed" }, { status: 403 });

    if (!checkApiKey(req, cfg)) return withCors(json({ ok: false, error: "Unauthorized" }, { status: 401 }), origin);

    const body = (await req.json().catch(() => ({}))) as { site?: string; q?: string; k?: number };
    const q = (body.q || "").trim();
    const k = Number(body.k || cfg.topK || 6);
    const site = cfg.site_key;
    if (!q) return withCors(json({ ok: false, error: "Missing q" }, { status: 400 }), origin);

    try {
        const vec = await embed(env, cfg.embed_model!, q);

        const out = await env.VECTORIZE.query(vec, {
            topK: isFinite(k) && k > 0 ? k : (cfg.topK || 6),
            filter: siteFilter(site),
            returnMetadata: true
        } as any);

        const matches = out.matches ?? [];

        const contexts = matches.map((m: any) => {
            const md = m.metadata || {};
            const head = md.title || md.path || md.url || m.id;
            const prev = md.preview ? `\n${md.preview}` : "";
            return `### ${head}${prev}`;
        });

        const system = cfg.system_prompt || `You are a helpful assistant. Answer using only the provided context. If unsure, say you don't know. Keep answers concise.`;
        const user = `Question: ${q}\n\nContext:\n${contexts.join("\n\n")}`;

        const chat = await env.AI.run(cfg.chat_model! as any, {
            messages: [
                { role: "system", content: system },
                { role: "user", content: user }
            ],
            temperature: cfg.chat_temperature ?? 0.1,
            max_output_tokens: 350
        } as any);

        const answer =
            (chat as any).response ??
            (Array.isArray((chat as any).output) ? (chat as any).output.map((t: any) => t?.content ?? "").join("\n") : String(chat));

        return withCors(json({
            ok: true,
            site,
            q,
            k,
            answer,
            citations: matches.map((m: any) => `[${m.id}] ${fmtCitation(m.metadata || {})}`)
        }), origin);
    } catch (e: any) {
        return withCors(json({ ok: false, error: String(e?.message || e) }, { status: 500 }), origin);
    }
}

async function handleDebugQueryById(req: Request, env: Env, cfg: SiteSearchConfig) {
    const u = new URL(req.url);
    const origin = pickCorsOrigin(req, cfg);
    if (!origin) return json({ ok: false, error: "CORS: origin not allowed" }, { status: 403 });

    const id = u.searchParams.get("id") || "";
    const k = Number(u.searchParams.get("k") || cfg.topK || 6);
    if (!id) return withCors(json({ ok: false, error: "Missing id" }, { status: 400 }), origin);

    try {
        const out = await (env.VECTORIZE as any).query({
            vectorId: id,
            topK: isFinite(k) && k > 0 ? k : (cfg.topK || 6),
            includeMetadata: true,
            filter: siteFilter(cfg.site_key)
        });
        const matches = (out as any).matches ?? out;
        return withCors(json({ ok: true, site: cfg.site_key, mode: "vectorId", id, k, results: matches }), origin);
    } catch (e: any) {
        return withCors(json({ ok: false, error: String(e?.message || e) }, { status: 500 }), origin);
    }
}

// ---------- router ----------
export default {
    async fetch(req: Request, env: Env): Promise<Response> {
        try {
            const u = new URL(req.url);
            const { pathname } = u;
            const site = (u.searchParams.get("site") || "").trim();

            // Status is allowed without site (to read defaults), but if provided, CORS respects site.
            if (req.method === "GET" && pathname === "/status") {
                // If site present, we’ll decorate response with CORS for that site
                if (!site) return json(await (async () => JSON.parse(await (await handleStatus(req, env)).text()))());
                const cfg = await loadSiteCfg(env, site);
                const origin = pickCorsOrigin(req, cfg);
                const res = await handleStatus(req, env);
                return withCors(res, origin);
            }

            // Everything else requires site
            if (!site) return json({ ok: false, error: "Missing site" }, { status: 400 });

            const cfg = await loadSiteCfg(env, site);

            if (req.method === "OPTIONS") return handlePreflight(req, cfg);

            if (req.method === "GET" && pathname === "/search") return handleSearch(req, env, cfg);
            if (req.method === "GET" && pathname === "/debug/query-by-id") return handleDebugQueryById(req, env, cfg);

            if (req.method === "POST" && pathname === "/ask") return handleAsk(req, env, cfg);

            return json({ ok: false, error: "Not found" }, { status: 404 });
        } catch (e: any) {
            return json({ ok: false, error: String(e?.message || e) }, { status: 500 });
        }
    },
};