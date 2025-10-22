// WORX AI Search Worker — multi-tenant + API key + CORS + Vectorize v2 (ENRICHED CONTEXT)

// ---- Type declarations ----
declare interface VectorizeIndex {
    query(arg: number[] | { vectorId: string } | any, opts?: any): Promise<{ matches?: Array<any> }>;
}

declare interface Ai {
    run(model: any, input: any): Promise<any>;
}

declare interface KVNamespace {
    get<T = string>(key: string, type?: "text" | "json"): Promise<T | null>;
    put?(key: string, value: string, opts?: any): Promise<void>;
}

export interface Env {
    VECTORIZE: VectorizeIndex;
    AI: Ai;
    CONFIG: KVNamespace;
    ALLOWED_ORIGINS?: string;
}

// ---- Config types ----
type SiteConfig = {
    site_key: string;
    vectorize: { index_name: string; dims: number; metric?: "cosine" | "euclidean" | "dot" };
    ai: { embed_model: string };
    api_key?: string;
    search?: {
        allowed_origins?: string[];
        api_key?: string;
        chat_model?: string;
        chat_temperature?: number;
        system_prompt?: string;
        topK?: number;
        max_output_tokens?: number;
    };
};

type SearchMatch = { id: string; score: number; metadata?: Record<string, any> };

const json = (o: unknown, init: ResponseInit = {}) =>
    new Response(JSON.stringify(o, null, 2), {
        headers: { "content-type": "application/json; charset=utf-8" },
        ...init,
    });

// ---- Small utils ----
async function loadSiteConfig(env: Env, site: string): Promise<SiteConfig> {
    const key = `cfg:${site}`;
    const cfg = await env.CONFIG.get<SiteConfig>(key, "json");
    if (!cfg) throw new Error(`Missing CONFIG KV entry for ${key}`);
    if (!cfg.vectorize?.dims) throw new Error(`CONFIG ${key} missing vectorize.dims`);
    if (!cfg.ai?.embed_model) throw new Error(`CONFIG ${key} missing ai.embed_model`);
    return cfg;
}

function sitePrefixedId(site: string, id: string) {
    return id.startsWith(`${site}:`) ? id : `${site}:${id}`;
}

function filterToSite(site: string, matches: SearchMatch[]): SearchMatch[] {
    return (matches || []).filter((m) => {
        if (!m) return false;
        if (m.id?.startsWith(`${site}:`)) return true;
        const s = (m.metadata as any)?.site;
        return s === site;
    });
}

function allowOrigin(origin: string | null, cfg: SiteConfig, env: Env): string | null {
    const fromCfg = cfg.search?.allowed_origins || [];
    let allowed = fromCfg.length ? fromCfg : (env.ALLOWED_ORIGINS || "").split(",").map(s => s.trim()).filter(Boolean);
    if (!origin) return null;
    if (allowed.includes("*")) return "*";
    return allowed.includes(origin) ? origin : null;
}

function wantApiKey(cfg: SiteConfig): string {
    return (cfg.search?.api_key || cfg.api_key || "").trim();
}
async function requireApiKey(req: Request, cfg: SiteConfig): Promise<Response | null> {
    const want = wantApiKey(cfg);
    if (!want) return null;
    const got = (req.headers.get("x-api-key") || "").trim();
    if (got === want) return null;
    return json({ ok: false, error: "API key required or invalid" }, { status: 401 });
}

function isNumArray(a: any): a is number[] {
    return Array.isArray(a) && a.every((x) => typeof x === "number" && Number.isFinite(x));
}

function pluckEmbedding(raw: any, dims: number): number[] | null {
    if (Array.isArray(raw) && isNumArray(raw) && raw.length === dims) return raw;
    if (raw && typeof raw === "object") {
        const d = (raw as any).data;
        if (Array.isArray(d) && isNumArray(d) && d.length === dims) return d;
        if (Array.isArray(d) && d.length > 0 && Array.isArray(d[0]) && isNumArray(d[0]) && d[0].length === dims) return d[0];
        if (Array.isArray(d) && d.length > 0 && d[0] && typeof d[0] === "object") {
            const emb = (d[0] as any).embedding;
            if (Array.isArray(emb) && isNumArray(emb) && emb.length === dims) return emb;
        }
        const e = (raw as any).embeddings;
        if (Array.isArray(e) && isNumArray(e) && e.length === dims) return e;
        if (Array.isArray(e) && e.length > 0 && Array.isArray(e[0]) && isNumArray(e[0]) && e[0].length === dims) return e[0];
    }
    return null;
}

async function embed(env: Env, model: string, dims: number, text: string): Promise<number[]> {
    const raw = await env.AI.run(model as any, { text });
    const arr = pluckEmbedding(raw, dims);
    if (!arr) throw new Error("Embedding dims mismatch or invalid shape");
    return arr;
}

function normalizeUrl(u: string): string {
    try {
        const url = new URL(u);
        url.hash = "";
        url.search = "";
        url.hostname = url.hostname.toLowerCase();
        return url.toString().replace(/\/+$/, "");
    } catch {
        return "";
    }
}

// ---- Build enriched document context ----
async function buildDocContext(env: Env, md: Record<string, any>): Promise<string> {
    let parts: string[] = [];

    const url = md.url ? normalizeUrl(md.url) : "";
    const title = md.title || "Untitled";
    const seoTitle = md.seo_title || "";
    const seoDesc = md.seo_description || "";
    const collection = md.collection || "";
    const subcat = md.subcategory || "";
    const parent = md.parent_title || "";
    const breadcrumbs = (md.breadcrumbs || []).join(" > ");
    const preview = md.preview || "";
    const children = md.children_md || "";
    const path = md.path || "";

    parts.push(`**${title}** (${url})`);
    if (seoTitle && seoTitle !== title) parts.push(`SEO Title: ${seoTitle}`);
    if (seoDesc) parts.push(`SEO Description: ${seoDesc}`);
    if (collection) parts.push(`Collection: ${collection}`);
    if (subcat) parts.push(`Subcategory: ${subcat}`);
    if (parent) parts.push(`Parent: ${parent}`);
    if (breadcrumbs) parts.push(`Breadcrumbs: ${breadcrumbs}`);
    if (path) parts.push(`Path: ${path}`);

    const kvKey = md.doc_key;
    if (kvKey) {
        const txt = await env.CONFIG.get<string>(kvKey, "text");
        if (txt && txt.trim()) parts.push(`Full Text:\n${txt.slice(0, 3000)}`);
    }
    if (preview) parts.push(`Preview:\n${preview}`);
    if (children) parts.push(`Children:\n${children}`);

    return parts.join("\n\n");
}

// ---- Handlers ----
async function handleStatus(req: Request, env: Env) {
    const u = new URL(req.url);
    const site = (u.searchParams.get("site") || "").trim();
    if (!site) return json({ ok: false, error: "Missing ?site=" }, { status: 400 });
    const cfg = await loadSiteConfig(env, site);
    return json({
        ok: true,
        site,
        vectorize: cfg.vectorize.index_name,
        dims: cfg.vectorize.dims,
        embed_model: cfg.ai.embed_model,
        chat_model: cfg.search?.chat_model || "@cf/meta/llama-3.1-8b-instruct",
        requires_api_key: !!wantApiKey(cfg),
    });
}

async function handleSearch(req: Request, env: Env, cfg: SiteConfig) {
    const u = new URL(req.url);
    const site = (u.searchParams.get("site") || "").trim();
    const q = (u.searchParams.get("q") || "").trim();
    const k = cfg.search?.topK;
    const wantDebug = (u.searchParams.get("debug") || "") === "1";
    if (!site) return json({ ok: false, error: "Missing ?site=" }, { status: 400 });
    if (!q) return json({ ok: false, error: "Missing ?q=" }, { status: 400 });

    const vec = await embed(env, cfg.ai.embed_model, cfg.vectorize.dims, q);
    // @ts-ignore
    const out = await env.VECTORIZE.query(vec, { topK: k, returnMetadata: true });
    const pre = (out?.matches || []) as SearchMatch[];
    const post = filterToSite(site, pre);

    const body: any = { ok: true, site, q, k, results: post };
    if (wantDebug) body._debug = { total_pre: pre.length, total_post: post.length, sample_ids: post.slice(0, 3).map((m) => m.id) };
    return json(body);
}

// ---- Enhanced ASK ----
async function handleAsk(req: Request, env: Env, cfg: SiteConfig) {
    const body = (await req.json().catch(() => ({}))) as { site?: string; q?: string; k?: number };
    const site = (body.site || "").trim();
    const q = (body.q || "").trim();
    const k = Math.max(1, Math.min(24, Number(body.k || cfg.search?.topK || 6)));
    if (!site) return json({ ok: false, error: "Missing field 'site'" }, { status: 400 });
    if (!q) return json({ ok: false, error: "Missing field 'q'" }, { status: 400 });

    const vec = await embed(env, cfg.ai.embed_model, cfg.vectorize.dims, q);
    // @ts-ignore
    const out = await env.VECTORIZE.query(vec, { topK: k, includeMetadata: true, returnMetadata: true });
    const matches = filterToSite(site, out?.matches || []);

    const contexts: string[] = [];
    for (let i = 0; i < Math.min(matches.length, 6); i++) {
        const m = matches[i];
        const md = (m.metadata || {}) as any;
        const docContext = await buildDocContext(env, md);
        contexts.push(`[#${i + 1}] ${docContext}`);
    }

    const allowedUrls = matches.map((m) => normalizeUrl((m.metadata as any)?.url || "")).filter(Boolean);
    const linkHints = ["You may ONLY use these URLs in links:", ...allowedUrls.map((u) => `- ${u}`)].join("\n");

    const system = cfg.search?.system_prompt || `You are a helpful assistant. Use all provided context and metadata to answer accurately.
Include inline Markdown links only from this list:
${linkHints}
If unsure, respond: "I'm sorry, I couldn't find an answer based on the available information."`;

    const user = `Question: ${q}\n\nContext:\n${contexts.join("\n\n")}`;
    const chatModel = cfg.search?.chat_model || "@cf/meta/llama-3.1-8b-instruct";
    const temperature = Number(cfg.search?.chat_temperature ?? 0.1);
    const max_output_tokens = Math.max(128, Math.min(2048, Number(cfg.search?.max_output_tokens ?? 1024)));

    const chat = await env.AI.run(chatModel as any, {
        messages: [
            { role: "system", content: system },
            { role: "user", content: user },
        ],
        temperature,
        max_output_tokens,
    } as any);

    const answer = (chat as any).response || "I'm sorry, I couldn't find an answer based on the available information.";
    return json({ ok: true, q, k, answer, _debug: { matches: matches.slice(0, 3) } });
}

// ---- Debug ----
async function handleDebugEmbed(req: Request, env: Env, cfg: SiteConfig) {
    const u = new URL(req.url);
    const q = (u.searchParams.get("q") || "").trim();
    if (!q) return json({ ok: false, error: "Missing ?q=" }, { status: 400 });
    try {
        const arr = await embed(env, cfg.ai.embed_model, cfg.vectorize.dims, q);
        return json({ ok: true, q, result: { valid: Array.isArray(arr) && arr.length === cfg.vectorize.dims, gotLen: arr.length, expected: cfg.vectorize.dims } });
    } catch (e: any) {
        return json({ ok: false, q, error: String(e?.message || e) }, { status: 500 });
    }
}

async function handleDebugQueryById(req: Request, env: Env, cfg: SiteConfig) {
    const u = new URL(req.url);
    const site = (u.searchParams.get("site") || "").trim();
    let id = (u.searchParams.get("id") || "").trim();
    const k = Math.max(1, Math.min(24, Number(u.searchParams.get("k") || 3)));
    if (!site) return json({ ok: false, error: "Missing ?site=" }, { status: 400 });
    if (!id) return json({ ok: false, error: "Missing ?id=" }, { status: 400 });

    id = sitePrefixedId(site, id);
    // @ts-ignore
    const out = await env.VECTORIZE.query({ vectorId: id, topK: k, returnMetadata: true });
    const post = filterToSite(site, out?.matches || []);
    return json({ ok: true, id, k, results: post });
}

// ---- CORS + Router ----
function withCors(originAllow: string | null, res: Response): Response {
    const headers = new Headers(res.headers);
    if (originAllow) headers.set("Access-Control-Allow-Origin", originAllow);
    headers.set("access-control-allow-credentials", "true");
    headers.set("access-control-allow-headers", "content-type, authorization, x-api-key");
    headers.set("access-control-allow-methods", "GET, POST, OPTIONS");
    return new Response(res.body, { status: res.status, headers });
}

export default {
    async fetch(req: Request, env: Env): Promise<Response> {
        const u = new URL(req.url);
        const { pathname } = u;
        const site = (u.searchParams.get("site") || "").trim();

        let cfg: SiteConfig | undefined;
        if (site) {
            try { cfg = await loadSiteConfig(env, site); }
            catch (e: any) { return json({ ok: false, error: String(e?.message || e) }, { status: 500 }); }
        }

        const origin = req.headers.get("Origin");
        const originAllow = cfg ? allowOrigin(origin, cfg, env) : null;

        if (req.method === "OPTIONS") {
            if (origin && !originAllow) return json({ ok: false, error: "CORS: origin not allowed" }, { status: 403 });
            return withCors(originAllow || "*", new Response(null, { status: 200 }));
        }

        try {
            if (req.method === "GET" && pathname === "/status") {
                const res = await handleStatus(req, env);
                return withCors(originAllow, res);
            }

            if (!cfg) return withCors(originAllow, json({ ok: false, error: "Missing or invalid site" }, { status: 400 }));
            const auth = await requireApiKey(req, cfg);
            if (auth) return withCors(originAllow, auth);

            if (req.method === "GET" && pathname === "/search") return withCors(originAllow, await handleSearch(req, env, cfg));
            if (req.method === "POST" && pathname === "/ask") return withCors(originAllow, await handleAsk(req, env, cfg));
            if (req.method === "GET" && pathname === "/debug/embed") return withCors(originAllow, await handleDebugEmbed(req, env, cfg));
            if (req.method === "GET" && pathname === "/debug/query-by-id") return withCors(originAllow, await handleDebugQueryById(req, env, cfg));

            return withCors(originAllow, new Response("Not found", { status: 404 }));
        } catch (e: any) {
            return withCors(originAllow, json({ ok: false, error: String(e?.message || e) }, { status: 500 }));
        }
    },
};