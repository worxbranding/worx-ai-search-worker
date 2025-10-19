// worx-search/src/index.ts

export interface Env {
    VECTORIZE: VectorizeIndex;
    AI: Ai;
    ALLOWED_ORIGINS?: string; // comma-separated, or "*" in dev
}

const EMBED_MODEL = "@cf/baai/bge-base-en-v1.5"; // 768 dims
const CHAT_MODEL  = "@cf/meta/llama-3.1-8b-instruct";
const DIMS = 768;

// =============== small utils ===============
const json = (o: unknown, init: ResponseInit = {}) =>
    new Response(JSON.stringify(o, null, 2), {
        headers: { "content-type": "application/json; charset=utf-8" },
        ...init,
    });

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

// =============== robust embed ===============
/**
 * Accept all known shapes Cloudflare AI can return:
 *  1) number[768]
 *  2) { data: number[768], ... }
 *  3) { data: [ number[768] ] } or { data: [ [ number[768] ] ] }
 *  4) { data: [{ embedding: number[768] }], ... }
 *  5) { embeddings: number[768] } or embeddings: [ number[768] ]
 */
function pluckEmbedding(raw: any): number[] | null {
    // 1) raw array
    if (isNumArray(raw) && raw.length === DIMS) return raw;

    if (raw && typeof raw === "object") {
        const d = (raw as any).data;

        // 2) data is flat array
        if (isNumArray(d) && d.length === DIMS) return d;

        // 3a) data is [ number[] ]
        if (Array.isArray(d) && d.length > 0 && isNumArray(d[0]) && d[0].length === DIMS) return d[0] as number[];

        // 3b) data is [ [ number[] ] ]
        if (Array.isArray(d) && d.length > 0 && Array.isArray(d[0]) && isNumArray(d[0][0]) && d[0][0].length === DIMS) {
            return d[0][0] as number[];
        }

        // 4) data is [{ embedding: [...] }]
        if (Array.isArray(d) && d.length > 0 && d[0] && typeof d[0] === "object" && isNumArray(d[0].embedding) && d[0].embedding.length === DIMS) {
            return d[0].embedding as number[];
        }

        // 5) embeddings key
        const e = (raw as any).embeddings;
        if (isNumArray(e) && e.length === DIMS) return e;
        if (Array.isArray(e) && e.length > 0 && isNumArray(e[0]) && e[0].length === DIMS) return e[0];
    }
    return null;
}

async function embed(env: Env, text: string): Promise<number[]> {
    const raw = await env.AI.run(EMBED_MODEL as any, { text });
    const arr = pluckEmbedding(raw);
    if (!arr) {
        const hint =
            raw && typeof raw === "object"
                ? { keys: Object.keys(raw).slice(0, 6), typeOfData0: Array.isArray((raw as any).data) ? typeof (raw as any).data[0] : typeof raw }
                : { typeof: typeof raw };
        throw new Error(`Embedding unexpected shape: ${JSON.stringify(hint)}`);
    }
    ensureDims(arr);
    return arr;
}

// =============== CORS ===============
type CorsDecision =
    | { allowed: true; origin: string }
    | { allowed: false; reason: "no-origin" | "not-allowed" };

function parseAllowed(originsVar?: string): string[] | "*" {
    if (!originsVar) return [];
    const s = originsVar.trim();
    if (s === "*") return "*";
    return s
        .split(",")
        .map((x) => x.trim())
        .filter(Boolean);
}

function decideCors(env: Env, req: Request): CorsDecision {
    const origin = req.headers.get("Origin");
    if (!origin) return { allowed: false, reason: "no-origin" }; // non-browser (curl) – we won’t add ACAO

    const allowed = parseAllowed(env.ALLOWED_ORIGINS);
    if (allowed === "*") return { allowed: true, origin };

    if (Array.isArray(allowed) && allowed.includes(origin)) {
        return { allowed: true, origin };
    }
    return { allowed: false, reason: "not-allowed" };
}

function withCorsAllowed(res: Response, origin: string): Response {
    const h = new Headers(res.headers);
    h.set("Access-Control-Allow-Origin", origin);
    h.set("access-control-allow-credentials", "true");
    h.set("access-control-allow-headers", "content-type, authorization");
    h.set("access-control-allow-methods", "GET, POST, OPTIONS");
    return new Response(res.body, { ...res, headers: h });
}

function corsReject(): Response {
    // 403 and NO ACAO header (browsers will block)
    return json({ ok: false, error: "CORS: origin not allowed" }, { status: 403 });
}

// =============== endpoints ===============
async function handleStatus() {
    return json({
        ok: true,
        embed_model: EMBED_MODEL,
        chat_model: CHAT_MODEL,
        expected_dims: DIMS,
        endpoints: {
            status: "GET /status",
            debug_embed: "GET /debug/embed?q=...",
            debug_embed_raw: "GET /debug/embed/raw?q=...",
            debug_query_by_id: "GET /debug/query-by-id?id=...&k=8",
            search: "GET /search?q=...&k=8",
            search_raw: "GET /search/raw?q=...&k=8",
            ask: "POST /ask { q, k }",
        },
    });
}

async function handleDebugEmbed(req: Request, env: Env) {
    const q = new URL(req.url).searchParams.get("q") || "";
    try {
        const v = await embed(env, q);
        return json({ ok: true, q, result: { valid: v.length === DIMS, gotLen: v.length, expected: DIMS } });
    } catch (e: any) {
        return json({ ok: false, q, error: String(e?.message || e) }, { status: 500 });
    }
}

async function handleDebugEmbedRaw(req: Request, env: Env) {
    const q = new URL(req.url).searchParams.get("q") || "";
    try {
        const raw = await env.AI.run(EMBED_MODEL as any, { text: q });
        return json({ ok: true, q, raw });
    } catch (e: any) {
        return json({ ok: false, q, error: String(e?.message || e) }, { status: 500 });
    }
}

/** Query by stored vector id */
async function handleDebugQueryById(req: Request, env: Env) {
    const u = new URL(req.url);
    const id = u.searchParams.get("id") || "";
    const k = Number(u.searchParams.get("k") || 6);
    if (!id) return json({ ok: false, error: "Missing id" }, { status: 400 });

    try {
        const out = await (env.VECTORIZE as any).query({
            vectorId: id,
            topK: isFinite(k) && k > 0 ? k : 6,
            includeMetadata: true,
        });
        const matches = (out as any).matches ?? out;
        return json({ ok: true, mode: "vectorId", id, k, results: matches });
    } catch (e: any) {
        return json({ ok: false, error: String(e?.message || e) }, { status: 500 });
    }
}

async function handleSearch(req: Request, env: Env) {
    const u = new URL(req.url);
    const q = u.searchParams.get("q") || "";
    const k = Number(u.searchParams.get("k") || 8);
    if (!q) return json({ ok: false, error: "Missing q" }, { status: 400 });

    try {
        const vec = await embed(env, q);
        const out = await env.VECTORIZE.query(vec, {
            topK: isFinite(k) && k > 0 ? k : 8,
            returnMetadata: true,
        });
        const matches = out.matches ?? [];
        return json({
            ok: true,
            q,
            k,
            results: matches.map((m: any) => ({
                id: m.id,
                score: m.score,
                metadata: m.metadata,
            })),
        });
    } catch (e: any) {
        return json({ ok: false, error: String(e?.message || e) }, { status: 500 });
    }
}

async function handleSearchRaw(req: Request, env: Env) {
    const u = new URL(req.url);
    const q = u.searchParams.get("q") || "";
    const k = Number(u.searchParams.get("k") || 8);
    if (!q) return json({ ok: false, error: "Missing q" }, { status: 400 });

    try {
        const vec = await embed(env, q);
        const out = await env.VECTORIZE.query(vec, {
            topK: isFinite(k) && k > 0 ? k : 8,
            returnMetadata: true,
        });
        return json({ ok: true, route: "search/raw", body: out });
    } catch (e: any) {
        return json({ ok: false, route: "search/raw", body: { error: String(e?.message || e) } }, { status: 500 });
    }
}

async function handleAsk(req: Request, env: Env) {
    const body = (await req.json().catch(() => ({}))) as { q?: string; k?: number };
    const q = (body.q || "").trim();
    const k = Number(body.k || 6);
    if (!q) return json({ ok: false, error: "Missing q" }, { status: 400 });

    try {
        const vec = await embed(env, q);

        const out = await env.VECTORIZE.query(vec, {
            topK: isFinite(k) && k > 0 ? k : 6,
            returnMetadata: true,
        });
        const matches = out.matches ?? [];

        const contexts = matches.map((m: any) => {
            const md = m.metadata || {};
            const head = md.title || md.path || md.url || m.id;
            const prev = md.preview ? `\n${md.preview}` : "";
            return `### ${head}${prev}`;
        });

        const system = `You are a helpful assistant. Answer using only the provided context. If unsure, say you don't know. Keep answers concise.`;
        const user = `Question: ${q}\n\nContext:\n${contexts.join("\n\n")}`;

        const chat = await env.AI.run(CHAT_MODEL as any, {
            messages: [
                { role: "system", content: system },
                { role: "user", content: user },
            ],
            max_output_tokens: 300,
        } as any);

        const answer =
            (chat as any).response ??
            (Array.isArray((chat as any).output) ? (chat as any).output.map((t: any) => t?.content ?? "").join("\n") : String(chat));

        return json({
            ok: true,
            q,
            k,
            answer,
            citations: matches.map((m: any) => `[${m.id}] ${fmtCitation(m.metadata || {})}`),
        });
    } catch (e: any) {
        return json({ ok: false, error: String(e?.message || e), where: "vectorize.query" }, { status: 500 });
    }
}

// =============== router with CORS gate ===============
export default {
    async fetch(req: Request, env: Env): Promise<Response> {
        const url = new URL(req.url);
        const path = url.pathname;

        // Handle CORS preflight first
        if (req.method === "OPTIONS") {
            const decision = decideCors(env, req);
            if (!decision.allowed) return corsReject(); // invalid origin
            // valid preflight
            return withCorsAllowed(new Response(null, { status: 204 }), decision.origin);
        }

        // Route
        let res: Response;
        try {
            if (req.method === "GET" && path === "/status") res = await handleStatus();
            else if (req.method === "GET" && path === "/debug/embed") res = await handleDebugEmbed(req, env);
            else if (req.method === "GET" && path === "/debug/embed/raw") res = await handleDebugEmbedRaw(req, env);
            else if (req.method === "GET" && path === "/debug/query-by-id") res = await handleDebugQueryById(req, env);
            else if (req.method === "GET" && path === "/search") res = await handleSearch(req, env);
            else if (req.method === "GET" && path === "/search/raw") res = await handleSearchRaw(req, env);
            else if (req.method === "POST" && path === "/ask") res = await handleAsk(req, env);
            else if (req.method === "GET" && path === "/") res = json({ ok: true, see: "/status" });
            else res = new Response("Not found", { status: 404 });
        } catch (e: any) {
            res = json({ ok: false, error: String(e?.message || e) }, { status: 500 });
        }

        // Only add CORS headers if an Origin was sent and it’s allowed
        const origin = req.headers.get("Origin");
        if (!origin) return res; // curl / server-to-server

        const decision = decideCors(env, req);
        if (!decision.allowed) return corsReject();
        return withCorsAllowed(res, decision.origin);
    },
};