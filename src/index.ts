// src/index.ts (worx-search) — full file

export interface Env {
    VECTORIZE: VectorizeIndex;
    AI: Ai;
    ALLOWED_ORIGINS?: string; // comma-separated list; e.g. "http://localhost:5173,https://example.com"
}

const EMBED_MODEL = "@cf/baai/bge-base-en-v1.5"; // 768 dims
const CHAT_MODEL = "@cf/meta/llama-3.1-8b-instruct";
const DIMS = 768;

/* ------------------------- CORS helpers ------------------------- */

const DEFAULT_ALLOWED = ["http://localhost:5173", "http://localhost:3000"];

function corsHeaders(origin: string | null, allowed: string[]): Headers {
    const h = new Headers({ "content-type": "application/json; charset=utf-8" });
    const okOrigin = origin && (allowed.includes("*") || allowed.includes(origin));
    h.set("access-control-allow-origin", okOrigin ? origin : "null");
    h.set("access-control-allow-credentials", "true");
    h.set("access-control-allow-headers", "content-type, authorization");
    h.set("access-control-allow-methods", "GET, POST, OPTIONS");
    return h;
}

function withCors(origin: string | null, allowed: string[], body: unknown, init: ResponseInit = {}) {
    const headers = corsHeaders(origin, allowed);
    if (init.headers) {
        for (const [k, v] of Object.entries(init.headers as Record<string, string>)) headers.set(k, v as string);
    }
    return new Response(JSON.stringify(body, null, 2), { ...init, headers });
}

/* ------------------------- small utils ------------------------- */

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

/* ------------------------- robust embed ------------------------- */
/**
 * Cloudflare AI may return:
 * 1) number[768]
 * 2) { data: number[768], shape?: [1,768], ... }
 * 3) { data: [[number[768]]] }   // common
 * 4) { data: [{ embedding: number[768] }], ... }
 * 5) { embeddings: number[768] } or [[...]]
 */
function pluckEmbedding(raw: any): number[] | null {
    if (Array.isArray(raw) && isNumArray(raw) && raw.length === DIMS) return raw;

    if (raw && typeof raw === "object") {
        const d = (raw as any).data;

        if (Array.isArray(d) && isNumArray(d) && d.length === DIMS) return d;

        if (Array.isArray(d) && d.length > 0 && Array.isArray(d[0])) {
            const inner = d[0];
            if (isNumArray(inner) && inner.length === DIMS) return inner;
        }

        if (
            Array.isArray(d) &&
            d.length > 0 &&
            d[0] &&
            typeof d[0] === "object" &&
            Array.isArray(d[0].embedding) &&
            isNumArray(d[0].embedding) &&
            d[0].embedding.length === DIMS
        ) {
            return d[0].embedding as number[];
        }

        const e = (raw as any).embeddings;
        if (Array.isArray(e) && isNumArray(e) && e.length === DIMS) return e;
        if (Array.isArray(e) && e.length > 0 && Array.isArray(e[0]) && isNumArray(e[0]) && e[0].length === DIMS) return e[0];
    }

    return null;
}

async function embed(env: Env, text: string): Promise<number[]> {
    const raw = await env.AI.run(EMBED_MODEL as any, { text });

    const arr = pluckEmbedding(raw);
    if (!arr) {
        const hint =
            raw && typeof raw === "object"
                ? { keys: Object.keys(raw).slice(0, 6), sample: Array.isArray((raw as any).data) ? typeof (raw as any).data[0] : typeof raw }
                : { typeof: typeof raw };
        throw new Error(`Embedding unexpected shape: ${JSON.stringify(hint)}`);
    }
    ensureDims(arr);
    return arr; // keep as regular array for Vectorize.query
}

/* ------------------------- route handlers ------------------------- */

async function handleStatus(req: Request, allowed: string[]) {
    const origin = req.headers.get("origin");
    return withCors(origin, allowed, {
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
            ask: "POST /ask { q, k }"
        }
    });
}

async function handleDebugEmbed(req: Request, env: Env, allowed: string[]) {
    const origin = req.headers.get("origin");
    const q = new URL(req.url).searchParams.get("q") || "";
    try {
        const v = await embed(env, q);
        return withCors(origin, allowed, { ok: true, q, result: { valid: v.length === DIMS, gotLen: v.length, expected: DIMS } });
    } catch (e: any) {
        return withCors(origin, allowed, { ok: false, q, error: String(e?.message || e) }, { status: 500 });
    }
}

async function handleDebugEmbedRaw(req: Request, env: Env, allowed: string[]) {
    const origin = req.headers.get("origin");
    const q = new URL(req.url).searchParams.get("q") || "";
    try {
        const raw = await env.AI.run(EMBED_MODEL as any, { text: q });
        return withCors(origin, allowed, { ok: true, q, raw });
    } catch (e: any) {
        return withCors(origin, allowed, { ok: false, q, error: String(e?.message || e) }, { status: 500 });
    }
}

/** Query by stored vector id (uses vectorId) */
async function handleDebugQueryById(req: Request, env: Env, allowed: string[]) {
    const origin = req.headers.get("origin");
    const u = new URL(req.url);
    const id = u.searchParams.get("id") || "";
    const k = Number(u.searchParams.get("k") || 6);
    if (!id) return withCors(origin, allowed, { ok: false, error: "Missing id" }, { status: 400 });

    try {
        const out = await (env.VECTORIZE as any).query({
            vectorId: id,
            topK: isFinite(k) && k > 0 ? k : 6,
            includeMetadata: true
        });
        const matches = (out as any).matches ?? out;
        return withCors(origin, allowed, { ok: true, mode: "vectorId", id, k, results: matches });
    } catch (e: any) {
        return withCors(origin, allowed, { ok: false, error: String(e?.message || e) }, { status: 500 });
    }
}

async function handleSearch(req: Request, env: Env, allowed: string[]) {
    const origin = req.headers.get("origin");
    const u = new URL(req.url);
    const q = u.searchParams.get("q") || "";
    const k = Number(u.searchParams.get("k") || 8);
    if (!q) return withCors(origin, allowed, { ok: false, error: "Missing q" }, { status: 400 });

    try {
        const vec = await embed(env, q);

        const out = await env.VECTORIZE.query(vec, {
            topK: isFinite(k) && k > 0 ? k : 8,
            returnMetadata: true
        });

        const matches = out.matches ?? [];
        return withCors(origin, allowed, {
            ok: true,
            q,
            k,
            results: matches.map((m: any) => ({
                id: m.id,
                score: m.score,
                metadata: m.metadata
            }))
        });
    } catch (e: any) {
        return withCors(origin, allowed, { ok: false, error: String(e?.message || e) }, { status: 500 });
    }
}

async function handleSearchRaw(req: Request, env: Env, allowed: string[]) {
    const origin = req.headers.get("origin");
    const u = new URL(req.url);
    const q = u.searchParams.get("q") || "";
    const k = Number(u.searchParams.get("k") || 8);
    if (!q) return withCors(origin, allowed, { ok: false, error: "Missing q" }, { status: 400 });

    try {
        const vec = await embed(env, q);

        const out = await env.VECTORIZE.query(vec, {
            topK: isFinite(k) && k > 0 ? k : 8,
            returnMetadata: true
        });

        return withCors(origin, allowed, { ok: true, route: "search/raw", body: out });
    } catch (e: any) {
        return withCors(origin, allowed, { ok: false, route: "search/raw", body: { error: String(e?.message || e) } }, { status: 500 });
    }
}

async function handleAsk(req: Request, env: Env, allowed: string[]) {
    const origin = req.headers.get("origin");
    const body = (await req.json().catch(() => ({}))) as { q?: string; k?: number };
    const q = (body.q || "").trim();
    const k = Number(body.k || 6);
    if (!q) return withCors(origin, allowed, { ok: false, error: "Missing q" }, { status: 400 });

    try {
        const vec = await embed(env, q);

        const out = await env.VECTORIZE.query(vec, {
            topK: isFinite(k) && k > 0 ? k : 6,
            returnMetadata: true
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
                { role: "user", content: user }
            ],
            max_output_tokens: 300
        } as any);

        const answer =
            (chat as any).response ??
            (Array.isArray((chat as any).output) ? (chat as any).output.map((t: any) => t?.content ?? "").join("\n") : String(chat));

        return withCors(origin, allowed, {
            ok: true,
            q,
            k,
            answer,
            citations: matches.map((m: any) => `[${m.id}] ${fmtCitation(m.metadata || {})}`)
        });
    } catch (e: any) {
        return withCors(origin, allowed, { ok: false, error: String(e?.message || e), where: "vectorize.query" }, { status: 500 });
    }
}

/* ------------------------- router ------------------------- */

export default {
    async fetch(req: Request, env: Env): Promise<Response> {
        const allowed = (env.ALLOWED_ORIGINS?.split(",").map((s) => s.trim()).filter(Boolean) ??
            DEFAULT_ALLOWED);

        // Preflight
        if (req.method === "OPTIONS") {
            return new Response(null, { headers: corsHeaders(req.headers.get("origin"), allowed) });
        }

        const { pathname } = new URL(req.url);

        if (req.method === "GET" && pathname === "/status") return handleStatus(req, allowed);

        if (req.method === "GET" && pathname === "/debug/embed") return handleDebugEmbed(req, env, allowed);
        if (req.method === "GET" && pathname === "/debug/embed/raw") return handleDebugEmbedRaw(req, env, allowed);
        if (req.method === "GET" && pathname === "/debug/query-by-id") return handleDebugQueryById(req, env, allowed);

        if (req.method === "GET" && pathname === "/search") return handleSearch(req, env, allowed);
        if (req.method === "GET" && pathname === "/search/raw") return handleSearchRaw(req, env, allowed);

        if (req.method === "POST" && pathname === "/ask") return handleAsk(req, env, allowed);

        return withCors(req.headers.get("origin"), allowed, { ok: true, see: "/status" });
    }
};