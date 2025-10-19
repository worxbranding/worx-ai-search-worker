export interface Env {
    VECTORIZE: VectorizeIndex;
    AI: Ai;
}

const EMBED_MODEL = "@cf/baai/bge-base-en-v1.5"; // 768 dims
const CHAT_MODEL = "@cf/meta/llama-3.1-8b-instruct";
const DIMS = 768;

// ---------- small utils ----------
const json = (o: unknown, init: ResponseInit = {}) =>
    new Response(JSON.stringify(o, null, 2), {
        headers: { "content-type": "application/json; charset=utf-8" },
        ...init,
    });

function isNumArray(a: any): a is number[] {
    return Array.isArray(a) && a.every((x) => typeof x === "number" && Number.isFinite(x));
}
function toF32(v: number[]) {
    return new Float32Array(v);
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

// ---------- robust embed ----------
/**
 * Cloudflare AI can return several shapes. Accept all:
 * 1) number[768]
 * 2) { data: number[768], shape?: [1,768], ... }
 * 3) { data: [[number[768]]], ... }  // nested - THIS IS WHAT CF RETURNS
 * 4) { data: [{ embedding: number[768] }], ... }
 * 5) { embeddings: number[768] } (rare)
 */
function pluckEmbedding(raw: any): number[] | null {
    // 1) raw as array
    if (Array.isArray(raw) && isNumArray(raw) && raw.length === DIMS) return raw;

    // 2) object with data
    if (raw && typeof raw === "object") {
        const d = (raw as any).data;

        // 2a) data is flat array
        if (Array.isArray(d) && isNumArray(d) && d.length === DIMS) return d;

        // 2b) data is nested array [[...]] - MOST COMMON FROM CLOUDFLARE
        if (Array.isArray(d) && d.length > 0 && Array.isArray(d[0])) {
            const inner = d[0];
            if (isNumArray(inner) && inner.length === DIMS) return inner;
        }

        // 2c) data is [{ embedding: [...] }]
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

        // 3) embeddings key
        const e = (raw as any).embeddings;
        if (Array.isArray(e) && isNumArray(e) && e.length === DIMS) return e;
        if (Array.isArray(e) && e.length > 0 && Array.isArray(e[0]) && isNumArray(e[0]) && e[0].length === DIMS) return e[0];
    }

    return null;
}

async function embed(env: Env, text: string): Promise<number[]> {
    const raw = await env.AI.run(EMBED_MODEL as any, { text });

    console.log("Raw AI response:", JSON.stringify(raw).substring(0, 200)); // debug

    const arr = pluckEmbedding(raw);

    console.log("Plucked array:", arr ? `length=${arr.length}` : "null"); // debug

    if (!arr) {
        // include a tiny hint without dumping huge objects
        const hint =
            raw && typeof raw === "object"
                ? { keys: Object.keys(raw).slice(0, 6), sample: Array.isArray((raw as any).data) ? typeof (raw as any).data[0] : typeof raw }
                : { typeof: typeof raw };
        throw new Error(`Embedding unexpected shape: ${JSON.stringify(hint)}`);
    }
    ensureDims(arr);
    return arr;  // Return the regular array, NOT toF32(arr)
}

// ---------- endpoints ----------
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
            ask: "POST /ask { q, k }"
        }
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

/** Query by a stored vector id (uses vectorId; there is no .get()) */
async function handleDebugQueryById(req: Request, env: Env) {
    const u = new URL(req.url);
    const id = u.searchParams.get("id") || "";
    const k = Number(u.searchParams.get("k") || 6);
    if (!id) return json({ ok: false, error: "Missing id" }, { status: 400 });

    try {
        const out = await (env.VECTORIZE as any).query({
            vectorId: id,
            topK: isFinite(k) && k > 0 ? k : 6,
            includeMetadata: true
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
            returnMetadata: true
        });

        const matches = out.matches ?? [];
        return json({
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
            returnMetadata: true
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

        // Correct Vectorize v2 API: query(vector, options)
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

        return json({
            ok: true,
            q,
            k,
            answer,
            citations: matches.map((m: any) => `[${m.id}] ${fmtCitation(m.metadata || {})}`)
        });
    } catch (e: any) {
        return json({ ok: false, error: String(e?.message || e), where: "vectorize.query" }, { status: 500 });
    }
}

// ---------- router ----------
export default {
    async fetch(req: Request, env: Env): Promise<Response> {
        const { pathname } = new URL(req.url);

        if (req.method === "GET" && pathname === "/status") return handleStatus();

        if (req.method === "GET" && pathname === "/debug/embed") return handleDebugEmbed(req, env);
        if (req.method === "GET" && pathname === "/debug/embed/raw") return handleDebugEmbedRaw(req, env);
        if (req.method === "GET" && pathname === "/debug/query-by-id") return handleDebugQueryById(req, env);

        if (req.method === "GET" && pathname === "/search") return handleSearch(req, env);
        if (req.method === "GET" && pathname === "/search/raw") return handleSearchRaw(req, env);

        if (req.method === "POST" && pathname === "/ask") return handleAsk(req, env);

        if (req.method === "GET" && pathname === "/") return json({ ok: true, see: "/status" });
        return new Response("Not found", { status: 404 });
    },
};