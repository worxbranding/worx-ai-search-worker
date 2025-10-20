// WORX AI Search Worker — all routes require x-api-key per site
// Endpoints:
//   GET  /status?site=...                (requires x-api-key)
//   GET  /search?site=...&q=...&k=...    (requires x-api-key + CORS)
//   POST /ask   { site, q, k }           (requires x-api-key)

export interface Env {
    CONFIG: KVNamespace;
    VECTORIZE: VectorizeIndex;
    AI: Ai;
    ALLOWED_ORIGINS?: string;
}

type SiteConfig = {
    site_key: string;
    api_key?: string;
    vectorize: { index_name: string; dims: number };
    ai: { embed_model: string };
    search?: {
        allowed_origins?: string[];
        chat_model?: string;
        chat_temperature?: number;
        system_prompt?: string;
    };
};

const DEFAULTS = {
    embed_model: "@cf/baai/bge-base-en-v1.5",
    chat_model:  "@cf/meta/llama-3.1-8b-instruct",
    chat_temperature: 0.1
};

const json = (o: unknown, init: ResponseInit = {}) =>
    new Response(JSON.stringify(o, null, 2), {
        headers: { "content-type": "application/json; charset=utf-8" },
        ...init,
    });

const isNumArray = (a:any): a is number[] => Array.isArray(a) && a.every(x=>typeof x==="number" && Number.isFinite(x));

/* ---------- helpers ---------- */
async function loadSiteConfig(env: Env, site: string): Promise<SiteConfig> {
    const key = `cfg:${site}`;
    const cfg = await env.CONFIG.get<SiteConfig>(key, "json");
    if (!cfg) throw new Error(`Missing CONFIG KV entry for ${key}`);
    if (!cfg.api_key) throw new Error(`CONFIG ${key} missing api_key`);
    return cfg;
}

function requireApiKey(req: Request, cfg: SiteConfig): Response | null {
    const got = req.headers.get("x-api-key")?.trim() || "";
    if (got && got === cfg.api_key) return null;
    return json({ ok:false, error:"Unauthorized: invalid or missing x-api-key" }, { status:401 });
}

function parseAllowed(envAllowed?: string, cfgAllowed?: string[]): string[] {
    const a = new Set<string>();
    (envAllowed||"").split(",").map(s=>s.trim()).filter(Boolean).forEach(x=>a.add(x));
    (cfgAllowed||[]).forEach(x=>a.add(x));
    return [...a];
}

function corsHeaders(origin: string) {
    return {
        "Access-Control-Allow-Origin": origin || "null",
        "access-control-allow-headers": "content-type, x-api-key",
        "access-control-allow-methods": "GET, POST, OPTIONS",
    };
}

/* ---------- embedding ---------- */
function pluckEmbedding(raw:any, dims:number): number[] | null {
    if (Array.isArray(raw) && isNumArray(raw) && raw.length===dims) return raw;
    if (raw && typeof raw==="object") {
        const d = (raw as any).data;
        if (Array.isArray(d) && isNumArray(d) && d.length===dims) return d;
        if (Array.isArray(d) && d.length>0 && Array.isArray(d[0]) && isNumArray(d[0]) && d[0].length===dims) return d[0];
        const e = (raw as any).embeddings;
        if (Array.isArray(e) && isNumArray(e) && e.length===dims) return e;
        if (Array.isArray(e) && e.length>0 && Array.isArray(e[0]) && isNumArray(e[0]) && e[0].length===dims) return e[0];
    }
    return null;
}

async function embed(env: Env, model:string, dims:number, text:string): Promise<number[]> {
    const raw = await env.AI.run(model as any, { text });
    const arr = pluckEmbedding(raw, dims);
    if (!arr) throw new Error(`Embedding unexpected shape`);
    if (arr.length !== dims) throw new Error(`Embedding dims mismatch: got ${arr.length}, expected ${dims}`);
    return arr;
}

/* ---------- handlers ---------- */
async function handleStatus(req: Request, env: Env) {
    const u = new URL(req.url);
    const site = (u.searchParams.get("site")||"").trim();
    if (!site)
        return json({ ok:true, usage:"GET /status?site=... (requires x-api-key)", endpoints:["/search","/ask"] });

    try {
        const cfg = await loadSiteConfig(env, site);
        const auth = requireApiKey(req, cfg);
        if (auth) return auth;
        return json({
            ok:true,
            site: cfg.site_key,
            dims: cfg.vectorize.dims,
            model: cfg.ai.embed_model,
            requires_api_key: true
        });
    } catch (e:any) {
        return json({ ok:false, error:String(e?.message||e) }, { status:500 });
    }
}

async function handleSearch(req: Request, env: Env) {
    const u = new URL(req.url);
    const site = (u.searchParams.get("site")||"").trim();
    const q = (u.searchParams.get("q")||"").trim();
    const k = Number(u.searchParams.get("k")||6)||6;
    const origin = req.headers.get("Origin") || "";

    if (!site || !q) return json({ ok:false, error:"Missing site or q" }, { status:400 });

    try {
        const cfg = await loadSiteConfig(env, site);
        const auth = requireApiKey(req, cfg);
        if (auth) return auth;

        const allowed = parseAllowed(env.ALLOWED_ORIGINS, cfg.search?.allowed_origins);
        if (req.method==="OPTIONS") {
            if (!allowed.includes(origin)) return json({ ok:false, error:"CORS" }, { status:403 });
            return new Response(null, { headers: corsHeaders(origin) });
        }
        if (origin && !allowed.includes(origin)) return json({ ok:false, error:"CORS" }, { status:403 });

        const vec = await embed(env, cfg.ai.embed_model || DEFAULTS.embed_model, cfg.vectorize.dims, q);
        // @ts-ignore
        const out = await env.VECTORIZE.query(vec, { topK: k, returnMetadata: true });
        return new Response(JSON.stringify({ ok:true, results: out.matches ?? [] }, null, 2), {
            headers: { "content-type":"application/json", ...(origin?corsHeaders(origin):{}) }
        });
    } catch (e:any) {
        return json({ ok:false, error:String(e?.message||e) }, { status:500 });
    }
}

async function handleAsk(req: Request, env: Env) {
    const body = await req.json().catch(()=>({})) as { site?:string; q?:string; k?:number };
    const site = (body.site||"").trim();
    const q = (body.q||"").trim();
    const k = Number(body.k||6)||6;
    if (!site || !q) return json({ ok:false, error:"Missing site or q" }, { status:400 });

    try {
        const cfg = await loadSiteConfig(env, site);
        const auth = requireApiKey(req, cfg);
        if (auth) return auth;

        const vec = await embed(env, cfg.ai.embed_model || DEFAULTS.embed_model, cfg.vectorize.dims, q);
        // @ts-ignore
        const out = await env.VECTORIZE.query(vec, { topK:k, returnMetadata:true });
        const matches = out.matches ?? [];

        const context = matches.map((m:any)=>`${m.metadata?.title||m.id}\n${m.metadata?.preview||""}`).join("\n\n");
        const sys = cfg.search?.system_prompt || "You are a concise assistant.";
        const chat = await env.AI.run(cfg.search?.chat_model || DEFAULTS.chat_model, {
            messages:[
                { role:"system", content: sys },
                { role:"user", content:`Question: ${q}\n\nContext:\n${context}` }
            ],
            temperature: cfg.search?.chat_temperature ?? DEFAULTS.chat_temperature,
            max_output_tokens: 300
        } as any);
        const answer = (chat as any).response ?? JSON.stringify(chat);
        return json({ ok:true, q, k, answer, citations: matches.map((m:any)=>m.id) });
    } catch (e:any) {
        return json({ ok:false, error:String(e?.message||e) }, { status:500 });
    }
}

/* ---------- router ---------- */
export default {
    async fetch(req: Request, env: Env) {
        const { pathname } = new URL(req.url);
        if (req.method==="GET"     && pathname==="/status") return handleStatus(req, env);
        if ((req.method==="GET"||req.method==="OPTIONS") && pathname==="/search") return handleSearch(req, env);
        if (req.method==="POST"    && pathname==="/ask") return handleAsk(req, env);
        return new Response("Not found", { status:404 });
    }
};