// WORX AI Search Worker — multi-tenant + API key + CORS + Vectorize v2 (ENRICHED CONTEXT)

import { LOGGING_ENABLED, log, startTimer, time } from "./log";
import { isNoAnswer } from "./utils/isNoAnswer";

// ---- Type declarations ----
declare interface VectorizeIndex {
    query(arg: number[] | { vectorId: string } | any, opts?: any): Promise<{ matches?: Array<any> }>;
}

declare interface Ai {
    run(model: any, input: any): Promise<any>;
}

declare interface ExecutionContext {
    waitUntil(promise: Promise<any>): void;
}

declare interface KVNamespace {
    get<T = string>(key: string, type?: "text" | "json"): Promise<T | null>;
    put?(key: string, value: string, opts?: any): Promise<void>;
    delete?(key: string): Promise<void>;
    list?(opts?: { prefix?: string; limit?: number; cursor?: string }): Promise<{ keys: Array<{ name: string }>; list_complete: boolean; cursor?: string | null }>;
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
        // Performance tuning knobs (optional):
        max_context_docs?: number;          // default 6
        max_kv_text_chars?: number;         // default 3000
        answer_cache_ttl?: number;          // seconds, default 600
        caching?: boolean;                  // default false; can be overridden per-request via caching=1|0
    };
};

type SearchMatch = { id: string; score: number; metadata?: Record<string, any> };
type ChildLink = { title: string; url: string; normalizedUrl: string };

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

// Robust Vectorize v2 query-by-id wrapper: tries multiple request shapes
async function vectorizeQueryById(env: Env, id: string, k: number): Promise<{ matches?: Array<any> } | null> {
    const options = { topK: k, returnMetadata: true, includeMetadata: true } as any;
    const shapes: Array<{ desc: string; payload: any }> = [
        { desc: "{ id }", payload: { id, ...options } },
        { desc: "{ vectorId }", payload: { vectorId: id, ...options } },
        { desc: "{ vector: { id } }", payload: { vector: { id }, ...options } },
    ];

    let lastErr: any = null;
    for (const s of shapes) {
        try {
            // @ts-ignore Cloudflare Vectorize typings are permissive
            const out = await time(`VECTORIZE.query(byId:${s.desc})`, () => env.VECTORIZE.query(s.payload));
            if (out && Array.isArray(out.matches)) return out;
            lastErr = lastErr || new Error("No matches returned");
        } catch (e: any) {
            lastErr = e;
            const msg = String(e?.message || e);
            // Continue trying next shape on common id-not-found or invalid-vector errors
            if (msg.includes("40006") || msg.toLowerCase().includes("invalid query vector")) {
                log("[vectorizeQueryById] failed", s.desc, "for id", id, "->", msg);
                continue;
            }
            // Other errors: rethrow
            throw e;
        }
    }
    if (lastErr) throw lastErr;
    return null;
}

function allowOrigin(origin: string | null, cfg: SiteConfig, env: Env): string | null {
    const fromCfg = cfg.search?.allowed_origins || [];
    let allowed = fromCfg.length ? fromCfg : (env.ALLOWED_ORIGINS || "").split(",").map(s => s.trim()).filter(Boolean);
    if (!origin) return null;
    if (allowed.includes("*")) return "*";
    return allowed.includes(origin) ? origin : null;
}

// Resolve whether caching should be used for this request.
// Precedence: query param caching=1|0 overrides config (cfg.search.caching), default false.
function resolveCaching(u: URL, cfg: SiteConfig): boolean {
    const raw = (u.searchParams.get("caching") || "").trim();
    if (raw === "1") return true;
    if (raw === "0") return false;
    return !!cfg.search?.caching;
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
    const raw = await time("AI.run(embed)", () => env.AI.run(model as any, { text }));
    const arr = pluckEmbedding(raw, dims);
    if (!arr) throw new Error("Embedding dims mismatch or invalid shape");
    return arr;
}

/** SHA-1 hex for small cache keys (Workers supports crypto.subtle). */
async function sha1Hex(s: string): Promise<string> {
    const data = new TextEncoder().encode(s);
    const digest = await crypto.subtle.digest("SHA-1", data);
    return Array.from(new Uint8Array(digest)).map(b => b.toString(16).padStart(2, "0")).join("");
}

/** Try to cache embeddings in KV when available; no-ops if KV write isn’t bound. */
async function cachedEmbed(env: Env, model: string, dims: number, text: string, ctx?: ExecutionContext, useCache: boolean = false): Promise<number[]> {
    const key = `qemb:${await sha1Hex(text)}`;

    // If caching is not requested, bypass KV entirely
    if (!useCache) {
        return await time("embed(cachedEmbed)", () => embed(env, model, dims, text));
    }

    try {
        const cached = await env.CONFIG.get<number[]>(key, "json");
        if (cached && Array.isArray(cached) && cached.length === dims) {
            log("[cachedEmbed] HIT", key);
            return cached;
        }
    } catch (e) {
        log("[cachedEmbed] READ-ERROR", key, String((e as Error)?.message || e));
    }

    const emb = await time("embed(cachedEmbed)", () => embed(env, model, dims, text));
    log("[cachedEmbed] MISS", key, "(computed)");

    try {
        if (ctx?.waitUntil && env.CONFIG.put) {
            ctx.waitUntil(env.CONFIG.put(key, JSON.stringify(emb), { expirationTtl: 86400 }));
            log("[cachedEmbed] STORE-QUEUED", key, "ttl=86400");
        } else {
            const stop = startTimer("KV put qemb");
            await env.CONFIG.put!(key, JSON.stringify(emb), { expirationTtl: 86400 });
            stop();
            log("[cachedEmbed] STORED", key, "ttl=86400");
        }
    } catch (e) {
        log("[cachedEmbed] STORE-SKIP", key, "KV not writable or put() unavailable");
    }

    return emb;
}

type IntentKey =
    | "person"
    | "service"
    | "case_study"
    | "page_list"
    | "how_to"
    | "company_info"
    | "contact"
    | "default";

interface IntentResult {
    intent: IntentKey;
    keywords: string[];
}

const LEVEN_LIMIT = 2;
const INTENT_DEFAULT: IntentKey = "default";

const DEFAULT_SYSTEM_PROMPT = `You are WORX AI, a strategic assistant that answers questions using only content from WORX.
- Speak from the client partner perspective; avoid using “we”.
- Keep responses concise, confident, and focused on outcomes.
- Always include inline Markdown links to the specific page you cite.
- Use WORX in all caps.
- If relevant information is missing, reply exactly with: "I couldn’t locate that information in the current WORX content. Try a different phrasing or explore the site for more context."`;

const INTENT_GUIDANCE: Record<IntentKey, string> = {
    person: `Format your answer as a short paragraph that states the person’s role, highlights, and relationship to other team members when relevant. Include a direct inline link to the person’s page.`,
    service: `Summarize the service in one or two sentences, followed by a concise bullet list of key capabilities or benefits. Link to the service page.`,
    case_study: `Provide a bulleted list of two or three case studies with bold project names, the sector or challenge, and the measurable outcome. Each item should link to the specific case study.`,
    page_list: `Begin with one concise sentence introducing the list. Provide a Markdown bullet list with up to four items, each exactly in the form "- **[Page Title](https://example.com)**". Finish with the sentence "For more information, visit [Page Title](https://example.com)." using the main page from the context.`,
    how_to: `Outline the recommended steps in a numbered list. Each step should be brief and grounded in the provided content. Link to the source page that details the process.`,
    company_info: `Deliver a confident overview paragraph that highlights WORX positioning and differentiators. Link to the About or Leadership page as appropriate.`,
    contact: `Share the preferred contact method (phone, email, form) in sentence form and link directly to the contact page. Include location details when available.`,
    default: `Answer succinctly using the strongest supporting details. Highlight the most relevant facts and include inline links to the supporting page(s).`,
};

function normalizeForCompare(value: string): string {
    return value.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase().trim();
}

function tokenize(value: string): string[] {
    return normalizeForCompare(value)
        .split(/[^a-z0-9]+/)
        .filter(Boolean);
}

function levenshtein(a: string, b: string): number {
    if (a === b) return 0;
    if (!a.length) return b.length;
    if (!b.length) return a.length;
    const dp = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0));
    for (let i = 0; i <= a.length; i++) dp[i][0] = i;
    for (let j = 0; j <= b.length; j++) dp[0][j] = j;
    for (let i = 1; i <= a.length; i++) {
        for (let j = 1; j <= b.length; j++) {
            const cost = a[i - 1] === b[j - 1] ? 0 : 1;
            dp[i][j] = Math.min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            );
        }
    }
    return dp[a.length][b.length];
}

function detectIntent(raw: string): IntentResult {
    const query = raw.trim();
    const lower = query.toLowerCase();
    const keywords: string[] = [];

    const excludeWords = new Set(["who", "is", "was", "the", "a", "an", "about", "tell", "me", "what"]);
    const extractNameTokens = (input: string) =>
        tokenize(input)
            .filter((word) => !excludeWords.has(word))
            .slice(0, 4);

    const personPatterns = [
        /^(who\s+(?:is|was)\s+)(.+)$/i,
        /^(tell\s+me\s+about\s+)(.+)$/i,
        /(profile|biography|team member)/i,
    ];
    for (const pattern of personPatterns) {
        const match = lower.match(pattern);
        if (match) {
            const fragment = match[2] ?? match[0] ?? lower;
            extractNameTokens(fragment).forEach((kw) => keywords.push(kw));
            return { intent: "person", keywords };
        }
    }

    if (lower.includes("case study") || lower.includes("case studies") || lower.includes("examples of work")) {
        keywords.push("case");
        return { intent: "case_study", keywords };
    }

    if (/(services?|capabilities?|offerings?|solution|package|deliver)/i.test(lower)) {
        tokenize(lower).forEach((kw) => keywords.push(kw));
        return { intent: "service", keywords };
    }

    if (/(list|show|what|which)\s+(?:pages|sections|team|case studies|services)/i.test(lower)) {
        tokenize(lower).forEach((kw) => keywords.push(kw));
        return { intent: "page_list", keywords };
    }

    if (/^how\s+do|how\s+does|steps|process|method|approach/i.test(lower)) {
        tokenize(lower).forEach((kw) => keywords.push(kw));
        return { intent: "how_to", keywords };
    }

    if (/^(what\s+is\s+worx|about\s+worx|company|history|mission)/i.test(lower) || lower.includes("win more with worx")) {
        keywords.push("worx");
        return { intent: "company_info", keywords };
    }

    if (/(contact|reach|phone|email|address|visit)/i.test(lower)) {
        tokenize(lower).forEach((kw) => keywords.push(kw));
        return { intent: "contact", keywords };
    }

    tokenize(lower).forEach((kw) => keywords.push(kw));
    return { intent: INTENT_DEFAULT, keywords };
}

function metadataMatchesKeywords(metadata: Record<string, any> | undefined, intent: IntentKey, keywords: string[]): boolean {
    if (!metadata || !keywords.length) return true;
    const haystackParts: string[] = [];
    const fields = [
        "title",
        "collection",
        "subcategory",
        "preview",
        "seo_title",
        "seo_description",
        "children_md",
        "breadcrumbs",
        "parent_title",
        "path",
    ];
    for (const field of fields) {
        const value = metadata[field];
        if (typeof value === "string") haystackParts.push(value);
    }
    const haystack = haystackParts.join(" ").toLowerCase();
    if (!haystack) return false;

    for (const keyword of keywords) {
        const normalizedKeyword = normalizeForCompare(keyword);
        if (!normalizedKeyword) continue;

        if (haystack.includes(normalizedKeyword)) {
            return true;
        }

        if (intent === "person") {
            const words = tokenize(haystack);
            for (const word of words) {
                if (Math.abs(word.length - normalizedKeyword.length) > LEVEN_LIMIT + 1) continue;
                if (levenshtein(word, normalizedKeyword) <= LEVEN_LIMIT) {
                    return true;
                }
            }
        }
    }
    return false;
}

function pickMatchesByIntent(matches: SearchMatch[], intent: IntentKey, keywords: string[]): SearchMatch[] {
    if (!matches.length) return matches;

    const seen = new Set<string>();
    const preferredKeyword: SearchMatch[] = [];
    const preferredOther: SearchMatch[] = [];
    const tertiaryKeyword: SearchMatch[] = [];
    const tertiaryOther: SearchMatch[] = [];
    const primary: SearchMatch[] = [];
    const secondary: SearchMatch[] = [];

    for (const match of matches) {
        const key = (match.id || "") + ":" + (match.metadata?.title || "");
        if (seen.has(key)) continue;
        seen.add(key);

        const metadata = (match.metadata || {}) as Record<string, any>;
        const pageKindRaw = typeof metadata.page_kind === "string" ? metadata.page_kind.toLowerCase() : "";
        const isIndex = metadata.is_index === true || pageKindRaw === "index";
        const matchKeyword = metadataMatchesKeywords(metadata, intent, keywords);

        if (intent === "case_study") {
            if (pageKindRaw === "detail") {
                (matchKeyword ? preferredKeyword : preferredOther).push(match);
                continue;
            }
            if (isIndex) {
                (matchKeyword ? tertiaryKeyword : tertiaryOther).push(match);
                continue;
            }
        }

        if (intent === "page_list") {
            if (isIndex) {
                (matchKeyword ? preferredKeyword : preferredOther).push(match);
                continue;
            }
        }

        if (matchKeyword) {
            primary.push(match);
        } else {
            secondary.push(match);
        }
    }

    if (intent === "case_study") {
        const ordered = [
            ...preferredKeyword,
            ...preferredOther,
            ...tertiaryKeyword,
            ...tertiaryOther,
            ...primary,
            ...secondary,
        ];
        return ordered.length ? ordered : matches;
    }

    if (intent === "page_list") {
        const ordered = [
            ...preferredKeyword,
            ...preferredOther,
            ...primary,
            ...secondary,
        ];
        return ordered.length ? ordered : matches;
    }

    return primary.length ? primary : secondary;
}

function clampTopK(value: number, min = 1, max = 24): number {
    if (!Number.isFinite(value)) return min;
    return Math.min(Math.max(Math.floor(value), min), max);
}

function resolveTopKFromQuery(u: URL, fallback: number): number {
    const candidates = [
        u.searchParams.get("k"),
        u.searchParams.get("topK"),
        u.searchParams.get("top_k"),
        u.searchParams.get("limit"),
    ];
    for (const raw of candidates) {
        if (!raw) continue;
        const n = Number(raw);
        if (!Number.isFinite(n)) continue;
        const clamped = clampTopK(n);
        if (clamped > 0) return clamped;
    }
    return clampTopK(fallback);
}

function clampTemperature(value: number, fallback: number): number {
    if (!Number.isFinite(value)) return fallback;
    return Math.min(1, Math.max(0, Number(value)));
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

function parseChildrenMd(childrenMd: unknown): ChildLink[] {
    if (typeof childrenMd !== "string" || !childrenMd.trim()) return [];

    const lines = childrenMd.split(/\r?\n/);
    const links: ChildLink[] = [];

    for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line) continue;

        const withoutPrefix = line.replace(/^#?\s*(\d+[\.\)]?)\s*/, "").trim();
        if (!withoutPrefix) continue;

        const parts = withoutPrefix.split(/\s+[—-]\s+/);
        if (parts.length < 2) continue;

        const title = parts[0]?.trim();
        const urlCandidate = parts.slice(1).join(" — ").trim();
        if (!title || !urlCandidate) continue;

        const normalized = normalizeUrl(urlCandidate);
        const exists = links.find((l) => {
            if (normalized && l.normalizedUrl) return l.normalizedUrl === normalized;
            return l.url === urlCandidate;
        });
        if (exists) continue;

        links.push({
            title,
            url: urlCandidate,
            normalizedUrl: normalized,
        });
    }

    return links;
}

function limitChildrenForIntent(children: ChildLink[], intent: IntentKey): ChildLink[] {
    const limit = intent === "page_list" ? 4 : intent === "case_study" ? 4 : 0;
    return limit ? children.slice(0, limit) : children;
}

function stripHtmlTags(input: string): string {
    return input.replace(/<[^>]+>/g, "");
}

function ensureMarkdown(answer: string): string {
    if (typeof answer !== "string" || !answer.trim()) return answer;

    let out = answer;
    out = out.replace(/\r\n/g, "\n");
    out = out.replace(/&nbsp;/gi, " ");
    out = out.replace(/&amp;/gi, "&");
    out = out.replace(/&lt;/gi, "<");
    out = out.replace(/&gt;/gi, ">");

    out = out.replace(/<br\s*\/?>/gi, "\n");
    out = out.replace(/<\/p>\s*<p>/gi, "\n\n");
    out = out.replace(/<p>/gi, "");
    out = out.replace(/<\/p>/gi, "\n\n");
    out = out.replace(/<\/?strong>/gi, "**");
    out = out.replace(/<\/?b>/gi, "**");
    out = out.replace(/<\/?em>/gi, "*");
    out = out.replace(/<\/?i>/gi, "*");
    out = out.replace(/<li>\s*/gi, "- ");
    out = out.replace(/<\/li>/gi, "\n");
    out = out.replace(/<\/?ul>/gi, "\n");
    out = out.replace(/<\/?ol>/gi, "\n");

    out = out.replace(/<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)<\/a>/gi, (_, href: string, text: string) => {
        const cleanedText = ensureMarkdown(stripHtmlTags(text)).trim() || href.trim();
        return `[${cleanedText}](${href.trim()})`;
    });

    out = out.replace(/<h[1-6]>/gi, "\n\n**");
    out = out.replace(/<\/h[1-6]>/gi, "**\n\n");

    out = out.replace(/<div[^>]*>/gi, "\n");
    out = out.replace(/<\/div>/gi, "\n");
    out = out.replace(/<span[^>]*>/gi, "");
    out = out.replace(/<\/span>/gi, "");

    out = stripHtmlTags(out);
    out = out.replace(/\n{3,}/g, "\n\n");
    return out.trim();
}

// ---- Build enriched document context ----
function extractSnippet(text: string, keywords: string[], maxChars: number): string | null {
    if (!text) return null;
    const cleaned = text.replace(/\s+/g, " ").trim();
    if (!cleaned) return null;
    const sentences = cleaned.split(/(?<=[\.!\?])\s+/);
    const normalizedKeywords = keywords.map(normalizeForCompare).filter(Boolean);

    const chosen: string[] = [];

    if (normalizedKeywords.length) {
        for (const sentence of sentences) {
            const candidate = sentence.trim();
            if (!candidate) continue;
            const normalizedSentence = normalizeForCompare(candidate);
            if (normalizedKeywords.some((kw) => normalizedSentence.includes(kw))) {
                chosen.push(candidate);
                if (chosen.join(" ").length >= maxChars) break;
            }
        }
    }

    if (!chosen.length) {
        chosen.push(...sentences.slice(0, Math.min(3, sentences.length)));
    }

    const snippet = chosen.join(" ");
    return snippet.slice(0, Math.max(0, maxChars)) || null;
}

async function buildDocContext(
    env: Env,
    md: Record<string, any>,
    maxChars: number = 2000,
    keywords: string[] = [],
    intent: IntentKey = INTENT_DEFAULT,
    parsedChildren: ChildLink[] = []
): Promise<string> {
    const parts: string[] = [];

    const url = md.url ? normalizeUrl(md.url) : "";
    const title = md.title || "Untitled";
    const preview = md.preview || "";
    const collection = md.collection || "";
    const parent = md.parent_title || "";
    const breadcrumbs = Array.isArray(md.breadcrumbs) ? md.breadcrumbs.join(" > ") : (md.breadcrumbs || "");
    const childrenRaw = parsedChildren.length ? parsedChildren : parseChildrenMd(md.children_md);
    const baseUrl = url;
    const childrenFiltered =
        intent === "page_list" && baseUrl
            ? childrenRaw.filter((child) => {
                const childNormalized = child.normalizedUrl || normalizeUrl(child.url);
                return childNormalized !== baseUrl;
            })
            : childrenRaw;
    const children = limitChildrenForIntent(childrenFiltered, intent);

    if (url) {
        parts.push(`**${title}** (${url})`);
    } else {
        parts.push(`**${title}**`);
    }

    if (collection) parts.push(`Collection: ${collection}`);
    if (parent) parts.push(`Parent: ${parent}`);
    if (breadcrumbs) parts.push(`Breadcrumbs: ${breadcrumbs}`);

    if (preview) {
        parts.push(`Summary: ${preview}`);
    }

    if (intent === "page_list" && children.length) {
        const formatted = children
            .map(({ title: childTitle, url: childUrl }) => `- **[${childTitle}](${childUrl})**`);
        if (formatted.length) {
            parts.push(`Pages:\n${formatted.join("\n")}`);
        }
    }
    if (intent === "case_study" && children.length) {
        const formatted = children
            .map(({ title: childTitle, url: childUrl }) => `- [${childTitle}](${childUrl})`);
        if (formatted.length) {
            parts.push(`Case Studies:\n${formatted.join("\n")}`);
        }
    }

    const kvKey = md.doc_key;
    if (kvKey) {
        const txt = await env.CONFIG.get<string>(kvKey, "text");
        const snippet = extractSnippet(txt || "", keywords, maxChars);
        if (snippet) {
            parts.push(`Details: ${snippet}`);
        }
    }

    return parts.join("\n\n");
}

// ---- Handlers ----
async function handleStatus(req: Request, env: Env) {
    const stop = startTimer("handleStatus");
    const u = new URL(req.url);
    const site = (u.searchParams.get("site") || "").trim();
    if (!site) { stop(); return json({ ok: false, error: "Missing ?site=" }, { status: 400 }); }
    const cfg = await loadSiteConfig(env, site);
    const res = json({
        ok: true,
        site,
        vectorize: cfg.vectorize.index_name,
        dims: cfg.vectorize.dims,
        embed_model: cfg.ai.embed_model,
        chat_model: cfg.search?.chat_model || "@cf/meta/llama-3.1-8b-instruct",
        system_prompt: cfg.search?.system_prompt || null,
        chat_temperature: cfg.search?.chat_temperature ?? null,
        requires_api_key: !!wantApiKey(cfg),
    });
    stop();
    return res;
}

async function handleSearch(req: Request, env: Env, cfg: SiteConfig, ctx?: ExecutionContext) {
    const stop = startTimer("handleSearch");
    const u = new URL(req.url);
    const site = (u.searchParams.get("site") || "").trim();
    const q = (u.searchParams.get("q") || "").trim();
    const intentInfo = detectIntent(q);
    const cfgTopK = Number(cfg.search?.topK ?? 6);
    const defaultK = clampTopK(Number.isFinite(cfgTopK) ? cfgTopK : 6);
    const k = resolveTopKFromQuery(u, defaultK);
    const wantDebug = (u.searchParams.get("debug") || "") === "1";
    const wantCaching = resolveCaching(u, cfg);
    if (!site) { stop(); return json({ ok: false, error: "Missing ?site=" }, { status: 400 }); }
    if (!q) { stop(); return json({ ok: false, error: "Missing ?q=" }, { status: 400 }); }

    const vec = await time("cachedEmbed(search)", () => cachedEmbed(env, cfg.ai.embed_model, cfg.vectorize.dims, q, ctx, wantCaching));
    // @ts-ignore
    const out = await time("VECTORIZE.query(search)", () => env.VECTORIZE.query(vec, { topK: k, returnMetadata: true }));
    const pre = (out?.matches || []) as SearchMatch[];
    const post = filterToSite(site, pre);

    const body: any = { ok: true, site, q, k, results: post, intent: intentInfo.intent };
    if (wantDebug) body._debug = { total_pre: pre.length, total_post: post.length, sample_ids: post.slice(0, 3).map((m) => m.id) };
    stop();
    return json(body);
}

// ---- Enhanced ASK ----
async function handleAsk(req: Request, env: Env, cfg: SiteConfig, ctx?: ExecutionContext) {
    const stop = startTimer("handleAsk");
    const u = new URL(req.url);
    const wantDebug = (u.searchParams.get("debug") || "") === "1";
    const wantCaching = resolveCaching(u, cfg);
    const body = (await req.json().catch(() => ({}))) as { site?: string; q?: string; k?: number };
    const site = (body.site || "").trim();
    const q = (body.q || "").trim();
    if (!site) { stop(); return json({ ok: false, error: "Missing field 'site'" }, { status: 400 }); }
    if (!q) { stop(); return json({ ok: false, error: "Missing field 'q'" }, { status: 400 }); }

    const intentInfo = detectIntent(q);
    const baseTopK = clampTopK(Number(body.k || cfg.search?.topK || 6));
    const initialK = Math.min(baseTopK, 3);
    const fallbackK = Math.max(initialK + 3, Math.min(baseTopK, 6));

    const promptOverrideRaw = typeof body.systemPrompt === "string"
        ? body.systemPrompt
        : (typeof body.system_prompt === "string" ? body.system_prompt : "");
    const promptOverride = promptOverrideRaw ? String(promptOverrideRaw).trim() : "";
    const configPrompt = cfg.search?.system_prompt || DEFAULT_SYSTEM_PROMPT;

    const temperatureCandidateRaw =
        body.chatTemperature ??
        body.temperature ??
        body.chat_temperature ??
        null;
    const configTemperature = Number(cfg.search?.chat_temperature ?? 0.1);
    const temperatureCandidate =
        temperatureCandidateRaw !== null &&
        temperatureCandidateRaw !== undefined &&
        !Number.isNaN(Number(temperatureCandidateRaw))
            ? Number(temperatureCandidateRaw)
            : configTemperature;
    const temperature = clampTemperature(temperatureCandidate, configTemperature);

    const vec = await time("cachedEmbed(ask)", () => cachedEmbed(env, cfg.ai.embed_model, cfg.vectorize.dims, q, ctx, wantCaching));
    const runQuery = async (topK: number) => {
        // @ts-ignore
        const res = await time("VECTORIZE.query(ask)", () => env.VECTORIZE.query(vec, { topK, includeMetadata: true, returnMetadata: true }));
        return filterToSite(site, res?.matches || []);
    };

    let matches = pickMatchesByIntent(await runQuery(initialK), intentInfo.intent, intentInfo.keywords);
    if (!matches.length && fallbackK > initialK) {
        matches = pickMatchesByIntent(await runQuery(fallbackK), intentInfo.intent, intentInfo.keywords);
    }

    const resolvedK = matches.length ? Math.min(matches.length, baseTopK) : initialK;
    const maxDocs = Math.max(1, Math.min(10, Number(cfg.search?.max_context_docs ?? 6)));
    const maxChars = Math.max(200, Math.min(4000, Number(cfg.search?.max_kv_text_chars ?? 2000)));
    const selected = matches.slice(0, Math.min(matches.length, maxDocs));

    const allowedUrlSet = new Set<string>();
    const MAX_ALLOWED_URLS = 40;
    const addAllowedUrl = (candidate: unknown) => {
        if (typeof candidate !== "string") return;
        if (allowedUrlSet.size >= MAX_ALLOWED_URLS) return;
        const trimmed = candidate.trim();
        if (!trimmed) return;
        const normalized = normalizeUrl(trimmed);
        if (normalized) {
            allowedUrlSet.add(normalized);
        } else {
            allowedUrlSet.add(trimmed);
        }
    };

    const docStop = startTimer("buildDocContext(filtered)");
    const contexts = await Promise.all(
        selected.map(async (m, i) => {
            const md = (m.metadata || {}) as any;
            addAllowedUrl(md.url);
            addAllowedUrl(md.canonical);
            const childrenAll = parseChildrenMd(md.children_md);
            const baseUrl = normalizeUrl(md.url || md.canonical || "");
            const childrenFiltered =
                intentInfo.intent === "page_list" && baseUrl
                    ? childrenAll.filter((child) => {
                        const childNormalized = child.normalizedUrl || normalizeUrl(child.url);
                        return childNormalized !== baseUrl;
                    })
                    : childrenAll;
            const children = limitChildrenForIntent(childrenFiltered, intentInfo.intent);
            for (const child of children) {
                addAllowedUrl(child.url);
                if (child.normalizedUrl) addAllowedUrl(child.normalizedUrl);
            }
            const docContext = await buildDocContext(env, md, maxChars, intentInfo.keywords, intentInfo.intent, children);
            return `[#${i + 1}] ${docContext}`;
        })
    );
    docStop();

    const allowedUrls = Array.from(allowedUrlSet);
    const linkHints = ["Use only these URLs when linking:", ...allowedUrls.map((u) => `- ${u}`)].join("\n");

    const basePrompt = (promptOverride !== "" ? promptOverride : configPrompt).trim();
    const intentGuide = (INTENT_GUIDANCE[intentInfo.intent] || INTENT_GUIDANCE.default).trim();
    const system = `${basePrompt}

Link Guidance:
${linkHints}

Intent focus: ${intentInfo.intent}
${intentGuide}`.trim();

    const user = `Question: ${q}\n\nContext:\n${contexts.join("\n\n")}`;
    const chatModel = cfg.search?.chat_model || "@cf/meta/llama-3.1-8b-instruct";
    const max_output_tokens = Math.max(128, Math.min(2048, Number(cfg.search?.max_output_tokens ?? 1024)));

    // Answer cache: short TTL to speed up repeated questions while keeping freshness
    const ansKeyRaw = JSON.stringify({
        site,
        q,
        k: resolvedK,
        chatModel,
        temperature,
        max_output_tokens,
        intent: intentInfo.intent,
        systemHash: await sha1Hex(system)
    });
    const ansKey = `ans:${await sha1Hex(ansKeyRaw)}`;
    if (wantCaching) {
        try {
            const cachedAns = await env.CONFIG.get<string>(ansKey, "text");
            if (cachedAns && cachedAns.trim()) {
                const sanitizedCached = ensureMarkdown(cachedAns);
                log("[cachedAnswer] HIT", ansKey);
                const nowIso = new Date().toISOString();
                const noAnswer = isNoAnswer(sanitizedCached);
                const foundIndex = (Array.isArray(matches) && matches.length > 0) && !noAnswer;
                const stats = {
                    question: q,
                    found_index: foundIndex,
                    cached: true,
                    model: chatModel,
                    tokens_input: null,
                    tokens_output: null,
                    total_tokens: null,
                    timestamp: nowIso,
                    temperature,
                    intent: intentInfo.intent,
                };
                const bodyOut: any = { ok: true, q, k: resolvedK, answer: sanitizedCached, stats, intent: intentInfo.intent };
                if (wantDebug) bodyOut._debug = { matches: matches.slice(0, 3), cache: "hit", intent: intentInfo.intent };
                const response = json(bodyOut);
                stop();
                return response;
            }
        } catch (e) {
            log("[cachedAnswer] READ-ERROR", ansKey, String((e as Error)?.message || e));
        }
    }

    const chat = await time("AI.run(chat)", () => env.AI.run(chatModel as any, {
        messages: [
            { role: "system", content: system },
            { role: "user", content: user },
        ],
        temperature,
        max_output_tokens,
    } as any));

    const rawAnswer = (chat as any).response || "I couldn’t locate that information in the current WORX content. Try a different phrasing or explore the site for more context.";
    const answer = ensureMarkdown(rawAnswer);
    const noAnswer = isNoAnswer(answer);
    // Store answer in cache with TTL when KV is writable and caching is enabled
    const ansTtl = Math.max(60, Math.min(86400, Number(cfg.search?.answer_cache_ttl ?? 600)));
    if (wantCaching) {
        try {
            if (ctx?.waitUntil && env.CONFIG.put) {
                ctx.waitUntil(env.CONFIG.put(ansKey, answer, { expirationTtl: ansTtl }));
                log("[cachedAnswer] STORE-QUEUED", ansKey, `ttl=${ansTtl}`);
            } else {
                const stopPut = startTimer("KV put ans");
                await env.CONFIG.put!(ansKey, answer, { expirationTtl: ansTtl });
                stopPut();
                log("[cachedAnswer] STORED", ansKey, `ttl=${ansTtl}`);
            }
        } catch (e) {
            log("[cachedAnswer] STORE-SKIP", ansKey, "KV not writable or put() unavailable");
        }
    }

    const usage = (chat as any)?.usage || (chat as any)?.meta?.usage || {};
    const tokens_input = (usage?.input_tokens ?? usage?.prompt_tokens ?? usage?.inputTokens ?? null) as number | null;
    const tokens_output = (usage?.output_tokens ?? usage?.completion_tokens ?? usage?.outputTokens ?? null) as number | null;
    const total_tokens = (usage?.total_tokens ?? (tokens_input != null && tokens_output != null ? tokens_input + tokens_output : null)) as number | null;
    const nowIso2 = new Date().toISOString();
    const foundIndex2 = (Array.isArray(matches) && matches.length > 0) && !noAnswer;
    const stats = {
        question: q,
        found_index: foundIndex2,
        cached: false,
        model: chatModel,
        tokens_input,
        tokens_output,
        total_tokens,
        timestamp: nowIso2,
        temperature,
        intent: intentInfo.intent,
    };
    const bodyOut: any = { ok: true, q, k: resolvedK, answer, stats, intent: intentInfo.intent };
    if (wantDebug) bodyOut._debug = { matches: matches.slice(0, 3), cache: "miss", intent: intentInfo.intent };
    const response = json(bodyOut);
    stop();
    return response;
}

// ---- Admin: Clear Cache ----
async function kvDeleteByPrefix(ns: KVNamespace, prefix: string, ctx?: ExecutionContext): Promise<{ prefix: string; total: number; queued: number }> {
    if (!ns.list || !ns.delete) throw new Error("KV list/delete not available on this binding");
    let cursor: string | undefined = undefined;
    let total = 0;
    let queued = 0;
    do {
        const page = await ns.list!({ prefix, limit: 1000, cursor });
        const names = (page.keys || []).map(k => k.name);
        total += names.length;
        if (names.length) {
            const delAll = Promise.all(names.map(n => ns.delete!(n).catch(() => {})));
            if (ctx?.waitUntil) {
                ctx.waitUntil(delAll);
                queued += names.length;
            } else {
                await delAll;
                queued += names.length;
            }
        }
        cursor = (page as any).cursor || undefined;
        if (page.list_complete) break;
    } while (cursor);
    return { prefix, total, queued };
}

async function handleClearCache(req: Request, env: Env, cfg: SiteConfig, ctx?: ExecutionContext) {
    const stop = startTimer("handleClearCache");
    const u = new URL(req.url);
    const qsScope = (u.searchParams.get("scope") || "").trim();
    const body = (await req.json().catch(() => ({}))) as { scope?: string };
    const scope = String(body.scope || qsScope || "all").toLowerCase();

    if (!env.CONFIG.list || !env.CONFIG.delete) {
        stop();
        return json({ ok: false, error: "CONFIG KV does not support list/delete in this environment" }, { status: 501 });
    }

    const prefixes = scope === "answers" ? ["ans:"] : scope === "embeddings" ? ["qemb:"] : ["ans:", "qemb:"];

    const results: Array<{ prefix: string; total: number; queued: number }> = [];
    for (const p of prefixes) {
        try {
            const r = await time(`KV clear ${p}`, () => kvDeleteByPrefix(env.CONFIG, p, ctx));
            results.push(r);
            log("[clear-cache]", p, r);
        } catch (e: any) {
            results.push({ prefix: p, total: 0, queued: 0 });
            log("[clear-cache] ERROR", p, String(e?.message || e));
        }
    }

    stop();
    return json({ ok: true, scope, results, note: ctx?.waitUntil ? "Deletes queued asynchronously" : "Deletes completed synchronously" });
}

// ---- Debug ----
async function handleDebugEmbed(req: Request, env: Env, cfg: SiteConfig) {
    const stop = startTimer("handleDebugEmbed");
    const u = new URL(req.url);
    const q = (u.searchParams.get("q") || "").trim();
    if (!q) { stop(); return json({ ok: false, error: "Missing ?q=" }, { status: 400 }); }
    try {
        const arr = await embed(env, cfg.ai.embed_model, cfg.vectorize.dims, q);
        const res = json({ ok: true, q, result: { valid: Array.isArray(arr) && arr.length === cfg.vectorize.dims, gotLen: arr.length, expected: cfg.vectorize.dims } });
        stop();
        return res;
    } catch (e: any) {
        stop();
        return json({ ok: false, q, error: String(e?.message || e) }, { status: 500 });
    }
}

async function handleDebugQueryById(req: Request, env: Env, cfg: SiteConfig) {
    const stop = startTimer("handleDebugQueryById");
    const u = new URL(req.url);
    const site = (u.searchParams.get("site") || "").trim();
    const idRaw = (u.searchParams.get("id") || "").trim();
    const kRaw = (u.searchParams.get("k") || "").trim();
    const kDigits = (kRaw.match(/\d+/)?.[0] || "");
    const kParsed = parseInt(kDigits, 10);
    const k = Math.max(1, Math.min(24, Number.isFinite(kParsed) ? kParsed : 3));
    const useRaw = ["1", "true", "yes"].includes((u.searchParams.get("raw") || "").trim().toLowerCase());
    const forceDoc = ["1", "true", "yes"].includes((u.searchParams.get("doc") || "").trim().toLowerCase());
    const wantCaching = resolveCaching(u, cfg);
    if (!site) { stop(); return json({ ok: false, error: "Missing ?site=" }, { status: 400 }); }
    if (!idRaw) { stop(); return json({ ok: false, error: "Missing ?id" }, { status: 400 }); }

    // Build a robust list of IDs to try:
    // - If raw=1, only try the provided id exactly as-is.
    // - If doc=1, try only doc:-prefixed variants of the provided forms.
    // - Otherwise, try in order: site-prefixed, as-provided, stripped; then their doc:-prefixed counterparts.
    const prefixed = sitePrefixedId(site, idRaw);
    const stripped = idRaw.startsWith(`${site}:`) ? idRaw.slice(site.length + 1) : null;

    let candidateIds: string[] = [];
    if (useRaw) {
        candidateIds = [idRaw];
    } else if (forceDoc) {
        candidateIds = [
            `doc:${prefixed}`,
            `doc:${idRaw}`,
            ...(stripped ? [`doc:${stripped}`] : []),
        ];
    } else {
        candidateIds = [
            prefixed,
            idRaw,
            ...(stripped ? [stripped] : []),
            `doc:${prefixed}`,
            `doc:${idRaw}`,
            ...(stripped ? [`doc:${stripped}`] : []),
        ];
    }

    const tryIds: string[] = Array.from(new Set(candidateIds.filter(Boolean)));

    let lastErr: any = null;

    for (let i = 0; i < tryIds.length; i++) {
        const vectorId = tryIds[i];
        try {
            const out = await vectorizeQueryById(env, vectorId, k);
            const post = filterToSite(site, out?.matches || []);
            if ((out?.matches && out.matches.length > 0) || post.length > 0) {
                log("[debug/query-by-id] SUCCESS with", vectorId);
                const res = json({ ok: true, id: vectorId, k, results: post });
                stop();
                return res;
            }
            // If no matches, try next id if available
            lastErr = lastErr || new Error("No matches returned for this id");
        } catch (e: any) {
            lastErr = e;
            const msg = String(e?.message || e);
            // 40006 is commonly returned when the id is not found in the index
            if (msg.includes("40006") || msg.toLowerCase().includes("invalid query vector")) {
                log("[debug/query-by-id] attempt failed for", vectorId, "->", msg);
                continue; // try next id if available
            }
            // For any other error, break and report
            break;
        }
    }

    // Fallback: try embedding the KV doc text for candidate doc keys and query by vector
    try {
        // Build candidate KV doc keys from tried IDs
        const kvCandidates = Array.from(new Set(
            tryIds.flatMap((vid) => {
                const arr: string[] = [];
                if (vid.startsWith("doc:")) arr.push(vid);
                arr.push(`doc:${vid}`);
                if (!vid.startsWith(`${site}:`)) arr.push(`doc:${site}:${vid}`);
                return arr;
            })
        ));

        for (const kvKey of kvCandidates) {
            try {
                const txt = await env.CONFIG.get<string>(kvKey, "text");
                if (txt && txt.trim()) {
                    const snippet = txt.slice(0, 3000);
                    const vec = await cachedEmbed(env, cfg.ai.embed_model, cfg.vectorize.dims, snippet, undefined, wantCaching);
                    // @ts-ignore
                    const out = await env.VECTORIZE.query(vec, { topK: k, returnMetadata: true, includeMetadata: true });
                    const post = filterToSite(site, out?.matches || []);
                    if (post.length > 0) {
                        log("[debug/query-by-id] FALLBACK success via KV doc", kvKey);
                        const res = json({ ok: true, id: idRaw, k, results: post, note: `fallback: embedded KV text via ${kvKey}` });
                        stop();
                        return res;
                    }
                }
            } catch (e) {
                // ignore and try next
            }
        }
    } catch (e) {
        log("[debug/query-by-id] fallback error", String((e as any)?.message || e));
    }

    const hints = [
        "The provided id might not exist in the index.",
        "Try adding &doc=1 to query using doc:-prefixed ids (e.g., doc:<site>:page:<id>).",
        "If your vectors are stored without a site prefix, try adding &raw=1 to use the id exactly as provided.",
        `Tried ids: ${tryIds.join(", ")}`,
        "Note: We also attempted a fallback by embedding the KV document text when available."
    ];

    const errMsg = lastErr ? String(lastErr?.message || lastErr) : "No matches returned for any attempted id variant";
    stop();
    return json({ ok: false, error: errMsg, site, id: idRaw, k, hints, tried: tryIds }, { status: 400 });
}

// List candidate vector IDs by scanning CONFIG KV for document keys (doc:<site>:...)
async function handleDebugListIds(req: Request, env: Env, cfg: SiteConfig) {
    const stop = startTimer("handleDebugListIds");
    const u = new URL(req.url);
    const site = (u.searchParams.get("site") || "").trim();
    const limit = Math.max(1, Math.min(1000, Number(u.searchParams.get("limit") || 100)));
    const cursor = u.searchParams.get("cursor") || undefined;
    const wantStripped = ["1", "true", "yes"].includes((u.searchParams.get("stripped") || "").trim().toLowerCase());
    const wantValidate = ["1", "true", "yes"].includes((u.searchParams.get("validate") || "").trim().toLowerCase());
    if (!site) { stop(); return json({ ok: false, error: "Missing ?site=" }, { status: 400 }); }
    if (!env.CONFIG.list) { stop(); return json({ ok: false, error: "CONFIG KV does not support list() in this environment" }, { status: 501 }); }

    const prefix = `doc:${site}:`;
    const page = await time(`KV list ${prefix}`, () => env.CONFIG.list!({ prefix, limit, cursor }));
    const names = (page.keys || []).map(k => k.name);
    const idsPrefixed = names.map(n => n.startsWith("doc:") ? n.slice(4) : n);
    const idsStripped = idsPrefixed.map(v => v.startsWith(`${site}:`) ? v.slice(site.length + 1) : v);

    let validatedOk: string[] = [];
    let validatedFail: string[] = [];

    if (wantValidate && idsPrefixed.length) {
        for (const vid of idsPrefixed) {
            try {
                await vectorizeQueryById(env, vid, 1);
                validatedOk.push(vid);
            } catch (e: any) {
                const msg = String(e?.message || e);
                if (msg.includes("40006") || msg.toLowerCase().includes("invalid query vector")) {
                    validatedFail.push(vid);
                } else {
                    // Non-existence vs other error: still record as failure but annotate in logs
                    log("[debug/list-ids] validate error for", vid, "->", msg);
                    validatedFail.push(vid);
                }
            }
        }
    }

    const body: any = {
        ok: true,
        site,
        prefix,
        limit,
        total_page: names.length,
        list_complete: !!(page as any).list_complete,
        next_cursor: (page as any).cursor || null,
        ids_prefixed: idsPrefixed,
        ids_stripped: idsStripped,
    };
    if (wantStripped) body.ids = idsStripped; else body.ids = idsPrefixed;
    if (wantValidate) body.validation = { ok: validatedOk.length, fail: validatedFail.length, fail_sample: validatedFail.slice(0, 20) };

    stop();
    return json(body);
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
    async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
        const u = new URL(req.url);
        const { pathname } = u;
        const site = (u.searchParams.get("site") || "").trim();
        log("REQUEST", req.method, pathname, site ? `(site=${site})` : "");

        let cfg: SiteConfig | undefined;
        if (site) {
            try { cfg = await time("loadSiteConfig", () => loadSiteConfig(env, site)); }
            catch (e: any) { return json({ ok: false, error: String(e?.message || e) }, { status: 500 }); }
        }

        const origin = req.headers.get("Origin");
        const originAllow = cfg ? allowOrigin(origin, cfg, env) : null;

        if (req.method === "OPTIONS") {
            if (origin && !originAllow) return json({ ok: false, error: "CORS: origin not allowed" }, { status: 403 });
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

            if (req.method === "GET" && pathname === "/search") return withCors(originAllow, await time("route:/search", () => handleSearch(req, env, cfg!, ctx)));
            if (req.method === "POST" && pathname === "/ask") return withCors(originAllow, await time("route:/ask", () => handleAsk(req, env, cfg!, ctx)));
            if (req.method === "POST" && pathname === "/admin/clear-cache") return withCors(originAllow, await time("route:/admin/clear-cache", () => handleClearCache(req, env, cfg!, ctx)));
            if (req.method === "GET" && pathname === "/admin/clear-cache") return withCors(originAllow, await time("route:/admin/clear-cache", () => handleClearCache(req, env, cfg!, ctx)));
            if (req.method === "GET" && pathname === "/debug/embed") return withCors(originAllow, await time("route:/debug/embed", () => handleDebugEmbed(req, env, cfg!)));
            if (req.method === "GET" && pathname === "/debug/query-by-id") return withCors(originAllow, await time("route:/debug/query-by-id", () => handleDebugQueryById(req, env, cfg!)));
            if (req.method === "GET" && pathname === "/debug/list-ids") return withCors(originAllow, await time("route:/debug/list-ids", () => handleDebugListIds(req, env, cfg!)));

            return withCors(originAllow, new Response("Not found", { status: 404 }));
        } catch (e: any) {
            log("[ERROR]", req.method, pathname, String(e?.message || e));
            return withCors(originAllow, json({ ok: false, error: String(e?.message || e) }, { status: 500 }));
        }
    },
};
