import type { ChildLink, Env, IntentKey } from "../lib/types";
import { INTENT_DEFAULT, normalizeForCompare } from "./intent";

/** Normalize URLs so duplicate entries are easier to deduplicate. */
export function normalizeUrl(value: string): string {
  try {
    const url = new URL(value);
    url.hash = "";
    url.search = "";
    url.hostname = url.hostname.toLowerCase();
    return url.toString().replace(/\/*$/, "");
  } catch {
    return "";
  }
}

/** Parse the markdown-ish children list emitted by ingest into structured links. */
export function parseChildrenMd(childrenMd: unknown): ChildLink[] {
  if (typeof childrenMd !== "string" || !childrenMd.trim()) return [];

  const lines = childrenMd.split(/\r?\n/);
  const links: ChildLink[] = [];

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;

	const withoutPrefix = line.replace(/^#?\s*(\d+[.)]?)\s*/, "").trim();
    if (!withoutPrefix) continue;

    const parts = withoutPrefix.split(/\s+[—-]\s+/);
    if (parts.length < 2) continue;

    const title = parts[0]?.trim();
    const urlCandidate = parts.slice(1).join(" — ").trim();
    if (!title || !urlCandidate) continue;

    const normalized = normalizeUrl(urlCandidate);
    const exists = links.find((link) => {
      if (normalized && link.normalizedUrl) return link.normalizedUrl === normalized;
      return link.url === urlCandidate;
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

/** Trim the number of child links to intent-specific limits. */
export function limitChildrenForIntent(children: ChildLink[], intent: IntentKey | string): ChildLink[] {
  const limit = intent === "page_list" ? 6 : intent === "case_study" ? 4 : 3;
  return children.slice(0, limit);
}

/** Coarse HTML stripper used by Markdown conversion. */
export function stripHtmlTags(input: string): string {
  if (!input) return "";
  return input.replace(/<[^>]+>/g, "");
}

/** Convert HTML-ish model output into simple Markdown for the CMS. */
export function ensureMarkdown(answer: string): string {
  if (!answer) return "";
  let out = answer;
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

/** Grab a small snippet from KV-stored document text that matches keywords. */
export function extractSnippet(text: string, keywords: string[], maxChars: number): string | null {
  if (!text) return null;
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (!cleaned) return null;
	const sentences = cleaned.split(/(?<=[.!?])\s+/);
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

/**
 * Build the rich context blob fed into the chat model. It combines page
 * metadata, optional children lists, and KV snippets while respecting intent.
 *
 * @param fullText - Optional pre-fetched full text (from re-ranking). If provided,
 *                   avoids duplicate KV fetch. For detail pages, uses full text
 *                   without truncation.
 */
export async function buildDocContext(
  env: Env,
  metadata: Record<string, unknown>,
  maxChars = 2000,
  keywords: string[] = [],
  intent: IntentKey | string = INTENT_DEFAULT,
  parsedChildren: ChildLink[] = [],
  fullText: string | null = null
): Promise<string> {
  const parts: string[] = [];

  const url = metadata["url"] ? normalizeUrl(String(metadata["url"])) : "";
  const title = (metadata["title"] as string) || "Untitled";
  const preview = (metadata["preview"] as string) || "";
  const collection = (metadata["collection"] as string) || "";
  const parent = (metadata["parent_title"] as string) || "";
  const breadcrumbsRaw = metadata["breadcrumbs"];
  const breadcrumbs = Array.isArray(breadcrumbsRaw)
    ? (breadcrumbsRaw as string[]).join(" > ")
    : (breadcrumbsRaw as string) || "";
  const childrenRaw = parsedChildren.length ? parsedChildren : parseChildrenMd(metadata["children_md"]);
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
    const formatted = children.map(({ title: childTitle, url: childUrl }) => `- **[${childTitle}](${childUrl})**`);
    if (formatted.length) {
      parts.push(`Pages:\n${formatted.join("\n")}`);
    }
  }
  if (intent === "case_study" && children.length) {
    const formatted = children.map(({ title: childTitle, url: childUrl }) => `- [${childTitle}](${childUrl})`);
    if (formatted.length) {
      parts.push(`Case Studies:\n${formatted.join("\n")}`);
    }
  }

  // Fetch or use pre-fetched full text
  let txt = fullText; // Use pre-fetched text if available
  const kvKey = metadata["doc_key"];

  if (!txt && kvKey) {
    // Fallback: fetch from KV if not already provided
    txt = await env.CONTENT.get<string>(String(kvKey), "text");
  }

  if (txt && txt.trim()) {
    const pageKind = (metadata["page_kind"] as string) || "";
    const isDetailPage = pageKind === "detail";

    // For detail pages, use FULL TEXT (no truncation)
    // For other pages, use intelligent snippet extraction
    if (isDetailPage) {
      // Detail page: Use full content but cap at 4000 chars to avoid
      // sending excessive text to the LLM context window
      parts.push(`Details: ${txt.trim().substring(0, 4000)}`);
    } else {
      // Index/other pages: Use snippet extraction to focus on relevant content
      const snippet = extractSnippet(txt, keywords, maxChars);
      if (snippet) {
        parts.push(`Details: ${snippet}`);
      }
    }
  }

  return parts.join("\n\n");
}
