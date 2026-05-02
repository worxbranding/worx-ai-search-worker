/**
 * Guarantee every answer drives the user to a real page on the site.
 *
 * The widget IS a search — the goal is to surface the most relevant page,
 * give a confident summary, and send the user there to read the full
 * thing (and ideally fill out a lead form). If the LLM forgets to include
 * a link (small models drop this 10-20% of the time even when prompted),
 * append one referencing the top source.
 *
 * - If the answer already contains a markdown link `[text](url)` whose URL
 *   matches one of the sources, leave it alone.
 * - Otherwise append a "**Read more:** [Title](URL)" line pointing at the
 *   top-ranked source (sources are pre-sorted highest-score-first by the
 *   re-rank pipeline).
 *
 * Keep the appended copy short and call-to-action-shaped — this is a
 * lead-gen pattern.
 */

type Source = { title?: string; url?: string; score?: number; [k: string]: unknown };

const MARKDOWN_LINK_RE = /\[[^\]]+\]\(([^)\s]+)/g;

export function ensureSourceLink(answer: string, sources: Source[]): string {
  if (!answer || !sources || sources.length === 0) return answer;

  const sourceUrls = new Set(
    sources.map((s) => normalizeUrl(s.url)).filter((u): u is string => !!u),
  );
  if (sourceUrls.size === 0) return answer;

  // Already linking to at least one of the sources?
  for (const m of answer.matchAll(MARKDOWN_LINK_RE)) {
    const linkUrl = normalizeUrl(m[1]);
    if (linkUrl && sourceUrls.has(linkUrl)) return answer;
  }

  // Pick the top source (highest score = first after re-rank).
  const top = sources[0];
  if (!top?.url || !top?.title) return answer;

  // Trim trailing whitespace from the answer, then append the CTA on its own line.
  const trimmed = answer.replace(/\s+$/, "");
  return `${trimmed}\n\n**Read more:** [${top.title}](${top.url})`;
}

/** Normalize a URL for cheap "does this point at the same page" comparison. */
function normalizeUrl(u: unknown): string | null {
  if (typeof u !== "string" || !u) return null;
  try {
    const parsed = new URL(u);
    return parsed.origin + parsed.pathname.replace(/\/$/, "");
  } catch {
    return u.replace(/\/$/, "");
  }
}
