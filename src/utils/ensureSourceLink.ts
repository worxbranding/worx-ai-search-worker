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

export function ensureSourceLink(answer: string, sources: Source[]): string {
  if (!answer || !sources || sources.length === 0) return answer;

  const sourceUrls = sources
    .map((s) => normalizeUrl(s.url))
    .filter((u): u is string => !!u);
  if (sourceUrls.length === 0) return answer;

  // Skip the append whenever the answer already references any source URL
  // in *any* form — markdown link `[...](url)`, parenthetical `(url)`, or
  // bare URL. Small models often skip the proper `[anchor](url)` syntax
  // and just write `(https://...)` next to the text; that still drives
  // the user to the page, so don't pile a redundant "Read more:" on top.
  const answerNorm = answer.toLowerCase();
  for (const u of sourceUrls) {
    if (answerNorm.includes(u.toLowerCase())) return answer;
  }

  // No source URL appears anywhere in the text — append the top source as a CTA.
  const top = sources[0];
  if (!top?.url || !top?.title) return answer;
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
