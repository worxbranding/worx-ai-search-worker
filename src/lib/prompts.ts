/**
 * Combine the site-level system prompt with an optional intent-specific
 * prompt. The intent prompt is ADDITIVE — it never replaces the site
 * prompt. This lets intents specialize behavior (e.g. "for case studies,
 * lead with the outcome") while every answer still gets the site-wide
 * voice/guardrails.
 *
 * If neither is set, the caller's hard-coded fallback is used.
 *
 * Format-level guardrails (no footnote markers, always include a source
 * link, no meta-narration prefixes) live in the SITE system_prompt
 * editable from the CMS dashboard, NOT in worker code. Cleanup that's
 * not prompt-shaped (regex stripping `[^N]`, appending a source link
 * when the model omitted one) lives in extractResponse / ensureSourceLink.
 */
export function buildSystemPrompt(
  sitePrompt: string | undefined | null,
  intentPrompt: string | undefined | null,
  fallback: string,
): string {
  const site = (sitePrompt || "").trim();
  const intent = (intentPrompt || "").trim();
  if (site && intent) return `${site}\n\n${intent}`;
  if (site) return site;
  if (intent) return intent;
  return fallback;
}
