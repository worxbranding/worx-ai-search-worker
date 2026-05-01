/**
 * Combine the site-level system prompt with an optional intent-specific
 * prompt. The intent prompt is ADDITIVE — it never replaces the site
 * prompt. This lets intents specialize behavior (e.g. "for case studies,
 * lead with the outcome") while every answer still gets the site-wide
 * voice/guardrails.
 *
 * If neither is set, the caller's hard-coded fallback is used.
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
