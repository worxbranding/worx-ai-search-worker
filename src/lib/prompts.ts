/**
 * Format-level rules every answer needs regardless of site/intent. These
 * sit AFTER the site + intent prompts so they always reach the model and
 * survive whatever the CMS user wrote. Keep this list small and stable —
 * it's a guardrail, not a styling layer.
 *
 * Origins:
 *   - Footnote rule: small models (CF Llama 3B / 1B) sometimes inject
 *     `[^1]` markdown footnote markers as fake citations without the
 *     matching definition. The markdown renderer shows the literal text
 *     to the user. extractResponse() strips them as a safety net, but
 *     it's better to not produce them in the first place.
 */
const FORMAT_GUARDRAILS = `**Output format guardrails (always apply)**
- Do NOT use Markdown footnote syntax. Never write \`[^1]\`, \`[^note]\`, or any \`[^...]\` reference. Cite sources as inline Markdown links only.
- Do NOT prefix the answer with meta-narration ("We need to answer...", "Let me think...", "The user is asking..."). Start directly with the answer.`;

/**
 * Combine the site-level system prompt with an optional intent-specific
 * prompt + the always-appended format guardrails. The intent prompt is
 * ADDITIVE — it never replaces the site prompt. This lets intents
 * specialize behavior (e.g. "for case studies, lead with the outcome")
 * while every answer still gets the site-wide voice + guardrails.
 *
 * If neither site nor intent is set, the caller's hard-coded fallback is
 * used (still followed by guardrails).
 */
export function buildSystemPrompt(
  sitePrompt: string | undefined | null,
  intentPrompt: string | undefined | null,
  fallback: string,
): string {
  const site = (sitePrompt || "").trim();
  const intent = (intentPrompt || "").trim();
  const head =
    site && intent ? `${site}\n\n${intent}` :
    site ? site :
    intent ? intent :
    fallback;
  return `${head}\n\n${FORMAT_GUARDRAILS}`;
}
