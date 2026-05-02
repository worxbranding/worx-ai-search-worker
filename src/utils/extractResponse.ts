/**
 * Extract the generated text from a Cloudflare AI chat response.
 *
 * Different models return text in different fields:
 *   - Most Llama/Mistral: { response: "..." }
 *   - OpenAI-compatible: { choices: [{ message: { content: "..." } }] }
 *   - Some models: { result: "..." } or { text: "..." }
 */
export function extractResponse(chat: unknown): string | null {
  if (chat == null) return null;
  const c = chat as Record<string, unknown>;

  // Standard Cloudflare AI format
  if (typeof c.response === "string" && c.response.trim()) {
    return cleanLlmOutput(c.response);
  }

  // OpenAI-compatible format (choices array)
  if (Array.isArray(c.choices) && c.choices.length > 0) {
    const msg = (c.choices[0] as any)?.message;
    if (msg) {
      // Standard content field
      if (typeof msg.content === "string" && msg.content.trim()) {
        return cleanLlmOutput(msg.content);
      }
      // Reasoning models (GLM, DeepSeek-R1) put output in reasoning_content
      if (typeof msg.reasoning_content === "string" && msg.reasoning_content.trim()) {
        return cleanLlmOutput(msg.reasoning_content);
      }
    }
  }

  // Fallback fields some models use
  if (typeof c.result === "string" && c.result.trim()) {
    return cleanLlmOutput(c.result);
  }
  if (typeof c.text === "string" && c.text.trim()) {
    return cleanLlmOutput(c.text);
  }
  if (typeof c.content === "string" && c.content.trim()) {
    return cleanLlmOutput(c.content);
  }

  return null;
}

/**
 * Strip artifacts small models occasionally emit and the renderer can't
 * make sense of.
 *
 * - Markdown footnote markers (`[^1]`, `[^note]`) — small models randomly
 *   inject these as fake citations without the matching definitions, and
 *   the markdown renderer shows the literal `[^1]` to the user. Drop both
 *   the inline markers and any trailing definition lines (`[^1]: ...`).
 * - Bare numbered citation markers (`[1]`, `[2]`) — same pattern but
 *   without the caret. Skip when followed by `(` because that's a real
 *   markdown link with a numeric anchor, not a citation footnote.
 * - Trailing "Sources:" / "References:" / "Citations:" dump blocks where
 *   small models pile up `[1] https://... [2] https://...` lines that
 *   should have been inline links in the first place.
 * - Common chain-of-thought lead-ins from open-weights models that
 *   occasionally leak past the system prompt ("We need to answer:",
 *   "Okay, let me think...", etc.).
 */
function cleanLlmOutput(text: string): string {
  let s = text;
  // Drop footnote definition lines: `[^anything]: rest of line`
  s = s.replace(/^\s*\[\^[^\]]+\]:.*$/gm, "");
  // Drop inline footnote refs: `[^anything]`
  s = s.replace(/\[\^[^\]]+\]/g, "");
  // Drop bare numeric citation markers: `[1]`, `[12]` — but not `[1](url)`
  // (a real markdown link with a numeric anchor) and not `[1]:` (a markdown
  // reference-style link definition, which renders correctly).
  s = s.replace(/\[(\d{1,2})\](?![(\:])/g, "");
  // Drop a trailing citation dump block — "Sources:" / "References:" /
  // "Citations:" header followed only by lines that look like bare URLs or
  // numbered URL pairs. Stops at a blank line that isn't part of the dump.
  s = s.replace(
    /\n+\s*\*{0,2}(?:Sources|References|Citations)\*{0,2}:?\s*\n(?:\s*(?:\d+[.)]\s*|\[\d+\]\s*|-\s*)?https?:\/\/[^\s]+(?:\s+https?:\/\/[^\s]+)*\s*\n?)+\s*$/i,
    "",
  );
  // Drop a leading chain-of-thought block (gpt-oss tells)
  s = s.replace(
    /^(we need to answer|okay,?\s+let|so we|let me think|first,?\s+we|ok,?\s+the user|the user is asking)[^\n]*\n+/i,
    "",
  );
  // Collapse the doubled blank lines our regex leaves behind
  s = s.replace(/\n{3,}/g, "\n\n");
  return s.trim();
}
