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
    return c.response.trim();
  }

  // OpenAI-compatible format (choices array)
  if (Array.isArray(c.choices) && c.choices.length > 0) {
    const msg = (c.choices[0] as any)?.message;
    if (msg) {
      // Standard content field
      if (typeof msg.content === "string" && msg.content.trim()) {
        return msg.content.trim();
      }
      // Reasoning models (GLM, DeepSeek-R1) put output in reasoning_content
      if (typeof msg.reasoning_content === "string" && msg.reasoning_content.trim()) {
        return msg.reasoning_content.trim();
      }
    }
  }

  // Fallback fields some models use
  if (typeof c.result === "string" && c.result.trim()) {
    return c.result.trim();
  }
  if (typeof c.text === "string" && c.text.trim()) {
    return c.text.trim();
  }
  if (typeof c.content === "string" && c.content.trim()) {
    return c.content.trim();
  }

  return null;
}
