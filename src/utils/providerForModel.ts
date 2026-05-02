/**
 * Map a model id to its provider. Used by stats logging so the dashboard
 * can filter by provider without parsing model strings on the CMS side.
 */
export function providerForModel(model: string | null | undefined): string | null {
  if (!model) return null;
  const m = model.toLowerCase();
  if (m.startsWith("@cf/")) return "cloudflare";
  if (m.startsWith("claude")) return "anthropic";
  if (m.startsWith("gpt") || m.startsWith("o1") || m.startsWith("o3")) return "openai";
  if (m.startsWith("text-embedding")) return "openai";
  if (m.includes("meta-llama") || m.includes("mistral") || m.includes("qwen")) return "cloudflare";
  return null;
}
