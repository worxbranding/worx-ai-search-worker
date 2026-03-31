import type { BehaviorHandler, BehaviorContext, BehaviorResponse } from "./BehaviorHandler";
import { buildDocContext, normalizeUrl } from "../search/context";
import { extractResponse } from "../utils/extractResponse";

/**
 * COMPARISON Behavior
 *
 * Use Case: "X vs Y", "Difference between X and Y", "Compare X and Y"
 *
 * Algorithm:
 * 1. Detect two entities in query (if possible)
 * 2. Search for pages about each entity (top 4 results)
 * 3. Fetch content for both
 * 4. Generate side-by-side comparison
 * 5. Highlight key differences
 *
 * Token Usage: ~500-700 tokens
 * Optimized for comparative analysis.
 */
export class Comparison implements BehaviorHandler {
  readonly name = "comparison";

  async execute(context: BehaviorContext): Promise<BehaviorResponse> {
    const { query, matches, intent, config, env } = context;

    if (!matches || matches.length === 0) {
      return {
        answer: "I couldn't find information to compare.",
        behavior: this.name,
        intent: intent?.name || "compare",
      };
    }

    // Use top 4 matches to potentially find both entities
    const maxDocs = Math.min(4, matches.length);
    const maxChars = Math.max(200, Math.min(3000, Number(config.search?.max_kv_text_chars ?? 1500)));
    const selected = matches.slice(0, maxDocs);

    // Build allowed URL set
    const allowedUrlSet = new Set<string>();
    const addAllowedUrl = (candidate: unknown) => {
      if (typeof candidate !== "string") return;
      if (allowedUrlSet.size >= 40) return;
      const trimmed = candidate.trim();
      if (!trimmed) return;
      const normalized = normalizeUrl(trimmed);
      if (normalized) {
        allowedUrlSet.add(normalized);
      } else {
        allowedUrlSet.add(trimmed);
      }
    };

    // Build context for each document
    const contexts = await Promise.all(
      selected.map(async (match, idx) => {
        const metadata = (match.metadata || {}) as Record<string, unknown>;
        addAllowedUrl(metadata["url"]);
        addAllowedUrl(metadata["canonical"]);

        // Use pre-fetched full text from re-ranking (if available)
        const fullText = (match as any)._fullText as string | null;

        const docContext = await buildDocContext(
          env,
          metadata,
          maxChars,
          [],
          "default",
          [],
          fullText
        );
        return `[#${idx + 1}] ${docContext}`;
      })
    );

    const allowedUrls = Array.from(allowedUrlSet);
    const linkHints = ["Use only these URLs when linking:", ...allowedUrls.map((u) => `- ${u}`)].join("\n");

    // Build system prompt for comparison
    const basePrompt = intent?.system_prompt ||
      config.search?.system_prompt ||
      `You are WORX AI, a strategic assistant that provides clear comparisons.
- Identify the two items being compared
- Highlight key differences and similarities
- Use a structured format (bullet points or table-like structure)
- Include links to both items being compared
- Be objective and factual
- Use WORX in all caps`;

    const intentGuide = `Provide a side-by-side comparison highlighting the key differences between the two items. Use a structured format with clear sections or bullet points. Include links to both items.`;

    const system = `${basePrompt}

Link Guidance:
${linkHints}

Intent focus: comparison
${intentGuide}`.trim();

    const user = `Question: ${query}\n\nContext:\n${contexts.join("\n\n")}`;

    // Run LLM
    // Use intent-specific model, fallback to site default, then system default
    const chatModel = intent?.chat_model || config.search?.chat_model || "@cf/meta/llama-3.1-8b-instruct";
    const temperature = config.search?.chat_temperature ?? 0.1;
    const max_tokens = Math.max(384, Math.min(1536, Number(config.search?.max_output_tokens ?? 896)));

    const chat = await env.AI.run(chatModel as any, {
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature,
      max_tokens,
    } as any);

    const answer = extractResponse(chat) ||
      "I couldn't generate a comparison with the available information.";

    const usage = (chat as any)?.usage || (chat as any)?.meta?.usage || {};
    const tokens_input = (usage?.input_tokens ?? usage?.prompt_tokens ?? usage?.inputTokens ?? null) as number | null;
    const tokens_output = (usage?.output_tokens ?? usage?.completion_tokens ?? usage?.outputTokens ?? null) as number | null;
    const total_tokens = (usage?.total_tokens ??
      (tokens_input != null && tokens_output != null ? tokens_input + tokens_output : null)) as number | null;

    return {
      answer: answer as string,
      behavior: this.name,
      intent: intent?.name || "compare",
      sources: selected.map((match) => ({
        title: (match.metadata?.title as string) || undefined,
        url: (match.metadata?.url as string) || (match.metadata?.canonical as string) || undefined,
        score: match.score,
      })),
      tokens_input,
      tokens_output,
      total_tokens,
      model: chatModel,
      temperature,
    };
  }
}
