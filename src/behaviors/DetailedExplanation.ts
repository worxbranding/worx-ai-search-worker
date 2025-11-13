import type { BehaviorHandler, BehaviorContext, BehaviorResponse } from "./BehaviorHandler";
import { buildDocContext, normalizeUrl } from "../search/context";

/**
 * DETAILED_EXPLANATION Behavior
 *
 * Use Case: "How does X work?", Process questions, Step-by-step instructions
 *
 * Algorithm:
 * 1. Vector search with focus on instructional content
 * 2. Fetch top 3-4 pages with full KV text
 * 3. Generate structured response (numbered steps if appropriate)
 * 4. Include multiple supporting links
 *
 * Token Usage: ~600-800 tokens
 * Optimized for instructional and process-oriented content.
 */
export class DetailedExplanation implements BehaviorHandler {
  readonly name = "detailed_explanation";

  async execute(context: BehaviorContext): Promise<BehaviorResponse> {
    const { query, matches, intent, config, env } = context;

    if (!matches || matches.length === 0) {
      return {
        answer: "I couldn't find information about that process or topic.",
        behavior: this.name,
        intent: intent?.name || "how_to",
      };
    }

    // Use top 4 matches for detailed explanation
    const maxDocs = 4;
    const maxChars = Math.max(200, Math.min(4000, Number(config.search?.max_kv_text_chars ?? 2000)));
    const selected = matches.slice(0, Math.min(matches.length, maxDocs));

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
          "how_to",
          [],
          fullText
        );
        return `[#${idx + 1}] ${docContext}`;
      })
    );

    const allowedUrls = Array.from(allowedUrlSet);
    const linkHints = ["Use only these URLs when linking:", ...allowedUrls.map((u) => `- ${u}`)].join("\n");

    // Build system prompt emphasizing step-by-step structure
    const basePrompt = intent?.system_prompt ||
      config.search?.system_prompt ||
      `You are WORX AI, a strategic assistant that explains processes clearly and comprehensively.
- Provide step-by-step explanations when appropriate
- Use numbered lists for sequential processes
- Be detailed but focused on actionable information
- Include inline Markdown links to source pages
- Use WORX in all caps`;

    const intentGuide = `Outline the recommended steps or provide a detailed explanation of the process. Be thorough but organized. Use numbered lists for sequential steps. Link to the source page that details the process.`;

    const system = `${basePrompt}

Link Guidance:
${linkHints}

Intent focus: process/how-to
${intentGuide}`.trim();

    const user = `Question: ${query}\n\nContext:\n${contexts.join("\n\n")}`;

    // Run LLM with higher token limit for detailed explanations
    const chatModel = config.search?.chat_model || "@cf/meta/llama-3.1-8b-instruct";
    const temperature = config.search?.chat_temperature ?? 0.1;
    // Allow up to 4096 tokens for detailed explanations
    const configMaxTokens = Number(config.search?.max_output_tokens);
    const max_tokens = isNaN(configMaxTokens) || configMaxTokens <= 0
      ? 2048 // Default if not configured
      : Math.min(4096, configMaxTokens); // Clamp to 4096 max

    console.log("[DetailedExplanation] config.search?.max_output_tokens =", config.search?.max_output_tokens);
    console.log("[DetailedExplanation] computed max_tokens =", max_tokens);

    const chat = await env.AI.run(chatModel as any, {
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature,
      max_tokens,
    } as any);

    const answer = (chat as any).response ||
      "I couldn't find detailed information about that process.";

    const usage = (chat as any)?.usage || (chat as any)?.meta?.usage || {};
    const tokens_input = (usage?.input_tokens ?? usage?.prompt_tokens ?? usage?.inputTokens ?? null) as number | null;
    const tokens_output = (usage?.output_tokens ?? usage?.completion_tokens ?? usage?.outputTokens ?? null) as number | null;
    const total_tokens = (usage?.total_tokens ??
      (tokens_input != null && tokens_output != null ? tokens_input + tokens_output : null)) as number | null;

    return {
      answer: answer as string,
      behavior: this.name,
      intent: intent?.name || "how_to",
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
