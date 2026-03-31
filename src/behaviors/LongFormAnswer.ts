import type { BehaviorHandler, BehaviorContext, BehaviorResponse } from "./BehaviorHandler";
import { buildDocContext, normalizeUrl } from "../search/context";

/**
 * LONG_FORM_ANSWER Behavior
 *
 * Use Case: "Tell me about X", "Explain X", "What is your approach to X?"
 *
 * Algorithm:
 * 1. Vector search returns top matches (already done)
 * 2. Fetch full KV text for top 5-6 pages
 * 3. Build comprehensive context with multiple sources
 * 4. Generate thorough 2-3 paragraph answer with LLM
 * 5. Include citations to all source pages
 *
 * Token Usage: Flexible, uses config max_output_tokens (default: 800, max: 2048)
 * This is the default comprehensive response behavior.
 */
export class LongFormAnswer implements BehaviorHandler {
  readonly name = "long_form_answer";

  async execute(context: BehaviorContext): Promise<BehaviorResponse> {
    const { query, matches, intent, config, env } = context;

    if (!matches || matches.length === 0) {
      return {
        answer: "I couldn't locate that information in the current WORX content. Try a different phrasing or explore the site for more context.",
        behavior: this.name,
        intent: intent?.name || "default",
      };
    }

    // Use top 5 matches for comprehensive context
    const maxDocs = Math.max(1, Math.min(10, Number(config.search?.max_context_docs ?? 6)));
    const maxChars = Math.max(200, Math.min(4000, Number(config.search?.max_kv_text_chars ?? 2000)));
    const selected = matches.slice(0, Math.min(matches.length, maxDocs));

    // Build allowed URL set for link validation
    const allowedUrlSet = new Set<string>();
    const MAX_ALLOWED_URLS = 40;
    const addAllowedUrl = (candidate: unknown) => {
      if (typeof candidate !== "string") return;
      if (allowedUrlSet.size >= MAX_ALLOWED_URLS) return;
      const trimmed = candidate.trim();
      if (!trimmed) return;
      const normalized = normalizeUrl(trimmed);
      if (normalized) {
        allowedUrlSet.add(normalized);
      } else {
        allowedUrlSet.add(trimmed);
      }
    };

    // Build rich context for each document
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
          [], // No specific keywords for long form
          intent?.name || "default", // Pass detected intent for context building
          [],
          fullText
        );
        return `[#${idx + 1}] ${docContext}`;
      })
    );

    const allowedUrls = Array.from(allowedUrlSet);
    const linkHints = ["Use only these URLs when linking:", ...allowedUrls.map((u) => `- ${u}`)].join("\n");

    // Build system prompt
    const basePrompt = intent?.system_prompt ||
      config.search?.system_prompt ||
      `You are WORX AI, a strategic assistant that answers questions using only content from WORX.
- Speak from the client partner perspective; avoid using "we".
- Keep responses concise, confident, and focused on outcomes.
- Always include inline Markdown links to the specific page you cite.
- Use WORX in all caps.
- If relevant information is missing, reply exactly with: "I couldn't locate that information in the current WORX content. Try a different phrasing or explore the site for more context."`;

    const intentGuide = `CRITICAL LENGTH REQUIREMENT: Write 2-3 paragraphs for a comprehensive answer.
Your answer should be thorough and well-developed across multiple paragraphs.
Include the strongest supporting details and multiple inline links to relevant pages.
Each paragraph should cover a different aspect or provide additional depth.
This is a comprehensive response - more detailed than a single paragraph answer.`;

    const system = `${basePrompt}

Link Guidance:
${linkHints}

Intent focus: ${intent?.name || "default"}
${intentGuide}`.trim();

    const user = `Question: ${query}

Context:
${contexts.join("\n\n")}

Write a comprehensive answer with 2-3 paragraphs. Be thorough and include multiple supporting details.`;

    // Run LLM
    // Use intent-specific model, fallback to site default, then system default
    const chatModel = intent?.chat_model || config.search?.chat_model || "@cf/meta/llama-3.1-8b-instruct";
    const temperature = config.search?.chat_temperature ?? 0.1;
    // Cap at 600 tokens for faster responses while still being comprehensive
    const max_tokens = Math.max(300, Math.min(600, Number(config.search?.max_output_tokens ?? 500)));

    const chat = await env.AI.run(chatModel as any, {
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature,
      max_tokens,
    } as any);

    const answer = (chat as any).response ||
      "I couldn't locate that information in the current WORX content. Try a different phrasing or explore the site for more context.";

    // Extract token usage
    const usage = (chat as any)?.usage || (chat as any)?.meta?.usage || {};
    const tokens_input = (usage?.input_tokens ?? usage?.prompt_tokens ?? usage?.inputTokens ?? null) as number | null;
    const tokens_output = (usage?.output_tokens ?? usage?.completion_tokens ?? usage?.outputTokens ?? null) as number | null;
    const total_tokens = (usage?.total_tokens ??
      (tokens_input != null && tokens_output != null ? tokens_input + tokens_output : null)) as number | null;

    return {
      answer: answer as string,
      behavior: this.name,
      intent: intent?.name || "default",
      sources: selected.slice(0, 5).map((match) => ({
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
