import type { BehaviorHandler, BehaviorContext, BehaviorResponse } from "./BehaviorHandler";
import { buildDocContext, normalizeUrl } from "../search/context";
import { runChat, resolveAnswerModel } from "../lib/llm";
import { buildSystemPrompt } from "../lib/prompts";

/**
 * SINGLE_PAGE_SUMMARY Behavior
 *
 * Use Case: "Who is [person]?", Specific entity queries, Biography requests
 *
 * Algorithm:
 * 1. Vector search focused on exact match (use top result primarily)
 * 2. Fetch full KV text from single best match
 * 3. Deep dive on that one topic/person/entity
 * 4. Single source citation
 *
 * Token Usage: ~400-600 tokens
 * Optimized for biographical and entity-specific queries.
 */
export class SinglePageSummary implements BehaviorHandler {
  readonly name = "single_page_summary";

  async execute(context: BehaviorContext): Promise<BehaviorResponse> {
    const { query, matches, intent, config, env } = context;

    if (!matches || matches.length === 0) {
      return {
        answer: "I couldn't find information about that person or topic.",
        behavior: this.name,
        intent: intent?.name || "person",
      };
    }

    // Focus on the single best match
    const bestMatch = matches[0];
    const metadata = (bestMatch.metadata || {}) as Record<string, unknown>;

    // Build comprehensive context from this single page
    const maxChars = Math.max(1000, Math.min(4000, Number(config.search?.max_kv_text_chars ?? 2500)));
    const url = (metadata.url as string) || (metadata.canonical as string) || "";

    const allowedUrlSet = new Set<string>();
    if (url) allowedUrlSet.add(normalizeUrl(url) || url);
    if (metadata.canonical) allowedUrlSet.add(normalizeUrl(metadata.canonical as string) || metadata.canonical as string);

    // Use pre-fetched full text from re-ranking (if available)
    // For person bios, this ensures we get the complete bio text
    const fullText = (bestMatch as any)._fullText as string | null;

    const docContext = await buildDocContext(
      env,
      metadata,
      maxChars, // Higher char limit for deep dive
      [],
      "person",
      [],
      fullText
    );

    const allowedUrls = Array.from(allowedUrlSet);
    const linkHints = allowedUrls.length > 0
      ? ["Use only these URLs when linking:", ...allowedUrls.map((u) => `- ${u}`)].join("\n")
      : "Include a link to the source page.";

    // Build system prompt for biographical/entity summary
    const basePrompt = buildSystemPrompt(
      config.search?.system_prompt,
      intent?.system_prompt,
      `You are WORX AI, a strategic assistant that provides detailed summaries of people and entities.
- Provide a comprehensive paragraph about the person or topic
- Include their role, background, and key achievements
- Mention relationships to other team members or topics when relevant
- Include an inline Markdown link to their page
- Use WORX in all caps`,
    );

    const intentGuide = `Format your answer as a detailed paragraph (or two) that covers the person or entity comprehensively. Include role, highlights, background, and relevant connections. Include a direct inline link to the page.`;

    const system = `${basePrompt}

Link Guidance:
${linkHints}

Intent focus: biography/entity summary
${intentGuide}`.trim();

    const user = `Question: ${query}\n\nContext:\n${docContext}`;

    // Resolve provider+model via the multi-provider LlmClient
    const answerModel = resolveAnswerModel(intent, config);
    const temperature = config.search?.chat_temperature ?? 0.1;
    const max_tokens = Math.max(200, Math.min(500, Number(config.search?.max_output_tokens ?? 400)));

    const result = await runChat(env, {
      provider: answerModel.provider,
      model: answerModel.model,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature,
      max_tokens,
    });

    const answer = result.answer ||
      `I found information about ${metadata.title}, but couldn't generate a summary.`;

    const { tokens_input, tokens_output, total_tokens } = result.usage;

    return {
      answer: answer as string,
      behavior: this.name,
      intent: intent?.name || "person",
      sources: [
        {
          title: (metadata.title as string) || undefined,
          url,
          score: bestMatch.score,
        },
      ],
      tokens_input,
      tokens_output,
      total_tokens,
      model: answerModel.provider === "cloudflare"
        ? answerModel.model
        : `${answerModel.provider}/${answerModel.model}`,
      temperature,
    };
  }
}
