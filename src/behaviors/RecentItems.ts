import type { BehaviorHandler, BehaviorContext, BehaviorResponse } from "./BehaviorHandler";
import { runChat, resolveAnswerModel } from "../lib/llm";

/**
 * RECENT_ITEMS Behavior
 *
 * Use Case: "Latest X", "Recent X", "New X"
 *
 * Algorithm:
 * 1. Find pages with recent dates in metadata (updated_at, lastmod_iso)
 * 2. Sort by date (most recent first)
 * 3. Return blurb + concreteDirective for CMS to render sorted list
 *
 * Token Usage: ~150-300 tokens
 * Uses concreteDirective with date sorting for accurate recency.
 */
export class RecentItems implements BehaviorHandler {
  readonly name = "recent_items";

  async execute(context: BehaviorContext): Promise<BehaviorResponse> {
    const { query, matches, intent, config, env } = context;

    if (!matches || matches.length === 0) {
      return {
        answer: "I couldn't find recent items.",
        behavior: this.name,
        intent: intent?.name || "recent",
      };
    }

    // Find the collection/index page that contains recent items
    let bestMatch = matches[0];
    for (const match of matches.slice(0, 3)) {
      const metadata = (match.metadata || {}) as Record<string, unknown>;
      const isIndex = metadata.is_index === true || metadata.page_kind === "index";
      const hasChildren = Number(metadata.children_count || 0) > 0;

      if (isIndex && hasChildren) {
        bestMatch = match;
        break;
      }
    }

    const metadata = (bestMatch.metadata || {}) as Record<string, unknown>;
    const title = (metadata.title as string) || "items";
    const url = (metadata.url as string) || (metadata.canonical as string) || "";
    const childrenCount = Number(metadata.children_count || 0);
    const pageId = metadata.cID ? Number(metadata.cID) : undefined;
    const isIndex = metadata.is_index === true || metadata.page_kind === "index";

    // If we have an index page with children and pageId, use concreteDirective
    if (isIndex && childrenCount > 0 && pageId) {
      const systemPrompt = intent?.system_prompt ||
        `You are WORX AI. Provide a brief 1-2 sentence introduction for a list of recent items.
Be concise. The list will be rendered separately by date.`;

      const userPrompt = `Question: ${query}

Context:
Title: ${title}
URL: ${url}
Items: ${childrenCount}

Provide ONLY a brief introduction mentioning that these are the most recent items.`;

      // Resolve provider+model via the multi-provider LlmClient
      const answerModel = resolveAnswerModel(intent, config);
      const temperature = config.search?.chat_temperature ?? 0.1;

      const result = await runChat(env, {
        provider: answerModel.provider,
        model: answerModel.model,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
        temperature,
        max_tokens: 256,
      });

      const answerText = (result.answer || `Here are the most recent ${title.toLowerCase()}:`) as string;

      return {
        answer: answerText.trim(),
        concreteDirective: {
          type: "render_recent",
          pageId,
          sortBy: "date",
          limit: 5, // Show 5 most recent
        },
        behavior: this.name,
        intent: intent?.name || "recent",
        sources: [
          {
            title,
            url,
            score: bestMatch.score,
          },
        ],
      };
    }

    // Fallback: Generate answer about recent items without concreteDirective
    const systemPrompt = intent?.system_prompt ||
      config.search?.system_prompt ||
      `You are WORX AI. Provide information about recent items or updates.
Be concise and include dates when available.`;

    const userPrompt = `Question: ${query}

Context:
Title: ${title}
URL: ${url}

Provide information about recent items with a link.`;

    // Resolve provider+model via the multi-provider LlmClient
    const answerModel = resolveAnswerModel(intent, config);
    const temperature = config.search?.chat_temperature ?? 0.1;

    const result = await runChat(env, {
      provider: answerModel.provider,
      model: answerModel.model,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature,
      max_tokens: 384,
    });

    const answer = result.answer || `For recent ${title.toLowerCase()}, visit [${title}](${url}).`;

    const { tokens_input, tokens_output, total_tokens } = result.usage;

    return {
      answer: answer as string,
      behavior: this.name,
      intent: intent?.name || "recent",
      sources: [
        {
          title,
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
