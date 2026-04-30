import type { BehaviorHandler, BehaviorContext, BehaviorResponse } from "./BehaviorHandler";
import { runChat, resolveAnswerModel } from "../lib/llm";

/**
 * COLLECTION_OVERVIEW Behavior
 *
 * Use Case: Summarize a category/collection without listing all items
 *
 * Algorithm:
 * 1. Find collection/category page (index page)
 * 2. Analyze metadata of children (don't fetch full text)
 * 3. Generate overview: "You have X items in Y categories"
 * 4. Provide high-level summary with optional concreteDirective
 *
 * Token Usage: ~200-300 tokens
 * Optimized for collection summaries using metadata only.
 */
export class CollectionOverview implements BehaviorHandler {
  readonly name = "collection_overview";

  async execute(context: BehaviorContext): Promise<BehaviorResponse> {
    const { query, matches, intent, config, env } = context;

    if (!matches || matches.length === 0) {
      return {
        answer: "I couldn't find that collection or category.",
        behavior: this.name,
        intent: intent?.name || "collection",
      };
    }

    // Find the best collection/index page
    let bestMatch = matches[0];
    for (const match of matches.slice(0, 3)) {
      const metadata = (match.metadata || {}) as Record<string, unknown>;
      const isIndex = metadata.is_index === true || metadata.page_kind === "index";
      if (isIndex) {
        bestMatch = match;
        break;
      }
    }

    const metadata = (bestMatch.metadata || {}) as Record<string, unknown>;
    const title = (metadata.title as string) || "this collection";
    const url = (metadata.url as string) || (metadata.canonical as string) || "";
    const preview = (metadata.preview as string) || "";
    const collection = (metadata.collection as string) || "";
    const childrenCount = Number(metadata.children_count || 0);
    const isIndex = metadata.is_index === true || metadata.page_kind === "index";
    const pageId = metadata.cID ? Number(metadata.cID) : undefined;

    // Build context for overview
    const contextParts: string[] = [];
    contextParts.push(`Title: ${title}`);
    if (url) contextParts.push(`URL: ${url}`);
    if (preview) contextParts.push(`Summary: ${preview}`);
    if (collection) contextParts.push(`Collection: ${collection}`);
    if (childrenCount > 0) contextParts.push(`Items in collection: ${childrenCount}`);

    const systemPrompt = intent?.system_prompt ||
      `You are WORX AI. Provide a high-level overview of a collection or category.
Summarize what the collection contains without listing individual items.
Include statistics if available (e.g., "X items across Y categories").
Keep it concise (2-3 paragraphs) and include a link to the collection page.`;

    const userPrompt = `Question: ${query}

Context:
${contextParts.join("\n")}

Provide a high-level overview of this collection. DO NOT list individual items.`;

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
      max_tokens: 512,
    });

    const answerText = (result.answer || `${title} contains ${childrenCount} items.`) as string;

    // If this is an index with children and we have pageId, optionally include concreteDirective
    if (isIndex && childrenCount > 0 && pageId) {
      return {
        answer: answerText.trim(),
        concreteDirective: {
          type: "render_children",
          pageId,
          sortBy: "weight",
        },
        behavior: this.name,
        intent: intent?.name || "collection",
        sources: [
          {
            title,
            url,
            score: bestMatch.score,
          },
        ],
      };
    }

    // No concreteDirective - just the overview text
    const { tokens_input, tokens_output, total_tokens } = result.usage;

    return {
      answer: answerText.trim(),
      behavior: this.name,
      intent: intent?.name || "collection",
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
