import type { BehaviorHandler, BehaviorContext, BehaviorResponse } from "./BehaviorHandler";
import { runChat, resolveAnswerModel } from "../lib/llm";

/**
 * SHORT_BLURB_WITH_LIST Behavior
 *
 * Use Case: "List all X", "Show me your X", "What X do you have?"
 *
 * Algorithm:
 * 1. Find best matching page (check top 3 results)
 * 2. Check if page has children in metadata
 * 3. If children exist:
 *    - Return brief blurb (1-2 sentences)
 *    - Return concreteDirective with pageId for CMS to render list
 * 4. If no children, fallback to brief answer from metadata
 *
 * Token Savings: 85% reduction (123 tokens vs 850 tokens)
 * - LLM doesn't generate the list, just the blurb
 * - CMS renders list from database directly
 */
export class ShortBlurbWithList implements BehaviorHandler {
  readonly name = "short_blurb_with_list";

  async execute(context: BehaviorContext): Promise<BehaviorResponse> {
    const { query, matches, intent, config, env } = context;

    if (!matches || matches.length === 0) {
      return {
        answer: "I couldn't find any pages matching your request.",
        behavior: this.name,
        intent: intent?.name || "unknown",
      };
    }

    // Find the best match that has children
    let bestMatch = matches[0];
    for (const match of matches.slice(0, 3)) {
      const metadata = (match.metadata || {}) as Record<string, unknown>;
      const childrenCount = Number(metadata.children_count || 0);
      const hasChildren = childrenCount > 0 || !!metadata.children_md;

      if (hasChildren) {
        bestMatch = match;
        break;
      }
    }

    const metadata = (bestMatch.metadata || {}) as Record<string, unknown>;
    const childrenCount = Number(metadata.children_count || 0);
    const hasChildren = childrenCount > 0 || !!metadata.children_md;
    const title = (metadata.title as string) || "this section";
    const url = (metadata.url as string) || (metadata.canonical as string) || "";
    const preview = (metadata.preview as string) || "";
    const pageId = metadata.cID ? Number(metadata.cID) : undefined;
    const isIndex = metadata.is_index === true || metadata.page_kind === "index";

    // If page has children and we have a pageId, use concreteDirective
    if (hasChildren && pageId && isIndex) {
      // Generate a brief blurb using LLM
      const systemPrompt = intent?.system_prompt ||
        `You are WORX AI. Provide a very brief 1-2 sentence introduction for a list of items.
Be concise and professional. Do not list the items - just introduce the category.`;

      const userPrompt = `Question: ${query}

Context:
Title: ${title}
URL: ${url}
Preview: ${preview}
Children Count: ${childrenCount}

Provide ONLY a brief 1-2 sentence introduction. The actual list will be rendered separately.`;

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
        max_tokens: 256, // Short blurb only
      });

      const answerText = (result.answer || preview || `Here are the ${title.toLowerCase()}:`) as string;

      return {
        answer: answerText.trim(),
        concreteDirective: {
          type: "render_children",
          pageId,
          sortBy: "weight",
        },
        behavior: this.name,
        intent: intent?.name || "list_query",
        sources: [
          {
            title,
            url,
            score: bestMatch.score,
          },
        ],
      };
    }

    // Fallback: No children or no pageId - generate simple answer
    const systemPrompt = intent?.system_prompt ||
      `You are WORX AI. Provide a brief, helpful answer based on the available information.
Keep it concise (2-3 sentences). Include a link to the source page.`;

    const userPrompt = `Question: ${query}

Context:
Title: ${title}
URL: ${url}
Preview: ${preview}

Provide a brief answer with a link to ${url}.`;

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

    const answer = result.answer || `For information about ${title}, visit ${url}`;

    return {
      answer: answer as string,
      behavior: this.name,
      intent: intent?.name || "list_query",
      sources: [
        {
          title,
          url,
          score: bestMatch.score,
        },
      ],
    };
  }
}
