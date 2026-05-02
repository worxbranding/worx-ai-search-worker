import type { BehaviorHandler, BehaviorContext, BehaviorResponse } from "./BehaviorHandler";
import { runChat, resolveAnswerModel } from "../lib/llm";
import { buildSystemPrompt } from "../lib/prompts";
import { resolveSimpleBehaviorMaxTokens } from "../lib/configDefaults";

/**
 * NAVIGATION_HELP Behavior
 *
 * Use Case: "Where can I find X?", Navigation questions
 *
 * Algorithm:
 * 1. Search for pages matching topic
 * 2. Return breadcrumb/navigation path
 * 3. Direct link to section with path guidance
 *
 * Token Usage: ~100-200 tokens
 * Optimized for navigation and "where to find" queries.
 */
export class NavigationHelp implements BehaviorHandler {
  readonly name = "navigation_help";

  async execute(context: BehaviorContext): Promise<BehaviorResponse> {
    const { query, matches, intent, config, env } = context;

    if (!matches || matches.length === 0) {
      return {
        answer: "I couldn't find that section on the site.",
        behavior: this.name,
        intent: intent?.name || "navigation",
      };
    }

    // Use the best match for navigation
    const bestMatch = matches[0];
    const metadata = (bestMatch.metadata || {}) as Record<string, unknown>;

    const title = (metadata.title as string) || "that page";
    const url = (metadata.url as string) || (metadata.canonical as string) || "";
    const path = (metadata.path as string) || "";
    const breadcrumbsRaw = metadata.breadcrumbs;
    const breadcrumbs = Array.isArray(breadcrumbsRaw)
      ? (breadcrumbsRaw as string[]).join(" > ")
      : (breadcrumbsRaw as string) || "";
    const parentTitle = (metadata.parent_title as string) || "";

    // Build navigation context
    const contextParts: string[] = [];
    contextParts.push(`Page: ${title}`);
    if (url) contextParts.push(`URL: ${url}`);
    if (path) contextParts.push(`Path: ${path}`);
    if (breadcrumbs) contextParts.push(`Navigation: ${breadcrumbs}`);
    if (parentTitle) contextParts.push(`Under: ${parentTitle}`);

    const systemPrompt = buildSystemPrompt(
      config.search?.system_prompt,
      intent?.system_prompt,
      `You are WORX AI. Provide clear navigation guidance to help users find pages.
- Give the direct link to the page
- Mention the breadcrumb path if available
- Be concise and helpful (2-3 sentences)
- Use clear directional language`,
    );

    const userPrompt = `Question: ${query}

Context:
${contextParts.join("\n")}

Provide navigation guidance with the breadcrumb path and direct link.`;

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
      // Per-intent max_output_tokens wins; default cfg.search.behavior_caps.navigation_help (256).
      max_tokens: resolveSimpleBehaviorMaxTokens(config, "navigation_help"),
    });

    // Generate answer with navigation path
    let answer = result.answer;

    // Fallback if LLM doesn't provide good answer
    if (!answer || answer.length < 20) {
      const parts: string[] = [];
      if (breadcrumbs) {
        parts.push(`You can find **${title}** by navigating: ${breadcrumbs}.`);
      } else {
        parts.push(`You can find **${title}** on the site.`);
      }
      parts.push(`[Visit ${title}](${url})`);
      answer = parts.join(" ");
    }

    const { tokens_input, tokens_output, total_tokens } = result.usage;

    return {
      answer: answer.trim(),
      behavior: this.name,
      intent: intent?.name || "navigation",
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
