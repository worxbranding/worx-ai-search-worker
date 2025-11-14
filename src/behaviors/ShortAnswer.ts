import type { BehaviorHandler, BehaviorContext, BehaviorResponse } from "./BehaviorHandler";

/**
 * SHORT_ANSWER Behavior
 *
 * Use Case: Quick facts, "What is X?", "Where is X?", "When is X?"
 *
 * Algorithm:
 * 1. Vector search returns top 2 matches
 * 2. Use ONLY metadata + preview (no KV text fetch for speed)
 * 3. Generate 1-2 sentence concise response
 * 4. Include single best link
 *
 * Token Usage: ~50-150 tokens (fixed max: 150)
 * Optimized for speed and extreme brevity.
 */
export class ShortAnswer implements BehaviorHandler {
  readonly name = "short_answer";

  async execute(context: BehaviorContext): Promise<BehaviorResponse> {
    const { query, matches, intent, config, env } = context;

    if (!matches || matches.length === 0) {
      return {
        answer: "I couldn't find that information.",
        behavior: this.name,
        intent: intent?.name || "quick_fact",
      };
    }

    // Use only top 2 matches for quick response
    const bestMatch = matches[0];
    const metadata = (bestMatch.metadata || {}) as Record<string, unknown>;

    const title = (metadata.title as string) || "this page";
    const url = (metadata.url as string) || (metadata.canonical as string) || "";
    const preview = (metadata.preview as string) || "";
    const collection = (metadata.collection as string) || "";

    // Build minimal context from metadata only
    const contextParts: string[] = [];
    if (title) contextParts.push(`Title: ${title}`);
    if (url) contextParts.push(`URL: ${url}`);
    if (preview) contextParts.push(`Summary: ${preview}`);
    if (collection) contextParts.push(`Category: ${collection}`);

    const systemPrompt = intent?.system_prompt ||
      `You are WORX AI. You MUST provide a VERY BRIEF answer.
CRITICAL: Keep your answer to 1-2 sentences ONLY. Be extremely concise.
Include ONE link to the most relevant page in Markdown format [text](url).
Do not elaborate or provide extra details. Answer the question directly and stop.`;

    const userPrompt = `Question: ${query}

Context:
${contextParts.join("\n")}

Provide ONLY a 1-2 sentence answer with one link to ${url}. Be extremely brief.`;

    const chatModel = config.search?.chat_model || "@cf/meta/llama-3.1-8b-instruct";
    const temperature = config.search?.chat_temperature ?? 0.1;

    const chat = await env.AI.run(chatModel as any, {
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature,
      max_tokens: 150, // Reduced from 256 - force brevity
    } as any);

    const answer = (chat as any).response || `For information about ${title}, visit [${title}](${url}).`;

    const usage = (chat as any)?.usage || (chat as any)?.meta?.usage || {};
    const tokens_input = (usage?.input_tokens ?? usage?.prompt_tokens ?? usage?.inputTokens ?? null) as number | null;
    const tokens_output = (usage?.output_tokens ?? usage?.completion_tokens ?? usage?.outputTokens ?? null) as number | null;
    const total_tokens = (usage?.total_tokens ??
      (tokens_input != null && tokens_output != null ? tokens_input + tokens_output : null)) as number | null;

    return {
      answer: answer as string,
      behavior: this.name,
      intent: intent?.name || "quick_fact",
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
      model: chatModel,
      temperature,
    };
  }
}
