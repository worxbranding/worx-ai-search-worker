import type { BehaviorHandler, BehaviorContext, BehaviorResponse } from "./BehaviorHandler";
import { buildDocContext, normalizeUrl } from "../search/context";
import { runChat, resolveAnswerModel } from "../lib/llm";
import { buildSystemPrompt } from "../lib/prompts";
import { resolveSimpleBehaviorMaxTokens } from "../lib/configDefaults";

/**
 * MEDIUM_ANSWER Behavior
 *
 * Use Case: Questions that need more than a quick fact but less than a comprehensive essay.
 * Perfect for "How does X work?", "What are the benefits of X?", "Can you explain X?"
 *
 * Algorithm:
 * 1. Vector search returns top matches (already done)
 * 2. Fetch full KV text for top 3 pages
 * 3. Build context with key sources
 * 4. Generate focused 1 paragraph answer (4-6 sentences) with LLM
 * 5. Include inline citations
 *
 * Token Usage: ~200-300 tokens (fixed max: 300)
 * Sweet spot between brevity and depth.
 */
export class MediumAnswer implements BehaviorHandler {
  readonly name = "medium_answer";

  async execute(context: BehaviorContext): Promise<BehaviorResponse> {
    const { query, matches, intent, config, env } = context;

    if (!matches || matches.length === 0) {
      return {
        answer: "I couldn't locate that information in the current WORX content. Try a different phrasing or explore the site for more context.",
        behavior: this.name,
        intent: intent?.name || "default",
      };
    }

    // Use top 3 matches for balanced context
    const maxDocs = 3;
    const maxChars = Math.max(200, Math.min(3000, Number(config.search?.max_kv_text_chars ?? 1500)));
    const selected = matches.slice(0, Math.min(matches.length, maxDocs));

    // Build allowed URL set for link validation
    const allowedUrlSet = new Set<string>();
    const MAX_ALLOWED_URLS = 20;
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
          intent?.name || "default",
          [],
          fullText
        );
        return `[#${idx + 1}] ${docContext}`;
      })
    );

    const allowedUrls = Array.from(allowedUrlSet);
    const linkHints = ["Use only these URLs when linking:", ...allowedUrls.map((u) => `- ${u}`)].join("\n");

    // Build system prompt
    const basePrompt = buildSystemPrompt(
      config.search?.system_prompt,
      intent?.system_prompt,
      `You are WORX AI, a strategic assistant that answers questions using only content from WORX.
- Speak from the client partner perspective; avoid using "we".
- Keep responses concise, confident, and focused on outcomes.
- Always include inline Markdown links to the specific page you cite.
- Use WORX in all caps.
- If relevant information is missing, reply exactly with: "I couldn't locate that information in the current WORX content. Try a different phrasing or explore the site for more context."`,
    );

    const intentGuide = `CRITICAL LENGTH REQUIREMENT: Write exactly ONE paragraph.
Your answer MUST be 4-6 sentences - no more, no less.
Include the most important details and 1-2 inline links to supporting pages.
Be thorough but stop after one paragraph. Do not write multiple paragraphs.`;

    const system = `${basePrompt}

Link Guidance:
${linkHints}

Intent focus: ${intent?.name || "default"}
${intentGuide}`.trim();

    const user = `Question: ${query}

Context:
${contexts.join("\n\n")}

Write ONE paragraph (4-6 sentences) answering this question. Stop after one paragraph.`;

    // Run LLM via the multi-provider LlmClient
    const answerModel = resolveAnswerModel(intent, config);
    const temperature = config.search?.chat_temperature ?? 0.1;
    // Per-intent max_output_tokens wins; default cfg.search.behavior_caps.medium_answer (300).
    const max_tokens = resolveSimpleBehaviorMaxTokens(config, "medium_answer");

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
      "I couldn't locate that information in the current WORX content. Try a different phrasing or explore the site for more context.";

    const { tokens_input, tokens_output, total_tokens } = result.usage;

    return {
      answer: answer as string,
      behavior: this.name,
      intent: intent?.name || "default",
      sources: selected.slice(0, 3).map((match) => ({
        title: (match.metadata?.title as string) || undefined,
        url: (match.metadata?.url as string) || (match.metadata?.canonical as string) || undefined,
        score: match.score,
      })),
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
