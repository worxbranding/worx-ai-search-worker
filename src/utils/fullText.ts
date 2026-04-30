import type { Env, SearchMatch } from "../lib/types";
import { log } from "../lib/logging";

/**
 * Fetch full text content from KV for multiple search matches in parallel.
 * Adds the full text to each match as `_fullText` property.
 *
 * @param env Worker environment with KV bindings
 * @param matches Search matches to fetch text for
 * @returns Matches with _fullText property added (or null if not found)
 */
export async function fetchFullTextForMatches(
  env: Env,
  matches: SearchMatch[]
): Promise<SearchMatch[]> {
  // Fetch all texts in parallel
  const fetchPromises = matches.map(async (match) => {
    const metadata = (match.metadata || {}) as Record<string, unknown>;
    const kvKey = metadata["doc_key"];

    if (!kvKey) {
      log("[FullText] No doc_key for:", match.id);
      return { ...match, _fullText: null };
    }

    try {
      const txt = await env.CONTENT.get<string>(String(kvKey), "text");
      if (!txt || !txt.trim()) {
        log("[FullText] Empty text for:", match.id);
        return { ...match, _fullText: null };
      }

      log("[FullText] Fetched", txt.length, "chars for:", (metadata.title as string) || match.id);
      return { ...match, _fullText: txt };
    } catch (error) {
      log("[FullText] Error fetching:", match.id, String((error as Error)?.message || error));
      return { ...match, _fullText: null };
    }
  });

  return Promise.all(fetchPromises);
}
