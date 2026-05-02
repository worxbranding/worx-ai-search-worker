/**
 * Substring patterns that indicate the model failed to answer from WORX
 * content. Smaller models (e.g. llama-3.2-3b) often emit the no-answer
 * phrase and then ramble with general-knowledge filler ("However, I can
 * still tell you that..."), so we cannot require an exact full-response
 * match. Substring presence is enough to flag found_index = false.
 */
const NO_ANSWER_SUBSTRINGS: string[] = [
    "couldn't find an answer based on the available information",
    "couldn't locate that information in the current worx content",
    "could not locate that information in the current worx content",
    "could not find an answer based on the available information",
];

/** Normalize punctuation and whitespace for comparison against patterns. */
function normalizeFallback(text: string): string {
    return text
        .trim()
        .replace(/\s+/g, " ")
        .replace(/’/g, "'")
        .toLowerCase();
}

/**
 * Returns true when the model response indicates it could not answer the
 * question from indexed content — even if the model also added unrelated
 * filler around the disclaimer.
 */
export function isNoAnswer(text: string | null | undefined): boolean {
    if (text == null) return true;

    const normalized = normalizeFallback(String(text));
    if (!normalized) return true;

    for (const pat of NO_ANSWER_SUBSTRINGS) {
        if (normalized.includes(pat)) return true;
    }
    return false;
}
