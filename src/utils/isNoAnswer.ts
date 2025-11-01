/** Known fallback phrases (normalized) returned when the worker has no answer. */
const FALLBACK_RESPONSES = new Set([
    "i'm sorry, i couldn't find an answer based on the available information",
    "im sorry, i couldn't find an answer based on the available information",
    "i couldn't find an answer based on the available information",
    "i couldn't locate that information in the current worx content. try a different phrasing or explore the site for more context",
]);

/** Normalize punctuation and whitespace for comparison against the fallback set. */
function normalizeFallback(text: string): string {
    return text
        .trim()
        .replace(/\s+/g, " ")
        .replace(/[’]/g, "'")
        .toLowerCase();
}

/**
 * Determines whether a model response represents an explicit "no answer" fallback.
 * Matches the exact fallback phrasing (with minor punctuation variance) to avoid
 * misclassifying real answers that mention similar wording.
 */
export function isNoAnswer(text: string | null | undefined): boolean {
    if (text == null) return true;

    const normalized = normalizeFallback(String(text));
    if (!normalized) return true;

    const canonical = normalized.replace(/[.!?\s]+$/g, "");
    if (!canonical) return true;

    if (FALLBACK_RESPONSES.has(canonical)) return true;

    return FALLBACK_RESPONSES.has(normalized);
}
