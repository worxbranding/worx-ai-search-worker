import type { IntentKey, IntentResult, SearchMatch } from "../lib/types";

// Intent detection and formatting hints that power `/ask` and `/search`.

export const LEVEN_LIMIT = 2;
export const INTENT_DEFAULT: IntentKey = "default";

export const DEFAULT_SYSTEM_PROMPT = `You are WORX AI, a strategic assistant that answers questions using only content from WORX.
- Speak from the client partner perspective; avoid using “we”.
- Keep responses concise, confident, and focused on outcomes.
- Always include inline Markdown links to the specific page you cite.
- Use WORX in all caps.
- If relevant information is missing, reply exactly with: "I couldn’t locate that information in the current WORX content. Try a different phrasing or explore the site for more context."`;

export const INTENT_GUIDANCE: Record<IntentKey, string> = {
  person: `Format your answer as a short paragraph that states the person’s role, highlights, and relationship to other team members when relevant. Include a direct inline link to the person’s page.`,
  service: `Summarize the service in one or two sentences, followed by a concise bullet list of key capabilities or benefits. Link to the service page.`,
  case_study: `Provide a bulleted list of two or three case studies with bold project names, the sector or challenge, and the measurable outcome. Each item should link to the specific case study.`,
  page_list: `Begin with one concise sentence introducing the list. Provide a Markdown bullet list with up to four items, each exactly in the form "- **[Page Title](https://example.com)**". Finish with the sentence "For more information, visit [Page Title](https://example.com)." using the main page from the context.`,
  how_to: `Outline the recommended steps in a numbered list. Each step should be brief and grounded in the provided content. Link to the source page that details the process.`,
  company_info: `Deliver a confident overview paragraph that highlights WORX positioning and differentiators. Link to the About or Leadership page as appropriate.`,
  contact: `Share the preferred contact method (phone, email, form) in sentence form and link directly to the contact page. Include location details when available.`,
  default: `Answer succinctly using the strongest supporting details. Highlight the most relevant facts and include inline links to the supporting page(s).`,
};

/** Normalize text so intent matching can compare tokens consistently. */
export function normalizeForCompare(value: string): string {
  return value.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase().trim();
}

/** Tokenize a string into alphanumeric words for keyword tracking. */
export function tokenize(value: string): string[] {
  return normalizeForCompare(value)
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
}

/** Classic Levenshtein distance, used to fuzzier match person names. */
export function levenshtein(a: string, b: string): number {
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;
  const dp = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0));
  for (let i = 0; i <= a.length; i++) dp[i][0] = i;
  for (let j = 0; j <= b.length; j++) dp[0][j] = j;
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost
      );
    }
  }
  return dp[a.length][b.length];
}

/** Bucket the user query into a high-level intent category. */
export function detectIntent(raw: string): IntentResult {
  const query = raw.trim();
  const lower = query.toLowerCase();
  const keywords: string[] = [];

  const excludeWords = new Set(["who", "is", "was", "the", "a", "an", "about", "tell", "me", "what"]);
  const extractNameTokens = (input: string) =>
    tokenize(input)
      .filter((word) => !excludeWords.has(word))
      .slice(0, 4);

  const personPatterns = [
    /^(who\s+(?:is|was)\s+)(.+)$/i,
    /^(tell\s+me\s+about\s+)(.+)$/i,
    /(profile|biography|team member)/i,
  ];
  for (const pattern of personPatterns) {
    const match = lower.match(pattern);
    if (match) {
      const fragment = match[2] ?? match[0] ?? lower;
      extractNameTokens(fragment).forEach((kw) => keywords.push(kw));
      return { intent: "person", keywords };
    }
  }

  if (lower.includes("case study") || lower.includes("case studies") || lower.includes("examples of work")) {
    keywords.push("case");
    return { intent: "case_study", keywords };
  }

  if (/(services?|capabilities?|offerings?|solution|package|deliver)/i.test(lower)) {
    tokenize(lower).forEach((kw) => keywords.push(kw));
    return { intent: "service", keywords };
  }

  if (/(list|show|what|which)\s+(?:pages|sections|team|case studies|services)/i.test(lower)) {
    tokenize(lower).forEach((kw) => keywords.push(kw));
    return { intent: "page_list", keywords };
  }

  if (/^how\s+do|how\s+does|steps|process|method|approach/i.test(lower)) {
    tokenize(lower).forEach((kw) => keywords.push(kw));
    return { intent: "how_to", keywords };
  }

  if (/^(what\s+is\s+worx|about\s+worx|company|history|mission)/i.test(lower) || lower.includes("win more with worx")) {
    keywords.push("worx");
    return { intent: "company_info", keywords };
  }

  if (/(contact|reach|phone|email|address|visit)/i.test(lower)) {
    tokenize(lower).forEach((kw) => keywords.push(kw));
    return { intent: "contact", keywords };
  }

  tokenize(lower).forEach((kw) => keywords.push(kw));
  return { intent: INTENT_DEFAULT, keywords };
}

/** Determine whether a match's metadata looks relevant to the detected intent. */
export function metadataMatchesKeywords(
  metadata: Record<string, unknown> | undefined,
  intent: IntentKey,
  keywords: string[]
): boolean {
  if (!metadata || !keywords.length) return true;
  const haystackParts: string[] = [];
  const fields = [
    "title",
    "collection",
    "subcategory",
    "preview",
    "seo_title",
    "seo_description",
    "children_md",
    "breadcrumbs",
    "parent_title",
    "path",
  ];
  for (const field of fields) {
    const value = metadata[field];
    if (typeof value === "string") haystackParts.push(value);
  }
  const haystack = haystackParts.join(" ").toLowerCase();
  if (!haystack) return false;

  for (const keyword of keywords) {
    const normalizedKeyword = normalizeForCompare(keyword);
    if (!normalizedKeyword) continue;

    if (haystack.includes(normalizedKeyword)) {
      return true;
    }

    if (intent === "person") {
      const words = tokenize(haystack);
      for (const word of words) {
        if (Math.abs(word.length - normalizedKeyword.length) > LEVEN_LIMIT + 1) continue;
        if (levenshtein(word, normalizedKeyword) <= LEVEN_LIMIT) {
          return true;
        }
      }
    }
  }
  return false;
}

/**
 * Re-rank matches according to the detected intent so the chat context starts
 * from the most relevant documents.
 */
export function pickMatchesByIntent(
  matches: SearchMatch[],
  intent: IntentKey,
  keywords: string[]
): SearchMatch[] {
  if (!matches.length) return matches;

  const seen = new Set<string>();
  const preferredKeyword: SearchMatch[] = [];
  const preferredOther: SearchMatch[] = [];
  const tertiaryKeyword: SearchMatch[] = [];
  const tertiaryOther: SearchMatch[] = [];
  const primary: SearchMatch[] = [];
  const secondary: SearchMatch[] = [];

  for (const match of matches) {
    const key = (match.id || "") + ":" + ((match.metadata as any)?.title || "");
    if (seen.has(key)) continue;
    seen.add(key);

    const metadata = (match.metadata || {}) as Record<string, unknown>;
    const pageKindRaw = typeof metadata.page_kind === "string" ? (metadata.page_kind as string).toLowerCase() : "";
    const isIndex = metadata["is_index"] === true || pageKindRaw === "index";
    const matchKeyword = metadataMatchesKeywords(metadata, intent, keywords);

    if (intent === "case_study") {
      if (pageKindRaw === "detail") {
        (matchKeyword ? preferredKeyword : preferredOther).push(match);
        continue;
      }
      if (isIndex) {
        (matchKeyword ? tertiaryKeyword : tertiaryOther).push(match);
        continue;
      }
    }

    if (intent === "page_list") {
      if (isIndex) {
        (matchKeyword ? preferredKeyword : preferredOther).push(match);
        continue;
      }
    }

    if (matchKeyword) {
      primary.push(match);
    } else {
      secondary.push(match);
    }
  }

  if (intent === "case_study") {
    const ordered = [
      ...preferredKeyword,
      ...preferredOther,
      ...tertiaryKeyword,
      ...tertiaryOther,
      ...primary,
      ...secondary,
    ];
    return ordered.length ? ordered : matches;
  }

  if (intent === "page_list") {
    const ordered = [
      ...preferredKeyword,
      ...preferredOther,
      ...primary,
      ...secondary,
    ];
    return ordered.length ? ordered : matches;
  }

  return primary.length ? primary : secondary;
}
