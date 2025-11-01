import { strict as assert } from "node:assert";

import { clampTemperature, resolveTopKFromQuery } from "../src/config/siteConfig";
import { detectIntent, INTENT_DEFAULT } from "../src/search/intent";
import { ensureMarkdown, normalizeUrl } from "../src/search/context";

/**
 * Simple smoke tests for the pure helper methods that power the worker. These
 * build confidence that refactors keep intent detection and prompt shaping
 * stable.
 */
(function testResolveTopKFromQuery() {
  const url = new URL("https://example.com/search?topK=12");
  assert.equal(resolveTopKFromQuery(url, 6), 12, "topK query param should override default");

  const urlFallback = new URL("https://example.com/search");
  assert.equal(resolveTopKFromQuery(urlFallback, 4), 4, "fallback should be used when param missing");
})();

(function testClampTemperature() {
  assert.equal(clampTemperature(0.7, 0.3), 0.7, "valid temperature should pass through");
  assert.equal(clampTemperature(2.5, 0.3), 1, "values above 1 clamp to 1");
  assert.equal(clampTemperature(-1, 0.3), 0, "values below 0 clamp to 0");
})();

(function testDetectIntent() {
  const person = detectIntent("Who is Jane Doe?");
  assert.equal(person.intent, "person", "person phrasing should detect person intent");
  assert(person.keywords.includes("jane"), "person intent should extract name keyword");

  const fallback = detectIntent("Insights on innovation strategy");
  assert.equal(fallback.intent, INTENT_DEFAULT, "unmatched phrasing should fall back to default intent");
})();

(function testNormalizeUrl() {
  assert.equal(
    normalizeUrl("https://Example.com/path/?utm=ad"),
    "https://example.com/path",
    "normalizeUrl should lowercase host and drop query/hash"
  );
})();

(function testEnsureMarkdown() {
  const html = "<strong>Hello</strong> <a href=\"https://worx.com\">World</a>";
  assert.equal(
    ensureMarkdown(html),
    "**Hello** [World](https://worx.com)",
    "ensureMarkdown should convert HTML to markdown"
  );
})();

console.log("helper tests passed");
