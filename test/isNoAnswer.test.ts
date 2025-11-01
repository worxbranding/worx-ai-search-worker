
import { strict as assert } from "node:assert";

import { isNoAnswer } from "../src/utils/isNoAnswer";

const fallback = "I'm sorry, I couldn't find an answer based on the available information.";
assert.equal(isNoAnswer(fallback), true, "Exact fallback should be treated as no-answer");

const fallbackNoPeriod = "I'm sorry, I couldn't find an answer based on the available information";
assert.equal(isNoAnswer(fallbackNoPeriod), true, "Fallback without trailing punctuation should still match");

const fallbackAlt = "I couldn't find an answer based on the available information.";
assert.equal(isNoAnswer(fallbackAlt), true, "Alternate fallback phrasing should match");

const fallbackNew = "I couldn’t locate that information in the current WORX content. Try a different phrasing or explore the site for more context.";
assert.equal(isNoAnswer(fallbackNew), true, "Updated fallback phrasing should match");

const helpfulAnswer = "Couldn\u2019t find an answer? Try these steps: open the app, navigate to Settings, and tap Help.";
assert.equal(isNoAnswer(helpfulAnswer), false, "Helpful answers mentioning the phrase should not be marked as no-answer");

const realAnswer = "I'm sorry this is happening, but you can reboot the device and it will resolve the issue.";
assert.equal(isNoAnswer(realAnswer), false, "Apologetic tone alone should not count as no-answer");

const blank = "   ";
assert.equal(isNoAnswer(blank), true, "Blank responses should count as no-answer");

console.log("isNoAnswer tests passed");
