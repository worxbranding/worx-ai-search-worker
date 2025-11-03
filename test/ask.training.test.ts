import { strict as assert } from "node:assert";

import { executeAsk } from "../src/handlers/ask";
import type { Env, SiteConfig } from "../src/lib/types";

async function testExecuteAskTrainingReRank() {
  const site = `training-site-${Date.now()}`;

  const env = {
    VECTORIZE: {
      async query() {
        return {
          matches: [
            {
              id: `${site}:doc1`,
              score: 0.2,
              metadata: {
                site,
                url: "https://example.com/doc1",
                title: "Doc One",
                freshness: 0.1,
              },
            },
            {
              id: `${site}:doc2`,
              score: 0.1,
              metadata: {
                site,
                url: "https://example.com/doc2",
                title: "Doc Two",
                freshness: 0.9,
              },
            },
          ],
        };
      },
    },
    AI: {
      async run(model: unknown) {
        if (model === "embed-model") {
          return { data: [[0.1, 0.2, 0.3]] };
        }
        return {
          response: "Policy adjusted answer",
          usage: { input_tokens: 10, output_tokens: 5 },
        };
      },
    },
    CONFIG: {
      async get(key: string) {
        if (key === `policy:${site}:active`) {
          return {
            version: "v-test",
            rerank: { weights: { base: 1, freshness: 10 } },
            prompts: {
              default: "default",
              variants: {
                default: { systemPrompt: "Policy system prompt" },
              },
            },
          };
        }
        return null;
      },
      async put() {
        /* noop */
      },
    },
  } as unknown as Env;

  const cfg: SiteConfig = {
    site_key: site,
    vectorize: { index_name: "idx", dims: 3 },
    ai: { embed_model: "embed-model" },
    search: {
      chat_model: "chat-model",
      chat_temperature: 0.5,
      system_prompt: "Base system prompt",
      max_context_docs: 2,
      max_kv_text_chars: 500,
      max_output_tokens: 512,
    },
    training: {
      policy: {
        defaultPromptVariant: "default",
      },
    },
  };

  const result = await executeAsk(
    env,
    cfg,
    {
      site,
      question: "Tell me about services",
      topK: 2,
      promptOverride: "",
      temperature: 0.4,
      wantCaching: false,
      training: true,
      promptVariant: null,
      disableCaching: false,
    },
    undefined
  );

  assert.equal(result.selected[0].id, `${site}:doc2`, "re-ranking should favor freshness-weighted doc");
  assert(result.trainingMetadata, "training metadata should be present when training flag set");
  assert.equal(
    result.trainingMetadata?.promptVariant?.key,
    "default",
    "policy-configured prompt variant should be used"
  );
  assert.equal(
    result.trainingMetadata?.matches[0].id,
    `${site}:doc2`,
    "training metadata should map reordered matches"
  );
  console.log("ask training tests passed");
}

(async function runAskTrainingTests() {
  await testExecuteAskTrainingReRank();
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
