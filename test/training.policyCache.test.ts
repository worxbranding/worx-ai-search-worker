import { strict as assert } from "node:assert";

import { getPolicySnapshot, pickPromptVariant, getReRankWeights } from "../src/training/policyCache";
import type { Env, SiteConfig } from "../src/lib/types";

async function testPolicySnapshotCaching() {
  let lookupCount = 0;
  const env = {
    CONFIG: {
      async get() {
        lookupCount += 1;
        return { version: "v1" };
      },
    },
  } as unknown as Env;

  const cfg = {
    site_key: "site-a",
    vectorize: { index_name: "idx", dims: 1 },
    ai: { embed_model: "model" },
    training: { policy: { cacheTtlSeconds: 60 } },
  } as unknown as SiteConfig;

  const site = `test-${Date.now()}`;
  const first = await getPolicySnapshot(env, cfg, site, { cacheTtlMs: 1_000 });
  const second = await getPolicySnapshot(env, cfg, site);

  assert.equal(lookupCount, 1, "policy snapshot should be cached for repeated lookups");
  assert.deepEqual(first, second, "cached result should match initial snapshot");
}

function testPickPromptVariant() {
  const snapshot = {
    prompts: {
      default: "alpha",
      variants: {
        alpha: { systemPrompt: "Prompt A" },
        beta: { systemPrompt: "Prompt B" },
      },
    },
  } as any;

  const direct = pickPromptVariant(snapshot, "beta");
  assert.equal(direct?.key, "beta", "explicit prompt variant should be returned");

  const fallback = pickPromptVariant(snapshot);
  assert.equal(fallback?.key, "alpha", "default variant should be used when no candidate provided");

  const firstAvailable = pickPromptVariant(
    { prompts: { variants: { gamma: { systemPrompt: "Prompt C" } } } } as any,
    "missing"
  );
  assert.equal(firstAvailable?.key, "gamma", "first variant should be used when default missing");
}

function testGetReRankWeights() {
  const cfg = {
    training: { policy: { weights: { base: 2, freshness: 1 } } },
  } as unknown as SiteConfig;

  const snapshotWeights = getReRankWeights(cfg, {
    rerank: { weights: { base: 5, popularity: 3 } },
  } as any);
  assert.equal(snapshotWeights.base, 5, "snapshot weights should override config");
  assert.equal(snapshotWeights.popularity, 3, "snapshot weights should include additional keys");

  const configWeights = getReRankWeights(cfg, null);
  assert.equal(configWeights.base, 2, "config weights should be used when snapshot missing");
}

(async function runPolicyCacheTests() {
  await testPolicySnapshotCaching();
  testPickPromptVariant();
  testGetReRankWeights();
  console.log("policy cache tests passed");
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
