import { strict as assert } from "node:assert";

import { logTrainingSession } from "../src/training/sessionLogger";
import type { Env, SiteConfig } from "../src/lib/types";

const originalFetch = globalThis.fetch;

async function testSessionLoggerDispatches() {
  const cached: Array<{ key: string; value: string }> = [];
  const env = {
    TRAINING_CACHE: {
      async put(key: string, value: string) {
        cached.push({ key, value });
      },
    },
  } as unknown as Env;

  const cfg = {
    site_key: "acme",
    vectorize: { index_name: "idx", dims: 1 },
    ai: { embed_model: "model" },
    training: {
      apiKey: "secret",
      logger: {
        endpoint: "https://api.internal/training/session-log",
        cacheKeyPrefix: "train:",
        ttlSeconds: 10,
        headers: { "x-extra": "1" },
        timeoutMs: 2000,
      },
    },
  } as unknown as SiteConfig;

  let fetchCalls = 0;
  globalThis.fetch = (async (input: RequestInfo, init?: RequestInit) => {
    fetchCalls += 1;
    assert.equal(input, "https://api.internal/training/session-log", "logger should call configured endpoint");
    const headers = init?.headers as Record<string, string>;
    assert.equal(headers["content-type"], "application/json", "logger should send JSON");
    assert.equal(headers["x-api-key"], "secret", "logger should attach training API key");
    assert.equal(headers["x-extra"], "1", "logger should merge custom headers");
    const body = JSON.parse(String(init?.body || "{}"));
    assert.equal(body.site, "acme", "payload should include site key");
    return {
      ok: true,
      status: 200,
      text: async () => "",
    } as Response;
  }) as typeof fetch;

  await logTrainingSession(
    env,
    cfg,
    {
      site: "acme",
      question: "Q1",
      answer: "A1",
      stats: {},
      matches: [],
      prompt: { system: "sys", temperature: 0.4, model: "llm" },
    },
    undefined
  );

  assert.equal(fetchCalls, 1, "logger should issue one fetch call");
  assert.equal(cached.length, 1, "payload should be cached when prefix configured");
}

async function testSessionLoggerNoEndpoint() {
  const cached: Array<{ key: string; value: string }> = [];
  const env = {
    CONFIG: {
      async put(key: string, value: string) {
        cached.push({ key, value });
      },
    },
  } as unknown as Env;

  const cfg = {
    site_key: "acme",
    vectorize: { index_name: "idx", dims: 1 },
    ai: { embed_model: "model" },
    training: {
      logger: {
        cacheKeyPrefix: "train:",
        ttlSeconds: 5,
      },
    },
  } as unknown as SiteConfig;

  let fetchCalled = false;
  globalThis.fetch = (async () => {
    fetchCalled = true;
    return { ok: true, status: 200, text: async () => "" } as Response;
  }) as typeof fetch;

  await logTrainingSession(
    env,
    cfg,
    {
      site: "acme",
      question: "Q2",
      answer: "A2",
      stats: {},
      matches: [],
      prompt: { system: "sys", temperature: 0.4, model: "llm" },
    },
    undefined
  );

  assert.equal(fetchCalled, false, "logger should skip fetch when endpoint missing");
  assert.equal(cached.length, 1, "payload should still be cached locally");
}

(async function runSessionLoggerTests() {
  try {
    await testSessionLoggerDispatches();
    await testSessionLoggerNoEndpoint();
    console.log("session logger tests passed");
  } finally {
    globalThis.fetch = originalFetch;
  }
})().catch((error) => {
  console.error(error);
  globalThis.fetch = originalFetch;
  process.exit(1);
});
