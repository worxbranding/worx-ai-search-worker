#!/usr/bin/env node
// Signed request to a worx-search worker (default: local `npm run dev`).
// Usage: node scripts/call.mjs <METHOD> <path?query> [json-body]
//   WORX_URL (default http://127.0.0.1:8787), secret from WORX_HMAC_SECRET or .dev.vars
import { createHmac } from "node:crypto";
import { readFileSync, existsSync } from "node:fs";

const [method = "GET", target = "/status", body = ""] = process.argv.slice(2);
let secret = process.env.WORX_HMAC_SECRET;
if (!secret && existsSync(".dev.vars")) {
  secret = readFileSync(".dev.vars", "utf8").match(/^WORX_HMAC_SECRET=(.*)$/m)?.[1]?.trim();
}
if (!secret) { console.error("No WORX_HMAC_SECRET (env or .dev.vars)"); process.exit(2); }

const url = new URL(target, process.env.WORX_URL || "http://127.0.0.1:8787");
const ts = String(Date.now());
const canonical = `${method.toUpperCase()}\n${url.pathname}\n${url.search.replace(/^\?/, "")}\n${ts}\n${body}`;
const sig = createHmac("sha256", secret).update(canonical).digest("hex");

const res = await fetch(url, {
  method,
  headers: { "X-Worx-Timestamp": ts, "X-Worx-Signature": sig, ...(body ? { "content-type": "application/json" } : {}) },
  body: body || undefined,
});
console.log(res.status, await res.text());
process.exit(res.ok ? 0 : 1);
