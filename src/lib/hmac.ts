/**
 * HMAC-SHA256 request verification for the WORX AI workers.
 *
 * Canonical signing string:
 *   METHOD\nPATH\nQUERY\nTIMESTAMP\nBODY
 *
 * - METHOD     uppercase HTTP method ("POST")
 * - PATH       URL pathname only ("/ask")
 * - QUERY      raw query string without leading "?" ("site=worxbranding-dev")
 * - TIMESTAMP  request milliseconds since epoch as decimal string
 * - BODY       exact request body bytes (empty string for GET)
 *
 * Wire format: hex-encoded HMAC-SHA256.
 * Headers: `X-Worx-Timestamp`, `X-Worx-Signature`.
 *
 * Verifier accepts a `current` secret and an optional `previous` secret so
 * rotation can run with overlap (stop-the-world rotation is operationally
 * brittle).
 */

const MAX_SKEW_MS = 5 * 60 * 1000; // 5 minutes
const HEX_RE = /^[0-9a-fA-F]+$/;

export interface HmacSecrets {
  current: string;
  previous?: string;
}

export type HmacVerifyResult =
  | { ok: true }
  | { ok: false; status: number; reason: string };

/** Verify the signature on an inbound worker request. */
export async function verifyHmac(
  req: Request,
  body: string,
  secrets: HmacSecrets,
  maxSkewMs: number = MAX_SKEW_MS
): Promise<HmacVerifyResult> {
  const ts = req.headers.get("X-Worx-Timestamp");
  const sig = req.headers.get("X-Worx-Signature");

  if (!ts || !sig) {
    return { ok: false, status: 401, reason: "missing X-Worx-Timestamp or X-Worx-Signature" };
  }

  if (!HEX_RE.test(sig) || sig.length !== 64) {
    return { ok: false, status: 401, reason: "signature must be 64-char hex" };
  }

  const tsNum = Number(ts);
  if (!Number.isFinite(tsNum) || tsNum <= 0) {
    return { ok: false, status: 401, reason: "invalid timestamp" };
  }

  const skew = Math.abs(Date.now() - tsNum);
  if (skew > maxSkewMs) {
    return { ok: false, status: 401, reason: `timestamp out of window (skew=${skew}ms)` };
  }

  if (!secrets.current) {
    return { ok: false, status: 500, reason: "WORX_HMAC_SECRET not configured" };
  }

  const url = new URL(req.url);
  const canonical = canonicalString(req.method, url.pathname, url.search, ts, body);

  if (await constantTimeMatch(secrets.current, canonical, sig)) {
    return { ok: true };
  }
  if (secrets.previous && await constantTimeMatch(secrets.previous, canonical, sig)) {
    return { ok: true };
  }

  return { ok: false, status: 401, reason: "signature mismatch" };
}

/** Build the canonical signing string. */
export function canonicalString(
  method: string,
  pathname: string,
  search: string,
  timestamp: string,
  body: string
): string {
  const query = search.startsWith("?") ? search.slice(1) : search;
  return `${method.toUpperCase()}\n${pathname}\n${query}\n${timestamp}\n${body}`;
}

/** Hex-encode HMAC-SHA256 of `message` with `secret`. */
export async function signHex(secret: string, message: string): Promise<string> {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(message));
  return bytesToHex(new Uint8Array(sig));
}

async function constantTimeMatch(secret: string, message: string, expectedHex: string): Promise<boolean> {
  const actualHex = await signHex(secret, message);
  if (actualHex.length !== expectedHex.length) return false;
  let diff = 0;
  for (let i = 0; i < actualHex.length; i++) {
    diff |= actualHex.charCodeAt(i) ^ expectedHex.charCodeAt(i);
  }
  return diff === 0;
}

function bytesToHex(bytes: Uint8Array): string {
  let out = "";
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i].toString(16);
    out += b.length === 1 ? "0" + b : b;
  }
  return out;
}
