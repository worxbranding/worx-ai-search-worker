// Small crypto helpers. Workers exposes WebCrypto, so we keep a thin wrapper to
// reuse SHA-1 hex generation wherever cache keys are constructed.

/** Produce a SHA-1 hex digest using the Workers runtime WebCrypto API. */
export async function sha1Hex(value: string): Promise<string> {
  const data = new TextEncoder().encode(value);
  const digest = await crypto.subtle.digest("SHA-1", data);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}
