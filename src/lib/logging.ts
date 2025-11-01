/**
 * Minimal logging helpers shared across the worker. They are intentionally
 * lightweight so modules can import them without pulling in additional
 * dependencies or breaking the Cloudflare runtime.
 */

/** Flip this flag when you need verbose diagnostics in production. */
export const LOGGING_ENABLED = false;

/**
 * Emit a timestamped console log only when logging is enabled. Using a helper
 * keeps noisy conditional checks out of the caller sites.
 */
export function log(...args: unknown[]): void {
  if (!LOGGING_ENABLED) return;
  const ts = new Date().toISOString();
  // eslint-disable-next-line no-console
  console.log(`[${ts}]`, ...args);
}

/**
 * Create a simple timer function that logs how long an operation took once the
 * returned callback executes. This makes request logging consistent.
 */
export function startTimer(label?: string): () => number {
  const t0 = Date.now();
  return () => {
    const ms = Date.now() - t0;
    if (LOGGING_ENABLED) {
      // eslint-disable-next-line no-console
      console.log(
        `[${new Date().toISOString()}]`,
        label ? `${label} took ${ms}ms` : `took ${ms}ms`
      );
    }
    return ms;
  };
}

/**
 * Wrap a synchronous or async operation and log the total runtime once it
 * completes. Callers use this around network requests and expensive helpers.
 */
export async function time<T>(label: string, fn: () => Promise<T> | T): Promise<T> {
  const stop = startTimer(label);
  try {
    return await fn();
  } finally {
    stop();
  }
}
