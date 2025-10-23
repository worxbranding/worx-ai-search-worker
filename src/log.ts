// Centralized logging control for the worker
// Toggle this flag to enable/disable logs across the codebase.
// Set to false to silence logs in production.
export const LOGGING_ENABLED = false;

// Prefix logs with ISO timestamp for easier tracing
export function log(...args: any[]) {
  if (!LOGGING_ENABLED) return;
  const ts = new Date().toISOString();
  // eslint-disable-next-line no-console
  console.log(`[${ts}]`, ...args);
}

// Simple timer utility
export function startTimer(label?: string) {
  const t0 = Date.now();
  return () => {
    const ms = Date.now() - t0;
    if (LOGGING_ENABLED) {
      // eslint-disable-next-line no-console
      console.log(`[${new Date().toISOString()}]`, label ? `${label} took ${ms}ms` : `took ${ms}ms`);
    }
    return ms;
  };
}

// Wrap a promise-returning function and log duration
export async function time<T>(label: string, fn: () => Promise<T> | T): Promise<T> {
  const stop = startTimer(label);
  try {
    return await fn();
  } finally {
    stop();
  }
}
