/**
 * Minimal Node.js ambient declarations for the non-browser edge of this
 * package: `playwright.config.ts` reads `process.env.CI`, and the vitest
 * unit tests load committed fixtures via `node:fs` / `node:path` /
 * `import.meta.dirname`.
 *
 * The browser bundle never uses these. This package deliberately avoids a
 * `@types/node` dependency (browser-first code, lean lockfile); if it is
 * ever added, DELETE this file — the real declarations supersede these.
 */

declare var process: {
  env: Record<string, string | undefined>;
};

declare module "node:fs" {
  export function readFileSync(path: string, encoding: "utf8"): string;
  export function readdirSync(path: string): string[];
}

declare module "node:path" {
  export function join(...parts: string[]): string;
}

interface ImportMeta {
  /** Directory pathname of the current module (Node >= 20.11, vitest). */
  readonly dirname: string;
}
