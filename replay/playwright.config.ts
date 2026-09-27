import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 60_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["line"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: "http://localhost:8971",
    viewport: { width: 1440, height: 860 },
    deviceScaleFactor: 1,
  },
  webServer: {
    command: "node scripts/serve.mjs",
    port: 8971,
    reuseExistingServer: !process.env.CI,
    timeout: 15_000,
  },
  outputDir: "test-results",
});
