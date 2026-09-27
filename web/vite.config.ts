import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// `base: "./"` keeps asset and fixture URLs relative so the built `dist/`
// works from any static mount point (GitHub Pages subpath, file share, ...).
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    outDir: "dist",
    sourcemap: false,
  },
  test: {
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
    environment: "node",
  },
});
