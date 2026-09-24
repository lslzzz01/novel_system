import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { resolve } from "node:path";

export default defineConfig({
  plugins: [vue()],
  root: resolve(process.cwd(), "frontend"),
  publicDir: false,
  build: {
    outDir: resolve(process.cwd(), "novel_agents/web_assets"),
    emptyOutDir: true,
    assetsDir: ".",
    rollupOptions: {
      output: {
        entryFileNames: "app.js",
        chunkFileNames: "chunk-[name].js",
        assetFileNames: (assetInfo) => {
          const extension = assetInfo.name?.split(".").pop() || "bin";
          return extension === "css" ? "app.css" : "asset-[name].[ext]";
        },
      },
    },
  },
});
