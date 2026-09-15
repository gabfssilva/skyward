import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../skyward/server/http/console',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/v1': { target: 'http://127.0.0.1:17590', changeOrigin: true },
    },
  },
})
