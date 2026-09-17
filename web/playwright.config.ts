import { defineConfig, devices } from '@playwright/test'

const PORT = 5390

/**
 * The console on example data, in Chromium and in WebKit, because the phones and iPads it is laid out for run Safari.
 * Playwright's WebKit is not Safari: the on-screen keyboard, Safari's toolbars and its scrolling are for a real device.
 */
export default defineConfig({
  testDir: 'tests',
  fullyParallel: true,
  reporter: 'list',
  use: { baseURL: `http://127.0.0.1:${PORT}` },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
  ],
  webServer: {
    command: `npx vite --host 127.0.0.1 --port ${PORT} --strictPort`,
    url: `http://127.0.0.1:${PORT}`,
    env: { VITE_MOCK: '1' },
    reuseExistingServer: false,
  },
})
