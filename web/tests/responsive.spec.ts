import { expect, test, type Page } from '@playwright/test'

const COMPUTE = '/computes/cmp_7f31ab'

const ROUTES = ['/', '/tasks', '/tasks/tk_9d21c4', '/functions', '/activity', '/market', '/providers', COMPUTE, `${COMPUTE}/nodes/0`] as const

/** A phone, an iPad standing and lying down, desktops, and a pixel either side of every width the layout turns at. */
const SCREENS = [
  { width: 320, height: 640, touch: true },
  { width: 390, height: 844, touch: true },
  { width: 600, height: 900, touch: true },
  { width: 601, height: 900, touch: true },
  { width: 820, height: 1180, touch: true },
  { width: 1080, height: 820, touch: true },
  { width: 1081, height: 820, touch: true },
  { width: 1180, height: 820, touch: true },
  { width: 1280, height: 900, touch: false },
  { width: 1281, height: 900, touch: false },
  { width: 1440, height: 900, touch: false },
  { width: 1920, height: 1080, touch: false },
] as const

const PHONE = { viewport: { width: 390, height: 844 }, hasTouch: true }
const IPAD = { viewport: { width: 820, height: 1180 }, hasTouch: true }
const IPAD_LANDSCAPE = { viewport: { width: 1180, height: 820 }, hasTouch: true }
const DESKTOP = { viewport: { width: 1440, height: 900 }, hasTouch: false }

const SHEETS: readonly (readonly [string, string, (page: Page) => Promise<void>])[] = [
  ['wizard', '/', (page) => page.getByRole('button', { name: 'New compute' }).first().click()],
  ['run', COMPUTE, (page) => page.getByRole('button', { name: 'Run', exact: true }).first().click()],
  ['confirm', COMPUTE, (page) => page.getByRole('button', { name: 'Delete', exact: true }).first().click()],
  ['write', '/functions', (page) => page.getByRole('button', { name: 'New function' }).first().click()],
  ['palette', '/', (page) => page.getByRole('button', { name: 'Search', exact: true }).click()],
]

async function open(page: Page, path: string): Promise<void> {
  await page.goto(path)
  await expect(page.locator('#stage .card').first()).toBeVisible()
  await page.waitForLoadState('networkidle')
  await page.evaluate(() => document.fonts.ready)
}

const sideways = (page: Page): Promise<number> => page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)

for (const screen of SCREENS) {
  test.describe(`${screen.width}×${screen.height}`, () => {
    test.use({ viewport: { width: screen.width, height: screen.height }, hasTouch: screen.touch })

    test('no page scrolls sideways', async ({ page }) => {
      for (const path of ROUTES) {
        await open(page, path)
        expect.soft(await sideways(page), path).toBe(0)
      }
    })

    test('no sheet scrolls sideways', async ({ page }) => {
      for (const [name, path, show] of SHEETS) {
        await open(page, path)
        await show(page)
        await expect(page.locator('dialog.scrim[open]')).toBeVisible()
        const dialog = page.locator('dialog.scrim[open]')
        expect.soft(await dialog.evaluate((d) => d.scrollWidth - d.clientWidth), name).toBe(0)
        expect.soft(await sideways(page), name).toBe(0)
      }
    })
  })
}

test.describe('phone', () => {
  test.use(PHONE)

  test('the nav is a bar at the bottom, without Providers, which the market links to', async ({ page }) => {
    await open(page, '/market')
    const nav = await page.locator('#nav').boundingBox()
    expect(nav).not.toBeNull()
    expect(Math.round((nav?.y ?? 0) + (nav?.height ?? 0))).toBe(844)
    await expect(page.getByRole('tab', { name: 'Tasks' })).toBeVisible()
    await expect(page.getByRole('tab', { name: 'Providers' })).toBeHidden()
    await page.getByRole('button', { name: 'Provider accounts' }).click()
    await expect(page).toHaveURL(/\/providers$/)
  })

  test('the nav steps aside while a field is being typed in', async ({ page }) => {
    await open(page, '/activity')
    await page.getByPlaceholder('filter lines').focus()
    await expect(page.locator('#nav')).toBeHidden()
    await page.getByPlaceholder('filter lines').blur()
    await expect(page.locator('#nav')).toBeVisible()
  })

  test('tasks and functions are cards, and the market scrolls inside its card', async ({ page }) => {
    await open(page, '/tasks')
    await expect(page.locator('table.tasklist thead')).toBeHidden()
    expect(await page.locator('table.tasklist tbody tr').first().evaluate((tr) => getComputedStyle(tr).display)).toBe('grid')

    await open(page, '/market')
    const offers = page.getByRole('region', { name: 'Offers' })
    expect(await offers.evaluate((box) => box.scrollWidth > box.clientWidth)).toBe(true)
    expect(await offers.locator('td').first().evaluate((td) => getComputedStyle(td).position)).toBe('sticky')
  })

  test('a form sheet takes the screen, and closes on Escape', async ({ page }) => {
    await open(page, '/')
    await page.getByRole('button', { name: 'New compute' }).first().click()
    const sheet = page.locator('dialog.scrim[open] .sheet')
    await expect(sheet).toBeVisible()
    expect(await sheet.boundingBox()).toEqual({ x: 0, y: 0, width: 390, height: 844 })
    await page.keyboard.press('Escape')
    await expect(page.locator('dialog.scrim')).toHaveCount(0)
  })

  test('a finger gets 44px targets and 16px fields', async ({ page }) => {
    await open(page, COMPUTE)
    const target = await page.locator('.btn.sm').first().evaluate((btn) => getComputedStyle(btn, '::after').height)
    expect(parseFloat(target)).toBeGreaterThanOrEqual(44)
    await open(page, '/activity')
    expect(await page.getByPlaceholder('filter lines').evaluate((input) => getComputedStyle(input).fontSize)).toBe('16px')
  })

  test('a tap shows a tooltip, and a tap elsewhere takes it away', async ({ page }) => {
    await open(page, `${COMPUTE}/nodes/0`)
    await page.locator('.mcard [data-tip]').last().tap()
    await expect(page.locator('#tip')).toBeVisible()
    await expect(page.locator('#tip')).toHaveText(/ago/)
    await page.locator('.mcard').first().getByText('GPU').tap()
    await expect(page.locator('#tip')).toBeHidden()
  })
})

test.describe('ipad', () => {
  test.use(IPAD)

  test('the nav keeps its icons, named for a screen reader, and the inspector drops below', async ({ page }) => {
    await open(page, '/tasks')
    await expect(page.getByRole('tab', { name: 'Providers' })).toBeVisible()
    await expect(page.locator('#nav span').first()).toBeHidden()
    const stage = await page.locator('#stage').boundingBox()
    const inspector = await page.locator('#inspector').boundingBox()
    expect((inspector?.y ?? 0) >= (stage?.y ?? 0) + (stage?.height ?? 0)).toBe(true)
  })

  test('a form sheet stays open on a click outside it, and the palette does not', async ({ page }) => {
    await open(page, '/')
    await page.getByRole('button', { name: 'New compute' }).first().click()
    await expect(page.locator('dialog.scrim[open]')).toBeVisible()
    await page.mouse.click(6, 6)
    await expect(page.locator('dialog.scrim[open]')).toBeVisible()
    await page.keyboard.press('Escape')

    await page.getByRole('button', { name: 'Search', exact: true }).click()
    await expect(page.locator('dialog.scrim[open] .palette')).toBeVisible()
    await page.mouse.click(6, 6)
    await expect(page.locator('dialog.scrim')).toHaveCount(0)
  })

  test('opening a form does not put the caret in a field', async ({ page }) => {
    await open(page, '/')
    await page.getByRole('button', { name: 'New compute' }).first().click()
    await expect(page.locator('dialog.scrim[open]')).toBeVisible()
    expect(await page.evaluate(() => document.activeElement?.tagName)).not.toMatch(/^(INPUT|TEXTAREA|SELECT)$/)
  })
})

test.describe('ipad lying down', () => {
  test.use(IPAD_LANDSCAPE)

  test('the daemon and the theme fold into a menu', async ({ page }) => {
    await open(page, '/tasks')
    await expect(page.locator('#daemon-tag')).toBeHidden()
    await page.getByRole('button', { name: 'Daemon and theme' }).click()
    await expect(page.locator('#daemon-tag')).toBeVisible()
    await page.locator('#theme').getByTitle('dark').click()
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
    await page.keyboard.press('Escape')
    await expect(page.locator('#daemon-tag')).toBeHidden()
  })
})

test.describe('desktop', () => {
  test.use(DESKTOP)

  test('the nav carries its labels, and the inspector sits beside the stage', async ({ page }) => {
    await open(page, '/tasks')
    await expect(page.locator('#nav span').first()).toBeVisible()
    const stage = await page.locator('#stage').boundingBox()
    const inspector = await page.locator('#inspector').boundingBox()
    expect((inspector?.x ?? 0) >= (stage?.x ?? 0) + (stage?.width ?? 0)).toBe(true)
  })

  test('on a wide screen the content keeps to its measure, centred', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 })
    await open(page, '/tasks')
    const stage = await page.locator('#stage').boundingBox()
    const inspector = await page.locator('#inspector').boundingBox()
    const left = stage?.x ?? 0
    const right = (inspector?.x ?? 0) + (inspector?.width ?? 0)
    expect(Math.round(right - left)).toBe(1280)
    expect(Math.round(left)).toBe(Math.round(1920 - right))
  })

  test('a pointer resting on a hex shows its tooltip', async ({ page }) => {
    await open(page, '/')
    await page.locator('.hx[data-tip]').first().hover()
    await expect(page.locator('#tip')).toBeVisible()
    await expect(page.locator('#tip')).toHaveText(/rank/)
  })
})
