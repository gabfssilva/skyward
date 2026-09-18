import { expect, test, type Page } from '@playwright/test'

const COMPUTE = '/computes/cmp_7f31ab'

const ROUTES = ['/', '/tasks', '/tasks/tk_9d21c4', '/activity', '/market', '/market/accounts', COMPUTE, `${COMPUTE}/nodes/0`] as const

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

/** Every sheet, and what opens it now that a destructive action lives in the page's menu. */
const SHEETS: readonly (readonly [string, string, (page: Page) => Promise<void>])[] = [
  ['wizard', '/', (page) => page.getByRole('button', { name: 'New compute' }).first().click()],
  ['run', COMPUTE, (page) => page.getByRole('button', { name: 'Run', exact: true }).first().click()],
  [
    'confirm',
    COMPUTE,
    async (page) => {
      await page.getByRole('button', { name: 'More actions' }).first().click()
      await page.getByRole('menuitem', { name: 'Delete compute' }).click()
    },
  ],
  [
    'write',
    '/tasks',
    async (page) => {
      /* the function list is a phone's page head instead of a column, so New function is in its menu there */
      const listed = page.getByRole('button', { name: 'New function' }).first()
      if (await listed.isVisible()) return listed.click()
      await page.getByRole('button', { name: 'More actions' }).first().click()
      await page.getByRole('menuitem', { name: 'New function' }).click()
    },
  ],
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

test.describe('every width', () => {
  test.use(DESKTOP)

  test('a page opens on its head: where it is, what it is called, and what can be done to it', async ({ page }) => {
    await open(page, COMPUTE)
    await expect(page.locator('#stage .head .title')).toHaveText(/llama-3-sft/)
    await expect(page.locator('#stage .head .facts')).toContainText('64 nodes')
    await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeVisible()

    await open(page, `${COMPUTE}/nodes/0`)
    await expect(page.locator('#stage .head .crumb')).toHaveText(/llama-3-sft/)
    await expect(page.locator('#stage .head .title')).toHaveText('rank 0')
  })

  test('the pages under a bar item carry no title of their own', async ({ page }) => {
    for (const path of ['/', '/tasks', '/activity', '/market']) {
      await open(page, path)
      await expect(page.locator('#stage .head .title'), path).toHaveCount(0)
    }
  })

  test("a compute's logs and events read in the order they happened", async ({ page }) => {
    await open(page, COMPUTE)
    const card = page.locator('#stage section.card').last()
    /* the example fleet prints a line or so a second, and the order is only a question once there are a few */
    await expect(card.locator('.logline').nth(2)).toBeVisible({ timeout: 15_000 })
    /* read rather than rendered: a window this long skips what is off screen, and skipped text comes back empty */
    const stamps = await card.locator('.logline > span:first-child').evaluateAll((all) => all.map((at) => at.textContent ?? ''))
    expect(stamps.length).toBeGreaterThan(1)
    expect([...stamps].sort()).toEqual(stamps)

    await page.getByRole('tab', { name: /^Events/ }).click()
    const said = await card.locator('.evline > span:first-child').evaluateAll((all) => all.map((at) => at.textContent ?? ''))
    expect(said.length).toBeGreaterThan(1)
    expect([...said].sort()).toEqual(said)
  })

  test('tasks and functions are one page: every task until a function is picked', async ({ page }) => {
    await open(page, '/tasks')
    await expect(page.locator('.fnlist .fn[aria-selected="true"]')).toHaveText(/All functions/)
    await expect(page.locator('table.tasklist thead th').first()).toHaveText('Function')

    await page.locator('.fnlist .fn', { hasText: 'evaluate' }).click()
    await expect(page.locator('table.tasklist thead th').first()).toHaveText('State')
    await expect(page.locator('.feed .facts')).toContainText('tasks')
  })

  test('the market holds the provider accounts, and an old link to them still lands', async ({ page }) => {
    await open(page, '/market')
    await page.getByRole('tab', { name: /^Accounts/ }).click()
    await expect(page).toHaveURL(/\/market\/accounts$/)
    await expect(page.getByRole('button', { name: 'Add account' })).toBeVisible()

    await page.goto('/providers')
    await expect(page).toHaveURL(/\/market\/accounts$/)
    await page.goto('/functions')
    await expect(page).toHaveURL(/\/tasks$/)
  })

  test('every metric the nodes measure is drawn against a scale the page owns', async ({ page }) => {
    await open(page, COMPUTE)
    const metrics = page.locator('#stage section.card', { has: page.getByText('Metrics') })
    await expect(metrics.locator('.mrow')).toHaveCount(3)
    await expect(metrics.locator('.chart')).toHaveCount(10)
    await expect(metrics.locator('.chart', { hasText: 'Memory' }).first()).toContainText('of 640 GB')
    await metrics.getByRole('button', { name: '15m' }).click()
    await expect(metrics.locator('.plot svg').first()).toBeVisible()
  })
})

test.describe('phone', () => {
  test.use(PHONE)

  test('the nav is a bar at the bottom with four items', async ({ page }) => {
    await open(page, '/market')
    const nav = await page.locator('#nav').boundingBox()
    expect(nav).not.toBeNull()
    expect(Math.round((nav?.y ?? 0) + (nav?.height ?? 0))).toBe(844)
    await expect(page.locator('#nav').getByRole('tab')).toHaveCount(4)
    await expect(page.locator('#nav').getByRole('tab', { name: 'Market' })).toHaveAttribute('aria-selected', 'true')
  })

  test('the nav steps aside while a field is being typed in', async ({ page }) => {
    await open(page, '/activity')
    await page.getByPlaceholder('filter lines').focus()
    await expect(page.locator('#nav')).toBeHidden()
    await page.getByPlaceholder('filter lines').blur()
    await expect(page.locator('#nav')).toBeVisible()
  })

  test('the task list is a card per row, the function list a select, and the market scrolls in its card', async ({ page }) => {
    await open(page, '/tasks')
    await expect(page.locator('table.tasklist thead')).toBeHidden()
    expect(await page.locator('table.tasklist tbody tr').first().evaluate((tr) => getComputedStyle(tr).display)).toBe('grid')
    await expect(page.locator('.fnlist')).toHaveCount(0)
    await expect(page.locator('.picker select')).toBeVisible()

    await open(page, '/market')
    const offers = page.getByRole('region', { name: 'Offers' })
    expect(await offers.evaluate((box) => box.scrollWidth > box.clientWidth)).toBe(true)
    expect(await offers.locator('td').first().evaluate((td) => getComputedStyle(td).position)).toBe('sticky')
  })

  test('a page keeps one primary action, with the rest in a menu', async ({ page }) => {
    await open(page, COMPUTE)
    await expect(page.locator('.head .acts .btn')).toHaveCount(2)
    await page.getByRole('button', { name: 'More actions' }).click()
    await expect(page.getByRole('menuitem', { name: 'Scale' })).toBeVisible()
    await expect(page.getByRole('menuitem', { name: 'Delete compute' })).toBeVisible()
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
    const target = await page.locator('.btn').first().evaluate((btn) => getComputedStyle(btn, '::after').height)
    expect(parseFloat(target)).toBeGreaterThanOrEqual(44)
    await open(page, '/activity')
    expect(await page.getByPlaceholder('filter lines').evaluate((input) => getComputedStyle(input).fontSize)).toBe('16px')
  })

  test('a tap shows a tooltip, and a tap elsewhere takes it away', async ({ page }) => {
    await open(page, `${COMPUTE}/nodes/0`)
    await page.locator('.slothive [data-tip]').first().tap()
    await expect(page.locator('#tip')).toBeVisible()
    await page.locator('.bar .brand').tap()
    await expect(page.locator('#tip')).toBeHidden()
  })
})

test.describe('ipad', () => {
  test.use(IPAD)

  test('the nav keeps its icons, named for a screen reader, and the page runs the width', async ({ page }) => {
    await open(page, '/tasks')
    await expect(page.locator('#nav span').first()).toBeHidden()
    await expect(page.getByRole('tab', { name: 'Activity' })).toBeVisible()
    const stage = await page.locator('#stage').boundingBox()
    expect(Math.round(stage?.width ?? 0)).toBe(820 - 32)
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

  test('the nav carries its labels', async ({ page }) => {
    await open(page, '/tasks')
    await expect(page.locator('#nav span').first()).toBeVisible()
  })

  test('on a wide screen the content keeps to its measure, centred', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 })
    await open(page, '/tasks')
    const stage = await page.locator('#stage').boundingBox()
    expect(Math.round(stage?.width ?? 0)).toBe(1280)
    expect(Math.round(stage?.x ?? 0)).toBe(Math.round(1920 - (stage?.x ?? 0) - (stage?.width ?? 0)))
  })

  test('a pointer resting on a hex shows its tooltip', async ({ page }) => {
    await open(page, '/')
    await page.locator('.hx[data-tip]').first().hover()
    await expect(page.locator('#tip')).toBeVisible()
    await expect(page.locator('#tip')).toHaveText(/rank/)
  })

  test('every compute on the home is drawn at the same cell size, so 284 nodes take more room than 64', async ({ page }) => {
    await open(page, '/')
    const boxes = await page.locator('.tile .comb svg').evaluateAll((all) => all.map((svg) => svg.getBoundingClientRect().width))
    expect(boxes.length).toBe(4)
    expect(boxes[0]).toBeLessThan(boxes[1])
    expect(boxes[3]).toBeLessThan(boxes[2])
  })
})
