import { expect, test, type Page } from '@playwright/test'

/**
 * Resizing a compute from the console.
 *
 * The compute is the mock's ``embed-fleet``: it opened at 284 and is held between 240 and 320. What a resize moves
 * is that range — ``initial`` is the size it opened at, decided once at creation, and a form that writes it is a
 * form that can reopen a pool that is already standing. What is asserted here is that the numbers on the screen are
 * the numbers that travel: a bound the form keeps behind your back is what turned "scale to 50" into a pool asked
 * for fifty under a ceiling of twenty, bought and then counted as surplus.
 */
const ELASTIC = '/computes/cmp_9ac410'

test.use({ viewport: { width: 1440, height: 900 }, hasTouch: false })

async function scale(page: Page): Promise<void> {
  await page.goto(ELASTIC)
  await expect(page.locator('#stage .card').first()).toBeVisible()
  /* a wide page keeps the first two actions beside the primary one; a narrow one folds them into the menu */
  const beside = page.locator('.head .acts').getByRole('button', { name: 'Scale', exact: true })
  if (await beside.isVisible()) await beside.click()
  else {
    await page.getByRole('button', { name: 'More actions' }).first().click()
    await page.getByRole('menuitem', { name: 'Scale' }).click()
  }
  await expect(page.locator('dialog.scrim[open] .sheet')).toBeVisible()
}

test.describe('the scale sheet', () => {
  test('opens on the range the compute is held to, and offers no way to reopen it', async ({ page }) => {
    await scale(page)

    await expect(page.locator('#sc-nodes')).toHaveValue('240')
    await expect(page.locator('#sc-up')).toHaveValue('320')
    await expect(page.locator('.sheet input')).toHaveCount(2)
    await expect(page.locator('.sheet')).toContainText('It opened at 284, which only its creation decides.')
  })

  test('refuses a ceiling under the size asked for', async ({ page }) => {
    await scale(page)
    await page.locator('#sc-nodes').fill('400')

    await expect(page.getByText('A ceiling of 320 is below the 400 asked for')).toBeVisible()
    await expect(page.getByRole('button', { name: /^Scale to/ })).toBeDisabled()
  })

  test('a size with no ceiling is the size the pool is held at', async ({ page }) => {
    await scale(page)
    await page.locator('#sc-nodes').fill('400')
    await page.locator('#sc-up').fill('')
    await page.getByRole('button', { name: 'Scale to 400 nodes' }).click()

    await expect(page.locator('dialog.scrim')).toHaveCount(0)
    await expect(page.locator('.head')).toContainText('400 nodes')
    await expect(page.locator('.head')).not.toContainText('elastic')
  })

  test('a range is kept as a range', async ({ page }) => {
    await scale(page)
    await page.locator('#sc-nodes').fill('300')
    await page.locator('#sc-up').fill('400')
    await page.getByRole('button', { name: 'Scale to 300–400 nodes' }).click()

    await expect(page.locator('dialog.scrim')).toHaveCount(0)
    await expect(page.locator('.head')).toContainText('400 nodes, elastic 300 to 400')
  })
})
