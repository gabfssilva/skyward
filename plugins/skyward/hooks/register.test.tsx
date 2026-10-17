import type { HttpResponse } from 'claude-code'
import { expect, mock, test } from 'claude-code/testing'

const PANE = { title: 'Skyward', isFocused: false, bodyColumns: 120, placement: 'dock', scroll: { offset: 0, bodyRows: 80 }, view: {} } as const
const AT = '2026-10-17T12:00:00Z'
const COMPUTE = { id: 'cmp_1', name: 'train', status: { state: 'ready' }, created_at: AT, nodes: [{ id: 'node_1', rank: 0, state: 'ready' }] }

// What tqdm and a colored logger print, as the worker records it: a bar redrawn after a carriage return, the cursor moved
// up over the bars above it, and a line in color.
const PRINTED = ['\rFetching 9 files: 100%|##########| 9/9', '\x1b[A\x1b[A', '\x1b[31mloss\x1b[0m 0.5', 'epoch 1']
// The panel reads `Date.now()` and keeps the failures of the last day, so this one happened as the test runs.
const failed = () => ({ sequence: 1, type: 'task.failed', at: new Date().toISOString(), data: { compute: 'cmp_1', error: '\x1b[1mValueError\x1b[0m: bad shape' } })

const ok = (body: unknown): HttpResponse => ({ status: 200, ok: true, headers: {}, text: JSON.stringify(body) })

const daemon = (url: string): HttpResponse => {
  const { pathname, searchParams } = new URL(url)
  if (pathname === '/v1/computes') return ok({ items: searchParams.get('live') === 'true' ? [COMPUTE] : [] })
  if (pathname === '/v1/computes/cmp_1') return ok(COMPUTE)
  if (pathname === '/v1/computes/cmp_1/metrics') return ok({ series: [] })
  if (pathname === '/v1/tasks') return ok({ items: [], total: 0 })
  if (pathname === '/v1/events/log' && searchParams.has('types')) return ok({ items: [failed()] })
  if (pathname === '/v1/events/log') return ok({ items: PRINTED.map((content) => ({ type: 'node.console', at: AT, data: { node: 'node_1', content } })) })
  return { status: 404, ok: false, headers: {}, text: '{"message":"not found"}' }
}

for (const surface of ['terminal', 'desktop'] as const) {
  test(`a log holding escape sequences is drawn on the ${surface}`, async ($, on) => {
    mock.store(on)
    const clock = mock.clock(on)
    on('http.fetch', (_$, e) => ({ value: daemon(e.url) }))
    on('ui.open', () => ({ value: { isPlaced: true } }))
    await $.command.run({ command: 'sky', args: '', origin: { kind: 'composer' }, presentation: { isFullscreen: true, columns: 200 } })
    await clock.settle()

    const ui = await $.ui.mount({ plugin: 'skyward', surface, component: 'Pane', props: PANE, requestId: 'skyward' })
    await ui.press({ key: 'cmp_1' })
    await clock.settle()

    await expect(ui.drawn()).resolves.toMatchObject({ type: 'Box' })
    // One clock per log row: the line that only moved the cursor is not one.
    expect(await ui.findAll({ type: 'Text', text: /^\d{2}:\d{2}:\d{2}$/ })).toHaveLength(3)
    expect(await ui.find({ type: 'Text', text: 'Fetching 9 files: 100%|##########| 9/9' })).toBeDefined()
    expect(await ui.find({ type: 'Text', text: 'loss 0.5' })).toBeDefined()
    expect(await ui.find({ type: 'Text', text: 'epoch 1' })).toBeDefined()
    expect(await ui.find({ type: 'Text', text: 'ValueError: bad shape' })).toBeDefined()
  })
}
