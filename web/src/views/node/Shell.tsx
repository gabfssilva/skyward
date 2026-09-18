import { FitAddon } from '@xterm/addon-fit'
import { Terminal } from '@xterm/xterm'
import '@xterm/xterm/css/xterm.css'
import { useEffect, useRef, useState } from 'react'
import { BASE, type Compute, type Node, type WireError } from '../../api/client'
import { MOCK } from '../../api/mock'
import { clamp, holderOf, nodeHeld, readyOf } from '../../state/model'
import { useStore } from '../../state/store'

/** How much memory each accelerator carries, the way the prototype's catalog reads. */
export const VRAM: Record<string, number> = { b200: 192, h200: 141, h100: 80, a100: 80, l40s: 48, a10g: 24, l4: 24, mi300x: 192 }
const SMI_NAME: Record<string, string> = {
  h100: 'NVIDIA H100 80GB',
  a100: 'NVIDIA A100-SXM4',
  l40s: 'NVIDIA L40S',
  a10g: 'NVIDIA A10G',
  l4: 'NVIDIA L4',
  b200: 'NVIDIA B200',
}

const nameOf = (c: Compute): string => c.name ?? c.id
const promptOf = (c: Compute, rank: number): string => `root@${nameOf(c)}-${rank}:~# `

/** A CSS custom property's value, so the terminal is painted in the theme the page is. */
const token = (name: string): string => getComputedStyle(document.documentElement).getPropertyValue(name).trim()

/**
 * Where the daemon opens a terminal on a machine.
 *
 * A socket rather than the two half-duplex streams the CLI uses, because a browser
 * cannot write a request body it is still reading the answer to: a streaming body
 * needs HTTP/2, and the daemon serves 1.1. The size travels in the query so the pty
 * opens at the shape it will be drawn at, and again in a frame whenever that changes.
 */
const attach = (computeId: string, rank: number, columns: number, rows: number): string => {
  const query = new URLSearchParams({ node: String(rank), columns: String(columns), rows: String(rows), term: 'xterm-256color' })
  return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}${BASE}/computes/${computeId}/shell/attach?${query}`
}

/** The refusal the daemon sends as a frame before it closes, in the shape every endpoint answers with. */
const reasonOf = (frame: string): string => {
  try {
    const wire = JSON.parse(frame) as Partial<WireError>
    return wire.message ?? frame
  } catch {
    return frame
  }
}

type Context = { compute: Compute; node: Node; rank: number; count: number; peers: number; concurrency: number; head: string; readings: Record<string, number> }

/**
 * The machine's own terminal, carried over one socket.
 *
 * Nothing is interpreted on this side: what arrives is what the pty painted, escape
 * codes and all, which is why it is drawn by a terminal emulator and not by a list of
 * lines. The session is over when the socket closes — the shell exiting, the machine
 * going away, or this card unmounting.
 */
function live(term: Terminal, computeId: string, rank: number, say: (said: string | null) => void): () => void {
  const socket = new WebSocket(attach(computeId, rank, term.cols, term.rows))
  socket.binaryType = 'arraybuffer'
  let refused = false

  const shape = (): void => {
    if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ columns: term.cols, rows: term.rows }))
  }

  const typed = term.onData((data) => {
    if (socket.readyState === WebSocket.OPEN) socket.send(new TextEncoder().encode(data))
  })
  const moved = term.onResize(shape)

  socket.onopen = () => {
    say(null)
    shape()
    term.focus()
  }
  socket.onmessage = (event: MessageEvent<string | ArrayBuffer>) => {
    if (typeof event.data !== 'string') return term.write(new Uint8Array(event.data))
    refused = true
    say(reasonOf(event.data))
  }
  socket.onclose = (event) => {
    if (!refused && event.code !== 1000) say(`the session ended (${event.code})`)
  }

  return () => {
    typed.dispose()
    moved.dispose()
    socket.onclose = null
    socket.close()
  }
}

/**
 * The prototype's machine, answering into the same terminal a real one would.
 *
 * Only under ``VITE_MOCK``, where there is no daemon and so no socket: the example
 * data has always included a shell, and a terminal that only ever says it cannot
 * connect would be a worse prototype than one that answers.
 */
function fake(term: Terminal, context: () => Context): () => void {
  const prompt = promptOf(context().compute, context().rank)
  term.writeln(`pty opened on rank ${context().rank} — xterm-256color, example data`)
  term.writeln('nvidia-smi · ls · uv pip list · env | grep SKYWARD · help')
  term.write(prompt)

  let line = ''
  const typed = term.onData((data) => {
    for (const key of data) {
      if (key === '\r') {
        term.write('\r\n')
        const said = answer(context(), line)
        if (said) term.writeln(said.replaceAll('\n', '\r\n'))
        line = ''
        term.write(prompt)
      } else if (key === '\x7f') {
        if (line) {
          line = line.slice(0, -1)
          term.write('\b \b')
        }
      } else if (key >= ' ') {
        line += key
        term.write(key)
      }
    }
  })

  term.focus()
  return () => typed.dispose()
}

function smi(n: Node, count: number, m: Record<string, number>): string {
  const accel = n.accelerator ?? '?'
  const name = SMI_NAME[accel] ?? accel.toUpperCase()
  const total = (VRAM[accel] ?? 24) * 1024
  const util = m['gpu_util'] ?? 0
  const temp = m['gpu_temp_c'] ?? 42
  const memory = m['gpu_mem_mb'] ?? 0
  const out = [
    '+-----------------------------------------------------------------------------+',
    '| NVIDIA-SMI 560.35.03      Driver Version: 560.35.03      CUDA Version: 12.8  |',
    '|-------------------------------+----------------------+----------------------+',
    '| GPU  Name            Persist-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |',
    '| Fan  Temp  Perf  Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |',
    '|===============================+======================+======================|',
  ]
  for (let i = 0; i < count; i++) {
    const u = Math.round(clamp(util + (Math.random() - 0.5) * 8, 0, 100))
    const used = Math.round(memory / count) || Math.round(total * 0.6)
    out.push(`|   ${i}  ${name.padEnd(19).slice(0, 19)} On | 00000000:${(0x53 + i * 8).toString(16).toUpperCase()}:00.0 Off |                    0 |`)
    out.push(
      `| N/A  ${Math.round(temp)}C   P0     ${(380 + u * 3).toFixed(0)}W / 700W |  ${String(used).padStart(6)}MiB / ${total}MiB |     ${String(u).padStart(3)}%      Default |`,
    )
    out.push('+-------------------------------+----------------------+----------------------+')
  }
  return out.join('\n')
}

/** What the prototype's machine says to one command line. */
function answer(ctx: Context, raw: string): string {
  const { compute: c, node: n, rank } = ctx
  const cmd = raw.trim()
  const head = cmd.split(/\s+/)[0] ?? ''
  if (!cmd) return ''
  if (cmd === 'clear') return '\x1b[2J\x1b[H'
  if (head === 'help')
    return 'nvidia-smi · ls · pwd · whoami · uname -a · df -h · free -g · uv pip list · python -c "…" · env | grep SKYWARD · cat train.py · clear'
  if (head === 'nvidia-smi') return smi(n, ctx.count, ctx.readings)
  if (head === 'ls') return 'checkpoints/  data/  skyward/  train.py  pyproject.toml  events.jsonl'
  if (head === 'pwd') return '/root'
  if (head === 'whoami') return 'root'
  if (head === 'uname') return `Linux ${n.machine ?? n.id} 6.8.0-51-generic #52-Ubuntu SMP x86_64 GNU/Linux`
  if (head === 'df') return 'Filesystem      Size  Used Avail Use% Mounted on\n/dev/root       1.8T  412G  1.4T  23% /\ntmpfs           1.0T     0  1.0T   0% /dev/shm'
  if (head === 'free') return '              total   used   free  shared  buff/cache  available\nMem:           2015    998    612      12         405         1002'
  if (/^(uv )?pip list/.test(cmd))
    return 'Package              Version\n-------------------- --------\ncasty                0.22.1\ncloudpickle          3.1.2\nlz4                  4.4.4\nmsgspec              0.19.0\nskyward              0.9.3\ntorch                2.8.0\ntransformers         4.57.1'
  if (cmd.startsWith('env'))
    return [
      `SKYWARD_COMPUTE=${c.id}`,
      `SKYWARD_NODE=${n.id}`,
      `SKYWARD_RANK=${rank}`,
      `SKYWARD_NODES=${ctx.peers}`,
      `SKYWARD_WORKERS_PER_NODE=${ctx.concurrency}`,
      `SKYWARD_HEAD_ADDR=${ctx.head}`,
      'SKYWARD_HEAD_PORT=29500',
    ].join('\n')
  if (head === 'python' || head === 'python3') {
    if (cmd.includes('device_count')) return String(ctx.count)
    if (cmd.includes('instance_info'))
      return `Info(node='${n.id}', compute='${c.id}', rank=${rank}, peers=${ctx.peers}, worker=0, workers_per_node=${ctx.concurrency})`
    return '/root/.skyward/venv/bin/python — Python 3.12.7'
  }
  if (head === 'cat' && cmd.includes('train.py'))
    return 'import skyward as sky\n\n@sky.function\ndef train_step(batch):\n    info = sky.instance_info()\n    return model.step(sky.shard(batch)[info.rank])'
  if (head === 'sky') return 'bash: sky: command not found — a node runs the worker, not the CLI'
  return `bash: ${head}: command not found`
}

/** The terminal itself: an emulator sized to its box, and whatever is feeding it. */
function Screen({ ctx }: { ctx: Context }) {
  const mount = useRef<HTMLDivElement>(null)
  const [said, setSaid] = useState<string | null>('opening a terminal…')
  const latest = useRef(ctx)
  latest.current = ctx
  const { id: computeId } = ctx.compute
  const { rank } = ctx

  useEffect(() => {
    const host = mount.current
    if (!host) return

    const term = new Terminal({
      fontFamily: token('--mono') || 'monospace',
      fontSize: 12,
      cursorBlink: true,
      theme: { background: token('--term-bg'), foreground: token('--term-ink'), cursor: token('--term-p') },
    })
    const fit = new FitAddon()
    term.loadAddon(fit)
    term.open(host)

    let stop = (): void => {}
    let gone = false
    let queued = 0

    /*
     * Refitting is deferred to the next frame, and the box it watches is bounded from
     * the outside. A terminal is a fixed grid of cells, so it has a width of its own
     * to contribute upward; measuring it inside the box it just widened is how a
     * resize becomes a loop that never settles.
     */
    const resized = new ResizeObserver(() => {
      cancelAnimationFrame(queued)
      queued = requestAnimationFrame(() => fit.fit())
    })

    /* The font is a webfont, and a terminal measured before it lands is measured against the fallback. */
    void document.fonts.ready.then(() => {
      if (gone) return
      fit.fit()
      resized.observe(host)
      stop = MOCK ? fake(term, () => latest.current) : live(term, computeId, rank, setSaid)
      if (MOCK) setSaid(null)
    })

    return () => {
      gone = true
      cancelAnimationFrame(queued)
      resized.disconnect()
      stop()
      term.dispose()
    }
  }, [computeId, rank])

  return (
    <div className="term">
      <div className="term-screen" ref={mount} />
      {said ? <div className="term-said">{said}</div> : null}
    </div>
  )
}

/**
 * Why there is no terminal, for a machine the daemon holds no link to.
 *
 * Worth telling apart: one that has not been bought yet is a wait, and one the daemon
 * has let go of is not — the same distinction the daemon draws between a channel that
 * is not up and a channel that is finished.
 */
const why = (n: Node): string =>
  ['requested', 'provisioning'].includes(n.state)
    ? 'This machine is still being bought. A terminal opens the moment it answers SSH, which is well before its bootstrap finishes.'
    : 'The daemon has no link to this machine. Every machine that answers SSH takes a terminal, and this one is not answering.'

/** A pty on one rank, in the node page's Shell tab. */
export function Shell({ computeId, rank }: { computeId: string; rank: number }) {
  const compute = useStore((s) => s.computes.find((c) => c.id === computeId))
  const nodes = useStore((s) => s.nodes[computeId])
  const readings = useStore((s) => s.readings)

  const node = holderOf(nodes ?? [], rank)
  if (!compute || !node || !nodeHeld(node)) return <div className="sub">{!compute || !node ? 'Only a live compute takes a shell.' : why(node)}</div>

  const all = nodes ?? []
  const spec = compute.spec.specs[0]

  return (
    <Screen
      ctx={{
        compute,
        node,
        rank,
        count: spec?.accelerator_count ?? 1,
        peers: readyOf(all).length,
        concurrency: compute.spec.worker?.concurrency ?? 1,
        head: holderOf(all, 0)?.address ?? '—',
        readings: readings[`${computeId}/${rank}`] ?? {},
      }}
    />
  )
}
