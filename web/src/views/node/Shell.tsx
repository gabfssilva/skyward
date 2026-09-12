import { useEffect, useRef, useState } from 'react'
import { useStore } from '../../state/store'
import type { Compute, Node } from '../../api/client'
import type { NodeMetrics } from '../../state/model'
import { clamp, readyOf } from '../../state/model'
import { Icon } from '../../ui/icons'

type TermLine = { c: string; t: string }
type ShellState = { lines: TermLine[]; history: string[] }

const shells: Record<string, ShellState> = {}

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
const promptOf = (c: Compute, rank: number): string => `root@${nameOf(c)}-${rank}:~#`

function shellFor(computeId: string, rank: number): ShellState {
  const k = `${computeId}/${rank}`
  const found = shells[k]
  if (found) return found
  const fresh: ShellState = {
    lines: [
      { c: 'd', t: `pty opened on rank ${rank} — xterm-256color, keystrokes over POST /shell/up` },
      { c: 'd', t: 'mock shell: nvidia-smi, ls, uv pip list, env | grep SKYWARD, help' },
    ],
    history: [],
  }
  shells[k] = fresh
  return fresh
}

function smi(n: Node, count: number, m: NodeMetrics): string {
  const accel = n.accelerator ?? '?'
  const name = SMI_NAME[accel] ?? accel.toUpperCase()
  const total = (VRAM[accel] ?? 24) * 1024
  const out = [
    '+-----------------------------------------------------------------------------+',
    '| NVIDIA-SMI 560.35.03      Driver Version: 560.35.03      CUDA Version: 12.8  |',
    '|-------------------------------+----------------------+----------------------+',
    '| GPU  Name            Persist-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |',
    '| Fan  Temp  Perf  Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |',
    '|===============================+======================+======================|',
  ]
  for (let i = 0; i < count; i++) {
    const u = Math.round(clamp(m.gpu + (Math.random() - 0.5) * 8, 0, 100))
    const used = Math.round((total * m.vram) / 100)
    out.push(`|   ${i}  ${name.padEnd(19).slice(0, 19)} On | 00000000:${(0x53 + i * 8).toString(16).toUpperCase()}:00.0 Off |                    0 |`)
    out.push(
      `| N/A  ${Math.round(m.temp)}C   P0     ${(380 + u * 3).toFixed(0)}W / 700W |  ${String(used).padStart(6)}MiB / ${total}MiB |     ${String(u).padStart(3)}%      Default |`,
    )
    out.push('+-------------------------------+----------------------+----------------------+')
  }
  return out.join('\n')
}

type Context = { compute: Compute; node: Node; rank: number; count: number; peers: number; concurrency: number; head: string; metrics: NodeMetrics }

function runCommand(ctx: Context, raw: string): void {
  const { compute: c, node: n, rank } = ctx
  const sh = shellFor(c.id, rank)
  const cmd = raw.trim()
  sh.lines.push({ c: 'p', t: `${promptOf(c, rank)} ${cmd}` })
  const say = (t: string, cls?: string): number => sh.lines.push({ c: cls ?? '', t })
  const head = cmd.split(/\s+/)[0] ?? ''
  if (!cmd) {
    /* an empty line only prints the prompt */
  } else if (cmd === 'clear') sh.lines = []
  else if (head === 'help')
    say('nvidia-smi · ls · pwd · whoami · uname -a · df -h · free -g · uv pip list · python -c "…" · env | grep SKYWARD · cat train.py · clear', 'd')
  else if (head === 'nvidia-smi') say(smi(n, ctx.count, ctx.metrics))
  else if (head === 'ls') say('checkpoints/  data/  skyward/  train.py  pyproject.toml  events.jsonl')
  else if (head === 'pwd') say('/root')
  else if (head === 'whoami') say('root')
  else if (head === 'uname') say(`Linux ${n.machine ?? n.id} 6.8.0-51-generic #52-Ubuntu SMP x86_64 GNU/Linux`)
  else if (head === 'df') say('Filesystem      Size  Used Avail Use% Mounted on\n/dev/root       1.8T  412G  1.4T  23% /\ntmpfs           1.0T     0  1.0T   0% /dev/shm')
  else if (head === 'free') say('              total   used   free  shared  buff/cache  available\nMem:           2015    998    612      12         405         1002')
  else if (/^(uv )?pip list/.test(cmd))
    say(
      'Package              Version\n-------------------- --------\ncasty                0.22.1\ncloudpickle          3.1.2\nlz4                  4.4.4\nmsgspec              0.19.0\nskyward              0.9.3\ntorch                2.8.0\ntransformers         4.57.1',
    )
  else if (cmd.startsWith('env'))
    say(
      [
        `SKYWARD_COMPUTE=${c.id}`,
        `SKYWARD_NODE=${n.id}`,
        `SKYWARD_RANK=${rank}`,
        `SKYWARD_NODES=${ctx.peers}`,
        `SKYWARD_WORKERS_PER_NODE=${ctx.concurrency}`,
        `SKYWARD_HEAD_ADDR=${ctx.head}`,
        'SKYWARD_HEAD_PORT=29500',
      ].join('\n'),
    )
  else if (head === 'python' || head === 'python3') {
    if (cmd.includes('device_count')) say(String(ctx.count))
    else if (cmd.includes('instance_info'))
      say(`Info(node='${n.id}', compute='${c.id}', rank=${rank}, peers=${ctx.peers}, worker=0, workers_per_node=${ctx.concurrency})`)
    else say('/root/.skyward/venv/bin/python — Python 3.12.7')
  } else if (head === 'cat' && cmd.includes('train.py'))
    say('import skyward as sky\n\n@sky.function\ndef train_step(batch):\n    info = sky.instance_info()\n    return model.step(sky.shard(batch)[info.rank])')
  else if (head === 'sky') say('bash: sky: command not found — a node runs the worker, not the CLI', 'e')
  else say(`bash: ${head}: command not found`, 'e')
  if (cmd) sh.history.push(cmd)
}

/** The prototype's ``shellCard``: a pty on one rank, in a card of its own on the node page. */
export function ShellCard({ computeId, rank, onClose }: { computeId: string; rank: number; onClose: () => void }) {
  const compute = useStore((s) => s.computes.find((c) => c.id === computeId))
  const nodes = useStore((s) => s.nodes[computeId])
  const metrics = useStore((s) => s.metrics)
  const [, bump] = useState(0)
  const out = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (out.current) out.current.scrollTop = out.current.scrollHeight
  })

  useEffect(() => {
    input.current?.focus()
  }, [computeId, rank])

  const node = (nodes ?? []).find((n) => n.rank === rank)
  const head = (
    <div className="row" style={{ justifyContent: 'space-between', marginBottom: 8 }}>
      <span className="cap">Shell on rank {rank}</span>
      <span className="mono faint">{node?.address ?? '—'} · pty over the daemon</span>
      <button className="btn sm ghost" onClick={onClose}>
        <Icon name="close" />
        Close
      </button>
    </div>
  )

  if (!compute || !node || node.state !== 'ready')
    return (
      <section className="card">
        {head}
        <div className="sub">{compute ? 'This node is not ready to take a shell.' : 'Only a live compute takes a shell.'}</div>
      </section>
    )

  const all = nodes ?? []
  const spec = compute.spec.specs[0]
  const sh = shellFor(compute.id, rank)
  const context: Context = {
    compute,
    node,
    rank,
    count: spec?.accelerator_count ?? 1,
    peers: readyOf(all).length,
    concurrency: compute.spec.worker?.concurrency ?? 1,
    head: all[0]?.address ?? '—',
    metrics: metrics[`${compute.id}/${rank}`] ?? { gpu: 0, vram: 0, cpu: 0, temp: 0, net: 0 },
  }

  const submit = (e: React.FormEvent): void => {
    e.preventDefault()
    const field = input.current
    if (!field) return
    runCommand(context, field.value)
    field.value = ''
    field.focus()
    bump((n) => n + 1)
  }

  return (
    <section className="card">
      {head}
      <div className="term">
        <div className="term-out" id="term-out" ref={out}>
          {sh.lines.map((l, i) => (
            <div key={i} className={l.c}>
              {l.t}
            </div>
          ))}
        </div>
        <form className="term-in" onSubmit={submit}>
          <span>{promptOf(compute, rank)}</span>
          <input id="term-input" name="cmd" autoComplete="off" spellCheck={false} aria-label="shell command" ref={input} />
        </form>
      </div>
    </section>
  )
}
