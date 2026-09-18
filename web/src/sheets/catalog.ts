/** The static vocabulary the console offers, ported from the prototype's data block. */
export const ACCELS: Record<string, { vram: number; arch: string }> = {
  b200: { vram: 192, arch: 'Blackwell' },
  h200: { vram: 141, arch: 'Hopper' },
  h100: { vram: 80, arch: 'Hopper' },
  a100: { vram: 80, arch: 'Ampere' },
  l40s: { vram: 48, arch: 'Ada Lovelace' },
  a10g: { vram: 24, arch: 'Ampere' },
  l4: { vram: 24, arch: 'Ada Lovelace' },
  mi300x: { vram: 192, arch: 'CDNA3' },
}

export const PLUGINS: readonly string[] = ['torch', 'jax', 'keras', 'accelerate', 'huggingface', 'joblib', 'cuml', 'sklearn', 'mig', 'mps']

export const COLLECTIVE: ReadonlySet<string> = new Set(['torch', 'jax', 'accelerate'])

export const VIEWS: readonly (readonly [string, string])[] = [
  ['/', 'Computes'],
  ['/tasks', 'Tasks'],
  ['/activity', 'Activity'],
  ['/market', 'Market'],
  ['/market/accounts', 'Provider accounts'],
]
