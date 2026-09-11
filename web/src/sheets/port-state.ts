import { create } from 'zustand'

/**
 * The bridged ports of a compute.
 *
 * The daemon's forwarding endpoint is a raw socket bridge with no listing in the
 * HTTP contract, so what is open is remembered here, in the client that opened it.
 */
export type Port = { route: string; remote: number; local: number; node: number; state: 'open' }

export type PortsStore = {
  ports: Record<string, Port[]>
  add: (computeId: string, port: Port) => void
  close: (computeId: string, remote: number) => void
}

export const usePorts = create<PortsStore>((set) => ({
  ports: {},
  add: (computeId, port) => set((s) => ({ ports: { ...s.ports, [computeId]: [...(s.ports[computeId] ?? []), port] } })),
  close: (computeId, remote) =>
    set((s) => ({ ports: { ...s.ports, [computeId]: (s.ports[computeId] ?? []).filter((p) => p.remote !== remote) } })),
}))
