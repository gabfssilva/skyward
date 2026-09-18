import { useEffect, useState } from 'react'

const EVERY = 30_000

/**
 * What ``read`` answers now and every half minute after, while the caller is up and ``key`` stays the same.
 *
 * For the two things no event carries: a machine that was lost has left the node listing, and how fast a
 * compute is finishing tasks is a question about the daemon's own records. A read that fails keeps the last answer.
 */
export function usePolled<T>(read: () => Promise<T>, key: string, initial: T): T {
  const [value, setValue] = useState(initial)
  useEffect(() => {
    let live = true
    const once = () =>
      read().then(
        (answer) => live && setValue(answer),
        () => undefined,
      )
    void once()
    const every = setInterval(once, EVERY)
    return () => {
      live = false
      clearInterval(every)
    }
  }, [key])
  return value
}
