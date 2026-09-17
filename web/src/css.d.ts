import type {} from 'react'

declare module 'react' {
  interface CSSProperties {
    [property: `--${string}`]: string | number
  }
}
