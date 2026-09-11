export type Theme = 'auto' | 'light' | 'dark'

const KEY = 'sky-theme'

export function getTheme(): Theme {
  const raw = localStorage.getItem(KEY)
  return raw === 'light' || raw === 'dark' || raw === 'auto' ? raw : 'auto'
}

export function setTheme(theme: Theme): void {
  localStorage.setItem(KEY, theme)
  if (theme === 'auto') document.documentElement.removeAttribute('data-theme')
  else document.documentElement.setAttribute('data-theme', theme)
}

export function applyStoredTheme(): Theme {
  const forced = new URLSearchParams(window.location.search).get('theme')
  if (forced === 'light' || forced === 'dark' || forced === 'auto') {
    setTheme(forced)
    return forced
  }
  const theme = getTheme()
  setTheme(theme)
  return theme
}
