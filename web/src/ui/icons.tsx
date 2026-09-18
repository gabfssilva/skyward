export const ICON = {
  fleet: '<path d="M12 3l7 4v8l-7 4-7-4V7z"/><path d="M12 11l7-4M12 11v8M12 11L5 7"/>',
  market: '<path d="M3 9l1.5-5h15L21 9"/><path d="M3 9v11h18V9"/><path d="M3 9a3 3 0 006 0 3 3 0 006 0 3 3 0 006 0"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  shell: '<path d="M5 7l5 5-5 5M12 17h7"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M20 14.5A8 8 0 019.5 4a8 8 0 1010.5 10.5z"/>',
  auto: '<circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 010 18z" fill="currentColor" stroke="none"/>',
  activity: '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
  tasks: '<path d="M4 7l2 2 4-4M4 17l2 2 4-4M13 7h7M13 17h7"/>',
  ports: '<path d="M4 12h6M14 12h6"/><circle cx="12" cy="12" r="3"/><path d="M4 8v8M20 8v8"/>',
  close: '<path d="M6 6l12 12M18 6L6 18"/>',
  back: '<path d="M15 5l-7 7 7 7"/>',
  more: '<circle cx="5" cy="12" r="1.5" fill="currentColor"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/><circle cx="19" cy="12" r="1.5" fill="currentColor"/>',
  check: '<path d="M5 12l5 5 9-10"/>',
  alert: '<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17h.01"/>',
  refresh: '<path d="M20 12a8 8 0 01-14 5.3M4 12a8 8 0 0114-5.3"/><path d="M4 4v5h5M20 20v-5h-5"/>',
  run: '<path d="M7 5l12 7-12 7z"/>',
  next: '<path d="M9 6l6 6-6 6"/>',
  drain: '<path d="M12 3v10M8 9l4 4 4-4M4 19h16"/>',
} as const

export type IconName = keyof typeof ICON

export function Icon({ name }: { name: IconName }) {
  return <svg className="ic" viewBox="0 0 24 24" aria-hidden="true" dangerouslySetInnerHTML={{ __html: ICON[name] }} />
}
