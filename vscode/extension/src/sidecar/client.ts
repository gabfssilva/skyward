/**
 * Re-export for easy swapping between mock and real daemon client.
 *
 * Change the import here when a real sidecar client is implemented.
 */

export { MockClient as createClient } from "./mock-client";
export type { SidecarClient } from "./protocol";
