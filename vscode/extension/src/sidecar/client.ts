/**
 * Factory for creating a {@link SidecarClient}.
 *
 * Switches between the {@link MockClient} (hardcoded data for UI dev)
 * and the {@link RealClient} (spawns the Python bridge process) based
 * on the `skyward.useMock` configuration setting.
 */

import { MockClient } from "./mock-client";
import type { SidecarClient } from "./protocol";
import { RealClient } from "./real-client";

export function createClient(
  workspaceRoot: string,
  useMock: boolean,
): SidecarClient {
  return useMock ? new MockClient() : new RealClient(workspaceRoot);
}

export type { SidecarClient };
