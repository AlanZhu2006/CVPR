export type ArtifactSupport = "native" | "derived" | "passthrough" | "fallback" | "none";
export type IntegrationMode = "scene-pack" | "ros-live" | "playback-cache" | "hybrid" | "importer-only";
export type ArtifactKey =
  | "gaussian"
  | "mesh"
  | "occupancy"
  | "rawPointCloud"
  | "semantics"
  | "trajectory"
  | "rgb"
  | "depth"
  | "evaluation"
  | "externalViewer";

export interface ArtifactCapability {
  support: ArtifactSupport;
  formats?: string[];
  notes?: string;
}

export interface AdapterCapability {
  adapterId: string;
  displayName: string;
  sourceKind: string;
  integrationMode: IntegrationMode;
  artifacts: Partial<Record<ArtifactKey, ArtifactCapability>>;
  live: {
    profiles: string[];
    transports: string[];
    inboundChannels: string[];
    outboundChannels: string[];
    canonicalContract?: string | null;
    notes?: string;
  };
  navigation?: {
    publishGoal?: boolean;
    publishInitialPose?: boolean;
    publishCameraPose?: boolean;
  };
  sceneManifest?: {
    produces: boolean;
    producerScript?: string | null;
    notes?: string;
  };
  examples?: string[];
  notes?: string;
}

export interface CapabilityMatrix {
  schemaVersion: string;
  matrixId: string;
  description?: string;
  adapters: AdapterCapability[];
}

export const DEFAULT_CAPABILITY_MATRIX_URL = "/contracts/adapter-capability-matrix.default.json";

export function parseCapabilityMatrix(raw: unknown): CapabilityMatrix {
  if (!isRecord(raw)) {
    throw new Error("Capability matrix must be a JSON object.");
  }
  if (typeof raw.schemaVersion !== "string" || raw.schemaVersion.length < 1) {
    throw new Error("Capability matrix is missing schemaVersion.");
  }
  if (typeof raw.matrixId !== "string" || raw.matrixId.length < 1) {
    throw new Error("Capability matrix is missing matrixId.");
  }
  if (!Array.isArray(raw.adapters)) {
    throw new Error("Capability matrix is missing adapters.");
  }

  return {
    schemaVersion: raw.schemaVersion,
    matrixId: raw.matrixId,
    description: typeof raw.description === "string" ? raw.description : undefined,
    adapters: raw.adapters.map((adapter, index) => parseAdapterCapability(adapter, index))
  };
}

export function resolveAdapterCapability(
  matrix: CapabilityMatrix | null | undefined,
  sourceKind: string | null | undefined
): AdapterCapability | null {
  if (!matrix || !sourceKind) {
    return null;
  }

  return matrix.adapters.find((adapter) => adapter.sourceKind === sourceKind) ?? null;
}

export function adapterSupportsArtifact(
  adapter: AdapterCapability | null | undefined,
  artifact: ArtifactKey
): boolean {
  const support = adapter?.artifacts?.[artifact]?.support;
  return Boolean(support && support !== "none");
}

export function adapterSupportsLiveProfile(
  adapter: AdapterCapability | null | undefined,
  profile: string
): boolean {
  return Boolean(adapter?.live.profiles.includes(profile));
}

export function adapterSupportsInboundChannel(
  adapter: AdapterCapability | null | undefined,
  channel: string
): boolean {
  return Boolean(adapter?.live.inboundChannels.includes(channel));
}

export function adapterSupportsOutboundChannel(
  adapter: AdapterCapability | null | undefined,
  channel: string
): boolean {
  return Boolean(adapter?.live.outboundChannels.includes(channel));
}

function parseAdapterCapability(raw: unknown, index: number): AdapterCapability {
  if (!isRecord(raw)) {
    throw new Error(`Capability matrix adapter #${index} must be an object.`);
  }
  if (typeof raw.adapterId !== "string" || raw.adapterId.length < 1) {
    throw new Error(`Capability matrix adapter #${index} is missing adapterId.`);
  }
  if (typeof raw.displayName !== "string" || raw.displayName.length < 1) {
    throw new Error(`Capability matrix adapter "${raw.adapterId}" is missing displayName.`);
  }
  if (typeof raw.sourceKind !== "string" || raw.sourceKind.length < 1) {
    throw new Error(`Capability matrix adapter "${raw.adapterId}" is missing sourceKind.`);
  }
  if (!isIntegrationMode(raw.integrationMode)) {
    throw new Error(`Capability matrix adapter "${raw.adapterId}" has an invalid integrationMode.`);
  }
  if (!isRecord(raw.artifacts)) {
    throw new Error(`Capability matrix adapter "${raw.adapterId}" is missing artifacts.`);
  }
  if (!isRecord(raw.live)) {
    throw new Error(`Capability matrix adapter "${raw.adapterId}" is missing live.`);
  }

  const live = raw.live;
  return {
    adapterId: raw.adapterId,
    displayName: raw.displayName,
    sourceKind: raw.sourceKind,
    integrationMode: raw.integrationMode,
    artifacts: parseArtifacts(raw.artifacts),
    live: {
      profiles: toStringArray(live.profiles),
      transports: toStringArray(live.transports),
      inboundChannels: toStringArray(live.inboundChannels),
      outboundChannels: toStringArray(live.outboundChannels),
      canonicalContract: typeof live.canonicalContract === "string" || live.canonicalContract === null
        ? live.canonicalContract
        : undefined,
      notes: typeof live.notes === "string" ? live.notes : undefined
    },
    navigation: isRecord(raw.navigation)
      ? {
          publishGoal:
            typeof raw.navigation.publishGoal === "boolean" ? raw.navigation.publishGoal : undefined,
          publishInitialPose:
            typeof raw.navigation.publishInitialPose === "boolean"
              ? raw.navigation.publishInitialPose
              : undefined,
          publishCameraPose:
            typeof raw.navigation.publishCameraPose === "boolean"
              ? raw.navigation.publishCameraPose
              : undefined
        }
      : undefined,
    sceneManifest: isRecord(raw.sceneManifest)
      ? {
          produces: Boolean(raw.sceneManifest.produces),
          producerScript:
            typeof raw.sceneManifest.producerScript === "string" || raw.sceneManifest.producerScript === null
              ? raw.sceneManifest.producerScript
              : undefined,
          notes: typeof raw.sceneManifest.notes === "string" ? raw.sceneManifest.notes : undefined
        }
      : undefined,
    examples: toStringArray(raw.examples),
    notes: typeof raw.notes === "string" ? raw.notes : undefined
  };
}

function parseArtifacts(raw: Record<string, unknown>): Partial<Record<ArtifactKey, ArtifactCapability>> {
  const result: Partial<Record<ArtifactKey, ArtifactCapability>> = {};
  for (const key of Object.keys(raw) as ArtifactKey[]) {
    const value = raw[key];
    if (!isRecord(value) || !isArtifactSupport(value.support)) {
      continue;
    }
    result[key] = {
      support: value.support,
      formats: toStringArray(value.formats),
      notes: typeof value.notes === "string" ? value.notes : undefined
    };
  }
  return result;
}

function toStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];
}

function isRecord(value: unknown): value is Record<string, any> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isArtifactSupport(value: unknown): value is ArtifactSupport {
  return (
    value === "native" ||
    value === "derived" ||
    value === "passthrough" ||
    value === "fallback" ||
    value === "none"
  );
}

function isIntegrationMode(value: unknown): value is IntegrationMode {
  return (
    value === "scene-pack" ||
    value === "ros-live" ||
    value === "playback-cache" ||
    value === "hybrid" ||
    value === "importer-only"
  );
}
