export type LiveContractChannelKey =
  | "robot_pose"
  | "trajectory"
  | "live_point_cloud"
  | "rgb_frame"
  | "depth_frame"
  | "playback_odometry"
  | "playback_scan_body"
  | "playback_scan_body_web"
  | "playback_rgb"
  | "playback_global_map"
  | "playback_current_scan_world"
  | "external_viewer"
  | "nav_goal"
  | "initial_pose"
  | "camera_pose";

export type LiveContractDirection = "subscribe" | "publish" | "request" | "response";
export type LiveContractTransport = "ros_topic" | "foxglove_topic" | "websocket" | "http" | "custom";

export interface LiveContractChannel {
  key: string;
  profile?: string;
  direction: LiveContractDirection;
  transport: LiveContractTransport;
  topic?: string;
  endpoint?: string;
  messageType?: string;
  semanticType: string;
  frameIdPolicy?: string;
  fixedFrameId?: string;
  required: boolean;
  enabledByDefault?: boolean;
  fallbackKeys?: string[];
  rateHz?: number;
  notes?: string;
}

export interface LiveContract {
  schemaVersion: string;
  contractId: string;
  displayName?: string;
  transportHints?: {
    preferredBridge?: string;
    defaultUrlEnv?: string;
    notes?: string;
  };
  selectionPolicy?: {
    profilePriority?: string[];
    preferFirstActive?: boolean;
  };
  channels: LiveContractChannel[];
}

export const DEFAULT_LIVE_CONTRACT_URL = "/contracts/live-contract.default.json";

export function parseLiveContract(raw: unknown): LiveContract {
  if (!isRecord(raw)) {
    throw new Error("Live contract must be a JSON object.");
  }

  if (typeof raw.schemaVersion !== "string" || raw.schemaVersion.length < 1) {
    throw new Error("Live contract is missing schemaVersion.");
  }
  if (typeof raw.contractId !== "string" || raw.contractId.length < 1) {
    throw new Error("Live contract is missing contractId.");
  }
  if (!Array.isArray(raw.channels)) {
    throw new Error("Live contract is missing channels.");
  }

  return {
    schemaVersion: raw.schemaVersion,
    contractId: raw.contractId,
    displayName: typeof raw.displayName === "string" ? raw.displayName : undefined,
    transportHints: isRecord(raw.transportHints)
      ? {
          preferredBridge:
            typeof raw.transportHints.preferredBridge === "string"
              ? raw.transportHints.preferredBridge
              : undefined,
          defaultUrlEnv:
            typeof raw.transportHints.defaultUrlEnv === "string"
              ? raw.transportHints.defaultUrlEnv
              : undefined,
          notes: typeof raw.transportHints.notes === "string" ? raw.transportHints.notes : undefined
        }
      : undefined,
    selectionPolicy: isRecord(raw.selectionPolicy)
      ? {
          profilePriority: Array.isArray(raw.selectionPolicy.profilePriority)
            ? raw.selectionPolicy.profilePriority.filter((value): value is string => typeof value === "string")
            : undefined,
          preferFirstActive:
            typeof raw.selectionPolicy.preferFirstActive === "boolean"
              ? raw.selectionPolicy.preferFirstActive
              : undefined
        }
      : undefined,
    channels: raw.channels.map((channel, index) => parseLiveContractChannel(channel, index))
  };
}

export function getLiveContractChannel(
  contract: LiveContract | null | undefined,
  key: string,
  direction?: LiveContractDirection
): LiveContractChannel | null {
  if (!contract) {
    return null;
  }

  for (const channel of contract.channels) {
    if (channel.key !== key) {
      continue;
    }
    if (direction && channel.direction !== direction) {
      continue;
    }
    return channel;
  }

  return null;
}

export function getLiveRosTopicName(
  contract: LiveContract | null | undefined,
  key: string,
  fallbackTopic: string,
  direction?: LiveContractDirection
): string | null {
  if (!contract) {
    return fallbackTopic;
  }

  const channel = getLiveContractChannel(contract, key, direction);
  if (!channel || channel.transport !== "ros_topic") {
    return null;
  }

  return channel.topic ?? fallbackTopic;
}

export function getLiveRosMessageType(
  contract: LiveContract | null | undefined,
  key: string,
  fallbackMessageType: string,
  direction?: LiveContractDirection
): string | null {
  if (!contract) {
    return fallbackMessageType;
  }

  const channel = getLiveContractChannel(contract, key, direction);
  if (!channel || channel.transport !== "ros_topic") {
    return null;
  }

  return channel.messageType ?? fallbackMessageType;
}

export function getLiveHttpEndpoint(
  contract: LiveContract | null | undefined,
  key: string,
  direction?: LiveContractDirection
): string | null {
  if (!contract) {
    return null;
  }

  const channel = getLiveContractChannel(contract, key, direction);
  if (!channel || (channel.transport !== "http" && channel.transport !== "custom")) {
    return null;
  }

  return channel.endpoint ?? null;
}

export function resolveLiveContractUrl(reference: string, baseUrl?: string): string {
  if (/^[a-z]+:\/\//i.test(reference)) {
    return reference;
  }

  if (reference.startsWith("/")) {
    return reference;
  }

  if (baseUrl) {
    return new URL(reference, baseUrl).toString();
  }

  return reference;
}

function parseLiveContractChannel(raw: unknown, index: number): LiveContractChannel {
  if (!isRecord(raw)) {
    throw new Error(`Live contract channel #${index} must be an object.`);
  }
  if (typeof raw.key !== "string" || raw.key.length < 1) {
    throw new Error(`Live contract channel #${index} is missing key.`);
  }
  if (!isDirection(raw.direction)) {
    throw new Error(`Live contract channel "${raw.key}" has an invalid direction.`);
  }
  if (!isTransport(raw.transport)) {
    throw new Error(`Live contract channel "${raw.key}" has an invalid transport.`);
  }
  if (typeof raw.semanticType !== "string" || raw.semanticType.length < 1) {
    throw new Error(`Live contract channel "${raw.key}" is missing semanticType.`);
  }
  if (typeof raw.required !== "boolean") {
    throw new Error(`Live contract channel "${raw.key}" is missing required.`);
  }

  return {
    key: raw.key,
    profile: typeof raw.profile === "string" ? raw.profile : undefined,
    direction: raw.direction,
    transport: raw.transport,
    topic: typeof raw.topic === "string" ? raw.topic : undefined,
    endpoint: typeof raw.endpoint === "string" ? raw.endpoint : undefined,
    messageType: typeof raw.messageType === "string" ? raw.messageType : undefined,
    semanticType: raw.semanticType,
    frameIdPolicy: typeof raw.frameIdPolicy === "string" ? raw.frameIdPolicy : undefined,
    fixedFrameId: typeof raw.fixedFrameId === "string" ? raw.fixedFrameId : undefined,
    required: raw.required,
    enabledByDefault: typeof raw.enabledByDefault === "boolean" ? raw.enabledByDefault : undefined,
    fallbackKeys: Array.isArray(raw.fallbackKeys)
      ? raw.fallbackKeys.filter((value): value is string => typeof value === "string")
      : undefined,
    rateHz: typeof raw.rateHz === "number" ? raw.rateHz : undefined,
    notes: typeof raw.notes === "string" ? raw.notes : undefined
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isDirection(value: unknown): value is LiveContractDirection {
  return value === "subscribe" || value === "publish" || value === "request" || value === "response";
}

function isTransport(value: unknown): value is LiveContractTransport {
  return (
    value === "ros_topic" ||
    value === "foxglove_topic" ||
    value === "websocket" ||
    value === "http" ||
    value === "custom"
  );
}
