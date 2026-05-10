import { useEffect, useMemo, useState } from "react";
import type { Pose3D } from "../gs/gsSdfSceneAdapter";
import type { DecodedPointCloud } from "../ros/pointCloud2";
import {
  getLiveHttpEndpoint,
  resolveLiveContractUrl,
  type LiveContract
} from "../contracts/liveContract";

type ConnectionState = "connecting" | "connected" | "closed" | "error";
type StreamProfile = "idle" | "neural" | "playback";

export interface HttpLiveState {
  connectionState: ConnectionState;
  streamProfile: StreamProfile;
  robotPose: Pose3D | null;
  trajectory: Pose3D[];
  livePointCloud: DecodedPointCloud | null;
  liveMetrics: HttpLiveMetrics | null;
  errorMessage: string | null;
}

export interface HttpLiveMetrics {
  frameIndex?: number;
  frameCount?: number;
  currentPointCount?: number;
  globalPointCount?: number;
}

export function useHttpLiveState(liveContract: LiveContract | null): HttpLiveState {
  const poseEndpoint = useMemo(
    () => {
      const endpoint = getLiveHttpEndpoint(liveContract, "robot_pose", "subscribe");
      return endpoint ? resolveLiveContractUrl(endpoint, window.location.href) : null;
    },
    [liveContract]
  );
  const trajectoryEndpoint = useMemo(
    () => {
      const endpoint = getLiveHttpEndpoint(liveContract, "trajectory", "subscribe");
      return endpoint ? resolveLiveContractUrl(endpoint, window.location.href) : null;
    },
    [liveContract]
  );
  const livePointCloudEndpoint = useMemo(
    () => {
      const endpoint = getLiveHttpEndpoint(liveContract, "live_point_cloud", "subscribe");
      return endpoint ? resolveLiveContractUrl(endpoint, window.location.href) : null;
    },
    [liveContract]
  );
  const hasHttpFeed = Boolean(poseEndpoint || trajectoryEndpoint || livePointCloudEndpoint);

  const [connectionState, setConnectionState] = useState<ConnectionState>(hasHttpFeed ? "connecting" : "closed");
  const [streamProfile, setStreamProfile] = useState<StreamProfile>(hasHttpFeed ? "neural" : "idle");
  const [robotPose, setRobotPose] = useState<Pose3D | null>(null);
  const [trajectory, setTrajectory] = useState<Pose3D[]>([]);
  const [livePointCloud, setLivePointCloud] = useState<DecodedPointCloud | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<HttpLiveMetrics | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!hasHttpFeed) {
      setConnectionState("closed");
      setStreamProfile("idle");
      setRobotPose(null);
      setTrajectory([]);
      setLivePointCloud(null);
      setLiveMetrics(null);
      setErrorMessage(null);
      return;
    }

    let disposed = false;
    let poseTimer = 0;
    let trajectoryTimer = 0;
    let pointCloudTimer = 0;

    setConnectionState("connecting");
    setStreamProfile("neural");
    setErrorMessage(null);

    const fetchPose = async () => {
      if (!poseEndpoint || disposed) {
        return;
      }
      try {
        const response = await fetch(poseEndpoint, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`robot_pose returned ${response.status}`);
        }
        const payload = (await response.json()) as unknown;
        const nextPose = extractPose3D(payload);
        if (disposed) {
          return;
        }
        if (nextPose) {
          setRobotPose(nextPose);
          setConnectionState("connected");
          setErrorMessage(null);
        }
      } catch (error) {
        if (!disposed) {
          setConnectionState("error");
          setErrorMessage(error instanceof Error ? error.message : "Failed to fetch robot_pose.");
        }
      }
    };

    const fetchTrajectory = async () => {
      if (!trajectoryEndpoint || disposed) {
        return;
      }
      try {
        const response = await fetch(trajectoryEndpoint, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`trajectory returned ${response.status}`);
        }
        const payload = (await response.json()) as unknown;
        const nextTrajectory = extractTrajectory(payload);
        if (disposed) {
          return;
        }
        setTrajectory(nextTrajectory);
        if (nextTrajectory.length > 0) {
          setConnectionState("connected");
          setErrorMessage(null);
        }
      } catch (error) {
        if (!disposed && !poseEndpoint) {
          setConnectionState("error");
          setErrorMessage(error instanceof Error ? error.message : "Failed to fetch trajectory.");
        }
      }
    };

    const fetchLivePointCloud = async () => {
      if (!livePointCloudEndpoint || disposed) {
        return;
      }
      try {
        const response = await fetch(livePointCloudEndpoint, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`live_point_cloud returned ${response.status}`);
        }
        const payload = (await response.json()) as unknown;
        const nextPointCloud = extractPointCloud(payload);
        const nextMetrics = extractLiveMetrics(payload);
        if (disposed) {
          return;
        }
        setLivePointCloud(nextPointCloud);
        setLiveMetrics(nextMetrics);
        if ((nextPointCloud && nextPointCloud.renderedPointCount > 0) || nextMetrics) {
          setConnectionState("connected");
          setErrorMessage(null);
        }
      } catch (error) {
        if (!disposed && !poseEndpoint && !trajectoryEndpoint) {
          setConnectionState("error");
          setErrorMessage(error instanceof Error ? error.message : "Failed to fetch live_point_cloud.");
        }
      }
    };

    void fetchPose();
    void fetchTrajectory();
    void fetchLivePointCloud();
    poseTimer = window.setInterval(fetchPose, 200);
    trajectoryTimer = window.setInterval(fetchTrajectory, 900);
    pointCloudTimer = window.setInterval(fetchLivePointCloud, 900);

    return () => {
      disposed = true;
      window.clearInterval(poseTimer);
      window.clearInterval(trajectoryTimer);
      window.clearInterval(pointCloudTimer);
    };
  }, [hasHttpFeed, livePointCloudEndpoint, poseEndpoint, trajectoryEndpoint]);

  return useMemo(
    () => ({
      connectionState,
      streamProfile,
      robotPose,
      trajectory,
      livePointCloud,
      liveMetrics,
      errorMessage
    }),
    [connectionState, errorMessage, liveMetrics, livePointCloud, robotPose, streamProfile, trajectory]
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function extractPose3D(payload: unknown): Pose3D | null {
  const raw = isRecord(payload) && isRecord(payload.pose) ? payload.pose : payload;
  if (!isRecord(raw) || !isRecord(raw.position) || !isRecord(raw.orientation)) {
    return null;
  }
  return {
    frameId: typeof raw.frameId === "string" ? raw.frameId : "world",
    position: {
      x: asNumber(raw.position.x, 0),
      y: asNumber(raw.position.y, 0),
      z: asNumber(raw.position.z, 0)
    },
    orientation: {
      x: asNumber(raw.orientation.x, 0),
      y: asNumber(raw.orientation.y, 0),
      z: asNumber(raw.orientation.z, 0),
      w: asNumber(raw.orientation.w, 1)
    },
    stampMs: typeof raw.stampMs === "number" ? raw.stampMs : undefined
  };
}

function extractTrajectory(payload: unknown): Pose3D[] {
  const rawPoses = Array.isArray(payload)
    ? payload
    : isRecord(payload) && Array.isArray(payload.poses)
      ? payload.poses
      : [];
  return rawPoses.map((entry) => extractPose3D(entry)).filter((entry): entry is Pose3D => entry !== null);
}

function extractPointCloud(payload: unknown): DecodedPointCloud | null {
  const raw = isRecord(payload) && isRecord(payload.pointCloud) ? payload.pointCloud : payload;
  if (!isRecord(raw) || !Array.isArray(raw.positions) || raw.positions.length < 3) {
    return null;
  }

  const positions = Float32Array.from(
    raw.positions.filter((value): value is number => typeof value === "number" && Number.isFinite(value))
  );
  if (positions.length < 3) {
    return null;
  }

  const renderedPointCount =
    typeof raw.renderedPointCount === "number" && raw.renderedPointCount > 0
      ? raw.renderedPointCount
      : Math.floor(positions.length / 3);
  const colors =
    Array.isArray(raw.colors) && raw.colors.length === positions.length
      ? Float32Array.from(
          raw.colors.map((value) =>
            typeof value === "number" && Number.isFinite(value)
              ? value > 1
                ? value / 255
                : value
              : 0
          )
        )
      : undefined;

  return {
    frameId: typeof raw.frameId === "string" ? raw.frameId : "world",
    stampMs: typeof raw.stampMs === "number" ? raw.stampMs : 0,
    sourcePointCount:
      typeof raw.sourcePointCount === "number" && raw.sourcePointCount >= renderedPointCount
        ? raw.sourcePointCount
        : renderedPointCount,
    renderedPointCount,
    positions,
    colors
  };
}

function extractLiveMetrics(payload: unknown): HttpLiveMetrics | null {
  if (!isRecord(payload)) {
    return null;
  }
  const frameIndex = asOptionalNumber(payload.frame_index);
  const frameCount = asOptionalNumber(payload.frame_count);
  const currentPointCount = asOptionalNumber(payload.current_point_count);
  const globalPointCount = asOptionalNumber(payload.global_point_count);
  if (
    frameIndex === undefined &&
    frameCount === undefined &&
    currentPointCount === undefined &&
    globalPointCount === undefined
  ) {
    return null;
  }
  return {
    frameIndex,
    frameCount,
    currentPointCount,
    globalPointCount
  };
}

function asOptionalNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}
