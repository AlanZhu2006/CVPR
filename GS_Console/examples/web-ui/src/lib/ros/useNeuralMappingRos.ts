import { useEffect, useMemo, useRef, useState } from "react";
import ROSLIB from "roslib";
import type { Pose3D } from "../gs/gsSdfSceneAdapter";
import type { RosClient } from "./navGoalPublisher";
import {
  getLiveRosMessageType,
  getLiveRosTopicName,
  type LiveContract
} from "../contracts/liveContract";
import { decodePointCloud2, type DecodedPointCloud, type PointCloud2Message } from "./pointCloud2";
import { projectPlaybackCloudToWorld } from "./playbackProjection";
import { decodeRosImage, type DecodedRosImage, type RosImageMessage } from "./rosImage";

type ConnectionState = "connecting" | "connected" | "closed" | "error";
type StreamProfile = "idle" | "neural" | "playback";

export interface PlaybackCloudOptions {
  enabled: boolean;
  maxInputPoints: number;
  maxAccumulatedPoints: number;
  voxelSize: number;
  publishEveryFrames: number;
  decayTimeSec: number;
  minVoxelObservations: number;
}

interface RosHookOptions {
  playbackCloud?: Partial<PlaybackCloudOptions>;
  playbackTargetStampMs?: number | null;
  liveContract?: LiveContract | null;
  disableRgbFrame?: boolean;
  disableDepthFrame?: boolean;
  disablePlaybackTopics?: boolean;
  disableLivePointCloud?: boolean;
  disableRos?: boolean;
}

export interface RosLiveState {
  ros: RosClient | null;
  connectionState: ConnectionState;
  streamProfile: StreamProfile;
  robotPose: Pose3D | null;
  trajectory: Pose3D[];
  livePointCloud: DecodedPointCloud | null;
  playbackAccumulatedPointCloud: DecodedPointCloud | null;
  playbackScanPointCloud: DecodedPointCloud | null;
  rgbFrame: DecodedRosImage | null;
  depthFrame: DecodedRosImage | null;
  errorMessage: string | null;
}

interface RosPoseMessage {
  header?: {
    frame_id?: string;
    stamp?: {
      secs?: number;
      nsecs?: number;
    };
  };
  pose?: {
    position?: { x?: number; y?: number; z?: number };
    orientation?: { x?: number; y?: number; z?: number; w?: number };
  };
}

interface RosPathMessage {
  poses?: RosPoseMessage[];
}

interface RosOdometryMessage {
  header?: {
    frame_id?: string;
    stamp?: {
      secs?: number;
      nsecs?: number;
    };
  };
  pose?: {
    pose?: {
      position?: { x?: number; y?: number; z?: number };
      orientation?: { x?: number; y?: number; z?: number; w?: number };
    };
  };
}

interface PlaybackVoxel {
  ix: number;
  iy: number;
  iz: number;
  x: number;
  y: number;
  z: number;
  positionSamples: number;
  color?: [number, number, number];
  colorSamples: number;
  lastSeenStampMs: number;
}

const defaultPlaybackCloudOptions: PlaybackCloudOptions = {
  enabled: true,
  maxInputPoints: 2_600,
  maxAccumulatedPoints: 180_000,
  voxelSize: 0.24,
  publishEveryFrames: 4,
  decayTimeSec: 0,
  minVoxelObservations: 2
};

export function useNeuralMappingRos(wsUrl: string, options?: RosHookOptions): RosLiveState {
  const [reconnectToken, setReconnectToken] = useState(0);
  const [ros, setRos] = useState<RosClient | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>("connecting");
  const [streamProfile, setStreamProfile] = useState<StreamProfile>("idle");
  const [robotPose, setRobotPose] = useState<Pose3D | null>(null);
  const [trajectory, setTrajectory] = useState<Pose3D[]>([]);
  const [livePointCloud, setLivePointCloud] = useState<DecodedPointCloud | null>(null);
  const [playbackAccumulatedPointCloud, setPlaybackAccumulatedPointCloud] = useState<DecodedPointCloud | null>(null);
  const [playbackScanPointCloud, setPlaybackScanPointCloud] = useState<DecodedPointCloud | null>(null);
  const [rgbFrame, setRgbFrame] = useState<DecodedRosImage | null>(null);
  const [depthFrame, setDepthFrame] = useState<DecodedRosImage | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const liveContract = options?.liveContract ?? null;
  const disableRgbFrame = options?.disableRgbFrame ?? false;
  const disableDepthFrame = options?.disableDepthFrame ?? false;
  const disablePlaybackTopics = options?.disablePlaybackTopics ?? false;
  const disableLivePointCloud = options?.disableLivePointCloud ?? false;
  const disableRos = options?.disableRos ?? false;
  const playbackTrajectoryRef = useRef<Pose3D[]>([]);
  const lastPlaybackPoseRef = useRef<Pose3D | null>(null);
  const latestPlaybackPoseRef = useRef<Pose3D | null>(null);
  const latestPlaybackRgbRef = useRef<DecodedRosImage | null>(null);
  const playbackRgbHistoryRef = useRef<DecodedRosImage[]>([]);
  const playbackWebCloudActiveUntilRef = useRef(0);
  const playbackCloudVoxelsRef = useRef<Map<string, PlaybackVoxel>>(new Map());
  const playbackCloudOrderRef = useRef<string[]>([]);
  const playbackCloudTickRef = useRef(0);
  const playbackSidecarMapActiveRef = useRef(false);
  const playbackSidecarScanActiveRef = useRef(false);
  const playbackScanHistoryRef = useRef<DecodedPointCloud[]>([]);
  const lastPlaybackOdomStampRef = useRef<number | null>(null);
  const lastPlaybackCloudStampRef = useRef<number | null>(null);
  const playbackTargetStampMsRef = useRef<number | null>(options?.playbackTargetStampMs ?? null);
  const playbackCloudOptions = useMemo(
    () => ({
      ...defaultPlaybackCloudOptions,
      ...options?.playbackCloud
    }),
    [options?.playbackCloud]
  );
  const playbackCloudOptionsRef = useRef<PlaybackCloudOptions>(playbackCloudOptions);

  useEffect(() => {
    playbackTargetStampMsRef.current = options?.playbackTargetStampMs ?? null;
    if (!playbackSidecarScanActiveRef.current) {
      return;
    }
    const selectedScan = selectPlaybackScanForTarget(
      playbackScanHistoryRef.current,
      playbackTargetStampMsRef.current
    );
    setPlaybackScanPointCloud(selectedScan);
  }, [options?.playbackTargetStampMs]);

  useEffect(() => {
    playbackCloudOptionsRef.current = playbackCloudOptions;
    if (playbackSidecarMapActiveRef.current || playbackSidecarScanActiveRef.current) {
      return;
    }
    playbackCloudVoxelsRef.current = new Map();
    playbackCloudOrderRef.current = [];
    playbackCloudTickRef.current = 0;
    playbackScanHistoryRef.current = [];
    lastPlaybackCloudStampRef.current = null;
    playbackWebCloudActiveUntilRef.current = 0;
    playbackSidecarMapActiveRef.current = false;
    playbackSidecarScanActiveRef.current = false;
    setLivePointCloud(null);
    setPlaybackAccumulatedPointCloud(null);
    setPlaybackScanPointCloud(null);
  }, [playbackCloudOptions]);

  const resetPlaybackAccumulation = () => {
    playbackTrajectoryRef.current = [];
    lastPlaybackPoseRef.current = null;
    playbackCloudVoxelsRef.current = new Map();
    playbackCloudOrderRef.current = [];
    playbackCloudTickRef.current = 0;
    playbackScanHistoryRef.current = [];
    lastPlaybackOdomStampRef.current = null;
    lastPlaybackCloudStampRef.current = null;
    playbackWebCloudActiveUntilRef.current = 0;
    playbackRgbHistoryRef.current = [];
    playbackSidecarMapActiveRef.current = false;
    playbackSidecarScanActiveRef.current = false;
    setTrajectory([]);
    setLivePointCloud(null);
    setPlaybackAccumulatedPointCloud(null);
    setPlaybackScanPointCloud(null);
  };

  useEffect(() => {
    let disposed = false;
    let reconnectTimer: number | null = null;
    if (disableRos) {
      setRos(null);
      setConnectionState("closed");
      setStreamProfile("idle");
      setErrorMessage(null);
      return;
    }
    const client = new ROSLIB.Ros({ url: wsUrl });
    setRos(client);
    setConnectionState("connecting");
    setStreamProfile("idle");
    setErrorMessage(null);

    const poseTopic = createConfiguredRosTopic(client, liveContract, {
      key: "robot_pose",
      fallbackTopic: "/neural_mapping/pose",
      fallbackMessageType: "geometry_msgs/PoseStamped"
    });

    const pathTopic = createConfiguredRosTopic(client, liveContract, {
      key: "trajectory",
      fallbackTopic: "/neural_mapping/path",
      fallbackMessageType: "nav_msgs/Path"
    });

    const pointCloudTopic = disableLivePointCloud
      ? null
      : createConfiguredRosTopic(client, liveContract, {
          key: "live_point_cloud",
          fallbackTopic: "/neural_mapping/pointcloud",
          fallbackMessageType: "sensor_msgs/PointCloud2",
          throttleRate: 350,
          queueLength: 1,
          compression: "none"
        });

    const rgbTopic = disableRgbFrame
      ? null
      : createConfiguredRosTopic(client, liveContract, {
          key: "rgb_frame",
          fallbackTopic: "/neural_mapping/rgb",
          fallbackMessageType: "sensor_msgs/Image",
          throttleRate: 120,
          queueLength: 1,
          compression: "png"
        });

    const depthTopic = disableDepthFrame
      ? null
      : createConfiguredRosTopic(client, liveContract, {
          key: "depth_frame",
          fallbackTopic: "/neural_mapping/depth",
          fallbackMessageType: "sensor_msgs/Image",
          throttleRate: 250,
          queueLength: 1,
          compression: "none"
        });

    const playbackOdomTopic = disablePlaybackTopics
      ? null
      : createConfiguredRosTopic(client, liveContract, {
          key: "playback_odometry",
          fallbackTopic: "/aft_mapped_to_init",
          fallbackMessageType: "nav_msgs/Odometry"
        });

    const playbackPointCloudTopic = disablePlaybackTopics
      ? null
      : createConfiguredRosTopic(client, liveContract, {
          key: "playback_scan_body",
          fallbackTopic: "/cloud_registered_body",
          fallbackMessageType: "sensor_msgs/PointCloud2",
          throttleRate: 150,
          queueLength: 1,
          compression: "none"
        });

    const playbackWebPointCloudTopic = disablePlaybackTopics
      ? null
      : createConfiguredRosTopic(client, liveContract, {
          key: "playback_scan_body_web",
          fallbackTopic: "/cloud_registered_body_web",
          fallbackMessageType: "sensor_msgs/PointCloud2",
          throttleRate: 45,
          queueLength: 1,
          compression: "none"
        });

    const playbackRgbTopic = disablePlaybackTopics
      ? null
      : createConfiguredRosTopic(client, liveContract, {
          key: "playback_rgb",
          fallbackTopic: "/origin_img",
          fallbackMessageType: "sensor_msgs/Image",
          throttleRate: 70,
          queueLength: 1,
          compression: "png"
        });

    const playbackGlobalMapTopic = disablePlaybackTopics
      ? null
      : createConfiguredRosTopic(client, liveContract, {
          key: "playback_global_map",
          fallbackTopic: "/fastlivo/global_map_web",
          fallbackMessageType: "sensor_msgs/PointCloud2",
          throttleRate: 55,
          queueLength: 1,
          compression: "none"
        });

    const playbackCurrentScanWorldTopic = disablePlaybackTopics
      ? null
      : createConfiguredRosTopic(client, liveContract, {
          key: "playback_current_scan_world",
          fallbackTopic: "/fastlivo/current_scan_world_web",
          fallbackMessageType: "sensor_msgs/PointCloud2",
          throttleRate: 35,
          queueLength: 1,
          compression: "none"
        });

    let neuralProfileActive = false;

    const handleConnection = () => {
      setConnectionState("connected");
      setErrorMessage(null);
      poseTopic?.subscribe((message: unknown) => {
        neuralProfileActive = true;
        setStreamProfile("neural");
        setRobotPose(toPose3D(message as RosPoseMessage));
      });

      pathTopic?.subscribe((message: unknown) => {
        neuralProfileActive = true;
        setStreamProfile("neural");
        const poses = ((message as RosPathMessage).poses ?? []).map((poseMessage) =>
          toPose3D(poseMessage)
        );
        setTrajectory(poses);
      });

      pointCloudTopic?.subscribe((message: unknown) => {
        try {
          neuralProfileActive = true;
          setStreamProfile("neural");
          const decoded = decodePointCloud2(message as PointCloud2Message, {
            maxPoints: 90_000
          });
          setLivePointCloud(decoded);
          setPlaybackAccumulatedPointCloud(null);
          setPlaybackScanPointCloud(null);
        } catch (error) {
          console.warn("Failed to decode PointCloud2 message from rosbridge.", error);
        }
      });

      rgbTopic?.subscribe((message: unknown) => {
        try {
          neuralProfileActive = true;
          setStreamProfile("neural");
          setRgbFrame(decodeRosImage(message as RosImageMessage));
        } catch (error) {
          console.warn("Failed to decode RGB image from rosbridge.", error);
        }
      });

      depthTopic?.subscribe((message: unknown) => {
        try {
          neuralProfileActive = true;
          setStreamProfile("neural");
          setDepthFrame(decodeRosImage(message as RosImageMessage));
        } catch (error) {
          console.warn("Failed to decode depth image from rosbridge.", error);
        }
      });

      playbackOdomTopic?.subscribe((message: unknown) => {
        if (neuralProfileActive) {
          return;
        }

        const pose = toPose3DFromOdometry(message as RosOdometryMessage);
        const odomStampMs = pose.stampMs ?? 0;
        const previousStampMs = lastPlaybackOdomStampRef.current;
        if (previousStampMs !== null && odomStampMs + 400 < previousStampMs) {
          resetPlaybackAccumulation();
        }

        setStreamProfile("playback");
        setRobotPose(pose);
        latestPlaybackPoseRef.current = pose;
        lastPlaybackOdomStampRef.current = odomStampMs;

        if (!shouldAppendPlaybackPose(lastPlaybackPoseRef.current, pose)) {
          return;
        }

        playbackTrajectoryRef.current = [...playbackTrajectoryRef.current.slice(-2047), pose];
        lastPlaybackPoseRef.current = pose;
        setTrajectory(playbackTrajectoryRef.current);
      });

      const handlePlaybackCloudMessage = (message: unknown, source: "raw" | "web") => {
        if (neuralProfileActive) {
          return;
        }

        if (playbackSidecarMapActiveRef.current || playbackSidecarScanActiveRef.current) {
          return;
        }

        try {
          setStreamProfile("playback");
          if (!playbackCloudOptionsRef.current.enabled) {
            return;
          }
          if (!latestPlaybackPoseRef.current) {
            return;
          }
          if (source === "raw" && Date.now() < playbackWebCloudActiveUntilRef.current) {
            return;
          }
          const decoded = decodePointCloud2(message as PointCloud2Message, {
            maxPoints: playbackCloudOptionsRef.current.maxInputPoints
          });
          if (source === "web") {
            playbackWebCloudActiveUntilRef.current = Date.now() + 1800;
          }
          const cloudStampMs = decoded.stampMs ?? 0;
          const previousCloudStampMs = lastPlaybackCloudStampRef.current;
          if (previousCloudStampMs !== null && cloudStampMs + 400 < previousCloudStampMs) {
            resetPlaybackAccumulation();
          }

          const worldProjected = projectPlaybackCloudToWorld(
            decoded,
            latestPlaybackPoseRef.current,
            selectPlaybackRgbFrame(playbackRgbHistoryRef.current, decoded.stampMs)
          );
          const currentScan = simplifyPlaybackCurrentScan(
            worldProjected,
            Math.max(600, Math.round(playbackCloudOptionsRef.current.maxInputPoints * 0.85)),
            Math.max(playbackCloudOptionsRef.current.voxelSize * 0.9, 0.08)
          );
          const accumulated = accumulatePlaybackCloud(
            playbackCloudVoxelsRef.current,
            playbackCloudOrderRef.current,
            worldProjected,
            playbackCloudOptionsRef.current
          );
          lastPlaybackCloudStampRef.current = cloudStampMs;
          setPlaybackScanPointCloud(currentScan);
          setPlaybackAccumulatedPointCloud(accumulated);
          playbackCloudTickRef.current += 1;
          if (
            playbackCloudTickRef.current === 1 ||
            playbackCloudTickRef.current % Math.max(playbackCloudOptionsRef.current.publishEveryFrames, 1) === 0
          ) {
            setLivePointCloud(accumulated);
          }
        } catch (error) {
          console.warn("Failed to decode playback PointCloud2 message from rosbridge.", error);
        }
      };

      playbackPointCloudTopic?.subscribe((message: unknown) => {
        handlePlaybackCloudMessage(message, "raw");
      });

      playbackWebPointCloudTopic?.subscribe((message: unknown) => {
        handlePlaybackCloudMessage(message, "web");
      });

      playbackRgbTopic?.subscribe((message: unknown) => {
        if (neuralProfileActive) {
          return;
        }

        try {
          setStreamProfile("playback");
          const decoded = decodeRosImage(message as RosImageMessage);
          latestPlaybackRgbRef.current = decoded;
          playbackRgbHistoryRef.current = appendPlaybackRgbFrame(playbackRgbHistoryRef.current, decoded);
          setRgbFrame(decoded);
        } catch (error) {
          console.warn("Failed to decode playback RGB image from rosbridge.", error);
        }
      });

      playbackGlobalMapTopic?.subscribe((message: unknown) => {
        if (neuralProfileActive) {
          return;
        }

        try {
          setStreamProfile("playback");
          const decoded = decodePointCloud2(message as PointCloud2Message, {
            maxPoints: 80_000
          });
          const previousStampMs = lastPlaybackCloudStampRef.current;
          if (previousStampMs !== null && (decoded.stampMs ?? 0) + 400 < previousStampMs) {
            resetPlaybackAccumulation();
          }
          playbackSidecarMapActiveRef.current = true;
          lastPlaybackCloudStampRef.current = decoded.stampMs ?? null;
          setPlaybackAccumulatedPointCloud(decoded);
          setLivePointCloud(decoded);
        } catch (error) {
          console.warn("Failed to decode sidecar playback global map from rosbridge.", error);
        }
      });

      playbackCurrentScanWorldTopic?.subscribe((message: unknown) => {
        if (neuralProfileActive) {
          return;
        }

        try {
          setStreamProfile("playback");
          const decoded = decodePointCloud2(message as PointCloud2Message, {
            maxPoints: 16_000
          });
          playbackSidecarScanActiveRef.current = true;
          playbackScanHistoryRef.current = appendPlaybackScanHistory(
            playbackScanHistoryRef.current,
            decoded
          );
          const selectedScan = selectPlaybackScanForTarget(
            playbackScanHistoryRef.current,
            playbackTargetStampMsRef.current ?? latestPlaybackPoseRef.current?.stampMs ?? null
          );
          setPlaybackScanPointCloud(selectedScan);
        } catch (error) {
          console.warn("Failed to decode sidecar playback current scan from rosbridge.", error);
        }
      });
    };

    const handleClose = () => {
      if (disposed) {
        return;
      }
      setConnectionState("closed");
      setStreamProfile("idle");
      if (reconnectTimer === null) {
        reconnectTimer = window.setTimeout(() => {
          reconnectTimer = null;
          setConnectionState("connecting");
          setReconnectToken((current) => current + 1);
        }, 1200);
      }
    };

    const handleError = (error: unknown) => {
      if (disposed) {
        return;
      }
      setConnectionState("error");
      setErrorMessage(error instanceof Error ? error.message : "Failed to connect to rosbridge.");
      if (reconnectTimer === null) {
        reconnectTimer = window.setTimeout(() => {
          reconnectTimer = null;
          setConnectionState("connecting");
          setReconnectToken((current) => current + 1);
        }, 1500);
      }
    };

    client.on("connection", handleConnection);
    client.on("close", handleClose);
    client.on("error", handleError);

    return () => {
      disposed = true;
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
      }
      poseTopic?.unsubscribe();
      pathTopic?.unsubscribe();
      pointCloudTopic?.unsubscribe();
      rgbTopic?.unsubscribe();
      depthTopic?.unsubscribe();
      playbackOdomTopic?.unsubscribe();
      playbackPointCloudTopic?.unsubscribe();
      playbackWebPointCloudTopic?.unsubscribe();
      playbackRgbTopic?.unsubscribe();
      playbackGlobalMapTopic?.unsubscribe();
      playbackCurrentScanWorldTopic?.unsubscribe();
      client.close();
      setRos(null);
    };
  }, [disableDepthFrame, disableLivePointCloud, disablePlaybackTopics, disableRgbFrame, disableRos, liveContract, reconnectToken, wsUrl]);

  return useMemo(
    () => ({
      ros,
      connectionState,
      streamProfile,
      robotPose,
      trajectory,
      livePointCloud,
      playbackAccumulatedPointCloud,
      playbackScanPointCloud,
      rgbFrame,
      depthFrame,
      errorMessage
    }),
    [
      connectionState,
      depthFrame,
      errorMessage,
      livePointCloud,
      playbackAccumulatedPointCloud,
      playbackScanPointCloud,
      rgbFrame,
      robotPose,
      ros,
      streamProfile,
      trajectory
    ]
  );
}

interface ConfiguredRosTopicOptions {
  key: string;
  fallbackTopic: string;
  fallbackMessageType: string;
  throttleRate?: number;
  queueLength?: number;
  compression?: string;
}

function createConfiguredRosTopic(
  ros: RosClient,
  liveContract: LiveContract | null,
  options: ConfiguredRosTopicOptions
): any | null {
  const topicName = getLiveRosTopicName(liveContract, options.key, options.fallbackTopic, "subscribe");
  const messageType = getLiveRosMessageType(
    liveContract,
    options.key,
    options.fallbackMessageType,
    "subscribe"
  );
  if (!topicName || !messageType) {
    return null;
  }

  return new ROSLIB.Topic({
    ros,
    name: topicName,
    messageType,
    throttle_rate: options.throttleRate,
    queue_length: options.queueLength,
    compression: options.compression
  });
}

function toPose3D(message: RosPoseMessage): Pose3D {
  const secs = message.header?.stamp?.secs ?? 0;
  const nsecs = message.header?.stamp?.nsecs ?? 0;

  return {
    frameId: message.header?.frame_id ?? "world",
    position: {
      x: message.pose?.position?.x ?? 0,
      y: message.pose?.position?.y ?? 0,
      z: message.pose?.position?.z ?? 0
    },
    orientation: {
      x: message.pose?.orientation?.x ?? 0,
      y: message.pose?.orientation?.y ?? 0,
      z: message.pose?.orientation?.z ?? 0,
      w: message.pose?.orientation?.w ?? 1
    },
    stampMs: secs * 1000 + Math.round(nsecs / 1_000_000)
  };
}

function toPose3DFromOdometry(message: RosOdometryMessage): Pose3D {
  const secs = message.header?.stamp?.secs ?? 0;
  const nsecs = message.header?.stamp?.nsecs ?? 0;

  return {
    frameId: message.header?.frame_id ?? "world",
    position: {
      x: message.pose?.pose?.position?.x ?? 0,
      y: message.pose?.pose?.position?.y ?? 0,
      z: message.pose?.pose?.position?.z ?? 0
    },
    orientation: {
      x: message.pose?.pose?.orientation?.x ?? 0,
      y: message.pose?.pose?.orientation?.y ?? 0,
      z: message.pose?.pose?.orientation?.z ?? 0,
      w: message.pose?.pose?.orientation?.w ?? 1
    },
    stampMs: secs * 1000 + Math.round(nsecs / 1_000_000)
  };
}

function shouldAppendPlaybackPose(lastPose: Pose3D | null, nextPose: Pose3D): boolean {
  if (!lastPose) {
    return true;
  }

  const nextStampMs = nextPose.stampMs ?? 0;
  const lastStampMs = lastPose.stampMs ?? 0;
  const dt = Math.abs(nextStampMs - lastStampMs);
  const dx = nextPose.position.x - lastPose.position.x;
  const dy = nextPose.position.y - lastPose.position.y;
  const dz = nextPose.position.z - lastPose.position.z;
  const yawDelta = Math.abs(normalizeAngle(quaternionToYaw(nextPose.orientation) - quaternionToYaw(lastPose.orientation)));

  if (nextStampMs <= lastStampMs && Math.hypot(dx, dy, dz) < 1e-3 && yawDelta < 0.01) {
    return false;
  }

  return dt >= 45 || Math.hypot(dx, dy, dz) >= 0.025 || yawDelta >= 0.04;
}

function appendPlaybackScanHistory(
  history: DecodedPointCloud[],
  nextCloud: DecodedPointCloud
): DecodedPointCloud[] {
  const nextStampMs = nextCloud.stampMs ?? 0;
  const trimmed = history.filter((cloud) => {
    const stampMs = cloud.stampMs ?? 0;
    return nextStampMs <= 0 || stampMs >= nextStampMs - 1_200;
  });
  const last = trimmed[trimmed.length - 1];
  if (last && Math.abs((last.stampMs ?? 0) - nextStampMs) <= 1) {
    trimmed[trimmed.length - 1] = nextCloud;
    return trimmed;
  }
  trimmed.push(nextCloud);
  return trimmed.slice(-36);
}

function selectPlaybackScanForTarget(
  history: DecodedPointCloud[],
  targetStampMs: number | null
): DecodedPointCloud | null {
  if (history.length < 1) {
    return null;
  }
  if (!targetStampMs) {
    return history[history.length - 1] ?? null;
  }

  let best: DecodedPointCloud | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = history.length - 1; index >= 0; index -= 1) {
    const cloud = history[index];
    const distance = Math.abs((cloud.stampMs ?? 0) - targetStampMs);
    if (distance <= bestDistance) {
      best = cloud;
      bestDistance = distance;
    }
  }

  if (!best) {
    return null;
  }

  return bestDistance <= 220 ? best : null;
}

function quaternionToYaw(orientation: Pose3D["orientation"]): number {
  const { x, y, z, w } = orientation;
  const siny = 2 * (w * z + x * y);
  const cosy = 1 - 2 * (y * y + z * z);
  return Math.atan2(siny, cosy);
}

function normalizeAngle(angle: number): number {
  let normalized = angle;
  while (normalized > Math.PI) {
    normalized -= Math.PI * 2;
  }
  while (normalized < -Math.PI) {
    normalized += Math.PI * 2;
  }
  return normalized;
}

function accumulatePlaybackCloud(
  voxels: Map<string, PlaybackVoxel>,
  order: string[],
  cloud: DecodedPointCloud,
  options: PlaybackCloudOptions
): DecodedPointCloud {
  const voxelSize = options.voxelSize;
  const maxPoints = options.maxAccumulatedPoints;
  const positions = cloud.positions;
  const colors = cloud.colors;
  const cloudStampMs = cloud.stampMs ?? Date.now();

  for (let index = 0; index < cloud.renderedPointCount; index += 1) {
    const offset = index * 3;
    const x = positions[offset];
    const y = positions[offset + 1];
    const z = positions[offset + 2];
    const ix = Math.round(x / voxelSize);
    const iy = Math.round(y / voxelSize);
    const iz = Math.round(z / voxelSize);
    const key = `${ix}:${iy}:${iz}`;

    const existing = voxels.get(key);
    if (existing) {
      existing.positionSamples += 1;
      const mix = 1 / existing.positionSamples;
      existing.x += (x - existing.x) * mix;
      existing.y += (y - existing.y) * mix;
      existing.z += (z - existing.z) * mix;
      existing.lastSeenStampMs = cloudStampMs;

      if (colors) {
        const nextColor = [colors[offset], colors[offset + 1], colors[offset + 2]] as [
          number,
          number,
          number
        ];
        if (existing.color) {
          existing.colorSamples += 1;
          const colorMix = 1 / existing.colorSamples;
          existing.color[0] += (nextColor[0] - existing.color[0]) * colorMix;
          existing.color[1] += (nextColor[1] - existing.color[1]) * colorMix;
          existing.color[2] += (nextColor[2] - existing.color[2]) * colorMix;
        } else {
          existing.color = nextColor;
          existing.colorSamples = 1;
        }
      }
      continue;
    }

    const color = colors
      ? ([colors[offset], colors[offset + 1], colors[offset + 2]] as [number, number, number])
      : undefined;

    voxels.set(key, {
      ix,
      iy,
      iz,
      x,
      y,
      z,
      positionSamples: 1,
      color,
      colorSamples: color ? 1 : 0,
      lastSeenStampMs: cloudStampMs
    });
    order.push(key);

  }

  trimPlaybackVoxelBudget(voxels, order, maxPoints);

  if (options.decayTimeSec > 0) {
    const cutoff = cloudStampMs - options.decayTimeSec * 1000;
    let writeIndex = 0;
    for (let index = 0; index < order.length; index += 1) {
      const key = order[index];
      const voxel = voxels.get(key);
      if (voxel && voxel.lastSeenStampMs >= cutoff) {
        order[writeIndex] = key;
        writeIndex += 1;
      } else if (voxel) {
        voxels.delete(key);
      }
    }
    order.length = writeIndex;
  }

  const visibleKeys = order.filter((key) => {
    const voxel = voxels.get(key);
    if (!voxel) {
      return false;
    }
    if (voxel.positionSamples >= options.minVoxelObservations) {
      return true;
    }
    return countNeighborVoxels(voxels, voxel) >= 2;
  });

  const accumulatedPositions = new Float32Array(visibleKeys.length * 3);
  let accumulatedColors: Float32Array | undefined;
  let hasAnyColor = false;

  for (let index = 0; index < visibleKeys.length; index += 1) {
    const voxel = voxels.get(visibleKeys[index]);
    if (!voxel) {
      continue;
    }

    const offset = index * 3;
    accumulatedPositions[offset] = voxel.x;
    accumulatedPositions[offset + 1] = voxel.y;
    accumulatedPositions[offset + 2] = voxel.z;

    if (voxel.color) {
      if (!accumulatedColors) {
        accumulatedColors = new Float32Array(visibleKeys.length * 3);
      }
      accumulatedColors[offset] = voxel.color[0];
      accumulatedColors[offset + 1] = voxel.color[1];
      accumulatedColors[offset + 2] = voxel.color[2];
      hasAnyColor = true;
    }
  }

  return {
    frameId: cloud.frameId || "camera_init",
    stampMs: cloud.stampMs,
    sourcePointCount: visibleKeys.length,
    renderedPointCount: visibleKeys.length,
    positions: accumulatedPositions,
    colors: hasAnyColor ? accumulatedColors : undefined
  };
}

function simplifyPlaybackCurrentScan(
  cloud: DecodedPointCloud,
  maxPoints: number,
  voxelSize: number
): DecodedPointCloud {
  if (cloud.renderedPointCount <= 0) {
    return cloud;
  }

  const maxAllowedPoints = Math.max(1, maxPoints);
  const voxelScale = Math.max(voxelSize, 1e-3);
  const positions = cloud.positions;
  const colors = cloud.colors;
  const voxelSeen = new Set<string>();
  const kept: number[] = [];
  const stride = Math.max(1, Math.ceil(cloud.renderedPointCount / maxAllowedPoints));

  for (let index = 0; index < cloud.renderedPointCount; index += stride) {
    const offset = index * 3;
    const key = [
      Math.round(positions[offset] / voxelScale),
      Math.round(positions[offset + 1] / voxelScale),
      Math.round(positions[offset + 2] / voxelScale)
    ].join(":");

    if (voxelSeen.has(key)) {
      continue;
    }
    voxelSeen.add(key);
    kept.push(index);
    if (kept.length >= maxAllowedPoints) {
      break;
    }
  }

  const nextPositions = new Float32Array(kept.length * 3);
  const nextColors = colors ? new Float32Array(kept.length * 3) : undefined;

  for (let writeIndex = 0; writeIndex < kept.length; writeIndex += 1) {
    const readOffset = kept[writeIndex] * 3;
    const writeOffset = writeIndex * 3;
    nextPositions[writeOffset] = positions[readOffset];
    nextPositions[writeOffset + 1] = positions[readOffset + 1];
    nextPositions[writeOffset + 2] = positions[readOffset + 2];

    if (nextColors && colors) {
      nextColors[writeOffset] = colors[readOffset];
      nextColors[writeOffset + 1] = colors[readOffset + 1];
      nextColors[writeOffset + 2] = colors[readOffset + 2];
    }
  }

  return {
    ...cloud,
    sourcePointCount: kept.length,
    renderedPointCount: kept.length,
    positions: nextPositions,
    colors: nextColors
  };
}

function appendPlaybackRgbFrame(
  frames: DecodedRosImage[],
  nextFrame: DecodedRosImage
): DecodedRosImage[] {
  const nextFrames = [...frames, nextFrame];
  return nextFrames.slice(-80);
}

function selectPlaybackRgbFrame(
  frames: DecodedRosImage[],
  stampMs: number,
  toleranceMs = 140
): DecodedRosImage | null {
  if (!frames.length) {
    return null;
  }

  let best: DecodedRosImage | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = frames.length - 1; index >= 0; index -= 1) {
    const frame = frames[index];
    const distance = Math.abs(frame.stampMs - stampMs);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = frame;
    }
    if (distance === 0) {
      break;
    }
  }

  return best && bestDistance <= toleranceMs ? best : null;
}

function countNeighborVoxels(voxels: Map<string, PlaybackVoxel>, voxel: PlaybackVoxel): number {
  let count = 0;
  if (voxels.has(`${voxel.ix + 1}:${voxel.iy}:${voxel.iz}`)) {
    count += 1;
  }
  if (voxels.has(`${voxel.ix - 1}:${voxel.iy}:${voxel.iz}`)) {
    count += 1;
  }
  if (voxels.has(`${voxel.ix}:${voxel.iy + 1}:${voxel.iz}`)) {
    count += 1;
  }
  if (voxels.has(`${voxel.ix}:${voxel.iy - 1}:${voxel.iz}`)) {
    count += 1;
  }
  if (voxels.has(`${voxel.ix}:${voxel.iy}:${voxel.iz + 1}`)) {
    count += 1;
  }
  if (voxels.has(`${voxel.ix}:${voxel.iy}:${voxel.iz - 1}`)) {
    count += 1;
  }
  return count;
}

function trimPlaybackVoxelBudget(
  voxels: Map<string, PlaybackVoxel>,
  order: string[],
  maxPoints: number
): void {
  if (order.length <= maxPoints) {
    return;
  }

  let scanStart = 0;
  while (order.length > maxPoints) {
    let removed = false;
    const scanEnd = Math.min(order.length, scanStart + 512);
    for (let index = scanStart; index < scanEnd; index += 1) {
      const key = order[index];
      const voxel = voxels.get(key);
      if (!voxel) {
        continue;
      }
      if (
        voxel.positionSamples >= 2 ||
        countNeighborVoxels(voxels, voxel) >= 2
      ) {
        continue;
      }
      voxels.delete(key);
      order.splice(index, 1);
      removed = true;
      break;
    }

    if (removed) {
      continue;
    }

    const evicted = order.shift();
    if (evicted) {
      voxels.delete(evicted);
    } else {
      break;
    }
    scanStart = 0;
  }
}
