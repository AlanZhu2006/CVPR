import type { Pose3D } from "../gs/gsSdfSceneAdapter";
import type { DecodedPointCloud } from "./pointCloud2";
import type { DecodedRosImage } from "./rosImage";

export interface PlaybackProjectionCalibration {
  camera: {
    width: number;
    height: number;
    fx: number;
    fy: number;
    cx: number;
    cy: number;
  };
  extrinsic: {
    tCL: number[];
    tBL: number[];
  };
}

export const defaultPlaybackCalibration: PlaybackProjectionCalibration = {
  // Mirrors config/fastlivo_cbd_host.yaml for the current FAST-LIVO2 bag.
  camera: {
    width: 640,
    height: 512,
    fx: 588.143714607,
    fy: 588.107927227,
    cx: 296.059369138,
    cy: 254.543215481
  },
  extrinsic: {
    tCL: [
      -0.002, -0.99975, -0.02211, 0.0026,
      -0.00366, 0.02212, -0.99975, 0.05057,
      0.99999, -0.00192, -0.00371, -0.00587,
      0.0, 0.0, 0.0, 1.0
    ],
    tBL: [
      1.0, 0.0, 0.0, 0.04165,
      0.0, 1.0, 0.0, 0.02326,
      0.0, 0.0, 1.0, -0.0284,
      0.0, 0.0, 0.0, 1.0
    ]
  }
};

const PLAYBACK_RGB_SYNC_WINDOW_MS = 40;

export function projectPlaybackCloudToWorld(
  cloud: DecodedPointCloud,
  poseWB: Pose3D | null,
  rgbFrame: DecodedRosImage | null,
  calibration: PlaybackProjectionCalibration = defaultPlaybackCalibration
): DecodedPointCloud {
  if (!poseWB || cloud.renderedPointCount === 0) {
    return cloud;
  }

  const positions = new Float32Array(cloud.renderedPointCount * 3);
  const colors = new Float32Array(cloud.renderedPointCount * 3);
  const sourceColors = cloud.colors;
  const tWL = multiplyMat4(poseToMat4(poseWB), calibration.extrinsic.tBL);
  const tCL = calibration.extrinsic.tCL;
  const useRgbFrame =
    rgbFrame &&
    Math.abs((rgbFrame.stampMs ?? 0) - (cloud.stampMs ?? 0)) <= PLAYBACK_RGB_SYNC_WINDOW_MS &&
    rgbFrame.width > 0 &&
    rgbFrame.height > 0
      ? rgbFrame
      : null;

  for (let index = 0; index < cloud.renderedPointCount; index += 1) {
    const offset = index * 3;
    const xL = cloud.positions[offset];
    const yL = cloud.positions[offset + 1];
    const zL = cloud.positions[offset + 2];

    const world = transformPoint(tWL, xL, yL, zL);
    positions[offset] = world[0];
    positions[offset + 1] = world[1];
    positions[offset + 2] = world[2];

    const sampledColor = useRgbFrame
      ? sampleProjectedColor(useRgbFrame, calibration.camera, tCL, xL, yL, zL)
      : null;
    if (sampledColor) {
      colors[offset] = sampledColor[0];
      colors[offset + 1] = sampledColor[1];
      colors[offset + 2] = sampledColor[2];
      continue;
    }

    if (sourceColors) {
      colors[offset] = sourceColors[offset];
      colors[offset + 1] = sourceColors[offset + 1];
      colors[offset + 2] = sourceColors[offset + 2];
      continue;
    }

    colors[offset] = 0.8;
    colors[offset + 1] = 0.84;
    colors[offset + 2] = 0.9;
  }

  return {
    frameId: poseWB.frameId || "camera_init",
    stampMs: cloud.stampMs,
    sourcePointCount: cloud.sourcePointCount,
    renderedPointCount: cloud.renderedPointCount,
    positions,
    colors
  };
}

function sampleProjectedColor(
  frame: DecodedRosImage,
  camera: PlaybackProjectionCalibration["camera"],
  tCL: number[],
  xL: number,
  yL: number,
  zL: number
): [number, number, number] | null {
  const cameraPoint = transformPoint(tCL, xL, yL, zL);
  const zC = cameraPoint[2];
  if (!Number.isFinite(zC) || zC <= 0.05) {
    return null;
  }

  const u = camera.fx * (cameraPoint[0] / zC) + camera.cx;
  const v = camera.fy * (cameraPoint[1] / zC) + camera.cy;
  const pixelX = Math.floor(u);
  const pixelY = Math.floor(v);

  if (pixelX < 0 || pixelY < 0 || pixelX >= frame.width - 1 || pixelY >= frame.height - 1) {
    return null;
  }

  const tx = Math.min(Math.max(u - pixelX, 0), 1);
  const ty = Math.min(Math.max(v - pixelY, 0), 1);

  const c00 = readFrameRgb(frame, pixelX, pixelY);
  const c10 = readFrameRgb(frame, pixelX + 1, pixelY);
  const c01 = readFrameRgb(frame, pixelX, pixelY + 1);
  const c11 = readFrameRgb(frame, pixelX + 1, pixelY + 1);

  return [
    bilerp(c00[0], c10[0], c01[0], c11[0], tx, ty),
    bilerp(c00[1], c10[1], c01[1], c11[1], tx, ty),
    bilerp(c00[2], c10[2], c01[2], c11[2], tx, ty)
  ];
}

function poseToMat4(pose: Pose3D): number[] {
  const { x, y, z, w } = pose.orientation;
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const wx = w * x;
  const wy = w * y;
  const wz = w * z;

  return [
    1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), pose.position.x,
    2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), pose.position.y,
    2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), pose.position.z,
    0, 0, 0, 1
  ];
}

function multiplyMat4(a: number[], b: number[]): number[] {
  const out = new Array<number>(16).fill(0);
  for (let row = 0; row < 4; row += 1) {
    for (let col = 0; col < 4; col += 1) {
      for (let k = 0; k < 4; k += 1) {
        out[row * 4 + col] += a[row * 4 + k] * b[k * 4 + col];
      }
    }
  }
  return out;
}

function transformPoint(m: number[], x: number, y: number, z: number): [number, number, number] {
  return [
    m[0] * x + m[1] * y + m[2] * z + m[3],
    m[4] * x + m[5] * y + m[6] * z + m[7],
    m[8] * x + m[9] * y + m[10] * z + m[11]
  ];
}

function readFrameRgb(frame: DecodedRosImage, x: number, y: number): [number, number, number] {
  const offset = (y * frame.width + x) * 4;
  return [
    (frame.rgba[offset] ?? 0) / 255,
    (frame.rgba[offset + 1] ?? 0) / 255,
    (frame.rgba[offset + 2] ?? 0) / 255
  ];
}

function bilerp(c00: number, c10: number, c01: number, c11: number, tx: number, ty: number): number {
  const top = c00 * (1 - tx) + c10 * tx;
  const bottom = c01 * (1 - tx) + c11 * tx;
  return top * (1 - ty) + bottom * ty;
}
