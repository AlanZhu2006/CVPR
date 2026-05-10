export interface RosImageMessage {
  header?: {
    stamp?: {
      secs?: number;
      nsecs?: number;
    };
  };
  width?: number;
  height?: number;
  encoding?: string;
  step?: number;
  data?: string | number[] | Uint8Array;
}

export interface DecodedRosImage {
  width: number;
  height: number;
  encoding: string;
  stampMs: number;
  rgba: Uint8ClampedArray;
}

export function decodeRosImage(message: RosImageMessage): DecodedRosImage {
  const width = message.width ?? 0;
  const height = message.height ?? 0;
  const encoding = (message.encoding ?? "").toLowerCase();
  if (width <= 0 || height <= 0) {
    throw new Error("ROS image message is missing width or height.");
  }

  const bytes = toUint8Array(message.data);
  const rgba = new Uint8ClampedArray(width * height * 4);

  switch (encoding) {
    case "bgr8":
      writeColorImage(rgba, bytes, width, height, message.step ?? width * 3, "bgr");
      break;
    case "rgb8":
      writeColorImage(rgba, bytes, width, height, message.step ?? width * 3, "rgb");
      break;
    case "mono8":
      writeMono8Image(rgba, bytes, width, height, message.step ?? width);
      break;
    case "mono16":
      writeMono16Image(rgba, bytes, width, height, message.step ?? width * 2);
      break;
    default:
      throw new Error(`Unsupported ROS image encoding: ${message.encoding ?? "unknown"}`);
  }
  const secs = message.header?.stamp?.secs ?? 0;
  const nsecs = message.header?.stamp?.nsecs ?? 0;
  return {
    width,
    height,
    encoding,
    stampMs: secs * 1000 + Math.round(nsecs / 1_000_000),
    rgba
  };
}

function toUint8Array(data: RosImageMessage["data"]): Uint8Array {
  if (!data) {
    return new Uint8Array();
  }
  if (data instanceof Uint8Array) {
    return data;
  }
  if (Array.isArray(data)) {
    return Uint8Array.from(data);
  }
  if (typeof data === "string") {
    const decoded = atob(data);
    const bytes = new Uint8Array(decoded.length);
    for (let index = 0; index < decoded.length; index += 1) {
      bytes[index] = decoded.charCodeAt(index);
    }
    return bytes;
  }
  throw new Error("Unsupported ROS image payload.");
}

function writeColorImage(
  rgba: Uint8ClampedArray,
  bytes: Uint8Array,
  width: number,
  height: number,
  step: number,
  order: "rgb" | "bgr"
): void {
  for (let y = 0; y < height; y += 1) {
    const rowStart = y * step;
    for (let x = 0; x < width; x += 1) {
      const source = rowStart + x * 3;
      const target = (y * width + x) * 4;
      const first = bytes[source] ?? 0;
      const second = bytes[source + 1] ?? 0;
      const third = bytes[source + 2] ?? 0;
      rgba[target] = order === "rgb" ? first : third;
      rgba[target + 1] = second;
      rgba[target + 2] = order === "rgb" ? third : first;
      rgba[target + 3] = 255;
    }
  }
}

function writeMono8Image(
  rgba: Uint8ClampedArray,
  bytes: Uint8Array,
  width: number,
  height: number,
  step: number
): void {
  for (let y = 0; y < height; y += 1) {
    const rowStart = y * step;
    for (let x = 0; x < width; x += 1) {
      const value = bytes[rowStart + x] ?? 0;
      const target = (y * width + x) * 4;
      rgba[target] = value;
      rgba[target + 1] = value;
      rgba[target + 2] = value;
      rgba[target + 3] = 255;
    }
  }
}

function writeMono16Image(
  rgba: Uint8ClampedArray,
  bytes: Uint8Array,
  width: number,
  height: number,
  step: number
): void {
  for (let y = 0; y < height; y += 1) {
    const rowStart = y * step;
    for (let x = 0; x < width; x += 1) {
      const source = rowStart + x * 2;
      const raw = (bytes[source] ?? 0) | ((bytes[source + 1] ?? 0) << 8);
      const value = Math.max(0, Math.min(255, Math.round(raw / 256)));
      const target = (y * width + x) * 4;
      rgba[target] = value;
      rgba[target + 1] = value;
      rgba[target + 2] = value;
      rgba[target + 3] = 255;
    }
  }
}
