export interface PointField {
  name: string;
  offset: number;
  datatype: number;
  count: number;
}

export interface PointCloud2Message {
  header?: {
    frame_id?: string;
    stamp?: {
      secs?: number;
      nsecs?: number;
    };
  };
  height?: number;
  width?: number;
  fields?: PointField[];
  is_bigendian?: boolean;
  point_step?: number;
  row_step?: number;
  data?: string | number[];
  is_dense?: boolean;
}

export interface DecodedPointCloud {
  frameId: string;
  stampMs: number;
  receivedAtMs?: number;
  sourcePointCount: number;
  renderedPointCount: number;
  positions: Float32Array;
  colors?: Float32Array;
}

interface DecodeOptions {
  maxPoints?: number;
}

const INT8 = 1;
const UINT8 = 2;
const INT16 = 3;
const UINT16 = 4;
const INT32 = 5;
const UINT32 = 6;
const FLOAT32 = 7;
const FLOAT64 = 8;

export function decodePointCloud2(
  message: PointCloud2Message,
  options: DecodeOptions = {}
): DecodedPointCloud {
  const width = message.width ?? 0;
  const height = message.height ?? 1;
  const sourcePointCount = width * height;
  const pointStep = message.point_step ?? 0;
  const fields = message.fields ?? [];
  const frameId = message.header?.frame_id ?? "world";
  const secs = message.header?.stamp?.secs ?? 0;
  const nsecs = message.header?.stamp?.nsecs ?? 0;
  const stampMs = secs * 1000 + Math.round(nsecs / 1_000_000);
  const receivedAtMs = Date.now();

  if (!sourcePointCount || !pointStep || !message.data) {
    return {
      frameId,
      stampMs,
      receivedAtMs,
      sourcePointCount: 0,
      renderedPointCount: 0,
      positions: new Float32Array()
    };
  }

  const xField = fields.find((field) => field.name === "x" && field.datatype === FLOAT32);
  const yField = fields.find((field) => field.name === "y" && field.datatype === FLOAT32);
  const zField = fields.find((field) => field.name === "z" && field.datatype === FLOAT32);
  const packedRgbField = findField(fields, ["rgb", "rgba"]);
  const redField = findField(fields, ["r", "red"]);
  const greenField = findField(fields, ["g", "green"]);
  const blueField = findField(fields, ["b", "blue"]);
  const intensityField = findField(fields, ["intensity"]);

  if (!xField || !yField || !zField) {
    throw new Error("PointCloud2 message is missing float32 x/y/z fields.");
  }

  const raw = decodeBinary(message.data);
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const littleEndian = !message.is_bigendian;
  const maxPoints = options.maxPoints ?? 120_000;
  const stride = Math.max(1, Math.ceil(sourcePointCount / maxPoints));
  const sampleCount = Math.ceil(sourcePointCount / stride);
  const positions = new Float32Array(sampleCount * 3);
  const hasColorFields = Boolean(
    packedRgbField || (redField && greenField && blueField) || intensityField
  );
  const colors = hasColorFields ? new Float32Array(sampleCount * 3) : undefined;

  let renderedIndex = 0;

  for (let pointIndex = 0; pointIndex < sourcePointCount; pointIndex += stride) {
    const base = pointIndex * pointStep;
    const x = view.getFloat32(base + xField.offset, littleEndian);
    const y = view.getFloat32(base + yField.offset, littleEndian);
    const z = view.getFloat32(base + zField.offset, littleEndian);

    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      continue;
    }

    const offset = renderedIndex * 3;
    positions[offset] = x;
    positions[offset + 1] = y;
    positions[offset + 2] = z;

    if (colors) {
      const color = readPointColor(view, base, littleEndian, {
        packedRgbField,
        redField,
        greenField,
        blueField,
        intensityField
      });
      if (color) {
        colors[offset] = color[0];
        colors[offset + 1] = color[1];
        colors[offset + 2] = color[2];
      }
    }

    renderedIndex += 1;
  }

  return {
    frameId,
    stampMs,
    receivedAtMs,
    sourcePointCount,
    renderedPointCount: renderedIndex,
    positions: renderedIndex === sampleCount ? positions : positions.slice(0, renderedIndex * 3),
    colors: colors
      ? renderedIndex === sampleCount
        ? colors
        : colors.slice(0, renderedIndex * 3)
      : undefined
  };
}

interface PointColorFields {
  packedRgbField?: PointField;
  redField?: PointField;
  greenField?: PointField;
  blueField?: PointField;
  intensityField?: PointField;
}

function findField(fields: PointField[], names: string[]): PointField | undefined {
  const expected = new Set(names.map((name) => name.toLowerCase()));
  return fields.find((field) => expected.has(field.name.toLowerCase()));
}

function readPointColor(
  view: DataView,
  base: number,
  littleEndian: boolean,
  fields: PointColorFields
): [number, number, number] | null {
  if (fields.packedRgbField) {
    const packed = readPackedRgb(view, base, fields.packedRgbField, littleEndian);
    if (packed !== null) {
      return [
        ((packed >> 16) & 0xff) / 255,
        ((packed >> 8) & 0xff) / 255,
        (packed & 0xff) / 255
      ];
    }
  }

  if (fields.redField && fields.greenField && fields.blueField) {
    const red = readScalar(view, base + fields.redField.offset, fields.redField.datatype, littleEndian);
    const green = readScalar(
      view,
      base + fields.greenField.offset,
      fields.greenField.datatype,
      littleEndian
    );
    const blue = readScalar(
      view,
      base + fields.blueField.offset,
      fields.blueField.datatype,
      littleEndian
    );

    if (red !== null && green !== null && blue !== null) {
      return [normalizeChannel(red), normalizeChannel(green), normalizeChannel(blue)];
    }
  }

  if (fields.intensityField) {
    const intensity = readScalar(
      view,
      base + fields.intensityField.offset,
      fields.intensityField.datatype,
      littleEndian
    );
    if (intensity !== null) {
      return colorizeIntensity(normalizeChannel(intensity));
    }
  }

  return null;
}

function readPackedRgb(
  view: DataView,
  base: number,
  field: PointField,
  littleEndian: boolean
): number | null {
  const offset = base + field.offset;
  try {
    if (field.datatype === FLOAT32 || field.datatype === UINT32) {
      return view.getUint32(offset, littleEndian);
    }
    if (field.datatype === INT32) {
      return view.getInt32(offset, littleEndian) >>> 0;
    }
  } catch {
    return null;
  }
  return null;
}

function readScalar(
  view: DataView,
  offset: number,
  datatype: number,
  littleEndian: boolean
): number | null {
  try {
    switch (datatype) {
      case INT8:
        return view.getInt8(offset);
      case UINT8:
        return view.getUint8(offset);
      case INT16:
        return view.getInt16(offset, littleEndian);
      case UINT16:
        return view.getUint16(offset, littleEndian);
      case INT32:
        return view.getInt32(offset, littleEndian);
      case UINT32:
        return view.getUint32(offset, littleEndian);
      case FLOAT32:
        return view.getFloat32(offset, littleEndian);
      case FLOAT64:
        return view.getFloat64(offset, littleEndian);
      default:
        return null;
    }
  } catch {
    return null;
  }
}

function normalizeChannel(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  const normalized = value <= 1 && value >= 0 ? value : value / 255;
  return Math.max(0, Math.min(1, normalized));
}

function colorizeIntensity(value: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, Math.sqrt(value)));

  if (t < 0.33) {
    const local = t / 0.33;
    return [
      0.05 * (1 - local) + 0.08 * local,
      0.18 * (1 - local) + 0.62 * local,
      0.55 * (1 - local) + 0.92 * local
    ];
  }

  if (t < 0.66) {
    const local = (t - 0.33) / 0.33;
    return [
      0.08 * (1 - local) + 0.39 * local,
      0.62 * (1 - local) + 0.9 * local,
      0.92 * (1 - local) + 0.22 * local
    ];
  }

  const local = (t - 0.66) / 0.34;
  return [
    0.39 * (1 - local) + 0.95 * local,
    0.9 * (1 - local) + 0.98 * local,
    0.22 * (1 - local) + 0.18 * local
  ];
}

function decodeBinary(data: string | number[]): Uint8Array {
  if (Array.isArray(data)) {
    return Uint8Array.from(data);
  }

  const normalized = normalizeBase64(data);
  const binary = atob(normalized);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function normalizeBase64(input: string): string {
  const padding = input.length % 4;
  if (padding === 0) {
    return input;
  }
  return input + "=".repeat(4 - padding);
}
