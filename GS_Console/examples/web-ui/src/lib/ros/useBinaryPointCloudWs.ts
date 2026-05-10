import { useEffect, useState } from "react";
import type { DecodedPointCloud } from "./pointCloud2";

const MAGIC = "LBPC1";

export interface BinaryPointCloudWsState {
  cloud: DecodedPointCloud | null;
  connectionState: "idle" | "connecting" | "connected" | "closed" | "error";
  errorMessage: string | null;
}

export function useBinaryPointCloudWs(url: string | null, fallbackUrl?: string | null): BinaryPointCloudWsState {
  const [cloud, setCloud] = useState<DecodedPointCloud | null>(null);
  const [connectionState, setConnectionState] =
    useState<BinaryPointCloudWsState["connectionState"]>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!url) {
      setCloud(null);
      setConnectionState("idle");
      setErrorMessage(null);
      return;
    }

    let disposed = false;
    let reconnectTimer = 0;
    let socket: WebSocket | null = null;

    const connect = () => {
      if (disposed) {
        return;
      }
      setConnectionState("connecting");
      socket = new WebSocket(url);
      socket.binaryType = "arraybuffer";
      socket.onopen = () => {
        if (!disposed) {
          setConnectionState("connected");
          setErrorMessage(null);
        }
      };
      socket.onmessage = (event) => {
        if (disposed || !(event.data instanceof ArrayBuffer)) {
          return;
        }
        try {
          setCloud(decodeBinaryPointCloud(event.data));
        } catch (error) {
          setErrorMessage(error instanceof Error ? error.message : "Failed to decode binary cloud.");
        }
      };
      socket.onerror = () => {
        if (!disposed) {
          setConnectionState("error");
          setErrorMessage("Binary cloud socket unavailable.");
        }
      };
      socket.onclose = () => {
        if (disposed) {
          return;
        }
        setConnectionState("closed");
        reconnectTimer = window.setTimeout(connect, 1200);
      };
    };

    connect();
    return () => {
      disposed = true;
      window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [url]);

  useEffect(() => {
    if (!fallbackUrl) {
      return;
    }

    let disposed = false;
    let interval = 0;

    const fetchFallbackCloud = async () => {
      if (disposed || (connectionState === "connected" && cloud)) {
        return;
      }
      try {
        const response = await fetch(fallbackUrl, { cache: "no-store" });
        if (!response.ok) {
          return;
        }
        const buffer = await response.arrayBuffer();
        if (disposed) {
          return;
        }
        setCloud(decodeBinaryPointCloud(buffer));
        setErrorMessage(null);
      } catch (error) {
        if (!disposed) {
          setErrorMessage(error instanceof Error ? error.message : "Binary cloud fallback unavailable.");
        }
      }
    };

    fetchFallbackCloud();
    interval = window.setInterval(fetchFallbackCloud, 650);
    return () => {
      disposed = true;
      window.clearInterval(interval);
    };
  }, [cloud, connectionState, fallbackUrl]);

  return { cloud, connectionState, errorMessage };
}

function decodeBinaryPointCloud(buffer: ArrayBuffer): DecodedPointCloud {
  const view = new DataView(buffer);
  if (buffer.byteLength < 9) {
    throw new Error("Binary cloud frame is too small.");
  }
  const magic = new TextDecoder().decode(buffer.slice(0, 5));
  if (magic !== MAGIC) {
    throw new Error("Binary cloud magic mismatch.");
  }
  const headerLength = view.getUint32(5, true);
  const headerStart = 9;
  const headerEnd = headerStart + headerLength;
  if (headerEnd > buffer.byteLength) {
    throw new Error("Binary cloud header is truncated.");
  }
  const header = JSON.parse(new TextDecoder().decode(buffer.slice(headerStart, headerEnd))) as {
    frameId?: string;
    stampMs?: number;
    sourcePointCount?: number;
    renderedPointCount?: number;
    xyzBytes?: number;
    rgbBytes?: number;
  };
  const count = Math.max(0, Math.floor(header.renderedPointCount ?? 0));
  const xyzBytes = header.xyzBytes ?? count * 12;
  const rgbBytes = header.rgbBytes ?? count * 3;
  const xyzStart = headerEnd;
  const rgbStart = xyzStart + xyzBytes;
  const rgbEnd = rgbStart + rgbBytes;
  if (rgbEnd > buffer.byteLength || xyzBytes < count * 12 || rgbBytes < count * 3) {
    throw new Error("Binary cloud payload is truncated.");
  }

  const positions = new Float32Array(buffer.slice(xyzStart, xyzStart + count * 12));
  const rgb = new Uint8Array(buffer, rgbStart, count * 3);
  const colors = new Float32Array(count * 3);
  for (let index = 0; index < rgb.length; index += 1) {
    colors[index] = rgb[index] / 255;
  }

  return {
    frameId: header.frameId ?? "map",
    stampMs: header.stampMs ?? Date.now(),
    receivedAtMs: Date.now(),
    sourcePointCount: header.sourcePointCount ?? count,
    renderedPointCount: count,
    positions,
    colors
  };
}
