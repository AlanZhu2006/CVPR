import { BufferAttribute, BufferGeometry } from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import type { Point3 } from "../map/occupancyProjection";

const loader = new PLYLoader();

export async function loadPlyGeometry(url: string): Promise<BufferGeometry> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch PLY ${url}: ${response.status}`);
  }

  const buffer = await response.arrayBuffer();
  return loader.parse(buffer);
}

export function geometryToPoints(geometry: BufferGeometry): Point3[] {
  const position = geometry.getAttribute("position");
  if (!(position instanceof BufferAttribute)) {
    return [];
  }

  const points: Point3[] = [];
  for (let index = 0; index < position.count; index += 1) {
    points.push({
      x: position.getX(index),
      y: position.getY(index),
      z: position.getZ(index)
    });
  }

  return points;
}
