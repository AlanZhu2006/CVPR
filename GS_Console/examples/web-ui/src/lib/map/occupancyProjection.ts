export type OccupancyValue = -1 | 0 | 100;

export interface Point3 {
  x: number;
  y: number;
  z: number;
}

export interface Bounds3 {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  minZ: number;
  maxZ: number;
}

export interface OccupancyProjectionOptions {
  resolution: number;
  bounds: Bounds3;
  obstacleBand: {
    minZ: number;
    maxZ: number;
  };
  robotRadius: number;
}

export interface OccupancyGrid {
  width: number;
  height: number;
  resolution: number;
  origin: {
    x: number;
    y: number;
  };
  data: Int8Array;
  hits: Uint16Array;
}

export interface CellIndex {
  x: number;
  y: number;
}

export function createGrid(options: OccupancyProjectionOptions): OccupancyGrid {
  const width = Math.ceil((options.bounds.maxX - options.bounds.minX) / options.resolution);
  const height = Math.ceil((options.bounds.maxY - options.bounds.minY) / options.resolution);

  const data = new Int8Array(width * height);
  data.fill(0);
  const hits = new Uint16Array(width * height);

  return {
    width,
    height,
    resolution: options.resolution,
    origin: {
      x: options.bounds.minX,
      y: options.bounds.minY
    },
    data,
    hits
  };
}

export function worldToCell(grid: OccupancyGrid, x: number, y: number): CellIndex | null {
  const cellX = Math.floor((x - grid.origin.x) / grid.resolution);
  const cellY = Math.floor((y - grid.origin.y) / grid.resolution);

  if (cellX < 0 || cellY < 0 || cellX >= grid.width || cellY >= grid.height) {
    return null;
  }

  return { x: cellX, y: cellY };
}

export function cellToLinearIndex(grid: OccupancyGrid, cell: CellIndex): number {
  return cell.y * grid.width + cell.x;
}

export function projectOccPriorToGrid(
  points: Point3[],
  options: OccupancyProjectionOptions
): OccupancyGrid {
  const grid = createGrid(options);

  for (const point of points) {
    if (point.z < options.obstacleBand.minZ || point.z > options.obstacleBand.maxZ) {
      continue;
    }

    const cell = worldToCell(grid, point.x, point.y);
    if (!cell) {
      continue;
    }

    const index = cellToLinearIndex(grid, cell);
    grid.hits[index] = Math.min(grid.hits[index] + 1, 65535);
    grid.data[index] = 100;
  }

  dilateOccupiedCells(grid, options.robotRadius);
  return grid;
}

export function occupancyValueAtWorld(
  grid: OccupancyGrid,
  x: number,
  y: number
): OccupancyValue | null {
  const cell = worldToCell(grid, x, y);
  if (!cell) {
    return null;
  }
  return grid.data[cellToLinearIndex(grid, cell)] as OccupancyValue;
}

function dilateOccupiedCells(grid: OccupancyGrid, robotRadius: number): void {
  const radiusCells = Math.ceil(robotRadius / grid.resolution);
  if (radiusCells <= 0) {
    return;
  }

  const source = new Int8Array(grid.data);

  for (let y = 0; y < grid.height; y += 1) {
    for (let x = 0; x < grid.width; x += 1) {
      const index = y * grid.width + x;
      if (source[index] !== 100) {
        continue;
      }
      rasterizeDisc(grid, x, y, radiusCells);
    }
  }
}

function rasterizeDisc(grid: OccupancyGrid, cx: number, cy: number, radiusCells: number): void {
  const radiusSquared = radiusCells * radiusCells;

  for (let y = cy - radiusCells; y <= cy + radiusCells; y += 1) {
    if (y < 0 || y >= grid.height) {
      continue;
    }

    for (let x = cx - radiusCells; x <= cx + radiusCells; x += 1) {
      if (x < 0 || x >= grid.width) {
        continue;
      }

      const dx = x - cx;
      const dy = y - cy;
      if (dx * dx + dy * dy > radiusSquared) {
        continue;
      }

      grid.data[y * grid.width + x] = 100;
    }
  }
}
