import type { OccupancyGrid } from "./occupancyProjection";
import { cellToLinearIndex, worldToCell } from "./occupancyProjection";
import type { GoalPose2D } from "../ros/navGoalPublisher";
import type { ObstacleRect2D } from "../../components/TopDownMap";

export interface PlanPoint2D {
  x: number;
  y: number;
}

export interface LocalPlanResult {
  status: "idle" | "ready" | "blocked";
  path: PlanPoint2D[];
  reason?: string;
}

interface Cell {
  x: number;
  y: number;
}

interface OpenNode extends Cell {
  index: number;
  priority: number;
}

const neighborOffsets = [
  { x: -1, y: -1, cost: Math.SQRT2 },
  { x: 0, y: -1, cost: 1 },
  { x: 1, y: -1, cost: Math.SQRT2 },
  { x: -1, y: 0, cost: 1 },
  { x: 1, y: 0, cost: 1 },
  { x: -1, y: 1, cost: Math.SQRT2 },
  { x: 0, y: 1, cost: 1 },
  { x: 1, y: 1, cost: Math.SQRT2 }
];

export function planPathOnGrid(input: {
  grid: OccupancyGrid | null;
  start: PlanPoint2D | null;
  goal: GoalPose2D | null;
  obstacles?: ObstacleRect2D[];
  robotRadius?: number;
}): LocalPlanResult {
  const { grid, start, goal, obstacles = [], robotRadius = 0 } = input;
  if (!grid || !start || !goal) {
    return { status: "idle", path: [] };
  }

  const occupancy = new Uint8Array(grid.data.length);
  for (let index = 0; index < grid.data.length; index += 1) {
    occupancy[index] = grid.data[index] === 100 ? 1 : 0;
  }
  applyObstacleRects(occupancy, grid, obstacles);
  inflateOccupancy(occupancy, grid, robotRadius);

  const rawStart = worldToCell(grid, start.x, start.y);
  const rawGoal = worldToCell(grid, goal.x, goal.y);
  if (!rawStart || !rawGoal) {
    return { status: "blocked", path: [], reason: "Goal or start lies outside the map bounds." };
  }

  const startCell = findNearestFreeCell(rawStart, occupancy, grid, 18);
  const goalCell = findNearestFreeCell(rawGoal, occupancy, grid, 18);
  if (!startCell || !goalCell) {
    return { status: "blocked", path: [], reason: "Could not find a free start or goal cell." };
  }

  const route = runAStar(startCell, goalCell, occupancy, grid);
  if (!route.length) {
    return { status: "blocked", path: [], reason: "No collision-free path found on the occupancy grid." };
  }

  const smoothed = smoothCellPath(route, occupancy, grid);
  const path = smoothed.map((cell) => cellCenterToWorld(grid, cell));
  if (path.length > 0) {
    path[0] = isFreeCell(rawStart, occupancy, grid) ? { x: start.x, y: start.y } : cellCenterToWorld(grid, startCell);
    path[path.length - 1] = isFreeCell(rawGoal, occupancy, grid) ? { x: goal.x, y: goal.y } : cellCenterToWorld(grid, goalCell);
  }

  return { status: "ready", path };
}

function isFreeCell(cell: Cell, occupancy: Uint8Array, grid: OccupancyGrid): boolean {
  return occupancy[cellToLinearIndex(grid, cell)] === 0;
}

function inflateOccupancy(occupancy: Uint8Array, grid: OccupancyGrid, robotRadius: number): void {
  const radiusCells = Math.ceil(Math.max(0, robotRadius) / grid.resolution);
  if (radiusCells <= 0) {
    return;
  }

  const occupiedCells: Cell[] = [];
  for (let y = 0; y < grid.height; y += 1) {
    for (let x = 0; x < grid.width; x += 1) {
      if (occupancy[y * grid.width + x]) {
        occupiedCells.push({ x, y });
      }
    }
  }

  for (const cell of occupiedCells) {
    for (let dy = -radiusCells; dy <= radiusCells; dy += 1) {
      for (let dx = -radiusCells; dx <= radiusCells; dx += 1) {
        if (dx * dx + dy * dy > radiusCells * radiusCells) {
          continue;
        }
        const x = cell.x + dx;
        const y = cell.y + dy;
        if (x < 0 || y < 0 || x >= grid.width || y >= grid.height) {
          continue;
        }
        occupancy[y * grid.width + x] = 1;
      }
    }
  }
}

function applyObstacleRects(
  occupancy: Uint8Array,
  grid: OccupancyGrid,
  obstacles: ObstacleRect2D[]
): void {
  for (const obstacle of obstacles) {
    const min = worldToCell(grid, obstacle.minX, obstacle.minY) ?? { x: 0, y: 0 };
    const max = worldToCell(grid, obstacle.maxX, obstacle.maxY) ?? {
      x: grid.width - 1,
      y: grid.height - 1
    };

    const minX = Math.max(0, Math.min(min.x, max.x));
    const maxX = Math.min(grid.width - 1, Math.max(min.x, max.x));
    const minY = Math.max(0, Math.min(min.y, max.y));
    const maxY = Math.min(grid.height - 1, Math.max(min.y, max.y));

    for (let y = minY; y <= maxY; y += 1) {
      for (let x = minX; x <= maxX; x += 1) {
        occupancy[y * grid.width + x] = 1;
      }
    }
  }
}

function runAStar(
  start: Cell,
  goal: Cell,
  occupancy: Uint8Array,
  grid: OccupancyGrid
): Cell[] {
  const total = grid.width * grid.height;
  const gScore = new Float32Array(total);
  gScore.fill(Number.POSITIVE_INFINITY);
  const cameFrom = new Int32Array(total);
  cameFrom.fill(-1);
  const closed = new Uint8Array(total);

  const startIndex = cellToLinearIndex(grid, start);
  const goalIndex = cellToLinearIndex(grid, goal);
  const open = new MinHeap();

  gScore[startIndex] = 0;
  open.push({
    ...start,
    index: startIndex,
    priority: heuristic(start, goal)
  });

  while (!open.isEmpty()) {
    const current = open.pop();
    if (!current) {
      break;
    }
    if (closed[current.index]) {
      continue;
    }
    closed[current.index] = 1;

    if (current.index === goalIndex) {
      return reconstructPath(cameFrom, goalIndex, grid);
    }

    for (const neighbor of neighborOffsets) {
      const nextX = current.x + neighbor.x;
      const nextY = current.y + neighbor.y;
      if (nextX < 0 || nextY < 0 || nextX >= grid.width || nextY >= grid.height) {
        continue;
      }

      if (neighbor.x !== 0 && neighbor.y !== 0) {
        const sideA = current.y * grid.width + nextX;
        const sideB = nextY * grid.width + current.x;
        if (occupancy[sideA] || occupancy[sideB]) {
          continue;
        }
      }

      const nextIndex = nextY * grid.width + nextX;
      if (closed[nextIndex] || occupancy[nextIndex]) {
        continue;
      }

      const tentativeScore = gScore[current.index] + neighbor.cost;
      if (tentativeScore >= gScore[nextIndex]) {
        continue;
      }

      cameFrom[nextIndex] = current.index;
      gScore[nextIndex] = tentativeScore;
      open.push({
        x: nextX,
        y: nextY,
        index: nextIndex,
        priority: tentativeScore + heuristic({ x: nextX, y: nextY }, goal)
      });
    }
  }

  return [];
}

function reconstructPath(cameFrom: Int32Array, goalIndex: number, grid: OccupancyGrid): Cell[] {
  const path: Cell[] = [];
  let current = goalIndex;

  while (current >= 0) {
    path.push({
      x: current % grid.width,
      y: Math.floor(current / grid.width)
    });
    current = cameFrom[current];
  }

  path.reverse();
  return path;
}

function smoothCellPath(path: Cell[], occupancy: Uint8Array, grid: OccupancyGrid): Cell[] {
  if (path.length <= 2) {
    return path;
  }

  const smoothed: Cell[] = [path[0]];
  let anchor = 0;

  while (anchor < path.length - 1) {
    let furthest = anchor + 1;
    for (let candidate = path.length - 1; candidate > anchor + 1; candidate -= 1) {
      if (hasLineOfSight(path[anchor], path[candidate], occupancy, grid)) {
        furthest = candidate;
        break;
      }
    }
    smoothed.push(path[furthest]);
    anchor = furthest;
  }

  return smoothed;
}

function hasLineOfSight(
  start: Cell,
  end: Cell,
  occupancy: Uint8Array,
  grid: OccupancyGrid
): boolean {
  let x0 = start.x;
  let y0 = start.y;
  const x1 = end.x;
  const y1 = end.y;
  const dx = Math.abs(x1 - x0);
  const dy = Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;
  let err = dx - dy;

  while (true) {
    const index = y0 * grid.width + x0;
    if (occupancy[index]) {
      return false;
    }
    if (x0 === x1 && y0 === y1) {
      return true;
    }
    const err2 = err * 2;
    if (err2 > -dy) {
      err -= dy;
      x0 += sx;
    }
    if (err2 < dx) {
      err += dx;
      y0 += sy;
    }
  }
}

function findNearestFreeCell(
  cell: Cell,
  occupancy: Uint8Array,
  grid: OccupancyGrid,
  maxRadius: number
): Cell | null {
  const baseIndex = cellToLinearIndex(grid, cell);
  if (!occupancy[baseIndex]) {
    return cell;
  }

  for (let radius = 1; radius <= maxRadius; radius += 1) {
    for (let y = cell.y - radius; y <= cell.y + radius; y += 1) {
      if (y < 0 || y >= grid.height) {
        continue;
      }
      for (let x = cell.x - radius; x <= cell.x + radius; x += 1) {
        if (x < 0 || x >= grid.width) {
          continue;
        }
        const dx = x - cell.x;
        const dy = y - cell.y;
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== radius) {
          continue;
        }
        const index = y * grid.width + x;
        if (!occupancy[index]) {
          return { x, y };
        }
      }
    }
  }

  return null;
}

function heuristic(left: Cell, right: Cell): number {
  return Math.hypot(right.x - left.x, right.y - left.y);
}

function cellCenterToWorld(grid: OccupancyGrid, cell: Cell): PlanPoint2D {
  return {
    x: grid.origin.x + (cell.x + 0.5) * grid.resolution,
    y: grid.origin.y + (cell.y + 0.5) * grid.resolution
  };
}

class MinHeap {
  private readonly values: OpenNode[] = [];

  push(node: OpenNode): void {
    this.values.push(node);
    this.bubbleUp(this.values.length - 1);
  }

  pop(): OpenNode | undefined {
    if (this.values.length === 0) {
      return undefined;
    }
    const first = this.values[0];
    const last = this.values.pop();
    if (this.values.length && last) {
      this.values[0] = last;
      this.bubbleDown(0);
    }
    return first;
  }

  isEmpty(): boolean {
    return this.values.length === 0;
  }

  private bubbleUp(index: number): void {
    let current = index;
    while (current > 0) {
      const parent = Math.floor((current - 1) / 2);
      if (this.values[parent].priority <= this.values[current].priority) {
        break;
      }
      [this.values[parent], this.values[current]] = [this.values[current], this.values[parent]];
      current = parent;
    }
  }

  private bubbleDown(index: number): void {
    let current = index;
    while (true) {
      const left = current * 2 + 1;
      const right = left + 1;
      let next = current;

      if (left < this.values.length && this.values[left].priority < this.values[next].priority) {
        next = left;
      }
      if (right < this.values.length && this.values[right].priority < this.values[next].priority) {
        next = right;
      }
      if (next === current) {
        break;
      }

      [this.values[current], this.values[next]] = [this.values[next], this.values[current]];
      current = next;
    }
  }
}
