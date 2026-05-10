import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Pose3D, SceneManifest } from "../lib/gs/gsSdfSceneAdapter";
import {
  getLiveRosTopicName,
  type LiveContract
} from "../lib/contracts/liveContract";
import type {
  GoalPose2D,
  InitialPose2D,
  OccupancyMapMeta,
  ViewportTransform
} from "../lib/ros/navGoalPublisher";
import type { RosClient } from "../lib/ros/navGoalPublisher";
import {
  createInitialPoseTopic,
  createMoveBaseGoalTopic,
  publishInitialPose,
  publishMoveBaseSimpleGoal
} from "../lib/ros/navGoalPublisher";
import {
  cellToLinearIndex,
  createGrid,
  type OccupancyGrid,
  projectOccPriorToGrid,
  worldToCell,
  type Point3
} from "../lib/map/occupancyProjection";
import { planPathOnGrid, type PlanPoint2D } from "../lib/map/localPlanner";
import { geometryToPoints, loadPlyGeometry } from "../lib/ply/loadPly";
import type { DecodedPointCloud } from "../lib/ros/pointCloud2";

export type MapTool = "goal" | "initialPose" | "measure" | "obstacle" | "erase";

export interface Measurement2D {
  id: string;
  start: { x: number; y: number };
  end: { x: number; y: number };
  distance: number;
}

export interface ObstacleRect2D {
  id: string;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

interface TopDownMapProps {
  manifest: SceneManifest | null;
  ros: RosClient | null;
  liveContract?: LiveContract | null;
  streamProfile?: "idle" | "neural" | "playback";
  robotPose: Pose3D | null;
  trajectory: Pose3D[];
  mapPointCloud: DecodedPointCloud | null;
  scanPointCloud?: DecodedPointCloud | null;
  goals: GoalPose2D[];
  plannedPath: PlanPoint2D[];
  initialPose: InitialPose2D | null;
  measurements: Measurement2D[];
  obstacles: ObstacleRect2D[];
  erasures: ObstacleRect2D[];
  selectedTool: MapTool;
  followRobot: boolean;
  exportRequestVersion?: number;
  onGoal(goal: GoalPose2D): void;
  onPlannedPath(path: PlanPoint2D[]): void;
  onPlanningState(status: "idle" | "ready" | "blocked", reason?: string): void;
  onInitialPose(pose: InitialPose2D): void;
  onMeasurement(measurement: Measurement2D): void;
  onObstacle(obstacle: ObstacleRect2D): void;
  onErase(eraseRect: ObstacleRect2D): void;
  onExported?(message: string, ok: boolean): void;
}

interface XYBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

interface DragState {
  pointerId: number;
  kind: "panOrPlace" | "measure" | "obstacle" | "erase";
  startX: number;
  startY: number;
  startTranslateX: number;
  startTranslateY: number;
  startWorld: { x: number; y: number };
  moved: boolean;
}

const defaultViewport: ViewportTransform = {
  scale: 1,
  translateX: 0,
  translateY: 0
};

export default function TopDownMap(props: TopDownMapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [viewport, setViewport] = useState<ViewportTransform>(defaultViewport);
  const [baseGrid, setBaseGrid] = useState<OccupancyGrid | null>(null);
  const [draftMeasurement, setDraftMeasurement] = useState<Measurement2D | null>(null);
  const [draftObstacle, setDraftObstacle] = useState<ObstacleRect2D | null>(null);
  const [draftErase, setDraftErase] = useState<ObstacleRect2D | null>(null);
  const dragStateRef = useRef<DragState | null>(null);
  const playbackViewportFitRef = useRef(false);
  const playbackStreamRefitPendingRef = useRef(true);
  const lastExportVersionRef = useRef(0);
  const streamBuildRefLight = useRef<{
    cloud: DecodedPointCloud | null;
    manifest: SceneManifest | null;
    meta: OccupancyMapMeta | null;
  }>({ cloud: null, manifest: null, meta: null });
  const playbackGridRafRef = useRef(0);
  const playbackGridLastBuildRef = useRef(0);
  const playbackGridLastStampRef = useRef<number | null>(null);
  const drawModelRef = useRef<{
    effectiveGrid: OccupancyGrid | null;
    rasterCanvas: HTMLCanvasElement | null;
    viewport: ViewportTransform;
    draftObstacle: ObstacleRect2D | null;
    draftErase: ObstacleRect2D | null;
    props: TopDownMapProps;
  } | null>(null);

  const mapMeta = useMemo<OccupancyMapMeta | null>(() => {
    if (!props.manifest?.meta?.leafSize || !props.manifest.meta.mapSize || !props.manifest.meta.mapOrigin) {
      return null;
    }

    const width = Math.ceil(props.manifest.meta.mapSize / props.manifest.meta.leafSize);
    const height = width;
    return {
      width,
      height,
      resolution: props.manifest.meta.leafSize,
      origin: {
        x: props.manifest.meta.mapOrigin.x,
        y: props.manifest.meta.mapOrigin.y
      },
      frameId: props.manifest.frameId
    };
  }, [props.manifest]);

  /** Manifest meta is optional for live/playback streaming — derive grid frame from the point cloud. */
  const streamingMapMeta = useMemo(() => {
    if (mapMeta) {
      return mapMeta;
    }
    if (!props.mapPointCloud || props.mapPointCloud.renderedPointCount < 1) {
      return null;
    }
    return buildMapMetaFromPointCloud(props.mapPointCloud, props.manifest);
  }, [mapMeta, props.mapPointCloud, props.manifest]);

  const hasPrebuiltOccupancy = props.manifest?.occupancy?.source === "prebuilt_grid";
  const mapCloudForGrid =
    hasPrebuiltOccupancy && props.streamProfile !== "playback" ? null : props.mapPointCloud;

  streamBuildRefLight.current.cloud = mapCloudForGrid ?? null;
  streamBuildRefLight.current.manifest = props.manifest;
  streamBuildRefLight.current.meta = streamingMapMeta;

  useEffect(() => {
    if (mapCloudForGrid && streamingMapMeta && props.manifest) {
      if (props.streamProfile === "playback") {
        if (playbackGridRafRef.current) {
          return;
        }
        playbackGridRafRef.current = requestAnimationFrame(() => {
          playbackGridRafRef.current = 0;
          const { cloud, manifest, meta } = streamBuildRefLight.current;
          if (!cloud || !meta || !manifest) {
            return;
          }
          const nextStampMs = cloud.stampMs ?? null;
          if (nextStampMs !== null && playbackGridLastStampRef.current === nextStampMs) {
            return;
          }
          const now = performance.now();
          if (
            playbackGridLastBuildRef.current > 0 &&
            now - playbackGridLastBuildRef.current < 28
          ) {
            return;
          }
          playbackGridLastBuildRef.current = now;
          playbackGridLastStampRef.current = nextStampMs;
          const nextGrid = projectLiveCloudToGrid(
            sampleDecodedCloudForGrid(cloud, 24_000),
            manifest,
            meta
          );
          setBaseGrid(nextGrid);
          const c = canvasRef.current;
          if (
            (playbackStreamRefitPendingRef.current || !playbackViewportFitRef.current) &&
            canvasHasLayout(c)
          ) {
            setViewport(fitViewport(nextGrid, c));
            playbackViewportFitRef.current = true;
            playbackStreamRefitPendingRef.current = false;
          }
        });
        return () => {
          if (playbackGridRafRef.current) {
            cancelAnimationFrame(playbackGridRafRef.current);
            playbackGridRafRef.current = 0;
          }
        };
      }
      const nextGrid = projectLiveCloudToGrid(
        sampleDecodedCloudForGrid(mapCloudForGrid, 36_000),
        props.manifest,
        streamingMapMeta
      );
      setBaseGrid(nextGrid);
      const c = canvasRef.current;
      if (
        (playbackStreamRefitPendingRef.current || !playbackViewportFitRef.current) &&
        canvasHasLayout(c)
      ) {
        setViewport(fitViewport(nextGrid, c));
        playbackViewportFitRef.current = true;
        playbackStreamRefitPendingRef.current = false;
      }
      return;
    }

    playbackViewportFitRef.current = false;
    playbackStreamRefitPendingRef.current = true;
    playbackGridLastStampRef.current = null;

    if (!props.manifest?.occupancy || !mapMeta) {
      setBaseGrid(null);
      return;
    }

    const eagerFallbackGrid = createEmptyGrid(props.manifest, mapMeta);
    setBaseGrid((current) => current ?? eagerFallbackGrid);
    const existingCanvas = canvasRef.current;
    if (!playbackViewportFitRef.current && canvasHasLayout(existingCanvas)) {
      setViewport((current) =>
        current === defaultViewport ? fitViewport(eagerFallbackGrid, existingCanvas) : current
      );
      playbackViewportFitRef.current = true;
    }

    let cancelled = false;

    const load = async () => {
      const occupancy = props.manifest!.occupancy!;
      const nextGrid =
        occupancy.source === "prebuilt_grid"
          ? await loadPrebuiltOccupancyGrid(occupancy.url, props.manifest!, mapMeta)
          : projectPointsToGrid(
              geometryToPoints(await loadPlyGeometry(occupancy.url)),
              props.manifest!,
              mapMeta
            );
      if (!cancelled) {
        setBaseGrid(nextGrid);
        setViewport(fitViewport(nextGrid, canvasRef.current));
      }
    };

    load().catch((error) => {
      console.error("Failed to load occupancy prior.", error);
      if (!cancelled) {
        const fallbackGrid = createEmptyGrid(props.manifest!, mapMeta);
        setBaseGrid(fallbackGrid);
        setViewport(fitViewport(fallbackGrid, canvasRef.current));
      }
    });

    return () => {
      cancelled = true;
    };
  }, [mapCloudForGrid, mapMeta, streamingMapMeta, props.manifest, props.streamProfile]);

  const effectiveGrid = useMemo(() => {
    if (!baseGrid) {
      return null;
    }
    return applyGridEdits(baseGrid, props.obstacles, props.erasures);
  }, [baseGrid, props.obstacles, props.erasures]);

  const gridRasterCanvas = useMemo(
    () => (effectiveGrid ? createGridRasterCanvas(effectiveGrid) : null),
    [effectiveGrid]
  );

  drawModelRef.current = {
    effectiveGrid,
    rasterCanvas: gridRasterCanvas,
    viewport,
    draftObstacle,
    draftErase,
    props
  };

  const paintCanvas = () => {
    const canvas = canvasRef.current;
    const parent = canvas?.parentElement;
    if (!canvas || !parent) {
      return;
    }
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    const d = drawModelRef.current;
    if (!d) {
      return;
    }
    canvas.width = parent.clientWidth;
    canvas.height = parent.clientHeight;

    let viewportForDraw = d.viewport;
    if (
      d.props.streamProfile === "playback" &&
      d.effectiveGrid &&
      canvas.clientWidth > 0 &&
      canvas.clientHeight > 0 &&
      !playbackViewportFitRef.current
    ) {
      viewportForDraw = fitViewport(d.effectiveGrid, canvas);
      playbackViewportFitRef.current = true;
      setViewport(viewportForDraw);
    }

    if (d.effectiveGrid) {
      drawMap(
        context,
        canvas,
        d.effectiveGrid,
        d.rasterCanvas,
        viewportForDraw,
        d.props,
        d.draftObstacle,
        d.draftErase
      );
    } else {
      drawMapLoadingPlaceholder(context, canvas, d.props.mapPointCloud);
    }
  };

  const paintCanvasRef = useRef(paintCanvas);
  paintCanvasRef.current = paintCanvas;

  useLayoutEffect(() => {
    paintCanvasRef.current();
  }, [
    effectiveGrid,
    viewport,
    draftObstacle,
    draftErase,
    props.goals,
    props.plannedPath,
    props.trajectory,
    props.robotPose,
    props.obstacles,
    props.erasures,
    props.measurements,
    props.initialPose,
    props.mapPointCloud,
    props.scanPointCloud,
    gridRasterCanvas
  ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const parent = canvas?.parentElement;
    if (!canvas || !parent) {
      return;
    }
    const observer = new ResizeObserver(() => paintCanvasRef.current());
    observer.observe(parent);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!props.followRobot || !effectiveGrid || !props.robotPose || !canvasRef.current) {
      return;
    }

    setViewport((current) =>
      centerViewportOnWorld(
        current,
        canvasRef.current!,
        effectiveGrid,
        props.robotPose!.position.x,
        props.robotPose!.position.y
      )
    );
  }, [effectiveGrid, props.followRobot, props.robotPose]);

  useEffect(() => {
    if (!effectiveGrid || props.goals.length === 0) {
      props.onPlannedPath([]);
      props.onPlanningState("idle");
      return;
    }

    const latestGoal = props.goals[props.goals.length - 1];
    const start = getPlannerStart(props);
    const result = planPathOnGrid({
      grid: effectiveGrid,
      start,
      goal: latestGoal,
      robotRadius: getLocalPlannerInflationRadius(props.manifest)
    });

    props.onPlannedPath(result.path);
    props.onPlanningState(result.status, result.reason);
  }, [
    effectiveGrid,
    props.goals,
    props.initialPose,
    props.manifest,
    props.robotPose,
    props.trajectory
  ]);

  useEffect(() => {
    const exportVersion = props.exportRequestVersion ?? 0;
    if (exportVersion <= 0 || exportVersion === lastExportVersionRef.current) {
      return;
    }
    lastExportVersionRef.current = exportVersion;

    if (!effectiveGrid) {
      props.onExported?.("Map is not ready yet.", false);
      return;
    }

    const filename = `gs_sdf_map_${Date.now()}`;
    downloadGridAsPgm(effectiveGrid, filename);
    props.onExported?.(`Saved ${filename}.pgm`, true);
  }, [effectiveGrid, props]);

  const beginInteraction = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !effectiveGrid) {
      return;
    }

    canvas.setPointerCapture(event.pointerId);
    const startWorld = screenToWorld(event.nativeEvent.offsetX, event.nativeEvent.offsetY, viewport, effectiveGrid);
    const forcePan = event.shiftKey || event.button === 1;
    const kind =
      forcePan || props.selectedTool === "goal" || props.selectedTool === "initialPose"
        ? "panOrPlace"
        : props.selectedTool === "measure"
          ? "measure"
          : props.selectedTool === "erase"
            ? "erase"
            : "obstacle";

    dragStateRef.current = {
      pointerId: event.pointerId,
      kind,
      startX: event.clientX,
      startY: event.clientY,
      startTranslateX: viewport.translateX,
      startTranslateY: viewport.translateY,
      startWorld,
      moved: false
    };
  };

  const updateInteraction = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragStateRef.current;
    if (!drag || drag.pointerId !== event.pointerId || !effectiveGrid) {
      return;
    }

    const deltaX = event.clientX - drag.startX;
    const deltaY = event.clientY - drag.startY;
    const moved = Math.abs(deltaX) > 4 || Math.abs(deltaY) > 4;
    const currentWorld = screenToWorld(event.nativeEvent.offsetX, event.nativeEvent.offsetY, viewport, effectiveGrid);
    dragStateRef.current = { ...drag, moved };

    if (drag.kind === "panOrPlace") {
      setViewport((current) => ({
        ...current,
        translateX: drag.startTranslateX + deltaX,
        translateY: drag.startTranslateY + deltaY
      }));
      return;
    }

    if (drag.kind === "measure") {
      setDraftMeasurement({
        id: "draft-measure",
        start: drag.startWorld,
        end: currentWorld,
        distance: distance2D(drag.startWorld, currentWorld)
      });
      return;
    }

    if (drag.kind === "erase") {
      setDraftErase({
        id: "draft-erase",
        minX: Math.min(drag.startWorld.x, currentWorld.x),
        maxX: Math.max(drag.startWorld.x, currentWorld.x),
        minY: Math.min(drag.startWorld.y, currentWorld.y),
        maxY: Math.max(drag.startWorld.y, currentWorld.y)
      });
      return;
    }

    setDraftObstacle({
      id: "draft-obstacle",
      minX: Math.min(drag.startWorld.x, currentWorld.x),
      maxX: Math.max(drag.startWorld.x, currentWorld.x),
      minY: Math.min(drag.startWorld.y, currentWorld.y),
      maxY: Math.max(drag.startWorld.y, currentWorld.y)
    });
  };

  const endInteraction = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const drag = dragStateRef.current;
    dragStateRef.current = null;
    if (!canvas || !effectiveGrid || !mapMeta || !drag) {
      return;
    }

    canvas.releasePointerCapture(event.pointerId);
    const world = screenToWorld(event.nativeEvent.offsetX, event.nativeEvent.offsetY, viewport, effectiveGrid);

    if (drag.kind === "panOrPlace") {
      if (drag.moved) {
        return;
      }

      if (props.selectedTool === "initialPose") {
        const pose = {
          x: world.x,
          y: world.y,
          yaw: 0
        };
        if (props.ros) {
          const topicName = getLiveRosTopicName(
            props.liveContract,
            "initial_pose",
            "/initialpose",
            "publish"
          );
          const topic = topicName ? createInitialPoseTopic(props.ros, topicName) : null;
          if (topic) {
            publishInitialPose(topic, pose, mapMeta.frameId);
          }
        }
        props.onInitialPose(pose);
        return;
      }

      const goal = {
        x: world.x,
        y: world.y,
        yaw: 0
      };

      if (props.ros) {
        const topicName = getLiveRosTopicName(
          props.liveContract,
          "nav_goal",
          "/move_base_simple/goal",
          "publish"
        );
        const topic = topicName ? createMoveBaseGoalTopic(props.ros, topicName) : null;
        if (topic) {
          publishMoveBaseSimpleGoal(topic, goal, mapMeta.frameId);
        }
      }
      props.onGoal(goal);
      return;
    }

    if (drag.kind === "measure") {
      const measurement = {
        id: `measure-${Date.now()}`,
        start: drag.startWorld,
        end: world,
        distance: distance2D(drag.startWorld, world)
      };
      setDraftMeasurement(null);
      if (measurement.distance > effectiveGrid.resolution * 0.75) {
        props.onMeasurement(measurement);
      }
      return;
    }

    if (drag.kind === "erase") {
      const eraseRect = {
        id: `erase-${Date.now()}`,
        minX: Math.min(drag.startWorld.x, world.x),
        maxX: Math.max(drag.startWorld.x, world.x),
        minY: Math.min(drag.startWorld.y, world.y),
        maxY: Math.max(drag.startWorld.y, world.y)
      };
      setDraftErase(null);
      if (eraseRect.maxX - eraseRect.minX > effectiveGrid.resolution && eraseRect.maxY - eraseRect.minY > effectiveGrid.resolution) {
        props.onErase(eraseRect);
      }
      return;
    }

    const obstacle = {
      id: `obstacle-${Date.now()}`,
      minX: Math.min(drag.startWorld.x, world.x),
      maxX: Math.max(drag.startWorld.x, world.x),
      minY: Math.min(drag.startWorld.y, world.y),
      maxY: Math.max(drag.startWorld.y, world.y)
    };
    setDraftObstacle(null);
    if (
      obstacle.maxX - obstacle.minX > effectiveGrid.resolution &&
      obstacle.maxY - obstacle.minY > effectiveGrid.resolution
    ) {
      props.onObstacle(obstacle);
    }
  };

  return (
    <canvas
      ref={canvasRef}
      className="topdown-canvas"
      onPointerDown={beginInteraction}
      onPointerMove={updateInteraction}
      onPointerUp={endInteraction}
      onPointerCancel={() => {
        dragStateRef.current = null;
        setDraftMeasurement(null);
        setDraftObstacle(null);
        setDraftErase(null);
      }}
      onWheel={(event) => {
        event.preventDefault();
        const canvas = canvasRef.current;
        if (!canvas) {
          return;
        }

        const cursorX = event.nativeEvent.offsetX;
        const cursorY = event.nativeEvent.offsetY;
        const zoomFactor = event.deltaY < 0 ? 1.1 : 0.92;

        setViewport((current) => {
          const nextScale = clamp(current.scale * zoomFactor, 0.35, 12);
          const scaleRatio = nextScale / current.scale;
          return {
            scale: nextScale,
            translateX: cursorX - (cursorX - current.translateX) * scaleRatio,
            translateY: cursorY - (cursorY - current.translateY) * scaleRatio
          };
        });
      }}
    />
  );
}

function createEmptyGrid(manifest: SceneManifest | null, mapMeta: OccupancyMapMeta): OccupancyGrid {
  return createGrid({
    resolution: mapMeta.resolution,
    bounds: {
      minX: mapMeta.origin.x,
      maxX: mapMeta.origin.x + mapMeta.width * mapMeta.resolution,
      minY: mapMeta.origin.y,
      maxY: mapMeta.origin.y + mapMeta.height * mapMeta.resolution,
      minZ: manifest?.meta?.mapOrigin?.z ?? 0,
      maxZ: (manifest?.meta?.mapOrigin?.z ?? 0) + 3
    },
    obstacleBand: {
      minZ: (manifest?.meta?.mapOrigin?.z ?? 0) - 0.1,
      maxZ: (manifest?.meta?.mapOrigin?.z ?? 0) + 1.3
    },
    robotRadius: manifest?.robot?.radius ?? 0.28
  });
}

function getLocalPlannerInflationRadius(manifest: SceneManifest | null): number {
  // Prebuilt PGM maps are generated for the robot footprint already. Inflating
  // them again in the browser makes valid NuRec corridors look blocked.
  if (manifest?.occupancy?.source === "prebuilt_grid") {
    return 0;
  }
  return manifest?.robot?.radius ?? 0.28;
}

async function loadPrebuiltOccupancyGrid(
  url: string,
  manifest: SceneManifest | null,
  mapMeta: OccupancyMapMeta
): Promise<OccupancyGrid> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to fetch occupancy grid ${url}: ${response.status}`);
  }
  const pgm = parsePgm(new Uint8Array(await response.arrayBuffer()));
  const resolution = manifest?.occupancy?.resolution ?? mapMeta.resolution;
  const origin = manifest?.occupancy?.origin ?? mapMeta.origin;
  const grid = createGrid({
    resolution,
    bounds: {
      minX: origin.x,
      maxX: origin.x + pgm.width * resolution,
      minY: origin.y,
      maxY: origin.y + pgm.height * resolution,
      minZ: manifest?.meta?.mapOrigin?.z ?? 0,
      maxZ: (manifest?.meta?.mapOrigin?.z ?? 0) + 3
    },
    obstacleBand: {
      minZ: (manifest?.meta?.mapOrigin?.z ?? 0) - 0.1,
      maxZ: (manifest?.meta?.mapOrigin?.z ?? 0) + 1.3
    },
    robotRadius: manifest?.robot?.radius ?? 0.28
  });

  for (let row = 0; row < pgm.height; row += 1) {
    const cellY = pgm.height - 1 - row;
    for (let x = 0; x < pgm.width; x += 1) {
      const pixel = pgm.pixels[row * pgm.width + x];
      const normalized = pgm.maxValue > 0 ? (pixel / pgm.maxValue) * 255 : pixel;
      const index = cellY * pgm.width + x;
      if (normalized < 128) {
        grid.data[index] = 100;
        grid.hits[index] = 8;
      } else {
        grid.data[index] = 0;
      }
    }
  }

  return grid;
}

function parsePgm(bytes: Uint8Array): {
  width: number;
  height: number;
  maxValue: number;
  pixels: Uint16Array;
} {
  let offset = 0;

  const isWhitespace = (value: number) =>
    value === 9 || value === 10 || value === 13 || value === 32;

  const nextToken = () => {
    while (offset < bytes.length) {
      if (bytes[offset] === 35) {
        while (offset < bytes.length && bytes[offset] !== 10) {
          offset += 1;
        }
      }
      if (!isWhitespace(bytes[offset])) {
        break;
      }
      offset += 1;
    }
    const start = offset;
    while (offset < bytes.length && !isWhitespace(bytes[offset]) && bytes[offset] !== 35) {
      offset += 1;
    }
    return new TextDecoder("ascii").decode(bytes.subarray(start, offset));
  };

  const magic = nextToken();
  const width = Number.parseInt(nextToken(), 10);
  const height = Number.parseInt(nextToken(), 10);
  const maxValue = Number.parseInt(nextToken(), 10);
  if ((magic !== "P5" && magic !== "P2") || !Number.isFinite(width) || !Number.isFinite(height) || !Number.isFinite(maxValue)) {
    throw new Error("Unsupported PGM occupancy grid.");
  }

  const pixelCount = width * height;
  const pixels = new Uint16Array(pixelCount);
  if (magic === "P2") {
    for (let index = 0; index < pixelCount; index += 1) {
      pixels[index] = Number.parseInt(nextToken(), 10);
    }
    return { width, height, maxValue, pixels };
  }

  while (offset < bytes.length && isWhitespace(bytes[offset])) {
    offset += 1;
  }
  if (maxValue <= 255) {
    for (let index = 0; index < pixelCount; index += 1) {
      pixels[index] = bytes[offset + index] ?? 0;
    }
  } else {
    for (let index = 0; index < pixelCount; index += 1) {
      const hi = bytes[offset + index * 2] ?? 0;
      const lo = bytes[offset + index * 2 + 1] ?? 0;
      pixels[index] = hi * 256 + lo;
    }
  }
  return { width, height, maxValue, pixels };
}

function projectPointsToGrid(
  points: Point3[],
  manifest: SceneManifest | null,
  mapMeta: OccupancyMapMeta
): OccupancyGrid {
  const narrowBand = {
    minZ: (manifest?.meta?.mapOrigin?.z ?? 0) - 0.2,
    maxZ: (manifest?.meta?.mapOrigin?.z ?? 0) + 1.4
  };

  let autoBounds = derivePointBounds(points, narrowBand);
  /** Z band passed to projectOccPriorToGrid — must include points after XY slice. */
  let rasterBand = narrowBand;

  if (!autoBounds && points.length > 0) {
    let zMin = Number.POSITIVE_INFINITY;
    let zMax = Number.NEGATIVE_INFINITY;
    for (const point of points) {
      zMin = Math.min(zMin, point.z);
      zMax = Math.max(zMax, point.z);
    }
    if (Number.isFinite(zMin) && Number.isFinite(zMax)) {
      const span = Math.max(2, zMax - zMin, 1.5);
      const wideSlice = { minZ: zMin - span, maxZ: zMax + span };
      autoBounds = derivePointBounds(points, wideSlice);
      rasterBand = { minZ: zMin - span * 0.35, maxZ: zMax + span * 0.35 };
    }
  }

  const bounds = autoBounds
    ? {
        minX: autoBounds.minX,
        maxX: autoBounds.maxX,
        minY: autoBounds.minY,
        maxY: autoBounds.maxY,
        minZ: manifest?.meta?.mapOrigin?.z ?? 0,
        maxZ: (manifest?.meta?.mapOrigin?.z ?? 0) + 3
      }
    : {
        minX: mapMeta.origin.x,
        maxX: mapMeta.origin.x + mapMeta.width * mapMeta.resolution,
        minY: mapMeta.origin.y,
        maxY: mapMeta.origin.y + mapMeta.height * mapMeta.resolution,
        minZ: manifest?.meta?.mapOrigin?.z ?? 0,
        maxZ: (manifest?.meta?.mapOrigin?.z ?? 0) + 3
      };

  return projectOccPriorToGrid(points, {
    resolution: mapMeta.resolution,
    bounds,
    obstacleBand: rasterBand,
    robotRadius: manifest?.robot?.radius ?? 0.28
  });
}

function projectLiveCloudToGrid(
  cloud: DecodedPointCloud,
  manifest: SceneManifest | null,
  mapMeta: OccupancyMapMeta
): OccupancyGrid {
  const narrowBand = {
    minZ: (manifest?.meta?.mapOrigin?.z ?? 0) - 0.2,
    maxZ: (manifest?.meta?.mapOrigin?.z ?? 0) + 1.4
  };

  let autoBounds = deriveDecodedCloudBounds(cloud, narrowBand);
  let rasterBand = narrowBand;

  if (!autoBounds && cloud.renderedPointCount > 0) {
    let zMin = Number.POSITIVE_INFINITY;
    let zMax = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < cloud.renderedPointCount; index += 1) {
      const z = cloud.positions[index * 3 + 2];
      zMin = Math.min(zMin, z);
      zMax = Math.max(zMax, z);
    }
    if (Number.isFinite(zMin) && Number.isFinite(zMax)) {
      const span = Math.max(2, zMax - zMin, 1.5);
      const wideSlice = { minZ: zMin - span, maxZ: zMax + span };
      autoBounds = deriveDecodedCloudBounds(cloud, wideSlice);
      rasterBand = { minZ: zMin - span * 0.35, maxZ: zMax + span * 0.35 };
    }
  }

  const bounds = autoBounds
    ? {
        minX: autoBounds.minX,
        maxX: autoBounds.maxX,
        minY: autoBounds.minY,
        maxY: autoBounds.maxY,
        minZ: manifest?.meta?.mapOrigin?.z ?? 0,
        maxZ: (manifest?.meta?.mapOrigin?.z ?? 0) + 3
      }
    : {
        minX: mapMeta.origin.x,
        maxX: mapMeta.origin.x + mapMeta.width * mapMeta.resolution,
        minY: mapMeta.origin.y,
        maxY: mapMeta.origin.y + mapMeta.height * mapMeta.resolution,
        minZ: manifest?.meta?.mapOrigin?.z ?? 0,
        maxZ: (manifest?.meta?.mapOrigin?.z ?? 0) + 3
      };

  const grid = createGrid({
    resolution: mapMeta.resolution,
    bounds,
    obstacleBand: rasterBand,
    robotRadius: manifest?.robot?.radius ?? 0.28
  });

  for (let index = 0; index < cloud.renderedPointCount; index += 1) {
    const offset = index * 3;
    const x = cloud.positions[offset];
    const y = cloud.positions[offset + 1];
    const z = cloud.positions[offset + 2];
    if (z < rasterBand.minZ || z > rasterBand.maxZ) {
      continue;
    }
    const cell = worldToCell(grid, x, y);
    if (!cell) {
      continue;
    }
    const linear = cellToLinearIndex(grid, cell);
    grid.hits[linear] = Math.min(grid.hits[linear] + 1, 65535);
    grid.data[linear] = 100;
  }

  dilateOccupiedCellsLocal(grid, manifest?.robot?.radius ?? 0.28);
  return grid;
}

function sampleDecodedCloudForGrid(
  cloud: DecodedPointCloud,
  maxPoints: number
): DecodedPointCloud {
  const limit = Math.max(1, maxPoints);
  if (cloud.renderedPointCount <= limit) {
    return cloud;
  }

  const stride = Math.max(1, Math.ceil(cloud.renderedPointCount / limit));
  const kept = Math.min(limit, Math.ceil(cloud.renderedPointCount / stride));
  const positions = new Float32Array(kept * 3);
  const colors = cloud.colors ? new Float32Array(kept * 3) : undefined;
  let writeIndex = 0;

  for (let index = 0; index < cloud.renderedPointCount && writeIndex < kept; index += stride) {
    const readOffset = index * 3;
    const writeOffset = writeIndex * 3;
    positions[writeOffset] = cloud.positions[readOffset];
    positions[writeOffset + 1] = cloud.positions[readOffset + 1];
    positions[writeOffset + 2] = cloud.positions[readOffset + 2];
    if (colors && cloud.colors) {
      colors[writeOffset] = cloud.colors[readOffset];
      colors[writeOffset + 1] = cloud.colors[readOffset + 1];
      colors[writeOffset + 2] = cloud.colors[readOffset + 2];
    }
    writeIndex += 1;
  }

  return {
    ...cloud,
    sourcePointCount: writeIndex,
    renderedPointCount: writeIndex,
    positions: positions.subarray(0, writeIndex * 3),
    colors: colors ? colors.subarray(0, writeIndex * 3) : undefined
  };
}

function buildMapMetaFromPointCloud(cloud: DecodedPointCloud, manifest: SceneManifest | null): OccupancyMapMeta | null {
  if (cloud.renderedPointCount < 1) {
    return null;
  }
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < cloud.renderedPointCount; index += 1) {
    const offset = index * 3;
    const x = cloud.positions[offset];
    const y = cloud.positions[offset + 1];
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  if (!Number.isFinite(minX) || maxX <= minX || maxY <= minY) {
    return null;
  }
  const resolution = manifest?.meta?.leafSize ?? 0.2;
  const pad = resolution * 6;
  const widthM = maxX - minX + 2 * pad;
  const heightM = maxY - minY + 2 * pad;
  const width = Math.max(8, Math.ceil(widthM / resolution));
  const height = Math.max(8, Math.ceil(heightM / resolution));
  return {
    width,
    height,
    resolution,
    origin: {
      x: minX - pad,
      y: minY - pad
    },
    frameId: manifest?.frameId ?? cloud.frameId
  };
}

function derivePointBounds(
  points: Point3[],
  obstacleBand: { minZ: number; maxZ: number }
): XYBounds | null {
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let count = 0;

  for (const point of points) {
    if (point.z < obstacleBand.minZ || point.z > obstacleBand.maxZ) {
      continue;
    }

    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
    count += 1;
  }

  if (count === 0) {
    return null;
  }

  const padding = 2;
  return {
    minX: minX - padding,
    maxX: maxX + padding,
    minY: minY - padding,
    maxY: maxY + padding
  };
}

function deriveDecodedCloudBounds(
  cloud: DecodedPointCloud,
  obstacleBand: { minZ: number; maxZ: number }
): XYBounds | null {
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let count = 0;

  for (let index = 0; index < cloud.renderedPointCount; index += 1) {
    const offset = index * 3;
    const z = cloud.positions[offset + 2];
    if (z < obstacleBand.minZ || z > obstacleBand.maxZ) {
      continue;
    }
    const x = cloud.positions[offset];
    const y = cloud.positions[offset + 1];
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
    count += 1;
  }

  if (count === 0) {
    return null;
  }

  const padding = 2;
  return {
    minX: minX - padding,
    maxX: maxX + padding,
    minY: minY - padding,
    maxY: maxY + padding
  };
}

function dilateOccupiedCellsLocal(grid: OccupancyGrid, robotRadius: number): void {
  const radiusCells = Math.ceil(robotRadius / grid.resolution);
  if (radiusCells <= 0) {
    return;
  }

  const source = new Int8Array(grid.data);
  const radiusSquared = radiusCells * radiusCells;
  for (let y = 0; y < grid.height; y += 1) {
    for (let x = 0; x < grid.width; x += 1) {
      const index = y * grid.width + x;
      if (source[index] !== 100) {
        continue;
      }
      for (let oy = y - radiusCells; oy <= y + radiusCells; oy += 1) {
        if (oy < 0 || oy >= grid.height) {
          continue;
        }
        for (let ox = x - radiusCells; ox <= x + radiusCells; ox += 1) {
          if (ox < 0 || ox >= grid.width) {
            continue;
          }
          const dx = ox - x;
          const dy = oy - y;
          if (dx * dx + dy * dy > radiusSquared) {
            continue;
          }
          grid.data[oy * grid.width + ox] = 100;
        }
      }
    }
  }
}

function canvasHasLayout(canvas: HTMLCanvasElement | null): canvas is HTMLCanvasElement {
  return Boolean(canvas && canvas.clientWidth > 0 && canvas.clientHeight > 0);
}

function fitViewport(grid: OccupancyGrid, canvas: HTMLCanvasElement | null): ViewportTransform {
  if (!canvas) {
    return defaultViewport;
  }

  const scaleX = canvas.clientWidth / grid.width;
  const scaleY = canvas.clientHeight / grid.height;
  const scale = Math.max(0.2, Math.min(scaleX, scaleY));

  return {
    scale,
    translateX: (canvas.clientWidth - grid.width * scale) * 0.5,
    translateY: (canvas.clientHeight - grid.height * scale) * 0.5
  };
}

function centerViewportOnWorld(
  current: ViewportTransform,
  canvas: HTMLCanvasElement,
  grid: OccupancyGrid,
  x: number,
  y: number
): ViewportTransform {
  const pixel = worldToGridPixel(grid, x, y);
  return {
    ...current,
    translateX: canvas.clientWidth * 0.5 - pixel.x * current.scale,
    translateY: canvas.clientHeight * 0.5 - pixel.y * current.scale
  };
}

function drawMapLoadingPlaceholder(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  mapPointCloud: DecodedPointCloud | null
): void {
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#3d3830";
  context.fillRect(0, 0, canvas.width, canvas.height);
  const fontPx = Math.max(12, Math.min(16, Math.floor(canvas.width / 26)));
  context.font = `${fontPx}px "IBM Plex Sans","Segoe UI",sans-serif`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillStyle = "#e9e1d4";
  const line1 = mapPointCloud ? "Building 2D map from stream…" : "Waiting for map / point cloud…";
  context.fillText(line1, canvas.width / 2, canvas.height * 0.42);
  context.font = `${Math.max(10, fontPx - 2)}px "IBM Plex Sans","Segoe UI",sans-serif`;
  context.fillStyle = "#9a9288";
  const line2 = mapPointCloud ? "No points yet — check ROS topics" : "Load scene manifest or connect playback cloud";
  context.fillText(line2, canvas.width / 2, canvas.height * 0.42 + fontPx + 10);
}

function createGridRasterCanvas(grid: OccupancyGrid): HTMLCanvasElement | null {
  const imageData = new ImageData(grid.width, grid.height);
  for (let y = 0; y < grid.height; y += 1) {
    for (let x = 0; x < grid.width; x += 1) {
      const sourceIndex = y * grid.width + x;
      const targetIndex = ((grid.height - 1 - y) * grid.width + x) * 4;
      const occupied = grid.data[sourceIndex] === 100;
      const hit = grid.hits[sourceIndex];
      const strength = occupied ? Math.min(1, Math.log2(hit + 1) / 5) : 0;
      imageData.data[targetIndex] = occupied ? Math.round(16 + strength * 78) : 12;
      imageData.data[targetIndex + 1] = occupied ? Math.round(82 + strength * 185) : 62;
      imageData.data[targetIndex + 2] = occupied ? Math.round(68 + (1 - strength) * 120) : 92;
      imageData.data[targetIndex + 3] = 255;
    }
  }

  const bufferCanvas = document.createElement("canvas");
  bufferCanvas.width = grid.width;
  bufferCanvas.height = grid.height;
  const bufferContext = bufferCanvas.getContext("2d");
  if (!bufferContext) {
    return null;
  }
  bufferContext.putImageData(imageData, 0, 0);
  return bufferCanvas;
}

function drawMap(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  grid: OccupancyGrid,
  rasterCanvas: HTMLCanvasElement | null,
  viewport: ViewportTransform,
  props: TopDownMapProps,
  draftObstacle: ObstacleRect2D | null,
  draftErase: ObstacleRect2D | null
): void {
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#f1e6d0";
  context.fillRect(0, 0, canvas.width, canvas.height);

  context.save();
  context.translate(viewport.translateX, viewport.translateY);
  context.scale(viewport.scale, viewport.scale);
  context.imageSmoothingEnabled = false;
  if (rasterCanvas) {
    context.drawImage(rasterCanvas, 0, 0);
  }

  drawPlaybackScanOverlay(context, grid, props.scanPointCloud ?? null);
  drawEraseZones(context, grid, props.erasures, draftErase);
  drawObstacles(context, grid, props.obstacles);
  if (draftObstacle) {
    drawObstacles(context, grid, [draftObstacle], true);
  }
  drawTrajectory(context, grid, props.trajectory);
  drawPlannedPath(context, grid, props.plannedPath);
  drawMeasurements(context, grid, props.measurements);
  drawGoals(context, grid, props.goals);
  drawInitialPose(context, grid, props.initialPose);
  drawRobot(context, grid, props.robotPose);
  context.restore();
}

function drawPlaybackScanOverlay(
  context: CanvasRenderingContext2D,
  grid: OccupancyGrid,
  cloud: DecodedPointCloud | null
): void {
  if (!cloud || cloud.renderedPointCount < 1) {
    return;
  }

  const maxOverlayPoints = 2_400;
  const stride = Math.max(1, Math.ceil(cloud.renderedPointCount / maxOverlayPoints));
  const scale = Math.max(context.getTransform().a, 1e-3);
  const size = Math.max(1.55 / scale, 0.85);
  context.fillStyle = "rgba(255, 198, 96, 0.72)";

  for (let index = 0; index < cloud.renderedPointCount; index += stride) {
    const offset = index * 3;
    const x = cloud.positions[offset];
    const y = cloud.positions[offset + 1];
    const cell = worldToCell(grid, x, y);
    if (!cell) {
      continue;
    }
    const px = cell.x + 0.5;
    const py = grid.height - cell.y - 0.5;
    context.fillRect(px - size * 0.5, py - size * 0.5, size, size);
  }
}

function drawTrajectory(context: CanvasRenderingContext2D, grid: OccupancyGrid, trajectory: Pose3D[]): void {
  if (trajectory.length < 2) {
    return;
  }

  const points = trajectory.map((pose) => worldToGridPixel(grid, pose.position.x, pose.position.y));
  const scale = Math.max(context.getTransform().a, 1e-3);
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = "rgba(11, 16, 22, 0.55)";
  context.lineWidth = Math.max(4.2 / scale, 1.6);
  beginSmoothTrajectoryPath(context, points);
  context.stroke();

  context.strokeStyle = "#ffd166";
  context.lineWidth = Math.max(2.2 / scale, 1);
  beginSmoothTrajectoryPath(context, points);
  context.stroke();

  const markerStride = Math.max(1, Math.floor(trajectory.length / 18));
  context.fillStyle = "#fff4da";
  const markerRadius = Math.max(2.1 / scale, 0.95);
  for (let index = 0; index < trajectory.length; index += markerStride) {
    const point = points[index];
    context.beginPath();
    context.arc(point.x, point.y, markerRadius, 0, Math.PI * 2);
    context.fill();
  }

  const end = points[points.length - 1];
  context.fillStyle = "#ff8f52";
  context.beginPath();
  context.arc(end.x, end.y, Math.max(3.4 / scale, 1.8), 0, Math.PI * 2);
  context.fill();
}

function beginSmoothTrajectoryPath(
  context: CanvasRenderingContext2D,
  points: Array<{ x: number; y: number }>
): void {
  context.beginPath();
  context.moveTo(points[0].x, points[0].y);
  if (points.length === 2) {
    context.lineTo(points[1].x, points[1].y);
    return;
  }

  for (let index = 1; index < points.length - 1; index += 1) {
    const current = points[index];
    const next = points[index + 1];
    const midX = (current.x + next.x) * 0.5;
    const midY = (current.y + next.y) * 0.5;
    context.quadraticCurveTo(current.x, current.y, midX, midY);
  }

  const penultimate = points[points.length - 2];
  const last = points[points.length - 1];
  context.quadraticCurveTo(penultimate.x, penultimate.y, last.x, last.y);
}

function drawPlannedPath(context: CanvasRenderingContext2D, grid: OccupancyGrid, path: PlanPoint2D[]): void {
  if (path.length < 2) {
    return;
  }

  context.strokeStyle = "#ffd166";
  context.lineWidth = Math.max(2.4 / context.getTransform().a, 1.2);
  context.beginPath();
  path.forEach((point2D, index) => {
    const point = worldToGridPixel(grid, point2D.x, point2D.y);
    if (index === 0) {
      context.moveTo(point.x, point.y);
    } else {
      context.lineTo(point.x, point.y);
    }
  });
  context.stroke();

  context.fillStyle = "#ffd166";
  const end = worldToGridPixel(grid, path[path.length - 1].x, path[path.length - 1].y);
  const radius = Math.max(4.5 / context.getTransform().a, 2.3);
  context.beginPath();
  context.arc(end.x, end.y, radius, 0, Math.PI * 2);
  context.fill();
}

function drawRobot(context: CanvasRenderingContext2D, grid: OccupancyGrid, pose: Pose3D | null): void {
  if (!pose) {
    return;
  }

  const point = worldToGridPixel(grid, pose.position.x, pose.position.y);
  const radius = Math.max(4 / context.getTransform().a, 2.5);
  const yaw = quaternionToYaw(pose.orientation);

  context.fillStyle = "#da5a2a";
  context.beginPath();
  context.arc(point.x, point.y, radius, 0, Math.PI * 2);
  context.fill();

  context.strokeStyle = "#fff4da";
  context.lineWidth = Math.max(1.8 / context.getTransform().a, 1);
  context.beginPath();
  context.moveTo(point.x, point.y);
  context.lineTo(point.x + Math.cos(yaw) * radius * 2.3, point.y - Math.sin(yaw) * radius * 2.3);
  context.stroke();
}

function drawGoals(context: CanvasRenderingContext2D, grid: OccupancyGrid, goals: GoalPose2D[]): void {
  context.fillStyle = "#246bce";
  goals.forEach((goal) => {
    const point = worldToGridPixel(grid, goal.x, goal.y);
    const size = Math.max(5 / context.getTransform().a, 2.8);
    context.beginPath();
    context.moveTo(point.x, point.y - size);
    context.lineTo(point.x + size, point.y);
    context.lineTo(point.x, point.y + size);
    context.lineTo(point.x - size, point.y);
    context.closePath();
    context.fill();
  });
}

function getPlannerStart(props: TopDownMapProps): PlanPoint2D | null {
  if (props.robotPose) {
    return {
      x: props.robotPose.position.x,
      y: props.robotPose.position.y
    };
  }

  if (props.initialPose) {
    return {
      x: props.initialPose.x,
      y: props.initialPose.y
    };
  }

  const fallbackPose = props.trajectory.at(-1);
  if (fallbackPose) {
    return {
      x: fallbackPose.position.x,
      y: fallbackPose.position.y
    };
  }

  return null;
}

function drawInitialPose(
  context: CanvasRenderingContext2D,
  grid: OccupancyGrid,
  initialPose: InitialPose2D | null
): void {
  if (!initialPose) {
    return;
  }

  const point = worldToGridPixel(grid, initialPose.x, initialPose.y);
  const radius = Math.max(5 / context.getTransform().a, 3);
  context.strokeStyle = "#f59e0b";
  context.lineWidth = Math.max(2 / context.getTransform().a, 1);
  context.beginPath();
  context.arc(point.x, point.y, radius, 0, Math.PI * 2);
  context.stroke();
  context.beginPath();
  context.moveTo(point.x, point.y);
  context.lineTo(point.x + Math.cos(initialPose.yaw) * radius * 2.2, point.y - Math.sin(initialPose.yaw) * radius * 2.2);
  context.stroke();
}

function drawMeasurements(
  context: CanvasRenderingContext2D,
  grid: OccupancyGrid,
  measurements: Measurement2D[]
): void {
  if (measurements.length === 0) {
    return;
  }

  context.strokeStyle = "#4f46e5";
  context.fillStyle = "#312e81";
  context.lineWidth = Math.max(1.5 / context.getTransform().a, 1);
  context.font = `${Math.max(12 / context.getTransform().a, 4)}px IBM Plex Sans, sans-serif`;
  measurements.forEach((measurement) => {
    const start = worldToGridPixel(grid, measurement.start.x, measurement.start.y);
    const end = worldToGridPixel(grid, measurement.end.x, measurement.end.y);
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.stroke();
    const labelX = (start.x + end.x) * 0.5;
    const labelY = (start.y + end.y) * 0.5;
    context.fillText(`${measurement.distance.toFixed(1)} m`, labelX + 4 / context.getTransform().a, labelY - 4 / context.getTransform().a);
  });
}

function drawObstacles(
  context: CanvasRenderingContext2D,
  grid: OccupancyGrid,
  obstacles: ObstacleRect2D[],
  draft = false
): void {
  context.fillStyle = draft ? "rgba(220, 38, 38, 0.12)" : "rgba(220, 38, 38, 0.2)";
  context.strokeStyle = "#dc2626";
  context.lineWidth = Math.max(1.4 / context.getTransform().a, 1);
  obstacles.forEach((obstacle) => {
    const min = worldToGridPixel(grid, obstacle.minX, obstacle.maxY);
    const max = worldToGridPixel(grid, obstacle.maxX, obstacle.minY);
    context.beginPath();
    context.rect(min.x, min.y, max.x - min.x, max.y - min.y);
    context.fill();
    context.stroke();
  });
}

function drawEraseZones(
  context: CanvasRenderingContext2D,
  grid: OccupancyGrid,
  erasures: ObstacleRect2D[],
  draftErase: ObstacleRect2D | null
): void {
  const zones = draftErase ? [...erasures, draftErase] : erasures;
  context.fillStyle = "rgba(14, 165, 233, 0.14)";
  context.strokeStyle = "#0ea5e9";
  context.lineWidth = Math.max(1.4 / context.getTransform().a, 1);
  zones.forEach((zone) => {
    const min = worldToGridPixel(grid, zone.minX, zone.maxY);
    const max = worldToGridPixel(grid, zone.maxX, zone.minY);
    context.beginPath();
    context.rect(min.x, min.y, max.x - min.x, max.y - min.y);
    context.fill();
    context.stroke();
  });
}

function worldToGridPixel(grid: OccupancyGrid, x: number, y: number): { x: number; y: number } {
  return {
    x: (x - grid.origin.x) / grid.resolution,
    y: grid.height - (y - grid.origin.y) / grid.resolution
  };
}

function screenToWorld(
  pixelX: number,
  pixelY: number,
  viewport: ViewportTransform,
  grid: OccupancyGrid
): { x: number; y: number } {
  const mapPixelX = (pixelX - viewport.translateX) / viewport.scale;
  const mapPixelY = (pixelY - viewport.translateY) / viewport.scale;

  return {
    x: grid.origin.x + mapPixelX * grid.resolution,
    y: grid.origin.y + (grid.height - mapPixelY) * grid.resolution
  };
}

function quaternionToYaw(quaternion: Pose3D["orientation"]): number {
  const sinyCosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y);
  const cosyCosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z);
  return Math.atan2(sinyCosp, cosyCosp);
}

function distance2D(start: { x: number; y: number }, end: { x: number; y: number }): number {
  return Math.hypot(end.x - start.x, end.y - start.y);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function applyGridEdits(
  grid: OccupancyGrid,
  obstacles: ObstacleRect2D[],
  erasures: ObstacleRect2D[]
): OccupancyGrid {
  if (obstacles.length === 0 && erasures.length === 0) {
    return grid;
  }

  const nextGrid: OccupancyGrid = {
    ...grid,
    data: new Int8Array(grid.data),
    hits: new Uint16Array(grid.hits)
  };

  applyRectEdits(nextGrid, erasures, "clear");
  applyRectEdits(nextGrid, obstacles, "occupy");
  return nextGrid;
}

function applyRectEdits(
  grid: OccupancyGrid,
  rects: ObstacleRect2D[],
  mode: "clear" | "occupy"
): void {
  for (const rect of rects) {
    const bounds = rectToCellBounds(grid, rect);
    if (!bounds) {
      continue;
    }
    for (let y = bounds.minY; y <= bounds.maxY; y += 1) {
      for (let x = bounds.minX; x <= bounds.maxX; x += 1) {
        const index = cellToLinearIndex(grid, { x, y });
        if (mode === "occupy") {
          grid.data[index] = 100;
          grid.hits[index] = Math.max(grid.hits[index], 1);
        } else {
          grid.data[index] = 0;
          grid.hits[index] = 0;
        }
      }
    }
  }
}

function rectToCellBounds(
  grid: OccupancyGrid,
  rect: ObstacleRect2D
): { minX: number; maxX: number; minY: number; maxY: number } | null {
  const clampCell = (cell: { x: number; y: number } | null, fallback: { x: number; y: number }) =>
    cell ?? fallback;

  const min = clampCell(worldToCell(grid, rect.minX, rect.minY), { x: 0, y: 0 });
  const max = clampCell(worldToCell(grid, rect.maxX, rect.maxY), {
    x: grid.width - 1,
    y: grid.height - 1
  });

  const minX = Math.max(0, Math.min(min.x, max.x));
  const maxX = Math.min(grid.width - 1, Math.max(min.x, max.x));
  const minY = Math.max(0, Math.min(min.y, max.y));
  const maxY = Math.min(grid.height - 1, Math.max(min.y, max.y));

  if (minX > maxX || minY > maxY) {
    return null;
  }

  return { minX, maxX, minY, maxY };
}

function downloadGridAsPgm(grid: OccupancyGrid, baseName: string): void {
  const header = new TextEncoder().encode(`P5\n${grid.width} ${grid.height}\n255\n`);
  const pixels = new Uint8Array(grid.width * grid.height);

  for (let y = 0; y < grid.height; y += 1) {
    for (let x = 0; x < grid.width; x += 1) {
      const sourceIndex = y * grid.width + x;
      const targetIndex = (grid.height - 1 - y) * grid.width + x;
      const occupancy = grid.data[sourceIndex];
      pixels[targetIndex] = occupancy === 100 ? 0 : occupancy === -1 ? 205 : 254;
    }
  }

  const blob = new Blob([header, pixels], {
    type: "image/x-portable-graymap"
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${baseName}.pgm`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.setTimeout(() => URL.revokeObjectURL(url), 500);
}
