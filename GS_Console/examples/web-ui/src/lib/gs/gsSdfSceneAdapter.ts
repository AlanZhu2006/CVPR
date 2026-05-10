export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Quaternion {
  x: number;
  y: number;
  z: number;
  w: number;
}

export interface Bounds3Box {
  min: Vec3;
  max: Vec3;
}

export interface GaussianChunk {
  id?: string;
  url: string;
  bytes?: number;
  splats?: number;
  center?: Vec3;
  bounds?: Bounds3Box;
}

export interface GaussianSource {
  format: "gs-ply" | "spz" | "splat" | "gs-chunks";
  url?: string;
  chunks?: GaussianChunk[];
  shDegree: number;
  variant?: string;
  label?: string;
  frameRange?: {
    start: number;
    end: number;
  };
  atlasRole?: "local_chunk" | "global" | string;
  keyframeIndices?: number[];
  alignment?: Record<string, unknown>;
}

export interface Pose3D {
  position: Vec3;
  orientation: Quaternion;
  frameId: string;
  stampMs?: number;
}

export interface SceneInitialView {
  pose: Pose3D;
  target?: Vec3;
}

export interface SceneManifest {
  schemaVersion?: string;
  sceneId: string;
  frameId: string;
  camera?: {
    width?: number | null;
    height?: number | null;
    fx?: number | null;
    fy?: number | null;
    cx?: number | null;
    cy?: number | null;
    tCL?: number[] | null;
    tBL?: number[] | null;
  };
  initialView?: SceneInitialView;
  gaussian?: GaussianSource;
  gaussianVariants?: Record<string, GaussianSource>;
  occupancy?: {
    source: "as_occ_prior" | "mesh_projection" | "prebuilt_grid";
    url: string;
    resolution: number;
    origin: { x: number; y: number };
  };
  mesh?: {
    format: "ply" | "glb";
    url: string;
  };
  rawPointCloud?: {
    format: "ply" | "pcd";
    url: string;
  };
  externalViewer?: {
    kind?: string;
    label?: string;
    url: string;
    notes?: string;
  };
  training?: {
    status: string;
    outputDir: string;
    configPath?: string | null;
  };
  robot?: {
    radius: number;
    height: number;
  };
  meta?: {
    leafSize?: number | null;
    mapOrigin?: Vec3;
    mapSize?: number | null;
  };
  assets?: Record<string, string>;
  source?: {
    kind?: string;
    outputDir?: string;
    repoPath?: string;
    repoCommit?: string | null;
    visualManifestUrl?: string;
    liveGaussianPatchMode?: "replace" | "fallback" | "disabled";
    liveGaussianPolicy?: "chunk_atlas" | "single_global" | string;
    chunkCrossFadeFrames?: number;
    globalGaussianPolicy?: string;
    liveContractUrl?: string;
    capabilityMatrixUrl?: string;
  };
}

export interface LayerVisibility {
  gaussian: boolean;
  sdfMesh: boolean;
  occupancyCloud: boolean;
  liveCloud: boolean;
  trajectory: boolean;
  robot: boolean;
}

export interface GaussianScene {
  chunkCount: number;
  splatCount: number;
}

export interface CameraState {
  position: Vec3;
  target: Vec3;
  up: Vec3;
  fovYRad: number;
}

export interface GaussianRendererBackend {
  initialize(canvas: HTMLCanvasElement, options: {
    frameId: string;
    shDegree: number;
  }): Promise<void>;
  loadScene(source: SceneManifest["gaussian"]): Promise<GaussianScene>;
  setLayerVisibility(visibility: LayerVisibility): void;
  updateCamera(camera: CameraState): void;
  updateRobotPose(pose: Pose3D | null): void;
  updateTrajectory(path: Pose3D[]): void;
  render(): void;
}

export interface OverlaySceneBackend {
  loadMesh(url: string, format: "ply" | "glb"): Promise<void>;
  loadPointCloud(url: string, format: "ply" | "pcd"): Promise<void>;
  setLayerVisibility(visibility: LayerVisibility): void;
  updateRobotPose(pose: Pose3D | null): void;
  updateTrajectory(path: Pose3D[]): void;
}

export interface TopDownSyncBackend {
  setOccupancySource(url: string): Promise<void>;
  updateRobotPose(pose: Pose3D | null): void;
  updateTrajectory(path: Pose3D[]): void;
}

export class GsSdfSceneAdapter {
  private manifest: SceneManifest | null = null;
  private canvasInitialized = false;
  private gaussianUrl: string | null = null;
  private meshUrl: string | null = null;
  private pointCloudUrl: string | null = null;
  private occupancyUrl: string | null = null;
  private visibility: LayerVisibility = {
    gaussian: true,
    sdfMesh: true,
    occupancyCloud: false,
    liveCloud: false,
    trajectory: true,
    robot: true
  };

  constructor(
    private readonly gaussianRenderer: GaussianRendererBackend,
    private readonly overlayScene: OverlaySceneBackend,
    private readonly topDownSync: TopDownSyncBackend
  ) {}

  async bootstrap(canvas: HTMLCanvasElement, manifestUrl: string): Promise<SceneManifest> {
    const manifest = await fetchJson<SceneManifest>(manifestUrl);
    await this.ensureRendererInitialized(canvas, manifest);
    await this.syncAssets(manifest);
    this.applyVisibility(this.visibility);
    return manifest;
  }

  async refreshManifest(manifestUrl: string): Promise<SceneManifest> {
    if (!this.canvasInitialized) {
      throw new Error("Call bootstrap() before refreshManifest().");
    }

    const manifest = await fetchJson<SceneManifest>(manifestUrl);
    await this.syncAssets(manifest);
    this.applyVisibility(this.visibility);
    return manifest;
  }

  applyVisibility(next: Partial<LayerVisibility>): void {
    this.visibility = { ...this.visibility, ...next };
    this.gaussianRenderer.setLayerVisibility(this.visibility);
    this.overlayScene.setLayerVisibility(this.visibility);
  }

  updateRobotPose(pose: Pose3D | null): void {
    this.gaussianRenderer.updateRobotPose(pose);
    this.overlayScene.updateRobotPose(pose);
    this.topDownSync.updateRobotPose(pose);
  }

  updateTrajectory(path: Pose3D[]): void {
    this.gaussianRenderer.updateTrajectory(path);
    this.overlayScene.updateTrajectory(path);
    this.topDownSync.updateTrajectory(path);
  }

  updateCamera(camera: CameraState): void {
    this.gaussianRenderer.updateCamera(camera);
  }

  renderFrame(): void {
    this.gaussianRenderer.render();
  }

  getSceneFrameId(): string {
    if (!this.manifest) {
      throw new Error("Scene manifest has not been loaded yet.");
    }
    return this.manifest.frameId;
  }

  private async ensureRendererInitialized(
    canvas: HTMLCanvasElement,
    manifest: SceneManifest
  ): Promise<void> {
    if (this.canvasInitialized) {
      return;
    }

    await this.gaussianRenderer.initialize(canvas, {
      frameId: manifest.frameId,
      shDegree: manifest.gaussian?.shDegree ?? 3
    });

    this.canvasInitialized = true;
  }

  private async syncAssets(manifest: SceneManifest): Promise<void> {
    this.manifest = manifest;

    if (manifest.gaussian?.url && manifest.gaussian.url !== this.gaussianUrl) {
      await this.gaussianRenderer.loadScene(manifest.gaussian);
      this.gaussianUrl = manifest.gaussian.url;
    }

    if (manifest.mesh && manifest.mesh.url !== this.meshUrl) {
      await this.overlayScene.loadMesh(manifest.mesh.url, manifest.mesh.format);
      this.meshUrl = manifest.mesh.url;
    }

    if (manifest.rawPointCloud && manifest.rawPointCloud.url !== this.pointCloudUrl) {
      await this.overlayScene.loadPointCloud(
        manifest.rawPointCloud.url,
        manifest.rawPointCloud.format
      );
      this.pointCloudUrl = manifest.rawPointCloud.url;
    }

    if (manifest.occupancy && manifest.occupancy.url !== this.occupancyUrl) {
      await this.topDownSync.setOccupancySource(manifest.occupancy.url);
      this.occupancyUrl = manifest.occupancy.url;
    }
  }
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }
  return (await response.json()) as T;
}
