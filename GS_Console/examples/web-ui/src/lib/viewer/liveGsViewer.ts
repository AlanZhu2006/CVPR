import * as THREE from "three";
import * as Spark from "@sparkjsdev/spark";
import { MOUSE } from "three";
import { Line2 } from "three/examples/jsm/lines/Line2.js";
import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type {
  GaussianChunk,
  LayerVisibility,
  Pose3D,
  SceneManifest
} from "../gs/gsSdfSceneAdapter";
import type { PlanPoint2D } from "../map/localPlanner";
import { loadPlyGeometry } from "../ply/loadPly";
import type { DecodedPointCloud } from "../ros/pointCloud2";

export type ViewerMode = "live" | "playback" | "gs";
export type GsControlMode = "orbit" | "fps";
export type PresentationMode = "default" | "showcase";
export type LivePointCloudRenderStyle = "cloud" | "gaussian";

interface KeyboardState {
  forward: boolean;
  backward: boolean;
  left: boolean;
  right: boolean;
  up: boolean;
  down: boolean;
  boost: boolean;
}

interface PlaybackScanTrailEntry {
  object: THREE.Points;
  insertedAtMs: number;
  stampMs: number;
}

const GAUSSIAN_INITIAL_BYTE_BUDGET = {
  balanced: 10 * 1024 * 1024,
  quality: 16 * 1024 * 1024,
  ultra: 20 * 1024 * 1024,
  raw: 16 * 1024 * 1024
} as const;

const ROBOT_SMOOTHING_POSITION_ALPHA = 0.18;
const ROBOT_SMOOTHING_ROTATION_ALPHA = 0.22;

export class LiveGsViewer {
  private static readonly MAX_FPS_PITCH = THREE.MathUtils.degToRad(85);
  private static readonly FPS_MOUSE_SENSITIVITY = 0.0024;
  private static readonly INTERACTION_LOD_MS = 140;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(65, 1, 0.05, 1000);
  private readonly clock = new THREE.Clock();
  private readonly pointSpriteTexture = createPointSpriteTexture();
  private readonly orbitControls: OrbitControls;
  private readonly gaussianLayer = new THREE.Group();
  private readonly meshLayer = new THREE.Group();
  private readonly pointCloudLayer = new THREE.Group();
  private readonly sceneGuidesLayer = new THREE.Group();
  private sceneGuidesEnabled = true;
  private autoFrameGaussianOnLoad = true;
  private readonly staticPointCloudLayer = new THREE.Group();
  private readonly livePointCloudLayer = new THREE.Group();
  private readonly playbackMapPointCloudLayer = new THREE.Group();
  private readonly playbackScanGhostLayer = new THREE.Group();
  private readonly playbackScanPointCloudLayer = new THREE.Group();
  private readonly robotLayer = new THREE.Group();
  private readonly trajectoryLayer = new THREE.Group();
  private readonly trajectoryPoseMarkerLayer = new THREE.Group();
  private readonly plannedPathLayer = new THREE.Group();
  private readonly keyboard: KeyboardState = {
    forward: false,
    backward: false,
    left: false,
    right: false,
    up: false,
    down: false,
    boost: false
  };
  private animationHandle = 0;
  private manifest: SceneManifest | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private mode: ViewerMode = "live";
  private gsControlMode: GsControlMode = "orbit";
  private presentationMode: PresentationMode = "default";
  private livePointCloudRenderStyle: LivePointCloudRenderStyle = "cloud";
  /** When true in playback workspace, load Spark gaussian in-browser (cinema “Web GS”) instead of HiFi-only. */
  private playbackCompareUsesWebGs = false;
  private gaussianVariant = "balanced";
  private visibility: LayerVisibility = {
    gaussian: true,
    sdfMesh: true,
    occupancyCloud: false,
    liveCloud: false,
    trajectory: true,
    robot: true
  };
  private gaussianSourceKey: string | null = null;
  private gaussianLoadingKey: string | null = null;
  private gaussianLoadTask: Promise<void> | null = null;
  private gaussianLoadGeneration = 0;
  private meshUrl: string | null = null;
  private rawPointCloudUrl: string | null = null;
  private meshBounds: THREE.Box3 | null = null;
  private rawPointCloudBounds: THREE.Box3 | null = null;
  private gaussianBounds: THREE.Box3 | null = null;
  private robotMarker: THREE.Object3D;
  private trajectoryLine: Line2 | null = null;
  private plannedPathLine: THREE.Line | null = null;
  private latestTrajectory: Pose3D[] = [];
  private targetRobotPosition = new THREE.Vector3();
  private targetRobotQuaternion = new THREE.Quaternion();
  private hasTargetRobotPose = false;
  private livePointCloudObject: THREE.Points | null = null;
  private playbackMapPointCloudObject: THREE.Points | null = null;
  private playbackScanPointCloudObject: THREE.Points | null = null;
  private playbackScanPointCloudStampMs: number | null = null;
  private playbackScanTrailEntries: PlaybackScanTrailEntry[] = [];
  private latestLivePointCloud: DecodedPointCloud | null = null;
  private latestPlaybackMapPointCloud: DecodedPointCloud | null = null;
  private latestPlaybackScanPointCloud: DecodedPointCloud | null = null;
  private livePointCloudRevealCount = 0;
  private livePointCloudTargetCount = 0;
  private livePointCloudBounds: THREE.Box3 | null = null;
  private playbackMapPointCloudBounds: THREE.Box3 | null = null;
  private followRobot = false;
  private poseDrivenCamera = false;
  private cameraInteractionUntil = 0;
  private interactionRenderActive = false;
  private readonly followCameraOffset = new THREE.Vector3(-6, -6, 4);
  private readonly prefetchedGaussianUrls = new Set<string>();
  private gaussianPrefetchTask: Promise<void> | null = null;
  private trajectoryHeadLine: THREE.Line | null = null;
  private trajectoryHeadAnchor: THREE.Vector3 | null = null;
  private readonly sparkRenderer: (THREE.Object3D & {
    originDistance?: number;
    maxStdDev?: number;
    maxPixelRadius?: number;
    focalAdjustment?: number;
    minAlpha?: number;
    clipXY?: number;
    defaultView?: {
      sortRadial?: boolean;
      sortDistance?: number;
      sort32?: boolean;
      stochastic?: boolean;
    };
  }) | null = null;
  private fpsYaw = 0;
  private fpsPitch = 0;
  private pointerLocked = false;
  private readonly keyDownHandler = (event: KeyboardEvent) => this.handleKeyChange(true, event);
  private readonly keyUpHandler = (event: KeyboardEvent) => this.handleKeyChange(false, event);
  private readonly orbitStartHandler = () => this.bumpInteractionWindow(220);
  private readonly orbitChangeHandler = () => this.bumpInteractionWindow();
  private readonly orbitEndHandler = () => this.bumpInteractionWindow(90);
  private readonly pointerLockChangeHandler = () => {
    this.pointerLocked = document.pointerLockElement === this.canvas;
  };
  private readonly mouseMoveHandler = (event: MouseEvent) => this.handlePointerLook(event);

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.renderer = createWebGlRenderer(canvas);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.scene.background = new THREE.Color("#111318");
    this.scene.fog = new THREE.Fog("#111318", 30, 140);

    const SparkRendererCtor = (Spark as Record<string, unknown>).SparkRenderer as
      | (new (options: {
          renderer: THREE.WebGLRenderer;
          originDistance?: number;
          maxStdDev?: number;
          maxPixelRadius?: number;
          focalAdjustment?: number;
          premultipliedAlpha?: boolean;
          minAlpha?: number;
          clipXY?: number;
          view?: {
            sortRadial?: boolean;
            sortDistance?: number;
            sort32?: boolean;
            stochastic?: boolean;
          };
        }) => THREE.Object3D)
      | undefined;
    if (SparkRendererCtor) {
      const sparkRenderer = new SparkRendererCtor({
        renderer: this.renderer,
        premultipliedAlpha: true,
        originDistance: 0.85,
        maxStdDev: Math.sqrt(5.9),
        maxPixelRadius: 132,
        focalAdjustment: 1.22,
        minAlpha: 0.005,
        clipXY: 1.12,
        view: {
          sortRadial: false,
          sortDistance: 0.004,
          sort32: true,
          stochastic: false
        }
      }) as THREE.Object3D & {
        originDistance?: number;
        maxStdDev?: number;
        maxPixelRadius?: number;
        focalAdjustment?: number;
        minAlpha?: number;
        clipXY?: number;
        defaultView?: {
          sortRadial?: boolean;
          sortDistance?: number;
          sort32?: boolean;
          stochastic?: boolean;
        };
      };
      if (sparkRenderer.defaultView) {
        sparkRenderer.defaultView.sortRadial = false;
        sparkRenderer.defaultView.sortDistance = 0.004;
        sparkRenderer.defaultView.sort32 = true;
        sparkRenderer.defaultView.stochastic = false;
      }
      this.sparkRenderer = sparkRenderer;
      this.scene.add(sparkRenderer);
    }

    this.camera.position.set(-8, -10, 7);
    this.camera.up.set(0, 0, 1);

    this.orbitControls = new OrbitControls(this.camera, canvas);
    this.orbitControls.enableDamping = true;
    this.orbitControls.screenSpacePanning = true;
    this.orbitControls.mouseButtons = {
      LEFT: MOUSE.ROTATE,
      MIDDLE: MOUSE.PAN,
      RIGHT: MOUSE.PAN
    };
    this.orbitControls.target.set(0, 0, 1.2);
    this.orbitControls.update();
    this.orbitControls.addEventListener("start", this.orbitStartHandler);
    this.orbitControls.addEventListener("change", this.orbitChangeHandler);
    this.orbitControls.addEventListener("end", this.orbitEndHandler);

    this.canvas.tabIndex = 0;
    this.canvas.addEventListener("click", this.handleCanvasClick);
    window.addEventListener("keydown", this.keyDownHandler);
    window.addEventListener("keyup", this.keyUpHandler);
    document.addEventListener("pointerlockchange", this.pointerLockChangeHandler);
    document.addEventListener("mousemove", this.mouseMoveHandler);

    this.scene.add(new THREE.HemisphereLight("#f3f2d7", "#1a1d22", 1.8));

    const directional = new THREE.DirectionalLight("#fff5d0", 1.5);
    directional.position.set(8, -4, 16);
    this.scene.add(directional);

    const grid = new THREE.GridHelper(120, 120, "#7a91bd", "#314158");
    grid.rotateX(Math.PI / 2);
    const gridMaterial = grid.material as THREE.Material;
    gridMaterial.transparent = true;
    gridMaterial.opacity = 0.6;
    this.sceneGuidesLayer.add(grid);

    const axes = new THREE.AxesHelper(3);
    axes.position.set(0, 0, 0.02);
    this.sceneGuidesLayer.add(axes);

    this.robotMarker = createRobotMarker();
    this.robotLayer.add(this.robotMarker);

    this.pointCloudLayer.add(this.staticPointCloudLayer);
    this.pointCloudLayer.add(this.livePointCloudLayer);
    this.pointCloudLayer.add(this.playbackMapPointCloudLayer);
    this.pointCloudLayer.add(this.playbackScanGhostLayer);
    this.pointCloudLayer.add(this.playbackScanPointCloudLayer);

    this.scene.add(this.gaussianLayer);
    this.scene.add(this.meshLayer);
    this.scene.add(this.pointCloudLayer);
    this.scene.add(this.sceneGuidesLayer);
    this.scene.add(this.trajectoryLayer);
    this.scene.add(this.trajectoryPoseMarkerLayer);
    this.scene.add(this.plannedPathLayer);
    this.scene.add(this.robotLayer);

    this.setMode("live");
    this.applyPresentationMode();
    this.installResizeObserver();
    this.applyRenderProfile();
    this.renderLoop();
  }

  setMode(mode: ViewerMode): void {
    this.mode = mode;
    this.orbitControls.enabled = mode !== "gs" || this.gsControlMode === "orbit";
    this.applyPresentationMode();

    if (mode === "gs") {
      if (this.gsControlMode === "fps") {
        this.snapCameraToRobot();
      } else {
        this.frameGsScene();
      }
    } else if (mode === "playback") {
      this.unlockPointerLook();
      if (!this.isPlaybackCompareWebGsActive()) {
        this.framePlaybackScene();
      }
    } else {
      this.unlockPointerLook();
      this.frameLiveScene();
    }

    this.applyRenderProfile();
    if (!this.shouldLoadGaussian()) {
      this.gaussianLoadGeneration += 1;
      this.gaussianLoadingKey = null;
    }
    void this.ensureDeferredAssets();
  }

  setGsControlMode(controlMode: GsControlMode): void {
    this.gsControlMode = controlMode;
    this.orbitControls.enabled = this.mode !== "gs" || controlMode === "orbit";

    if (controlMode === "fps") {
      this.snapCameraToRobot();
    } else {
      this.unlockPointerLook();
      if (this.mode === "gs") {
        this.frameGsScene();
      }
    }

    this.applyRenderProfile();
  }

  setPresentationMode(mode: PresentationMode): void {
    this.presentationMode = mode;
    this.applyPresentationMode();
    this.refreshDynamicPointClouds();
    if (this.latestTrajectory.length > 0) {
      this.updateTrajectory(this.latestTrajectory);
    }
    this.applyRenderProfile();
    if (this.mode === "playback" && !this.isPlaybackCompareWebGsActive()) {
      this.framePlaybackScene();
    }
    void this.ensureDeferredAssets();
  }

  setSceneGuidesEnabled(enabled: boolean): void {
    this.sceneGuidesEnabled = enabled;
    this.applyPresentationMode();
  }

  setAutoFrameGaussianOnLoad(enabled: boolean): void {
    this.autoFrameGaussianOnLoad = enabled;
  }

  setPoseDrivenCamera(enabled: boolean): void {
    this.poseDrivenCamera = enabled;
    this.orbitControls.enabled = !enabled;
  }

  setCameraModel(camera: SceneManifest["camera"] | undefined): void {
    const height = typeof camera?.height === "number" ? camera.height : null;
    const fy = typeof camera?.fy === "number" ? camera.fy : null;
    if (!height || !fy || height <= 0 || fy <= 0) {
      return;
    }

    this.camera.fov = THREE.MathUtils.clamp(
      THREE.MathUtils.radToDeg(2 * Math.atan(height / (2 * fy))),
      15,
      100
    );
    this.camera.updateProjectionMatrix();
  }

  setPlaybackCompareUsesWebGs(enabled: boolean): void {
    if (this.playbackCompareUsesWebGs === enabled) {
      return;
    }
    this.playbackCompareUsesWebGs = enabled;
    if (this.mode === "playback" && !this.shouldLoadGaussian()) {
      this.gaussianLoadGeneration += 1;
      this.gaussianLoadingKey = null;
      this.clearGroup(this.gaussianLayer);
      this.gaussianSourceKey = null;
      this.gaussianBounds = null;
    }
    void this.ensureDeferredAssets();
    this.applyRenderProfile();
    if (this.mode === "playback" && !enabled) {
      this.framePlaybackScene();
    }
  }

  setLivePointCloudRenderStyle(style: LivePointCloudRenderStyle): void {
    if (this.livePointCloudRenderStyle === style) {
      return;
    }
    this.livePointCloudRenderStyle = style;
    this.refreshDynamicPointClouds();
  }

  setGaussianVariant(variant: string | null): void {
    this.gaussianVariant = variant ?? "balanced";
    this.applyRenderProfile();
  }

  setVisibility(visibility: LayerVisibility): void {
    this.visibility = visibility;
    this.gaussianLayer.visible = visibility.gaussian;
    this.meshLayer.visible = visibility.sdfMesh;
    this.pointCloudLayer.visible = visibility.occupancyCloud || visibility.liveCloud;
    const playbackMode = this.mode === "playback";
    this.staticPointCloudLayer.visible = visibility.occupancyCloud && !playbackMode;
    this.livePointCloudLayer.visible = visibility.liveCloud && !playbackMode;
    this.playbackMapPointCloudLayer.visible = visibility.occupancyCloud && playbackMode;
    this.playbackScanGhostLayer.visible = visibility.liveCloud && playbackMode;
    this.playbackScanPointCloudLayer.visible = visibility.liveCloud && playbackMode;
    if (visibility.liveCloud && !playbackMode && this.latestLivePointCloud) {
      this.renderLivePointCloud(this.latestLivePointCloud);
    }
    if (visibility.occupancyCloud && playbackMode && this.latestPlaybackMapPointCloud) {
      this.renderPlaybackMapPointCloud(this.latestPlaybackMapPointCloud);
    }
    if (visibility.liveCloud && playbackMode && this.latestPlaybackScanPointCloud) {
      this.renderPlaybackScanPointCloud(this.latestPlaybackScanPointCloud);
    } else if (!visibility.liveCloud && playbackMode) {
      this.clearPlaybackScanTrail();
    }
    this.robotLayer.visible = visibility.robot;
    this.trajectoryLayer.visible = visibility.trajectory;
    this.trajectoryPoseMarkerLayer.visible = visibility.trajectory;
    this.plannedPathLayer.visible = visibility.trajectory;
    this.applyRenderProfile();
    if (!this.shouldLoadGaussian()) {
      this.gaussianLoadGeneration += 1;
      this.gaussianLoadingKey = null;
    }
    void this.ensureDeferredAssets();
  }

  setFollowRobot(followRobot: boolean): void {
    this.followRobot = followRobot;

    if (!followRobot) {
      return;
    }

    const currentOffset = this.camera.position.clone().sub(this.orbitControls.target);
    if (currentOffset.lengthSq() > 0.5) {
      this.followCameraOffset.copy(currentOffset);
    }
    this.syncFollowCamera(true);
  }

  async loadManifest(manifest: SceneManifest): Promise<void> {
    this.manifest = manifest;

    const work: Promise<void>[] = [];
    const nextGaussianKey = getGaussianSourceKey(manifest.gaussian);
    if (nextGaussianKey !== this.gaussianSourceKey && nextGaussianKey !== this.gaussianLoadingKey) {
      this.gaussianLoadGeneration += 1;
      this.gaussianSourceKey = null;
      this.gaussianLoadingKey = null;
    }

    if (manifest.mesh?.url && manifest.mesh.url !== this.meshUrl) {
      work.push(this.loadMesh(manifest.mesh.url));
    }

    if (manifest.rawPointCloud?.url && manifest.rawPointCloud.url !== this.rawPointCloudUrl) {
      work.push(this.loadRawPointCloud(manifest.rawPointCloud.url));
    }

    await Promise.all(work);
    void this.prefetchGaussianAssets(manifest);
    await this.ensureDeferredAssets();
    this.setVisibility(this.visibility);
  }

  updateRobotPose(pose: Pose3D | null): void {
    if (!pose) {
      return;
    }

    this.targetRobotPosition.set(pose.position.x, pose.position.y, pose.position.z);
    this.targetRobotQuaternion.set(
      pose.orientation.x,
      pose.orientation.y,
      pose.orientation.z,
      pose.orientation.w
    );
    if (!this.hasTargetRobotPose) {
      this.robotLayer.position.copy(this.targetRobotPosition);
      this.robotMarker.quaternion.copy(this.targetRobotQuaternion);
      this.hasTargetRobotPose = true;
    }
    this.syncFollowCamera();
  }

  updateTrajectory(path: Pose3D[]): void {
    this.latestTrajectory = path;
    if (this.trajectoryLine) {
      this.trajectoryLayer.remove(this.trajectoryLine);
      this.trajectoryLine.geometry.dispose();
      this.trajectoryLine.material.dispose();
      this.trajectoryLine = null;
    }
    this.clearGroup(this.trajectoryPoseMarkerLayer);
    this.trajectoryHeadAnchor = null;

    if (path.length < 1) {
      this.updateTrajectoryHead();
      return;
    }

    const lastPose = path[path.length - 1];
    this.trajectoryHeadAnchor = new THREE.Vector3(
      lastPose.position.x,
      lastPose.position.y,
      lastPose.position.z + 0.07
    );

    if (path.length < 2) {
      this.updateTrajectoryHead();
      return;
    }

    const sampledPath = smoothTrajectoryPath(
      path,
      this.presentationMode === "showcase" && this.mode === "playback" ? 6 : 4
    );
    const positions = new Array<number>(sampledPath.length * 3);
    const colors = new Array<number>(sampledPath.length * 3);

    for (let index = 0; index < sampledPath.length; index += 1) {
      const pose = sampledPath[index];
      const offset = index * 3;
      positions[offset] = pose.position.x;
      positions[offset + 1] = pose.position.y;
      positions[offset + 2] = pose.position.z + 0.07;

      const t = sampledPath.length <= 1 ? 0 : index / (sampledPath.length - 1);
      const color = new THREE.Color().setHSL((1 - t) * 0.74, 0.88, 0.58);
      colors[offset] = color.r;
      colors[offset + 1] = color.g;
      colors[offset + 2] = color.b;
    }

    const geometry = new LineGeometry();
    geometry.setPositions(positions);
    geometry.setColors(colors);

    const resolution = this.getViewportSize();
    const material = new LineMaterial({
      color: "#ffffff",
      vertexColors: true,
      linewidth: this.presentationMode === "showcase" && this.mode === "playback" ? 14 : 8,
      transparent: true,
      opacity: this.presentationMode === "showcase" && this.mode === "playback" ? 0.98 : 0.92,
      depthWrite: false,
      depthTest: false,
      dashed: false
    });
    material.resolution.set(resolution.x, resolution.y);

    this.trajectoryLine = new Line2(geometry, material);
    this.trajectoryLine.computeLineDistances();
    this.trajectoryLine.renderOrder = 28;
    this.trajectoryLayer.add(this.trajectoryLine);

    const markerIndices = pickTrajectoryMarkerIndices(
      path,
      this.presentationMode === "showcase" && this.mode === "playback" ? 144 : 64
    );
    for (const index of markerIndices) {
      const pose = path[index];
      const t = path.length <= 1 ? 0 : index / (path.length - 1);
      const color = new THREE.Color().setHSL((1 - t) * 0.74, 0.88, 0.58);
      const marker = createTrajectoryPoseMarker(
        color,
        this.presentationMode === "showcase" && this.mode === "playback" ? 1.55 : 1
      );
      marker.position.set(pose.position.x, pose.position.y, pose.position.z + 0.075);
      marker.quaternion.set(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
      );
      marker.renderOrder = 29;
      this.trajectoryPoseMarkerLayer.add(marker);
    }

    this.updateTrajectoryHead();
  }

  updatePlannedPath(path: PlanPoint2D[]): void {
    if (this.plannedPathLine) {
      this.plannedPathLayer.remove(this.plannedPathLine);
      this.plannedPathLine.geometry.dispose();
      (this.plannedPathLine.material as THREE.Material).dispose();
      this.plannedPathLine = null;
    }

    if (path.length < 2) {
      return;
    }

    const points = path.map((point) => new THREE.Vector3(point.x, point.y, 0.12));
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineDashedMaterial({
      color: "#ffd166",
      dashSize: 0.45,
      gapSize: 0.24
    });
    this.plannedPathLine = new THREE.Line(geometry, material);
    this.plannedPathLine.computeLineDistances();
    this.plannedPathLayer.add(this.plannedPathLine);
  }

  updateLivePointCloud(cloud: DecodedPointCloud | null): void {
    if (!cloud || cloud.renderedPointCount === 0) {
      this.latestLivePointCloud = null;
      this.livePointCloudRevealCount = 0;
      this.livePointCloudTargetCount = 0;
      if (this.livePointCloudObject) {
        this.livePointCloudLayer.remove(this.livePointCloudObject);
        this.livePointCloudObject.geometry.dispose();
        (this.livePointCloudObject.material as THREE.Material).dispose();
        this.livePointCloudObject = null;
      }
      return;
    }
    this.latestLivePointCloud = cloud;
    if (!this.visibility.liveCloud) {
      return;
    }
    this.renderLivePointCloud(cloud);
  }

  updatePlaybackMapPointCloud(cloud: DecodedPointCloud | null): void {
    this.latestPlaybackMapPointCloud = cloud;

    if (!cloud || cloud.renderedPointCount === 0) {
      this.playbackMapPointCloudBounds = null;
      this.disposePointCloudObject("playbackMap");
      return;
    }

    if (this.mode !== "playback" || !this.visibility.occupancyCloud) {
      return;
    }
    this.renderPlaybackMapPointCloud(cloud);
  }

  updatePlaybackScanPointCloud(cloud: DecodedPointCloud | null): void {
    this.latestPlaybackScanPointCloud = cloud;

    if (!cloud || cloud.renderedPointCount === 0) {
      this.disposePointCloudObject("playbackScan");
      return;
    }

    if (this.mode !== "playback" || !this.visibility.liveCloud) {
      return;
    }
    const showcasePlayback = this.presentationMode === "showcase" && this.mode === "playback";
    const currentStampMs = cloud.stampMs ?? 0;
    let previousSnapshot: THREE.Points | null = null;
    if (
      showcasePlayback &&
      this.playbackScanPointCloudObject &&
      this.playbackScanPointCloudStampMs !== null &&
      this.playbackScanPointCloudStampMs !== currentStampMs
    ) {
      previousSnapshot = clonePointCloudSnapshot(this.playbackScanPointCloudObject);
    }
    this.renderPlaybackScanPointCloud(cloud);
    if (showcasePlayback && previousSnapshot && this.playbackScanPointCloudStampMs !== null) {
      const material = previousSnapshot.material as THREE.PointsMaterial;
      material.transparent = true;
      material.opacity = 0.34;
      material.depthWrite = false;
      material.depthTest = false;
      material.needsUpdate = true;
      previousSnapshot.renderOrder = 24;
      this.playbackScanGhostLayer.add(previousSnapshot);
      this.playbackScanTrailEntries.push({
        object: previousSnapshot,
        insertedAtMs: performance.now(),
        stampMs: this.playbackScanPointCloudStampMs
      });
    }
    this.playbackScanPointCloudStampMs = currentStampMs;
    this.updatePlaybackScanTrail();
  }

  private renderLivePointCloud(cloud: DecodedPointCloud): void {
    const gaussianStyle = this.livePointCloudRenderStyle === "gaussian";
    const renderCloud = samplePointCloudForRender(
      cloud,
      gaussianStyle
        ? this.interactionRenderActive
          ? 36_000
          : 110_000
        : this.interactionRenderActive
          ? 28_000
          : 72_000
    );
    const hadBounds = Boolean(this.livePointCloudBounds);
    const nextBounds = computePointCloudBounds(renderCloud.positions);
    this.livePointCloudObject = this.renderPointCloudObject(
      this.livePointCloudObject,
      this.livePointCloudLayer,
      renderCloud,
      {
        defaultColor: "#78d8ff",
        sizeNear: gaussianStyle ? 0.064 : 0.024,
        sizeFar: gaussianStyle ? 0.036 : 0.014,
        opacity: gaussianStyle ? 0.72 : 0.96,
        transparent: gaussianStyle,
        depthWrite: !gaussianStyle,
        depthTest: true,
        alphaTest: gaussianStyle ? 0.06 : 0.42,
        crispSprite: true,
        bounds: nextBounds,
        renderOrder: gaussianStyle ? 18 : 0
      }
    );
    this.livePointCloudBounds = nextBounds;
    this.livePointCloudTargetCount = renderCloud.renderedPointCount;
    this.livePointCloudRevealCount = renderCloud.renderedPointCount;
    if (this.mode === "live" && !this.followRobot && !hadBounds && nextBounds && isFiniteBox(nextBounds)) {
      this.frameLiveScene();
    }
  }

  private renderPlaybackMapPointCloud(cloud: DecodedPointCloud): void {
    const showcasePlayback = this.presentationMode === "showcase" && this.mode === "playback";
    const renderPointBudget = showcasePlayback
      ? this.interactionRenderActive
        ? 260_000
        : 820_000
      : this.interactionRenderActive
        ? 64_000
        : 160_000;
    const renderCloud = showcasePlayback
      ? samplePointCloudPrioritizingColor(cloud, renderPointBudget)
      : samplePointCloudForRender(cloud, renderPointBudget);
    const hadBounds = Boolean(this.playbackMapPointCloudBounds);
    const nextBounds = computePointCloudBounds(renderCloud.positions);
    this.playbackMapPointCloudObject = this.renderPointCloudObject(
      this.playbackMapPointCloudObject,
      this.playbackMapPointCloudLayer,
      renderCloud,
      {
        defaultColor: "#d8ecff",
        sizeNear: showcasePlayback ? 0.019 : 0.022,
        sizeFar: showcasePlayback ? 0.011 : 0.013,
        opacity: showcasePlayback ? 0.97 : 0.985,
        transparent: false,
        depthWrite: true,
        depthTest: true,
        alphaTest: 0.42,
        crispSprite: true,
        bounds: nextBounds,
        renderOrder: 8
      }
    );
    this.playbackMapPointCloudBounds = nextBounds;
    if (
      this.mode === "playback" &&
      !this.followRobot &&
      !this.isPlaybackCompareWebGsActive() &&
      !hadBounds &&
      this.playbackMapPointCloudBounds
    ) {
      this.framePlaybackScene();
    }
  }

  private renderPlaybackScanPointCloud(cloud: DecodedPointCloud): void {
    const showcasePlayback = this.presentationMode === "showcase" && this.mode === "playback";
    const renderCloud = samplePointCloudForRender(
      cloud,
      showcasePlayback
        ? this.interactionRenderActive
          ? 48_000
          : 140_000
        : this.interactionRenderActive
          ? 6_000
          : 14_000
    );
    this.playbackScanPointCloudObject = this.renderPointCloudObject(
      this.playbackScanPointCloudObject,
      this.playbackScanPointCloudLayer,
      renderCloud,
      {
        defaultColor: "#ffd38c",
        sizeNear: showcasePlayback ? 0.024 : 0.03,
        sizeFar: showcasePlayback ? 0.014 : 0.019,
        opacity: showcasePlayback ? 0.995 : 0.84,
        transparent: showcasePlayback,
        depthWrite: !showcasePlayback,
        depthTest: !showcasePlayback,
        alphaTest: 0.48,
        crispSprite: true,
        bounds: this.playbackMapPointCloudBounds,
        renderOrder: showcasePlayback ? 32 : 12
      }
    );
  }

  private renderPointCloudObject(
    object: THREE.Points | null,
    layer: THREE.Group,
    cloud: DecodedPointCloud,
    options: {
      defaultColor: string;
      sizeNear: number;
      sizeFar: number;
      opacity: number;
      transparent?: boolean;
      depthWrite?: boolean;
      depthTest?: boolean;
      alphaTest?: number;
      crispSprite?: boolean;
      bounds?: THREE.Box3 | null;
      renderOrder?: number;
    }
  ): THREE.Points {
    const useVertexColors = Boolean(cloud.colors && cloud.colors.length === cloud.positions.length);
    if (!object) {
      const geometry = new THREE.BufferGeometry();
      const material = new THREE.PointsMaterial({
        color: useVertexColors ? "#ffffff" : options.defaultColor,
        vertexColors: useVertexColors,
        size: options.sizeNear,
        sizeAttenuation: true,
        opacity: options.opacity,
        transparent: options.transparent ?? true,
        map: options.crispSprite ? this.pointSpriteTexture : null,
        alphaMap: options.crispSprite ? this.pointSpriteTexture : null,
        alphaTest: options.alphaTest ?? 0,
        fog: false,
        depthWrite: options.depthWrite ?? false,
        depthTest: options.depthTest ?? true
      });
      object = new THREE.Points(geometry, material);
      object.frustumCulled = false;
      layer.add(object);
    }

    const geometry = object.geometry as THREE.BufferGeometry;
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(cloud.positions, 3).setUsage(THREE.DynamicDrawUsage)
    );
    if (useVertexColors && cloud.colors) {
      geometry.setAttribute(
        "color",
        new THREE.BufferAttribute(cloud.colors, 3).setUsage(THREE.DynamicDrawUsage)
      );
    } else if (geometry.getAttribute("color")) {
      geometry.deleteAttribute("color");
    }
    geometry.setDrawRange(0, cloud.renderedPointCount);

    const material = object.material as THREE.PointsMaterial;
    material.vertexColors = useVertexColors;
    material.color.set(useVertexColors ? "#ffffff" : options.defaultColor);
    material.opacity = options.opacity;
    material.transparent = options.transparent ?? true;
    material.depthWrite = options.depthWrite ?? false;
    material.depthTest = options.depthTest ?? true;
    material.alphaTest = options.alphaTest ?? 0;
    object.renderOrder = options.renderOrder ?? 0;
    material.map = options.crispSprite ? this.pointSpriteTexture : null;
    material.alphaMap = options.crispSprite ? this.pointSpriteTexture : null;
    material.size =
      this.resolvePointSize(
        cloud.renderedPointCount > 40_000
          ? options.sizeFar
          : cloud.renderedPointCount > 12_000
            ? (options.sizeNear + options.sizeFar) * 0.5
            : options.sizeNear,
        options.bounds ?? null
      );
    material.needsUpdate = true;
    return object;
  }

  getCameraPose(): Pose3D {
    return {
      frameId: "world",
      position: {
        x: this.camera.position.x,
        y: this.camera.position.y,
        z: this.camera.position.z
      },
      orientation: {
        x: this.camera.quaternion.x,
        y: this.camera.quaternion.y,
        z: this.camera.quaternion.z,
        w: this.camera.quaternion.w
      },
      stampMs: Date.now()
    };
  }

  isInteractionActive(): boolean {
    return this.interactionRenderActive || this.pointerLocked;
  }

  setCameraPose(pose: Pose3D, target?: { x: number; y: number; z: number }): void {
    this.camera.position.set(
      pose.position.x,
      pose.position.y,
      pose.position.z
    );

    const exactPose = !target;
    if (target) {
      this.orbitControls.target.set(target.x, target.y, target.z);
      this.camera.up.set(0, 0, 1);
      this.camera.lookAt(target.x, target.y, target.z);
    } else {
      this.camera.quaternion.set(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
      );
      const forwardTarget = new THREE.Vector3(0, 0, -1)
        .applyQuaternion(this.camera.quaternion)
        .add(this.camera.position);
      this.orbitControls.target.copy(forwardTarget);
    }

    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion).normalize();
    this.fpsYaw = Math.atan2(forward.y, forward.x);
    this.fpsPitch = THREE.MathUtils.clamp(
      Math.asin(THREE.MathUtils.clamp(forward.z, -1, 1)),
      -LiveGsViewer.MAX_FPS_PITCH,
      LiveGsViewer.MAX_FPS_PITCH
    );
    this.camera.up.set(0, 0, 1);
    if (!exactPose || !this.poseDrivenCamera) {
      this.orbitControls.update();
    } else {
      this.camera.updateMatrixWorld(true);
    }
  }

  dispose(): void {
    cancelAnimationFrame(this.animationHandle);
    this.gaussianLoadGeneration += 1;
    this.resizeObserver?.disconnect();
    this.canvas.removeEventListener("click", this.handleCanvasClick);
    window.removeEventListener("keydown", this.keyDownHandler);
    window.removeEventListener("keyup", this.keyUpHandler);
    document.removeEventListener("pointerlockchange", this.pointerLockChangeHandler);
    document.removeEventListener("mousemove", this.mouseMoveHandler);
    this.orbitControls.removeEventListener("start", this.orbitStartHandler);
    this.orbitControls.removeEventListener("change", this.orbitChangeHandler);
    this.orbitControls.removeEventListener("end", this.orbitEndHandler);
    this.orbitControls.dispose();
    this.pointSpriteTexture.dispose();
    this.renderer.dispose();
  }

  private async createGaussianObject(url: string): Promise<{
    object: THREE.Object3D;
    bounds: THREE.Box3 | null;
  }> {
    try {
      const SplatMeshCtor = (Spark as Record<string, unknown>).SplatMesh as
        | (new (options: { url: string }) => THREE.Object3D)
        | undefined;

      if (!SplatMeshCtor) {
        throw new Error("Spark SplatMesh export was not found.");
      }

      const splatMesh = new SplatMeshCtor({ url }) as THREE.Object3D & {
        initialized?: Promise<unknown>;
        getBoundingBox?: (centersOnly?: boolean) => THREE.Box3;
      };
      splatMesh.frustumCulled = false;
      if (splatMesh.initialized) {
        await splatMesh.initialized;
      }
      const splatBounds = splatMesh.getBoundingBox?.();
      return {
        object: splatMesh,
        bounds: splatBounds && isFiniteBox(splatBounds) ? splatBounds : null
      };
    } catch (error) {
      console.warn("Failed to initialize Spark gaussian layer.", error);
      try {
        const geometry = await loadPlyGeometry(url);
        geometry.computeBoundingBox();
        const useVertexColors = ensureGeometryVertexColors(geometry);
        const fallbackPoints = new THREE.Points(
          geometry,
          new THREE.PointsMaterial({
            size: 0.045,
            color: useVertexColors ? "#ffffff" : "#d8dee6",
            vertexColors: useVertexColors,
            map: this.pointSpriteTexture,
            alphaMap: this.pointSpriteTexture,
            alphaTest: 0.35,
            fog: false,
            depthWrite: true,
            transparent: false
          })
        );
        return {
          object: fallbackPoints,
          bounds: geometry.boundingBox ?? new THREE.Box3().setFromObject(fallbackPoints)
        };
      } catch (fallbackError) {
        console.warn("Failed to load gaussian PLY fallback.", fallbackError);
        const fallback = new THREE.Mesh(
          new THREE.BoxGeometry(1.2, 1.2, 1.2),
          new THREE.MeshStandardMaterial({
            color: "#c78f54",
            wireframe: true
          })
        );
        fallback.position.set(0, 0, 0.6);
        return {
          object: fallback,
          bounds: new THREE.Box3().setFromObject(fallback)
        };
      }
    }
  }

  private async loadSingleGaussian(url: string, sourceKey: string, generation: number): Promise<void> {
    const { object, bounds } = await this.createGaussianObject(url);
    if (generation !== this.gaussianLoadGeneration) {
      disposeObject(object);
      return;
    }
    this.clearGroup(this.gaussianLayer);
    this.gaussianLayer.add(object);
    this.gaussianBounds = bounds;
    if (bounds && isFiniteBox(bounds) && this.autoFrameGaussianOnLoad) {
      if (this.mode === "playback" && !this.isPlaybackCompareWebGsActive()) {
        this.framePlaybackScene();
      } else if (this.mode !== "playback") {
        this.fitCameraToBox(bounds, 1.2);
      }
    }
    this.gaussianSourceKey = sourceKey;
  }

  private async loadGaussianChunks(
    chunks: GaussianChunk[],
    sourceKey: string,
    generation: number
  ): Promise<void> {
    const nextGroup = new THREE.Group();
    const combinedBounds = combineChunkBounds(chunks);
    const orderedChunks = orderGaussianChunks(chunks, this.camera.position, combinedBounds);

    if (combinedBounds && isFiniteBox(combinedBounds)) {
      this.gaussianBounds = combinedBounds;
      if (!this.autoFrameGaussianOnLoad) {
        // Camera is being driven externally, for example by a synced RGB feed.
      } else if (this.mode === "playback" && !this.isPlaybackCompareWebGsActive()) {
        this.framePlaybackScene();
      } else if (this.mode !== "playback") {
        this.fitCameraToBox(combinedBounds, 1.05);
      }
    }

    const [initialBatch, deferredBatch] = splitGaussianChunksByBudget(
      orderedChunks,
      this.gaussianVariant
    );

    await this.loadGaussianChunkBatch(initialBatch, generation, 2, nextGroup);
    if (generation !== this.gaussianLoadGeneration) {
      disposeObject(nextGroup);
      return;
    }

    this.clearGroup(this.gaussianLayer);
    this.gaussianLayer.add(nextGroup);
    this.gaussianSourceKey = sourceKey;

    window.setTimeout(async () => {
      await this.loadGaussianChunkBatch(deferredBatch, generation, 3, nextGroup);
      if (generation === this.gaussianLoadGeneration) {
        this.gaussianBounds = combinedBounds;
        this.gaussianSourceKey = sourceKey;
      }
    }, 0);

    this.gaussianBounds = combinedBounds;
    this.gaussianSourceKey = sourceKey;
  }

  private async loadGaussianChunkBatch(
    chunks: GaussianChunk[],
    generation: number,
    concurrency: number,
    targetGroup: THREE.Group
  ): Promise<void> {
    if (!chunks.length) {
      return;
    }

    let nextIndex = 0;
    const workerCount = Math.min(Math.max(concurrency, 1), chunks.length);
    const workers = new Array(workerCount).fill(0).map(async () => {
      while (true) {
        const currentIndex = nextIndex;
        nextIndex += 1;
        if (currentIndex >= chunks.length || generation !== this.gaussianLoadGeneration) {
          return;
        }

        const chunk = chunks[currentIndex];
        const { object } = await this.createGaussianObject(chunk.url);
        if (generation !== this.gaussianLoadGeneration) {
          disposeObject(object);
          return;
        }

        targetGroup.add(object);
        await new Promise((resolve) => window.setTimeout(resolve, 0));
      }
    });

    await Promise.all(workers);
  }

  private async loadMesh(url: string): Promise<void> {
    this.clearGroup(this.meshLayer);
    const geometry = await loadPlyGeometry(url);
    geometry.computeBoundingBox();
    const position = geometry.getAttribute("position");
    const vertexCount = position?.count ?? 0;
    const useVertexColors = ensureGeometryVertexColors(geometry);
    const shouldComputeNormals =
      Boolean(geometry.index) && !geometry.getAttribute("normal") && vertexCount <= 600_000;

    if (shouldComputeNormals) {
      geometry.computeVertexNormals();
    }

    const material = geometry.index
      ? new THREE.MeshBasicMaterial({
          color: "#ffffff",
          vertexColors: useVertexColors,
          transparent: true,
          opacity: 0.96,
          side: THREE.DoubleSide,
          fog: false
        })
      : new THREE.PointsMaterial({
          color: useVertexColors ? "#ffffff" : "#d8dee6",
          vertexColors: useVertexColors,
          size: 0.034,
          map: this.pointSpriteTexture,
          alphaMap: this.pointSpriteTexture,
          alphaTest: 0.4,
          fog: false,
          depthWrite: true,
          transparent: false
        });

    const object = geometry.index
      ? new THREE.Mesh(geometry, material as THREE.MeshBasicMaterial)
      : new THREE.Points(geometry, material as THREE.PointsMaterial);

    this.meshLayer.add(object);
    const bounds = geometry.boundingBox ?? new THREE.Box3().setFromObject(object);
    this.meshBounds = bounds;
    if (isFiniteBox(bounds)) {
      if (this.mode === "gs") {
        this.frameGsScene();
      } else if (this.mode === "playback") {
        this.framePlaybackScene();
      } else {
        this.frameLiveScene();
      }
    }
    this.meshUrl = url;
  }

  private async loadRawPointCloud(url: string): Promise<void> {
    this.clearGroup(this.staticPointCloudLayer);
    const geometry = await loadPlyGeometry(url);
    geometry.computeBoundingBox();
    const useVertexColors = ensureGeometryVertexColors(geometry);
    const object = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        size: 0.032,
        color: useVertexColors ? "#ffffff" : "#d7dee8",
        vertexColors: useVertexColors,
        map: this.pointSpriteTexture,
        alphaMap: this.pointSpriteTexture,
        alphaTest: 0.4,
        fog: false,
        depthWrite: true,
        transparent: false
      })
    );
    this.staticPointCloudLayer.add(object);
    const bounds = geometry.boundingBox ?? new THREE.Box3().setFromObject(object);
    this.rawPointCloudBounds = bounds;
    if (isFiniteBox(bounds)) {
      if (this.mode === "gs") {
        this.frameGsScene();
      } else if (this.mode === "playback") {
        this.framePlaybackScene();
      } else {
        this.frameLiveScene();
      }
    }
    this.rawPointCloudUrl = url;
  }

  private clearGroup(group: THREE.Group): void {
    for (const child of [...group.children]) {
      group.remove(child);
      if ("geometry" in child && child.geometry instanceof THREE.BufferGeometry) {
        child.geometry.dispose();
      }
      if ("material" in child) {
        const material = child.material;
        if (Array.isArray(material)) {
          material.forEach((item) => item.dispose());
        } else if (material instanceof THREE.Material) {
          material.dispose();
        }
      }
    }
  }

  private renderLoop = (): void => {
    const delta = this.clock.getDelta();
    this.animationHandle = requestAnimationFrame(this.renderLoop);
    this.syncInteractionRenderState();

    const orbitStyleCamera =
      this.mode === "live" ||
      this.mode === "playback" ||
      (this.mode === "gs" && this.gsControlMode === "orbit");

    if (orbitStyleCamera) {
      if (!this.poseDrivenCamera) {
        this.orbitControls.update();
      }
      this.updateRobotSmoothing();
      this.updateTrajectoryHead();
      this.updatePlaybackScanTrail();
      this.syncFollowCamera();
    } else {
      this.updateFps(delta);
      this.clampFpsOrientation();
      this.updatePlaybackScanTrail();
    }

    this.updateLivePointCloudReveal(delta);

    this.renderer.render(this.scene, this.camera);
  };

  private updateLivePointCloudReveal(delta: number): void {
    if (!this.livePointCloudObject) {
      return;
    }

    if (this.livePointCloudRevealCount >= this.livePointCloudTargetCount) {
      return;
    }

    const remaining = this.livePointCloudTargetCount - this.livePointCloudRevealCount;
    const step = Math.max(
      Math.ceil(remaining * Math.min(delta * 14, 0.55)),
      Math.ceil(9000 * delta)
    );
    this.livePointCloudRevealCount = Math.min(
      this.livePointCloudRevealCount + step,
      this.livePointCloudTargetCount
    );
    (this.livePointCloudObject.geometry as THREE.BufferGeometry).setDrawRange(
      0,
      this.livePointCloudRevealCount
    );
  }

  private updateRobotSmoothing(): void {
    if (!this.hasTargetRobotPose) {
      return;
    }

    this.robotLayer.position.lerp(this.targetRobotPosition, ROBOT_SMOOTHING_POSITION_ALPHA);
    this.robotMarker.quaternion.slerp(this.targetRobotQuaternion, ROBOT_SMOOTHING_ROTATION_ALPHA);
  }

  private updateTrajectoryHead(): void {
    if (!this.trajectoryHeadAnchor || !this.hasTargetRobotPose || !this.visibility.trajectory) {
      if (this.trajectoryHeadLine) {
        this.trajectoryHeadLine.visible = false;
      }
      return;
    }

    const headPosition = this.robotLayer.position.clone();
    headPosition.z += 0.07;
    if (headPosition.distanceToSquared(this.trajectoryHeadAnchor) <= 1e-6) {
      if (this.trajectoryHeadLine) {
        this.trajectoryHeadLine.visible = false;
      }
      return;
    }

    if (!this.trajectoryHeadLine) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(6), 3));
      const material = new THREE.LineBasicMaterial({
        color: "#ff8d72",
        transparent: true,
        opacity: 0.92,
        depthWrite: false,
        depthTest: false
      });
      this.trajectoryHeadLine = new THREE.Line(geometry, material);
      this.trajectoryHeadLine.renderOrder = 30;
      this.trajectoryLayer.add(this.trajectoryHeadLine);
    }

    const geometry = this.trajectoryHeadLine.geometry as THREE.BufferGeometry;
    const position = geometry.getAttribute("position") as THREE.BufferAttribute;
    position.setXYZ(0, this.trajectoryHeadAnchor.x, this.trajectoryHeadAnchor.y, this.trajectoryHeadAnchor.z);
    position.setXYZ(1, headPosition.x, headPosition.y, headPosition.z);
    position.needsUpdate = true;
    geometry.computeBoundingSphere();
    this.trajectoryHeadLine.visible = true;
  }

  private updatePlaybackScanTrail(): void {
    if (!this.playbackScanTrailEntries.length) {
      return;
    }

    if (this.mode !== "playback" || this.presentationMode !== "showcase" || !this.visibility.liveCloud) {
      this.clearPlaybackScanTrail();
      return;
    }

    const now = performance.now();
    const lifetimeMs = 700;
    const maxTrailEntries = 5;
    const survivors: PlaybackScanTrailEntry[] = [];
    for (let index = 0; index < this.playbackScanTrailEntries.length; index += 1) {
      const entry = this.playbackScanTrailEntries[index];
      const ageMs = now - entry.insertedAtMs;
      const expired = ageMs >= lifetimeMs;
      const overBudget = this.playbackScanTrailEntries.length - index > maxTrailEntries;
      if (expired || overBudget) {
        this.playbackScanGhostLayer.remove(entry.object);
        entry.object.geometry.dispose();
        (entry.object.material as THREE.Material).dispose();
        continue;
      }
      const progress = THREE.MathUtils.clamp(ageMs / lifetimeMs, 0, 1);
      const material = entry.object.material as THREE.PointsMaterial;
      material.opacity = (1 - progress) * 0.34;
      material.needsUpdate = true;
      survivors.push(entry);
    }
    this.playbackScanTrailEntries = survivors;
  }

  private updateFps(delta: number): void {
    if (this.mode !== "gs" || this.gsControlMode !== "fps") {
      return;
    }

    const speed = this.keyboard.boost ? 12 : 4.5;
    const distance = speed * delta;

    if (!this.pointerLocked) {
      return;
    }

    const forward = new THREE.Vector3(Math.cos(this.fpsYaw), Math.sin(this.fpsYaw), 0);
    const strafe = new THREE.Vector3(-Math.sin(this.fpsYaw), Math.cos(this.fpsYaw), 0);

    if (this.keyboard.forward) {
      this.camera.position.addScaledVector(forward, distance);
    }
    if (this.keyboard.backward) {
      this.camera.position.addScaledVector(forward, -distance);
    }
    if (this.keyboard.left) {
      this.camera.position.addScaledVector(strafe, distance);
    }
    if (this.keyboard.right) {
      this.camera.position.addScaledVector(strafe, -distance);
    }
    if (this.keyboard.up) {
      this.camera.position.z += distance;
    }
    if (this.keyboard.down) {
      this.camera.position.z -= distance;
    }
  }

  private handleCanvasClick = (): void => {
    if (this.mode !== "gs" || this.gsControlMode !== "fps" || this.pointerLocked) {
      return;
    }
    void this.canvas.requestPointerLock();
  };

  private handleKeyChange(active: boolean, event: KeyboardEvent): void {
    switch (event.code) {
      case "KeyW":
        this.keyboard.forward = active;
        break;
      case "KeyS":
        this.keyboard.backward = active;
        break;
      case "KeyA":
        this.keyboard.left = active;
        break;
      case "KeyD":
        this.keyboard.right = active;
        break;
      case "KeyQ":
        this.keyboard.down = active;
        break;
      case "KeyE":
        this.keyboard.up = active;
        break;
      case "ShiftLeft":
      case "ShiftRight":
        this.keyboard.boost = active;
        break;
      default:
        break;
    }
  }

  private snapCameraToRobot(): void {
    const position = this.robotLayer.position;
    this.camera.position.set(position.x - 1.2, position.y - 1.2, position.z + 1.3);
    this.camera.up.set(0, 0, 1);

    const forward = this.orbitControls.target.clone().sub(this.camera.position);
    if (forward.lengthSq() < 1e-4) {
      forward.set(1, 0, 0);
    }
    forward.normalize();

    this.fpsYaw = Math.atan2(forward.y, forward.x);
    this.fpsPitch = THREE.MathUtils.clamp(
      Math.asin(THREE.MathUtils.clamp(forward.z, -1, 1)),
      -LiveGsViewer.MAX_FPS_PITCH,
      LiveGsViewer.MAX_FPS_PITCH
    );
    this.applyFpsOrientation();
  }

  private clampFpsOrientation(): void {
    this.fpsPitch = THREE.MathUtils.clamp(
      this.fpsPitch,
      -LiveGsViewer.MAX_FPS_PITCH,
      LiveGsViewer.MAX_FPS_PITCH
    );
    this.applyFpsOrientation();
  }

  private applyFpsOrientation(): void {
    const horizontal = Math.cos(this.fpsPitch);
    const direction = new THREE.Vector3(
      Math.cos(this.fpsYaw) * horizontal,
      Math.sin(this.fpsYaw) * horizontal,
      Math.sin(this.fpsPitch)
    );
    const lookTarget = this.camera.position.clone().add(direction);
    this.camera.up.set(0, 0, 1);
    this.camera.lookAt(lookTarget);
  }

  private handlePointerLook(event: MouseEvent): void {
    if (!this.pointerLocked || this.mode !== "gs" || this.gsControlMode !== "fps") {
      return;
    }

    this.fpsYaw -= event.movementX * LiveGsViewer.FPS_MOUSE_SENSITIVITY;
    this.fpsPitch = THREE.MathUtils.clamp(
      this.fpsPitch - event.movementY * LiveGsViewer.FPS_MOUSE_SENSITIVITY,
      -LiveGsViewer.MAX_FPS_PITCH,
      LiveGsViewer.MAX_FPS_PITCH
    );
    this.applyFpsOrientation();
  }

  private unlockPointerLook(): void {
    if (document.pointerLockElement === this.canvas) {
      void document.exitPointerLock();
    }
    this.pointerLocked = false;
  }

  private frameLiveScene(): void {
    if (this.mode !== "live") {
      return;
    }

    if (this.followRobot) {
      this.syncFollowCamera(true);
      return;
    }

    const preferredBounds = this.getPreferredLiveBounds();
    if (preferredBounds) {
      this.fitCameraToBox(preferredBounds, 1.08);
      return;
    }

    this.camera.position.set(-8, -10, 7);
    this.camera.up.set(0, 0, 1);
    this.orbitControls.target.set(0, 0, 1.2);
    this.orbitControls.update();
  }

  private framePlaybackScene(): void {
    if (this.mode !== "playback") {
      return;
    }

    if (this.isPlaybackCompareWebGsActive()) {
      return;
    }

    if (this.followRobot) {
      this.syncFollowCamera(true);
      return;
    }

    const mapBox = this.playbackMapPointCloudBounds;
    const gaussianBox = this.visibility.gaussian ? this.gaussianBounds : null;
    const meshBox = this.visibility.sdfMesh ? this.meshBounds : null;
    let preferredBounds: THREE.Box3 | null = null;

    if (mapBox && isFiniteBox(mapBox)) {
      preferredBounds = mapBox.clone();
      if (gaussianBox && isFiniteBox(gaussianBox)) {
        preferredBounds.union(gaussianBox);
      }
      if (meshBox && isFiniteBox(meshBox)) {
        preferredBounds.union(meshBox);
      }
    } else if (gaussianBox && isFiniteBox(gaussianBox)) {
      preferredBounds = gaussianBox.clone();
    } else if (meshBox && isFiniteBox(meshBox)) {
      preferredBounds = meshBox.clone();
    } else {
      preferredBounds = this.getPreferredLiveBounds();
    }

    if (preferredBounds && isFiniteBox(preferredBounds)) {
      this.fitCameraToBox(preferredBounds, 1.1);
      return;
    }

    this.camera.position.set(-8, -10, 7);
    this.camera.up.set(0, 0, 1);
    this.orbitControls.target.set(0, 0, 1.2);
    this.orbitControls.update();
  }

  private frameGsScene(): void {
    if (this.mode !== "gs") {
      return;
    }

    if (this.followRobot && this.gsControlMode !== "fps") {
      this.syncFollowCamera(true);
      return;
    }

    const preferredBounds = this.gaussianBounds ?? this.meshBounds ?? this.rawPointCloudBounds;
    if (preferredBounds && isFiniteBox(preferredBounds)) {
      this.fitCameraToBox(preferredBounds, 1.04);
      return;
    }

    this.camera.position.set(-8, -10, 7);
    this.camera.up.set(0, 0, 1);
    this.orbitControls.target.set(0, 0, 1.2);
    this.orbitControls.update();
  }

  private getPreferredLiveBounds(): THREE.Box3 | null {
    const bounds = [
      this.visibility.liveCloud ? this.livePointCloudBounds : null,
      this.visibility.sdfMesh ? this.meshBounds : null,
      this.visibility.occupancyCloud ? this.rawPointCloudBounds : null,
      this.visibility.gaussian ? this.gaussianBounds : null
    ].filter((box): box is THREE.Box3 => Boolean(box && isFiniteBox(box)));

    if (!bounds.length) {
      return null;
    }

    const combined = bounds[0].clone();
    for (let index = 1; index < bounds.length; index += 1) {
      combined.union(bounds[index]);
    }
    return combined;
  }

  private fitCameraToBox(box: THREE.Box3, padding = 1.15): void {
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());

    const maxDimension = Math.max(size.x, size.y, size.z, 1);
    const distance =
      (maxDimension * padding) / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov * 0.5)));

    const direction = new THREE.Vector3(-1, -1, 0.6).normalize();
    const position = center.clone().add(direction.multiplyScalar(distance * 1.4));

    this.camera.position.copy(position);
    this.camera.near = Math.max(distance / 500, 0.05);
    this.camera.far = Math.max(distance * 25, 1000);
    this.camera.updateProjectionMatrix();

    this.orbitControls.target.copy(center);
    this.orbitControls.update();
  }

  private syncFollowCamera(force = false): void {
    if (!this.followRobot) {
      return;
    }
    if (this.mode === "gs" && this.gsControlMode === "fps") {
      return;
    }

    const target = this.robotLayer.position.clone();
    target.z += 0.9;
    let desiredPosition = target.clone().add(this.followCameraOffset);
    if (this.presentationMode === "showcase" && this.mode === "playback") {
      const localOffset = new THREE.Vector3(-4.6, -1.2, 2.45);
      const rotatedOffset = localOffset.applyQuaternion(this.robotMarker.quaternion.clone());
      desiredPosition = target.clone().add(rotatedOffset);
    }

    if (force) {
      this.camera.position.copy(desiredPosition);
      this.orbitControls.target.copy(target);
      this.orbitControls.update();
      return;
    }

    this.camera.position.lerp(desiredPosition, 0.14);
    this.orbitControls.target.lerp(target, 0.18);
  }

  private applyRenderProfile(): void {
    const gaussianActive = this.visibility.gaussian && this.shouldLoadGaussian();
    const heavyLiveCloud =
      (
        (this.mode === "playback"
          ? (this.latestPlaybackMapPointCloud?.renderedPointCount ?? 0)
          : (this.latestLivePointCloud?.renderedPointCount ?? 0)
        ) > 45_000
      ) && (this.visibility.liveCloud || this.visibility.occupancyCloud);
    const maxPixelRatio = gaussianActive
      ? this.gsControlMode === "fps"
        ? 0.8
        : this.gaussianVariant === "quality"
          ? 1.0
          : this.gaussianVariant === "fast"
            ? 0.82
            : 0.9
      : this.interactionRenderActive
        ? 0.68
      : heavyLiveCloud
        ? 0.8
        : 1.0;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, maxPixelRatio));
    this.applySparkRenderTuning(gaussianActive);

    const parent = this.canvas.parentElement;
    if (!parent) {
      return;
    }
    const width = Math.max(parent.clientWidth, 1);
    const height = Math.max(parent.clientHeight, 1);
    this.renderer.setSize(width, height, false);
    this.updateLineMaterialResolution(width, height);
  }

  private applyPresentationMode(): void {
    const showcasePlayback = this.presentationMode === "showcase" && this.mode === "playback";
    const background = showcasePlayback ? "#f7f5ef" : "#111318";
    this.scene.background = new THREE.Color(background);
    this.scene.fog = new THREE.Fog(background, showcasePlayback ? 58 : 30, showcasePlayback ? 260 : 140);
    this.sceneGuidesLayer.visible = this.sceneGuidesEnabled && !showcasePlayback && this.mode !== "gs";
  }

  private installResizeObserver(): void {
    const resize = () => {
      const parent = this.canvas.parentElement;
      if (!parent) {
        return;
      }
      const width = Math.max(parent.clientWidth, 1);
      const height = Math.max(parent.clientHeight, 1);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height, false);
      this.updateLineMaterialResolution(width, height);
    };

    resize();
    this.resizeObserver = new ResizeObserver(resize);
    const parent = this.canvas.parentElement;
    if (parent) {
      this.resizeObserver.observe(parent);
    }
  }

  private getViewportSize(): THREE.Vector2 {
    const parent = this.canvas.parentElement;
    return new THREE.Vector2(
      Math.max(parent?.clientWidth ?? this.canvas.clientWidth, 1),
      Math.max(parent?.clientHeight ?? this.canvas.clientHeight, 1)
    );
  }

  private updateLineMaterialResolution(width: number, height: number): void {
    if (this.trajectoryLine) {
      this.trajectoryLine.material.resolution.set(width, height);
    }
  }

  private bumpInteractionWindow(durationMs = LiveGsViewer.INTERACTION_LOD_MS): void {
    this.cameraInteractionUntil = Math.max(this.cameraInteractionUntil, performance.now() + durationMs);
  }

  private syncInteractionRenderState(): void {
    const nextState = performance.now() < this.cameraInteractionUntil;
    if (nextState === this.interactionRenderActive) {
      return;
    }

    this.interactionRenderActive = nextState;
    this.refreshDynamicPointClouds();
    this.applyRenderProfile();
  }

  private refreshDynamicPointClouds(): void {
    if (this.mode === "playback") {
      if (this.latestPlaybackMapPointCloud && this.visibility.occupancyCloud) {
        this.renderPlaybackMapPointCloud(this.latestPlaybackMapPointCloud);
      }
      if (this.latestPlaybackScanPointCloud && this.visibility.liveCloud) {
        this.renderPlaybackScanPointCloud(this.latestPlaybackScanPointCloud);
      }
      return;
    }

    if (this.latestLivePointCloud && this.visibility.liveCloud) {
      this.renderLivePointCloud(this.latestLivePointCloud);
    }
  }

  private resolvePointSize(baseSize: number, bounds: THREE.Box3 | null): number {
    const interactionScale = this.interactionRenderActive ? 0.78 : 1;
    if (!bounds || !isFiniteBox(bounds)) {
      return baseSize * interactionScale;
    }

    const center = bounds.getCenter(new THREE.Vector3());
    const diagonal = Math.max(bounds.getSize(new THREE.Vector3()).length(), 1);
    const distance = this.camera.position.distanceTo(center);
    const distanceScale = THREE.MathUtils.clamp(0.46 + (distance / diagonal) * 0.34, 0.5, 1.18);
    return baseSize * distanceScale * interactionScale;
  }

  private disposePointCloudObject(kind: "live" | "playbackMap" | "playbackScan"): void {
    if (kind === "live" && this.livePointCloudObject) {
      this.livePointCloudLayer.remove(this.livePointCloudObject);
      this.livePointCloudObject.geometry.dispose();
      (this.livePointCloudObject.material as THREE.Material).dispose();
      this.livePointCloudObject = null;
      this.livePointCloudBounds = null;
      this.livePointCloudRevealCount = 0;
      this.livePointCloudTargetCount = 0;
      return;
    }

    if (kind === "playbackMap" && this.playbackMapPointCloudObject) {
      this.playbackMapPointCloudLayer.remove(this.playbackMapPointCloudObject);
      this.playbackMapPointCloudObject.geometry.dispose();
      (this.playbackMapPointCloudObject.material as THREE.Material).dispose();
      this.playbackMapPointCloudObject = null;
      return;
    }

    if (kind === "playbackScan") {
      if (this.playbackScanPointCloudObject) {
        this.playbackScanPointCloudLayer.remove(this.playbackScanPointCloudObject);
        this.playbackScanPointCloudObject.geometry.dispose();
        (this.playbackScanPointCloudObject.material as THREE.Material).dispose();
        this.playbackScanPointCloudObject = null;
      }
      this.playbackScanPointCloudStampMs = null;
      this.clearPlaybackScanTrail();
    }
  }

  private clearPlaybackScanTrail(): void {
    for (const entry of this.playbackScanTrailEntries) {
      this.playbackScanGhostLayer.remove(entry.object);
      entry.object.geometry.dispose();
      (entry.object.material as THREE.Material).dispose();
    }
    this.playbackScanTrailEntries = [];
  }

  private shouldLoadGaussian(): boolean {
    const gaussian = this.manifest?.gaussian;
    if (!this.visibility.gaussian || !(gaussian?.url || gaussian?.chunks?.length)) {
      return false;
    }
    if (this.mode === "gs") {
      return true;
    }
    if (this.mode === "live") {
      return true;
    }
    if (this.mode !== "playback") {
      return false;
    }
    if (this.presentationMode === "showcase") {
      return true;
    }
    return this.playbackCompareUsesWebGs;
  }

  private isPlaybackCompareWebGsActive(): boolean {
    return this.mode === "playback" && this.playbackCompareUsesWebGs;
  }

  private async ensureDeferredAssets(): Promise<void> {
    const gaussian = this.manifest?.gaussian;
    if (!gaussian || !this.shouldLoadGaussian()) {
      return;
    }
    const sourceKey = getGaussianSourceKey(gaussian);
    if (!sourceKey) {
      return;
    }

    if (this.gaussianSourceKey === sourceKey) {
      return;
    }

    if (this.gaussianLoadingKey === sourceKey && this.gaussianLoadTask) {
      if (gaussian.format === "gs-chunks") {
        return;
      }
      await this.gaussianLoadTask;
      return;
    }

    this.gaussianLoadingKey = sourceKey;
    const generation = this.gaussianLoadGeneration;
    if (gaussian.format === "gs-chunks" && gaussian.chunks?.length) {
      this.gaussianLoadTask = this.loadGaussianChunks(gaussian.chunks, sourceKey, generation).finally(() => {
        if (this.gaussianLoadingKey === sourceKey) {
          this.gaussianLoadingKey = null;
        }
        this.gaussianLoadTask = null;
      });
      return;
    }

    if (!gaussian.url) {
      return;
    }

    this.gaussianLoadTask = this.loadSingleGaussian(gaussian.url, sourceKey, generation).finally(() => {
      if (this.gaussianLoadingKey === sourceKey) {
        this.gaussianLoadingKey = null;
      }
      this.gaussianLoadTask = null;
    });
    await this.gaussianLoadTask;
  }

  private async prefetchGaussianAssets(manifest: SceneManifest): Promise<void> {
    const gaussian = manifest.gaussian;
    if (!gaussian || !this.shouldLoadGaussian() || this.gaussianPrefetchTask) {
      return;
    }

    const urls =
      gaussian.format === "gs-chunks" && gaussian.chunks?.length
        ? orderGaussianChunks(gaussian.chunks, this.camera.position, combineChunkBounds(gaussian.chunks)).map(
            (chunk) => chunk.url
          )
        : gaussian.url
          ? [gaussian.url]
          : [];

    const pending = urls.filter((url) => !this.prefetchedGaussianUrls.has(url)).slice(0, 16);
    if (!pending.length) {
      return;
    }

    this.gaussianPrefetchTask = (async () => {
      let nextIndex = 0;
      const workerCount = Math.min(4, pending.length);
      await Promise.all(
        new Array(workerCount).fill(0).map(async () => {
          while (true) {
            const currentIndex = nextIndex;
            nextIndex += 1;
            if (currentIndex >= pending.length) {
              return;
            }
            const url = pending[currentIndex];
            try {
              await fetch(url, { cache: "force-cache" });
              this.prefetchedGaussianUrls.add(url);
            } catch (error) {
              console.warn("Failed to prefetch gaussian asset.", url, error);
              return;
            }
          }
        })
      );
    })().finally(() => {
      this.gaussianPrefetchTask = null;
    });
  }

  private applySparkRenderTuning(gaussianActive: boolean): void {
    if (!this.sparkRenderer) {
      return;
    }

    if (!gaussianActive) {
      this.sparkRenderer.originDistance = 1.1;
      this.sparkRenderer.maxStdDev = Math.sqrt(5.6);
      this.sparkRenderer.maxPixelRadius = 120;
      this.sparkRenderer.focalAdjustment = 1.05;
      this.sparkRenderer.minAlpha = 0.006;
      this.sparkRenderer.clipXY = 1.1;
      return;
    }

    const interacting = this.interactionRenderActive || this.gsControlMode === "fps";
    switch (this.gaussianVariant) {
      case "ultra":
        this.sparkRenderer.originDistance = interacting ? 1.3 : 1.0;
        this.sparkRenderer.maxStdDev = interacting ? Math.sqrt(5.4) : Math.sqrt(5.95);
        this.sparkRenderer.maxPixelRadius = interacting ? 104 : 132;
        this.sparkRenderer.focalAdjustment = 1.24;
        this.sparkRenderer.minAlpha = interacting ? 0.008 : 0.0065;
        this.sparkRenderer.clipXY = 1.08;
        break;
      case "quality":
        this.sparkRenderer.originDistance = interacting ? 1.15 : 0.92;
        this.sparkRenderer.maxStdDev = interacting ? Math.sqrt(5.25) : Math.sqrt(5.7);
        this.sparkRenderer.maxPixelRadius = interacting ? 96 : 122;
        this.sparkRenderer.focalAdjustment = 1.2;
        this.sparkRenderer.minAlpha = interacting ? 0.0085 : 0.007;
        this.sparkRenderer.clipXY = 1.08;
        break;
      default:
        this.sparkRenderer.originDistance = interacting ? 1.2 : 1.0;
        this.sparkRenderer.maxStdDev = interacting ? Math.sqrt(5.1) : Math.sqrt(5.45);
        this.sparkRenderer.maxPixelRadius = interacting ? 90 : 112;
        this.sparkRenderer.focalAdjustment = 1.16;
        this.sparkRenderer.minAlpha = interacting ? 0.009 : 0.0075;
        this.sparkRenderer.clipXY = 1.08;
        break;
    }

    if (this.manifest?.source?.kind === "citygaussian") {
      this.sparkRenderer.originDistance = Math.max(this.sparkRenderer.originDistance ?? 0, interacting ? 1.22 : 1.04);
      this.sparkRenderer.maxStdDev = Math.min(this.sparkRenderer.maxStdDev ?? Math.sqrt(5), interacting ? Math.sqrt(4.55) : Math.sqrt(4.9));
      this.sparkRenderer.maxPixelRadius = Math.min(this.sparkRenderer.maxPixelRadius ?? 100, interacting ? 68 : 82);
      this.sparkRenderer.minAlpha = Math.max(this.sparkRenderer.minAlpha ?? 0.01, interacting ? 0.015 : 0.012);
      this.sparkRenderer.clipXY = Math.min(this.sparkRenderer.clipXY ?? 1.08, 1.02);
    }
  }
}

function createRobotMarker(): THREE.Object3D {
  return createFrustumMarker(new THREE.Color("#4bd0ff"), 0.8, 0.95);
}

function createTrajectoryPoseMarker(color: THREE.Color, scale = 1): THREE.Object3D {
  return createFrustumMarker(color, 0.42 * scale, 0.8);
}

function createWireBox(width: number, depth: number, height: number, color: THREE.Color, opacity: number): THREE.Object3D {
  const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(width, depth, height));
  return new THREE.LineSegments(
    edges,
    new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity,
      depthWrite: false
    })
  );
}

function createSolidBoxMarker(
  width: number,
  depth: number,
  height: number,
  fillColor: THREE.Color,
  fillOpacity: number,
  outlineColor: THREE.Color,
  outlineOpacity: number
): THREE.Object3D {
  const group = new THREE.Group();
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(width, depth, height),
    new THREE.MeshStandardMaterial({
      color: fillColor,
      transparent: true,
      opacity: fillOpacity,
      roughness: 0.42,
      metalness: 0.08
    })
  );
  const outline = createWireBox(width, depth, height, outlineColor, outlineOpacity);
  group.add(mesh);
  group.add(outline);
  return group;
}

function createFrustumMarker(color: THREE.Color, scale: number, opacity: number): THREE.Object3D {
  const group = new THREE.Group();
  const near = 0.02 * scale;
  const far = 0.32 * scale;
  const halfNearY = 0.035 * scale;
  const halfNearZ = 0.025 * scale;
  const halfFarY = 0.14 * scale;
  const halfFarZ = 0.11 * scale;

  const corners = {
    nl: new THREE.Vector3(near, -halfNearY, halfNearZ),
    nr: new THREE.Vector3(near, halfNearY, halfNearZ),
    nbl: new THREE.Vector3(near, -halfNearY, -halfNearZ),
    nbr: new THREE.Vector3(near, halfNearY, -halfNearZ),
    fl: new THREE.Vector3(far, -halfFarY, halfFarZ),
    fr: new THREE.Vector3(far, halfFarY, halfFarZ),
    fbl: new THREE.Vector3(far, -halfFarY, -halfFarZ),
    fbr: new THREE.Vector3(far, halfFarY, -halfFarZ),
  };

  const segments = [
    new THREE.Vector3(0, 0, 0), corners.fl,
    new THREE.Vector3(0, 0, 0), corners.fr,
    new THREE.Vector3(0, 0, 0), corners.fbl,
    new THREE.Vector3(0, 0, 0), corners.fbr,
    corners.fl, corners.fr,
    corners.fr, corners.fbr,
    corners.fbr, corners.fbl,
    corners.fbl, corners.fl,
    corners.nl, corners.nr,
    corners.nr, corners.nbr,
    corners.nbr, corners.nbl,
    corners.nbl, corners.nl,
  ];

  const geometry = new THREE.BufferGeometry().setFromPoints(segments);
  const lines = new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity,
      depthTest: false,
      depthWrite: false
    })
  );
  lines.renderOrder = 22;

  const body = new THREE.Mesh(
    new THREE.ConeGeometry(0.04 * scale, 0.12 * scale, 4),
    new THREE.MeshStandardMaterial({
      color,
      transparent: true,
      opacity: Math.min(opacity, 0.72),
      depthTest: false,
      depthWrite: false,
      roughness: 0.5,
      metalness: 0.08
    })
  );
  body.rotation.z = -Math.PI / 2;
  body.position.set(-0.02 * scale, 0, 0);
  body.renderOrder = 21;

  group.add(lines);
  group.add(body);
  return group;
}

function createPointSpriteTexture(): THREE.Texture {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  if (!context) {
    return new THREE.Texture();
  }

  context.clearRect(0, 0, size, size);
  const gradient = context.createRadialGradient(
    size * 0.5,
    size * 0.5,
    size * 0.06,
    size * 0.5,
    size * 0.5,
    size * 0.5
  );
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.72, "rgba(255,255,255,0.96)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  context.fillStyle = gradient;
  context.beginPath();
  context.arc(size * 0.5, size * 0.5, size * 0.5, 0, Math.PI * 2);
  context.fill();

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

function isFiniteBox(box: THREE.Box3): boolean {
  return Number.isFinite(box.min.x) &&
    Number.isFinite(box.min.y) &&
    Number.isFinite(box.min.z) &&
    Number.isFinite(box.max.x) &&
    Number.isFinite(box.max.y) &&
    Number.isFinite(box.max.z) &&
    box.max.x >= box.min.x &&
    box.max.y >= box.min.y &&
    box.max.z >= box.min.z;
}

function getGaussianSourceKey(source: SceneManifest["gaussian"] | undefined): string | null {
  if (!source) {
    return null;
  }
  if (source.format === "gs-chunks" && source.chunks?.length) {
    return `chunks:${source.chunks.map((chunk) => chunk.url).join("|")}`;
  }
  if (source.url) {
    return `${source.format}:${source.url}`;
  }
  return null;
}

function splitGaussianChunksByBudget(
  chunks: GaussianChunk[],
  variant: string
): [GaussianChunk[], GaussianChunk[]] {
  if (!chunks.length) {
    return [[], []];
  }

  const budget =
    GAUSSIAN_INITIAL_BYTE_BUDGET[
      (variant in GAUSSIAN_INITIAL_BYTE_BUDGET
        ? variant
        : "quality") as keyof typeof GAUSSIAN_INITIAL_BYTE_BUDGET
    ];

  const initial: GaussianChunk[] = [];
  const deferred: GaussianChunk[] = [];
  let usedBytes = 0;

  for (const chunk of chunks) {
    const bytes = chunk.bytes ?? 0;
    if (!initial.length || (initial.length < 4 && usedBytes + bytes <= budget)) {
      initial.push(chunk);
      usedBytes += bytes;
    } else {
      deferred.push(chunk);
    }
  }

  return [initial, deferred];
}

function boxFromChunkBounds(bounds: NonNullable<GaussianChunk["bounds"]>): THREE.Box3 {
  return new THREE.Box3(
    new THREE.Vector3(bounds.min.x, bounds.min.y, bounds.min.z),
    new THREE.Vector3(bounds.max.x, bounds.max.y, bounds.max.z)
  );
}

function compareChunks(
  left: GaussianChunk,
  right: GaussianChunk,
  cameraPosition: THREE.Vector3,
  sceneCenter: THREE.Vector3
): number {
  const leftSplats = left.splats ?? 0;
  const rightSplats = right.splats ?? 0;
  if (leftSplats !== rightSplats) {
    return rightSplats - leftSplats;
  }

  const leftCenter = left.center ?? centerFromBounds(left.bounds);
  const rightCenter = right.center ?? centerFromBounds(right.bounds);
  const leftDistance = leftCenter
    ? cameraPosition.distanceToSquared(new THREE.Vector3(leftCenter.x, leftCenter.y, leftCenter.z))
    : Number.POSITIVE_INFINITY;
  const rightDistance = rightCenter
    ? cameraPosition.distanceToSquared(new THREE.Vector3(rightCenter.x, rightCenter.y, rightCenter.z))
    : Number.POSITIVE_INFINITY;
  const leftSceneDistance = leftCenter
    ? sceneCenter.distanceToSquared(new THREE.Vector3(leftCenter.x, leftCenter.y, leftCenter.z))
    : Number.POSITIVE_INFINITY;
  const rightSceneDistance = rightCenter
    ? sceneCenter.distanceToSquared(new THREE.Vector3(rightCenter.x, rightCenter.y, rightCenter.z))
    : Number.POSITIVE_INFINITY;

  if (leftSceneDistance !== rightSceneDistance) {
    return leftSceneDistance - rightSceneDistance;
  }

  if (leftDistance !== rightDistance) {
    return leftDistance - rightDistance;
  }

  return 0;
}

function centerFromBounds(bounds: GaussianChunk["bounds"]): { x: number; y: number; z: number } | null {
  if (!bounds) {
    return null;
  }
  return {
    x: (bounds.min.x + bounds.max.x) * 0.5,
    y: (bounds.min.y + bounds.max.y) * 0.5,
    z: (bounds.min.z + bounds.max.z) * 0.5
  };
}

function combineChunkBounds(chunks: GaussianChunk[]): THREE.Box3 | null {
  const validBounds = chunks
    .map((chunk) => (chunk.bounds ? boxFromChunkBounds(chunk.bounds) : null))
    .filter((box): box is THREE.Box3 => Boolean(box && isFiniteBox(box)));

  if (!validBounds.length) {
    return null;
  }

  const combined = validBounds[0].clone();
  for (let index = 1; index < validBounds.length; index += 1) {
    combined.union(validBounds[index]);
  }
  return combined;
}

function clonePointCloudSnapshot(source: THREE.Points): THREE.Points {
  const geometry = (source.geometry as THREE.BufferGeometry).clone();
  geometry.setDrawRange(
    source.geometry.drawRange.start,
    source.geometry.drawRange.count
  );
  const material = (source.material as THREE.PointsMaterial).clone();
  const clone = new THREE.Points(geometry, material);
  clone.frustumCulled = false;
  clone.renderOrder = source.renderOrder;
  clone.position.copy(source.position);
  clone.quaternion.copy(source.quaternion);
  clone.scale.copy(source.scale);
  return clone;
}

function orderGaussianChunks(
  chunks: GaussianChunk[],
  cameraPosition: THREE.Vector3,
  combinedBounds: THREE.Box3 | null
): GaussianChunk[] {
  const sceneCenter = combinedBounds?.getCenter(new THREE.Vector3()) ?? new THREE.Vector3();
  return [...chunks].sort((left, right) => compareChunks(left, right, cameraPosition, sceneCenter));
}

function disposeObject(object: THREE.Object3D): void {
  object.traverse((child) => {
    if ("geometry" in child && child.geometry instanceof THREE.BufferGeometry) {
      child.geometry.dispose();
    }
    if ("material" in child) {
      const material = child.material;
      if (Array.isArray(material)) {
        material.forEach((item) => item.dispose());
      } else if (material instanceof THREE.Material) {
        material.dispose();
      }
    }
  });
}

function createWebGlRenderer(canvas: HTMLCanvasElement): THREE.WebGLRenderer {
  const contextAttributes: WebGLContextAttributes = {
    alpha: false,
    antialias: false,
    depth: true,
    stencil: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
    failIfMajorPerformanceCaveat: false,
    powerPreference: "high-performance"
  };

  const contextNames: Array<"webgl2" | "webgl" | "experimental-webgl"> = [
    "webgl2",
    "webgl",
    "experimental-webgl"
  ];

  let context: WebGLRenderingContext | WebGL2RenderingContext | null = null;
  for (const contextName of contextNames) {
    try {
      context = canvas.getContext(
        contextName,
        contextAttributes
      ) as WebGLRenderingContext | WebGL2RenderingContext | null;
    } catch {
      context = null;
    }
    if (context) {
      break;
    }
  }

  if (!context) {
    throw new Error(
      "WebGL is unavailable in this browser session. Enable hardware acceleration or use a browser/device with WebGL support."
    );
  }

  return new THREE.WebGLRenderer({
    canvas,
    context,
    antialias: false,
    alpha: false,
    powerPreference: "high-performance"
  });
}

function smoothTrajectoryPath(path: Pose3D[], densityMultiplier: number): Pose3D[] {
  if (path.length < 3) {
    return path;
  }

  const points = path.map(
    (pose) => new THREE.Vector3(pose.position.x, pose.position.y, pose.position.z)
  );
  const curve = new THREE.CatmullRomCurve3(points, false, "centripetal", 0.45);
  const sampleCount = Math.min(Math.max(path.length * densityMultiplier, path.length + 1, 24), 4096);
  const sampled = curve.getPoints(sampleCount - 1);
  if (sampled.length <= path.length) {
    return path;
  }

  const smoothed: Pose3D[] = [];
  for (let index = 0; index < sampled.length; index += 1) {
    const sourceFloat = (index / Math.max(sampled.length - 1, 1)) * (path.length - 1);
    const leftIndex = Math.floor(sourceFloat);
    const rightIndex = Math.min(path.length - 1, leftIndex + 1);
    const mix = sourceFloat - leftIndex;
    const leftPose = path[leftIndex];
    const rightPose = path[rightIndex];
    const point = sampled[index];
    const orientation = new THREE.Quaternion(
      leftPose.orientation.x,
      leftPose.orientation.y,
      leftPose.orientation.z,
      leftPose.orientation.w
    ).slerp(
      new THREE.Quaternion(
        rightPose.orientation.x,
        rightPose.orientation.y,
        rightPose.orientation.z,
        rightPose.orientation.w
      ),
      mix
    );

    smoothed.push({
      frameId: leftPose.frameId || rightPose.frameId,
      stampMs:
        leftPose.stampMs !== undefined && rightPose.stampMs !== undefined
          ? Math.round(leftPose.stampMs + (rightPose.stampMs - leftPose.stampMs) * mix)
          : leftPose.stampMs ?? rightPose.stampMs,
      position: {
        x: point.x,
        y: point.y,
        z: point.z
      },
      orientation: {
        x: orientation.x,
        y: orientation.y,
        z: orientation.z,
        w: orientation.w
      }
    });
  }

  return smoothed;
}

function pickTrajectoryMarkerIndices(path: Pose3D[], targetMarkers: number): number[] {
  if (path.length <= 1) {
    return path.length === 1 ? [0] : [];
  }

  const clampedTarget = Math.max(2, Math.min(targetMarkers, path.length));
  if (clampedTarget >= path.length) {
    return Array.from({ length: path.length }, (_, index) => index);
  }

  const cumulativeDistances = new Array<number>(path.length).fill(0);
  for (let index = 1; index < path.length; index += 1) {
    const prev = path[index - 1];
    const curr = path[index];
    cumulativeDistances[index] =
      cumulativeDistances[index - 1] +
      Math.hypot(
        curr.position.x - prev.position.x,
        curr.position.y - prev.position.y,
        curr.position.z - prev.position.z
      );
  }

  const totalDistance = cumulativeDistances[path.length - 1];
  if (totalDistance <= 1e-6) {
    const stride = Math.max(1, Math.floor(path.length / clampedTarget));
    const indices: number[] = [];
    for (let index = 0; index < path.length; index += stride) {
      indices.push(index);
    }
    if (indices[indices.length - 1] !== path.length - 1) {
      indices.push(path.length - 1);
    }
    return indices;
  }

  const indices: number[] = [0];
  let nextTargetDistance = totalDistance / (clampedTarget - 1);
  for (let index = 1; index < path.length - 1; index += 1) {
    if (cumulativeDistances[index] + 1e-6 < nextTargetDistance) {
      continue;
    }
    indices.push(index);
    nextTargetDistance += totalDistance / (clampedTarget - 1);
  }

  if (indices[indices.length - 1] !== path.length - 1) {
    indices.push(path.length - 1);
  }
  return indices;
}

function computePointCloudBounds(positions: Float32Array): THREE.Box3 | null {
  if (!positions.length) {
    return null;
  }

  const box = new THREE.Box3();
  const point = new THREE.Vector3();
  let hasPoint = false;

  for (let index = 0; index < positions.length; index += 3) {
    point.set(positions[index], positions[index + 1], positions[index + 2]);
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.z)) {
      continue;
    }
    if (!hasPoint) {
      box.min.copy(point);
      box.max.copy(point);
      hasPoint = true;
    } else {
      box.expandByPoint(point);
    }
  }

  return hasPoint ? box : null;
}

function samplePointCloudForRender(cloud: DecodedPointCloud, maxPoints: number): DecodedPointCloud {
  if (cloud.renderedPointCount <= maxPoints) {
    return cloud;
  }

  const stride = Math.max(1, Math.ceil(cloud.renderedPointCount / maxPoints));
  const sampleCount = Math.ceil(cloud.renderedPointCount / stride);
  const positions = new Float32Array(sampleCount * 3);
  const colors =
    cloud.colors && cloud.colors.length === cloud.positions.length
      ? new Float32Array(sampleCount * 3)
      : undefined;

  let renderIndex = 0;
  for (let pointIndex = 0; pointIndex < cloud.renderedPointCount; pointIndex += stride) {
    const sourceOffset = pointIndex * 3;
    const targetOffset = renderIndex * 3;
    positions[targetOffset] = cloud.positions[sourceOffset];
    positions[targetOffset + 1] = cloud.positions[sourceOffset + 1];
    positions[targetOffset + 2] = cloud.positions[sourceOffset + 2];

    if (colors && cloud.colors) {
      colors[targetOffset] = cloud.colors[sourceOffset];
      colors[targetOffset + 1] = cloud.colors[sourceOffset + 1];
      colors[targetOffset + 2] = cloud.colors[sourceOffset + 2];
    }
    renderIndex += 1;
  }

  return {
    frameId: cloud.frameId,
    stampMs: cloud.stampMs,
    sourcePointCount: cloud.sourcePointCount,
    renderedPointCount: renderIndex,
    positions: renderIndex === sampleCount ? positions : positions.slice(0, renderIndex * 3),
    colors:
      colors && renderIndex === sampleCount
        ? colors
        : colors
          ? colors.slice(0, renderIndex * 3)
          : undefined
  };
}

function samplePointCloudPrioritizingColor(
  cloud: DecodedPointCloud,
  maxPoints: number
): DecodedPointCloud {
  if (
    cloud.renderedPointCount <= maxPoints ||
    !cloud.colors ||
    cloud.colors.length !== cloud.positions.length
  ) {
    return samplePointCloudForRender(cloud, maxPoints);
  }

  const vividIndices: number[] = [];
  const neutralIndices: number[] = [];
  for (let pointIndex = 0; pointIndex < cloud.renderedPointCount; pointIndex += 1) {
    const offset = pointIndex * 3;
    const r = cloud.colors[offset];
    const g = cloud.colors[offset + 1];
    const b = cloud.colors[offset + 2];
    const span = Math.max(r, g, b) - Math.min(r, g, b);
    const neutralDistance =
      Math.abs(r - 160 / 255) + Math.abs(g - 165 / 255) + Math.abs(b - 174 / 255);
    if (span >= 0.12 || neutralDistance >= 0.18) {
      vividIndices.push(pointIndex);
    } else {
      neutralIndices.push(pointIndex);
    }
  }

  const selected = new Array<number>();
  if (vividIndices.length >= maxPoints) {
    for (let index = 0; index < maxPoints; index += 1) {
      selected.push(vividIndices[Math.floor((index * vividIndices.length) / maxPoints)]);
    }
  } else {
    selected.push(...vividIndices);
    const remaining = maxPoints - selected.length;
    if (remaining > 0) {
      if (neutralIndices.length <= remaining) {
        selected.push(...neutralIndices);
      } else {
        for (let index = 0; index < remaining; index += 1) {
          selected.push(neutralIndices[Math.floor((index * neutralIndices.length) / remaining)]);
        }
      }
    }
  }

  selected.sort((left, right) => left - right);
  const positions = new Float32Array(selected.length * 3);
  const colors = new Float32Array(selected.length * 3);
  for (let renderIndex = 0; renderIndex < selected.length; renderIndex += 1) {
    const sourceOffset = selected[renderIndex] * 3;
    const targetOffset = renderIndex * 3;
    positions[targetOffset] = cloud.positions[sourceOffset];
    positions[targetOffset + 1] = cloud.positions[sourceOffset + 1];
    positions[targetOffset + 2] = cloud.positions[sourceOffset + 2];
    colors[targetOffset] = cloud.colors[sourceOffset];
    colors[targetOffset + 1] = cloud.colors[sourceOffset + 1];
    colors[targetOffset + 2] = cloud.colors[sourceOffset + 2];
  }

  return {
    frameId: cloud.frameId,
    stampMs: cloud.stampMs,
    sourcePointCount: cloud.sourcePointCount,
    renderedPointCount: selected.length,
    positions,
    colors
  };
}

function ensureGeometryVertexColors(geometry: THREE.BufferGeometry): boolean {
  const existingColors = geometry.getAttribute("color");
  if (existingColors && existingColors.count > 0) {
    return true;
  }

  const position = geometry.getAttribute("position");
  if (!(position instanceof THREE.BufferAttribute) || position.count === 0) {
    return false;
  }

  const bounds = geometry.boundingBox ?? new THREE.Box3().setFromBufferAttribute(position);
  const zMin = bounds.min.z;
  const zMax = bounds.max.z;
  const zRange = Math.max(1e-5, zMax - zMin);
  const colors = new Float32Array(position.count * 3);
  const color = new THREE.Color();

  for (let index = 0; index < position.count; index += 1) {
    const z = position.getZ(index);
    const t = THREE.MathUtils.clamp((z - zMin) / zRange, 0, 1);
    color.setHSL(0.62 - t * 0.54, 0.78, 0.54);
    const offset = index * 3;
    colors[offset] = color.r;
    colors[offset + 1] = color.g;
    colors[offset + 2] = color.b;
  }

  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  return true;
}
