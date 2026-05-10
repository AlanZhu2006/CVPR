import ROSLIB from "roslib";
import type { Pose3D } from "../gs/gsSdfSceneAdapter";

export type RosClient = any;
export type RosTopic = any;

export interface ScreenPoint {
  x: number;
  y: number;
}

export interface ViewportTransform {
  scale: number;
  translateX: number;
  translateY: number;
}

export interface OccupancyMapMeta {
  width: number;
  height: number;
  resolution: number;
  origin: {
    x: number;
    y: number;
  };
  frameId: string;
}

export interface GoalPose2D {
  x: number;
  y: number;
  yaw: number;
}

export interface InitialPose2D extends GoalPose2D {
  covariance?: number[];
}

export interface PoseSubscription {
  dispose(): void;
}

export function createMoveBaseGoalTopic(
  ros: RosClient,
  topicName = "/move_base_simple/goal"
): RosTopic {
  return new ROSLIB.Topic({
    ros,
    name: topicName,
    messageType: "geometry_msgs/PoseStamped"
  });
}

export function createCurrentCameraPoseTopic(
  ros: RosClient,
  topicName = "/rviz/current_camera_pose"
): RosTopic {
  return new ROSLIB.Topic({
    ros,
    name: topicName,
    messageType: "geometry_msgs/Pose"
  });
}

export function createInitialPoseTopic(
  ros: RosClient,
  topicName = "/initialpose"
): RosTopic {
  return new ROSLIB.Topic({
    ros,
    name: topicName,
    messageType: "geometry_msgs/PoseWithCovarianceStamped"
  });
}

export function subscribeOdometry(
  ros: RosClient,
  onPose: (pose: GoalPose2D) => void,
  topicName = "/odom"
): PoseSubscription {
  const topic = new ROSLIB.Topic({
    ros,
    name: topicName,
    messageType: "nav_msgs/Odometry"
  });

  const handler = (message: unknown) => {
    const odom = message as unknown as {
      pose: {
        pose: {
          position: { x: number; y: number };
          orientation: { x: number; y: number; z: number; w: number };
        };
      };
    };

    onPose({
      x: odom.pose.pose.position.x,
      y: odom.pose.pose.position.y,
      yaw: quaternionToYaw(odom.pose.pose.orientation)
    });
  };

  topic.subscribe(handler);

  return {
    dispose() {
      topic.unsubscribe(handler);
    }
  };
}

export function screenToWorldOn2DMap(
  screen: ScreenPoint,
  canvasRect: DOMRect,
  viewport: ViewportTransform,
  map: OccupancyMapMeta
): { x: number; y: number } {
  const localX = screen.x - canvasRect.left;
  const localY = screen.y - canvasRect.top;

  const mapPixelX = (localX - viewport.translateX) / viewport.scale;
  const mapPixelY = (localY - viewport.translateY) / viewport.scale;

  const worldX = map.origin.x + mapPixelX * map.resolution;
  const worldY = map.origin.y + (map.height - mapPixelY) * map.resolution;

  return { x: worldX, y: worldY };
}

export function publishMoveBaseSimpleGoal(
  topic: RosTopic,
  goal: GoalPose2D,
  frameId = "map"
): void {
  const q = yawToQuaternion(goal.yaw);

  topic.publish(
    new (ROSLIB as any).Message({
      header: {
        frame_id: frameId
      },
      pose: {
        position: {
          x: goal.x,
          y: goal.y,
          z: 0
        },
        orientation: q
      }
    })
  );
}

export function publishInitialPose(
  topic: RosTopic,
  pose: InitialPose2D,
  frameId = "map"
): void {
  const q = yawToQuaternion(pose.yaw);
  const covariance = pose.covariance ?? [
    0.25, 0, 0, 0, 0, 0,
    0, 0.25, 0, 0, 0, 0,
    0, 0, 0.0, 0, 0, 0,
    0, 0, 0, 0.0, 0, 0,
    0, 0, 0, 0, 0.0, 0,
    0, 0, 0, 0, 0, 0.0685
  ];

  topic.publish(
    new (ROSLIB as any).Message({
      header: {
        frame_id: frameId
      },
      pose: {
        pose: {
          position: {
            x: pose.x,
            y: pose.y,
            z: 0
          },
          orientation: q
        },
        covariance
      }
    })
  );
}

export function publishGoalFromClick(input: {
  ros: RosClient;
  screen: ScreenPoint;
  canvasRect: DOMRect;
  viewport: ViewportTransform;
  map: OccupancyMapMeta;
  yaw?: number;
  topicName?: string;
}): void {
  const topic = createMoveBaseGoalTopic(input.ros, input.topicName);
  const world = screenToWorldOn2DMap(
    input.screen,
    input.canvasRect,
    input.viewport,
    input.map
  );

  publishMoveBaseSimpleGoal(
    topic,
    {
      x: world.x,
      y: world.y,
      yaw: input.yaw ?? 0
    },
    input.map.frameId
  );
}

export function publishCurrentCameraPose(topic: RosTopic, pose: Pose3D): void {
  topic.publish(
    new (ROSLIB as any).Message({
      position: {
        x: pose.position.x,
        y: pose.position.y,
        z: pose.position.z
      },
      orientation: {
        x: pose.orientation.x,
        y: pose.orientation.y,
        z: pose.orientation.z,
        w: pose.orientation.w
      }
    })
  );
}

export function yawToQuaternion(yaw: number): {
  x: number;
  y: number;
  z: number;
  w: number;
} {
  const half = yaw * 0.5;
  return {
    x: 0,
    y: 0,
    z: Math.sin(half),
    w: Math.cos(half)
  };
}

function quaternionToYaw(quaternion: {
  x: number;
  y: number;
  z: number;
  w: number;
}): number {
  const sinyCosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y);
  const cosyCosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z);
  return Math.atan2(sinyCosp, cosyCosp);
}
