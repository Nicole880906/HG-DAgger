from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from rclpy.node import Node
from typing import Tuple, Optional
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from transforms import *  # noqa: F403
except ModuleNotFoundError:
    def pose_to_matrix(pos, quat_wxyz):
        pos = np.asarray(pos, dtype=np.float32).reshape(3)
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
            dtype=np.float32,
        )
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)
        T[:3, 3] = pos
        return T


# The vendored original hard-coded /docker-ros/ws, the bind-mount path used
# inside the robodiff_ros container. That path does not exist when the deploy
# node runs on the dVRK desktop's native ROS install, so the calibration
# silently fell back to identity and every camera-frame conversion was wrong.
# Resolve against this checkout first and keep the container path as a fallback,
# so the same file works in both places.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ASSET_DIRS = (
    _REPO_ROOT / "src" / "arclab_dvrk" / "assets",
    Path("/docker-ros/ws/src/arclab_dvrk/assets"),
)


def asset_path(name: str) -> Path:
    """First existing location of a calibration asset; last candidate if none."""
    for directory in _ASSET_DIRS:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return _ASSET_DIRS[-1] / name



@staticmethod
def quaternion_multiply(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions in XYZW format.
    
    Args:
        q1_xyzw: First quaternion [x, y, z, w]
        q2_xyzw: Second quaternion [x, y, z, w]
    
    Returns:
        Result quaternion [x, y, z, w]
    """
    x1, y1, z1, w1 = q1_xyzw
    x2, y2, z2, w2 = q2_xyzw
    
    # Hamilton product
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([x, y, z, w], dtype=np.float32)

class PSM1Controller:
    def __init__(self, node: Node, ns: str = "PSM1", frame_id: str = "PSM1_base"):
        self.node = node
        self.frame_id = frame_id
        self.pub = node.create_publisher(PoseStamped, f"/{ns}/move_cp", 10)
        self._last_q = (0.0, 0.0, 0.0, 1.0)
        self._last_pos = (0.0, 0.0, 0.0)
        self._last_jaw: Optional[float] = None
        self.offset = (0.0, 0.0, 0.0)  # offset in PSM1 frame
        self.jaw_min = -1.0
        self.jaw_max =  1.5
        self.jaw_margin = 0.02
        self.jaw_vmax = 2.0
        self._last_jaw_time = None
        self._jaw_goal = None
        self._jaw_snap_done = False
        self._jaw_force_goal_next = False
        self._prev_jaw_goal = None

        # self.servo_cp_pub = node.create_publisher(PoseStamped, f"/{ns}/servo_cp", 10)
        self.servo_pub = node.create_publisher(PoseStamped, f"/{ns}/servo_cp", 10)

        self._js_names = None
        self._js_pos   = None


        node.create_subscription(PoseStamped, f"/{ns}/measured_cp", self._on_measured, 10)


        self.jaw_pub = node.create_publisher(JointState, f"/{ns}/jaw/servo_jp", 30)
        node.create_subscription(JointState, f"/{ns}/jaw/measured_js", self._on_jaw_measured, 10)

        try:
            data = np.load(asset_path("cam_pose_psm1.npz"))
            pos = data["pos"].astype(np.float32)
            quat_wxyz = data["quat"].astype(np.float32)
            # quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)  # convert to w,x,y,z
            self.psm1_T_cam = pose_to_matrix(pos, quat_wxyz)
            self.cam_T_psm1 = np.linalg.inv(self.psm1_T_cam)  # PSM1 to camera

        except Exception as e:
            self.node.get_logger().warn(f"cam_pose_psm1.npz missing ({e}); using identity.")
            self.psm1_T_cam = np.eye(4, dtype=np.float32)
            # The original left cam_T_psm1 unset here, so psm1_pos_to_cam()
            # raised AttributeError on the fallback path instead of degrading
            # to identity the way the PSM2 branch does.
            self.cam_T_psm1 = np.eye(4, dtype=np.float32)



    def psm1_pos_to_cam(self, pos_psm1_xyz) -> np.ndarray:
        """Convert a position from PSM1 base frame to CAMERA frame (no rotation)."""
        p = np.asarray(pos_psm1_xyz, dtype=np.float32).reshape(3)
        ph = np.hstack([p, 1.0]).astype(np.float32)
        return (self.cam_T_psm1 @ ph)[:3].astype(np.float32)

    def _on_measured(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        self._last_pos = (p.x, p.y, p.z)
        self._last_q = (q.x, q.y, q.z, q.w)

    def current_orientation(self) -> Tuple[float,float,float,float]:
        return self._last_q
    
    def current_pos(self) -> Tuple[float,float,float]:
        return self._last_pos
    
    def cam_pos_psm1(self, pos) -> Tuple[float,float,float]:

        pos_psm1 = np.matmul(self.cam_T_psm1, np.hstack((pos, 1.0)))[:3]  # apply T_cam_psm1
        return pos_psm1.astype(np.float32)
    
    @staticmethod
    def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / (n + eps)
    
    def heading_xy_in_psm1(self, target_cam_xyz: np.ndarray) -> np.ndarray:
        """
        ee->target, return direction
        """
        tgt_psm1 = self.cam_pos_psm1(np.asarray(target_cam_xyz, dtype=np.float32))
        cur_psm1 = np.asarray(self.current_pos(), dtype=np.float32)
        cur_psm1 += np.asarray(self.offset, dtype=np.float32)  # apply offset in PSM1 frame
        d = tgt_psm1 - cur_psm1                   # EE -> target (base frame)
        d_xy = np.array([d[0], d[1], 0.0], np.float32)
        if np.linalg.norm(d_xy) < 1e-6:
            # Degenerate: target is vertically above/below EE; choose default heading +X
            return np.array([1.0, 0.0, 0.0], np.float32)
        return self._normalize(d_xy)
    
    def quat_x_up_z_heading(self, target_cam_xyz: np.ndarray) -> np.ndarray:
        """
        x up, z to direction
        """
        x_axis = np.array([0.0, 0.0, 1.0], np.float32)  # "sky" in base frame
        z_axis = self.heading_xy_in_psm1(target_cam_xyz)
        # Ensure orthonormal, right-handed basis: columns [X Y Z]
        y_axis = np.cross(z_axis, x_axis)
        y_n = np.linalg.norm(y_axis)
        if y_n < 1e-6:
            # Shouldn’t happen because z has zero Z and x is [0,0,1], but guard anyway
            y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        else:
            y_axis = (y_axis / y_n).astype(np.float32)

        R_world_tool = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
        q_xyzw = R.from_matrix(R_world_tool).as_quat().astype(np.float32)  # [x,y,z,w]
        q_xyzw /= np.linalg.norm(q_xyzw) + 1e-9
        return q_xyzw
    
    def send_pose_with_heading(self, target_cam_xyz: np.ndarray):
        quat = tuple(self.quat_x_up_z_heading(target_cam_xyz))
        self.send_pose(pos_xyz=target_cam_xyz, quat=quat)
    
    def send_pose(self, pos_xyz, quat=None):
        pos_psm1 = self.cam_pos_psm1(np.asarray(pos_xyz, dtype=np.float32))
        pos_psm1 += np.asarray(self.offset, dtype=np.float32)  # apply offset in PSM1 frame
        if quat is None: quat = self.current_orientation()
        ps = PoseStamped()
        ps.header.stamp = self.node.get_clock().now().to_msg()
        ps.header.frame_id = self.frame_id
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, pos_psm1)
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, quat)
        self.pub.publish(ps)
        self.node.get_logger().info(f"move_cp → pos={tuple(round(x,4) for x in pos_psm1)}")


    def send_servo_pose(self, pos_cam_xyz, quat=None):
        ps = PoseStamped()
        ps.header.stamp = self.node.get_clock().now().to_msg()
        ps.header.frame_id = self.frame_id  # e.g., "PSM1_base" (driver interprets frames)
        # pos in CAMERA frame; your driver maps internally like your move_cp
        x, y, z = map(float, pos_cam_xyz)
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, z
        if quat is None:
            quat = self.current_orientation()
        qx, qy, qz, qw = quat
        ps.pose.orientation.x = float(qx)
        ps.pose.orientation.y = float(qy)
        ps.pose.orientation.z = float(qz)
        ps.pose.orientation.w = float(qw)
        self.servo_pub.publish(ps)

    def send_servo_rel_pose(self, dpos, dquat=None):
        ps = PoseStamped()
        ps.header.stamp = self.node.get_clock().now().to_msg()
        ps.header.frame_id = self.frame_id  # e.g., "PSM1_base" (driver interprets frames)
        # pos in CAMERA frame; your driver maps internally like your move_cp
        cur_pos = np.asarray(self.current_pos(), dtype=np.float32)
        target_pos = cur_pos + np.asarray(dpos, dtype=np.float32)
        print("target pos:", target_pos)
        cur_quat = self.current_orientation()
        if dquat is None:
            target_quat = cur_quat
        else:
            target_quat = quaternion_multiply(cur_quat, dquat)
            target_quat /= np.linalg.norm(target_quat) + 1e-9

        # x, y, z = map(float, pos_cam_xyz)
        # ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, z
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, target_pos)
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, target_quat)
        # if quat is None:
        #     quat = self.current_orientation()
        # qx, qy, qz, qw = quat
        # ps.pose.orientation.x = float(qx)
        # ps.pose.orientation.y = float(qy)
        # ps.pose.orientation.z = float(qz)
        # ps.pose.orientation.w = float(qw)
        self.servo_pub.publish(ps)

    def send_pose_psm1(self, pos_psm1_xyz: np.ndarray, quat=None):
        """
        Send a pose in PSM1 base frame. If quat is None, use current orientation.
        """
        pos_psm1 = np.asarray(pos_psm1_xyz, dtype=np.float32).reshape(3)
        # pos_psm1 += np.asarray(self.offset, dtype=np.float32)
        if quat is None:
            quat = self.current_orientation()
        ps = PoseStamped()
        ps.header.stamp = self.node.get_clock().now().to_msg()
        ps.header.frame_id = self.frame_id
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, pos_psm1)
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, quat)
        self.pub.publish(ps)
        self.node.get_logger().info(f"move_cp → pos={tuple(round(x,4) for x in pos_psm1)}")

    def _on_jaw_measured(self, msg: JointState):
        """
        Store last jaw angle in radians. Robustly pick the jaw index.
        """
        idx = 0
        if msg.name:
            try:
                idx = next(i for i, n in enumerate(msg.name) if "jaw" in n.lower())
            except StopIteration:
                idx = 0
        if len(msg.position) > idx:
            self._last_jaw = float(msg.position[idx])

    def current_jaw(self) -> Optional[float]:
        """Last measured jaw (radians), or None if unseen."""
        return self._last_jaw

    def set_jaw_servo(self, position_rad: float, name: str = "PSM1_jaw"):
        """
        Command jaw via /PSM1/jaw/servo_jp. position_rad is **radians**.
        """
        js = JointState()
        js.name = [name]  # name is optional; many drivers ignore it, but it's nice to include
        js.position = [float(position_rad)]
        self.jaw_pub.publish(js)

    # # Convenience helpers
    # def open_jaw(self, degrees: float = 30.0):
    #     self.set_jaw_servo(np.deg2rad(degrees))

    # def close_jaw(self, rad=0.0):
    #     self.set_jaw_servo(rad)


    def set_jaw_goal(self, goal_rad: float):
        """Set an absolute jaw goal (rad). Will be held by calling hold_jaw_goal() each tick."""
        lo = self.jaw_min + self.jaw_margin
        hi = self.jaw_max - self.jaw_margin
        self._jaw_goal = float(np.clip(goal_rad, lo, hi))

    def hold_jaw_goal(self, dt: float):
        """
        Advance toward the absolute goal with rate limiting and clamping.
        Special case:
        if goal < -0.5 rad and current < 0.1 rad:
            - send -0.5 (clamped) once,
            - next tick, send the goal directly,
            - then resume normal rate limiting.
        """
        # lazy init (in case you didn't set these in __init__)
        if not hasattr(self, "_jaw_snap_done"):
            self._jaw_snap_done = False
            self._jaw_force_goal_next = False
            self._prev_jaw_goal = None

        if self._jaw_goal is None:
            return

        # soft limits
        lo = self.jaw_min + self.jaw_margin
        hi = self.jaw_max - self.jaw_margin

        # reset snap logic when the goal changes
        if self._prev_jaw_goal is None or float(self._prev_jaw_goal) != float(self._jaw_goal):
            self._jaw_snap_done = False
            self._jaw_force_goal_next = False
            self._prev_jaw_goal = float(self._jaw_goal)

        q = self.current_jaw()

        # If we don't have feedback yet, just stream the goal (clamped)
        if q is None:
            self.set_jaw_servo(float(np.clip(self._jaw_goal, lo, hi)))
            return

        # ---- special "snap then goal" override ----
        if (self._jaw_goal < -0.5) and (q < 0.1):
            # step 1: snap to -0.5 once
            if not self._jaw_snap_done:
                snap_cmd = float(np.clip(-0.5, lo, hi))
                self.set_jaw_servo(snap_cmd)
                self._jaw_snap_done = True
                self._jaw_force_goal_next = True
                return
            # step 2: on very next tick, send the goal directly (no rate limit)
            if self._jaw_force_goal_next:
                goal_cmd = float(np.clip(self._jaw_goal, lo, hi))
                self.set_jaw_servo(goal_cmd)
                self._jaw_force_goal_next = False
                return
        # -------------------------------------------

        # normal rate-limited tracking toward goal.
        # Use real elapsed wall-time, not the nominal control period: the loop
        # actually runs much slower than `rate` Hz (inference + move_pause), so
        # the nominal dt throttles the jaw far below jaw_vmax. Clamp to avoid a
        # huge first step after a long gap / snap override.
        now = time.monotonic()
        real_dt = dt if self._last_jaw_time is None else now - self._last_jaw_time
        self._last_jaw_time = now
        dq_max = self.jaw_vmax * float(np.clip(real_dt, 1e-3, 0.5))
        q_next = q + np.clip(self._jaw_goal - q, -dq_max, dq_max)
        q_next = float(np.clip(q_next, lo, hi))
        self.set_jaw_servo(q_next)


    def open_jaw(self, rad: float = 1.0):
        # Map desired opening to an absolute goal in radians.
        # Adjust mapping/sign for your instrument; here we assume "more negative = more closed".
        # Example: fully open ~ +0.5 rad, fully closed ~ 0.0 rad. TUNE THESE:
        jaw_open_rad = rad
        self.set_jaw_goal(jaw_open_rad)

    def close_jaw(self):
        jaw_close_rad = -1.0  # TUNE: “tight” close that’s still inside limits (with margin)
        self.set_jaw_goal(jaw_close_rad)


    def move_delta(self, delta_xyz: Tuple[float, float, float], quat=None):

        pos = np.asarray(self.current_pos(), dtype=np.float32)
        pos += np.asarray(delta_xyz, dtype=np.float32)
        pos -= np.asarray(self.offset, dtype=np.float32)
        self.send_pose(pos_xyz=pos, quat=quat)


class PSM2Controller:
    def __init__(self, node: Node, ns: str = "PSM2", frame_id: str = "PSM2_base"):
        self.node = node
        self.frame_id = frame_id
        self.pub = node.create_publisher(PoseStamped, f"/{ns}/move_cp", 10)
        self._last_q = (0.0, 0.0, 0.0, 1.0)
        self._last_pos = (0.0, 0.0, 0.0)
        self._last_jaw: Optional[float] = None
        self.offset = (0.0, 0.0, 0.0)  # offset in PSM2 frame
        self.jaw_min = -1.0
        self.jaw_max =  1.5
        self.jaw_margin = 0.02
        self.jaw_vmax = 2.0
        self._last_jaw_time = None
        self._jaw_goal = None

        self.servo_pub = node.create_publisher(PoseStamped, f"/{ns}/servo_cp", 10)

        self._js_names = None
        self._js_pos   = None

        node.create_subscription(PoseStamped, f"/{ns}/measured_cp", self._on_measured, 10)

        self.jaw_pub = node.create_publisher(JointState, f"/{ns}/jaw/servo_jp", 10)
        node.create_subscription(JointState, f"/{ns}/jaw/measured_js", self._on_jaw_measured, 10)

        # Load cam->psm2 transform
        try:
            data = np.load(asset_path("cam_pose_psm2.npz"))
            pos = data["pos"].astype(np.float32)
            quat_wxyz = data["quat"].astype(np.float32)
            self.psm2_T_cam = pose_to_matrix(pos, quat_wxyz)
            self.cam_T_psm2 = np.linalg.inv(self.psm2_T_cam)  # PSM2 to camera

        except Exception as e:
            self.node.get_logger().warn(f"cam_pose_psm2.npz missing ({e}); using identity.")
            self.psm2_T_cam = np.eye(4, dtype=np.float32)
            self.cam_T_psm2 = np.eye(4, dtype=np.float32)

    def psm2_pos_to_cam(self, pos_psm2_xyz) -> np.ndarray:
        """Convert a position from PSM2 base frame to CAMERA frame (no rotation)."""
        p = np.asarray(pos_psm2_xyz, dtype=np.float32).reshape(3)
        ph = np.hstack([p, 1.0]).astype(np.float32)
        return (self.cam_T_psm2 @ ph)[:3].astype(np.float32)

    def _on_measured(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        self._last_pos = (p.x, p.y, p.z)
        self._last_q = (q.x, q.y, q.z, q.w)

    def current_orientation(self) -> Tuple[float,float,float,float]:
        return self._last_q
    
    def current_pos(self) -> Tuple[float,float,float]:
        return self._last_pos
    
    def cam_pos_psm2(self, pos) -> Tuple[float,float,float]:
        pos_psm2 = np.matmul(self.cam_T_psm2, np.hstack((pos, 1.0)))[:3]  # apply T_cam_psm2
        return pos_psm2.astype(np.float32)
    
    @staticmethod
    def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / (n + eps)
    
    def heading_xy_in_psm2(self, target_cam_xyz: np.ndarray) -> np.ndarray:
        """
        ee->target, return direction
        """
        tgt_psm2 = self.cam_pos_psm2(np.asarray(target_cam_xyz, dtype=np.float32))
        cur_psm2 = np.asarray(self.current_pos(), dtype=np.float32)
        cur_psm2 += np.asarray(self.offset, dtype=np.float32)  # apply offset in PSM2 frame
        d = tgt_psm2 - cur_psm2                   # EE -> target (base frame)
        d_xy = np.array([d[0], d[1], 0.0], np.float32)
        if np.linalg.norm(d_xy) < 1e-6:
            # Degenerate: target is vertically above/below EE; choose default heading +X
            return np.array([1.0, 0.0, 0.0], np.float32)
        return self._normalize(d_xy)
    
    def quat_x_up_z_heading(self, target_cam_xyz: np.ndarray) -> np.ndarray:
        """
        x up, z to direction
        """
        x_axis = np.array([0.0, 0.0, 1.0], np.float32)  # "sky" in base frame
        z_axis = self.heading_xy_in_psm2(target_cam_xyz)
        # Ensure orthonormal, right-handed basis: columns [X Y Z]
        y_axis = np.cross(z_axis, x_axis)
        y_n = np.linalg.norm(y_axis)
        if y_n < 1e-6:
            # Shouldn't happen because z has zero Z and x is [0,0,1], but guard anyway
            y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        else:
            y_axis = (y_axis / y_n).astype(np.float32)

        R_world_tool = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
        q_xyzw = R.from_matrix(R_world_tool).as_quat().astype(np.float32)  # [x,y,z,w]
        q_xyzw /= np.linalg.norm(q_xyzw) + 1e-9
        return q_xyzw
    
    def send_pose_with_heading(self, target_cam_xyz: np.ndarray):
        quat = tuple(self.quat_x_up_z_heading(target_cam_xyz))
        self.send_pose(pos_xyz=target_cam_xyz, quat=quat)
    
    def send_pose(self, pos_xyz, quat=None):
        pos_psm2 = self.cam_pos_psm2(np.asarray(pos_xyz, dtype=np.float32))
        pos_psm2 += np.asarray(self.offset, dtype=np.float32)  # apply offset in PSM2 frame
        if quat is None: quat = self.current_orientation()
        ps = PoseStamped()
        ps.header.stamp = self.node.get_clock().now().to_msg()
        ps.header.frame_id = self.frame_id
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, pos_psm2)
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, quat)
        self.pub.publish(ps)
        self.node.get_logger().info(f"move_cp → pos={tuple(round(x,4) for x in pos_psm2)}")

    def send_pose_psm2(self, pos_psm2_xyz: np.ndarray, quat=None):
        """
        Send a pose in PSM2 base frame. If quat is None, use current orientation.
        """
        pos_psm2 = np.asarray(pos_psm2_xyz, dtype=np.float32).reshape(3)
        if quat is None:
            quat = self.current_orientation()
        ps = PoseStamped()
        ps.header.stamp = self.node.get_clock().now().to_msg()
        ps.header.frame_id = self.frame_id
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, pos_psm2)
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, quat)
        self.pub.publish(ps)
        self.node.get_logger().info(f"move_cp → pos={tuple(round(x,4) for x in pos_psm2)}")


    def send_servo_pose(self, pos_cam_xyz, quat=None):
        ps = PoseStamped()
        ps.header.stamp = self.node.get_clock().now().to_msg()
        ps.header.frame_id = self.frame_id  # e.g., "PSM1_base" (driver interprets frames)
        # pos in CAMERA frame; your driver maps internally like your move_cp
        x, y, z = map(float, pos_cam_xyz)
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, z
        if quat is None:
            quat = self.current_orientation()
        qx, qy, qz, qw = quat
        ps.pose.orientation.x = float(qx)
        ps.pose.orientation.y = float(qy)
        ps.pose.orientation.z = float(qz)
        ps.pose.orientation.w = float(qw)
        self.servo_pub.publish(ps)
        
    def _on_jaw_measured(self, msg: JointState):
        """
        Store last jaw angle in radians. Robustly pick the jaw index.
        """
        idx = 0
        if msg.name:
            try:
                idx = next(i for i, n in enumerate(msg.name) if "jaw" in n.lower())
            except StopIteration:
                idx = 0
        if len(msg.position) > idx:
            self._last_jaw = float(msg.position[idx])

    def current_jaw(self) -> Optional[float]:
        """Last measured jaw (radians), or None if unseen."""
        return self._last_jaw

    def set_jaw_servo(self, position_rad: float, name: str = "PSM2_jaw"):
        """
        Command jaw via /PSM2/jaw/servo_jp. position_rad is **radians**.
        """
        js = JointState()
        js.name = [name]  # name is optional; many drivers ignore it, but it's nice to include
        js.position = [float(position_rad)]
        self.jaw_pub.publish(js)

    def set_jaw_goal(self, goal_rad: float):
        """Set an absolute jaw goal (rad). Will be held by calling hold_jaw_goal() each tick."""
        lo = self.jaw_min + self.jaw_margin
        hi = self.jaw_max - self.jaw_margin
        self._jaw_goal = float(np.clip(goal_rad, lo, hi))

    def hold_jaw_goal(self, dt: float):
        """
        Advance toward the absolute goal with rate limiting and clamping.
        Call this every _tick(dt).
        """
        if self._jaw_goal is None:
            return
        q = self.current_jaw()
        if q is None:
            # no feedback yet: send the goal directly
            self.set_jaw_servo(self._jaw_goal)
            return
        # rate limit using real elapsed wall-time, not the nominal control
        # period: the loop runs much slower than `rate` Hz (inference +
        # move_pause), so the nominal dt throttles the jaw far below jaw_vmax.
        now = time.monotonic()
        real_dt = dt if self._last_jaw_time is None else now - self._last_jaw_time
        self._last_jaw_time = now
        dq_max = self.jaw_vmax * float(np.clip(real_dt, 1e-3, 0.5))
        q_next = q + np.clip(self._jaw_goal - q, -dq_max, dq_max)
        # soft clamp
        lo = self.jaw_min + self.jaw_margin
        hi = self.jaw_max - self.jaw_margin
        q_next = float(np.clip(q_next, lo, hi))
        self.set_jaw_servo(q_next)

    def open_jaw(self, rad: float = 1.0):
        jaw_open_rad = rad
        self.set_jaw_goal(jaw_open_rad)

    def close_jaw(self):
        jaw_close_rad = -0.5  # TUNE: "tight" close that's still inside limits (with margin)
        self.set_jaw_goal(jaw_close_rad)

    def move_delta(self, delta_xyz: Tuple[float, float, float], quat=None):
        pos = np.asarray(self.current_pos(), dtype=np.float32)
        pos += np.asarray(delta_xyz, dtype=np.float32)
        pos -= np.asarray(self.offset, dtype=np.float32)
        self.send_pose(pos_xyz=pos, quat=quat)


class SurgFlowJointController:
    """Dual-arm CRTK joint controller for SurgFlow's 14D action layout.

    Layout:
      0..5  cutter/PSM2 joints, 6 cutter jaw
      7..12 retractor/PSM1 joints, 13 retractor jaw
    """

    CUTTER_JOINTS = slice(0, 6)
    CUTTER_JAW = 6
    RETRACT_JOINTS = slice(7, 13)
    RETRACT_JAW = 13
    ACTION_DIM = 14

    def __init__(
        self,
        node: Node,
        cutter_measured_js_topic: str = "/PSM2/measured_js",
        cutter_measured_jaw_topic: str = "/PSM2/jaw/measured_js",
        retract_measured_js_topic: str = "/PSM1/measured_js",
        retract_measured_jaw_topic: str = "/PSM1/jaw/measured_js",
        cutter_move_topic: str = "/PSM2/move_jp",
        retract_move_topic: str = "/PSM1/move_jp",
        cutter_servo_topic: str = "/PSM2/servo_jp",
        retract_servo_topic: str = "/PSM1/servo_jp",
        cutter_jaw_cmd_topic: str = "/PSM2/jaw/servo_jp",
        retract_jaw_cmd_topic: str = "/PSM1/jaw/servo_jp",
    ):
        self.node = node
        self.cutter_js: np.ndarray | None = None
        self.cutter_jaw: float | None = None
        self.retract_js: np.ndarray | None = None
        self.retract_jaw: float | None = None

        node.create_subscription(JointState, cutter_measured_js_topic, self._on_cutter_js, 10)
        node.create_subscription(JointState, cutter_measured_jaw_topic, self._on_cutter_jaw, 10)
        node.create_subscription(JointState, retract_measured_js_topic, self._on_retract_js, 10)
        node.create_subscription(JointState, retract_measured_jaw_topic, self._on_retract_jaw, 10)

        self.cutter_move_pub = node.create_publisher(JointState, cutter_move_topic, 10)
        self.retract_move_pub = node.create_publisher(JointState, retract_move_topic, 10)
        self.cutter_servo_pub = node.create_publisher(JointState, cutter_servo_topic, 10)
        self.retract_servo_pub = node.create_publisher(JointState, retract_servo_topic, 10)
        self.cutter_jaw_pub = node.create_publisher(JointState, cutter_jaw_cmd_topic, 10)
        self.retract_jaw_pub = node.create_publisher(JointState, retract_jaw_cmd_topic, 10)

    @staticmethod
    def _joint_state(names: list[str], positions) -> JointState:
        msg = JointState()
        pos = np.asarray(positions, dtype=float).reshape(-1)
        msg.name = list(names)
        msg.position = pos.tolist()
        msg.velocity = [0.0] * len(pos)
        msg.effort = [0.0] * len(pos)
        return msg

    def _on_cutter_js(self, msg: JointState) -> None:
        if len(msg.position) >= 6:
            self.cutter_js = np.asarray(msg.position[:6], dtype=np.float32)

    def _on_cutter_jaw(self, msg: JointState) -> None:
        if msg.position:
            self.cutter_jaw = float(msg.position[0])

    def _on_retract_js(self, msg: JointState) -> None:
        if len(msg.position) >= 6:
            self.retract_js = np.asarray(msg.position[:6], dtype=np.float32)

    def _on_retract_jaw(self, msg: JointState) -> None:
        if msg.position:
            self.retract_jaw = float(msg.position[0])

    def current_agent_pos(self) -> np.ndarray | None:
        if any(v is None for v in (self.cutter_js, self.cutter_jaw, self.retract_js, self.retract_jaw)):
            return None
        return np.concatenate(
            [
                self.cutter_js, [self.cutter_jaw],
                self.retract_js, [self.retract_jaw],
            ],
            axis=0,
        ).astype(np.float32)

    def publish_target(self, target: np.ndarray, use_move: bool = False) -> None:
        target = np.asarray(target, dtype=np.float32).reshape(self.ACTION_DIM)
        cutter = self._joint_state(["cutter"], target[self.CUTTER_JOINTS])
        retract = self._joint_state(["retractor"], target[self.RETRACT_JOINTS])
        cutter_jaw = self._joint_state(["jaw"], [target[self.CUTTER_JAW]])
        retract_jaw = self._joint_state(["jaw"], [target[self.RETRACT_JAW]])

        if use_move:
            self.cutter_move_pub.publish(cutter)
            self.retract_move_pub.publish(retract)
        else:
            self.cutter_servo_pub.publish(cutter)
            self.retract_servo_pub.publish(retract)
        self.cutter_jaw_pub.publish(cutter_jaw)
        self.retract_jaw_pub.publish(retract_jaw)

    def wait_for_feedback(self, timeout_s: float, spin_once) -> bool:
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            if self.current_agent_pos() is not None:
                return True
            spin_once(self.node, timeout_sec=0.05)
        return self.current_agent_pos() is not None
