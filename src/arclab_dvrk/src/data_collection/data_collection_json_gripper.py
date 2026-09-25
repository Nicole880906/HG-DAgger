import os
import time
import argparse
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from writer import EpisodeWriter

_HERE = os.path.dirname(os.path.abspath(__file__))
# src/arclab_dvrk/src/data_collection -> repository root, so the default output
# follows the checkout instead of a path baked in from another machine.
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 4)))
DEFAULT_TASK_DIR = os.path.join(_REPO_ROOT, "data", "diffusion_one_task_data")


class DVRKDataCollector(Node):

    '''
    Ros2 node for collecting dVRK data.

    Identical to data_collection_json_keyboard.py in what it records; only
    the start/end control differs:
    - 3 fast PSM1 jaw open-close cycles (within `time_window` s): start episode
    - 3 fast PSM2 jaw open-close cycles (within `time_window` s): end + save
    '''

    def __init__(self, task_dir=DEFAULT_TASK_DIR):
        super().__init__('dvrk_data_collector')
        self.bridge = CvBridge()
        self.task_dir = task_dir

        # Subscribers
        self.create_subscription(Image, '/stereo/left/rectified_downscaled_image', self.left_image_callback, 10)
        self.create_subscription(Image, '/stereo/right/rectified_downscaled_image', self.right_image_callback, 10)
        self.create_subscription(JointState, '/PSM1/measured_js', self.psm1_js_callback, 10)
        self.create_subscription(JointState, '/PSM2/measured_js', self.psm2_js_callback, 10)
        self.create_subscription(PoseStamped, '/PSM1/measured_cp', self.psm1_cp_callback, 10)
        self.create_subscription(PoseStamped, '/PSM2/measured_cp', self.psm2_cp_callback, 10)
        self.create_subscription(JointState, '/PSM1/jaw/measured_js', self.jaw1_callback, 10)
        self.create_subscription(JointState, '/PSM2/jaw/measured_js', self.jaw2_callback, 10)

        # Publishers used to move PSMs back to the saved initial pose before
        # each episode starts — see _move_to_initial_position().
        self.retractor_move_pub = self.create_publisher(
            JointState, '/PSM1/move_jp', 10)
        self.cutter_move_pub = self.create_publisher(
            JointState, '/PSM2/move_jp', 10)
        self.retractor_jaw_servo_pub = self.create_publisher(
            JointState, '/PSM1/jaw/servo_jp', 10)
        self.cutter_jaw_servo_pub = self.create_publisher(
            JointState, '/PSM2/jaw/servo_jp', 10)

        # Saved initial joint/jaw states, checked in alongside this file. The
        # recorder that produced them (data_collection_json_cpw.py) has been
        # removed; re-record by saving np.array(joint_states) / np.array(
        # jaw_states) to these names if the home pose ever changes.
        self.initial_retractor_joint_state = self._safe_load(
            'psm1_initial_joint_state.npy')
        self.initial_retractor_jaw_state = self._safe_load(
            'psm1_initial_jaw_state.npy')
        self.initial_cutter_joint_state = self._safe_load(
            'psm2_initial_joint_state.npy')
        self.initial_cutter_jaw_state = self._safe_load(
            'psm2_initial_jaw_state.npy')

        # Timer: run at 30 Hz
        self.frequency = 30.
        self.timer = self.create_timer(1 / self.frequency, self.timer_callback)

        self.recording = False

        # Gripper-pinch trigger state
        self.jaw1_transition_times = deque()
        self.jaw1_state = 'open'
        self.jaw2_transition_times = deque()
        self.jaw2_state = 'open'
        self.n = 3                # number of open-close cycles required
        self.time_window = 2.0    # all 2n transitions must fall in this window (s)
        self.jaw_threshold = 0.05 # rad; state-change threshold

        self.writer = EpisodeWriter(task_dir=self.task_dir)
        self.get_logger().info(f"Episodes will be written to: {self.task_dir}")

        self.get_logger().info(
            "Ready. Pinch PSM1 jaw 3x fast to start, PSM2 jaw 3x fast to "
            "end+save. Camera frames are saved unmodified."
        )

    # ------------------------------------------------------------------
    # Initial-pose helpers
    # ------------------------------------------------------------------

    def _safe_load(self, fname: str):
        """Load .npy from this file's directory; return None + warn if missing."""
        path = os.path.join(_HERE, fname)
        if not os.path.exists(path):
            self.get_logger().warn(
                f"Initial-state file not found: {path} — episodes will NOT "
                f"be reset to a saved pose before recording.")
            return None
        return np.load(path)

    def _PSMMove(self, pub, joint_name: str, goal_state, sleep_time: float):
        """Publish a joint-space target to `/<arm>/move_jp` and wait for the
        arm to settle. `goal_state` is the saved joint vector (radians)."""
        if goal_state is None:
            return
        msg = JointState()
        msg.name = [joint_name]
        msg.position = np.asarray(goal_state).astype(float).tolist()
        msg.velocity = [0.0]
        msg.effort = [0.0]
        pub.publish(msg)
        time.sleep(sleep_time)

    def _jawServo(self, pub, end_pos, sleep_time: float):
        """Publish a jaw-angle target. `end_pos` may be scalar or 1-elem array."""
        if end_pos is None:
            return
        msg = JointState()
        msg.name = ['jaw']
        msg.position = [float(np.asarray(end_pos).reshape(-1)[0])]
        msg.velocity = [0.0]
        msg.effort = [0.0]
        pub.publish(msg)
        time.sleep(sleep_time)

    def _move_to_initial_position(self):
        """Move both PSMs to the saved joint pose + initial jaw angle. Blocks
        for a few seconds while the arms travel — called before each episode
        so every recording starts from the same pose."""
        self.get_logger().info("Moving PSMs to initial position...")
        self._PSMMove(self.retractor_move_pub, 'retractor',
                      self.initial_retractor_joint_state, sleep_time=3.0)
        self._PSMMove(self.cutter_move_pub, 'cutter',
                      self.initial_cutter_joint_state, sleep_time=3.0)
        self._jawServo(self.retractor_jaw_servo_pub,
                       self.initial_retractor_jaw_state, sleep_time=0.1)
        self._jawServo(self.cutter_jaw_servo_pub,
                       self.initial_cutter_jaw_state, sleep_time=0.1)
        self.get_logger().info("PSMs at initial position.")

    # ------------------------------------------------------------------
    # Gripper-pinch detection
    # ------------------------------------------------------------------

    def _register_jaw_transition(self, transitions: deque, now: float) -> bool:
        """Append `now` to `transitions`, drop entries older than `time_window`,
        and return True iff the deque now holds at least 2*n transitions."""
        transitions.append(now)
        while transitions and now - transitions[0] > self.time_window:
            transitions.popleft()
        return len(transitions) >= self.n * 2

    def _print_topic_status(self):
        topics = {
            'left_image':     hasattr(self, 'left_image'),
            'right_image':    hasattr(self, 'right_image'),
            'psm1_js':        hasattr(self, 'psm1_js'),
            'psm2_js':        hasattr(self, 'psm2_js'),
            'psm1_cp':        hasattr(self, 'psm1_cp'),
            'psm2_cp':        hasattr(self, 'psm2_cp'),
            'jaw1_position':  hasattr(self, 'jaw1_position'),
            'jaw2_position':  hasattr(self, 'jaw2_position'),
        }
        if all(topics.values()):
            self.get_logger().info("All topics received.")
        else:
            missing = [name for name, received in topics.items() if not received]
            self.get_logger().warn(f"Missing topics: {missing}")

    # ------------------------------------------------------------------
    # Sensor callbacks
    # ------------------------------------------------------------------

    def left_image_callback(self, msg):
        self.left_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def right_image_callback(self, msg):
        self.right_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def psm1_js_callback(self, msg: JointState):
        self.psm1_js = msg.position
        self.psm1_js_vel = msg.velocity
        self.psm1_js_effort = msg.effort

    def psm2_js_callback(self, msg: JointState):
        self.psm2_js = msg.position
        self.psm2_js_vel = msg.velocity
        self.psm2_js_effort = msg.effort

    def psm1_cp_callback(self, msg: PoseStamped):
        self.psm1_cp = msg.pose

    def psm2_cp_callback(self, msg: PoseStamped):
        self.psm2_cp = msg.pose

    def jaw1_callback(self, msg: JointState):
        self.jaw1_position = msg.position[0]
        self.jaw1_effort = msg.effort[0]

        new_state = 'closed' if self.jaw1_position > self.jaw_threshold else 'open'
        if new_state != self.jaw1_state:
            self.jaw1_state = new_state
            if self._register_jaw_transition(self.jaw1_transition_times, time.time()):
                self.jaw1_transition_times.clear()
                self.get_logger().info('PSM1 gripper pattern detected')
                if not self.recording:
                    self._print_topic_status()
                    self.start_recording()
                else:
                    self.get_logger().warn(
                        "Already recording — pinch PSM2 jaw 3x to end and save."
                    )

    def jaw2_callback(self, msg: JointState):
        self.jaw2_position = msg.position[0]
        self.jaw2_effort = msg.effort[0]

        new_state = 'closed' if self.jaw2_position > self.jaw_threshold else 'open'
        if new_state != self.jaw2_state:
            self.jaw2_state = new_state
            if self._register_jaw_transition(self.jaw2_transition_times, time.time()):
                self.jaw2_transition_times.clear()
                self.get_logger().info('PSM2 gripper pattern detected')
                if self.recording:
                    self.stop_and_save()
                else:
                    self.get_logger().warn(
                        "Not recording — pinch PSM1 jaw 3x to start an episode."
                    )

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def timer_callback(self):
        if not self.recording:
            return

        if hasattr(self, 'left_image') and \
           hasattr(self, 'right_image') and \
           hasattr(self, 'psm1_js') and \
           hasattr(self, 'psm2_js') and \
           hasattr(self, 'psm1_cp') and \
           hasattr(self, 'psm2_cp') and \
           hasattr(self, 'jaw1_position') and \
           hasattr(self, 'jaw2_position'):

            colors = {
                "left_image": self.left_image,
                "right_image": self.right_image,
            }

            states = {
                "psm_cutter_js": {
                    "qpos": self.psm2_js.tolist(),
                    "qvel": self.psm2_js_vel.tolist(),
                    "qeffort": self.psm2_js_effort.tolist(),
                    "gripper": self.jaw2_position,
                    "gripper_effort": self.jaw2_effort,
                },
                "psm_retraction_js": {
                    "qpos": self.psm1_js.tolist(),
                    "qvel": self.psm1_js_vel.tolist(),
                    "qeffort": self.psm1_js_effort.tolist(),
                    "gripper": self.jaw1_position,
                    "gripper_effort": self.jaw1_effort,
                },
                "psm_cutter_ee": {
                    "psm_cutter_pos": [self.psm2_cp.position.x,
                                       self.psm2_cp.position.y,
                                       self.psm2_cp.position.z],
                    "psm_cutter_quat": [self.psm2_cp.orientation.w,
                                        self.psm2_cp.orientation.x,
                                        self.psm2_cp.orientation.y,
                                        self.psm2_cp.orientation.z],
                },
                "psm_retraction_ee": {
                    "psm_retraction_pos": [self.psm1_cp.position.x,
                                           self.psm1_cp.position.y,
                                           self.psm1_cp.position.z],
                    "psm_retraction_quat": [self.psm1_cp.orientation.w,
                                            self.psm1_cp.orientation.x,
                                            self.psm1_cp.orientation.y,
                                            self.psm1_cp.orientation.z],
                },
            }

            self.writer.add_item(colors=colors, states=states)

    # ------------------------------------------------------------------
    # Episode control
    # ------------------------------------------------------------------

    def start_recording(self):
        # Recording starts immediately from wherever the arms currently are.
        # The _move_to_initial_position() helper is still available — call it
        # manually if you want the arms to home before recording.
        if not self.writer.create_episode():
            return
        self.recording = True
        self.get_logger().info(f"Started recording episode_{self.writer.episode_id:04d}")

    def stop_and_save(self):
        self.writer.save_episode()
        self.recording = False
        self.get_logger().info(f"Saved episode_{self.writer.episode_id:04d}")

    def destroy_node(self):
        self.writer.close()
        super().destroy_node()


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Collect dVRK episodes with gripper-pinch start/stop."
    )
    parser.add_argument(
        "--task-dir",
        default=DEFAULT_TASK_DIR,
        help=f"where episodes are written (default: {DEFAULT_TASK_DIR})",
    )
    return parser.parse_known_args(args)


def resolve_task_dir(task_dir):
    """Absolute output directory for this task's episodes.

    A relative --task-dir is anchored to the repository root, not the cwd, so
    the collector writes to the same place regardless of where it is launched.
    """
    path = os.path.expanduser(task_dir)
    if not os.path.isabs(path):
        path = os.path.join(_REPO_ROOT, path)
    return os.path.abspath(path)


def main(args=None):
    parsed_args, ros_args = parse_args(args)
    rclpy.init(args=ros_args)
    collector = DVRKDataCollector(
        task_dir=resolve_task_dir(parsed_args.task_dir),
    )
    try:
        rclpy.spin(collector)
    except KeyboardInterrupt:
        collector.get_logger().info("Interrupted. Exiting.")
    finally:
        collector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
