import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

import argparse
import os
import sys
import tty
import termios
import threading

from writer import EpisodeWriter

_HERE = os.path.dirname(os.path.abspath(__file__))
# src/arclab_dvrk/src/data_collection -> repository root, so the default output
# follows the checkout instead of a path baked in from another machine.
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 4)))
DEFAULT_TASK_DIR = os.path.join(_REPO_ROOT, "data", "diffusion_one_task_data")


class KeyboardListener(threading.Thread):
    """
    Background thread that reads single keypresses from stdin (no Enter needed).
    Calls on_start() when 's' is pressed, on_end() when 'e' is pressed.
    """

    def __init__(self, on_start, on_end):
        super().__init__(daemon=True)
        self.on_start = on_start
        self.on_end = on_end
        self._stop_event = threading.Event()

    def run(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not self._stop_event.is_set():
                ch = sys.stdin.read(1)
                if ch == 's':
                    self.on_start()
                elif ch == 'e':
                    self.on_end()
                elif ch == '\x03':  # Ctrl+C
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def stop(self):
        self._stop_event.set()


class DVRKDataCollector(Node):

    '''
    Ros2 node for collecting dVRK data.

    control:
    - press 's': start episode
    - press 'e': end and save episode
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

        # Timer: run at 30 Hz
        self.frequency = 30.
        self.timer = self.create_timer(1 / self.frequency, self.timer_callback)

        self.recording = False

        self.writer = EpisodeWriter(task_dir=self.task_dir)
        self.get_logger().info(f"Episodes will be written to: {self.task_dir}")

        self._keyboard = KeyboardListener(
            on_start=self._keyboard_start,
            on_end=self._keyboard_end,
        )
        self._keyboard.start()

        self.get_logger().info("Ready. Press 's' to start episode, 'e' to end and save.")

    # ------------------------------------------------------------------
    # Keyboard callbacks (called from KeyboardListener thread)
    # ------------------------------------------------------------------

    def _keyboard_start(self):
        if self.recording:
            self.get_logger().warn("Already recording — press 'e' to end the current episode first.")
            return
        self._print_topic_status()
        self.start_recording()

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
        all_received = all(topics.values())
        if all_received:
            self.get_logger().info("All topics received.")
        else:
            missing = [name for name, received in topics.items() if not received]
            self.get_logger().warn(f"Missing topics: {missing}")

    def _keyboard_end(self):
        if not self.recording:
            self.get_logger().warn("Not recording — press 's' to start an episode first.")
            return
        self.stop_and_save()

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

    def jaw2_callback(self, msg: JointState):
        self.jaw2_position = msg.position[0]
        self.jaw2_effort = msg.effort[0]

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
        self.writer.create_episode()
        self.recording = True
        self.get_logger().info(f"Started recording episode_{self.writer.episode_id:04d}")

    def stop_and_save(self):
        self.writer.save_episode()
        self.recording = False
        self.get_logger().info(f"Saved episode_{self.writer.episode_id:04d}")

    def destroy_node(self):
        self._keyboard.stop()
        self.writer.close()
        super().destroy_node()


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Collect dVRK episodes with keyboard start/stop."
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
