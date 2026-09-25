import sys
import cv2
import rclpy
import PyKDL
import numpy as np
import transforms3d.quaternions as quaternions

from rclpy.node import Node
from rclpy.time import Time
from rclpy.task import Future
from rclpy.duration import Duration
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Transform, TransformStamped
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException


def imgmsg_to_cv2(img_msg: Image):
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    if img_msg.encoding == 'rgb8': 
        image_opencv = cv2.cvtColor(image_opencv, cv2.COLOR_RGB2BGR)
    return image_opencv


def transform_point(source_frame: str,
                 target_frame: str,
                 p : np.ndarray,
                 is_vector=False) -> np.ndarray:
    t_pose_s : TransformStamped = lookup_transform(source_frame, target_frame)
    t_H_s = convert_pose_rep("Transform", "H", t_pose_s.transform)
    s_p = np.array([p[0], p[1], p[2], 1]) if not is_vector else np.array([p[0], p[1], p[2], 0])
    t_p = np.dot(t_H_s, s_p)
    return t_p[:3]    

def wait_for_message(node: Node, topic, msg_type, timeout=None):
    """
    Waits for a single message on the specified topic.

    :param node: The rclpy node instance.
    :param topic: The topic name to subscribe to.
    :param msg_type: The message type (e.g., std_msgs.msg.String).
    :param timeout: Timeout in seconds, or None for no timeout.
    :return: The received message, or None if the timeout occurs.
    """
    future = Future()

    def callback(msg):
        if not future.done():
            future.set_result(msg)

    # Create a temporary subscription
    subscription = node.create_subscription(msg_type, topic, callback, 10)

    # Spin until a message is received or timeout occurs
    try:
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    finally:
        # Clean up subscription
        node.destroy_subscription(subscription)

    return future.result() if future.done() else None



def lookup_transform(node: Node,
                    source_frame: str,
                    target_frame: str,
                    timeout: float = None,
                    tf_buffer: Buffer = None) -> TransformStamped:
    """
    Get a transform between two frames (t_pose_s).

    :param source_frame: The source frame name.
    :param target_frame: The target frame name.
    :param timeout: Optional timeout in seconds to wait for the transform.
    :return: The TransformStamped object containing the transform, or None if not found.
    """

    # Set up a Buffer and TransformListener
    if tf_buffer is None:
        tf_buffer = Buffer()
        tf_listener = TransformListener(tf_buffer, node)
    
    end_time = node.get_clock().now() + Duration(seconds=timeout) if timeout else None

    while rclpy.ok():
        # If there's a timeout specified, break if time is up
        if end_time and node.get_clock().now() > end_time:
            node.get_logger().error(f"Timeout reached while waiting for transform from '{source_frame}' to '{target_frame}'")
            return None

        # Look up the transform
        try:
            transform = tf_buffer.lookup_transform(target_frame, source_frame, Time())
            return transform
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # node.get_logger().warn(f"Waiting for transform from '{source_frame}' to '{target_frame}': {e}")
            pass
        
        # Sleep for a short duration before retrying
        rclpy.spin_once(node, timeout_sec=0.1)


# source and traget are in ["posquat", "pose", "Transform", "PYKDLFrame"]
def convert_pose_rep(source: str, target: str, pose):
    if source != "posquat" and target != "posquat":
        pos, quat = globals()[f"{source}_to_posquat"](pose)
        return globals()[f"posquat_to_{target}"](pos, quat)
    
    if source == "posquat":
        return globals()[f"{source}_to_{target}"](*pose)
    else:
        return globals()[f"{source}_to_{target}"](pose)

def Pose_to_posquat(pose: Pose):
    pos = np.array([pose.position.x, pose.position.y, pose.position.z])
    quat = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
    return pos, quat

def posquat_to_Pose(pos, quat) -> Pose:
    pose = Pose()
    pose.position.x = pos[0]
    pose.position.y = pos[1]
    pose.position.z = pos[2]
    pose.orientation.w = quat[0]
    pose.orientation.x = quat[1]
    pose.orientation.y = quat[2]
    pose.orientation.z = quat[3]
    return pose

def posquat_to_Transform(pos, quat) -> Transform:
    transform = Transform()
    transform.translation.x = pos[0]
    transform.translation.y = pos[1]
    transform.translation.z = pos[2]
    transform.rotation.w = quat[0]
    transform.rotation.x = quat[1]
    transform.rotation.y = quat[2]
    transform.rotation.z = quat[3]
    return transform

def Transform_to_posquat(transform: Transform):
    pos = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
    quat = np.array([transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z])
    return pos, quat

def posquat_to_H(pos, quat): 
    H = np.zeros([4,4])
    H[:3,3] = pos
    H[:3,:3] = quaternions.quat2mat(quat)
    H[3,3] = 1

    return H

def H_to_posquat(H): 
    return H[:3,3], quaternions.mat2quat(H[:3,:3]) #quat: wxyz

def posquat_to_PYKDLFrame(pos, quat): 
    frame = PyKDL.Frame(PyKDL.Rotation.Quaternion(quat[1],
                                                  quat[2],
                                                  quat[3],
                                                  quat[0]),
                       PyKDL.Vector(pos[0],
                                    pos[1],
                                    pos[2]))
    return frame

def PYKDLFrame_to_posquat(frame):
    x, y, z, w = frame.M.GetQuaternion()
    p = frame.p
    return np.array([p[0], p[1], p[2]]), np.array([w, x, y, z])


def angleDist(q1, q2): 
    if np.dot(q1, q2) < 0: 
        q1 = -q1
    return np.arccos(np.clip(np.dot(q1, q2), -1, 1)), q1 #assume both q1 and q2 are unit-norm

def slerp(q_now, q_end, max_angle_step = 1*np.pi/180): 
    angle_dist, q_now = angleDist(q_now, q_end)
    move_angle = min(1*np.pi/180, angle_dist)
    on_q_end = q_end - np.dot(q_now, q_end) * q_now #(q_now, on_q_end) are not orthonormal
    on_q_end /= np.linalg.norm(on_q_end)

    return q_now * np.cos(move_angle) + on_q_end * np.sin(move_angle)

