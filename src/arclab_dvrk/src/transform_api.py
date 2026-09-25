import rclpy
import utils
import os
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionServer
# from arclab_dvrk.action import Transform
from interface.srv import TransformAPI
from geometry_msgs.msg import TransformStamped
from rclpy.executors import MultiThreadedExecutor
from tf2_ros.transform_broadcaster import TransformBroadcaster


from tf2_ros import BufferClient

class TransformApi(Node):

    '''
    Node that is a wrapper on top of buffer server to provide transform between different frames

    all broadcast calibration result
    '''

    def __init__(self):
        super().__init__('transform_api')

        self.cwd = os.path.dirname(os.path.abspath(__file__))
        self.asset_path = os.path.join(self.cwd, '..', 'assets')
        self.action_cb_group = ReentrantCallbackGroup()
        self.action_server = self.create_service(TransformAPI, 'transform_api', self.transform_service, callback_group=self.action_cb_group)

        self.tf_buffer_client = BufferClient(self, "/tf2_buffer_server", check_frequency=10)
        if not self.tf_buffer_client.action_client.wait_for_server(5):
            self.get_logger().error(f"Could not connect to TF buffer server")
        print("Connected to TF buffer server")

        self.dynamic_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(0.2, self.broadcast_transform)

    def transform_service(self, request, response):
        self.get_logger().info(f"Received request: {request.source_frame} -> {request.target_frame}")
        transform = self.tf_buffer_client.lookup_transform(request.source_frame,
                                                           request.target_frame,
                                                           rclpy.time.Time(),
                                                           rclpy.time.Duration(seconds=1))
        response.transform = transform
        return response
    
    def broadcast_transform(self):
        '''
        currently included:
        world_pose_cam, cam_pose_psm1, cam_pose_psm2
        
        '''
        # Create a TransformStamped message'
        data = np.load(os.path.join(self.asset_path, 'cam_pose_world.npz'))
        pos, quat = data['pos'], data['quat']
        world_pose_cam = TransformStamped()
        world_pose_cam.header.stamp = self.get_clock().now().to_msg()
        world_pose_cam.header.frame_id = 'dvrk_cam'
        world_pose_cam.child_frame_id = 'world'
        world_pose_cam.transform = utils.posquat_to_Transform(pos, quat)
        self.dynamic_broadcaster.sendTransform(world_pose_cam)

        data = np.load(os.path.join(self.asset_path, 'cam_pose_psm1.npz'))
        pos, quat = data['pos'], data['quat']
        cam_pose_psm1 = TransformStamped()
        cam_pose_psm1.header.stamp = self.get_clock().now().to_msg()
        cam_pose_psm1.header.frame_id = 'dvrk_cam'
        cam_pose_psm1.child_frame_id = 'PSM1_base'
        cam_pose_psm1.transform = utils.posquat_to_Transform(pos, quat)
        self.dynamic_broadcaster.sendTransform(cam_pose_psm1)

        data = np.load(os.path.join(self.asset_path, 'cam_pose_psm2.npz'))
        pos, quat = data['pos'], data['quat']
        cam_pose_psm2 = TransformStamped()
        cam_pose_psm2.header.stamp = self.get_clock().now().to_msg()
        cam_pose_psm2.header.frame_id = 'dvrk_cam'
        cam_pose_psm2.child_frame_id = 'PSM2_base'
        cam_pose_psm2.transform = utils.posquat_to_Transform(pos, quat)
        self.dynamic_broadcaster.sendTransform(cam_pose_psm2)


# class TransformAPI(Node):

#     '''
#     Node for managering calibration results
#     '''

#     def __init__(self):
#         super().__init__('transform_api')
#         self.action_cb_group = ReentrantCallbackGroup()
#         self.action_server = ActionServer(self, Transform, 'transform_api', self.action_callback, callback_group=self.action_cb_group)

#         self.tf_buffer_client = BufferClient(self, "/tf2_buffer_server", check_frequency=10)
#         self.waiting_rate = self.create_rate(100)
        
#         self.source = 'world'
#         self.target = 'dvrk_cam'

#         if not self.tf_buffer_client.action_client.wait_for_server(5):
#             self.get_logger().error(f"Could not connect to TF buffer server")
#         print("Connected to TF buffer server")


#     def action_callback(self, goal_handle):
#         self.get_logger().info(f"Received goal: {goal_handle.request.source_frame} -> {goal_handle.request.target_frame}")
#         transform = self.tf_buffer_client.lookup_transform(goal_handle.request.source_frame,
#                                                            goal_handle.request.target_frame,
#                                                            rclpy.time.Time(),
#                                                            rclpy.time.Duration(seconds=1))
#         result = Transform.Result()
#         result.transform = transform
#         goal_handle.succeed()
#         return result
    
def main(args=None):
    rclpy.init(args=args)

    node = TransformApi()    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()

    rclpy.shutdown()
    exit(0)

if __name__ == "__main__":
    main()