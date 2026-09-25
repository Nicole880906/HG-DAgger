import os
import rclpy
import scipy
import time
import numpy as np
import pyquaternion as pyquat

from tqdm import tqdm
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped, Transform, Vector3, Quaternion
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster

from arclab_python import utils
from arclab_python.psm_control import get_single_psm_control_application, PSMControlApplication




class CauterySpatula(Node):
    '''
    wrapper on top of the PSMControlApplication to control the cautery spatula
    '''
    def __init__(self, psm_id, psm_control_app : PSMControlApplication, penetration_offset=0.0):
        super().__init__('cautery_spatula_controller')
        self.psm_id = psm_id
        self.psm_control_app = psm_control_app
        assert(self.psm_id == self.psm_control_app.psm_id)

        self.create_subscription(PoseStamped, f'/PSM{self.psm_id}/measured_cp', self._psm_pose_ee, 1)
        self.psm_pose_tip_pub = self.create_publisher(PoseStamped, f'/PSM{self.psm_id}/spatula_tip_pose', 1)

        # spatula kinematic in translation and quaternion
        self.ee_t_tip = np.array([-1.5e-3, 0, 18e-3])
        self.ee_q_tip = np.array([1., 0., 0., 0.])

        self.dynamic_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.broadcast_psm_pose_tip()
        self.penetration_offset = penetration_offset



    def broadcast_psm_pose_tip(self):
        '''
        broadcast calculated spatula kinematic to tf tree 
        '''
        ee_pose_tip = TransformStamped(transform=Transform(translation=Vector3(x=self.ee_t_tip[0], y=self.ee_t_tip[1], z=self.ee_t_tip[2]),
                                                           rotation=Quaternion(x=self.ee_q_tip[1], y=self.ee_q_tip[2], z=self.ee_q_tip[3],
                                                                               w=self.ee_q_tip[0])))
        ee_pose_tip.header.stamp = self.get_clock().now().to_msg()
        ee_pose_tip.header.frame_id = f"PSM{self.psm_id}_tool_tip_link" # tool tip link same with ee
        ee_pose_tip.child_frame_id = "spatula_tip"
        self.static_broadcaster.sendTransform(ee_pose_tip)

    def _get_psm_pose_tip(self):
        ee_T_tip = utils.convert_pose_rep('posquat', 'H', (self.ee_t_tip, self.ee_q_tip))
        psm_T_ee = utils.convert_pose_rep('Pose', 'H', self.psm_pose_ee)
        psm_T_tip = psm_T_ee @ ee_T_tip
        return utils.convert_pose_rep('H', 'posquat', psm_T_tip)

    def _psm_pose_ee(self, msg):
        self.psm_pose_ee = msg.pose

    def compute_ee_pose_from_tip_motion(self,
                                        next,
                                        curr,
                                        normal,
                                        tangent=None,
                                        use_neg=False):
        '''
        compute the next ee pose given the next desired tip pose, all data should be in the base frame

        next: next desired tip position
        nomral: normal of the cutting surface, tip pose should be parallel to this normal
        tangent: tangent of the cutting trajectory, x-axis of the tip frame should be parallel to this tangent    
        '''
        normal = normal / np.linalg.norm(normal)
        # next, curr = next + self.penetration_offset * normal, curr + self.penetration_offset * normal
        if tangent is None:
            tangent = (next - curr) / np.linalg.norm(next - curr)

        # solve for orientation using wahba's algorithm
        p0, p1, p2, p3 = np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
        q0 = np.array([0, 0, 0])
        
        # TODO: use the cloest distant of -tangent or tangent
        if use_neg:
            q2 = -tangent
        else:
            q2 = tangent

        q3 = -normal
        q1 = np.cross(q2, q3)
        p, q = np.stack([p0, p1, p2, p3]), np.stack([q0, q1, q2, q3]) + next

        tip_T_ee = np.eye(4)
        tip_T_ee[:3, -1] = -self.ee_t_tip

        try:
            # more weight on the tangent direction
            weights = np.array([1, 1, 1.5, 1])
            rot = scipy.spatial.transform.Rotation.align_vectors(q, p, weights)[0].as_matrix()
        except np.linalg.LinAlgError:
            print('rotation alignment didn\'t converge')
            psm_T_ee = utils.convert_pose_rep('Pose', 'H', self.psm_pose_ee)
            curr_psm_T_tip = psm_T_ee @ np.linalg.inv(tip_T_ee)
            rot = curr_psm_T_tip[:3, :3]
            
        psm_T_tip = np.eye(4)
        psm_T_tip[:3, :3] = rot
        psm_T_tip[:3, -1] = next

        desired_psm_T_ee = psm_T_tip @ tip_T_ee

        desired_t, desired_q = utils.convert_pose_rep('H', 'posquat', desired_psm_T_ee)
        return desired_psm_T_ee, desired_t, desired_q
    
    def compute_shortest_ee_pose_from_tip_motion(self, next, curr, normal, tangent=None, vis=False, move=False):
        '''
        input:
            because tangent can be either direction, compute the shortest distance between the current orientation and the desired orientation

        output:
            desired_psm_T_ee, desired_t, desired_q in psm frame
        '''
        desired_psm_T_ee_1, desired_t_1, desired_q_1 = self.compute_ee_pose_from_tip_motion(next, curr, normal, use_neg=True, tangent=tangent)
        desired_psm_T_ee_2, desired_t_2, desired_q_2 = self.compute_ee_pose_from_tip_motion(next, curr, normal, use_neg=False, tangent=tangent)
        
        q_curr = utils.convert_pose_rep('Pose', 'posquat', self.psm_pose_ee)[1]

        # compute shorter chordal distance between q_goal and q_curr
        _desired_q1 = pyquat.Quaternion(desired_q_1[0], desired_q_1[1], desired_q_1[2], desired_q_1[3])
        _desired_q2 = pyquat.Quaternion(desired_q_2[0], desired_q_2[1], desired_q_2[2], desired_q_2[3])
        _q_curr = pyquat.Quaternion(q_curr[0], q_curr[1], q_curr[2], q_curr[3])

        d1, d2 = pyquat.Quaternion.absolute_distance(_desired_q1, _q_curr), pyquat.Quaternion.absolute_distance(_desired_q2, _q_curr)
        desired_psm2_T_ee, desired_t, desired_q = (desired_psm_T_ee_1, desired_t_1, desired_q_1) if d1 < d2 else (desired_psm_T_ee_2, desired_t_2, desired_q_2)
        return desired_psm2_T_ee, desired_t, desired_q
    
    def move_tip(self, target_ee_pose, vis=False, move=True, max_pos_step=1e-3):
        '''
        move the tip to the desired position
        '''
        transform = utils.convert_pose_rep('H', 'Transform', target_ee_pose)
        if vis:
            ee_pose_tip = TransformStamped(transform=transform)
            ee_pose_tip.header.stamp = self.get_clock().now().to_msg()
            ee_pose_tip.header.frame_id = f"PSM{self.psm_id}_base" # tool tip link same with ee
            ee_pose_tip.child_frame_id = "target_spatula_ee"
            self.dynamic_broadcaster.sendTransform(ee_pose_tip)
            if not move: time.sleep(0.5)

        if move: self.psm_control_app.control_base_pose_ee_interpolate(utils.convert_pose_rep('H', 'PYKDLFrame', target_ee_pose),
                                                                       max_pos_step=max_pos_step)

        


    def execute_trajectory(self,
                           trajectory,
                           segment_normals,
                           move_to_start=False,
                           vis=False,
                           move=True,
                           max_pos_step=1e-3):
        '''
        execute the trajectory
        '''
        if move_to_start:
            trajectory += self.penetration_offset * segment_normals[0] / np.linalg.norm(segment_normals[0])
            # approximate the initial pose wit
            tangent = trajectory[1] - trajectory[0]
            tangent = tangent / np.linalg.norm(tangent)
            normal = segment_normals[0]
            desired_psm_T_ee, _, _ = self.compute_shortest_ee_pose_from_tip_motion(trajectory[0],
                                                                                   trajectory[0],
                                                                                   normal,
                                                                                   tangent=tangent)
            


            self.move_tip(desired_psm_T_ee, vis=vis, move=move, max_pos_step=max_pos_step*2)

        for i in tqdm(range(len(trajectory) - 1)):
            normal = segment_normals[i]
            desired_psm_T_ee, _, _ = self.compute_shortest_ee_pose_from_tip_motion(trajectory[i + 1],
                                                                                   trajectory[i],
                                                                                   normal,
                                                                                   tangent=None)

            self.move_tip(desired_psm_T_ee, vis=vis, move=move, max_pos_step=max_pos_step)

    def execute_trajectory_zigzag(self,
                           trajectory,
                           segment_normals,
                           move_to_start=False,
                           vis=False,
                           move=True,
                           max_pos_step=1e-3,
                           zigzag=1,
                           amplitude=2e-3):
        '''
        execute the trajectory
        '''
        if move_to_start:
            trajectory += self.penetration_offset * segment_normals[0] / np.linalg.norm(segment_normals[0])
            # approximate the initial pose wit
            tangent = trajectory[1] - trajectory[0]
            tangent = tangent / np.linalg.norm(tangent)
            normal = segment_normals[0]
            desired_psm_T_ee, _, _ = self.compute_shortest_ee_pose_from_tip_motion(trajectory[0],
                                                                                   trajectory[0],
                                                                                   normal,
                                                                                   tangent=tangent)
            


            self.move_tip(desired_psm_T_ee, vis=vis, move=move, max_pos_step=max_pos_step*2)

        for i in tqdm(range(len(trajectory) - 1)):
            normal = segment_normals[i]
            start, end = trajectory[i], trajectory[i + 1]
            tangent = (end - start) / np.linalg.norm(end - start)

            zigzag_waypoints = np.linspace(start, end, zigzag*4+1)
            for j in range(zigzag*4):
                inter_start, inter_end = zigzag_waypoints[j], zigzag_waypoints[j + 1]
                if j % 2 == 0:
                    offset = np.zeros(3)
                elif j % 4 == 1:
                    offset = amplitude * normal / np.linalg.norm(normal)
                elif j % 4 == 3:
                    offset = -amplitude * normal / np.linalg.norm(normal) 
                inter_end += offset
                desired_psm_T_ee, _, _ = self.compute_shortest_ee_pose_from_tip_motion(inter_end,
                                                                                       inter_start,
                                                                                       normal,
                                                                                       tangent=tangent)
                self.move_tip(desired_psm_T_ee, vis=vis, move=move, max_pos_step=max_pos_step)



if __name__ == '__main__':
    rclpy.init()
    psm_control_app = get_single_psm_control_application('1')
    cautery_spatula_controller = CauterySpatula('1', psm_control_app)
    rclpy.spin(cautery_spatula_controller)
    rclpy.shutdown()
