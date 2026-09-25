#!/usr/bin/env python

import crtk
import dvrk
import PyKDL
import sys
import time
import math
import numpy as np
from sensor_msgs.msg import JointState
from typing import Tuple
from arclab_python import utils
from crtk_msgs.msg import CartesianImpedance

class PSMControlApplication:

    '''
    dvrk control application for PSM arms

    Supported functionalities:
        - gripper close and open
        - end effector pose control in base frames
    
    TODO:
        - Camera Calibrationmath.radians(-20.0)
        - ECM
        - MTM

    This code is based on ctrk and dvrk package, which according to dvrk's website, provide full functionality for dvrk,
    but might introduce performance penalty (https://dvrk.readthedocs.io/en/latest/pages/development/ros-clients/python.html#performance).

    '''

    def __init__(self, ral, expected_interval, psm_id):
        self.psm_id = psm_id
        self.ral = ral
        self.expected_interval = expected_interval
        self.psm = dvrk.psm(ral = ral,
                            arm_name = f"PSM{self.psm_id}",
                            expected_interval = expected_interval)
        
        self.jaw_pub = self.ral._node.create_publisher(
            JointState, f'/PSM{self.psm_id}/jaw/move_jp', 10)
        self.ci_pub = self.ral._node.create_publisher(
            CartesianImpedance, f'/PSM{self.psm_id}/servo_ci', 10)
        

        stiff_pos, damp_pos = -200.0, -5.0
        stiff_ori, damp_ori = -0.2, -0.01
        self.gains = CartesianImpedance()
        self.gains.position_negative.p.x = stiff_pos
        self.gains.position_positive.p.x = stiff_pos
        self.gains.position_negative.d.x = damp_pos
        self.gains.position_positive.d.x = damp_pos
        self.gains.position_negative.p.y = stiff_pos
        self.gains.position_positive.p.y = stiff_pos
        self.gains.position_negative.d.y = damp_pos
        self.gains.position_positive.d.y = damp_pos
        self.gains.position_negative.p.z = stiff_pos
        self.gains.position_positive.p.z = stiff_pos
        self.gains.position_negative.d.z = damp_pos
        self.gains.position_positive.d.z = damp_pos
        self.gains.orientation_negative.p.x = stiff_ori
        self.gains.orientation_positive.p.x = stiff_ori
        self.gains.orientation_negative.d.x = damp_ori
        self.gains.orientation_positive.d.x = damp_ori
        self.gains.orientation_negative.p.y = stiff_ori
        self.gains.orientation_positive.p.y = stiff_ori
        self.gains.orientation_negative.d.y = damp_ori
        self.gains.orientation_positive.d.y = damp_ori
        self.gains.orientation_negative.p.z = stiff_ori
        self.gains.orientation_positive.p.z = stiff_ori
        self.gains.orientation_negative.d.z = damp_ori
        self.gains.orientation_positive.d.z = damp_ori

    def __del__(self):
        self.ral.shutdown()

    def home(self):
        '''
        check connections, home the psm1not and psm2, and move to starting position
        '''
        self.ral.check_connections()
        if not self.psm.enable(10):
            sys.exit('psm1 failed to enable within 10 seconds')
        if not self.psm.home(10):
            sys.exit('psm1 failed to home within 10 seconds')
        print('psm1 move to starting position')
        goal = np.copy(self.psm.setpoint_jp())
        goal.fill(0)
        goal[2] = 0.12
        self.psm.move_jp(goal)
    
    def gripper_action(self, close):
        '''
        close or open the gripper of the psm
        '''
        # JointState_msg = JointState()
        # JointState_msg.name = ['jaw']
        current_js = self.psm.jaw.measured_js()[0].item()
        if close:
            # JointState_msg.position = [np.deg2rad(-20.0)]
            # JointState_msg.velocity = [0.0]
            # JointState_msg.effort = [0.0]
            # self.psm.jaw.servo_jf(np.array([-0.02]))
            # target = np.deg2rad(-10.0)
            # if target < current_js:
            #     intermediate = np.arange(current_js, target, -np.deg2rad(3))
            #     for joint_states in intermediate:
            #         self.psm.jaw.servo_jp(np.array([joint_states]))
            #         time.sleep(0.01)
            # self.psm.jaw.servo_jp(np.array([target]))
            self.psm.jaw.servo_jf(np.array([-0.05]))
        else:
            # JointState_msg.position = [np.deg2rad(45.0)]
            # JointState_msg.velocity = [0.0]
            # JointState_msg.effort = [0.0]
            # self.psm.jaw.servo_jf(np.array([-0.0]))
            # self.psm.jaw.servo_jp(np.array([np.deg2rad(45.0)]))
            target = np.deg2rad(45.0)
            if target > current_js:
                intermediate = np.arange(current_js, target, np.deg2rad(3))
                for joint_states in intermediate:
                    self.psm.jaw.servo_jp(np.array([joint_states]))
                    time.sleep(0.01)
            self.psm.jaw.servo_jp(np.array([target]))
        # self.jaw_pub.publish(JointState_msg)

    def get_ee_pose_base(self) -> PyKDL.Frame:
        '''
        get the current end effector pose in psm base frame
        '''
        return self.psm.measured_cp(age=0.01)
    
    def get_joint_states(self) -> Tuple[np.ndarray]:
        '''
        get the current joint states
        '''
        return self.psm.measured_js()
    
    def control_joint_states(self, target_joint_states: np.ndarray, wait=True, steps=1):
        '''
        move the psm to the target joint states

        args:
            target_joint_states: the target joint states
            wait: whether to wait for the movement to finish
            step: a way to make the movement slower
        '''
        curr_joint_states = self.get_joint_states()[0]

        intermediate = np.linspace(curr_joint_states, target_joint_states, steps+1)[1:]
        for joint_states in intermediate:
            if wait:
                self.psm.move_jp(joint_states).wait()
            else:
                self.psm.move_jp(joint_states)
            time.sleep(0.05)

    def servo_joint_states(self, target_joint_states: np.ndarray, wait=True, steps=1):
        '''
        move the psm to the target joint states

        args:
            target_joint_states: the target joint states
            wait: whether to wait for the movement to finish
            step: a way to make the movement slower
        '''
        curr_joint_states = self.get_joint_states()[0]

        intermediate = np.linspace(curr_joint_states, target_joint_states, steps+1)[1:]
        for joint_states in intermediate:
            if wait:
                self.psm.servo_jp(joint_states).wait()
            else:
                self.psm.servo_jp(joint_states)
            time.sleep(0.05)

    def control_base_pose_ee(self,
                             target_pose: PyKDL.Frame,
                             wait=True):
        '''
        move the end effector to the target pose in the base frame

        args:
            target_pose: the target pose in the base frame
            wait: whether to wait for the movement to finish
        '''
        if wait:
            self.psm.move_cp(target_pose).wait()
        else:
            self.psm.move_cp(target_pose)

    def control_base_pose_ee_impedance(self, goal_base_pose_ee: PyKDL.Frame):
        goal_ee_pos, goal_ee_quat = utils.PYKDLFrame_to_posquat(goal_base_pose_ee)  

        self.gains.force_position.x = goal_ee_pos[0]
        self.gains.force_position.y = goal_ee_pos[1]
        self.gains.force_position.z = goal_ee_pos[2]
        self.gains.force_orientation.x = goal_ee_quat[1]
        self.gains.force_orientation.y = goal_ee_quat[2]
        self.gains.force_orientation.z = goal_ee_quat[3]
        self.gains.force_orientation.w = goal_ee_quat[0]
        self.gains.torque_orientation.x = goal_ee_quat[1]
        self.gains.torque_orientation.y = goal_ee_quat[2]
        self.gains.torque_orientation.z = goal_ee_quat[3]
        self.gains.torque_orientation.w = goal_ee_quat[0]

        self.ci_pub.publish(self.gains)

    def control_base_pose_ee_interpolate(
        self, 
        goal_base_pose_ee: PyKDL.Frame, 
        pos_dist_th: float = 1e-3, 
        angle_dist_th: float = np.deg2rad(1),
        max_pos_step: float = 1e-3,
        max_angle_step: float = np.deg2rad(1)
    ): 
        """
        control the end effector to the goal pose in the base frame using interpolation
        (not sure if this is mathematically different from control_base_pose_ee)

        TODO: adjust linear and slerp interpolation step sizes

        args: 
            goal_ee_pose_base: (qw, qx, qy, qz, x, y, z)
            pos_dist_th: threshold for position distance
            angle_dist_th: threshold for angle distance
        """

        goal_ee_pos, goal_ee_quat = utils.PYKDLFrame_to_posquat(goal_base_pose_ee)

        reach_goal = False
        while not reach_goal: 

            curr_base_pose_ee = self.get_ee_pose_base()
            curr_ee_pos, curr_ee_quat = utils.PYKDLFrame_to_posquat(curr_base_pose_ee)

            pos_dist = np.linalg.norm(goal_ee_pos - curr_ee_pos)
            angle_dist, _ = utils.angleDist(curr_ee_quat, goal_ee_quat)

            if pos_dist < pos_dist_th and angle_dist < angle_dist_th: 
                reach_goal = True
                break

            ac_pos_base_fee = (goal_ee_pos - curr_ee_pos) * min(max_pos_step, pos_dist) / pos_dist
            next_pos_base_fee = curr_ee_pos + ac_pos_base_fee
            next_quat_base_fee = utils.slerp(curr_ee_quat, goal_ee_quat, max_angle_step)

            next_base_pose_ee = utils.posquat_to_PYKDLFrame(next_pos_base_fee, next_quat_base_fee)
            self.control_base_pose_ee(next_base_pose_ee, wait=False)
            
            # print(f"pos_dist: {pos_dist}, angle_dist: {angle_dist}")
            time.sleep(0.01)
    

def get_single_psm_control_application(psm_id):
    ral = crtk.ral('dvrk_psm_control')
    app = PSMControlApplication(ral, 0.01, psm_id)
    ral.spin()
    return app

def get_psm12_control_application():
    ral = crtk.ral('dvrk_psm_control')
    app1 = PSMControlApplication(ral, 0.01, '1')
    app2 = PSMControlApplication(ral, 0.01, '2')    
    ral.spin()
    return app1, app2