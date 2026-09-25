import cv2
import os
import cv_bridge
import numpy as np
import time
import rclpy
import tf2_ros
import utils
import transforms3d as tf3d
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped, Transform, Vector3, Quaternion


# NOTE: make sure the tool's end effector is at 0 joint angle

class MarkerCalibration(Node):
    
    def __init__(self, id='1'):
        super().__init__('marker_ctb_calibration') 
        self.cwd = os.path.dirname(os.path.abspath(__file__))
        self.id = id
        self.br = cv_bridge.CvBridge()
        self.broadcaster = tf2_ros.TransformBroadcaster(self)
        # subscribers
        self.P1 = np.array(utils.wait_for_message(self, '/stereo/rectified/P1', Float32MultiArray, 5).data).reshape(3,4)
        
        self.create_subscription(Image, '/stereo/left/rectified_downscaled_image', self._image_callback, 1)
        

        # makers kinematics to the ee frame, only m and h need to be modified
        if id != 'world':
            self.create_subscription(PoseStamped, f'/PSM{self.id}/measured_cp', self._psm_ee_pose, 1)
            if id == '2':
                self.h = 28e-3 # 28e-3 for cautery tool, 25e-3 for LND
            elif id == '1':
                self.h = 25e-3

            self.m = 38.1e-3
            
            self.ring_T_m = np.array([[1, 0, 0, 0.08 - self.m/2],
                                [0, 1, 0, (-0.0547+self.m)/2],
                                [0, 0, 1, (0.007)],
                                [0, 0, 0, 1]])
            self.m_T_ring = np.linalg.inv(self.ring_T_m)
            self.ring_T_ee = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, -self.h],
                                [0, 0, 0, 1]])
            def rotate_x(theta):
                return np.array([[1, 0, 0],
                                [0, np.cos(theta), -np.sin(theta)],
                                [0, np.sin(theta), np.cos(theta)]])
            self.ring_T_ee[:3, :3] = rotate_x(-np.pi)
        self.cam_T_marker = None

        self.create_timer(1/100, self.calibrate)
  
    @staticmethod
    def estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
        '''
        This will estimate the rvec and tvec for each of the marker corners detected by:
        corners, ids, rejectedImgPoints = detector.detectMarkers(image)
        corners - is an array of detected corners for each detected marker in the image
        marker_size - is the size of the detected markers
        mtx - is the camera matrix
        distortion - is the camera distortion matrix
        RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
        '''
        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
        trash = []
        rvecs = []
        tvecs = []
        for c in corners:
            nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
            rvecs.append(R)
            tvecs.append(t)
            trash.append(nada)
        return rvecs, tvecs, trash


    def _image_callback(self, img):
        self.left_image = self.br.imgmsg_to_cv2(img)
        gray = cv2.cvtColor(self.left_image, cv2.COLOR_RGB2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        parameters =  cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
        cam_mtx = self.P1[0:3, 0:3]
        dist = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])
        marker_size = 0.0381

        if np.all(ids != None):
            for i in range(ids.reshape(-1).shape[0]):
                id = ids.reshape(-1)[i]
                
                rvec, tvec, _ = self.estimatePoseSingleMarkers(corners[i], marker_size, cam_mtx, dist)
                rmat = cv2.Rodrigues(rvec[0])
                tvec = tvec[0]
                rmat = rmat[0]
                self.cam_T_marker = np.eye(4)
                self.cam_T_marker[:3, :3] = rmat
                self.cam_T_marker[:3, -1:] = tvec
                print(f'detected maker id: {id}')

    def _psm_ee_pose(self, msg):
        self.psm_ee_pose = msg.pose
        self.base_T_ee = utils.convert_pose_rep('Pose', 'H', self.psm_ee_pose)
        self.ee_T_base = np.linalg.inv(self.base_T_ee)
    
    def calibrate(self):
        if self.cam_T_marker is not None:
            if self.id != 'world' and hasattr(self, 'ee_T_base'):
                self.compute_cam_T_base()
                self.broadcast_cam_T_base()
            else:
                self.broadcast_cam_T_world()

    def compute_cam_T_base(self):
        self.cam_T_base = self.cam_T_marker @ self.m_T_ring @ self.ring_T_ee @ self.ee_T_base

    def broadcast_cam_T_world(self):
        cam_T_world = self.cam_T_marker
        t = cam_T_world[:3, -1]
        R = cam_T_world[:3, :3]
        q = tf3d.quaternions.mat2quat(R)

        c_to_w = TransformStamped(transform=Transform(translation=Vector3(x=t[0], y=t[1], z=t[2]),
                                                    rotation=Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])))
        c_to_w.header.stamp = self.get_clock().now().to_msg()
        c_to_w.header.frame_id = "/dvrk_cam"
        c_to_w.child_frame_id = "world"
        self.broadcaster.sendTransform(c_to_w)
        np.savez(os.path.join(self.cwd, f'../assets/cam_pose_world.npz'), pos=t, quat=q)


    def broadcast_cam_T_base(self):
        t = self.cam_T_base[:3, -1]
        R = self.cam_T_base[:3, :3]
        q = tf3d.quaternions.mat2quat(R)

        c_to_b = TransformStamped(transform=Transform(translation=Vector3(x=t[0], y=t[1], z=t[2]),
                                                    rotation=Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])))
        c_to_b.header.stamp = self.get_clock().now().to_msg()
        c_to_b.header.frame_id = "/dvrk_cam"
        c_to_b.child_frame_id = f"/PSM{self.id}_base"
        self.broadcaster.sendTransform(c_to_b)
        np.savez(os.path.join(self.cwd, f'../assets/cam_pose_psm{self.id}.npz'), pos=t, quat=q)
    
        t = self.cam_T_marker[:3, -1]
        R = self.cam_T_marker[:3, :3]
        q = tf3d.quaternions.mat2quat(R)

        c_to_m = TransformStamped(transform=Transform(translation=Vector3(x=t[0], y=t[1], z=t[2]),
                                                    rotation=Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])))
        c_to_m.header.stamp = self.get_clock().now().to_msg()
        c_to_m.header.frame_id = "/dvrk_cam"
        c_to_m.child_frame_id = "/mounted_marker"
        self.broadcaster.sendTransform(c_to_m)

if __name__ == '__main__':
    rclpy.init()
    marker_ctb_calibration = MarkerCalibration(id='1')
    rclpy.spin(marker_ctb_calibration)
    marker_ctb_calibration.destroy_node()
    rclpy.shutdown()

