#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Vector3
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import numpy as np

class ArucoDetect(Node):

    def __init__(self, dictionary_type, marker_length_m):
        super().__init__('detect')
        self.cv_bridge = CvBridge()
        self.image = None
        self.camera_info = None
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.marker_length_m = marker_length_m

        # Create subscribers for image and camera info
        self.image_raw_subscriber_ = self.create_subscription(Image,'/camera/color/image_raw', self.image_callback, 10)
        self.camera_info_subscriber_ = self.create_subscription(CameraInfo,'/camera/color/camera_info', self.camera_info_callback, 10)
    
        # Create a publisher for Aruco marker ID and pose
        self.rvecs_publisher_ = self.create_publisher(Vector3, 'aruco_info/rvecs', 10)
        self.tvecs_publisher_ = self.create_publisher(Vector3, 'aruco_info/tvecs', 10)
        self.marker_id_publisher_ = self.create_publisher(Int32, 'aruco_info/marker_id', 10)

    def image_callback(self, data):
        self.image = data
        self.aruco_detect()

    def camera_info_callback(self, data):
        self.camera_info = data
        self.aruco_detect()

    def aruco_detect(self):

        if self.image is None:
            self.get_logger().warn('No image')
            return

        image = self.cv_bridge.imgmsg_to_cv2(self.image, 'bgr8')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        cv2.imshow("output", image)
        cv2.waitKey(10)

        if ids is None or ids.size == 0:
            self.get_logger().warn('No ArUco markers detected in the frame')
            return

        k = self.camera_info.k
        d = self.camera_info.d
        self.distortion_coefficients = np.array(d, dtype=np.float32).reshape(1, 5)
        self.camera_matrix = np.array(k, dtype=np.float32).reshape(3, 3)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length_m, self.camera_matrix,
                                                            self.distortion_coefficients)

        ids_list = ids.flatten().tolist()

        for i in range(len(ids_list)):
            # Publish rvecs for each marker
            rvecs_msg = Vector3()
            rvecs_msg.x = float(rvecs[i][0][0])  # Extract the first element of the first dimension
            rvecs_msg.y = float(rvecs[i][0][1])  # Extract the second element of the first dimension
            rvecs_msg.z = float(rvecs[i][0][2])  # Extract the third element of the first dimension
            self.rvecs_publisher_.publish(rvecs_msg)

            # Publish tvecs for each marker
            tvecs_msg = Vector3()
            tvecs_msg.x = float(tvecs[i][0][0])
            tvecs_msg.y = float(tvecs[i][0][1])
            tvecs_msg.z = float(tvecs[i][0][2])
            self.tvecs_publisher_.publish(tvecs_msg)

            # Publish marker ID for each marker
            marker_id_msg = Int32()
            marker_id_msg.data = ids_list[i]
            self.marker_id_publisher_.publish(marker_id_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetect(dictionary_type=cv2.aruco.DICT_6X6_250, marker_length_m=0.1)
    rclpy.spin(node)
    rclpy.shutdown()