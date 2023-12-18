#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import Vector3
from std_msgs.msg import Int32
from std_msgs.msg import Float64
import numpy as np
import cv2
import tf2_ros
from geometry_msgs.msg import TransformStamped

class ArucoDetect(Node):

    def __init__(self, dictionary_type, marker_length_m):
        super().__init__('data')
        self.image = None
        self.camera_info = None
        self.cv_bridge = CvBridge()
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.marker_length_m = marker_length_m

        self.image_raw_subscriber_ = self.create_subscription(Image,'/camera/color/image_raw', self.image_callback, 10)
        self.camera_info_subscriber_ = self.create_subscription(CameraInfo,'/camera/color/camera_info', self.camera_info_callback, 10)
        self.detected_markers = []

        self.ID_publisher_ = self.create_publisher(Int32, 'aruco_info/ID', 10)
        self.roll_publisher_ = self.create_publisher(Float64, 'aruco_info/roll', 10)
        self.pitch_publisher_ = self.create_publisher(Float64, 'aruco_info/pitch', 10)
        self.yaw_publisher_ = self.create_publisher(Float64, 'aruco_info/yaw', 10)
        self.distance_publisher_ = self.create_publisher(Float64, 'aruco_info/distance', 10)
        self.rvecs_publisher_ = self.create_publisher(Vector3, 'aruco_info/rvecs', 10)
        self.tvecs_publisher_ = self.create_publisher(Vector3, 'aruco_info/tvecs', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

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
        cv2.imshow('output', image)
        cv2.waitKey(10)

        k = self.camera_info.k

        if k is None:
            self.get_logger().warn('Camera matrix is None')
            return
        d = self.camera_info.d
        self.distortion_coefficients = np.array(d, dtype=np.float32).reshape(1, 5)
        self.camera_matrix = np.array(k, dtype=np.float32).reshape(3, 3)

        if ids is not None:
            self.detected_markers = []
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length_m, self.camera_matrix, self.distortion_coefficients)

            for i in range(len(ids)):
                ID_msg = Int32()
                ID_msg.data = int(ids[i][0])
                roll_msg = Float64()
                pitch_msg = Float64()
                yaw_msg = Float64()
                distance_msg = Float64()
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]
             
                rvecs_msg = Vector3()
                rvecs_msg.x = float(rvec[0])   
                rvecs_msg.y = float(rvec[1])  
                rvecs_msg.z = float(rvec[2])  
                self.rvecs_publisher_.publish(rvecs_msg)

                tvecs_msg = Vector3()
                tvecs_msg.x = float(tvec[0])
                tvecs_msg.y = float(tvec[1])
                tvecs_msg.z = float(tvec[2])
                self.tvecs_publisher_.publish(tvecs_msg)
                
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                roll_pitch_yaw = cv2.RQDecomp3x3(rotation_matrix)[0]

                roll_msg.data = float(np.degrees(roll_pitch_yaw[0])%360)
                pitch_msg.data = float(np.degrees(roll_pitch_yaw[1]))
                yaw_msg.data = float(np.degrees(roll_pitch_yaw[2]))
                distance_msg.data = float(np.linalg.norm(tvec))

                self.ID_publisher_.publish(ID_msg)
                self.roll_publisher_.publish(roll_msg)
                self.pitch_publisher_.publish(pitch_msg)
                self.yaw_publisher_.publish(yaw_msg)
                self.distance_publisher_.publish(distance_msg)

                transform_stamped = TransformStamped()
                transform_stamped.header.stamp = self.get_clock().now().to_msg()
                transform_stamped.header.frame_id = "camera_color_frame" 
                transform_stamped.child_frame_id = f"aruco_marker_{ids[i][0]}"
                transform_stamped.transform.translation.x = tvec[0]
                transform_stamped.transform.translation.y = tvec[1]
                transform_stamped.transform.translation.z = tvec[2]
                transform_stamped.transform.rotation.x = rvec[0]
                transform_stamped.transform.rotation.y = rvec[1]
                transform_stamped.transform.rotation.z = rvec[2]
                transform_stamped.transform.rotation.w = 1.0 

                self.tf_broadcaster.sendTransform(transform_stamped)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetect(dictionary_type=cv2.aruco.DICT_6X6_250, marker_length_m=0.05)
    rclpy.spin(node)
    rclpy.shutdown()
