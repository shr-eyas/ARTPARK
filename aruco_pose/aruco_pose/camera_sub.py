#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image      
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageSubsriber(Node):
    
    def __init__(self, dictionary_type):
        super().__init__('camera_sub')
        self.sub_ = self.create_subscription(
            Image, 'image_raw', self.listener_callback, 10)
        self.cv_bridge = CvBridge()
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.parameters = cv2.aruco.DetectorParameters_create()

    def listener_callback(self, data):
        self.get_logger().info('Recieving video frame')
        image = self.cv_bridge.imgmsg_to_cv2(data, 'bgr8')
        self.aruco_detect(image)

    def aruco_detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        cv2.imshow("output", image)                             
        cv2.waitKey(10)


def main(args=None):
    rclpy.init(args=args)
    dictionary_type = cv2.aruco.DICT_6X6_250
    node = ImageSubsriber(dictionary_type)
    rclpy.spin(node)
    node.destroy_node()                                   
    rclpy.shutdown



