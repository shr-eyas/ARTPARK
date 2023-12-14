#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge # cv_bridge converts between ROS 2 image messages and OpenCV image representations

class ImagePublisher(Node):

    def __init__(self):
        super().__init__('camera_pub')
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.cap = cv2.VideoCapture(0)
        self.cv_bridge = CvBridge()

    def timer_callback(self):
        ret, frame = self.cap.read()

        if ret == True:
            self.publisher_.publish(
                self.cv_bridge.cv2_to_imgmsg(frame, 'bgr8')
            )
        
        self.get_logger().info('Publishing image frame')


def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()