#!/usr/bin/env python3
PKG = 'tfg'
from ctypes import sizeof
from numpy.core.fromnumeric import var
from numpy.lib.arraysetops import isin
import roslib; roslib.load_manifest(PKG)
import rospy
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2
import sensor_msgs.point_cloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
import cv2
import numpy as np
import ros_numpy
import sys
import os
import time
import message_filters

from pytorch_segmentation.inference import inference_init
from pytorch_segmentation.inference import inference_segment_image
from pytorch_segmentation.utils.helpers import show_images
from pytorch_segmentation.utils.palette import CityScpates_palette

num_classes = None
device = None # Torch device
palette = None # Color palette
bridge = None

# Utility functions
to_tensor = None
normalize = None

class Listener:
    def __init__(self):
        self.init_image_inference()
        self.bridge = CvBridge()
        self.mobile_classes = [11, 12, 13, 14, 15, 16, 17, 18, 18]
        self.points = []

        rospy.init_node('point_cloud_listener')
        odom_sub = message_filters.Subscriber("/vn100/odometry", Odometry)
        image_sub = message_filters.Subscriber("/gmsl/A0/image_color", Image)
        pc_sub = message_filters.Subscriber("/gmsl/A0/image_color/pixel_pointcloud", PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, pc_sub, odom_sub], 10, 0.1)
        ts.registerCallback(self.all_callback)
        rospy.spin()

    def get_indices_from_segment_image(self, image):
        segmented_image = inference_segment_image(image, 'multiscale')
        # segmented_image = segmented_image.convert(palette=CityScpates_palette)
        segmented_image = np.array(segmented_image)

        # Paint all pixels from mobile classes white
        indices = np.isin(segmented_image, self.mobile_classes)
        segmented_image[indices] = 255
        return indices

    def init_image_inference(self):
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        print('Initializing model...')
        model_path = os.path.join('pytorch_segmentation', 'best_model.pth')
        inference_init(model_path)
        print('Model initialized!')

    def all_callback(self, image, pc, odom):
        image = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        pos = odom.pose.pose.position
        indices = self.get_indices_from_segment_image(image)
        count = 0
        for point in point_cloud2.read_points(pc, skip_nans=True):
            image_x = point[5]
            image_y = point[6]
            if not indices[round(image_y-1)][round(image_x-1)]:
                count += 1
                self.points.append((point[0], point[1], point[2]))

        print(count)

        #print("({0},{1},{2})".format(pos.x, pos.y, pos.z))
        
        # Create new point cloud
        # header = Header()
        # header.stamp = rospy.Time.now()
        # header.frame_id = "velodyne"
        # new_point_cloud = point_cloud2.create_cloud_xyz32(header, self.points)
        

if __name__ == '__main__':
    listener = Listener()
