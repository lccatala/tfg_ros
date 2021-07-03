#!/usr/bin/env python3
PKG = 'tfg'
import ctypes
import roslib; roslib.load_manifest(PKG)
import rospy
import open3d as o3d
import trimesh
import struct
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs import point_cloud2
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from tf.transformations import euler_from_quaternion
import numpy as np
import os
import message_filters

from pytorch_segmentation.inference import inference_init
from pytorch_segmentation.inference import inference_segment_image
from pytorch_segmentation.utils.helpers import show_images
from pytorch_segmentation.utils.palette import CityScpates_palette

import atexit
import matplotlib.pyplot as plt

num_classes = None
device = None # Torch device
palette = None # Color palette
bridge = None

# Utility functions
to_tensor = None
normalize = None

counts = []

class Listener:
    def __init__(self):
        self.init_image_inference()
        self.bridge = CvBridge()
        self.mobile_classes = [10, 11, 12, 13, 14, 15, 16, 17, 18]
        self.class_colors = [0xA52A2A, # Road: dark red
                             0xFFC0CB, # Sidewalk: pink
                             0xFF7F50, # Building: orange
                             0xCFC87C, # Wall: light-yellow
                             0xCC6F4E, # Fence: light-brown
                             0x475470, # Pole: blue-grey
                             0xFF0000, # Traffic light: red
                             0xCFD600, # Traffic sign: yellow
                             0x00FF00, # Vegetation: green
                             0x428000  # Terrain: dark green
                            ]
        self.points = []
        self.counts = []
        self.fields = [PointField('x',    0, PointField.FLOAT32, 1),
                       PointField('y',    4, PointField.FLOAT32, 1),
                       PointField('z',    8, PointField.FLOAT32, 1),
                       PointField('rgb', 12, PointField.UINT32, 1)]

        self.header = Header()
        self.header.frame_id = "gmsl_centre_link"

        self.current_point_cloud = None
        rospy.init_node('point_cloud_listener')
        self.pub = rospy.Publisher('final_cloud_publisher', PointCloud2, queue_size=1)
        odom_sub = message_filters.Subscriber("/vn100/odometry", Odometry)
        image_sub0 = message_filters.Subscriber("/gmsl/A0/image_color", Image)
        image_sub1 = message_filters.Subscriber("/gmsl/A1/image_color", Image)
        image_sub2 = message_filters.Subscriber("/gmsl/A2/image_color", Image)
        pc_sub0 = message_filters.Subscriber("/gmsl/A0/image_color/pixel_pointcloud", PointCloud2)
        pc_sub1 = message_filters.Subscriber("/gmsl/A1/image_color/pixel_pointcloud", PointCloud2)
        pc_sub2 = message_filters.Subscriber("/gmsl/A2/image_color/pixel_pointcloud", PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer([image_sub0, image_sub1, image_sub2, 
                                                          pc_sub0, pc_sub1, pc_sub2, 
                                                          odom_sub], 10, 0.1)
        ts.registerCallback(self.all_callback)
        rospy.spin()

    def update_point_cloud(self):
        self.header.stamp = rospy.Time.now()
        self.current_point_cloud = point_cloud2.create_cloud(self.header, self.fields, self.points)

    def publish_point_cloud(self):
        self.pub.publish(self.current_point_cloud)

    def get_correct_indices_and_segmented_image(self, image):
        segmented_image = inference_segment_image(image, 'multiscale')
        segmented_image = np.array(segmented_image)
        indices = np.isin(segmented_image, self.mobile_classes)
        return indices, segmented_image

    def init_image_inference(self):
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        print('Initializing model...')
        model_path = os.path.join('pytorch_segmentation', 'best_model.pth')
        inference_init(model_path)
        print('Model initialized!')

    def transform_pc(self, pc, odom):
        trans = TransformStamped()
        trans.header = odom.header
        trans.transform.translation = odom.pose.pose.position
        trans.transform.rotation    = odom.pose.pose.orientation
        
        # trans.transform.rotation.w = odom.pose.pose.orientation.w # Do not change, probably
        # trans.transform.rotation.x = odom.pose.pose.orientation.x - (np.pi/4) - 0.122# More or less correct i guess
        # trans.transform.rotation.y = odom.pose.pose.orientation.y + (np.pi/2)
        # trans.transform.rotation.z = odom.pose.pose.orientation.z - (np.pi/2) # Correct I guess

        transformed_cloud = do_transform_cloud(pc, trans)
        return transformed_cloud

    def paint_point(self, point, point_class):
        color = self.class_colors[point_class]
        return [point[0], point[1], point[2], color]

    def process_image_and_pointcloud(self, image, pc, odom):
        image = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        indices, segmented_image = self.get_correct_indices_and_segmented_image(image)
        pc = self.transform_pc(pc, odom) #TODO re-enable

        for point in point_cloud2.read_points(pc, skip_nans=True):
            image_x = round(point[5]-1)
            image_y = round(point[6]-1)
            is_mobile = indices[image_y][image_x]
            if not is_mobile:
                point_class = segmented_image[image_y][image_x]
                point = self.paint_point(point, point_class)
                self.points.append(point)


    def all_callback(self, image0, image1, image2, pc0, pc1, pc2, odom):
        self.process_image_and_pointcloud(image0, pc0, odom)
        self.process_image_and_pointcloud(image1, pc1, odom)
        self.process_image_and_pointcloud(image2, pc2, odom)
        # TODO: rotate odometry for each side image
        
        self.update_point_cloud()
        self.publish_point_cloud()


if __name__ == '__main__':
    listener = Listener()
    # atexit.register(listener.save_point_cloud)
