#!/usr/bin/env python3
PKG = 'tfg'
import roslib; roslib.load_manifest(PKG)
import rospy
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import cv2
import numpy as np
import os
import message_filters
import pypcd

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
        self.mobile_classes = [11, 12, 13, 14, 15, 16, 17, 18, 18]
        self.points = []
        self.counts = []

        rospy.init_node('point_cloud_listener')
        self.pub = rospy.Publisher('final_cloud_publisher', PointCloud2, queue_size=1)
        odom_sub = message_filters.Subscriber("/vn100/odometry", Odometry)
        image_sub = message_filters.Subscriber("/gmsl/A0/image_color", Image)
        pc_sub = message_filters.Subscriber("/gmsl/A0/image_color/pixel_pointcloud", PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, pc_sub, odom_sub], 10, 0.1)
        ts.registerCallback(self.all_callback)
        rospy.spin()

    def generate_point_cloud(self):
        print('Publishing point cloud')
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "velodyne_front_link"
        new_point_cloud = point_cloud2.create_cloud_xyz32(header, self.points)

        print(len(point_cloud2.read_points_list(new_point_cloud, skip_nans=True)))

        self.pub.publish(new_point_cloud)
        rospy.sleep(2.0)
        print('published')

    def get_indices_from_segmented_image(self, image):
        segmented_image = inference_segment_image(image, 'multiscale')
        segmented_image = np.array(segmented_image)
        indices = np.isin(segmented_image, self.mobile_classes)
        return indices

    def init_image_inference(self):
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        print('Initializing model...')
        model_path = os.path.join('pytorch_segmentation', 'best_model.pth')
        inference_init(model_path)
        print('Model initialized!')

    def transform_pc(self, pc, odom):
        trans = TransformStamped()
        trans.header = odom.header
        trans.child_frame_id = odom.child_frame_id
        trans.transform.translation.x = odom.pose.pose.position.x
        trans.transform.translation.y = odom.pose.pose.position.y
        trans.transform.translation.z = odom.pose.pose.position.z
        trans.transform.rotation = odom.pose.pose.orientation

        transformed_cloud = do_transform_cloud(pc, trans)
        return transformed_cloud

    def all_callback(self, image, pc, odom):
        image = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        indices = self.get_indices_from_segmented_image(image)
        pc = self.transform_pc(pc, odom)
        
        # count = 0
        for point in point_cloud2.read_points(pc, skip_nans=True):
            image_x = point[5]
            image_y = point[6]
            if not indices[round(image_y-1)][round(image_x-1)]:
                self.points.append((point[0], point[1], point[2]))
                # count += 1

        # print(self.points[-1])
        # self.counts.append(count)
        self.generate_point_cloud()

        #print("({0},{1},{2})".format(pos.x, pos.y, pos.z))
                


if __name__ == '__main__':
    listener = Listener()
    atexit.register(listener.generate_point_cloud)
