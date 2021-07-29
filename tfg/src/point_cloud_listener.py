#!/usr/bin/env python3
PKG = 'tfg'
import roslib; roslib.load_manifest(PKG)
import rospy
import ctypes
import open3d as o3d
import trimesh
import struct
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs import point_cloud2
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import numpy as np
import os
import message_filters
import open3d as o3d

from pytorch_segmentation.inference import inference_init
from pytorch_segmentation.inference import inference_segment_image

import atexit

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

        # ID for each mobile class (pedestrian, car...)
        self.mobile_classes = [10, 11, 12, 13, 14, 15, 16, 17, 18]

        # Colors to differenciate 3D points of each class
        self.class_colors = [0xFFFFFF,#0xA52A2A, # Road: dark red
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

        # Data fields for the published pointcloud
        self.fields = [PointField('x',    0, PointField.FLOAT32, 1),
                       PointField('y',    4, PointField.FLOAT32, 1),
                       PointField('z',    8, PointField.FLOAT32, 1),
                       PointField('rgb', 12, PointField.UINT32, 1)]
        self.header = Header()
        self.header.frame_id = "base_link"

        # For tilt correction in each generated cloud
        self.forward_tilt = 0.13

        # For filtering out large and sudden changes in rotation
        self.rotation_threshold = 1.0
        self.last_rotation_x = 0.0
        self.last_rotation_y = 0.0
        self.last_rotation_z = 0.0

        self.current_points = []
        self.combined_points = []
        self.current_pc = None
        self.combined_pc = None
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
                                                          odom_sub], 10, 0.05)
        ts.registerCallback(self.all_callback)
        rospy.spin()

    def __get_correct_indices_and_segmented_image(self, image):
        """
        Segments the input image and returns:
        - A numpy array of all indices corresponding to a mobile class
        - The segmented image
        """
        segmented_image = inference_segment_image(image, 'multiscale')
        segmented_image = np.array(segmented_image)
        indices = np.isin(segmented_image, self.mobile_classes)
        return indices, segmented_image

    def init_image_inference(self):
        """
        Initalize image inference model
        """
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        print('Initializing model...')
        model_filename = 'best_model.pth'
        inference_init(model_filename)
        print('Model initialized!')

    def __transform_pc(self, pc, odom):
        """
        Change position and rotation of a given point cloud (pc)
        to those of the odometry (odom)
        """
        trans = TransformStamped()
        trans.child_frame_id = 'gmsl_centre_link'
        trans.transform.translation = odom.pose.pose.position
        trans.transform.rotation = odom.pose.pose.orientation

        transformed_cloud = do_transform_cloud(pc, trans)
        transformed_cloud.header.frame_id = 'gmsl_centre_link'
        return transformed_cloud

    def __reformat_point(self, point, point_class):
        """
        - Change color of point according to it's class
        - Swap point axis values for correct positioning
        """
        color = self.class_colors[point_class]
        return [point[2], -point[0], -point[1], color]

    def _process_image_and_pointcloud(self, image, pc, odom):
        """
        Handle an image and it's corresponding projected point cloud
        Should be used also for side images and clouds
        """
        image = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        indices, segmented_image = self.__get_correct_indices_and_segmented_image(image)

        self.current_points.clear()
        for point in point_cloud2.read_points(pc, skip_nans=True):
            image_x = round(point[5]-1)
            image_y = round(point[6]-1)
            is_mobile = indices[image_y][image_x]
            if not is_mobile:
                point_class = segmented_image[image_y][image_x]
                point = self.__reformat_point(point, point_class)
                self.current_points.append(point)

        # Create new cloud with current points
        self.header.stamp = rospy.Time.now()
        self.current_pc = point_cloud2.create_cloud(self.header, self.fields, self.current_points)
        self.current_pc.header.frame_id = 'gmsl_centre_link'

        self.current_pc = self.__transform_pc(self.current_pc, odom)

        # Add points from new cloud to final points
        self.combined_points.extend(point_cloud2.read_points_list(self.current_pc))

    def rotations_below_threshold(self, odom):
        """
        Returns False if the current rotation of odometry hasn't changed
        too abruptly from the last received one
        """
        delta_rotation_x = abs(self.last_rotation_x - odom.pose.pose.orientation.x)
        delta_rotation_y = abs(self.last_rotation_y - odom.pose.pose.orientation.y)
        delta_rotation_z = abs(self.last_rotation_z - odom.pose.pose.orientation.z)

        if delta_rotation_x >= self.rotation_threshold or delta_rotation_y >= self.rotation_threshold or delta_rotation_z >= self.rotation_threshold:
            return False
        self.last_rotation_x = odom.pose.pose.orientation.x
        self.last_rotation_y = odom.pose.pose.orientation.y
        self.last_rotation_z = odom.pose.pose.orientation.z
        return True

    def all_callback(self, image0, image1, image2, pc0, pc1, pc2, odom):
        """
        Receives 3 simultaneous images and their corresponding point clouds
        Only handles one of them
        """

        # Filter rotation outliers
        if (not self.rotations_below_threshold(odom)):
            return

        # Prepare odometry for and segment central image
        odom.pose.pose.orientation.y += self.forward_tilt
        self._process_image_and_pointcloud(image0, pc0, odom)

        # Create combined cloud
        self.header.stamp = rospy.Time.now()
        self.combined_pc = point_cloud2.create_cloud(self.header, self.fields, self.combined_points)
        self.combined_pc.header.frame_id = 'gmsl_centre_link'

        # Publish combined cloud
        self.pub.publish(self.combined_pc)


    def save_point_cloud(self):
        """
        Runs once the node is attempted to shut down (ctrl+c)
        Saves the combined point cloud to cloud_color.ply
        """

        filename = 'cloud_color.ply'
        print('Saving cloud to ', filename)
        pcd = o3d.geometry.PointCloud()
        print('Creating points...')
        xyz = []
        rgb = []
        for p in self.combined_points:
            xyz.append([p[0], p[1], p[2]])

            r = (p[3] & 0x00FF0000) #>> 16
            g = (p[3] & 0x0000FF00) #>> 8
            b = (p[3] & 0x000000FF)
            rgb.append([r, g, b])
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.io.write_point_cloud("/home/alpasfly/" + filename, pcd)
        print('cloud saved')


if __name__ == '__main__':
    listener = Listener()
    atexit.register(listener.save_point_cloud)
