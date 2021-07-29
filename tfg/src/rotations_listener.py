#!/usr/bin/env python3
PKG = 'tfg'
import roslib; roslib.load_manifest(PKG)
import rospy
from nav_msgs.msg import Odometry
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import numpy as np
import matplotlib.pyplot as plt

import atexit

num_classes = None
device = None # Torch device
palette = None # Color palette
bridge = None

class Listener:
    def __init__(self):
        self.max_rot_x = 0.0
        self.max_rot_y = 0.0
        self.max_rot_z = 0.0

        self.last_rot_x = 0.0
        self.last_rot_y = 0.0
        self.last_rot_z = 0.0
        self.rotations_x = []
        self.rotations_y = []
        self.rotations_z = []
        rospy.init_node('rotations_listener_listener')
        rospy.Subscriber("/vn100/odometry", Odometry, self.callback, queue_size=1)
        rospy.spin()


    def callback(self, odom):
        delta_x = abs(odom.pose.pose.orientation.x - self.last_rot_x)
        delta_y = abs(odom.pose.pose.orientation.y - self.last_rot_y)
        delta_z = abs(odom.pose.pose.orientation.z - self.last_rot_z)

        self.rotations_x.append(delta_x)
        self.rotations_y.append(delta_y)
        self.rotations_z.append(delta_z)

        self.last_rot_x = odom.pose.pose.orientation.x
        self.last_rot_y = odom.pose.pose.orientation.y
        self.last_rot_z = odom.pose.pose.orientation.z

        self.max_rot_x = max(delta_x, self.max_rot_x)
        self.max_rot_y = max(delta_y, self.max_rot_y)
        self.max_rot_z = max(delta_z, self.max_rot_z)

    def save_plot(self):
        plt.plot(self.rotations_x, label='x')
        plt.plot(self.rotations_y, label='y')
        plt.plot(self.rotations_z, label='z')
        plt.legend(loc="upper left")
        plt.savefig('rotations.png')
        plt.show()
        print('max x:', self.max_rot_x)
        print('max y:', self.max_rot_y)
        print('max z:', self.max_rot_z)



if __name__ == '__main__':
    listener = Listener()
    atexit.register(listener.save_plot)
