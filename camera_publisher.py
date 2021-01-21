#!/usr/bin/env python3
PKG = 'tfg'
import roslib; roslib.load_manifest(PKG)
import rosbag

import rospy
from rospy.numpy_msg import numpy_msg

from sensor_msgs.msg import Image
import os
import cv2
from cv_bridge import CvBridge, CvBridgeError
from utilities import show_images

def get_images():
    vidcap = cv2.VideoCapture('2018-03-08-14-30-07_Dataset_year_-A0.h264')
    success, image = vidcap.read()
    while success:
        images.append(image)
        success, image = vidcap.read()

    # TODO: maybe saving GIGABYTES OF FRAMES IN RAM is not a great idea
    # images = []
    # bag = rosbag.Bag('2018-03-08-14-30-07_Dataset_year_.bag', "r")
    # for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
    #     img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    #     images.append(img)

    # bag.close()
    print('Done!')
    # return images

def talker():
    bridge = CvBridge()
    pub = rospy.Publisher('stream', Image, queue_size=10)
    rospy.init_node('talker',anonymous=True)
    r = rospy.Rate(10) # 10hz
    
    vidcap = cv2.VideoCapture('2018-03-08-14-30-07_Dataset_year_-A0.h264')
    success, image = vidcap.read()
    while not rospy.is_shutdown() and success:
        try:
            pub.publish(bridge.cv2_to_imgmsg(image))
            success, image = vidcap.read()
        except CvBridgeError as e:
            print(e) 
        r.sleep()

if __name__ == '__main__':
    talker()
