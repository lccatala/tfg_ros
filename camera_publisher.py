#!/usr/bin/env python3
PKG = 'tfg'
import roslib; roslib.load_manifest(PKG)
#import rosbag

import numpy as np
import rospy
from rospy.numpy_msg import numpy_msg

from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
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
    #bridge = CvBridge()
    pub = rospy.Publisher('stream', CompressedImage, queue_size=10)
    rospy.init_node('talker',anonymous=True)
    r = rospy.Rate(30) # 30hz
    
    vidcap = cv2.VideoCapture('2018-03-08-14-30-07_Dataset_year_-A0.h264')
    success, image = vidcap.read()
    msg = CompressedImage()
    msg.format = "jpeg"
    image_index = 0
    while not rospy.is_shutdown() and success:
        try:
            #pub.publish(bridge.cv2_to_imgmsg(image))
            image_index += 1
            if image_index % 10 == 0:
                msg.header.stamp = rospy.Time.now()
                msg.data = np.array(cv2.imencode('.jpg', image)[1]).tostring()
                pub.publish(msg)
                image_index = 0
            success, image = vidcap.read()
        except CvBridgeError as e:
            print(e) 
        r.sleep()

if __name__ == '__main__':
    talker()
