#!/usr/bin/env python3
PKG = 'tfg'
import roslib; roslib.load_manifest(PKG)

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
import sys
import os
from cv_bridge import CvBridge, CvBridgeError
from pytorch_segmentation.inference import inference_init
from pytorch_segmentation.inference import inference_segment_image
from pytorch_segmentation.utils.helpers import show_images
from pytorch_segmentation.utils.palette import CityScpates_palette

import matplotlib.pyplot as plt
import PIL

# model = None # Model for performing inference
num_classes = None
device = None # Torch device
palette = None # Color palette
bridge = None

# Utility functions
to_tensor = None 
normalize = None

def callback(data):
    image = bridge.imgmsg_to_cv2(data)

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image.astype(np.float32)
    # image /= 255

    segmented_image = inference_segment_image(image, 'multiscale')
    segmented_image = segmented_image.convert(palette=CityScpates_palette)
    segmented_image = np.array(segmented_image)
    cv2.imshow('Live', segmented_image)
    cv2.waitKey(20)

    # TODO: this is temporal code
    # images = {
    #    0 : ['Input Image', image],
    #    1 : ['Segmented Image', segmented_image],
    #    2 : ['np Segmented Image', np_segmented_image],
    # }
    # show_images(images)


    # plt.imshow(segmented_image)
    # plt.draw()
    # plt.pause(2)
    

def listener():
    rospy.init_node('listener')
    rospy.Subscriber("stream", Image, callback)
    rospy.spin()

if __name__ == '__main__':
    print('Initializing model...')
    model_path = os.path.join('pytorch_segmentation', 'best_model.pth')
    inference_init(model_path)
    bridge = CvBridge()
    plt.ion()
    print('Model initialized!')
    listener()
