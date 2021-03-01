#!/usr/bin/env python3
PKG = 'tfg'
import roslib; roslib.load_manifest(PKG)

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
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

import time

# model = None # Model for performing inference
num_classes = None
device = None # Torch device
palette = None # Color palette
bridge = None

# Utility functions
to_tensor = None 
normalize = None

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def callback(data):
    np_array = np.fromstring(data.data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    #image = bridge.imgmsg_to_cv2(data)
    image = resize_image(image, 25)
   

    start = time.monotonic()
    segmented_image = inference_segment_image(image, 'multiscale')
    #end_segmentation = time.monotonic()
    segmented_image = segmented_image.convert(palette=CityScpates_palette)
    #end_conversion = time.monotonic()
    segmented_image = np.array(segmented_image)

    #print("conversion:", end_conversion - end_segmentation)
    #print("np.array:", end_nparray - end_conversion)
    #print()

    
    #segmented_image = resize_image(segmented_image, 125)
    #image = resize_image(image, 125)

    end = time.monotonic()

    print("segmentation time:", end - start)
    images = np.concatenate((image, segmented_image), axis=1)
    cv2.imshow('Live segmentation', images)

    cv2.waitKey(1)

def listener():
    rospy.init_node('listener')
    rospy.Subscriber("stream", CompressedImage, callback, queue_size=1)
    rospy.spin()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    print(sys.version)
    print('Initializing model...')
    model_path = os.path.join('pytorch_segmentation', 'best_model.pth')
    inference_init(model_path)
    #bridge = CvBridge()
    plt.ion()
    print('Model initialized!')
    listener()
