#!/usr/bin/env python

# __author__: Muhannad Alomari
# __author__: Umer Rafi
# __email__:  scmara@leeds.ac.uk
# __email__:  rafi@vision.rwth-aachen.de

import cv2
import rospy
import numpy as np
#import tensorflow as tf
import cnn_live_functions

class live_cnn():
    def __init__(self):
        cam = rospy.get_param("~camera_calibration","")
        pub = rospy.get_param("~publish_images","True")
        sav = rospy.get_param("~save_images","")
        topic = rospy.get_param("~image","/head_xtion/rgb/image_raw")
        self.sk_cnn = cnn_live_functions.skeleton_cnn(cam,topic,pub,sav)
        r = rospy.Rate(30) # 30hz
        while not rospy.is_shutdown():
            if self.sk_cnn.image_ready:
                print "image ready"
                #self.sk_cnn._process_images(self.sk_cnn.image)
            else:
                r.sleep()


def main():
    rospy.init_node('cnn_live')
    print "initialising CNN"
    cnn = live_cnn()

if __name__ == '__main__':
    main()
    

