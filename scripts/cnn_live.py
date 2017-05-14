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
        save = rospy.get_param("~save_images","")
        im_topic = rospy.get_param("~image","/head_xtion/rgb/image_raw")		# subscribed to image topic
        dp_topic = rospy.get_param("~depth","/head_xtion/depth_registered/sw_registered/image_rect")	# subscribed to depth topic
        sk_topic = rospy.get_param("~skeleton","/skeleton_data/incremental")		# subscribed to openni skeleton topic
        self.sk_cnn = cnn_live_functions.skeleton_cnn(cam, im_topic, dp_topic, sk_topic, pub, save)
        counter = 0
        r = rospy.Rate(15) # 30hz
        while not rospy.is_shutdown():
            imgs, depths, img_xys, userIDs = [],[],[],[]					# inputs to parallel processing cnn
            if self.sk_cnn.image_ready and self.sk_cnn.depth_ready and self.sk_cnn.openni_ready:
                users = self.sk_cnn.openni_data.keys()
                for userID in users:
                    #if userID in self.sk_cnn.openni_to_delete and userID in self.sk_cnn.openni_data:
                        self.sk_cnn.openni_data[userID]["cnt"] += 1			# counter to make sure this user is still there
                        if self.sk_cnn.openni_data[userID]["cnt"] == 30:			# no update to this ID for two seconds
                            self.sk_cnn.openni_data[userID]["msg"] = "Stopped tracking"
                        if self.sk_cnn.openni_data[userID]["msg"] not in ["Out of Scene","Stopped tracking"]:
                            if "img_xy" in self.sk_cnn.openni_data[userID].keys():
                                counter = 0
                                imgs.append(self.sk_cnn.openni_data[userID]["process_img"])
                                depths.append(self.sk_cnn.openni_data[userID]["process_depth"])
                                img_xys.append(self.sk_cnn.openni_data[userID]["img_xy"])
                                userIDs.append(userID)
            
            if imgs != []:									# any data for processing?
                self.sk_cnn._process_images(imgs, depths, img_xys, userIDs)			# process data
                self.sk_cnn._publish()								# publish results
            else:
                if self.sk_cnn.image_ready and counter>45:
                    self.sk_cnn._publish()
                counter += 1
                r.sleep()

def main():
    rospy.init_node('cnn_live')
    print "initialising CNN"
    cnn = live_cnn()

if __name__ == '__main__':
    main()
    

