# date    06-03-2017
# author  Muhannad Alomari
# email   scmara@leeds.ac.uk
# version 1.0
import rospy
import cv2
import numpy as np
import scipy
import math
#import caffe
import tensorflow as tf
from network import inception

import time
#from config_reader import config_reader
import util
import copy
import rospkg
import os
import glob
import getpass
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Header
from skeleton_tracker.msg import skeleton_message, joint_message, skeleton_tracker_state
from geometry_msgs.msg import Pose
#from cpm_skeleton.msg import cpm_pointer, cpmAction, cpmActionResult
import sys
import actionlib
import shutil
import skimage.io as io

class skeleton_cnn():
    """docstring for cnn"""
    def __init__(self, cam, im_topic, dp_topic, sk_topic, pub, save):
        
        # read camera calib
        self.camera_calib = util.read_yaml_calib(cam)

        # save cpm images
        # TO DO!
        self.save_cpm_img = save
        if self.save_cpm_img:
            rospy.loginfo("save cpm images.") 

        # get camera topic
        self.image_topic = rospy.resolve_name(im_topic)

        # get depth topic
        self.depth_topic = rospy.resolve_name(dp_topic)

        # get skeleton topic
        self.skeleton_topic = rospy.resolve_name(sk_topic)

        # initialize published
        self.pub = pub
        if self.pub:
            rospy.loginfo("publish cnn images")
            self.image_pub = rospy.Publisher("/cnn_skeleton_image", Image, queue_size=1)
            rospy.loginfo("publish cnn skeletons")
            self.skeleton_pub = rospy.Publisher("/skeleton_data/cnn", skeleton_message, queue_size=1)
        else:
            rospy.loginfo("don't publish cnn images")
            rospy.loginfo("don't publish cnn skeletons")

        # cnn init stuff
        self.bridge = CvBridge()
        self.rospack = rospkg.RosPack()
        self.cnn_path = self.rospack.get_path('cnn_skeleton')
        self.limbs_names = ['r_ankle', 'r_knee','r_hip', 'l_hip', 'l_knee', 'l_ankle', 'pelvis','thorax','upper_neck',
                                        'head_top','r_wrist', 'r_elbow','r_shoulder', 'l_shoulder','l_elbow','l_wrist']
        self.bone_con = [[9,8], [12,11], [11,10], [13,14], [14,15], [2,1], [1,0], [3,4], [4,5]]
        self.colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0],
        [255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255]] # note BGR ...
        self.stickwidth = 12
        self.dist_threshold = 1.5       # less than 1.5 meters ignore the skeleton
        self.depth_thresh = .35         # any more different in depth than this with openni, use openni
        self.finished_processing = 0    # a flag to indicate that we finished processing allavailable  data
        self.threshold = 10             # remove any folder <= 10 detections
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        self.sess = tf.Session()
        #self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.X = tf.placeholder(tf.float32, shape=None)
        self.logits = inception(self.X,False)
        self.preds = tf.nn.sigmoid(self.logits)
        self.saver = tf.train.Saver()
        self.teaser_batch = np.zeros([8, 256, 256,3], dtype = np.float32)
        self.img_small = np.zeros((256,256,3),dtype=np.uint8)
        self.conf_threshold = 0.5		# threshold for cnn detection
        self.conf_threshold2 = 0.1
        self._initiliase_cnn()
        self.processing = 0
        self.image_ready = 0
        self.depth_ready = 0
        self.openni_ready = 0
        self.openni_data = {}		# keeps track of openni_data
        self.openni_to_delete = {}

        # subscribe to camera topic
        rospy.Subscriber(self.image_topic, Image, self._get_rgb)

        # subscribe to depth topic
        rospy.Subscriber(self.depth_topic, Image, self._get_depth)

	# subscribe to openni state
        rospy.Subscriber("/skeleton_data/state", skeleton_tracker_state, self._get_openni_state)

        # subscribe to openni topic
        rospy.Subscriber(self.skeleton_topic, skeleton_message, self._get_openni)

    def _initiliase_cnn(self):
        rospack = rospkg.RosPack()
        path = rospack.get_path('cnn_skeleton')
        teaser_images_folder = path+'/teaser_images/'
        network_path = path + '/pose_model/pose_net.chkp'
        self.saver.restore(self.sess, network_path)
        print('Initializing and running on teaser batch')
        for t in range(1,7):
            path = teaser_images_folder + 'teaser' + str(t) + '.jpg'
            self.teaser_batch[t] = np.divide(np.array(io.imread(path)),255.0)
        start = time.time()
        #init = tf.global_variables_initializer()
        #self.sess.run(init)
        output = self.sess.run(self.preds, feed_dict={self.X: self.teaser_batch})
        print 'ini in: %3.3f sec' % (time.time()-start)
        print('initialization done')

    def _get_rgb(self,imgmsg):
        if not self.processing:
            img = self.bridge.imgmsg_to_cv2(imgmsg, desired_encoding="passthrough")
            img = img[:,:,0:3]
            #cv2.imshow('camera feed',img)
            #cv2.waitKey(1)
            self.image = img
            self.image_ready = 1

    def _get_depth(self,imgmsg):
        if not self.processing:
            self.depth = self.bridge.imgmsg_to_cv2(imgmsg, desired_encoding="passthrough")
            self.depth_ready = 1

    def _get_openni_state(self,msg):
        #print msg
        self.openni_to_delete[msg.userID] = msg.message
        print msg.userID,msg.message
        #if msg.message in ["Out of Scene","Stopped tracking"]:
        #    print 'yay'
        #    self.openni_to_delete.append(msg.userID)

    def _get_openni(self,msg):
        if not self.processing:
            [fx,fy,cx,cy] = self.camera_calib
            self.openni_data[msg.userID] = {}
            self.openni_data[msg.userID]["uuid"] = msg.uuid
            x_max = 0
            x_min = 1000
            y_max = 0
            y_min = 1000
            for j in msg.joints:
                pose = j.pose.position
                x2d = int(int(pose.x*fx/pose.z+cx))
                y2d = int(int(pose.y*fy/pose.z+cy))
                self.openni_data[msg.userID][j.name] = [x2d, y2d, pose.x, pose.y, pose.z]
                if x2d < x_min:		x_min=x2d
                if x2d > x_max:		x_max=x2d
                if y2d < y_min:		y_min=y2d
                if y2d > y_max:		y_max=y2d
            if self.image_ready and self.depth_ready:
                x_min = np.max([x_min-60,0])
                y_min = np.max([y_min-60,0])
                x_max = np.min([x_max+60,640])
                y_max = np.min([y_max+60,480])
                self.openni_data[msg.userID]["process_img"]   = self.image[y_min:y_max, x_min:x_max, :]
                self.openni_data[msg.userID]["process_depth"] = self.depth[y_min:y_max, x_min:x_max]
                self.openni_data[msg.userID]["img_xy"]        = [x_min, x_max, y_min, y_max]
                self.openni_ready = 1
            for ID in self.openni_to_delete:
                if self.openni_to_delete[ID] in ["Out of Scene","Stopped tracking"]:
                    self.openni_data.pop(ID, None)
            #self.openni_to_delete = []

    def _process_images(self, img, depth, img_xy, userID):
        self.processing = 1
        self.image_ready = 0
        self.depth_ready = 0
        self.openni_ready = 0

        # main loop
        start = time.time()

        self.img_small = cv2.resize(img, (256,256))#, fx=self.scale1, fy=self.scale2, interpolation=cv2.INTER_CUBIC)
        self.teaser_batch[0] = np.divide(self.img_small,255.0)
        output = self.sess.run(self.preds, feed_dict={self.X:self.teaser_batch})
        a,conf = self.get_skeleton(output[0])

        X = a[0]*img.shape[1]/256.0
        Y = a[1]*img.shape[0]/256.0
        X = map(int,X)
        Y = map(int,Y)

        # block 10
        canvas = img.copy()
        # check conf
        C_val = np.sum(conf)/float(len(conf))

        if C_val >= self.conf_threshold:
            canvas = np.multiply(canvas,0.2,casting="unsafe")
            cur_canvas = img.copy() #np.zeros(canvas.shape,dtype=np.uint8)
            for l,bone in enumerate(self.bone_con):
                a = bone[0]
                b = bone[1]
                if conf[a] >= self.conf_threshold2 and conf[b] >= self.conf_threshold2:
	            Yb = [X[a],X[b]]
	            Xb = [Y[a],Y[b]]
	            mX = np.mean(Xb)
	            mY = np.mean(Yb)
	            length = ((Xb[0] - Xb[1]) ** 2 + (Yb[0] - Yb[1]) ** 2) ** 0.5
	            angle = math.degrees(math.atan2(Xb[0] - Xb[1], Yb[0] - Yb[1]))
	            polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), self.stickwidth), int(angle), 0, 360, 1)
	            cv2.fillConvexPoly(cur_canvas, polygon, self.colors[l])
            canvas = np.add(canvas,np.multiply(cur_canvas,0.8,casting="unsafe"),casting="unsafe") # for transparency
            #for x,y,c in zip(X,Y,conf):
            #    if c >= self.conf_threshold:
            #        cv2.circle(canvas,(x,y), 5, (0,0,255), -1)

        canvas = canvas.astype(np.uint8)
        x_min, x_max, y_min, y_max = img_xy
        self.image.setflags(write=1)
        self.image[y_min:y_max, x_min:x_max, :] = canvas
        self.image.setflags(write=1)
        self.image[y_min:y_min+2, x_min:x_max, :] = self.colors[userID]
        self.image.setflags(write=1)
        self.image[y_max-2:y_max, x_min:x_max, :] = self.colors[userID]
        self.image.setflags(write=1)
        self.image[y_min:y_max, x_min:x_min+2, :] = self.colors[userID]
        self.image.setflags(write=1)
        self.image[y_min:y_max, x_max-2:x_max, :] = self.colors[userID]
        #print 'image processed in: %1.3f sec' % (time.time()-start)
        #util.showBGRimage(name+'_results',canvas,1)

    def _publish(self):
        if self.pub:
            sys.stdout = open(os.devnull, "w")
            msg = self.bridge.cv2_to_imgmsg(self.image, "rgb8")
            sys.stdout = sys.__stdout__
            self.image_pub.publish(msg)
        self.processing = 0

    def _get_depth_data(self, prediction, depthToTest, userID, img_xy, p):
        [fx,fy,cx,cy] = self.camera_calib
        x_min, x_max, y_min, y_max = img_xy
        
        # add the torso position
        #x2d = np.min([int(self.y[p]),367])
        #y2d = np.min([int(self.x[p]),490])     
        #z = depthToTest[x2d, y2d]
        #x = (y2d/self.scale-cx)*z/fx
        #y = (x2d/self.scale-cy)*z/fy
	# the rest of the body joints
        for part,jname in enumerate(self.limbs_names):
            x2d = np.min([int(prediction[part, 0, p]),367])
            y2d = np.min([int(prediction[part, 1, p]),490])
            z = depthToTest[x2d, y2d]
            if not np.abs(z-self.openni_data[userID][jname][4])<self.depth_thresh:
                z = self.openni_data[userID][jname][4]
            x2d += y_min
            y2d += x_min
            x = (y2d/self.scale-cx)*z/fx
            y = (x2d/self.scale-cy)*z/fy
            #j = joint_message
            #j.name = jname
            #j.pose.position.x = x
            #j.pose.position.y = y
            #j.pose.position.z = z
            #po = Pose
            #po.position.x = x
            #j.pose = po
            #print "person:",p,jname+','+str(y2d)+','+str(x2d)+','+str(x)+','+str(y)+','+str(z)
            #print "person:",p,jname, self.openni_data[userID][jname]
            #self.openni_data[userID][jname] = 

    def get_skeleton(self, preds):
        #print preds.shape
        preds_rewarped = np.ones((3, 16), dtype=np.float32)
        confidence = np.zeros((16), dtype=np.float32)
        for part in range(0, 16):
            coord = np.unravel_index(np.argmax(preds[:, :, part]), [256, 256])
            #print(np.round(coord))
            confidence[part] = np.max(preds[:, :, part])
            preds_rewarped[0, part] = coord[1]
            preds_rewarped[1, part] = coord[0]

        #preds_rewarped = np.matmul(tform, preds_rewarped)

        return preds_rewarped, confidence

    def crop(self, im, scale, croppos):
        shift_to_upper_left = np.identity(3)
        shift_to_center = np.identity(3)
        A = np.identity(3)
        #print(scale)
        scale = 200.0/scale #* float(256/640)
        #print(scale)
        #print(croppos)
        A[0][0] = scale
        A[1][1] = scale
        shift_to_upper_left[0][2] = -croppos[0]
        shift_to_upper_left[1][2] = -croppos[1]
        shift_to_center[0][2] = 128
        shift_to_center[1][2] = 128
        tform = np.matmul(A, shift_to_upper_left)
        tform = np.matmul(shift_to_center, tform)
        w = np.linalg.inv(tform)
        #tform = transform.SimilarityTransform(matrix=tform)
        #im_w = warp(im, tform.inverse, output_shape=(256, 256))
        im_w = cv2.warpAffine(im, tform[0:2,:],(256, 256))
        return im_w, w




