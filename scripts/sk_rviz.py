#!/usr/bin/env python

import rospy
import copy
import numpy as np
import util 
from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from tf.broadcaster import TransformBroadcaster
import glob
from tf.transformations import euler_from_quaternion
import math
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Header
from skeleton_tracker.msg import skeleton_message, joint_message, skeleton_tracker_state
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from tf.transformations import euler_from_quaternion

class people():
    """docstring for people."""
    def __init__(self):
        # self.menu_handler = MenuHandler()
        self.image_topic = "/head_xtion/rgb/image_raw"
        self.skeleton_topic = "/skeleton_data/incremental"
        #self.skeleton_topic = "/skeleton_data/cnn"

        self.fx, self.fy, self.cx, self.cy = util.read_yaml_calib("")        
        self.person = {}
        self.frame = 0
        self.counter = 0
        self.video = 1
        self.how_many_frames = 10
        #self._get_next_video()
        self.data = {}      # use this later to make it online
        self.robot = {}
        self.colour = {}
        self.bridge = CvBridge()
        self.action = ""

        self.image_ready = 0
        self.robot_ready = 0
        self.ptu_ready   = 0
        self.shirt_joints = ['torso', 'left_shoulder', 'right_shoulder']
        self.short_joints = ['left_knee', 'torso', 'right_knee']

        #self.openni_data = {} # keeps track of openni_data
        #for user in range(20):
        #    self.openni_data[user] = {}
        #    self.openni_data[user]["msg"] = "Out of Scene"   

        # subscribe to camera topic
        rospy.Subscriber(self.image_topic, Image, self._get_rgb)

        # subscribe to openni state
        rospy.Subscriber("/skeleton_data/state", skeleton_tracker_state, self._get_openni_state)

        # subscribe to openni topic
        rospy.Subscriber(self.skeleton_topic, skeleton_message, self._get_openni)

        # subscribe to robot pose
        rospy.Subscriber("/robot_pose", Pose, callback=self.robot_callback, queue_size=10)

        # subscribe to ptu state
        rospy.Subscriber("/ptu/state", JointState, callback=self.ptu_callback, queue_size=1)

    def ptu_callback(self, msg):
        self.ptu_pan, self.ptu_tilt = msg.position
        if not self.ptu_ready:
            print 'ptu ready'
            self.ptu_ready = 1

    def robot_callback(self, msg):
        self.robot_pose = msg
        if not self.robot_ready:
            print 'robot ready'
            self.robot_ready = 1

    def _get_rgb(self,imgmsg):
        img = self.bridge.imgmsg_to_cv2(imgmsg, desired_encoding="passthrough")
        img = img[:,:,0:3]
        self.image = img
        if not self.image_ready:
            print 'image ready'
            self.image_ready = 1

    def _get_openni_state(self,msg):
        msg.userID = np.mod(msg.userID, 20)
        #self.openni_data[msg.userID]["msg"] = msg.message
        #self.openni_data[msg.userID]["cnt"] = 0
        print msg.userID,msg.message
        if msg.message not in ["Out of Scene", "Stopped tracking"]:
            self._create_person([0,0,2], str(msg.userID))
        else:
            self._delete_person(str(msg.userID))

    def _get_openni(self,msg):
        msg.userID = np.mod(msg.userID, 20)
        name = str(msg.userID)
        if name in self.person:
            data = {}
            for j in msg.joints:
                pose = j.pose.position
                x2d = int( pose.x*self.fx/pose.z+self.cx ) #int( pose.x*self.fy/pose.z+self.cy )
                y2d = int( pose.y*self.fy/pose.z+self.cy ) #int( pose.y*self.fx/pose.z+self.cx )

                data[j.name] = self._convert_to_world_frame(j.pose.position)
                data[j.name].append(x2d)
                data[j.name].append(y2d)
            #print data["torso"]
            #img = self._get_shirt_short(data)
            #up, lo = self._get_colours_mean()            
            
            self.person[name]["lower"].scale = (data["head"][2]-.25)/2.0
            self.person[name]["lower"].pose.position.x = data["torso"][0]
            self.person[name]["lower"].pose.position.y = data["torso"][1]
            self.person[name]["lower"].pose.position.z = (data["head"][2]-.25)/4.0
            self.person[name]["lower"].controls[0].markers[0].scale.z = (data["head"][2]-.25)/2.0
            self.person[name]["lower"].controls[0].markers[0].color.r = .3 #lo[2]/255.0
            self.person[name]["lower"].controls[0].markers[0].color.g = .3 #lo[1]/255.0
            self.person[name]["lower"].controls[0].markers[0].color.b = .6 #lo[0]/255.0

            self.person[name]["upper"].scale = (data["head"][2]-.25)/2.0
            self.person[name]["upper"].pose.position.x = data["torso"][0]
            self.person[name]["upper"].pose.position.y = data["torso"][1]
            self.person[name]["upper"].pose.position.z = 3*(data["head"][2]-.25)/4.0
            self.person[name]["upper"].controls[0].markers[0].scale.z = (data["head"][2]-.25)/2.0
            self.person[name]["upper"].controls[0].markers[0].color.r = .4 #up[2]/255.0
            self.person[name]["upper"].controls[0].markers[0].color.g = .4 #up[1]/255.0
            self.person[name]["upper"].controls[0].markers[0].color.b = 1 #up[0]/255.0

            self.person[name]["head"].pose.position.x = data["torso"][0]
            self.person[name]["head"].pose.position.y = data["torso"][1]
            self.person[name]["head"].pose.position.z = data["head"][2]
            # self.action.replace("\n",",")
            self.person[name]["head"].description = "" #self.action

            self.person[name]["right_hand"].pose.position.x = data["right_hand"][0]
            self.person[name]["right_hand"].pose.position.y = data["right_hand"][1]
            self.person[name]["right_hand"].pose.position.z = data["right_hand"][2]

            self.person[name]["left_hand"].pose.position.x = data["left_hand"][0]
            self.person[name]["left_hand"].pose.position.y = data["left_hand"][1]
            self.person[name]["left_hand"].pose.position.z = data["left_hand"][2]

            for part in self.person[name]:
                self.server.insert(self.person[name][part], self.processFeedback)
        self.server.applyChanges()

    def _get_colours_mean(self):
        # finding shirt mean
        B = self.shirt[:,:,0]!=0
        G = self.shirt[:,:,1]!=0
        R = self.shirt[:,:,2]!=0
        B_mean = int(np.mean(self.shirt[R*B*G][:,0]))
        G_mean = int(np.mean(self.shirt[R*B*G][:,1]))
        R_mean = int(np.mean(self.shirt[R*B*G][:,2]))
        shirt_mean = [B_mean,G_mean,R_mean]
        # finding short mean
        B = self.short[:,:,0]!=0
        G = self.short[:,:,1]!=0
        R = self.short[:,:,2]!=0
        B_mean = int(np.mean(self.short[R*B*G][:,0]))
        G_mean = int(np.mean(self.short[R*B*G][:,1]))
        R_mean = int(np.mean(self.short[R*B*G][:,2]))
        short_mean = [B_mean,G_mean,R_mean]
        return shirt_mean, short_mean

    def _get_shirt_short(self, data):
        img = self.image.copy()
        #img = self._remove_black(img,[1,1,1])
        mask_shirt = np.zeros((480,640), np.uint8)
        points = []
        for j in self.shirt_joints:
            y = data[j][3]
            x = data[j][4]
            points.append([x,y])
        poly = np.array(points, np.int32)
        cv2.fillConvexPoly(mask_shirt, poly, (255,255,255))
        shirt = cv2.bitwise_and(img, img, mask=mask_shirt)
        self.shirt = self._remove_black(shirt,[0,0,0])
        #cv2.imshow("shirt", self.shirt)

        img = self.image.copy()
        mask_short = np.zeros((480,640), np.uint8)
        points = []
        for j in self.short_joints:
            x = data[j][3]
            y = data[j][4]
            points.append([x,y])
            cv2.circle(img,(x,y), 3, (0,0,255), -1)
        #cv2.imshow("short", img)
        poly = np.array(points, np.int32)
        cv2.fillConvexPoly(mask_short, poly, (255,255,255))
        short = cv2.bitwise_and(img, img, mask=mask_short)
        self.short = self._remove_black(short,[0,0,0])
        #cv2.waitKey(1)
        return img

    def _remove_black(self,img,val):
        B = img[:,:,0]==0
        G = img[:,:,1]==0
        R = img[:,:,2]==0
        img[B*G*R] = val
        return img

    def _convert_to_world_frame(self, xyz):
        """Convert a single camera frame coordinate into a map frame coordinate"""
        fx = 525.0
        fy = 525.0
        cx = 319.5
        cy = 239.5

        y,z,x = xyz.x, -xyz.y, xyz.z
 
        # transformation from camera to map
        xr = self.robot_pose.position.x
        yr = self.robot_pose.position.y
        zr = self.robot_pose.position.z

        ax = self.robot_pose.orientation.x
        ay = self.robot_pose.orientation.y
        az = self.robot_pose.orientation.z
        aw = self.robot_pose.orientation.w

        roll, pr, yawr = euler_from_quaternion([ax, ay, az, aw])    #odom
        yawr += self.ptu_pan
        pr += self.ptu_tilt
 
        rot_y = np.matrix([[np.cos(pr), 0, np.sin(pr)], [0, 1, 0], [-np.sin(pr), 0, np.cos(pr)]])
        rot_z = np.matrix([[np.cos(yawr), -np.sin(yawr), 0], [np.sin(yawr), np.cos(yawr), 0], [0, 0, 1]])
        rot = rot_z*rot_y

        pos_r = np.matrix([[xr], [yr], [zr+1.66]]) # robot's position in map frame
        pos_p = np.matrix([[x], [-y], [z]]) # person's position in camera frame

        map_pos = rot*pos_p+pos_r # person's position in map frame
        x_mf = map_pos[0,0]
        y_mf = map_pos[1,0]
        z_mf = map_pos[2,0]

        return [x_mf, y_mf, z_mf]

    def processFeedback( self, feedback ):
        pass
        
    def makeCyl( self, msg, rgb ):
        marker = Marker()
        marker.type = Marker.CYLINDER
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = msg.scale
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
        marker.color.a = 1.0
        return marker

    def makeCir( self, msg, rgb ):
        marker = Marker()
        marker.type = Marker.SPHERE
        marker.scale.x = msg.scale
        marker.scale.y = msg.scale
        marker.scale.z = msg.scale
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
        marker.color.a = 1.0
        return marker

    def makeBox( self, msg, rgb ):
        marker = Marker()
        marker.type = Marker.CUBE
        marker.scale.x = msg.scale
        marker.scale.y = msg.scale
        marker.scale.z = msg.scale
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
        marker.color.a = 1.0
        return marker

    def makeCirControl( self, msg, rgb ):
        control =  InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append( self.makeCir(msg,rgb) )
        msg.controls.append( control )
        return control

    def makeBoxControl( self, msg, rgb ):
        control =  InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append( self.makeBox(msg,rgb) )
        msg.controls.append( control )
        return control

    def makeCylControl( self, msg, rgb ):
        control =  InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append( self.makeCyl(msg,rgb) )
        msg.controls.append( control )
        return control

    def _make_lower(self,p,name):
        # make lower
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.pose.position = Point(p[0],p[1],(p[2]-.25)/4.0)
        int_marker.scale = (p[2]-.25)/2.0
        int_marker.name = "person_"+name+"_lower"
        color = [0,0,.5]
        self.makeCylControl(int_marker,color)
        int_marker.controls[0].interaction_mode = InteractiveMarkerControl.NONE
        return int_marker

    def _make_upper(self,p,name):
        # make upper
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.pose.position = Point(p[0],p[1],3*(p[2]-.25)/4.0)
        int_marker.scale = (p[2]-.25)/2.0
        int_marker.name = "person_"+name+"_upper"
        color = [.5,0,0]
        self.makeCylControl(int_marker,color)
        int_marker.controls[0].interaction_mode = InteractiveMarkerControl.NONE
        return int_marker

    def _make_head(self,p,name):
        # make upper
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.pose.position = Point(p[0],p[1],3*(p[2]-.25)/4.0)
        int_marker.scale = .45
        int_marker.name = "person_"+name+"_head"
        int_marker.description = "person_1"
        color = [.7,.7,.7]
        self.makeCirControl(int_marker,color)
        int_marker.controls[0].interaction_mode = InteractiveMarkerControl.NONE
        return int_marker

    def _make_RH(self,p,name):
        # make right hand
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.pose.position = Point(p[0],p[1],p[2])
        int_marker.scale = 0.15
        int_marker.name = "person_"+name+"_right_hand"
        color = [1,.3,.3]
        self.makeBoxControl(int_marker,color)
        int_marker.controls[0].interaction_mode = InteractiveMarkerControl.NONE
        return int_marker

    def _make_LH(self,p,name):
        # make right hand
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.pose.position = Point(p[0],p[1],p[2])
        int_marker.scale = 0.15
        int_marker.name = "person_"+name+"_left_hand"
        color = [1,.3,.3]
        self.makeBoxControl(int_marker,color)
        int_marker.controls[0].interaction_mode = InteractiveMarkerControl.NONE
        return int_marker

    #####################################################################
    # People Creation
    def _create_person(self, p, name):
        self.person[name] = {}
        self.person[name]['lower'] = self._make_lower(p,name)
        self.person[name]['upper'] = self._make_upper(p,name)
        self.person[name]['head'] = self._make_head(p,name)
        self.person[name]['right_hand'] = self._make_RH(p,name)
        self.person[name]['left_hand'] = self._make_LH(p,name)

        # # insert it
        for part in self.person[name]:
            self.server.insert(self.person[name][part], self.processFeedback)
        self.server.applyChanges()

    def _delete_person(self, name):
        if name in self.person:
            for part in self.person[name]:
                self.server.erase(self.person[name][part].name)
        self.server.applyChanges()	

if __name__=="__main__":
    rospy.init_node("rviz_skeleton")
    P = people()
    P.server = InteractiveMarkerServer("people_skeleton")
    #rospy.Timer(rospy.Duration(0.02), P.frameCallback)

    position = [-3, 3, 1.8]
    #P._create_person( position, '1')

    P.server.applyChanges()
    rospy.spin()

