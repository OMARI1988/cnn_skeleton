#!/usr/bin/env python

import rospy
import copy
import numpy as np

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
            for j in msg.joints:
                if j.name == "head":
                    head = j.pose.position
                    head = self._convert_to_world_frame(head)
                    #print "head", head
                if j.name == "torso":
                    torso = j.pose.position
                    torso = self._convert_to_world_frame(torso)
                    #print "torso", torso
                if j.name == "right_hand":
                    right_hand = j.pose.position
                    right_hand = self._convert_to_world_frame(right_hand)
                if j.name == "left_hand":
                    left_hand = j.pose.position
                    left_hand = self._convert_to_world_frame(left_hand)

            self.person[name]["lower"].scale = (head[2]-.25)/2.0
            self.person[name]["lower"].pose.position.x = torso[0]
            self.person[name]["lower"].pose.position.y = torso[1]
            self.person[name]["lower"].pose.position.z = (head[2]-.25)/4.0
            self.person[name]["lower"].controls[0].markers[0].scale.z = (head[2]-.25)/2.0
            self.person[name]["lower"].controls[0].markers[0].color.r = 0.3 #self.colour["lower"][2]
            self.person[name]["lower"].controls[0].markers[0].color.g = 0.3 #self.colour["lower"][1]
            self.person[name]["lower"].controls[0].markers[0].color.b = 0.3 #self.colour["lower"][0]

            self.person[name]["upper"].scale = (head[2]-.25)/2.0
            self.person[name]["upper"].pose.position.x = torso[0]
            self.person[name]["upper"].pose.position.y = torso[1]
            self.person[name]["upper"].pose.position.z = 3*(head[2]-.25)/4.0
            self.person[name]["upper"].controls[0].markers[0].scale.z = (head[2]-.25)/2.0
            self.person[name]["upper"].controls[0].markers[0].color.r = 1.0
            self.person[name]["upper"].controls[0].markers[0].color.g = 0.1
            self.person[name]["upper"].controls[0].markers[0].color.b = 0.1

            self.person[name]["head"].pose.position.x = torso[0]
            self.person[name]["head"].pose.position.y = torso[1]
            self.person[name]["head"].pose.position.z = head[2]
            # self.action.replace("\n",",")
            self.person[name]["head"].description = "" #self.action

            self.person[name]["right_hand"].pose.position.x = right_hand[0]
            self.person[name]["right_hand"].pose.position.y = right_hand[1]
            self.person[name]["right_hand"].pose.position.z = right_hand[2]

            self.person[name]["left_hand"].pose.position.x = left_hand[0]
            self.person[name]["left_hand"].pose.position.y = left_hand[1]
            self.person[name]["left_hand"].pose.position.z = left_hand[2]

            for part in self.person[name]:
                self.server.insert(self.person[name][part], self.processFeedback)
        self.server.applyChanges()

    def frameCallback(self, msg):
        self.get_sk_info()
        self._get_color_info()
        self._get_action_info()
        self._pub_image()
        for name in self.person:
            #UPDATE BODY
            pass

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

    def _get_robot_msg(self):
        f1 = open(self._rbt_files[self.frame],'r')
        for count, line in enumerate(f1):
            # read the x value
            if count == 1:
                a = float(line.split('\n')[0].split(':')[1])
                self.robot['x'] = a
            # read the y value
            elif count == 2:
                a = float(line.split('\n')[0].split(':')[1])
                self.robot['y'] = a
            # read the z value
            elif count == 3:
                a = float(line.split('\n')[0].split(':')[1])
                self.robot['z'] = a
            # read roll pitch yaw
            elif count == 5:
                ax = float(line.split('\n')[0].split(':')[1])
            elif count == 6:
                ay = float(line.split('\n')[0].split(':')[1])
            elif count == 7:
                az = float(line.split('\n')[0].split(':')[1])
            elif count == 8:
                aw = float(line.split('\n')[0].split(':')[1])
                # ax,ay,az,aw
                roll, pitch, yaw = euler_from_quaternion([ax, ay, az, aw])    #odom
                pitch = 10*math.pi / 180.   #we pointed the pan tilt 10 degrees
                self.robot['rol'] = roll
                self.robot['pit'] = pitch
                self.robot['yaw'] = yaw

    def get_sk_info(self):
        self._get_robot_msg()
        f1 = open(self._skl_files[self.frame],'r')
        joints = {}
        for count, line in enumerate(f1):
            if count == 0:
                t = np.float64(line.split(':')[1].split('\n')[0])
            # read the joint name
            elif (count-1)%10 == 0:
                j = line.split('\n')[0]
                joints[j] = []
            # read the x value
            elif (count-1)%10 == 2:
                a = float(line.split('\n')[0].split(':')[1])
                joints[j].append(a)
            # read the y value
            elif (count-1)%10 == 3:
                a = float(line.split('\n')[0].split(':')[1])
                joints[j].append(a)
            # read the z value
            elif (count-1)%10 == 4:
                a = float(line.split('\n')[0].split(':')[1])
                joints[j].append(a)
                self.data[j] = self._convert_to_world_frame(joints[j])


    # def get_openni_values(self):
    #     robot = self._get_robot_msg()
    #     f1 = open(self._skl_files[self.frame],'r')
    #     self.data = {}
    #     for count, line in enumerate(f1):
    #         if count == 0: continue
    #         line = line.split(',')
    #         joint_name = line[0]
    #         self.data[joint_name] = {}
    #         x2d = float(line[1])
    #         y2d = float(line[2])
    #         x = float(line[3])
    #         y = float(line[4])
    #         z = float(line[5])
    #         # self._convert_to_world_frame([x,y,z],[rx,ry,rz])
    #         self.data[joint_name] = [x,y,z]
    #     self.frame+=1
    #     if self.frame == len(self._skl_files):
    #         self.frame=0

    def processFeedback( self, feedback ):
        s = "Feedback from marker '" + feedback.marker_name
        s += "' / control '" + feedback.control_name + "'"

        mp = ""
        if feedback.mouse_point_valid:
            mp = " at " + str(feedback.mouse_point.x)
            mp += ", " + str(feedback.mouse_point.y)
            mp += ", " + str(feedback.mouse_point.z)
            mp += " in frame " + feedback.header.frame_id

        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            rospy.loginfo( s + ": button click" + mp + "." )
        elif feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
            rospy.loginfo( s + ": menu item " + str(feedback.menu_entry_id) + " clicked" + mp + "." )
        elif feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            rospy.loginfo( s + ": pose changed")

        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            rospy.loginfo( s + ": mouse down" + mp + "." )
        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            rospy.loginfo( s + ": mouse up" + mp + "." )
        server.applyChanges()

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
    rospy.init_node("basic_controls")
    P = people()
    P.server = InteractiveMarkerServer("basic_controls")
    #rospy.Timer(rospy.Duration(0.02), P.frameCallback)

    position = [-3, 3, 1.8]
    #P._create_person( position, '1')

    P.server.applyChanges()
    rospy.spin()

