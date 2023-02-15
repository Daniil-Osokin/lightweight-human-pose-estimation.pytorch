#! /usr/bin/env python3
import rospy
from geometry_msgs.msg import Point
# import sensor_msgs.msg
from sensor_msgs.msg import Image
# from std_msgs.msg import Int32MultiArray, MultiArrayLayout, MultiArrayDimension
from cv_bridge import CvBridge
# from cv_bridge import CvBridge, CvBridgeError
from lightweight_human_pose_estimation.msg import KeyPoint
from lightweight_human_pose_estimation.msg import KeyPoints
# import argparse
# import sys

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    # GPUはTrueのときはここに引っかかる`net(tensor_img)`
    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

class Flame :
    def __init__(self):
        # Get params from launch
        self.img_show_flag = rospy.get_param( rospy.get_name() + "/pose_img_show_flag", True )
        self.pub_result_flag = rospy.get_param( rospy.get_name() + "/pose_pub_result_image", True )
        self.needs_time_stamp = rospy.get_param( rospy.get_name() +  "needs_time_stamp", True )

        self.checkpoint_path = rospy.get_param( rospy.get_name() + "/checkpoint_path", "checkpoints/checkpoint_iter_370000.pth")
        self.height_size = rospy.get_param( rospy.get_name() + "/height_size", 256)
        self.cpu = rospy.get_param( rospy.get_name() + "/cpu", True )
        self.track = rospy.get_param( rospy.get_name() + "/track", 1 )
        self.smooth = rospy.get_param( rospy.get_name() + "/smooth", 1 )

        self.net = PoseEstimationWithMobileNet()
        self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        load_state(self.net, self.checkpoint)

        self.net = self.net.eval()

        if not self.cpu:
            self.net = self.net.cuda()

        self.stride = 8
        self.upsample_ratio = 4
        self.num_keypoints = Pose.num_kpts
        self.previous_poses = []
        self.delay = 1

        self.pub_flame_result = rospy.Publisher("pose",KeyPoints,queue_size=10)
        self.pub_result_img = rospy.Publisher("detect_result", Image, queue_size=10)

        self.sub_img_topic_name = rospy.get_param( rospy.get_name() + "/pose_image_topic_name", "/camera/rgb/image_raw" )
        self.sub_img = rospy.Subscriber(self.sub_img_topic_name, Image, self.img_cb)

        self.sum = 0
        self.counter = 0


    def img_cb(self, msg):
        # 読み込んだ動画像の配列
        orig_img = img = CvBridge().imgmsg_to_cv2(msg, "bgr8")

        heatmaps, pafs, scale, pad = infer_fast(self.net, img, self.height_size, self.stride, self.upsample_ratio, self.cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []

        for kpt_idx in range(self.num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
        
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)

        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio - pad[0]) / scale
        
        current_poses = []
        keypoints_msg = KeyPoints()

        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue

            pose_keypoints = np.ones((self.num_keypoints, 2), dtype=np.int32) * -1

            for kpt_id in range(self.num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

            # print(pose_keypoints)
            pose_keypoints_list = pose_keypoints.tolist()

            vec = list(range(18*2))
            for i in range(18):
                for j in range(2):
                    vec[i * 2 + j] = pose_keypoints_list[i][j]

            keypoint_msg = KeyPoint('person_{}'.format(n), Point(vec[28], vec[29], -1), Point(vec[30], vec[31], -1), 
                                        Point(vec[32], vec[33], -1), Point(vec[34], vec[35], -1), Point(vec[0] , vec[1] , -1), Point(vec[2] , vec[3] , -1),
                                        Point(vec[4] , vec[5] , -1), Point(vec[10], vec[11], -1), Point(vec[6] , vec[7] , -1), Point(vec[12], vec[13], -1), 
                                        Point(vec[8] , vec[9] , -1), Point(vec[14], vec[15], -1), Point(vec[16], vec[17], -1), Point(vec[22], vec[23], -1),
                                        Point(vec[18], vec[19], -1), Point(vec[24], vec[25], -1), Point(vec[20], vec[21], -1), Point(vec[26], vec[27], -1))

            keypoints_msg.key_point.append(keypoint_msg)
            keypoints_msg.header = msg.header

        self.pub_flame_result.publish(keypoints_msg)


        if self.track:
            track_poses(self.previous_poses, current_poses, smooth=self.smooth)
            self.previous_poses = current_poses

        for pose in current_poses:
            pose.draw(img)

        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)

        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if self.track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

        if self.pub_result_flag:
            result_img_msg = CvBridge().cv2_to_imgmsg(orig_img, "bgr8")
            result_img_msg.header.seq = self.counter
            result_img_msg.header.stamp = rospy.Time.now()
            self.pub_result_img.publish(result_img_msg)
            self.counter+=1

        if self.img_show_flag:
            cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
                
            key = cv2.waitKey(self.delay)
            if key == 27:  # esc
                rospy.signal_shutdown("Key [ESC] pressed to leave")
                return
            elif key == 112:  # 'p'
                if self.delay == 1:
                    self.delay = 0
                else:
                    self.delay = 1


if __name__ == '__main__':

    rospy.init_node("demo2_node")
    poseflame = Flame()
    try:
        rospy.spin()
    finally:
        cv2.destroyAllWindows()    