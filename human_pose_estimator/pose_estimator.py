import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from human_pose_estimator.models.with_mobilenet import PoseEstimationWithMobileNet
from human_pose_estimator.modules.keypoints import extract_keypoints, group_keypoints
from human_pose_estimator.modules.load_state import load_state
from human_pose_estimator.modules.pose import Pose
from human_pose_estimator.val import normalize, pad_width

REPO_ID = "cxeep/PIFuHD"


class PoseEstimator:
    def __init__(self, cpu=None):
        self.cpu = cpu
        self.net = PoseEstimationWithMobileNet()
        self.checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename="checkpoint_iter_370000.pth")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        load_state(self.net, checkpoint)
        self.net = self.net.eval()
        if not cpu:
            self.net = self.net.cuda()

    def infer_fast(self, img, net_input_height_size, stride, upsample_ratio,
                   pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1 / 256)):
        height, width, _ = img.shape
        scale = net_input_height_size / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if not self.cpu:
            tensor_img = tensor_img.cuda()

        stages_output = self.net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad

    def get_poses(self, img, height_size):

        stride = 8
        upsample_ratio = 4
        num_keypoints = Pose.num_kpts

        heatmaps, pafs, scale, pad = self.infer_fast(img, height_size, stride, upsample_ratio)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

        poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])

            pose = Pose(pose_keypoints, pose_entries[n][18])
            poses.append(pose)

        return poses, pose_entries, all_keypoints
