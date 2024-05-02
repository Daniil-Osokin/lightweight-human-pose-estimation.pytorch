import argparse
import os.path

import cv2
import filetype

from human_pose_estimator import PoseEstimator, VideoReader, ImageReader
from human_pose_estimator.modules.pose import track_poses


def make_estimation(estimator, image_provider, delay, height_size, track, smooth):
    previous_poses = []

    for img in image_provider:
        orig_img = img.copy()
        current_poses, _, _ = estimator.get_poses(img, height_size)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1


def main():
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                           This is just for quick results preview.
                           Please, consider c++ demo for the best performance.''')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', type=str, default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    output_delay = 0
    frame_provider = None

    if args.images != '':
        images = []
        if os.path.isdir(args.images):
            for file in os.listdir(args.images):
                img_path = os.path.join(args.images, file)
                if filetype.is_image(img_path):
                    images.append(img_path)
            frame_provider = ImageReader(images)
        elif os.path.isfile(args.images):
            if filetype.is_image(args.images):
                frame_provider = ImageReader([args.images])
        else:
            raise ValueError('No valid images were found.')

    if args.video != '':
        frame_provider = VideoReader(args.video)
        output_delay = 1
    else:
        args.track = 0

    pose_estimator = PoseEstimator(args.cpu)

    make_estimation(pose_estimator, frame_provider, output_delay, args.height_size, args.track, args.smooth)


if __name__ == '__main__':
    main()
