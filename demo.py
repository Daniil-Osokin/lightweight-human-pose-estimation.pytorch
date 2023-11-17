import cv2
from human_pose_estimator import PoseEstimator

img = cv2.imread("/home/leonel/poses/pose02.jpg")

pose_estimator = PoseEstimator("cpu")

poses, _, _ = pose_estimator.get_poses(img, height_size=256)

for pose in poses:
    pose.draw(img)

cv2.imshow('Human Pose Estimation', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
