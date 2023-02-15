#!/bin/bash

echo "╔══╣ Install: lightweight_human_pose_estimation (STARTING) ╠══╗"


sudo apt-get update

sudo apt-get install -y \
    v4l-utils
    
sudo apt-get install -y \
    ros-${ROS_DISTRO}-libuvc-camera \
    ros-${ROS_DISTRO}-camera-calibration \
    ros-${ROS_DISTRO}-image-proc

python3 -m pip install \
    pycocotools

mkdir script/checkpoints/
wget -P script/checkpoints/ https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth

# Seting dynamixel USB1 (SOBIT PRO arm_pantilt)
echo "SUBSYSTEMS==\"usb\", ENV{DEVTYPE}==\"usb_device\", ATTRS{idVendor}==\"0458\", ATTRS{idProduct}==\"708c\", MODE=\"0666\"" | sudo tee /etc/udev/rules.d/99-uvc.rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# USB Reload
sudo /etc/init.d/udev reload

v4l2-ctl --list-devices
v4l2-ctl --list-formats-ext


echo "╚══╣ Install: lightweight_human_pose_estimation (FINISHED) ╠══╝"