#include <ros/ros.h>
// #include <ros/package.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include <geometry_msgs/Point.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <lightweight_human_pose_estimation/KeyPoint.h>
#include <lightweight_human_pose_estimation/KeyPoints.h>
#include <lightweight_human_pose_estimation/KeyPoint_3d.h>
#include <lightweight_human_pose_estimation/KeyPoints_3d.h>
// #include <lightweight_human_pose_estimation/BodyPartElm.h>
// #include <lightweight_human_pose_estimation/BodyPartElm_3d.h>
// #include <lightweight_human_pose_estimation/GetPersonsKeyPoint.h>

#include <cv_bridge/cv_bridge.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

#include <tf/transform_listener.h>

#include <iostream>
#include <unordered_map>

typedef pcl::PointXYZ           PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef message_filters::sync_policies::ApproximateTime<lightweight_human_pose_estimation::KeyPoints, sensor_msgs::PointCloud2>
    KeyPointSPointSyncPolicy;

class Pose3D {
    private:
        ros::NodeHandle       nh_;
        tf::TransformListener tf_listener_;
        std::string           base_frame_name_;
        std::string           cloud_topic_name_;
        std::string           msg_topic_2d_;
        PointCloud::Ptr       cloud_transformed_;

        // ros::Publisher                                                                             pub_body_point_;
        ros::Publisher                                                                             pub_keypoints_3d_;
        std::unique_ptr<message_filters::Subscriber<lightweight_human_pose_estimation::KeyPoints>> sub_keypoints_2d_;
        std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>>                     sub_pcl_;
        std::shared_ptr<message_filters::Synchronizer<KeyPointSPointSyncPolicy>>                   sync_;


        void callbackKeyPointsPCL(const lightweight_human_pose_estimation::KeyPointsConstPtr &keypoints_msg,
                                  const sensor_msgs::PointCloud2ConstPtr                     &pcl_msg) {
            //std::cout << pcl_msg->header.stamp - keypoints_msg->header.stamp << std::endl;
            std::string frame_id = pcl_msg->header.frame_id;
            ros::Time frame_stamp = pcl_msg->header.stamp;
            PointCloud  cloud_src;

			// Transform ROS cloud to PCL
            pcl::fromROSMsg(*pcl_msg, cloud_src);

			// Check if TF is working properly
            bool key = tf_listener_.canTransform(base_frame_name_, frame_id, frame_stamp);
            if (!key) {
                ROS_ERROR("Human 3D Pose: PCL canTransform failed (base_frame and target frame related!)");
                
                return;
            }
            // else{
            //     ROS_INFO("TF can be transformed");
            // }

			// Copy Point Cloud to 'cloud_transformed_'
            if (!pcl_ros::transformPointCloud(base_frame_name_, cloud_src, *cloud_transformed_, tf_listener_)) {
                ROS_ERROR("PointCloud could not be transformed");

                return;
            }

			// Check if there is any keypoint
            lightweight_human_pose_estimation::KeyPoints keypoints_2d = *keypoints_msg;
            if (keypoints_2d.key_point.size() == 0) {
                ROS_ERROR("No human detected");

                return;
            }

            lightweight_human_pose_estimation::KeyPoints_3d keypoints_3d;
            lightweight_human_pose_estimation::KeyPoint_3d keypoint_3d;
            std::string body_part_list[18] = {"r_eye", "l_eye", "r_ear", "l_ear", "nose", "neck", "r_sho", "l_sho", "r_elb", "l_elb", "r_wri", "l_wri", "r_hip", "l_hip", "r_knee", "l_knee", "r_ank", "l_ank"};

            // Human ID
            int i_h = 0;
            // TODO: start here the loop for multiple humans
            for (int i_h = 0; i_h < keypoints_2d.key_point.size(); i_h++) {
                // for (int i_b = 0; i_b < 18; i_b++) {
                for (std::string body_part : body_part_list) {
                    geometry_msgs::Point body_point;
                    int pt_x, pt_y;
                    
                    // if (body_part_list[i_b] == "r_eye"){
                    if (body_part == "r_eye"){
                        pt_x = keypoints_2d.key_point[i_h].r_eye.x;
                        pt_y = keypoints_2d.key_point[i_h].r_eye.y;
                    }
                    // else if (body_part_list[i_b] == "l_eye"){
                    else if (body_part == "l_eye"){
                        pt_x = keypoints_2d.key_point[i_h].l_eye.x;
                        pt_y = keypoints_2d.key_point[i_h].l_eye.y;
                    }
                    // else if (body_part_list[i_b] == "r_ear"){
                    else if (body_part == "r_ear"){
                        pt_x = keypoints_2d.key_point[i_h].r_ear.x;
                        pt_y = keypoints_2d.key_point[i_h].r_ear.y;
                    }
                    // else if (body_part_list[i_b] == "l_ear"){
                    else if (body_part == "l_ear"){
                        pt_x = keypoints_2d.key_point[i_h].l_ear.x;
                        pt_y = keypoints_2d.key_point[i_h].l_ear.y;
                    }
                    // else if (body_part_list[i_b] == "nose"){
                    else if (body_part == "nose"){
                        pt_x = keypoints_2d.key_point[i_h].nose.x;
                        pt_y = keypoints_2d.key_point[i_h].nose.y;
                    }
                    // else if (body_part_list[i_b] == "neck"){
                    else if (body_part == "neck"){
                        pt_x = keypoints_2d.key_point[i_h].neck.x;
                        pt_y = keypoints_2d.key_point[i_h].neck.y;
                    }
                    // else if (body_part_list[i_b] == "r_sho"){
                    else if (body_part == "r_sho"){
                        pt_x = keypoints_2d.key_point[i_h].r_sho.x;
                        pt_y = keypoints_2d.key_point[i_h].r_sho.y;
                    }
                    // else if (body_part_list[i_b] == "l_sho"){
                    else if (body_part == "l_sho"){
                        pt_x = keypoints_2d.key_point[i_h].l_sho.x;
                        pt_y = keypoints_2d.key_point[i_h].l_sho.y;
                    }
                    // else if (body_part_list[i_b] == "r_elb"){
                    else if (body_part == "r_elb"){
                        pt_x = keypoints_2d.key_point[i_h].r_elb.x;
                        pt_y = keypoints_2d.key_point[i_h].r_elb.y;
                    }
                    // else if (body_part_list[i_b] == "l_elb"){
                    else if (body_part == "l_elb"){
                        pt_x = keypoints_2d.key_point[i_h].l_elb.x;
                        pt_y = keypoints_2d.key_point[i_h].l_elb.y;
                    }
                    // else if (body_part_list[i_b] == "r_wri"){
                    else if (body_part == "r_wri"){
                        pt_x = keypoints_2d.key_point[i_h].r_wri.x;
                        pt_y = keypoints_2d.key_point[i_h].r_wri.y;
                    }
                    // else if (body_part_list[i_b] == "l_wri"){
                    else if (body_part == "l_wri"){
                        pt_x = keypoints_2d.key_point[i_h].l_wri.x;
                        pt_y = keypoints_2d.key_point[i_h].l_wri.y;
                    }
                    // else if (body_part_list[i_b] == "r_hip"){
                    else if (body_part == "r_hip"){
                        pt_x = keypoints_2d.key_point[i_h].r_hip.x;
                        pt_y = keypoints_2d.key_point[i_h].r_hip.y;
                    }
                    // else if (body_part_list[i_b] == "l_hip"){
                    else if (body_part == "l_hip"){
                        pt_x = keypoints_2d.key_point[i_h].l_hip.x;
                        pt_y = keypoints_2d.key_point[i_h].l_hip.y;
                    }
                    // else if (body_part_list[i_b] == "r_knee"){
                    else if (body_part == "r_knee"){
                        pt_x = keypoints_2d.key_point[i_h].r_knee.x;
                        pt_y = keypoints_2d.key_point[i_h].r_knee.y;
                    }
                    // else if (body_part_list[i_b] == "l_knee"){
                    else if (body_part == "l_knee"){
                        pt_x = keypoints_2d.key_point[i_h].l_knee.x;
                        pt_y = keypoints_2d.key_point[i_h].l_knee.y;
                    }
                    // else if (body_part_list[i_b] == "r_ank"){
                    else if (body_part == "r_ank"){
                        pt_x = keypoints_2d.key_point[i_h].r_ank.x;
                        pt_y = keypoints_2d.key_point[i_h].r_ank.y;
                    }
                    // else if (body_part_list[i_b] == "l_ank"){
                    else if (body_part == "l_ank"){
                        pt_x = keypoints_2d.key_point[i_h].l_ank.x;
                        pt_y = keypoints_2d.key_point[i_h].l_ank.y;
                    }
                    else{
                        continue;
                    }

                    // Get the 3D Pose(x,y,z) from each 2D Pose(x,y) body part by refering to the Point Cloud
                    // TODO: change 640 to the proper size of the camera (launcher? automatic?)
                    PointT element_xyz = cloud_transformed_->points[pt_y * 1280 + pt_x];
                    if (std::isnan(element_xyz.x) || std::isnan(element_xyz.y) || std::isnan(element_xyz.z)) {
                        continue;
                    }

                    body_point.x = element_xyz.x;
                    body_point.y = element_xyz.y;
                    body_point.z = element_xyz.z;

                    // Copy the obtained data to the 3D Pose msg
                    // if (body_part_list[i_b] == "r_eye"){
                    if (body_part == "r_eye"){
                        keypoint_3d.r_eye.x = body_point.x;
                        keypoint_3d.r_eye.y = body_point.y;
                        keypoint_3d.r_eye.z = body_point.z;
                    }
                    // else if (body_part_list[i_b] == "l_eye"){
                    else if (body_part == "l_eye"){
                        keypoint_3d.l_eye.x = body_point.x;
                        keypoint_3d.l_eye.y = body_point.y;
                        keypoint_3d.l_eye.z = body_point.z;
                    }
                    // else if (body_part_list[i_b] == "r_ear"){
                    else if (body_part == "r_ear"){
                        keypoint_3d.r_ear.x = body_point.x;
                        keypoint_3d.r_ear.y = body_point.y;
                        keypoint_3d.r_ear.z = body_point.z;
                    }
                    // else if (body_part_list[i_b] == "l_ear"){
                    else if (body_part == "l_ear"){
                        keypoint_3d.l_ear.x = body_point.x;
                        keypoint_3d.l_ear.y = body_point.y;
                        keypoint_3d.l_ear.z = body_point.z;
                    }
                    // else if (body_part_list[i_b] == "nose"){
                    else if (body_part == "nose"){
                        keypoint_3d.nose.x = body_point.x;
                        keypoint_3d.nose.y = body_point.y;
                        keypoint_3d.nose.z = body_point.z;
                    }
                    // else if (body_part_list[i_b] == "neck"){
                    else if (body_part == "neck"){
                        keypoint_3d.neck.x = body_point.x;
                        keypoint_3d.neck.y = body_point.y;
                        keypoint_3d.neck.z = body_point.z;
                    }
                    // else if (body_part_list[i_b] == "r_sho"){
                    else if (body_part == "r_sho"){
                        keypoint_3d.r_sho.x = body_point.x;
                        keypoint_3d.r_sho.y = body_point.y;
                        keypoint_3d.r_sho.z = body_point.z;
                    }
                    // else if (body_part_list[i_b] == "l_sho"){
                    else if (body_part == "l_sho"){
                        keypoint_3d.l_sho.x = body_point.x;
                        keypoint_3d.l_sho.y = body_point.y;
                        keypoint_3d.l_sho.z = body_point.z;
                    }
                    // else if (body_part_list[i_b] == "r_elb"){
                    else if (body_part == "r_elb"){
                        keypoint_3d.r_elb.x = body_point.x;
                        keypoint_3d.r_elb.y = body_point.y;
                        keypoint_3d.r_elb.z = body_point.z;
                    }
                    // else if (body_part_list[i_b] == "l_elb"){
                    else if (body_part == "l_elb"){
                        keypoint_3d.l_elb.x = body_point.x;
                        keypoint_3d.l_elb.y = body_point.y;
                        keypoint_3d.l_elb.z = body_point.z;
                    }
                    // else if (body_part_list[i_b] == "r_wri"){
                    else if (body_part == "r_wri"){
                        keypoint_3d.r_wri.x = body_point.x;
                        keypoint_3d.r_wri.y = body_point.y;
                        keypoint_3d.r_wri.z = body_point.z;
                    }
                    // else if (body_part_list[i_b] == "l_wri"){
                    else if (body_part == "l_wri"){
                        keypoint_3d.l_wri.x = body_point.x;
                        keypoint_3d.l_wri.y = body_point.y;
                        keypoint_3d.l_wri.z = body_point.z;
                    }
                    // else if (body_part_list[i_b] == "r_hip"){
                    else if (body_part == "r_hip"){
                        keypoint_3d.r_hip.x = body_point.x;
                        keypoint_3d.r_hip.y = body_point.y;
                        keypoint_3d.r_hip.z = body_point.z;
                    }
                    // else if (body_part_list[i_b] == "l_hip"){
                    else if (body_part == "l_hip"){
                        keypoint_3d.l_hip.x = body_point.x;
                        keypoint_3d.l_hip.y = body_point.y;
                        keypoint_3d.l_hip.z = body_point.z;
                    }
                    // else if (body_part_list[i_b] == "r_knee"){
                    else if (body_part == "r_knee"){
                        keypoint_3d.r_knee.x = body_point.x;
                        keypoint_3d.r_knee.y = body_point.y;
                        keypoint_3d.r_knee.z = body_point.z;
                    }
                    // else if (body_part_list[i_b] == "l_knee"){
                    else if (body_part == "l_knee"){
                        keypoint_3d.l_knee.x = body_point.x;
                        keypoint_3d.l_knee.y = body_point.y;
                        keypoint_3d.l_knee.z = body_point.z;
                    }
                    // else if (body_part_list[i_b] == "r_ank"){
                    else if (body_part == "r_ank"){
                        keypoint_3d.r_ank.x = body_point.x;
                        keypoint_3d.r_ank.y = body_point.y;
                        keypoint_3d.r_ank.z = body_point.z;
                    }
                    // else if (body_part_list[i_b] == "l_ank"){
                    else if (body_part == "l_ank"){
                        keypoint_3d.l_ank.x = body_point.x;
                        keypoint_3d.l_ank.y = body_point.y;
                        keypoint_3d.l_ank.z = body_point.z;
                    }
                    else{
                        ROS_ERROR("It is not a body part! (No idea if this happens)");

                        return;

                    }
                }
                // Introduce the data into the msg
                keypoint_3d.c_class = keypoints_2d.key_point[i_h].c_class;
                keypoints_3d.key_point_3d.push_back(keypoint_3d);

            // TODO: end here the loop for multiple humans
            }

            // Publish the msg
            pub_keypoints_3d_.publish(keypoints_3d);
        }

    public:
        Pose3D() {
            // Default params
            base_frame_name_ = "base_footprint";
            // cloud_topic_name_ = "/points2";
            cloud_topic_name_ = "/camera/depth/points";
            msg_topic_2d_ = "/human_pose_estimation/pose";

            // Get params from launcher
            ros::param::get("/human_pose_tf_broadcaster/base_frame_name", base_frame_name_);
            ros::param::get("/human_pose_tf_broadcaster/cloud_topic_name", cloud_topic_name_);

            std::cout << "human_pose_tf_broadcaster[base_frame_name]: " << base_frame_name_ << std::endl;
            std::cout << "human_pose_tf_broadcaster[cloud_topic_name]: " << cloud_topic_name_ << std::endl;
            
            // pub_keypoints_3d_ = nh_.advertise<lightweight_human_pose_estimation::KeyPoints_3d>("keypoints_3d", 10);
            pub_keypoints_3d_ = nh_.advertise<lightweight_human_pose_estimation::KeyPoints_3d>("/lightweight_human_pose_estimation/human_pose_estimation/pose_3d", 10);
            sub_keypoints_2d_.reset(new message_filters::Subscriber<lightweight_human_pose_estimation::KeyPoints>(nh_, msg_topic_2d_, 5));
            sub_pcl_.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh_, cloud_topic_name_, 5));
            
            cloud_transformed_.reset(new PointCloud());
            sync_.reset(new message_filters::Synchronizer<KeyPointSPointSyncPolicy>(
                KeyPointSPointSyncPolicy(100), *sub_keypoints_2d_, *sub_pcl_));
            sync_->registerCallback(boost::bind(&Pose3D::callbackKeyPointsPCL, this, _1, _2));
        }
};



int main(int argc, char **argv) {
    ros::init(argc, argv, "human_pose_tf_broadcaster");
    // ros::Rate rate(5);

    Pose3D           pose_3d;
    ros::AsyncSpinner spinner(0);
    spinner.start();
    ros::waitForShutdown();

    return 0;
}