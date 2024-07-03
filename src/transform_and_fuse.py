# type: ignore
"""
This script transforms 2D bounding box coordinates from image space to 3D lidar space 
and integrates radar object data. It uses ROS (Robot Operating System) to synchronize 
and process data from lidar, image, and radar sensors. The result is a fused 3D bounding 
box representation of detected objects which is published as a ROS message.

Usage:
- This script should be executed within a ROS environment where the required
 topics (`/lidar_tc/velodyne_points`,`/yoloLiveNode/bboxInfo`, `/radar_fc/as_tx/objects`) 
 are being published.
- It assumes the presence of specific message types and sensor configurations 
as defined in the imported message types.

Example:
    python transform_and_fuse.py
"""
import math

import message_filters
import numpy as np
import ros_numpy
import rospy
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
import tf
import yaml
from derived_object_msgs.msg import ObjectWithCovarianceArray
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from scipy.optimize import linear_sum_assignment
from sensor_msgs.msg import PointCloud2
from yolov7ros.msg import BboxCentersClass

# Camera intrinsic parameters
rect = np.array(
    [
        [1757.3969095, 0.0, 548.469263, 0.0],
        [0.0, 1758.613861, 404.160806, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)

# Camera to lidar extrinsic transformation matrix
T1 = np.array(
    [
        [
            -0.01286594650077832,
            -0.0460667467684005,
            0.9988555061983764,
            1.343301892280579,
        ],
        [
            -0.9971783142793244,
            -0.07329508411852753,
            -0.01622467796607624,
            0.2386326789855957,
        ],
        [
            0.07395861648032626,
            -0.9962457957182222,
            -0.04499375025580721,
            -0.7371386885643005,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Point cloud limits
lim_x, lim_y, lim_z = [2.5, 100], [-10, 10], [-3.5, 5]
pixel_lim = 10

# Radar Limit Cutoff
radar_limit = 50

# Average Class Dimensions
with open("class_averages.yaml", "r") as file:
    average_dimensions = yaml.safe_load(file)


def inverse_rigid_transformation(arr):
    """
    Compute the inverse of a rigid transformation matrix.
    """
    Rt = arr[:3, :3].T
    tt = -Rt @ arr[:3, 3]
    return np.vstack((np.column_stack((Rt, tt)), [0, 0, 0, 1]))


# Inverse transformation matrix from camera to lidar
T_vel_cam = inverse_rigid_transformation(T1)


class RealCoor:
    """
    Class to handle the transformation of detected bounding box coordinates
    from 2D image space to 3D lidar space.
    """

    def __init__(self):
        # Initialize ROS publishers
        self.bbox_publish = rospy.Publisher(
            "/fused_bbox", BoundingBoxArray, queue_size=1
        )

        # Point field configuration for PointCloud2
        self.fields = [
            pc2.PointField(
                name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1
            ),
            pc2.PointField(
                name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1
            ),
            pc2.PointField(
                name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1
            ),
            pc2.PointField(
                name="intensity", offset=12, datatype=pc2.PointField.FLOAT32, count=1
            ),
        ]

        # Initialize ROS subscribers
        self.sub_lidar = message_filters.Subscriber(
            "/lidar_tc/velodyne_points", PointCloud2
        )
        self.sub_image = message_filters.Subscriber(
            "/yoloLiveNode/bboxInfo", BboxCentersClass, queue_size=1
        )
        self.sub_radar = message_filters.Subscriber(
            "/radar_fc/as_tx/objects", ObjectWithCovarianceArray
        )

        # Radar offsets
        self.offset_radar_x = 3.65 - 1.435
        self.offset_radar_y = -0.2
        self.offset_radar_z = -0.655 - 1.324

        # Initialize header
        self.header = std_msgs.msg.Header()
        self.header.frame_id = "lidar_tc"

        # Time synchronizer for lidar, image, and radar data
        ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_lidar, self.sub_image, self.sub_radar], 15, 0.4
        )
        ts.registerCallback(self.callback)

        # Visualization flag
        self.vis = True
        rospy.spin()

    def create_cloud(self, onRoad3d, which):
        """
        Create and publish a point cloud message.

        Args:
            onRoad3d (np.ndarray): Point cloud data.
            which (int): Flag to indicate which point cloud to create.
        """
        self.header.stamp = rospy.Time.now()
        if which == 0:
            point_cloud_msg = pc2.create_cloud(self.header, self.fields, onRoad3d)
            # Uncomment to publish point cloud
            # self.pclOnroad_pub.publish(point_cloud_msg)

    def callback(self, msgLidar, msgPoint, msgRadar):
        """
        Callback function for synchronized lidar, image, and radar data.

        Args:
            msgLidar (sensor_msgs.msg.PointCloud2): Lidar point cloud message.
            msgPoint (yolov7ros.msg.BboxCentersClass): Bounding box centers from image detection.
            msgRadar (derived_object_msgs.msg.ObjectWithCovarianceArray): Radar objects message.
        """
        # Convert lidar data to numpy array
        pc = ros_numpy.numpify(msgLidar)
        points = np.vstack((pc["x"], pc["y"], pc["z"], np.ones(pc["x"].shape[0]))).T

        # Crop point cloud and transform to camera frame
        pc_arr = self.crop_pointcloud(points)
        pc_arr_pick = np.transpose(pc_arr)
        m1 = np.matmul(T_vel_cam, pc_arr_pick)
        uv1 = np.matmul(rect, m1)
        uv1[:2, :] /= uv1[2, :]

        center_3d = []
        label = []
        u, v = uv1[0, :], uv1[1, :]

        # Match bounding box centers with point cloud data
        for point in msgPoint.CenterClass:
            idx = np.where(
                (u + pixel_lim >= point.x)
                & (u - pixel_lim <= point.x)
                & (v + pixel_lim >= point.y)
                & (v - pixel_lim <= point.y)
            )[0]

            if idx.size > 0:
                for i in idx:
                    center_3d.append(
                        [pc_arr_pick[0][i], pc_arr_pick[1][i], pc_arr_pick[2][i], 1]
                    )
                    label.append([point.z])
        print(center_3d)

        if self.vis:
            bbox_array = BoundingBoxArray()
            center_3d = np.array(center_3d)

            # Finding matching detections b/w camera and radar
            camera_detections = center_3d
            radar_detections = msgRadar.objects

            distance_matrix = self.compute_distance_matrix(
                camera_detections, radar_detections
            )

            # Use the Hungarian algorithm to find the optimal assignment
            row_ind, col_ind = linear_sum_assignment(distance_matrix)
            # print("row index and col index", row_ind, col_ind)
            # Print the matched pairs
            # for i, j in zip(row_ind, col_ind):
            #     print(f"Camera detection {i} is matched with Radar detection {j}")

            # Add 3D centers to bounding box array
            for i, box in enumerate(center_3d):

                class_ = int(label[i][0])
                x, y, z = box[:3]
                bbox = BoundingBox()
                bbox.header.stamp = msgLidar.header.stamp
                bbox.header.frame_id = msgLidar.header.frame_id
                bbox.pose.position.x, bbox.pose.position.y, bbox.pose.position.z = box[
                    :3
                ]

                # yaw = yaw_angle = math.atan2(y, x + bbox.dimensions.x / 2)

                # # quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
                # bbox.pose.orientation.x = quaternion[0]
                # bbox.pose.orientation.y = quaternion[1]
                # bbox.pose.orientation.z = quaternion[2]
                # bbox.pose.orientation.w = 1
                # estimated_yaw = math.atan2(y, x)

                # bbox.pose.orientation.w = 1 if estimated_yaw > 0 else -1

                bbox.pose.orientation.w = 1
                bbox.dimensions.x = average_dimensions[class_]["dimensions"][2]
                bbox.dimensions.y = average_dimensions[class_]["dimensions"][1]
                bbox.dimensions.z = average_dimensions[class_]["dimensions"][0]

                bbox.value = 1
                bbox.label = int(label[i][0])  # class number
                bbox_array.boxes.append(bbox)

            # Add radar objects to bounding box array
            for i, obj in enumerate(msgRadar.objects):
                if i not in col_ind and obj.pose.pose.position.x > radar_limit:
                    # if i not in col_ind:
                    bbox = BoundingBox()
                    bbox.header = msgRadar.header
                    bbox.pose.position.x = (
                        obj.pose.pose.position.x - self.offset_radar_x
                    )
                    bbox.pose.position.y = (
                        obj.pose.pose.position.y - self.offset_radar_y
                    )
                    bbox.pose.position.z = (
                        obj.pose.pose.position.z - self.offset_radar_z
                    )
                    bbox.dimensions.x = 1.5
                    bbox.dimensions.y = 1.5
                    bbox.dimensions.z = 1.5
                    bbox_array.boxes.append(bbox)

            bbox_array.header.frame_id = msgLidar.header.frame_id
            self.bbox_publish.publish(bbox_array)

    # Calculate the distance matrix
    def compute_distance_matrix(self, camera_detections, radar_detections):
        num_camera = camera_detections.shape[0]
        num_radar = len(radar_detections)

        distance_matrix = np.zeros((num_camera, num_radar))
        for i in range(num_camera):
            for j in range(num_radar):
                distance_matrix[i, j] = np.linalg.norm(
                    camera_detections[i][0] - radar_detections[j].pose.pose.position.x
                )
        return distance_matrix

    def crop_pointcloud(self, pointcloud):
        """
        Crop the point cloud to the region of interest.

        Args:
            pointcloud (np.ndarray): Input point cloud data.

        Returns:
            np.ndarray: Cropped point cloud data.
        """
        mask = np.all(
            [
                (pointcloud[:, 0] >= lim_x[0]),
                (pointcloud[:, 0] <= lim_x[1]),
                (pointcloud[:, 1] >= lim_y[0]),
                (pointcloud[:, 1] <= lim_y[1]),
                (pointcloud[:, 2] >= lim_z[0]),
                (pointcloud[:, 2] <= lim_z[1]),
            ],
            axis=0,
        )
        return pointcloud[mask]


if __name__ == "__main__":
    rospy.init_node("LaneTO3D")
    RealCoor()