# !/usr/bin/env python3
# type: ignore

import sys

import cython
import message_filters
import numpy as np
import ros_numpy
import rospy
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
from derived_object_msgs.msg import ObjectWithCovarianceArray
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from sensor_msgs.msg import PointCloud2
from yolov7ros.msg import BboxCentersClass

# sys.path.insert(1, "/home/avalocal/catkin_ws/src/YOLOv7_ROS/src/yolov7")

# Camera intrinsic parameters
# Transfer information from inside the camera to outside frame. This needs to rescaled based on the input image size
# explains how to make the image straight.
# rect = np.array(
#     [
#         [1760.027735, 0.0, 522.446495, 0.0],
#         [0.0, 1761.13935, 401.253765, 0.0],
#         [0.00000, 0.0000000, 1.00000000, 0.000000],
#     ]
# )
"""rect = np.array(
    [
        [1725.122315, 0.0, 522.961315, 0.0],
        [0.0, 1740.355955, 402.58045, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)"""
rect = np.array(
    [
        [3514.793819 / 2, 0.000000, 1096.938526 / 2, 0.0],
        [0.000000, 3517.227722 / 2, 808.321613 / 2, 0.0],
        [0.000000, 0.000000, 1.000000, 0.0],
    ]
)
# Camera extrinsic parameters (transformation matrix from lidar to camera)
# Explains the rotation and translation from the lidar to the camera
# T1 = np.array(
#     [
#         [0.038114, -0.027594, 0.998892, 1.057087],
#         [-0.998441, 0.039741, 0.039195, -0.425626],
#         [-0.040779, -0.998829, -0.026036, -0.809218],
#         [0.000000, 0.000000, 0.000000, 1.00000],
#     ]
# )
"""T1 = np.array(
    [
        [-0.022559, -0.030990, 0.999265, 1.691040],
        [-0.998779, -0.043245, -0.023890, 0.353791],
        [0.043958, -0.998584, -0.029977, -0.995298],
        [0.000000, 0.000000, 0.000000, 1.00000],
    ]
)"""
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
        [0, 0, 0, 1],
    ]
)


def inverse_rigid_transformation(arr):
    """
    Compute the inverse of a rigid transformation matrix.

    Args:
        arr (np.ndarray): Input transformation matrix.

    Returns:
        np.ndarray: Inverted transformation matrix.
    """
    irt = np.zeros_like(arr)
    Rt = np.transpose(arr[:3, :3])
    tt = -np.matmul(Rt, arr[:3, 3])
    irt[:3, :3] = Rt
    irt[0, 3] = tt[0]
    irt[1, 3] = tt[1]
    irt[2, 3] = tt[2]
    irt[3, 3] = 1
    return irt


# Inverse transformation matrix from camera to lidar
T_vel_cam = inverse_rigid_transformation(T1)

# Point cloud and image processing limits
lim_x = [2.5, 100]
lim_y = [-10, 10]
lim_z = [-3.5, 5]
height = 2048
width = 1544
pixel_lim = 12

class_names = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
    80: "cone",
    81: "Speed limit: 70",
    82: "Speed limit: 75",
    83: "Speed limit: 30",
    84: "Speed limit: 35",
    85: "Speed limit: 40",
    86: "Speed limit: 45",
    87: "Speed limit: 50",
    88: "Speed limit: 55",
    89: "Speed limit: 60",
    90: "Speed limit: 65",
}


class realCoor:
    """
    Class to handle the transformation of detected bounding box coordinates from 2D image space to 3D lidar space.
    """

    def __init__(self):
        # Publishers
        # self.pclOnroad_pub = rospy.Publisher("/onroad", PointCloud2, queue_size=1)
        self.bbox_publish = rospy.Publisher(
            "/fused_bbox", BoundingBoxArray, queue_size=1
        )

        # Point field configuration
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

        # Subscribers
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

        # Initialize point cloud message
        self.onroad_pointcloud = PointCloud2()
        self.header = std_msgs.msg.Header()
        self.header.frame_id = "lidar_tc"

        # Time synchronizer for lidar and image data
        ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_lidar, self.sub_image], 15, 0.4
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
            self.onroad_pointcloud = pc2.create_cloud(
                self.header, self.fields, onRoad3d
            )
            # self.pclOnroad_pub.publish(self.onroad_pointcloud)
        elif which == 1:
            pass

    def callback(self, msgLidar, msgPoint):
        """
        Callback function for synchronized lidar and image data.

        Args:
            msgLidar (sensor_msgs.msg.PointCloud2): Lidar point cloud message.
            msgPoint (yolov7ros.msg.BboxCentersClass): Bounding box centers from image detection.
        """
        # Convert lidar data to numpy array
        pc = ros_numpy.numpify(msgLidar)
        points = np.zeros((pc.shape[0], 4))
        points[:, 0] = pc["x"]
        points[:, 1] = pc["y"]
        points[:, 2] = pc["z"]
        points[:, 3] = 1

        # Transpose and transform point cloud data
        pc_arr = self.crop_pointcloud(points)  # to reduce computational expense
        pc_arr_pick = np.transpose(pc_arr)
        m1 = np.matmul(T_vel_cam, pc_arr_pick)
        uv1 = np.matmul(rect, m1)
        uv1[0, :] = np.divide(uv1[0, :], uv1[2, :])
        uv1[1, :] = np.divide(uv1[1, :], uv1[2, :])

        center_3d = []
        label = []
        u = uv1[0, :]
        v = uv1[1, :]

        # Match bounding box centers with point cloud data
        for point in msgPoint.CenterClass:
            # print("message point", class_names[msgPoint.CenterClass[0].z])
            idx = np.where(
                ((u + pixel_lim >= point.x) & (u - pixel_lim <= point.x))
                & ((v + pixel_lim >= point.y) & (v - pixel_lim <= point.y))
            )
            idx = np.array(idx)

            if idx.size > 0:
                for i in range(idx.size):
                    center_3d.append(
                        (
                            [
                                pc_arr_pick[0][idx[0, i]],
                                pc_arr_pick[1][idx[0, i]],
                                pc_arr_pick[2][idx[0, i]],
                                1,
                            ]
                        )
                    )
                    label.append([point.z])

        print("Center 3D", center_3d)

        # Publish bounding boxes if visualization is enabled
        if self.vis:
            center_3d = np.array(center_3d)
            bbox_array = BoundingBoxArray()

            for i, box in enumerate(center_3d):
                bbox = BoundingBox()
                bbox.header.stamp = msgLidar.header.stamp
                bbox.header.frame_id = msgLidar.header.frame_id

                bbox.pose.position.x = box[0]
                bbox.pose.position.y = box[1]
                bbox.pose.position.z = box[2]

                bbox.pose.orientation.w = 1
                bbox.dimensions.x = 1.5
                bbox.dimensions.y = 1.5
                bbox.dimensions.z = 1.5
                bbox.value = 1
                bbox.label = int(label[i][0])
                bbox_array.header = bbox.header
                bbox_array.boxes.append(bbox)

                # for i, obj in enumerate(msgRadar.objects):
                #     bbox = BoundingBox()
                #     bbox.header = msgRadar.header
                #     bbox.pose.position.x = obj.pose.pose.position.x - self.offset_radar_x
                #     bbox.pose.position.y = obj.pose.pose.position.y - self.offset_radar_y
                #     bbox.pose.position.z = obj.pose.pose.position.z - self.offset_radar_z
                #     bbox.dimensions.x = 1
                #     bbox.dimensions.y = 1
                #     bbox.dimensions.z = 1
                #     bbox_array.boxes.append(bbox)

                bbox_array.header.frame_id = msgLidar.header.frame_id
                # print("Bounding box array", bbox_array)
                self.bbox_publish.publish(bbox_array)

    def crop_pointcloud(self, pointcloud):
        """
        Crop the point cloud to the region of interest.

        Args:
            pointcloud (np.ndarray): Input point cloud data.

        Returns:
            np.ndarray: Cropped point cloud data.
        """
        mask = np.where(
            (pointcloud[:, 0] >= lim_x[0])
            & (pointcloud[:, 0] <= lim_x[1])
            & (pointcloud[:, 1] >= lim_y[0])
            & (pointcloud[:, 1] <= lim_y[1])
            & (pointcloud[:, 2] >= lim_z[0])
            & (pointcloud[:, 2] <= lim_z[1])
        )
        pointcloud = pointcloud[mask]
        return pointcloud


if __name__ == "__main__":
    rospy.init_node("LaneTO3D")
    realCoor()
