# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Run the below command to run inference on ROS image topic:
python inference_ros.py --weights yolov9c.pt --reg_weights weights/resnet18.pkl --model_select resnet18
"""


import cv2
import numpy as np
import ros_numpy
import rospy
import torch
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from library.Math import calc_location
from library.Plotting import plot_3d_box
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from torch_lib import ClassAverages, Model
from torch_lib.Dataset import DetectedObject, generate_bins
from torchvision.models import vgg
from ultralytics import YOLO


class Detection:
    def __init__(self, box_2d, class_):
        self.box_2d = box_2d
        self.detected_class = class_


def main():
    reg_weights = (
        "/media/avalocal/T9/harsha/datasets/3D-BoundingBox/weights/epoch_20.pkl"
    )
    InferenceROS(reg_weights)


class InferenceROS:
    def __init__(self, reg_weights):
        self.reg_weights = reg_weights
        self.sub_image = rospy.Subscriber(
            "/resized/camera_fl/image_color", Image, self.callback
        )
        self.pub_bboxes = rospy.Publisher(
            "/detected_objects_3d", BoundingBoxArray, queue_size=10
        )
        self.pub_imageBBox = rospy.Publisher("/image_bbox", Image, queue_size=10)

        # Load model
        self.weights = "/home/avalocal/Documents/yolov9_ros/src/yolov9ros/src/runs/detect/train3/weights/best.pt"

        self.device = 0
        # self.data = "data/coco128.yaml"
        # self.device = select_device(self.device)
        self.model = YOLO(self.weights)
        self.imgsz = 640
        my_vgg = vgg.vgg19_bn(pretrained=True)
        self.regressor = Model.Model(features=my_vgg.features, bins=2).cuda()
        # load weight
        checkpoint = torch.load(
            "/media/avalocal/T7_Jonas/harsha/3D-BoundingBox/weights/epoch_20.pkl"
        )
        self.regressor.load_state_dict(checkpoint["model_state_dict"])
        self.regressor.eval()

        self.averages = ClassAverages.ClassAverages()
        self.angle_bins = generate_bins(2)

        # rect = np.array(
        #     [
        #         [3450.24463 / 2 * 640 / 1032, 0.0, 1045.92263 / 2 * 640 / 772, 0.0],
        #         [
        #             0.0,
        #             3480.71191 / 2 * 640 / 1032,
        #             805.1609 / 2 * 640 / 772,
        #             0.0,
        #         ],
        #         [0.00000, 0.0000000, 1.00000000, 0.000000],
        #     ]
        # )  # white jeep camera_fl

        rect = np.array(
            [
                [1725.122315, 0.0, 522.961315, 0.0],
                [0.0, 1740.355955, 402.58045, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        self.T1 = np.array(
            [
                [-0.022559, -0.030990, 0.999265, 1.691040],
                [-0.998779, -0.043245, -0.023890, 0.353791],
                [0.043958, -0.998584, -0.029977, -0.995298],
                [0.000000, 0.000000, 0.000000, 1.00000],
            ]
        )
        # self.T1 = calibrationToMatrix(rpy, xyz)

        self.proj_matrix = rect
        # self.proj_matrix = self.T1

        # calib_file = '/media/avalocal/T9/harsha/datasets/3D-BoundingBox/camera_cal/calib_cam_to_cam.txt'
        rospy.spin()

    def callback(self, image):
        with torch.no_grad():

            self.view_img = True
            self.img = ros_numpy.numpify(image)
            self.img = cv2.resize(self.img, (640, 640))
            self.img0 = self.img.copy()
            self.img = self.img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = self.img
            self.img = np.ascontiguousarray(self.img)

            img_tensor = (
                torch.from_numpy(self.img).to(self.device).float()
            )  # uint8 to fp16/32
            img_tensor = img_tensor / 255.0  # normalize the pixel values

            im = img_tensor.unsqueeze(0)  # add batch dimension

            # Inference
            bbox_list = []
            detections = self.model(im)
            result = detections[0]
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
            bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")  # Bounding boxes
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox.tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = self.model.names[class_ids[i]]
                bbox_list.append(Detection([(x1, y1), (x2, y2)], label))

            self.proj_matrix = np.array(self.proj_matrix)

            # Run detection 3D
            bbox_array = BoundingBoxArray()
            bbox_array.header = image.header

            for det in bbox_list:
                if not self.averages.recognized_class(det.detected_class):
                    continue

                detectedObject = DetectedObject(
                    self.img0, det.detected_class, det.box_2d, self.proj_matrix
                )

                theta_ray = detectedObject.theta_ray
                input_img = detectedObject.img
                proj_matrix = detectedObject.proj_matrix
                box_2d = det.box_2d
                detected_class = det.detected_class

                input_tensor = torch.zeros([1, 3, 224, 224]).cuda()
                input_tensor[0, :, :, :] = input_img

                # predict orient, conf, and dim
                [orient, conf, dim] = self.regressor(input_tensor)
                orient = orient.cpu().data.numpy()[0, :, :]
                conf = conf.cpu().data.numpy()[0, :]
                dim = dim.cpu().data.numpy()[0, :]

                # dim += self.averages.get_item(detected_class)

                argmax = np.argmax(conf)
                orient = orient[argmax, :]
                cos = orient[0]
                sin = orient[1]
                alpha = np.arctan2(sin, cos)
                alpha += self.angle_bins[argmax]
                alpha -= np.pi

                dimensions = dim

                location, X = calc_location(
                    dimensions, proj_matrix, box_2d, alpha, theta_ray
                )

                orient = alpha + theta_ray

                plot_3d_box(self.img0, proj_matrix, orient, dimensions, location)

                # transfer dimensions from camera frame to velodyne frame using T1
                # T1 is 4x4 matrix

                # rotation_T1 = self.T1[:3, :3]  # 3x3
                # translation_T1 = self.T1[:3, 3]  # 3x1

                location = np.append(location, 1)  # 4x1

                # self.T1 is 4x4 matrix
                Lidar_location = np.matmul(
                    self.T1, location
                )  # this gives x, y, z in lidar frame

                constant_offset_camera_lidar = np.array([0.0, 0.0, 0.0])

                # bbox
                bbox = BoundingBox()
                bbox.header = image.header
                # bbox.label = detected_class #must be unsigned int
                bbox.pose.position.x = Lidar_location[0] + dimensions[2] / 2
                bbox.pose.position.y = Lidar_location[1] - dimensions[0] / 2
                bbox.pose.position.z = Lidar_location[2]

                # rotate box around z axis for 50 degrees
                bbox.pose.orientation.x = 0.0
                bbox.pose.orientation.y = 0.0
                bbox.pose.orientation.z = 0.1
                bbox.pose.orientation.w = 1.0
                bbox.dimensions.x = dimensions[2]
                bbox.dimensions.y = dimensions[1]
                bbox.dimensions.z = dimensions[0]

                bbox_array.boxes.append(bbox)

            if self.view_img:
                img = self.img0[:, :, ::-1].transpose(2, 0, 1)
                if type(img) == torch.Tensor:
                    img = img.cpu().numpy()
                if img.shape[0] == 1:
                    img = img.squeeze(0)
                else:
                    img = img

                img = img.transpose(1, 2, 0)  # 640, 640, 3
                img = PILImage.fromarray(img, "RGB")

                msg = Image()
                msg.header.stamp = image.header.stamp  # rospy.Time.now()
                msg.height = img.height
                msg.width = img.width
                msg.encoding = "rgb8"
                msg.is_bigendian = False
                msg.step = 3 * img.width
                msg.data = np.array(img).tobytes()
                self.pub_imageBBox.publish(msg)

            self.pub_bboxes.publish(bbox_array)


if __name__ == "__main__":
    # opt = parse_opt()
    # init node
    rospy.init_node("inference_ros", anonymous=True)
    main()
