#!/usr/bin/env python

from typing import List

import cv2
import geometry_msgs.msg
import numpy as np
import ros_numpy
import rospy
import torch
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from ultralytics import YOLO

from yolov9ros.msg import BboxCentersClass

# Configuration parameters
weights: str = (
    "/home/avalocal/Documents/yolov9_ros/src/yolov9ros/src/yolov9/runs/detect/train16/weights/best.pt"
)
img_size: int = 640
conf_thres: float = 0.6
iou_thres: float = 0.6
device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
view_img: bool = True
augment: bool = True


class Detect:
    def __init__(self) -> None:
        self.model = YOLO(weights).to(device)
        if device != "cpu":
            self.model.half()
        self.names: List[str] = self.model.names
        self.image_sub = rospy.Subscriber(
            "/resized/camera_fl/image_color",
            Image,
            self.camera_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.image_pub = rospy.Publisher("~published_image", Image, queue_size=1)
        self.bboxInfo_pub = rospy.Publisher("~bboxInfo", BboxCentersClass, queue_size=1)
        rospy.spin()

    def camera_callback(self, data: Image) -> None:
        img: np.ndarray = ros_numpy.numpify(data)
        img_resized: np.ndarray = cv2.resize(img, (img_size, img_size))
        img_rgb: np.ndarray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor: torch.Tensor = (
            torch.from_numpy(img_rgb).to(device).float().permute(2, 0, 1) / 255.0
        )
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            detections = self.model(img_tensor)[0]
            bboxes: np.ndarray = detections.boxes.xyxy.cpu().numpy().astype(int)
            class_ids: np.ndarray = detections.boxes.cls.cpu().numpy().astype(int)
            confidences: np.ndarray = detections.boxes.conf.cpu().numpy()

            for bbox, class_id, conf in zip(bboxes, class_ids, confidences):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label: str = f"{self.names[class_id]}: {conf:.2f}"
                cv2.putText(
                    img_resized,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 225),
                    2,
                )

            if len(bboxes) > 0:
                for bbox in detections.boxes.data:
                    self.publish_center_class(
                        bbox.cpu().numpy().tolist(), data.header.stamp
                    )

            if view_img:
                self.publish_image(img_resized, data.header.stamp)

    def publish_center_class(self, detections: List[float], stamp: rospy.Time) -> None:
        x1, y1, x2, y2, conf, cls = detections
        x_center: float = (x1 + x2) / 2
        y_center: float = (y1 + y2) / 2
        point: geometry_msgs.msg.Point32 = geometry_msgs.msg.Point32(
            x=x_center, y=y_center, z=cls
        )

        msg: BboxCentersClass = BboxCentersClass()
        msg.header.stamp = stamp
        msg.CenterClass = [point]
        self.bboxInfo_pub.publish(msg)

    def publish_image(self, img: np.ndarray, stamp: rospy.Time) -> None:
        img_pil: PILImage = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        msg: Image = Image()
        msg.header.stamp = stamp
        msg.height = img_pil.height
        msg.width = img_pil.width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = 3 * img_pil.width
        msg.data = np.array(img_pil).tobytes()
        self.image_pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("yoloLiveNode")
    Detect()
