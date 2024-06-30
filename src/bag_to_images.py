# """Extract images from a rosbag"""

import os

import cv2
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def extract_images_from_rosbag(
    bag_file: str, output_dir: str, image_topic: str, frequency: int = 5
) -> None:
    """Extract images from a rosbag."""
    print(f"Extracting images from {bag_file} on topic {image_topic} into {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()
    count = 0
    saved_count = 0

    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        if count % frequency == 0:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            image_path = os.path.join(output_dir, f"frame{saved_count:06}.png")
            cv2.imwrite(image_path, cv_img)
            print(f"Wrote image {saved_count}")
            saved_count += 1
        count += 1

    bag.close()


if __name__ == "__main__":
    bag_file = "/media/avalocal/T7_Jonas/calib_may/2024-05-02-10-37-54.bag"
    image_topic = "/resized/camera_fl/image_color"
    output_dir = "/media/avalocal/T7_Jonas/harsha/ros_images"
    extract_images_from_rosbag(bag_file, output_dir, image_topic)
