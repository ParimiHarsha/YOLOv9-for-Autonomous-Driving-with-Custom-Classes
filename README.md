# YOLOv9 for Autonomous Driving with Custom Classes

## Preface:

YOLOv9, released in February 2024, marks a significant advancement in the YOLO (You Only Look Once) series, a family of object detection models that have revolutionized the field of computer vision. According to the project research team, YOLOv9 achieves a higher mAP than existing popular YOLO models such as YOLOv8, YOLOv7, and YOLOv5, when benchmarked against the MS COCO dataset.

## Problem Statement:

YOLOv9 was trained on the [COCO Dataset](https://cocodataset.org/#home) which has 80 classes. It excels at detecting those classes. Our goal is to add new custom classes to the existing COCO classes and detect all classes effectively.

Our specific use case is in the field of autonomous driving (AD). We aim to detect additional classes of cones and traffic signs. Specifically, we want to detect the following 15 classes in addition to the existing 80 COCO classes:

```python
0: 'Speed limit: 70'
1: 'Speed limit: 75'
2: 'Speed limit: 30'
3: 'Speed limit: 35'
4: 'Speed limit: 40'
5: 'Speed limit: 45'
6: 'Speed limit: 50'
7: 'Speed limit: 55'
8: 'Speed limit: 60'
9: 'Speed limit: 65'
10: 'Stop'
11: 'Red'
12: 'Yellow'
13: 'Green'
14: 'cones'
```
We have curated datasets to help us train for the above classes:

- For the COCO classes, we used the [COCO Minitrain](https://github.com/giddyyupp/coco-minitrain) dataset. This is a curated mini training set (25K images ≈ 20% of `train2017`) for COCO, useful for hyperparameter tuning and reducing the cost of ablation experiments.
- For the signs and traffic lights, we created a dataset by capturing images across Texas. GitHub repo for the dataset and instructions: [Traffic-Sign-and-Light-Detection](https://github.com/ava-share/Traffic-Sign-and-Light-Detection)
- For the cones, we used the publicly available dataset from Roboflow: [Traffic Cones Dataset](https://universe.roboflow.com/robotica-xftin/traffic-cones-4laxg)
- Custom a nnotated dataset for cones and signs 

## Methodology

Using the LabelBox tool we have annotated images for our custom classes, `cones` and the `traffic signs`. Then we have trained the YOLOv9 model to detect the new classes along with the existing coco classes.

## References

- [Train YOLOv9 Model](https://blog.roboflow.com/train-yolov9-model/)
- [Fine-tuning YOLOv9](https://learnopencv.com/fine-tuning-yolov9/#aioseo-experiment-1-freezing-the-backbone-lower-learning-rate-at-0-001)
- [Adding Classes to YOLOv8](https://y-t-g.github.io/tutorials/yolov8n-add-classes/)
- [COCO Dataset](https://cocodataset.org/#home)
- [COCO Minitrain](https://github.com/giddyyupp/coco-minitrain)
- [Cone Detection Dataset](https://github.com/ikatsamenis/Cone-Detection)
