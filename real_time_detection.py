import cv2
import numpy as np
from ultralytics import YOLO

best_model = YOLO(
    "/home/avalocal/Documents/Sign-and-Cone-Detection/runs/detect/train13/weights/best.pt"
)
cap = cv2.VideoCapture(
    "/home/avalocal/Documents/yolov9_ros/src/yolov9ros/src/yolov9/San Francisco group placing traffic cones on self-driving cars to disable them.mp4"
)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    print("Shape", frame.shape)
    results = best_model(frame)
    result = results[0]
    class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")  # Bounding boxes
    for i, bbox in enumerate(bboxes):
        (x1, y1, x2, y2) = bbox
        class_id = class_ids[i]
        class_label = best_model.names[class_id]
        label = f"{class_label} ({result.boxes.conf[i]:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )

    # Write the frame to the output video file
    # out.write(frame)

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release everything if job is finished
cap.release()
# out.release()
cv2.destroyAllWindows()
