import cv2
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes

# Load a model
# model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("/home/yujiannan/Projects/ultralytics/runs/detect/train/weights/best.pt")  # load a custom model
model.val()
# Validate the model
# metrics = model.val()  # no arguments needed, dataset and settings remembered
# for file in Path("/home/yujiannan/桌面").glob("*.jpg"):
#     file = Path("/media/8TB/dataset/combined_lingya_new_yolo_add_ceph_cephalosome_add_curvi/images/test_images/1-10-1-50-3.jpg")
#     results: Results = model(file)[0]
#     results.boxes = [b for b in results.boxes if int(b.cls) in [17]]
#     r = results.plot(labels=False)
#     cv2.namedWindow("result", cv2.WINDOW_NORMAL)
#     cv2.imshow("result", r)
#     cv2.waitKey(0)