"""
教室人体检测
"""
from ultralytics import YOLO

model = YOLO("yolo11m.pt")

# Train the model
results = model.train(data=r"C:\Users\yujiannan\Downloads\Classroom mointoring.v1i.yolov11\data.yaml",
                      epochs=300,
                      batch=0.9,
                      imgsz=640 * 2,
                      project="human_detection",
                      name="2",
                      degrees=10,
                      shear=10,
                      perspective=0.0005,
                      mixup=0.1,
                      cutmix=0.1)