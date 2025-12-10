"""
老师检测
"""
import comet_ml
from ultralytics import YOLO

comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="teacher-detection")

model = YOLO(r"yolo11m.pt")

results = model.train(data=r"/DATA/yujiannan/Datasets/process_20251209_20251210/data.yml",
                      epochs=300,
                      imgsz=640*2,
                      batch=12,
                      exist_ok=True,
                      project="logs/teacher_detection",
                      name="1",
                      degrees=10,
                      shear=10,
                      perspective=0.0005,
                      mosaic=0)