"""
教室事件检测
"""
import comet_ml
from ultralytics import YOLO

comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="classroom-event-detection")

model = YOLO(r"yolo11s.pt")

results = model.train(data=r"/DATA/yujiannan/Datasets/process_20251021/data.yml",
                      epochs=300,
                      imgsz=640,
                      batch=24,
                      exist_ok=True,
                      project="logs/classroom_event_detection",
                      name="1",
                      degrees=10,
                      shear=10,
                      perspective=0.0005)