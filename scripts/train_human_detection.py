"""
教室人体检测
"""
import comet_ml
from ultralytics import YOLO

comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="human-detection")

model = YOLO(r"yolo11m.pt")

results = model.train(data=r"/DATA/yujiannan/Datasets/process_20251013_20260511_updating/data.yaml",
                      # lr0=0.001,
                      # lrf=0.001,
                      # freeze=10,
                      epochs=300,
                      imgsz=640*2,
                      batch=8,
                      exist_ok=True,
                      project="logs/human_detection",
                      name="17",
                      degrees=10,
                      shear=10,
                      perspective=0.0005)