"""
教室人体检测
"""
import comet_ml
from ultralytics import YOLO

comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="human-detection")

model = YOLO("yolo11m.pt")

results = model.train(data=r"D:\Datasets\process_20251013_20251014_20251018_20251023\data.yaml",
                      epochs=300,
                      imgsz=640*2,
                      batch=8,
                      exist_ok=True,
                      project="human_detection",
                      multi_scale=True,
                      name="3",
                      degrees=10,
                      shear=10,
                      perspective=0.0005,
                      mixup=0.1,
                      cutmix=0.1)