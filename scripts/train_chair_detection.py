"""
教室人体检测
"""
import comet_ml
from ultralytics import YOLO

comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="chair-detection")

model = YOLO("yolo11m.pt")

results = model.train(data=r"D:/Datasets/process_20251025_20251026/data.yaml",
                      epochs=300,
                      imgsz=640*2,
                      batch=12,
                      exist_ok=True,
                      project="chair_detection",
                      name="1",
                      degrees=10,
                      shear=10,
                      perspective=0.0005,
                      mixup=0.1,
                      cutmix=0.1)