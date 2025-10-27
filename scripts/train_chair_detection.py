"""
教室人体检测
"""
import comet_ml
from ultralytics import YOLO

comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="chair-detection")

model = YOLO(r"D:\Projects\ultralytics\scripts\human_detection\4\weights\best.pt")

results = model.train(data=r"D:\Datasets\process_20251013_20251014_20251018_20251023\data.yaml",
                      lr0=0.001,
                      lrf=0.001,
                      freeze=10,
                      epochs=50,
                      imgsz=640*2,
                      batch=12,
                      exist_ok=True,
                      project="human_detection",
                      name="7",
                      degrees=10,
                      shear=10,
                      perspective=0.0005,
                      mixup=0.1,
                      cutmix=0.1)