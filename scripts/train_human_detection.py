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
                      batch=4,
                      exist_ok=True,
                      project="logs/human_detection",
                      name="19",
                      cls=1.0,
                      cos_lr=True,
                      close_mosaic=60,
                      degrees=2,
                      translate=0.05,
                      scale=0.35,
                      shear=2,
                      perspective=0.0001,
                      mosaic=0.35,
                      erasing=0.1)
