"""
教室人体检测
"""
import comet_ml
from ultralytics import YOLO

comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="human-detection")

model = YOLO("yolo11m.pt")

results = model.train(data=r"/DATA/yujiannan/Datasets/process_20251013_20260511_updating/data.yaml",
                      epochs=300,
                      imgsz=640*2,
                      batch=4,
                      exist_ok=True,
                      save_period=10,
                      project="logs/human_detection",
                      name="25",
                      box=8.0,
                      cls=1.2,
                      cos_lr=True,
                      close_mosaic=60,
                      degrees=2,
                      translate=0.05,
                      scale=0.25,
                      shear=2,
                      perspective=0.0,
                      mosaic=0.25,
                      erasing=0.05)
