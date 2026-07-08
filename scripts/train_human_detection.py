"""
教室人体检测
"""
import comet_ml
from ultralytics import YOLO

comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="human-detection")

model = YOLO(r"/home/yujiannan/Projects/ultralytics/scripts/logs/human_detection/18/weights/best.pt")

results = model.train(data=r"/DATA/yujiannan/Datasets/process_20251013_20260511_updating/data.yaml",
                      lr0=0.001,
                      lrf=0.01,
                      # freeze=10,
                      epochs=120,
                      imgsz=640*2,
                      batch=4,
                      exist_ok=True,
                      save_period=10,
                      project="logs/human_detection",
                      name="21",
                      box=8.0,
                      cls=1.2,
                      cos_lr=True,
                      close_mosaic=40,
                      degrees=1,
                      translate=0.03,
                      scale=0.2,
                      shear=1,
                      perspective=0.0,
                      mosaic=0.2,
                      erasing=0.05)
