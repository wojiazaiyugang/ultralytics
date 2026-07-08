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
                      epochs=60,
                      imgsz=640*2,
                      batch=4,
                      optimizer="AdamW",
                      lr0=0.00005,
                      lrf=0.2,
                      warmup_epochs=1.0,
                      warmup_bias_lr=0.00005,
                      exist_ok=True,
                      save_period=5,
                      project="logs/human_detection",
                      name="24",
                      box=8.0,
                      cls=1.2,
                      cos_lr=True,
                      close_mosaic=0,
                      degrees=0.5,
                      translate=0.02,
                      scale=0.1,
                      shear=0.5,
                      perspective=0.0,
                      mosaic=0.0,
                      erasing=0.0)
