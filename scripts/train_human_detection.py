"""
教室人体检测
"""
import comet_ml
from ultralytics import YOLO

comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="human-detection")

model = YOLO(r"/home/yujiannan/Projects/ultralytics/scripts/logs/human_detection/4/weights/best.pt")

results = model.train(data=r"/DATA/yujiannan/Datasets/process_20260119_20260120_20260121_20260122/data.yaml",
                      lr0=0.001,
                      lrf=0.001,
                      # freeze=10,
                      epochs=50,
                      imgsz=640*2,
                      batch=12,
                      exist_ok=True,
                      project="logs/human_detection",
                      name="9",
                      degrees=10,
                      shear=10,
                      perspective=0.0005)