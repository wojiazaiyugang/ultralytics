"""

"""
import comet_ml
from ultralytics import YOLO

comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="seat-keypoints")

model = YOLO("yolo11m-pose.pt")

results = model.train(data=r"/DATA/yujiannan/Datasets/process_20251215/data.yaml",
                      epochs=300,
                      imgsz=640*2,
                      batch=12,
                      exist_ok=True,
                      project="seat_keypoints",
                      name="3",
                      degrees=10,
                      shear=10,
                      perspective=0.0005,
                      mixup=0,
                      cutmix=0)