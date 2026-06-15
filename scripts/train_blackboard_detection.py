
import comet_ml
from ultralytics import YOLO

comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="blackboard-detection")

model = YOLO("/home/yujiannan/Projects/ultralytics/scripts/logs/blackboard_detection/1/weights/best.pt")

results = model.train(data=r"/DATA/yujiannan/Datasets/process_20251117_20260615_updating/data.yaml",
                      epochs=300,
                      imgsz=640*2,
                      batch=6,
                      exist_ok=True,
                      project="logs/blackboard_detection",
                      name="4",
                      degrees=10,
                      shear=10,
                      perspective=0.0005,
                      # mixup=0.1,
                      # cutmix=0.1
                      )
