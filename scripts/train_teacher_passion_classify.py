import os

import comet_ml
from ultralytics import YOLO


comet_api_key = os.getenv("COMET_API_KEY")
if comet_api_key:
    comet_ml.login(api_key=comet_api_key,
                   workspace="wojiazaiyugang",
                   project_name="teacher-passion-classify")

model = YOLO("/home/yujiannan/Projects/ultralytics/scripts/logs/teacher_passion_classify/1/weights/best.pt")

# Train the model
results = model.train(data="/DATA/yujiannan/Datasets/20260818_teacher_passion_classify_stable_file_split",
                      epochs=50,
                      patience=15,
                      batch=32,
                      imgsz=224,
                      optimizer="AdamW",
                      lr0=1e-5,
                      lrf=0.1,
                      weight_decay=5e-4,
                      scale=0.5,
                      auto_augment="randaugment",
                      erasing=0,
                      save_period=1,
                      exist_ok=True,
                      project="logs/teacher_passion_classify",
                      name="2")

# 业务指标选择 epoch16.pt，has_passion 推理阈值为 0.1378861367702484。
