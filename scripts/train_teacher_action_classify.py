import comet_ml
from ultralytics import YOLO


comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="teacher-action-classify")

model = YOLO("yolo11m-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/DATA/yujiannan/Datasets/process_20260414_limited",
                      epochs=300,
                      imgsz=224,
                      exist_ok=True,
                      project="logs/teacher_action_classify",
                      erasing=0,
                      name="11")