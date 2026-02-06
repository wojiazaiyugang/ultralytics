import comet_ml
from ultralytics import YOLO


comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="teacher-passion-classify")

model = YOLO("yolo11s-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/DATA/yujiannan/Datasets/20260205",
                      epochs=300,
                      imgsz=224,
                      exist_ok=True,
                      project="logs/teacher_passion_classify",
                      erasing=0,
                      name="1")