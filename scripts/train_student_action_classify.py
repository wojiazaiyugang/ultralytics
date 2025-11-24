import comet_ml
from ultralytics import YOLO


comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="student-action-classify")

model = YOLO("yolo11m-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data=r"/DATA/yujiannan/Datasets/20251121",
                      epochs=300,
                      imgsz=224,
                      exist_ok=True,
                      project="logs/student_action_classify",
                      auto_augment="augmix",
                      name="4")