from ultralytics import YOLO
import comet_ml

comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="student-action-classify")

model = YOLO("yolo11m-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="20251028",
                      epochs=300,
                      imgsz=224,
                      exist_ok=True,
                      project="student_action_classify",
                      name="1")