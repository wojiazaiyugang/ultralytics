import comet_ml
from ultralytics import YOLO


comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="teacher-student-classify")

model = YOLO("yolo11m-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data=r"D:\Datasets\20251029",
                      epochs=300,
                      batch=128*4,
                      imgsz=224,
                      exist_ok=True,
                      project="teacher_student_classify",
                      name="1")