from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x.pt")

# Train the model
train_results = model.train(
    data="combined_lingya_new_yolo_add_ceph_cephalosome.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cuda:0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    # batch=15,
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("./curvilinearFault_1687357745359curvilinearFault.jpg")
results[0].show()

# Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model
