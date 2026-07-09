import comet_ml

from ultralytics import YOLO

from letterbox_classification import StudentActionBusinessLossTrainer

comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="student-action-classify")

PREPROCESS = "letterbox"  # 可选: "center_crop", "letterbox"


def main():
    model = YOLO("/home/yujiannan/Projects/ultralytics/scripts/logs/student_action_classify/66/weights/best.pt")

    train_kwargs = dict(
        data="/DATA/yujiannan/Datasets/process_20260709_0748_student_action",
        batch=96,
        epochs=180,
        imgsz=224,
        patience=45,
        exist_ok=False,
        project="logs/student_action_classify",
        name="84",
        dropout=0.1,
        weight_decay=0.001,
        cos_lr=True,
        optimizer="SGD",
        lr0=0.003,
        lrf=0.01,
        momentum=0.9,
        warmup_epochs=0.5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.003,
        erasing=0.0,
        auto_augment=None,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.2,
        hsv_v=0.15,
    )

    if PREPROCESS == "center_crop":
        # Ultralytics 默认分类预处理：train 使用轻量 RandomResizedCrop，val/predict 使用 Resize + CenterCrop。
        train_kwargs.update(scale=0.1)
    elif PREPROCESS == "letterbox":
        # 不使用 RandomResizedCrop，避免裁掉腿、头、桌面边界后破坏站立判断。
        StudentActionBusinessLossTrainer.focus_gain = 0.35
        StudentActionBusinessLossTrainer.headup_gain = 0.25
        StudentActionBusinessLossTrainer.stand_gain = 0.10
        train_kwargs.update(trainer=StudentActionBusinessLossTrainer, scale=0.0)
    else:
        raise ValueError(f"不支持的 PREPROCESS: {PREPROCESS}")

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
