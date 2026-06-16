import comet_ml

from ultralytics import YOLO

from letterbox_classification import LetterBoxClassificationTrainer

comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="student-action-classify")

PREPROCESS = "letterbox"  # 可选: "center_crop", "letterbox"


def main():
    model = YOLO("/home/yujiannan/Projects/ultralytics/scripts/logs/student_action_classify/39/weights/best.pt")

    train_kwargs = dict(
        data="/DATA/yujiannan/Datasets/process_20260422_updating_limited",
        batch=96,
        epochs=300,
        imgsz=224,
        patience=80,
        exist_ok=False,
        project="logs/student_action_classify",
        name="58",
        # 对比 57：显式关闭 optimizer=auto 和 warmup，做真正的小学习率微调。
        dropout=0.1,
        weight_decay=0.001,
        cos_lr=True,
        optimizer="SGD",
        lr0=0.0003,
        lrf=0.1,
        warmup_epochs=0.0,
        warmup_bias_lr=0.0,
        erasing=0.05,
        auto_augment="randaugment",
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.25,
    )

    if PREPROCESS == "center_crop":
        # Ultralytics 默认分类预处理：train 使用轻量 RandomResizedCrop，val/predict 使用 Resize + CenterCrop。
        train_kwargs.update(scale=0.1)
    elif PREPROCESS == "letterbox":
        # 不使用 RandomResizedCrop，避免裁掉腿、头、桌面边界后破坏站立判断。
        train_kwargs.update(trainer=LetterBoxClassificationTrainer, scale=0.0)
    else:
        raise ValueError(f"不支持的 PREPROCESS: {PREPROCESS}")

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
