import comet_ml

from ultralytics import YOLO

from letterbox_classification import LetterBoxClassificationTrainer

comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="student-action-classify")

PREPROCESS = "letterbox"  # 可选: "center_crop", "letterbox"


def main():
    model = YOLO("yolo11s-cls.pt")

    train_kwargs = dict(
        data="/DATA/yujiannan/Datasets/process_20260422_updating_limited",
        batch=96,
        epochs=300,
        imgsz=224,
        patience=80,
        exist_ok=False,
        project="logs/student_action_classify",
        name="52",
        # 对比 51：保持 11 类全量数据和增强策略，仅关闭 RandomErasing。
        dropout=0.1,
        weight_decay=0.001,
        cos_lr=True,
        erasing=0.0,
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
