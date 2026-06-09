import comet_ml

from ultralytics import YOLO

from letterbox_classification import LetterBoxClassificationTrainer


comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="teacher-action-classify")

PREPROCESS = "center_crop"  # 可选: "center_crop", "letterbox"


def main():
    model = YOLO("yolo11s-cls.pt")

    train_kwargs = dict(
        data="/DATA/yujiannan/Datasets/process_20260414_updating_limited",
        batch=16,
        epochs=300,
        imgsz=224,
        patience=100,
        exist_ok=False,
        project="logs/teacher_action_classify",
        name="19",
        # 最新 updated 数据重新训练。老师动作分类当前以 13 的 center crop 旧策略最稳。
        dropout=0.0,
        weight_decay=0.0005,
        cos_lr=False,
        erasing=0,
        auto_augment="randaugment",
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    if PREPROCESS == "center_crop":
        # Ultralytics 默认分类预处理：train 使用轻量 RandomResizedCrop，val/predict 使用 Resize + CenterCrop。
        train_kwargs.update(scale=0.5)
    elif PREPROCESS == "letterbox":
        # 不使用 RandomResizedCrop，避免裁掉老师全身轮廓后破坏坐/站/板书判断。
        train_kwargs.update(trainer=LetterBoxClassificationTrainer, scale=0.0)
    else:
        raise ValueError(f"不支持的 PREPROCESS: {PREPROCESS}")

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
